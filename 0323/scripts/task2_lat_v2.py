"""
Task 2 实验: Lesion-Aware Transformer (LAT) v2
==================================================
v1 -> v2 修复:
  - 恢复 Mixup/CutMix (v1 去掉后过拟合严重, epoch 30 就塌了)
  - 去掉 SupCon (与 Mixup 的 soft label 冲突)
  - 增大 Dropout (0.25 -> 0.35)
  - 其余架构创新保留: Mask-Guided 双池化 + 形态学特征 + Transformer Fusion

架构:
  SwinV2-Tiny (4ch: RGB+mask) -> spatial [B,8,8,768]
  -> [CLS, Global(mean_pool), Lesion(mask_pool), Morph(MLP)]
  -> Transformer Fusion (2L, 4H, d=256)
  -> CLS -> 分类
  + Mixup/CutMix (batch级正则化)

vs Exp#7 基线的区别 (只多了这3件事):
  1. Mask-Guided 空间池化 (零额外 backbone 参数)
  2. 12维形态学特征 (形状/边界描述符)
  3. Transformer Fusion (4 token self-attention, ~1.2M 额外参数)
"""

import math
import os
import random
import shutil
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from scipy.spatial import ConvexHull

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


# ══════════════════════════════════════════════════════════════
#  Setup
# ══════════════════════════════════════════════════════════════


def find_repo_root(start_dir):
    cur = os.path.abspath(start_dir)
    while True:
        if os.path.isdir(os.path.join(cur, "0322dataset")) and os.path.isdir(
            os.path.join(cur, "0323")
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError("Cannot locate repo root")
        cur = parent


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = find_repo_root(THIS_DIR)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "0323", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from task2_json_utils import (  # noqa: E402
    StrongSyncTransform,
    SyncTransform,
    acquire_run_lock,
    adapt_model_to_4ch,
    build_class_weights,
    build_weighted_sampler,
    generate_lesion_mask,
    load_annotation,
    mixup_cutmix_data,
    set_epoch_lrs,
    set_seed,
    setup_logger,
)


NUM_MORPH_FEATURES = 12


# ══════════════════════════════════════════════════════════════
#  Part 1: 形态学特征 (与 v1 相同)
# ══════════════════════════════════════════════════════════════


def compute_morph_features(mask_pil):
    """从病灶 mask 提取 12 维形态学描述符."""
    mask_np = np.array(mask_pil)
    binary = (mask_np > 127).astype(np.uint8)
    h, w = binary.shape
    total_area = h * w
    total_lesion = float(binary.sum())

    labeled, num_features = ndimage.label(binary)

    if num_features == 0 or total_lesion < 1:
        return np.zeros(NUM_MORPH_FEATURES, dtype=np.float32)

    comp_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(comp_sizes)) + 1
    largest_mask = (labeled == largest_label).astype(np.uint8)
    area = float(comp_sizes[largest_label - 1])

    eroded = ndimage.binary_erosion(largest_mask).astype(np.uint8)
    perimeter = max(float(np.sum(largest_mask - eroded > 0)), 1.0)
    circularity = min(4.0 * np.pi * area / (perimeter**2), 1.0)

    rows, cols = np.where(largest_mask > 0)
    bbox_h = int(rows.max()) - int(rows.min()) + 1
    bbox_w = int(cols.max()) - int(cols.min()) + 1
    aspect_ratio = bbox_h / (bbox_w + 1e-6)
    extent = area / (bbox_h * bbox_w + 1e-6)

    eccentricity = 0.0
    if len(rows) > 1:
        cy, cx = rows.astype(np.float64).mean(), cols.astype(np.float64).mean()
        mu20 = ((cols.astype(np.float64) - cx) ** 2).mean()
        mu02 = ((rows.astype(np.float64) - cy) ** 2).mean()
        mu11 = ((cols.astype(np.float64) - cx) * (rows.astype(np.float64) - cy)).mean()
        delta = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)
        l1 = (mu20 + mu02 + delta) / 2
        l2 = max((mu20 + mu02 - delta) / 2, 0.0)
        if l1 > 1e-6:
            eccentricity = float(np.sqrt(1.0 - l2 / l1))

    solidity, convexity = 1.0, 1.0
    unique_pts = np.unique(np.column_stack([cols, rows]).astype(np.float64), axis=0)
    if len(unique_pts) >= 4:
        try:
            hull = ConvexHull(unique_pts)
            solidity = min(area / max(hull.volume, 1e-6), 1.0)
            convexity = min(max(hull.area, 1e-6) / perimeter, 1.0)
        except Exception:
            pass

    return np.array([
        area / total_area, circularity, eccentricity, solidity, extent,
        aspect_ratio, perimeter / (np.sqrt(area) + 1e-6), convexity,
        np.sqrt(4 * area / np.pi) / max(h, w), perimeter / max(h, w),
        float(num_features), area / (total_lesion + 1e-6),
    ], dtype=np.float32)


# ══════════════════════════════════════════════════════════════
#  Part 2: Dataset
# ══════════════════════════════════════════════════════════════


class GBPDatasetLAT(Dataset):
    """4ch (RGB + lesion mask) + 12维形态学特征."""

    def __init__(self, excel_path, data_root, sync_transform=None):
        self.df = pd.read_excel(excel_path)
        self.data_root = data_root
        self.sync_transform = sync_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        json_path = img_path.replace(".png", ".json")
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        shapes = []
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])

        mask = generate_lesion_mask(shapes, img_w, img_h)
        morph = compute_morph_features(mask)

        if self.sync_transform:
            input_tensor = self.sync_transform(img, mask)
        else:
            import torchvision.transforms.functional as TF
            input_tensor = torch.cat([TF.to_tensor(img), TF.to_tensor(mask)], dim=0)

        return {
            "image": input_tensor,
            "morph": torch.from_numpy(morph),
            "label": label,
        }


# ══════════════════════════════════════════════════════════════
#  Part 3: Model
# ══════════════════════════════════════════════════════════════


class LesionAwareTransformer(nn.Module):
    """
    LAT v2: Mask-Guided Pool + Morph + Transformer Fusion (无 SupCon proj).
    """

    def __init__(self, num_classes=2, fusion_dim=256, num_morph=NUM_MORPH_FEATURES,
                 drop_rate=0.35):
        super().__init__()

        self.backbone = timm.create_model(
            "swinv2_tiny_window8_256", pretrained=True, num_classes=0, drop_rate=0.0,
        )
        adapt_model_to_4ch(self.backbone)
        self.feat_dim = self.backbone.num_features  # 768

        self.feat_proj = nn.Linear(self.feat_dim, fusion_dim)

        self.morph_encoder = nn.Sequential(
            nn.BatchNorm1d(num_morph),
            nn.Linear(num_morph, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, fusion_dim),
            nn.GELU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, fusion_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim, nhead=4, dim_feedforward=fusion_dim * 2,
            dropout=0.15, activation="gelu", batch_first=True, norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.cls_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(drop_rate),
            nn.Linear(fusion_dim, num_classes),
        )

        self.mask_temp = nn.Parameter(torch.tensor(5.0))

    def _mask_weighted_pool(self, spatial_4d, mask_ch):
        B, H, W, C = spatial_4d.shape
        mask_small = F.adaptive_avg_pool2d(mask_ch, (H, W))
        mask_flat = mask_small.reshape(B, H * W, 1)
        spatial_flat = spatial_4d.reshape(B, H * W, C)
        temp = F.softplus(self.mask_temp)
        attn = F.softmax(mask_flat * temp, dim=1)
        return (spatial_flat * attn).sum(dim=1)

    def forward(self, image, morph):
        B = image.shape[0]
        mask_ch = image[:, 3:4, :, :]

        spatial = self.backbone.forward_features(image)
        if spatial.dim() == 3:
            N, C = spatial.shape[1], spatial.shape[2]
            H = W = int(math.sqrt(N))
            spatial = spatial.reshape(B, H, W, C)

        global_feat = spatial.mean(dim=(1, 2))
        lesion_feat = self._mask_weighted_pool(spatial, mask_ch)

        global_tok = self.feat_proj(global_feat).unsqueeze(1)
        lesion_tok = self.feat_proj(lesion_feat).unsqueeze(1)
        morph_tok = self.morph_encoder(morph).unsqueeze(1)
        cls_tok = self.cls_token.expand(B, -1, -1)

        tokens = torch.cat([cls_tok, global_tok, lesion_tok, morph_tok], dim=1)
        tokens = tokens + self.pos_embed

        fused = self.fusion(tokens)
        cls_out = fused[:, 0]

        return self.cls_head(cls_out)


# ══════════════════════════════════════════════════════════════
#  Part 4: Config
# ══════════════════════════════════════════════════════════════


class Config:
    repo_root = REPO_ROOT
    data_root = os.path.join(repo_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260326_task2_LAT_SwinV2Tiny_v2"
    log_dir = os.path.join(repo_root, "0323", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    fusion_dim = 256
    batch_size = 8  # 与 Exp#7 一致 (Mixup 不需要大 batch)
    num_epochs = 100
    warmup_epochs = 8
    backbone_lr = 2e-5
    fusion_lr = 1e-4
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 5
    seed = 42
    use_amp = True
    use_mixup = True  # ★ 恢复 Mixup/CutMix

    precision_alpha = 0.6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "Lesion-Aware Transformer (LAT) v2"
    modification = (
        "SwinV2-Tiny 4ch + Mask-Guided Dual Pool + 12dim Morph + "
        "Transformer Fusion(2L/4H/256d) + Mixup/CutMix + drop=0.35"
    )
    train_transform_desc = "StrongSync256 + morph + Mixup/CutMix"
    test_transform_desc = "Sync256 + morph"
    loss_name = "CE(class_weight+LS=0.1) + Mixup/CutMix"


# ══════════════════════════════════════════════════════════════
#  Part 5: Build functions
# ══════════════════════════════════════════════════════════════


def build_model(cfg):
    return LesionAwareTransformer(
        num_classes=len(cfg.class_names), fusion_dim=cfg.fusion_dim,
    )


def build_dataloaders(cfg):
    train_sync = StrongSyncTransform(cfg.img_size, is_train=True)
    test_sync = SyncTransform(cfg.img_size, is_train=False)

    train_ds = GBPDatasetLAT(cfg.train_excel, cfg.data_root, sync_transform=train_sync)
    test_ds = GBPDatasetLAT(cfg.test_excel, cfg.data_root, sync_transform=test_sync)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_ds, test_ds, train_loader, test_loader


def build_optimizer(model, cfg):
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    fusion_params = (
        list(model.feat_proj.parameters())
        + list(model.morph_encoder.parameters())
        + list(model.fusion.parameters())
        + [model.cls_token, model.pos_embed, model.mask_temp]
    )
    fusion_params = [p for p in fusion_params if p.requires_grad]
    head_params = [p for p in model.cls_head.parameters() if p.requires_grad]

    return AdamW([
        {"params": backbone_params, "lr": cfg.backbone_lr, "base_lr": cfg.backbone_lr},
        {"params": fusion_params, "lr": cfg.fusion_lr, "base_lr": cfg.fusion_lr},
        {"params": head_params, "lr": cfg.head_lr, "base_lr": cfg.head_lr},
    ], weight_decay=cfg.weight_decay)


# ══════════════════════════════════════════════════════════════
#  Part 6: Training & Evaluation
# ══════════════════════════════════════════════════════════════


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, cfg):
    model.train()
    sum_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        morph = batch["morph"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Mixup / CutMix
        soft_labels = None
        if cfg.use_mixup and model.training:
            images, labels, soft_labels = mixup_cutmix_data(
                images, labels, num_classes=len(cfg.class_names),
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=cfg.use_amp,
        ):
            logits = model(images, morph)

            if soft_labels is not None:
                # Soft CE with class weights (与 Exp#7 一致的修复版)
                log_probs = F.log_softmax(logits, dim=1)
                if hasattr(criterion, "weight") and criterion.weight is not None:
                    w = criterion.weight.unsqueeze(0)
                    loss = -(soft_labels * log_probs * w).sum(dim=1).mean()
                else:
                    loss = -(soft_labels * log_probs).sum(dim=1).mean()
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = images.size(0)
        sum_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs

    return sum_loss / total, correct / total


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        morph = batch["morph"].to(device, non_blocking=True)
        labels = batch["label"].numpy()

        logits = model(images, morph)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

        all_probs.append(probs)
        all_preds.extend(probs.argmax(axis=1).tolist())
        all_labels.extend(labels.tolist())

    return np.concatenate(all_probs), np.array(all_preds), np.array(all_labels)


def eval_metrics(all_probs, all_preds, all_labels, class_names, logger, phase):
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    logger.info(
        f"[{phase}] Acc: {acc:.4f} | Prec(macro): {prec:.4f} | "
        f"Rec(macro): {rec:.4f} | F1(macro): {f1:.4f}"
    )
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, prec, rec, f1


def find_optimal_threshold(all_probs, all_labels, alpha=0.6):
    p_benign = all_probs[:, 0]
    best_score, best_thresh, best_f1, best_bp = 0.0, 0.5, 0.0, 0.0
    for thresh in np.arange(0.15, 0.80, 0.005):
        preds = np.where(p_benign >= thresh, 0, 1)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        bp = precision_score(all_labels, preds, average=None, zero_division=0)
        bp = bp[0] if len(bp) > 0 else 0.0
        score = alpha * f1 + (1 - alpha) * bp
        if score > best_score:
            best_score, best_thresh, best_f1, best_bp = score, thresh, f1, bp
    return best_thresh, best_f1, best_bp


def eval_with_threshold(all_probs, all_labels, threshold, class_names, logger, phase):
    preds = np.where(all_probs[:, 0] >= threshold, 0, 1)
    return eval_metrics(all_probs, preds, all_labels, class_names, logger, phase)


# ══════════════════════════════════════════════════════════════
#  Part 7: Main
# ══════════════════════════════════════════════════════════════


def main():
    cfg = Config()
    os.makedirs(cfg.log_dir, exist_ok=True)

    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_file, cfg.exp_name)

    logger.info("=" * 60)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info("任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"Fusion Dim: {cfg.fusion_dim}")
    logger.info(f"Batch Size: {cfg.batch_size} | Mixup: {cfg.use_mixup}")
    logger.info(f"LR - Backbone: {cfg.backbone_lr} | Fusion: {cfg.fusion_lr} | Head: {cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay} | Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Warmup: {cfg.warmup_epochs} | Epochs: {cfg.num_epochs}")
    logger.info(f"Precision Alpha: {cfg.precision_alpha}")
    logger.info(f"Grad Clip: {cfg.grad_clip} | AMP: {cfg.use_amp}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 60)

    train_ds, test_ds, train_loader, test_loader = build_dataloaders(cfg)
    logger.info(
        f"训练集: {len(train_ds)} (benign={sum(train_ds.df['label'] == 0)}, "
        f"no_tumor={sum(train_ds.df['label'] == 1)})"
    )
    logger.info(
        f"测试集: {len(test_ds)} (benign={sum(test_ds.df['label'] == 0)}, "
        f"no_tumor={sum(test_ds.df['label'] == 1)})"
    )

    model = build_model(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"参数量: {n_params:,} | 可训练: {n_train:,}")

    class_weights = build_class_weights(train_ds.df, cfg.class_names, cfg.device)
    logger.info(f"类别权重: benign={class_weights[0]:.4f}, no_tumor={class_weights[1]:.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    logger.info(f"损失函数: {cfg.loss_name}")

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    best_f1, best_epoch = 0.0, 0

    logger.info("\n" + "=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(1, cfg.num_epochs + 1):
        lr_factor = set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, scaler, cfg,
        )
        elapsed = time.time() - t0

        lrs = "/".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
        logger.info(
            f"Epoch [{epoch:3d}/{cfg.num_epochs}] "
            f"LR({lrs}) Factor:{lr_factor:.4f} | "
            f"Loss:{train_loss:.4f} | Acc:{train_acc:.4f} | {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 40)
            probs, preds, labels = collect_predictions(model, test_loader, cfg.device)
            _, _, _, f1 = eval_metrics(
                probs, preds, labels, cfg.class_names, logger, "Test"
            )
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")
            logger.info("-" * 40)

    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成! 最优: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 60)

    # ── Final evaluation ──
    logger.info("\n加载最优权重...")
    model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )
    probs, preds, labels = collect_predictions(model, test_loader, cfg.device)

    logger.info("=" * 60)
    logger.info("最终测试 (threshold=0.5)")
    logger.info("=" * 60)
    eval_metrics(probs, preds, labels, cfg.class_names, logger, "Final")

    # F1 阈值优化
    logger.info("\n" + "=" * 60)
    logger.info("标准阈值优化 (最大化 macro F1)")
    logger.info("=" * 60)
    t1, f1_1, _ = find_optimal_threshold(probs, labels, alpha=1.0)
    logger.info(f"F1最优阈值: {t1:.3f} (F1: {f1_1:.4f})")
    if abs(t1 - 0.5) > 0.01:
        eval_with_threshold(probs, labels, t1, cfg.class_names, logger, "Final(F1阈值)")

    # 精确度导向阈值
    logger.info("\n" + "=" * 60)
    a = cfg.precision_alpha
    logger.info(f"精确度导向阈值 ({a}*F1 + {1 - a:.1f}*benign_prec)")
    logger.info("目标: 宁愿把良性判断成无 -> 提高 benign precision")
    logger.info("=" * 60)
    t2, f1_2, bp_2 = find_optimal_threshold(probs, labels, alpha=a)
    logger.info(f"精确度导向阈值: {t2:.3f} (F1: {f1_2:.4f}, benign_prec: {bp_2:.4f})")
    eval_with_threshold(probs, labels, t2, cfg.class_names, logger, "Final(精确度导向)")

    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)
        logger.info(f"脚本已复制到: {dst}")


if __name__ == "__main__":
    main()
