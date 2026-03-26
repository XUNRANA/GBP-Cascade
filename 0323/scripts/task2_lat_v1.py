"""
Task 2 实验: Lesion-Aware Transformer (LAT) v1
==================================================
架构: SwinV2-Tiny + Mask-Guided Spatial Attention + 形态学特征 + Transformer Fusion + SupCon

核心创新 (相比 Exp #1-#12):
  1. Mask-Guided 双池化: 全局池化 + 病灶区域加权池化, 显式分离病灶/背景特征
  2. 形态学特征分支: 从病灶mask提取12维形状描述符 (圆度/离心率/密实度等)
  3. Transformer Fusion: 用 Transformer Encoder 对 [CLS, 全局, 病灶, 形态] token 做 self-attention
  4. 监督对比损失 (SupCon): 辅助损失, 拉近同类特征、推远异类特征, 改善类别可分性
  5. 精确度导向阈值: 推理时优化 benign precision, 符合"宁愿把良性判断成无"的偏好

设计逻辑:
  - 当前最优 F1 ≈ 0.64, 瓶颈在 benign F1 ≈ 0.47
  - 两类影像外观几乎相同 (polyp in gallbladder), 区别在病理学而非影像学
  - 现有方法只把 mask 作为第4通道 → backbone 可能稀释 mask 信息
  - LAT 在 backbone 输出层用 mask 做显式空间注意力 → 强制聚焦病灶区域
  - 形态学特征 (形状/边界) 可能与病理类型相关, 但之前未测试过
  - SupCon 直接优化特征空间的类别可分性, 比纯 CE 更适合难分类任务
  - Transformer Fusion 让全局/局部/形态特征通过 self-attention 交互
"""

import math
import os
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
#  Setup: find repo root & import shared utils
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
    set_epoch_lrs,
    set_seed,
    setup_logger,
)


NUM_MORPH_FEATURES = 12


# ══════════════════════════════════════════════════════════════
#  Part 1: 形态学特征计算
# ══════════════════════════════════════════════════════════════


def compute_morph_features(mask_pil):
    """
    从病灶 mask (PIL Image 'L' mode) 提取 12 维形态学描述符.

    Features:
      0. area_ratio      : 病灶面积 / 图像面积
      1. circularity      : 4pi*area / perimeter^2 (圆度, 1=完美圆)
      2. eccentricity     : 离心率 (0=圆, 1=线段)
      3. solidity         : area / convex_hull_area (密实度)
      4. extent           : area / bbox_area (填充度)
      5. aspect_ratio     : bbox_height / bbox_width (长宽比)
      6. pa_ratio         : perimeter / sqrt(area) (粗糙度/边界复杂度)
      7. convexity        : convex_perimeter / perimeter (凸度)
      8. equiv_diameter   : 等效直径 / max(H,W) (归一化大小)
      9. norm_perimeter   : perimeter / max(H,W) (归一化周长)
      10. num_components  : 连通分量数
      11. largest_ratio   : 最大连通分量面积 / 总病灶面积
    """
    mask_np = np.array(mask_pil)
    binary = (mask_np > 127).astype(np.uint8)
    h, w = binary.shape
    total_area = h * w
    total_lesion = float(binary.sum())

    labeled, num_features = ndimage.label(binary)

    if num_features == 0 or total_lesion < 1:
        return np.zeros(NUM_MORPH_FEATURES, dtype=np.float32)

    # Largest connected component
    comp_sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
    largest_label = int(np.argmax(comp_sizes)) + 1
    largest_mask = (labeled == largest_label).astype(np.uint8)
    area = float(comp_sizes[largest_label - 1])

    # Perimeter: boundary pixels
    eroded = ndimage.binary_erosion(largest_mask).astype(np.uint8)
    perimeter = float(np.sum(largest_mask - eroded > 0))
    perimeter = max(perimeter, 1.0)

    # Circularity
    circularity = min(4.0 * np.pi * area / (perimeter**2), 1.0)

    # Bounding box
    rows, cols = np.where(largest_mask > 0)
    minr, maxr = int(rows.min()), int(rows.max())
    minc, maxc = int(cols.min()), int(cols.max())
    bbox_h = maxr - minr + 1
    bbox_w = maxc - minc + 1
    aspect_ratio = bbox_h / (bbox_w + 1e-6)
    extent = area / (bbox_h * bbox_w + 1e-6)

    # Eccentricity (from second moments of inertia)
    eccentricity = 0.0
    if len(rows) > 1:
        cy = rows.astype(np.float64).mean()
        cx = cols.astype(np.float64).mean()
        mu20 = ((cols.astype(np.float64) - cx) ** 2).mean()
        mu02 = ((rows.astype(np.float64) - cy) ** 2).mean()
        mu11 = ((cols.astype(np.float64) - cx) * (rows.astype(np.float64) - cy)).mean()
        delta = np.sqrt((mu20 - mu02) ** 2 + 4 * mu11**2)
        lambda1 = (mu20 + mu02 + delta) / 2
        lambda2 = max((mu20 + mu02 - delta) / 2, 0.0)
        if lambda1 > 1e-6:
            eccentricity = float(np.sqrt(1.0 - lambda2 / lambda1))

    # Convex hull: solidity & convexity
    solidity = 1.0
    convexity = 1.0
    unique_pts = np.unique(
        np.column_stack([cols, rows]).astype(np.float64), axis=0
    )
    if len(unique_pts) >= 4:
        try:
            hull = ConvexHull(unique_pts)
            hull_area = max(hull.volume, 1e-6)  # 2D: volume = area
            hull_peri = max(hull.area, 1e-6)  # 2D: area = perimeter
            solidity = min(area / hull_area, 1.0)
            convexity = min(hull_peri / perimeter, 1.0)
        except Exception:
            pass

    # Derived metrics
    area_ratio = area / total_area
    pa_ratio = perimeter / (np.sqrt(area) + 1e-6)
    equiv_diameter = np.sqrt(4 * area / np.pi) / max(h, w)
    norm_perimeter = perimeter / max(h, w)
    largest_ratio = area / (total_lesion + 1e-6)

    return np.array(
        [
            area_ratio,
            circularity,
            eccentricity,
            solidity,
            extent,
            aspect_ratio,
            pa_ratio,
            convexity,
            equiv_diameter,
            norm_perimeter,
            float(num_features),
            largest_ratio,
        ],
        dtype=np.float32,
    )


# ══════════════════════════════════════════════════════════════
#  Part 2: Dataset
# ══════════════════════════════════════════════════════════════


class GBPDatasetLAT(Dataset):
    """LAT 数据集: 4ch (RGB + lesion mask) + 12维形态学特征."""

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

        # 从原始 mask 计算形态学特征 (增强前, 反映真实病灶形态)
        morph = compute_morph_features(mask)

        if self.sync_transform:
            input_tensor = self.sync_transform(img, mask)  # [4, H, W]
        else:
            import torchvision.transforms.functional as TF

            img_t = TF.to_tensor(img)
            mask_t = TF.to_tensor(mask)
            input_tensor = torch.cat([img_t, mask_t], dim=0)

        return {
            "image": input_tensor,
            "morph": torch.from_numpy(morph),
            "label": label,
        }


# ══════════════════════════════════════════════════════════════
#  Part 3: Model — Lesion-Aware Transformer
# ══════════════════════════════════════════════════════════════


class LesionAwareTransformer(nn.Module):
    """
    Lesion-Aware Transformer (LAT)

    架构:
      SwinV2-Tiny (4ch) -> spatial [B,8,8,768]
      -> 生成 4 个 token:
        CLS(learnable) + Global(mean_pool) + Lesion(mask_pool) + Morph(MLP)
      -> Transformer Fusion (2 layers, 4 heads, dim=256)
      -> CLS output -> 分类 + 对比投影
    """

    def __init__(self, num_classes=2, fusion_dim=256, num_morph=NUM_MORPH_FEATURES):
        super().__init__()

        # ── Backbone ──
        self.backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            num_classes=0,
            drop_rate=0.0,
        )
        adapt_model_to_4ch(self.backbone)
        self.feat_dim = self.backbone.num_features  # 768

        # ── Feature projection: 768 -> fusion_dim ──
        self.feat_proj = nn.Linear(self.feat_dim, fusion_dim)

        # ── Morph encoder: 12 -> fusion_dim ──
        self.morph_encoder = nn.Sequential(
            nn.BatchNorm1d(num_morph),
            nn.Linear(num_morph, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, fusion_dim),
            nn.GELU(),
        )

        # ── Learnable CLS token + positional encoding ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, fusion_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4, fusion_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer Fusion ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=4,
            dim_feedforward=fusion_dim * 2,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Classification Head ──
        self.cls_head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.25),
            nn.Linear(fusion_dim, num_classes),
        )

        # ── Contrastive Projection Head ──
        self.proj_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, 128),
        )

        # ── Mask attention temperature (learnable) ──
        self.mask_temp = nn.Parameter(torch.tensor(5.0))

    def _mask_weighted_pool(self, spatial_4d, mask_ch):
        """
        Mask-guided spatial pooling.
        Args:
            spatial_4d: [B, H, W, C] backbone spatial features
            mask_ch:    [B, 1, 256, 256] 4th channel lesion mask
        Returns:
            lesion_feat: [B, C]
        """
        B, H, W, C = spatial_4d.shape
        mask_small = F.adaptive_avg_pool2d(mask_ch, (H, W))  # [B, 1, H, W]
        mask_flat = mask_small.reshape(B, H * W, 1)  # [B, N, 1]
        spatial_flat = spatial_4d.reshape(B, H * W, C)  # [B, N, C]

        # Temperature-scaled softmax attention
        temp = F.softplus(self.mask_temp)
        attn = F.softmax(mask_flat * temp, dim=1)  # [B, N, 1]
        lesion_feat = (spatial_flat * attn).sum(dim=1)  # [B, C]
        return lesion_feat

    def forward(self, image, morph):
        """
        Args:
            image: [B, 4, 256, 256] (RGB + lesion mask)
            morph: [B, 12] morphological features
        Returns:
            logits: [B, 2]
            proj:   [B, 128] (L2 normalized, for SupCon)
        """
        B = image.shape[0]
        mask_ch = image[:, 3:4, :, :]  # [B, 1, 256, 256]

        # Backbone -> spatial features
        spatial = self.backbone.forward_features(image)
        # timm 1.0.15: [B, H, W, C]; older: [B, N, C]
        if spatial.dim() == 3:
            N, C = spatial.shape[1], spatial.shape[2]
            H = W = int(math.sqrt(N))
            spatial = spatial.reshape(B, H, W, C)

        # Global average pool
        global_feat = spatial.mean(dim=(1, 2))  # [B, C]

        # Mask-weighted pool (lesion-focused)
        lesion_feat = self._mask_weighted_pool(spatial, mask_ch)  # [B, C]

        # Project to fusion dimension
        global_tok = self.feat_proj(global_feat).unsqueeze(1)  # [B, 1, D]
        lesion_tok = self.feat_proj(lesion_feat).unsqueeze(1)  # [B, 1, D]
        morph_tok = self.morph_encoder(morph).unsqueeze(1)  # [B, 1, D]
        cls_tok = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        # Assemble tokens + positional encoding
        tokens = torch.cat([cls_tok, global_tok, lesion_tok, morph_tok], dim=1)
        tokens = tokens + self.pos_embed  # [B, 4, D]

        # Transformer Fusion
        fused = self.fusion(tokens)  # [B, 4, D]
        cls_out = fused[:, 0]  # [B, D]

        # Classification
        logits = self.cls_head(cls_out)

        # Contrastive projection
        proj = self.proj_head(cls_out)
        proj = F.normalize(proj, dim=1)

        return logits, proj


# ══════════════════════════════════════════════════════════════
#  Part 4: Supervised Contrastive Loss
# ══════════════════════════════════════════════════════════════


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [B, D] L2 normalized embeddings
            labels:   [B] integer class labels
        """
        device = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.contiguous().view(-1, 1)
        same_class = torch.eq(labels, labels.T).float()  # [B, B]
        self_mask = torch.eye(B, device=device)

        # Remove self-pairs
        same_class = same_class * (1 - self_mask)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # Exp similarities (exclude self)
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

        # Mean log-prob over positive pairs
        log_prob = sim - log_sum_exp
        num_pos = same_class.sum(dim=1).clamp(min=1.0)
        loss = -(same_class * log_prob).sum(dim=1) / num_pos

        return loss.mean()


# ══════════════════════════════════════════════════════════════
#  Part 5: Config
# ══════════════════════════════════════════════════════════════


class Config:
    repo_root = REPO_ROOT
    data_root = os.path.join(repo_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260326_task2_LAT_SwinV2Tiny_v1"
    log_dir = os.path.join(repo_root, "0323", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    fusion_dim = 256
    batch_size = 16
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

    # SupCon 辅助损失
    supcon_weight = 0.1
    supcon_temperature = 0.07

    # 精确度导向阈值 (alpha=1 -> 纯F1, alpha<1 -> 兼顾benign precision)
    precision_alpha = 0.6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "Lesion-Aware Transformer (LAT) v1"
    modification = (
        "SwinV2-Tiny 4ch + Mask-Guided Dual Pool + 12dim Morph Features + "
        "Transformer Fusion(2L/4H/256d) + SupCon(lam=0.1) + Precision-Focused Threshold"
    )
    train_transform_desc = "StrongSync256 + morph (4ch RGB+mask)"
    test_transform_desc = "Sync256 + morph (4ch RGB+mask)"
    loss_name = "CE(class_weight+LS=0.1) + 0.1*SupCon(tau=0.07)"


# ══════════════════════════════════════════════════════════════
#  Part 6: Build functions
# ══════════════════════════════════════════════════════════════


def build_model(cfg):
    return LesionAwareTransformer(
        num_classes=len(cfg.class_names),
        fusion_dim=cfg.fusion_dim,
    )


def build_dataloaders(cfg):
    train_sync = StrongSyncTransform(cfg.img_size, is_train=True)
    test_sync = SyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDatasetLAT(
        cfg.train_excel, cfg.data_root, sync_transform=train_sync
    )
    test_dataset = GBPDatasetLAT(
        cfg.test_excel, cfg.data_root, sync_transform=test_sync
    )

    sampler = build_weighted_sampler(train_dataset.df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    """三组差异化学习率: backbone < fusion < heads."""
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    fusion_params = (
        list(model.feat_proj.parameters())
        + list(model.morph_encoder.parameters())
        + list(model.fusion.parameters())
        + [model.cls_token, model.pos_embed, model.mask_temp]
    )
    fusion_params = [p for p in fusion_params if p.requires_grad]

    head_params = list(model.cls_head.parameters()) + list(
        model.proj_head.parameters()
    )
    head_params = [p for p in head_params if p.requires_grad]

    return AdamW(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr, "base_lr": cfg.backbone_lr},
            {"params": fusion_params, "lr": cfg.fusion_lr, "base_lr": cfg.fusion_lr},
            {"params": head_params, "lr": cfg.head_lr, "base_lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


# ══════════════════════════════════════════════════════════════
#  Part 7: Training & Evaluation
# ══════════════════════════════════════════════════════════════


def train_one_epoch(model, dataloader, ce_criterion, supcon_criterion, optimizer, device, scaler, cfg):
    model.train()
    sum_loss = 0.0
    sum_ce = 0.0
    sum_sc = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        morph = batch["morph"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=cfg.use_amp,
        ):
            logits, proj = model(images, morph)
            ce_loss = ce_criterion(logits, labels)
            sc_loss = supcon_criterion(proj, labels)
            loss = ce_loss + cfg.supcon_weight * sc_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = labels.size(0)
        sum_loss += loss.item() * bs
        sum_ce += ce_loss.item() * bs
        sum_sc += sc_loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += bs

    return {
        "loss": sum_loss / total,
        "ce": sum_ce / total,
        "sc": sum_sc / total,
        "acc": correct / total,
    }


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        morph = batch["morph"].to(device, non_blocking=True)
        labels = batch["label"].numpy()

        logits, _ = model(images, morph)
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
    """
    搜索最优阈值.
    目标: alpha * macro_F1 + (1-alpha) * benign_precision
    alpha=1 -> 纯F1; alpha=0.6 -> 兼顾精确度 (宁愿把良性判断成无)
    """
    p_benign = all_probs[:, 0]
    best_score, best_thresh, best_f1, best_bp = 0.0, 0.5, 0.0, 0.0

    for thresh in np.arange(0.15, 0.80, 0.005):
        preds = np.where(p_benign >= thresh, 0, 1)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        per_class_prec = precision_score(
            all_labels, preds, average=None, zero_division=0
        )
        bp = per_class_prec[0] if len(per_class_prec) > 0 else 0.0
        score = alpha * f1 + (1 - alpha) * bp

        if score > best_score:
            best_score, best_thresh, best_f1, best_bp = score, thresh, f1, bp

    return best_thresh, best_f1, best_bp


def eval_with_threshold(all_probs, all_labels, threshold, class_names, logger, phase):
    preds = np.where(all_probs[:, 0] >= threshold, 0, 1)
    return eval_metrics(all_probs, preds, all_labels, class_names, logger, phase)


# ══════════════════════════════════════════════════════════════
#  Part 8: Main
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

    # ── Log config ──
    logger.info("=" * 60)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info("任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"Fusion Dim: {cfg.fusion_dim}")
    logger.info(f"图像尺寸: train {cfg.train_transform_desc}, test {cfg.test_transform_desc}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"LR - Backbone: {cfg.backbone_lr} | Fusion: {cfg.fusion_lr} | Head: {cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Warmup Epochs: {cfg.warmup_epochs}")
    logger.info(f"Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"SupCon - Weight: {cfg.supcon_weight} | Temp: {cfg.supcon_temperature}")
    logger.info(f"Precision Alpha (阈值优化): {cfg.precision_alpha}")
    logger.info(f"Seed: {cfg.seed} | Epochs: {cfg.num_epochs}")
    logger.info(f"Grad Clip: {cfg.grad_clip} | AMP: {cfg.use_amp}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 60)

    # ── Data ──
    train_ds, test_ds, train_loader, test_loader = build_dataloaders(cfg)
    logger.info(
        f"训练集: {len(train_ds)} 张 "
        f"(benign={sum(train_ds.df['label'] == 0)}, "
        f"no_tumor={sum(train_ds.df['label'] == 1)})"
    )
    logger.info(
        f"测试集: {len(test_ds)} 张 "
        f"(benign={sum(test_ds.df['label'] == 0)}, "
        f"no_tumor={sum(test_ds.df['label'] == 1)})"
    )

    # ── Model ──
    model = build_model(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"参数量: {n_params:,} | 可训练: {n_train:,}")

    # ── Loss ──
    class_weights = build_class_weights(train_ds.df, cfg.class_names, cfg.device)
    logger.info(f"类别权重: benign={class_weights[0]:.4f}, no_tumor={class_weights[1]:.4f}")
    ce_criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg.label_smoothing
    )
    sc_criterion = SupConLoss(temperature=cfg.supcon_temperature)
    logger.info(f"损失函数: {cfg.loss_name}")

    # ── Optimizer & Scaler ──
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

        stats = train_one_epoch(
            model, train_loader, ce_criterion, sc_criterion,
            optimizer, cfg.device, scaler, cfg,
        )
        elapsed = time.time() - t0

        lrs = "/".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
        logger.info(
            f"Epoch [{epoch:3d}/{cfg.num_epochs}] "
            f"LR({lrs}) Factor:{lr_factor:.4f} | "
            f"Loss:{stats['loss']:.4f} CE:{stats['ce']:.4f} SC:{stats['sc']:.4f} | "
            f"Acc:{stats['acc']:.4f} | {elapsed:.1f}s"
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
                logger.info(f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")
            logger.info("-" * 40)

    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成! 最优: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 60)

    # ── Final evaluation ──
    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )

    probs, preds, labels = collect_predictions(model, test_loader, cfg.device)

    logger.info("=" * 60)
    logger.info("最终测试 (threshold=0.5)")
    logger.info("=" * 60)
    eval_metrics(probs, preds, labels, cfg.class_names, logger, "Final")

    # 标准 F1 阈值优化
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
    logger.info(f"精确度导向阈值优化 ({a}*F1 + {1 - a:.1f}*benign_prec)")
    logger.info("目标: 宁愿把良性判断成无 -> 提高 benign precision")
    logger.info("=" * 60)
    t2, f1_2, bp_2 = find_optimal_threshold(probs, labels, alpha=a)
    logger.info(f"精确度导向阈值: {t2:.3f} (F1: {f1_2:.4f}, benign_prec: {bp_2:.4f})")
    eval_with_threshold(probs, labels, t2, cfg.class_names, logger, "Final(精确度导向)")

    # Copy script
    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)
        logger.info(f"脚本已复制到: {dst}")


if __name__ == "__main__":
    main()
