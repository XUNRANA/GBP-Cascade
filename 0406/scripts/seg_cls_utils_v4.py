"""
Task 2 分割+分类联合训练工具库 v4
新增: 形态学特征 Dataset, Feature-Level Mixup 模型, Early Stopping, TTA, K-Fold

vs v3:
  - GBPDatasetSegCls4chWithExtMeta: 扩展 metadata (6D clinical + 12D morphological)
  - SwinV2SegGuidedCls4chModelV4: 支持 forward_features / forward_cls 拆分 (Feature Mixup)
  - EarlyStoppingRunner: 验证集 + patience-based early stopping + EMA
  - predict_with_tta: 测试时增强
  - run_kfold_experiment: K-Fold CV + TTA + 集成推理
"""

import copy
import os
import sys
import random
import shutil
import time

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

# Add 0402/scripts to path so seg_cls_utils_v3 → v2 → v1 chain resolves
_v4_dir = os.path.dirname(os.path.abspath(__file__))
_0402_scripts = os.path.join(_v4_dir, "../../0402/scripts")
if _0402_scripts not in sys.path:
    sys.path.insert(0, _0402_scripts)

# Re-export everything from v3 (which re-exports v2, which re-exports v1)
from seg_cls_utils_v3 import (
    # v1
    load_annotation, generate_lesion_mask, UNetDecoderBlock, DiceLoss,
    SegClsLoss, set_seed, setup_logger, acquire_run_lock,
    build_class_weights, cosine_warmup_factor, set_epoch_lrs,
    build_optimizer_with_diff_lr, compute_seg_metrics,
    find_optimal_threshold_v2, evaluate_with_threshold_v2,
    # v2
    SegCls4chSyncTransform, GBPDatasetSegCls4chWithMeta,
    seg_cls_4ch_meta_collate_fn,
    SwinV2SegGuidedCls4chModel,
    _unpack_batch, train_one_epoch_v2, evaluate_v2,
    # v3
    ModelEMA,
    # from test_yqh
    adapt_model_to_4ch, META_FEATURE_NAMES,
    build_case_meta_table, fit_meta_stats, encode_meta_row,
    extract_case_id_from_image_path,
)

# 形态学特征提取 (from LAT v1)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../0323/scripts"))
from task2_lat_v1 import compute_morph_features, NUM_MORPH_FEATURES


# ═══════════════════════════════════════════════════════════
#  扩展 Metadata Dataset (6D clinical + 12D morphological)
# ═══════════════════════════════════════════════════════════

EXT_META_FEATURE_NAMES = list(META_FEATURE_NAMES) + [
    "morph_area_ratio", "morph_circularity", "morph_eccentricity",
    "morph_solidity", "morph_extent", "morph_aspect_ratio",
    "morph_pa_ratio", "morph_convexity", "morph_equiv_diameter",
    "morph_norm_perimeter", "morph_num_components", "morph_largest_ratio",
]


def fit_ext_meta_stats(df, feature_names):
    """Compute mean/std for extended meta features (z-score normalization)."""
    stats = {}
    for col in feature_names:
        if col in df.columns:
            vals = df[col].dropna()
            stats[col] = {"mean": float(vals.mean()), "std": float(vals.std()) + 1e-8}
        else:
            stats[col] = {"mean": 0.0, "std": 1.0}
    return stats


def encode_ext_meta_row(row, stats, feature_names):
    """Encode a single row's extended meta features as a normalized tensor."""
    values = []
    for col in feature_names:
        val = row.get(col, np.nan)
        if pd.isna(val):
            values.append(0.0)
        else:
            s = stats[col]
            values.append((float(val) - s["mean"]) / s["std"])
    return torch.tensor(values, dtype=torch.float32)


class GBPDatasetSegCls4chWithExtMeta(Dataset):
    """4ch input + seg target + cls label + extended metadata (6D clinical + 12D morph)."""

    def __init__(self, excel_path, data_root, clinical_excel_path, json_feature_root,
                 sync_transform=None, meta_stats=None):
        self.df = pd.read_excel(excel_path).copy()
        self.data_root = data_root
        self.sync_transform = sync_transform

        # Clinical metadata
        meta_df = build_case_meta_table(clinical_excel_path, json_feature_root)
        self.df["case_id_norm"] = self.df["image_path"].map(extract_case_id_from_image_path)
        self.df = self.df.merge(meta_df, on="case_id_norm", how="left")

        # Precompute morphological features for each sample
        morph_names = EXT_META_FEATURE_NAMES[len(META_FEATURE_NAMES):]
        for col in morph_names:
            self.df[col] = 0.0

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            img_path = os.path.join(self.data_root, row["image_path"])
            json_path = img_path.replace(".png", ".json")

            shapes = []
            if os.path.exists(json_path):
                ann = load_annotation(json_path)
                shapes = ann.get("shapes", [])

            img = Image.open(img_path).convert("RGB")
            mask = generate_lesion_mask(shapes, img.size[0], img.size[1])
            morph = compute_morph_features(mask)

            for j, col in enumerate(morph_names):
                self.df.at[self.df.index[idx], col] = float(morph[j])

        self.ext_meta_feature_names = list(EXT_META_FEATURE_NAMES)
        for col in self.ext_meta_feature_names:
            if col not in self.df.columns:
                self.df[col] = np.nan

        self.meta_stats = (meta_stats if meta_stats is not None
                           else fit_ext_meta_stats(self.df, self.ext_meta_feature_names))
        self.meta_dim = len(self.ext_meta_feature_names)

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
        has_mask = False
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])
            has_mask = any(
                s["label"] != "gallbladder" and s["shape_type"] == "polygon" and len(s["points"]) >= 3
                for s in shapes
            )

        mask = generate_lesion_mask(shapes, img_w, img_h)

        if self.sync_transform:
            input_4ch, seg_target = self.sync_transform(img, mask)
        else:
            img_t = TF.to_tensor(img)
            mask_t = TF.to_tensor(mask)
            input_4ch = torch.cat([img_t, mask_t], dim=0)
            seg_target = (mask_t.squeeze(0) > 0.5).long()

        meta_tensor = encode_ext_meta_row(row, self.meta_stats, self.ext_meta_feature_names)
        return input_4ch, seg_target, meta_tensor, label, has_mask


# ═══════════════════════════════════════════════════════════
#  SwinV2 Model V4: 支持 forward_features / forward_cls 拆分
# ═══════════════════════════════════════════════════════════


class SwinV2SegGuidedCls4chModelV4(nn.Module):
    """SwinV2-Tiny@256 + 4ch + UNet seg + seg-guided attention cls + metadata.

    vs V2 版本: 支持 forward_features() / forward_cls() 拆分,
    便于在特征空间做 Mixup.
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=2, meta_dim=0,
                 meta_hidden=64, meta_dropout=0.2, cls_dropout=0.4, pretrained=True):
        super().__init__()
        self.meta_dim = meta_dim

        self.encoder = timm.create_model(
            "swinv2_tiny_window8_256", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
        )
        adapt_model_to_4ch(self.encoder)
        fc = [info["num_chs"] for info in self.encoder.feature_info]

        # Seg decoder
        self.dec3 = UNetDecoderBlock(fc[3], fc[2], fc[2])
        self.dec2 = UNetDecoderBlock(fc[2], fc[1], fc[1])
        self.dec1 = UNetDecoderBlock(fc[1], fc[0], fc[0])
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(fc[0], 48, kernel_size=4, stride=4),
            nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, num_seg_classes, 1),
        )

        # Seg-guided cls: attention pool on f2
        self.cls_proj = nn.Sequential(
            nn.Conv2d(fc[2], 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
        )

        # Metadata encoder
        self.meta_encoder = None
        fusion_in = 256
        if meta_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_dim, meta_hidden), nn.LayerNorm(meta_hidden),
                nn.GELU(), nn.Dropout(meta_dropout),
                nn.Linear(meta_hidden, meta_hidden), nn.GELU(), nn.Dropout(meta_dropout),
            )
            fusion_in = 256 + meta_hidden

        self.cls_mlp = nn.Sequential(
            nn.Linear(fusion_in, 128), nn.GELU(), nn.Dropout(cls_dropout),
            nn.Linear(128, 2),
        )
        self._fusion_in = fusion_in

    def _to_bchw(self, x):
        if x.ndim == 4 and x.shape[1] != x.shape[3]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward_features(self, x, metadata=None):
        """返回 (seg_logits, cls_feat) — cls_feat 是融合后的特征, 未过 cls_mlp."""
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        # Seg-guided attention
        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]
        attn = F.interpolate(seg_prob, size=f2.shape[2:], mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)

        f2_proj = self.cls_proj(f2)
        cls_feat = (f2_proj * attn).sum(dim=(2, 3))  # (B, 256)

        if self.meta_encoder is not None and metadata is not None:
            meta_feat = self.meta_encoder(metadata.float())
            cls_feat = torch.cat([cls_feat, meta_feat], dim=1)

        return seg_logits, cls_feat

    def forward_cls(self, cls_feat):
        """cls_feat → cls_logits."""
        return self.cls_mlp(cls_feat)

    def forward(self, x, metadata=None):
        seg_logits, cls_feat = self.forward_features(x, metadata)
        cls_logits = self.forward_cls(cls_feat)
        return seg_logits, cls_logits


import timm  # noqa: E402 (ensure available at module level)


# ═══════════════════════════════════════════════════════════
#  Feature-Level Mixup 训练函数
# ═══════════════════════════════════════════════════════════


def train_one_epoch_feature_mixup(model, dataloader, criterion, optimizer, device,
                                  scaler, use_amp, grad_clip=None,
                                  num_seg_classes=2, ema=None,
                                  mixup_prob=0.5, mixup_alpha=0.4):
    """Feature-level Mixup: 在 seg-guided attention 后的特征空间做 Mixup."""
    model.train()
    running_loss, running_seg_loss, running_cls_loss = 0.0, 0.0, 0.0
    cls_correct, cls_total = 0, 0
    all_seg_ious, all_seg_dices = [], []

    for batch in dataloader:
        imgs, masks, metas, labels, has_masks = _unpack_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        if metas is not None:
            metas = metas.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                enabled=use_amp):
            seg_logits, cls_feat = model.forward_features(imgs, metadata=metas)

            # Feature-level Mixup (only for classification branch)
            do_mixup = (random.random() < mixup_prob)
            if do_mixup:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx_perm = torch.randperm(cls_feat.size(0), device=device)
                cls_feat_mixed = lam * cls_feat + (1 - lam) * cls_feat[idx_perm]
                cls_logits = model.forward_cls(cls_feat_mixed)

                # Soft cross-entropy for mixed labels
                labels_onehot = F.one_hot(labels, num_classes=2).float()
                labels_perm = F.one_hot(labels[idx_perm], num_classes=2).float()
                labels_mixed = lam * labels_onehot + (1 - lam) * labels_perm
                cls_log_probs = F.log_softmax(cls_logits, dim=1)
                # Apply class weights
                cls_loss = -(labels_mixed * cls_log_probs * criterion.cls_ce.weight).sum(dim=1).mean()
            else:
                cls_logits = model.forward_cls(cls_feat)
                cls_loss = criterion.cls_ce(cls_logits, labels)

            # Seg loss (always unmixed)
            seg_loss = torch.tensor(0.0, device=device)
            if has_masks.any():
                mask_idx = has_masks.nonzero(as_tuple=True)[0]
                seg_logits_m = seg_logits[mask_idx]
                seg_targets_m = masks[mask_idx]
                seg_loss = criterion.seg_ce(seg_logits_m, seg_targets_m) + \
                           criterion.seg_dice(seg_logits_m, seg_targets_m)

            loss = seg_loss + criterion.lambda_cls * cls_loss

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_seg_loss += seg_loss.item() * bs
        running_cls_loss += cls_loss.item() * bs
        # For accuracy, use unmixed logits if mixup happened
        if do_mixup:
            with torch.no_grad():
                cls_logits_clean = model.forward_cls(cls_feat)
            cls_correct += (cls_logits_clean.argmax(dim=1) == labels).sum().item()
        else:
            cls_correct += (cls_logits.argmax(dim=1) == labels).sum().item()
        cls_total += bs

        if has_masks.any():
            with torch.no_grad():
                mask_idx = has_masks.nonzero(as_tuple=True)[0]
                metrics = compute_seg_metrics(seg_logits[mask_idx], masks[mask_idx], num_seg_classes)
                all_seg_ious.append(metrics["lesion_IoU"])
                all_seg_dices.append(metrics["lesion_Dice"])

    n = cls_total
    return {
        "loss": running_loss / n,
        "seg_loss": running_seg_loss / n,
        "cls_loss": running_cls_loss / n,
        "cls_acc": cls_correct / n,
        "seg_iou": np.mean(all_seg_ious) if all_seg_ious else 0.0,
        "seg_dice": np.mean(all_seg_dices) if all_seg_dices else 0.0,
    }


# ═══════════════════════════════════════════════════════════
#  TTA (Test-Time Augmentation)
# ═══════════════════════════════════════════════════════════


def tta_predict(model, imgs, metas, device):
    """对一个 batch 做 TTA (5 views), 返回平均 softmax 概率.

    views: 原图, hflip, vflip, hflip+vflip, 5-degree rotation
    """
    model.eval()
    all_probs = []

    with torch.no_grad():
        # View 1: 原图
        _, logits = model(imgs, metadata=metas)
        all_probs.append(F.softmax(logits, dim=1))

        # View 2: 水平翻转
        imgs_hf = torch.flip(imgs, dims=[3])
        _, logits = model(imgs_hf, metadata=metas)
        all_probs.append(F.softmax(logits, dim=1))

        # View 3: 垂直翻转
        imgs_vf = torch.flip(imgs, dims=[2])
        _, logits = model(imgs_vf, metadata=metas)
        all_probs.append(F.softmax(logits, dim=1))

        # View 4: 水平+垂直翻转
        imgs_hvf = torch.flip(imgs, dims=[2, 3])
        _, logits = model(imgs_hvf, metadata=metas)
        all_probs.append(F.softmax(logits, dim=1))

        # View 5: 轻微旋转 (+5 degrees, 通过 grid_sample)
        B, C, H, W = imgs.shape
        angle_rad = torch.tensor(5.0 * 3.14159 / 180.0)
        cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
        theta = torch.tensor([[cos_a, -sin_a, 0],
                              [sin_a, cos_a, 0]], dtype=torch.float32)
        theta = theta.unsqueeze(0).expand(B, -1, -1).to(device)
        grid = F.affine_grid(theta, imgs.size(), align_corners=False)
        imgs_rot = F.grid_sample(imgs, grid, mode="bilinear", align_corners=False)
        _, logits = model(imgs_rot, metadata=metas)
        all_probs.append(F.softmax(logits, dim=1))

    avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)  # (B, num_classes)
    return avg_probs


def evaluate_with_tta(model, dataloader, device, class_names, logger, phase="TTA Test"):
    """用 TTA 评估模型, 返回 F1."""
    model.eval()
    all_probs, all_labels = [], []

    for batch in dataloader:
        imgs, masks, metas, labels, has_masks = _unpack_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        if metas is not None:
            metas = metas.to(device, non_blocking=True)

        avg_probs = tta_predict(model, imgs, metas, device)
        all_probs.append(avg_probs.cpu())
        all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_labels = np.array(all_labels)
    all_preds = all_probs.argmax(axis=1)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(
        f"[{phase}] Cls — Acc: {acc:.4f} | P(macro): {precision:.4f} | "
        f"R(macro): {recall:.4f} | F1(macro): {f1:.4f}"
    )
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")

    # 阈值优化 (基于 TTA 概率)
    best_f1_t, best_thresh = 0.0, 0.5
    benign_probs = all_probs[:, 0]
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds_t = np.where(benign_probs >= thresh, 0, 1)
        f1_t = f1_score(all_labels, preds_t, average="macro", zero_division=0)
        if f1_t > best_f1_t:
            best_f1_t = f1_t
            best_thresh = thresh
    logger.info(f"[{phase}] 最优阈值: {best_thresh:.3f} (F1: {best_f1_t:.4f} vs 默认 F1: {f1:.4f})")

    return f1, best_f1_t, best_thresh


# ═══════════════════════════════════════════════════════════
#  Early Stopping 训练 Runner
# ═══════════════════════════════════════════════════════════


def run_experiment_with_early_stopping(
    cfg, build_model_fn, build_dataloaders_fn, build_optimizer_fn, script_path,
    train_fn=None,
):
    """v4 runner: 支持 train/val/test 三路, early stopping, EMA.

    build_dataloaders_fn 需返回:
      (train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader)
    """
    os.makedirs(cfg.log_dir, exist_ok=True)
    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_file, cfg.exp_name)

    logger.info("=" * 70)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"Batch Size: {cfg.batch_size} | Epochs: {cfg.num_epochs}")
    logger.info(f"LR: backbone={cfg.backbone_lr}, head={cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay} | Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Lambda Cls: {cfg.lambda_cls} | Grad Clip: {cfg.grad_clip}")
    patience = getattr(cfg, "patience", 10)
    ema_decay = getattr(cfg, "ema_decay", 0.9995)
    use_ema = getattr(cfg, "use_ema", True)
    logger.info(f"Early Stopping Patience: {patience}")
    logger.info(f"EMA: {use_ema} (decay={ema_decay})")
    logger.info(f"Seed: {cfg.seed} | Device: {cfg.device}")
    logger.info("=" * 70)

    result = build_dataloaders_fn(cfg)
    if len(result) == 6:
        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = result
    else:
        train_dataset, test_dataset, train_loader, test_loader = result
        val_dataset, val_loader = None, None

    logger.info(f"训练集: {len(train_dataset)} 张")
    if val_dataset is not None:
        logger.info(f"验证集: {len(val_dataset)} 张")
    logger.info(f"测试集: {len(test_dataset)} 张")

    model = build_model_fn(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {n_params:,}")

    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=ema_decay)
        logger.info(f"EMA 已启用 (decay={ema_decay})")

    # 用训练集 df 计算类别权重
    cls_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    logger.info(f"分类权重: benign={cls_weights[0]:.4f}, no_tumor={cls_weights[1]:.4f}")

    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight], dtype=torch.float32, device=cfg.device
    )
    criterion = SegClsLoss(
        cls_weights=cls_weights, lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing, seg_ce_weight=seg_ce_weight,
    )
    optimizer = build_optimizer_fn(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    eval_loader = val_loader if val_loader is not None else test_loader
    eval_phase = "Val" if val_loader is not None else "Test"

    if train_fn is None:
        train_fn = train_one_epoch_v2

    best_f1, best_epoch = 0.0, 0
    no_improve_count = 0

    logger.info("\n" + "=" * 70)
    logger.info("开始训练 (Early Stopping)")
    logger.info("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        # 选择训练函数
        if train_fn == train_one_epoch_feature_mixup:
            train_metrics = train_fn(
                model, train_loader, criterion, optimizer, cfg.device,
                scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
                grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
                ema=ema,
                mixup_prob=getattr(cfg, "mixup_prob", 0.5),
                mixup_alpha=getattr(cfg, "mixup_alpha", 0.4),
            )
        else:
            train_metrics = train_fn(
                model, train_loader, criterion, optimizer, cfg.device,
                scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
                grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
            )
            if ema is not None:
                ema.update(model)

        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f}, cls={train_metrics['cls_loss']:.4f}) "
            f"| Acc: {train_metrics['cls_acc']:.4f} "
            f"| Dice: {train_metrics['seg_dice']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            eval_model = ema.module if ema else model
            logger.info("-" * 50)
            acc, prec, rec, f1, seg_iou, seg_dice = evaluate_v2(
                eval_model, eval_loader, cfg.device, cfg.class_names, logger,
                phase=eval_phase + (" (EMA)" if ema else ""),
                num_seg_classes=cfg.num_seg_classes,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                no_improve_count = 0
                torch.save(eval_model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")
            else:
                no_improve_count += 1
                logger.info(f"无提升 ({no_improve_count}/{patience})")

            if no_improve_count >= patience:
                logger.info(f"Early stopping at epoch {epoch} (patience={patience})")
                break
            logger.info("-" * 50)

    logger.info("\n" + "=" * 70)
    logger.info(f"训练完成! 最优: Epoch {best_epoch}, {eval_phase} F1: {best_f1:.4f}")
    logger.info("=" * 70)

    # 加载最优权重, 在 test 集上最终评估
    eval_model = ema.module if ema else model
    eval_model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )

    logger.info("\n" + "=" * 70)
    logger.info("最终测试结果 (最优权重)")
    logger.info("=" * 70)
    evaluate_v2(eval_model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Final Test", num_seg_classes=cfg.num_seg_classes)

    # 阈值优化
    best_thresh, best_thresh_f1 = find_optimal_threshold_v2(eval_model, test_loader, cfg.device)
    logger.info(f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认 F1: {best_f1:.4f})")
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold_v2(
            eval_model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=best_thresh, phase="Final Test (最优阈值)",
        )

    # 复制脚本
    dst = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst):
        shutil.copy2(script_path, dst)
        logger.info(f"脚本已复制到: {dst}")
