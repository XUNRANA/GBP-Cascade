"""
Task 2 分割+分类联合训练工具库 v3
新增: 强增强, EMA, SupConLoss, 多尺度Attention模型, v3实验runner

vs v2:
  - SegCls4chStrongSyncTransform: 更强数据增强
  - ModelEMA: 指数滑动平均
  - SupConLoss: 监督对比学习损失
  - SwinV2MultiScaleSegGuidedCls4chModel: f2+f3 双尺度 seg-guided attention
  - run_seg_cls_experiment_v3: 支持 EMA
"""

import copy
import os
import sys
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import timm

# Re-export everything from v2
from seg_cls_utils_v2 import (
    # v1 re-exports
    load_annotation, generate_lesion_mask, UNetDecoderBlock, DiceLoss,
    SegClsLoss, set_seed, setup_logger, acquire_run_lock,
    build_class_weights, cosine_warmup_factor, set_epoch_lrs,
    build_optimizer_with_diff_lr, compute_seg_metrics,
    find_optimal_threshold_v2, evaluate_with_threshold_v2,
    # v2
    SegCls4chSyncTransform, GBPDatasetSegCls4ch, GBPDatasetSegCls4chWithMeta,
    seg_cls_4ch_collate_fn, seg_cls_4ch_meta_collate_fn,
    SwinV2SegCls4chModel, SwinV2SegGuidedCls4chModel,
    _unpack_batch, train_one_epoch_v2, evaluate_v2,
    run_seg_cls_experiment_v2,
    # from test_yqh
    adapt_model_to_4ch, META_FEATURE_NAMES,
    build_case_meta_table, fit_meta_stats, encode_meta_row,
    extract_case_id_from_image_path,
)

import shutil
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)


# ═══════════════════════════════════════════════════════════
#  Stronger Augmentation Transform
# ═══════════════════════════════════════════════════════════


class SegCls4chStrongSyncTransform:
    """更强的同步变换 (vs SegCls4chSyncTransform):
    - 旋转 ±30° (原 ±20°)
    - RRC scale (0.5, 1.0) (原 0.7, 1.0)
    - 仿射 scale (0.8, 1.2), shear ±10° (原 0.9/1.1, ±5°)
    - 更强色彩抖动
    - 更频繁模糊、擦除、噪声
    """

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        size = [self.img_size, self.img_size]

        if self.is_train:
            # Stronger random resized crop
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=(0.5, 1.0), ratio=(0.8, 1.2),
            )
            img = TF.resized_crop(img, i, j, h, w, size, InterpolationMode.BICUBIC)
            mask = TF.resized_crop(mask, i, j, h, w, size, InterpolationMode.NEAREST)

            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.4:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            # Stronger rotation ±30°
            if random.random() < 0.6:
                angle = random.uniform(-30, 30)
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BICUBIC, fill=0)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
            # Stronger affine
            if random.random() < 0.5:
                angle = random.uniform(-8, 8)
                max_t = 0.08 * self.img_size
                translate = [int(random.uniform(-max_t, max_t)),
                             int(random.uniform(-max_t, max_t))]
                scale = random.uniform(0.8, 1.2)
                shear = [random.uniform(-10, 10)]
                img = TF.affine(img, angle, translate, scale, shear,
                                interpolation=InterpolationMode.BICUBIC, fill=0)
                mask = TF.affine(mask, angle, translate, scale, shear,
                                 interpolation=InterpolationMode.NEAREST, fill=0)
            # Stronger color jitter
            if random.random() < 0.7:
                img = TF.adjust_brightness(img, random.uniform(0.6, 1.4))
                img = TF.adjust_contrast(img, random.uniform(0.6, 1.4))
                img = TF.adjust_saturation(img, random.uniform(0.7, 1.3))
                img = TF.adjust_hue(img, random.uniform(-0.05, 0.05))
            # More frequent blur
            if random.random() < 0.3:
                img = TF.gaussian_blur(img, kernel_size=3)
        else:
            img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
            mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        img_t = TF.normalize(img_t, self.mean, self.std)

        if self.is_train:
            # More aggressive erasing
            if random.random() < 0.3:
                img_t = T.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3))(img_t)
            # More noise
            if random.random() < 0.4:
                img_t = img_t + torch.randn_like(img_t) * 0.05
            # Random grayscale
            if random.random() < 0.1:
                gray = img_t.mean(dim=0, keepdim=True)
                img_t = gray.expand_as(img_t)

        input_4ch = torch.cat([img_t, mask_t], dim=0)
        seg_target = (mask_t.squeeze(0) > 0.5).long()
        return input_4ch, seg_target


# ═══════════════════════════════════════════════════════════
#  EMA (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════


class ModelEMA:
    """Exponential Moving Average of model parameters."""

    def __init__(self, model, decay=0.999):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)
        for ema_b, model_b in zip(self.module.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)


# ═══════════════════════════════════════════════════════════
#  Supervised Contrastive Loss
# ═══════════════════════════════════════════════════════════


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (SupCon)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D) — will be L2 normalized
        labels: (B,) — class labels
        """
        device = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        features = F.normalize(features, dim=1)
        sim = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        self_mask = torch.eye(B, device=device)
        pos_mask = pos_mask * (1 - self_mask)  # exclude self
        neg_mask = 1 - self_mask

        # For numerical stability
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * neg_mask
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-6)

        pos_count = pos_mask.sum(dim=1)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-6)

        # Only compute for samples that have at least 1 positive pair
        valid = pos_count > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device)

        loss = -mean_log_prob[valid].mean()
        return loss


# ═══════════════════════════════════════════════════════════
#  Multi-Scale Seg-Guided Attention Model
# ═══════════════════════════════════════════════════════════


class SwinV2MultiScaleSegGuidedCls4chModel(nn.Module):
    """SwinV2-Tiny@256 + 4ch + UNet seg + DUAL-SCALE seg-guided attention + metadata.

    Uses both f2 (384ch, 16x16) and f3 (768ch, 8x8) with seg-guided attention.
    Two-scale features concatenated → fusion MLP → classification.
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=2, meta_dim=0,
                 meta_hidden=64, meta_dropout=0.2, cls_dropout=0.4,
                 drop_path_rate=0.0, pretrained=True):
        super().__init__()
        self.meta_dim = meta_dim

        self.encoder = timm.create_model(
            "swinv2_tiny_window8_256", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate,
        )
        adapt_model_to_4ch(self.encoder)
        fc = [info["num_chs"] for info in self.encoder.feature_info]  # [96,192,384,768]

        # Seg decoder (same as v2)
        self.dec3 = UNetDecoderBlock(fc[3], fc[2], fc[2])
        self.dec2 = UNetDecoderBlock(fc[2], fc[1], fc[1])
        self.dec1 = UNetDecoderBlock(fc[1], fc[0], fc[0])
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(fc[0], 48, kernel_size=4, stride=4),
            nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, num_seg_classes, 1),
        )

        # Scale 1: f2 (384ch, 16x16) → 256d
        self.cls_proj_fine = nn.Sequential(
            nn.Conv2d(fc[2], 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
        )
        # Scale 2: f3 (768ch, 8x8) → 256d
        self.cls_proj_coarse = nn.Sequential(
            nn.Conv2d(fc[3], 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
        )

        # Optional metadata
        self.meta_encoder = None
        fusion_in = 256 + 256  # fine + coarse
        if meta_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_dim, meta_hidden), nn.LayerNorm(meta_hidden),
                nn.GELU(), nn.Dropout(meta_dropout),
                nn.Linear(meta_hidden, meta_hidden), nn.GELU(), nn.Dropout(meta_dropout),
            )
            fusion_in += meta_hidden

        self.cls_mlp = nn.Sequential(
            nn.Linear(fusion_in, 256), nn.GELU(), nn.Dropout(cls_dropout),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(cls_dropout * 0.5),
            nn.Linear(128, num_cls_classes),
        )

    def _to_bchw(self, x):
        if x.ndim == 4 and x.shape[1] != x.shape[3]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, metadata=None):
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]  # (B, 1, H, W)

        # Fine scale: f2 (16x16)
        attn_fine = F.interpolate(seg_prob, size=f2.shape[2:], mode="bilinear", align_corners=False)
        attn_fine = attn_fine + 0.1
        attn_fine = attn_fine / (attn_fine.sum(dim=(2, 3), keepdim=True) + 1e-6)
        fine_feat = (self.cls_proj_fine(f2) * attn_fine).sum(dim=(2, 3))  # (B, 256)

        # Coarse scale: f3 (8x8)
        attn_coarse = F.interpolate(seg_prob, size=f3.shape[2:], mode="bilinear", align_corners=False)
        attn_coarse = attn_coarse + 0.1
        attn_coarse = attn_coarse / (attn_coarse.sum(dim=(2, 3), keepdim=True) + 1e-6)
        coarse_feat = (self.cls_proj_coarse(f3) * attn_coarse).sum(dim=(2, 3))  # (B, 256)

        cls_feat = torch.cat([fine_feat, coarse_feat], dim=1)  # (B, 512)

        if self.meta_encoder is not None and metadata is not None:
            meta_feat = self.meta_encoder(metadata.float())
            cls_feat = torch.cat([cls_feat, meta_feat], dim=1)

        cls_logits = self.cls_mlp(cls_feat)
        return seg_logits, cls_logits


# ═══════════════════════════════════════════════════════════
#  v3 Training with EMA support
# ═══════════════════════════════════════════════════════════


def train_one_epoch_v3(model, dataloader, criterion, optimizer, device, scaler,
                       use_amp, grad_clip=None, num_seg_classes=2, ema=None):
    """Like v2, but updates EMA after each step."""
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
            seg_logits, cls_logits = model(imgs, metadata=metas)
            loss, seg_l, cls_l = criterion(seg_logits, cls_logits, masks, labels, has_masks)

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
        running_seg_loss += seg_l * bs
        running_cls_loss += cls_l * bs
        cls_correct += (cls_logits.argmax(dim=1) == labels).sum().item()
        cls_total += bs

        if has_masks.any():
            with torch.no_grad():
                mask_idx = has_masks.nonzero(as_tuple=True)[0]
                metrics = compute_seg_metrics(
                    seg_logits[mask_idx], masks[mask_idx], num_seg_classes
                )
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


def run_seg_cls_experiment_v3(cfg, build_model_fn, build_dataloaders_fn,
                              build_optimizer_fn, script_path):
    """v3 experiment runner — like v2 but supports EMA."""
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
    logger.info(f"输入通道: {cfg.in_channels}")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"分割类别: {cfg.num_seg_classes}")
    logger.info(f"分类类别: {cfg.class_names}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Backbone LR: {cfg.backbone_lr}")
    logger.info(f"Head LR: {cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Warmup Epochs: {cfg.warmup_epochs}")
    logger.info(f"Lambda Cls: {cfg.lambda_cls}")
    logger.info(f"Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"Grad Clip: {cfg.grad_clip}")
    use_ema = getattr(cfg, "use_ema", False)
    ema_decay = getattr(cfg, "ema_decay", 0.999)
    logger.info(f"EMA: {use_ema} (decay={ema_decay})")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 70)

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders_fn(cfg)

    logger.info(
        f"训练集: {len(train_dataset)} 张 "
        f"(benign={sum(train_dataset.df['label'] == 0)}, "
        f"no_tumor={sum(train_dataset.df['label'] == 1)})"
    )
    logger.info(
        f"测试集: {len(test_dataset)} 张 "
        f"(benign={sum(test_dataset.df['label'] == 0)}, "
        f"no_tumor={sum(test_dataset.df['label'] == 1)})"
    )

    model = build_model_fn(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {n_params:,}")

    ema = None
    if use_ema:
        ema = ModelEMA(model, decay=ema_decay)
        logger.info(f"EMA 已启用 (decay={ema_decay})")

    cls_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    logger.info(f"分类类别权重: benign={cls_weights[0]:.4f}, no_tumor={cls_weights[1]:.4f}")

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

    best_f1, best_epoch = 0.0, 0

    logger.info("\n" + "=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        train_metrics = train_one_epoch_v3(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
            ema=ema,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f}, cls={train_metrics['cls_loss']:.4f}) "
            f"| Cls Acc: {train_metrics['cls_acc']:.4f} "
            f"| Seg Dice: {train_metrics['seg_dice']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            eval_model = ema.module if ema else model
            logger.info("-" * 50)
            acc, prec, rec, f1, seg_iou, seg_dice = evaluate_v2(
                eval_model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Test" + (" (EMA)" if ema else ""),
                num_seg_classes=cfg.num_seg_classes,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                save_dict = eval_model.state_dict()
                torch.save(save_dict, cfg.best_weight_path)
                logger.info(
                    f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***"
                )
            logger.info("-" * 50)

    logger.info("\n" + "=" * 70)
    logger.info(f"训练完成! 最优模型: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 70)

    # Load best and final eval
    eval_model = ema.module if ema else model
    eval_model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )
    logger.info("=" * 70)
    logger.info("最终测试结果 (最优权重)")
    logger.info("=" * 70)
    evaluate_v2(eval_model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Final Test", num_seg_classes=cfg.num_seg_classes)

    logger.info("\n" + "=" * 70)
    logger.info("阈值优化搜索")
    logger.info("=" * 70)
    best_thresh, best_thresh_f1 = find_optimal_threshold_v2(eval_model, test_loader, cfg.device)
    logger.info(
        f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认0.5 F1: {best_f1:.4f})"
    )
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold_v2(
            eval_model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=best_thresh, phase="Final Test (最优阈值)",
        )

    dst = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst):
        shutil.copy2(script_path, dst)
        logger.info(f"训练脚本已复制到: {dst}")
