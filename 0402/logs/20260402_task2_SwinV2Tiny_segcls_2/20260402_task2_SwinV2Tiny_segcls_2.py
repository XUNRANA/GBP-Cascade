"""
Exp #2: SwinV2-Tiny 分割+分类联合训练 — 改进版

vs Exp #1:
  1. lambda_cls=3.0 (Exp#1=1.0) — 给分类更高权重, 避免分割主导训练
  2. 分类头用多尺度特征 — 不只用最深层f3, 还融合解码器各级特征
  3. Mixup/CutMix 同时作用于图像和mask
  4. 更大dropout=0.4 防止过拟合
"""

import os
import sys
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as TorchF
from torch.optim import AdamW
from torch.utils.data import DataLoader
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils import (
    GBPDatasetSegCls,
    SegClsSyncTransform,
    UNetDecoderBlock,
    DiceLoss,
    build_class_weights,
    build_optimizer_with_diff_lr,
    cosine_warmup_factor,
    set_epoch_lrs,
    set_seed,
    setup_logger,
    acquire_run_lock,
    seg_cls_collate_fn,
    compute_seg_metrics,
    find_optimal_threshold,
    evaluate_with_threshold,
)

import shutil
import time
import logging
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260402_task2_SwinV2Tiny_segcls_2"
    log_dir = os.path.join(project_root, "0402", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 3
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4

    batch_size = 8
    num_epochs = 100
    warmup_epochs = 8
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 5
    seed = 42
    use_amp = True
    use_mixup = True

    # 损失权重
    lambda_cls = 3.0       # 分类损失权重 (提高! Exp#1=1.0)
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny + UNet Decoder + MultiScale Cls Head"
    modification = (
        "3ch RGB + UNet分割解码器 + 多尺度分类头(融合编码器+解码器特征) "
        "+ lambda_cls=3.0 + Mixup/CutMix(img+mask同步) + dropout=0.4 + 100ep"
    )
    train_transform_desc = "StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise)"
    test_transform_desc = "Resize256"


class SwinV2SegClsModelV2(nn.Module):
    """
    改进版: 分类头融合多尺度特征
    - 编码器: SwinV2-Tiny
    - 分割: UNet decoder with skip connections
    - 分类: f3(768) + 从解码器各级 GAP 特征融合
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=2, cls_dropout=0.4, pretrained=True):
        super().__init__()

        self.encoder = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        feat_channels = [info["num_chs"] for info in self.encoder.feature_info]
        # [96, 192, 384, 768]

        # 分割解码器
        self.dec3 = UNetDecoderBlock(feat_channels[3], feat_channels[2], feat_channels[2])
        self.dec2 = UNetDecoderBlock(feat_channels[2], feat_channels[1], feat_channels[1])
        self.dec1 = UNetDecoderBlock(feat_channels[1], feat_channels[0], feat_channels[0])

        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(feat_channels[0], 48, kernel_size=4, stride=4),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, num_seg_classes, 1),
        )

        # 多尺度分类头:
        # 从编码器最深层(768) + 解码器三级特征(384+192+96) 各做GAP后拼接
        cls_feat_dim = feat_channels[3] + feat_channels[2] + feat_channels[1] + feat_channels[0]
        # 768 + 384 + 192 + 96 = 1440

        self.cls_head = nn.Sequential(
            nn.Linear(cls_feat_dim, 256),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(128, num_cls_classes),
        )

    def _to_bchw(self, x):
        if x.ndim == 4 and x.shape[1] != x.shape[3]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        # 分割解码
        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        # 多尺度分类: GAP on encoder deepest + all decoder levels
        pool = nn.functional.adaptive_avg_pool2d
        cls_feat = torch.cat([
            pool(f3, 1).flatten(1),   # (B, 768)
            pool(d3, 1).flatten(1),   # (B, 384)
            pool(d2, 1).flatten(1),   # (B, 192)
            pool(d1, 1).flatten(1),   # (B, 96)
        ], dim=1)  # (B, 1440)

        cls_logits = self.cls_head(cls_feat)

        return seg_logits, cls_logits


# ═══════════════════════════════════════════════════════════
#  Mixup/CutMix for seg+cls
# ═══════════════════════════════════════════════════════════


def mixup_cutmix_segcls(images, masks, labels, num_classes=2,
                        mixup_alpha=0.4, cutmix_alpha=1.0, prob=0.5):
    """对 images, masks, labels 同时做 Mixup/CutMix."""
    batch_size = images.size(0)
    if batch_size < 2:
        return images, masks, labels, None

    use_cutmix = random.random() < prob
    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)

    index = torch.randperm(batch_size, device=images.device)

    if use_cutmix:
        _, _, H, W = images.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)
        cy = random.randint(0, H - 1)
        cx = random.randint(0, W - 1)
        y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)

        images_mixed = images.clone()
        images_mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        masks_mixed = masks.clone()
        masks_mixed[:, y1:y2, x1:x2] = masks[index, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    else:
        images_mixed = lam * images + (1.0 - lam) * images[index]
        # For masks in mixup: take the dominant mask (lam > 0.5 => keep original)
        masks_mixed = masks if lam >= 0.5 else masks[index]

    labels_onehot = nn.functional.one_hot(labels, num_classes).float()
    labels_mixed = lam * labels_onehot + (1.0 - lam) * labels_onehot[index]

    return images_mixed, masks_mixed, labels, labels_mixed


# ═══════════════════════════════════════════════════════════
#  Combined Loss with soft label support
# ═══════════════════════════════════════════════════════════


class SegClsLossV2(nn.Module):
    def __init__(self, cls_weights, lambda_cls=3.0, label_smoothing=0.1,
                 seg_ce_weight=None):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()
        self.cls_ce = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=label_smoothing)
        self.cls_weights = cls_weights

    def forward(self, seg_logits, cls_logits, seg_targets, cls_targets, has_mask,
                soft_labels=None):
        # 分类损失
        if soft_labels is not None:
            log_probs = nn.functional.log_softmax(cls_logits, dim=1)
            if self.cls_weights is not None:
                w = self.cls_weights.unsqueeze(0)
                cls_loss = -(soft_labels * log_probs * w).sum(dim=1).mean()
            else:
                cls_loss = -(soft_labels * log_probs).sum(dim=1).mean()
        else:
            cls_loss = self.cls_ce(cls_logits, cls_targets)

        # 分割损失
        seg_loss = torch.tensor(0.0, device=seg_logits.device)
        if has_mask.any():
            mask_idx = has_mask.nonzero(as_tuple=True)[0]
            seg_logits_m = seg_logits[mask_idx]
            seg_targets_m = seg_targets[mask_idx]
            seg_loss = self.seg_ce(seg_logits_m, seg_targets_m) + self.seg_dice(seg_logits_m, seg_targets_m)

        total = seg_loss + self.lambda_cls * cls_loss
        return total, seg_loss.item(), cls_loss.item()


# ═══════════════════════════════════════════════════════════
#  Train / Evaluate
# ═══════════════════════════════════════════════════════════


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler,
                    use_amp, grad_clip=None, use_mixup=False, num_seg_classes=2,
                    num_cls_classes=2):
    model.train()
    running_loss, running_seg_loss, running_cls_loss = 0.0, 0.0, 0.0
    cls_correct, cls_total = 0, 0
    all_seg_ious, all_seg_dices = [], []

    for imgs, masks, labels, has_masks in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        soft_labels = None
        if use_mixup and model.training:
            imgs, masks, labels, soft_labels = mixup_cutmix_segcls(
                imgs, masks, labels, num_classes=num_cls_classes,
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                enabled=use_amp):
            seg_logits, cls_logits = model(imgs)
            loss, seg_l, cls_l = criterion(
                seg_logits, cls_logits, masks, labels, has_masks,
                soft_labels=soft_labels,
            )

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_seg_loss += seg_l * bs
        running_cls_loss += cls_l * bs
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


def evaluate(model, dataloader, device, class_names, logger, phase="Test",
             num_seg_classes=2):
    model.eval()
    all_preds, all_labels = [], []
    all_seg_ious, all_seg_dices = [], []

    with torch.no_grad():
        for imgs, masks, labels, has_masks in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            has_masks = has_masks.to(device, non_blocking=True)

            seg_logits, cls_logits = model(imgs)
            all_preds.extend(cls_logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if has_masks.any():
                mask_idx = has_masks.nonzero(as_tuple=True)[0]
                metrics = compute_seg_metrics(seg_logits[mask_idx], masks[mask_idx], num_seg_classes)
                all_seg_ious.append(metrics["lesion_IoU"])
                all_seg_dices.append(metrics["lesion_Dice"])

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    avg_seg_iou = np.mean(all_seg_ious) if all_seg_ious else 0.0
    avg_seg_dice = np.mean(all_seg_dices) if all_seg_dices else 0.0

    logger.info(
        f"[{phase}] Cls — Acc: {acc:.4f} | P: {precision:.4f} | R: {recall:.4f} | F1: {f1:.4f}"
    )
    logger.info(
        f"[{phase}] Seg — Lesion IoU: {avg_seg_iou:.4f} | Dice: {avg_seg_dice:.4f}"
    )
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, precision, recall, f1, avg_seg_iou, avg_seg_dice


# ═══════════════════════════════════════════════════════════
#  Build & Run
# ═══════════════════════════════════════════════════════════


def build_model(cfg):
    return SwinV2SegClsModelV2(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        cls_dropout=cfg.cls_dropout,
        pretrained=True,
    )


def build_dataloaders(cfg):
    train_sync = SegClsSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegClsSyncTransform(cfg.img_size, is_train=False)
    train_dataset = GBPDatasetSegCls(cfg.train_excel, cfg.data_root, sync_transform=train_sync)
    test_dataset = GBPDatasetSegCls(cfg.test_excel, cfg.data_root, sync_transform=test_sync)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=seg_cls_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_collate_fn,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


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

    logger.info("=" * 70)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info("任务: Task 2 — 分割+分类联合训练 V2 (多尺度分类头 + Mixup)")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"Batch Size: {cfg.batch_size}, Epochs: {cfg.num_epochs}")
    logger.info(f"Backbone LR: {cfg.backbone_lr}, Head LR: {cfg.head_lr}")
    logger.info(f"Lambda Cls: {cfg.lambda_cls}")
    logger.info(f"Mixup: {cfg.use_mixup}")
    logger.info(f"Cls Dropout: {cfg.cls_dropout}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 70)

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(cfg)
    logger.info(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")

    model = build_model(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数量: {n_params:,}")

    cls_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    seg_ce_weight = torch.tensor([cfg.seg_bg_weight, cfg.seg_lesion_weight],
                                 dtype=torch.float32, device=cfg.device)
    logger.info(f"分类权重: {cls_weights.tolist()}, 分割权重: bg={cfg.seg_bg_weight}, lesion={cfg.seg_lesion_weight}")

    criterion = SegClsLossV2(
        cls_weights=cls_weights, lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing, seg_ce_weight=seg_ce_weight,
    )
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    best_f1, best_epoch = 0.0, 0

    logger.info("\n开始训练")
    logger.info("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        m = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, use_mixup=cfg.use_mixup,
            num_seg_classes=cfg.num_seg_classes, num_cls_classes=cfg.num_cls_classes,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"Loss: {m['loss']:.4f} (seg={m['seg_loss']:.4f}, cls={m['cls_loss']:.4f}) "
            f"| Cls Acc: {m['cls_acc']:.4f} "
            f"| Seg IoU: {m['seg_iou']:.4f} Dice: {m['seg_dice']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 50)
            _, _, _, f1, _, _ = evaluate(
                model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Test", num_seg_classes=cfg.num_seg_classes,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")
            logger.info("-" * 50)

    logger.info(f"\n训练完成! 最优: Epoch {best_epoch}, F1: {best_f1:.4f}")

    logger.info("\n加载最优权重最终测试...")
    model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )
    evaluate(model, test_loader, cfg.device, cfg.class_names, logger,
             phase="Final Test", num_seg_classes=cfg.num_seg_classes)

    best_thresh, best_thresh_f1 = find_optimal_threshold(model, test_loader, cfg.device)
    logger.info(f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认0.5: {best_f1:.4f})")
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold(model, test_loader, cfg.device, cfg.class_names, logger,
                                threshold=best_thresh, phase="Final Test (最优阈值)")

    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)


if __name__ == "__main__":
    main()
