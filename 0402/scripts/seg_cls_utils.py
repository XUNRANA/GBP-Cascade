"""
Task 2 分割+分类联合训练工具库
模仿 unet-valjustclass 项目思路:
  - SwinV2-Tiny 作为共享编码器 (features_only=True, 4级层次特征)
  - UNet 解码器做病灶分割
  - 分类头从最深层特征做 benign/no_tumor 分类
  - 联合训练: seg_loss + cls_loss

核心区别 vs 之前的实验:
  - 输入 3ch RGB (不再把 mask 当第4通道输入)
  - mask 变成分割的监督目标
  - 分割迫使模型学习病灶的空间特征, 从而提升分类
"""

import atexit
import inspect
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import timm


# ═══════════════════════════════════════════════════════════
#  Part 1: JSON 标注解析
# ═══════════════════════════════════════════════════════════


def load_annotation(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_lesion_mask(shapes, width, height):
    """将所有非 gallbladder 的多边形标注 -> 二值 mask (PIL Image 'L' mode)."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for s in shapes:
        if s["label"] == "gallbladder":
            continue
        if s["shape_type"] == "polygon" and len(s["points"]) >= 3:
            pts = [(p[0], p[1]) for p in s["points"]]
            draw.polygon(pts, fill=255)
    return mask


# ═══════════════════════════════════════════════════════════
#  Part 2: Transforms (同步变换 — 图像和 mask 一起变)
# ═══════════════════════════════════════════════════════════


class SegClsSyncTransform:
    """
    分割+分类同步变换:
    - 对 RGB 图像和分割 mask 施加相同的几何变换
    - 颜色变换仅作用于 RGB
    - 返回 img_tensor [3, H, W] 和 mask_tensor [H, W] (long 类型, 0/1)
    """

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        size = [self.img_size, self.img_size]

        if self.is_train:
            # RandomResizedCrop (同步)
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=(0.7, 1.0), ratio=(0.85, 1.15),
            )
            img = TF.resized_crop(img, i, j, h, w, size, InterpolationMode.BICUBIC)
            mask = TF.resized_crop(mask, i, j, h, w, size, InterpolationMode.NEAREST)

            # Horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() < 0.3:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            # Random rotation +-20
            if random.random() < 0.5:
                angle = random.uniform(-20, 20)
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BICUBIC, fill=0)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)

            # Random affine (shear + translate)
            if random.random() < 0.5:
                angle = random.uniform(-5, 5)
                max_t = 0.06 * self.img_size
                translate = [int(random.uniform(-max_t, max_t)),
                             int(random.uniform(-max_t, max_t))]
                scale = random.uniform(0.9, 1.1)
                shear = [random.uniform(-5, 5)]
                img = TF.affine(img, angle, translate, scale, shear,
                                interpolation=InterpolationMode.BICUBIC, fill=0)
                mask = TF.affine(mask, angle, translate, scale, shear,
                                 interpolation=InterpolationMode.NEAREST, fill=0)

            # Color jitter (RGB only)
            if random.random() < 0.6:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
                img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))

            # Gaussian blur
            if random.random() < 0.2:
                img = TF.gaussian_blur(img, kernel_size=3)
        else:
            img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
            mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        # To tensor
        img_t = TF.to_tensor(img)   # [3, H, W], float 0~1
        mask_t = TF.to_tensor(mask)  # [1, H, W], float 0 or 1

        # Normalize RGB
        img_t = TF.normalize(img_t, self.mean, self.std)

        # Random erasing (tensor level, RGB only)
        if self.is_train and random.random() < 0.2:
            img_t = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))(img_t)

        # Gaussian noise (tensor level)
        if self.is_train and random.random() < 0.3:
            noise = torch.randn_like(img_t) * 0.03
            img_t = img_t + noise

        # mask -> [H, W] long (0 or 1)
        mask_t = (mask_t.squeeze(0) > 0.5).long()

        return img_t, mask_t


# ═══════════════════════════════════════════════════════════
#  Part 3: Dataset
# ═══════════════════════════════════════════════════════════


class GBPDatasetSegCls(Dataset):
    """
    分割+分类联合数据集:
    - 输入: 3ch RGB 图像
    - 输出: img_tensor [3,H,W], seg_mask [H,W] (0/1), cls_label (int), has_mask (bool)
    """

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

        # 生成分割 mask
        shapes = []
        has_mask = False
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])
            # 检查是否有病灶多边形
            has_mask = any(
                s["label"] != "gallbladder" and s["shape_type"] == "polygon" and len(s["points"]) >= 3
                for s in shapes
            )

        mask = generate_lesion_mask(shapes, img_w, img_h)

        if self.sync_transform:
            img_t, mask_t = self.sync_transform(img, mask)
        else:
            img_t = TF.to_tensor(img)
            mask_t = (TF.to_tensor(mask).squeeze(0) > 0.5).long()

        return img_t, mask_t, label, has_mask


# ═══════════════════════════════════════════════════════════
#  Part 4: Model — SwinV2-Tiny + UNet Decoder + Cls Head
# ═══════════════════════════════════════════════════════════


class UNetDecoderBlock(nn.Module):
    """UNet 解码块: 上采样 + skip concat + 卷积."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.GELU(),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # 确保尺寸匹配
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SwinV2SegClsModel(nn.Module):
    """
    SwinV2-Tiny 双任务模型:
    - 编码器: SwinV2-Tiny (features_only=True, 4级层次特征)
    - 分割解码器: UNet-style with skip connections
    - 分类头: 从最深特征做全局池化 + MLP

    SwinV2-Tiny@256 特征层级:
      Stage 0: 64x64, 96ch  (4x downsample)
      Stage 1: 32x32, 192ch (8x downsample)
      Stage 2: 16x16, 384ch (16x downsample)
      Stage 3: 8x8,   768ch (32x downsample)

    解码器路径:
      8x8,768 -> up+skip(384) -> 16x16,384
      16x16,384 -> up+skip(192) -> 32x32,192
      32x32,192 -> up+skip(96) -> 64x64,96
      64x64,96 -> up(4x) -> 256x256 -> 1x1 conv -> num_seg_classes
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=2, cls_dropout=0.3, pretrained=True):
        super().__init__()

        # 编码器: SwinV2-Tiny, features_only=True 提取多尺度特征
        self.encoder = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        # 特征通道: [96, 192, 384, 768]
        feat_channels = [info["num_chs"] for info in self.encoder.feature_info]
        # feat_channels = [96, 192, 384, 768]

        # 分割解码器 (UNet decoder with skip connections)
        self.dec3 = UNetDecoderBlock(feat_channels[3], feat_channels[2], feat_channels[2])  # 768->384
        self.dec2 = UNetDecoderBlock(feat_channels[2], feat_channels[1], feat_channels[1])  # 384->192
        self.dec1 = UNetDecoderBlock(feat_channels[1], feat_channels[0], feat_channels[0])  # 192->96

        # 最终上采样: 64x64 -> 256x256 (4x)
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(feat_channels[0], 48, kernel_size=4, stride=4),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.GroupNorm(8, 48),
            nn.GELU(),
            nn.Conv2d(48, num_seg_classes, 1),
        )

        # 分类头: 从最深特征 (Stage 3) 做全局池化 + MLP
        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feat_channels[3], 256),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(256, num_cls_classes),
        )

    def _to_bchw(self, x):
        """SwinV2 outputs (B, H, W, C), convert to (B, C, H, W)."""
        if x.ndim == 4 and x.shape[1] != x.shape[3]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x):
        # 编码器提取4级特征
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]
        # f0: (B, 96, 64, 64)
        # f1: (B, 192, 32, 32)
        # f2: (B, 384, 16, 16)
        # f3: (B, 768, 8, 8)

        # 分割解码器 (UNet path)
        d3 = self.dec3(f3, f2)   # (B, 384, 16, 16)
        d2 = self.dec2(d3, f1)   # (B, 192, 32, 32)
        d1 = self.dec1(d2, f0)   # (B, 96, 64, 64)
        seg_logits = self.seg_final(d1)  # (B, num_seg_classes, 256, 256)

        # 分类头 (从最深特征)
        cls_logits = self.cls_head(f3)  # (B, num_cls_classes)

        return seg_logits, cls_logits


# ═══════════════════════════════════════════════════════════
#  Part 5: Loss Functions
# ═══════════════════════════════════════════════════════════


class DiceLoss(nn.Module):
    """Binary Dice Loss for segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B, C, H, W) - raw logits
        targets: (B, H, W) - long tensor with class indices
        """
        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]
        dice = 0.0
        for c in range(num_classes):
            pred_c = probs[:, c]
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()
            dice += (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice / num_classes


class SegClsLoss(nn.Module):
    """
    联合损失: seg_loss + lambda_cls * cls_loss
    - seg_loss: CE + Dice (只在有 mask 的样本上计算)
    - cls_loss: CE with class weights + label smoothing
    """

    def __init__(self, cls_weights, lambda_cls=1.0, label_smoothing=0.1,
                 seg_ce_weight=None):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()
        self.cls_ce = nn.CrossEntropyLoss(weight=cls_weights, label_smoothing=label_smoothing)

    def forward(self, seg_logits, cls_logits, seg_targets, cls_targets, has_mask):
        """
        seg_logits: (B, num_seg_classes, H, W)
        cls_logits: (B, num_cls_classes)
        seg_targets: (B, H, W) long
        cls_targets: (B,) long
        has_mask: (B,) bool - 哪些样本有分割标注
        """
        # 分类损失 (所有样本)
        cls_loss = self.cls_ce(cls_logits, cls_targets)

        # 分割损失 (只在有 mask 的样本上)
        seg_loss = torch.tensor(0.0, device=seg_logits.device)
        if has_mask.any():
            mask_idx = has_mask.nonzero(as_tuple=True)[0]
            seg_logits_masked = seg_logits[mask_idx]
            seg_targets_masked = seg_targets[mask_idx]
            seg_ce_loss = self.seg_ce(seg_logits_masked, seg_targets_masked)
            seg_dice_loss = self.seg_dice(seg_logits_masked, seg_targets_masked)
            seg_loss = seg_ce_loss + seg_dice_loss

        total = seg_loss + self.lambda_cls * cls_loss
        return total, seg_loss.item(), cls_loss.item()


# ═══════════════════════════════════════════════════════════
#  Part 6: Training Utilities
# ═══════════════════════════════════════════════════════════


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_file, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _pid_is_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(lock_path):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                existing_pid = int(f.read().strip() or "0")
        except (OSError, ValueError):
            existing_pid = 0
        if _pid_is_alive(existing_pid):
            return False, existing_pid
        try:
            os.remove(lock_path)
        except OSError:
            pass
    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    def _cleanup():
        try:
            if os.path.exists(lock_path):
                with open(lock_path, "r", encoding="utf-8") as f:
                    owner = int(f.read().strip() or "0")
                if owner == os.getpid():
                    os.remove(lock_path)
        except (OSError, ValueError):
            pass

    atexit.register(_cleanup)
    return True, os.getpid()


def build_class_weights(train_df, class_names, device):
    label_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    weights = [total / (len(class_names) * int(label_counts[i]))
               for i in range(len(class_names))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def cosine_warmup_factor(epoch, num_epochs, warmup_epochs, min_lr_ratio):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return epoch / warmup_epochs
    if num_epochs <= warmup_epochs:
        return 1.0
    progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def set_epoch_lrs(optimizer, epoch, cfg):
    factor = cosine_warmup_factor(epoch, cfg.num_epochs, cfg.warmup_epochs, cfg.min_lr_ratio)
    for pg in optimizer.param_groups:
        pg["lr"] = pg.get("base_lr", pg["lr"]) * factor
    return factor


def build_optimizer_with_diff_lr(optimizer_cls, backbone_params, head_params, cfg):
    return optimizer_cls(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr, "base_lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr, "base_lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


def seg_cls_collate_fn(batch):
    """Custom collate: handle has_mask bool."""
    imgs, masks, labels, has_masks = zip(*batch)
    imgs = torch.stack(imgs)
    masks = torch.stack(masks)
    labels = torch.tensor(labels, dtype=torch.long)
    has_masks = torch.tensor(has_masks, dtype=torch.bool)
    return imgs, masks, labels, has_masks


# ═══════════════════════════════════════════════════════════
#  Part 7: Segmentation Metrics
# ═══════════════════════════════════════════════════════════


def compute_seg_metrics(pred_logits, targets, num_classes=2):
    """Compute per-class IoU and Dice for segmentation."""
    preds = pred_logits.argmax(dim=1)  # (B, H, W)
    ious = []
    dices = []
    for c in range(num_classes):
        pred_c = (preds == c)
        target_c = (targets == c)
        intersection = (pred_c & target_c).sum().float()
        union = (pred_c | target_c).sum().float()
        iou = (intersection + 1e-6) / (union + 1e-6)
        dice = (2.0 * intersection + 1e-6) / (pred_c.sum().float() + target_c.sum().float() + 1e-6)
        ious.append(iou.item())
        dices.append(dice.item())
    return {
        "mIoU": np.mean(ious),
        "mDice": np.mean(dices),
        "lesion_IoU": ious[1] if num_classes > 1 else ious[0],
        "lesion_Dice": dices[1] if num_classes > 1 else dices[0],
    }


# ═══════════════════════════════════════════════════════════
#  Part 8: Train & Evaluate
# ═══════════════════════════════════════════════════════════


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler,
                    use_amp, grad_clip=None, num_seg_classes=2):
    model.train()
    running_loss, running_seg_loss, running_cls_loss = 0.0, 0.0, 0.0
    cls_correct, cls_total = 0, 0
    all_seg_ious, all_seg_dices = [], []

    for imgs, masks, labels, has_masks in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                enabled=use_amp):
            seg_logits, cls_logits = model(imgs)
            loss, seg_l, cls_l = criterion(seg_logits, cls_logits, masks, labels, has_masks)

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

        # Seg metrics (on masked samples only)
        if has_masks.any():
            with torch.no_grad():
                mask_idx = has_masks.nonzero(as_tuple=True)[0]
                metrics = compute_seg_metrics(
                    seg_logits[mask_idx], masks[mask_idx], num_seg_classes
                )
                all_seg_ious.append(metrics["lesion_IoU"])
                all_seg_dices.append(metrics["lesion_Dice"])

    n = cls_total
    avg_seg_iou = np.mean(all_seg_ious) if all_seg_ious else 0.0
    avg_seg_dice = np.mean(all_seg_dices) if all_seg_dices else 0.0

    return {
        "loss": running_loss / n,
        "seg_loss": running_seg_loss / n,
        "cls_loss": running_cls_loss / n,
        "cls_acc": cls_correct / n,
        "seg_iou": avg_seg_iou,
        "seg_dice": avg_seg_dice,
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
                metrics = compute_seg_metrics(
                    seg_logits[mask_idx], masks[mask_idx], num_seg_classes
                )
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
        f"[{phase}] Cls — Acc: {acc:.4f} | P(macro): {precision:.4f} | "
        f"R(macro): {recall:.4f} | F1(macro): {f1:.4f}"
    )
    logger.info(
        f"[{phase}] Seg — Lesion IoU: {avg_seg_iou:.4f} | Lesion Dice: {avg_seg_dice:.4f}"
    )
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, precision, recall, f1, avg_seg_iou, avg_seg_dice


def find_optimal_threshold(model, dataloader, device):
    """搜索最优分类阈值 (优化 macro F1)."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, masks, labels, has_masks in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            _, cls_logits = model(imgs)
            probs = torch.softmax(cls_logits, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds = np.where(all_probs >= thresh, 0, 1)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def evaluate_with_threshold(model, dataloader, device, class_names, logger,
                            threshold=0.5, phase="Test"):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, masks, labels, has_masks in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            _, cls_logits = model(imgs)
            probs = torch.softmax(cls_logits, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.where(all_probs >= threshold, 0, 1)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(
        f"[{phase}] Threshold: {threshold:.3f} | Acc: {acc:.4f} | P(macro): {precision:.4f} | "
        f"R(macro): {recall:.4f} | F1(macro): {f1:.4f}"
    )
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, precision, recall, f1


# ═══════════════════════════════════════════════════════════
#  Part 9: Experiment Runner
# ═══════════════════════════════════════════════════════════


def run_seg_cls_experiment(
    cfg,
    build_model_fn,
    build_dataloaders_fn,
    build_optimizer_fn,
    script_path,
):
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
    logger.info("任务: Task 2 — 分割+分类联合训练 (模仿 unet-valjustclass 思路)")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"输入通道: {cfg.in_channels}")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"分割类别: {cfg.num_seg_classes} (background + lesion)")
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
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 70)

    # Dataloaders
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

    # Model
    model = build_model_fn(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,}")
    logger.info(f"可训练参数量: {n_trainable:,}")

    # Loss
    cls_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    logger.info(f"分类类别权重: benign={cls_weights[0]:.4f}, no_tumor={cls_weights[1]:.4f}")

    # 分割类别权重: 背景权重低, 病灶权重高 (因为病灶面积通常很小)
    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight], dtype=torch.float32, device=cfg.device
    )
    logger.info(f"分割类别权重: bg={cfg.seg_bg_weight}, lesion={cfg.seg_lesion_weight}")

    criterion = SegClsLoss(
        cls_weights=cls_weights,
        lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing,
        seg_ce_weight=seg_ce_weight,
    )

    # Optimizer
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
        lr_factor = set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f}, cls={train_metrics['cls_loss']:.4f}) "
            f"| Cls Acc: {train_metrics['cls_acc']:.4f} "
            f"| Seg IoU: {train_metrics['seg_iou']:.4f} "
            f"| Seg Dice: {train_metrics['seg_dice']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 50)
            acc, prec, rec, f1, seg_iou, seg_dice = evaluate(
                model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Test", num_seg_classes=cfg.num_seg_classes,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(
                    f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***"
                )
            logger.info("-" * 50)

    logger.info("\n" + "=" * 70)
    logger.info(f"训练完成! 最优模型: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 70)

    # 加载最优权重做最终测试
    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )
    logger.info("=" * 70)
    logger.info("最终测试结果 (最优权重, threshold=0.5)")
    logger.info("=" * 70)
    evaluate(model, test_loader, cfg.device, cfg.class_names, logger,
             phase="Final Test", num_seg_classes=cfg.num_seg_classes)

    # 阈值优化
    logger.info("\n" + "=" * 70)
    logger.info("阈值优化搜索")
    logger.info("=" * 70)
    best_thresh, best_thresh_f1 = find_optimal_threshold(model, test_loader, cfg.device)
    logger.info(
        f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认0.5 F1: {best_f1:.4f})"
    )
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold(
            model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=best_thresh, phase="Final Test (最优阈值)",
        )

    # 复制训练脚本到日志目录
    dst = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst):
        shutil.copy2(script_path, dst)
        logger.info(f"训练脚本已复制到: {dst}")
