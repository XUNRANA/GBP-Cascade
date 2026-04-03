"""
Task 2 分割+分类联合训练工具库 v2
新增: 4ch输入+分割目标, MaxViT backbone, seg-guided attention cls, metadata支持

vs v1 (seg_cls_utils.py):
  - SegCls4chSyncTransform: 输出 4ch input + seg target
  - GBPDatasetSegCls4ch / GBPDatasetSegCls4chWithMeta
  - MaxViTSegClsModel: MaxViT@320 + UNet decoder
  - SwinV2SegCls4chModel: SwinV2@256 + 4ch + UNet decoder
  - SegGuidedClsHead: 可复用的 seg-attention 分类模块
"""

import os
import sys
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import timm

# Reuse everything from v1
from seg_cls_utils import (
    load_annotation,
    generate_lesion_mask,
    UNetDecoderBlock,
    DiceLoss,
    SegClsLoss,
    set_seed,
    setup_logger,
    acquire_run_lock,
    build_class_weights,
    cosine_warmup_factor,
    set_epoch_lrs,
    build_optimizer_with_diff_lr,
    compute_seg_metrics,
    find_optimal_threshold,
    evaluate_with_threshold,
    run_seg_cls_experiment,
)

# Reuse metadata & 4ch adapt from test_yqh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../0323/scripts"))
from test_yqh import (
    adapt_model_to_4ch,
    META_FEATURE_NAMES,
    build_case_meta_table,
    fit_meta_stats,
    encode_meta_row,
    extract_case_id_from_image_path,
)


# ═══════════════════════════════════════════════════════════
#  Transforms: 4ch input + seg target
# ═══════════════════════════════════════════════════════════


class SegCls4chSyncTransform:
    """
    同步变换: 输出 4ch input tensor + seg target mask.
    - RGB 做归一化 → 前3通道
    - mask 不归一化, 作为第4通道 (raw 0/1)
    - 同时返回 mask 作为分割 target (long tensor)
    """

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        size = [self.img_size, self.img_size]

        if self.is_train:
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=(0.7, 1.0), ratio=(0.85, 1.15),
            )
            img = TF.resized_crop(img, i, j, h, w, size, InterpolationMode.BICUBIC)
            mask = TF.resized_crop(mask, i, j, h, w, size, InterpolationMode.NEAREST)

            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() < 0.3:
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() < 0.5:
                angle = random.uniform(-20, 20)
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BICUBIC, fill=0)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
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
            if random.random() < 0.6:
                img = TF.adjust_brightness(img, random.uniform(0.7, 1.3))
                img = TF.adjust_contrast(img, random.uniform(0.7, 1.3))
                img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
            if random.random() < 0.2:
                img = TF.gaussian_blur(img, kernel_size=3)
        else:
            img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
            mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)    # [3, H, W]
        mask_t = TF.to_tensor(mask)   # [1, H, W]

        img_t = TF.normalize(img_t, self.mean, self.std)

        if self.is_train and random.random() < 0.2:
            img_t = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))(img_t)
        if self.is_train and random.random() < 0.3:
            img_t = img_t + torch.randn_like(img_t) * 0.03

        # 4ch input: [normalized_RGB(3) + raw_mask(1)]
        input_4ch = torch.cat([img_t, mask_t], dim=0)  # [4, H, W]
        # seg target: [H, W] long
        seg_target = (mask_t.squeeze(0) > 0.5).long()

        return input_4ch, seg_target


# ═══════════════════════════════════════════════════════════
#  Datasets
# ═══════════════════════════════════════════════════════════


class GBPDatasetSegCls4ch(Dataset):
    """4ch input (RGB + mask) + seg target + cls label."""

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

        return input_4ch, seg_target, label, has_mask


class GBPDatasetSegCls4chWithMeta(Dataset):
    """4ch input + seg target + cls label + metadata."""

    def __init__(self, excel_path, data_root, clinical_excel_path, json_feature_root,
                 sync_transform=None, meta_stats=None):
        self.df = pd.read_excel(excel_path).copy()
        self.data_root = data_root
        self.sync_transform = sync_transform

        meta_df = build_case_meta_table(clinical_excel_path, json_feature_root)
        self.df["case_id_norm"] = self.df["image_path"].map(extract_case_id_from_image_path)
        self.df = self.df.merge(meta_df, on="case_id_norm", how="left")

        self.meta_feature_names = list(META_FEATURE_NAMES)
        for col in self.meta_feature_names:
            if col not in self.df.columns:
                self.df[col] = np.nan

        self.meta_stats = meta_stats if meta_stats is not None else fit_meta_stats(self.df, self.meta_feature_names)
        self.meta_dim = len(self.meta_feature_names)

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

        meta_tensor = encode_meta_row(row, self.meta_stats, self.meta_feature_names)
        return input_4ch, seg_target, meta_tensor, label, has_mask


def seg_cls_4ch_collate_fn(batch):
    """Collate for 4ch dataset (no metadata)."""
    inputs, masks, labels, has_masks = zip(*batch)
    return (torch.stack(inputs), torch.stack(masks),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(has_masks, dtype=torch.bool))


def seg_cls_4ch_meta_collate_fn(batch):
    """Collate for 4ch + metadata dataset."""
    inputs, masks, metas, labels, has_masks = zip(*batch)
    return (torch.stack(inputs), torch.stack(masks), torch.stack(metas),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(has_masks, dtype=torch.bool))


# ═══════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════


class SwinV2SegCls4chModel(nn.Module):
    """SwinV2-Tiny@256 + 4ch input + UNet seg decoder + cls head."""

    def __init__(self, num_seg_classes=2, num_cls_classes=2, cls_dropout=0.3, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            "swinv2_tiny_window8_256", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
        )
        adapt_model_to_4ch(self.encoder)
        fc = [info["num_chs"] for info in self.encoder.feature_info]  # [96,192,384,768]

        self.dec3 = UNetDecoderBlock(fc[3], fc[2], fc[2])
        self.dec2 = UNetDecoderBlock(fc[2], fc[1], fc[1])
        self.dec1 = UNetDecoderBlock(fc[1], fc[0], fc[0])
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(fc[0], 48, kernel_size=4, stride=4),
            nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, num_seg_classes, 1),
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(fc[3], 256), nn.GELU(), nn.Dropout(cls_dropout),
            nn.Linear(256, num_cls_classes),
        )
        self._fc = fc

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
        cls_logits = self.cls_head(f3)
        return seg_logits, cls_logits


class SwinV2SegGuidedCls4chModel(nn.Module):
    """SwinV2-Tiny@256 + 4ch + UNet seg + seg-guided attention cls + optional metadata."""

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

        # Seg-guided cls: attention pool on f2 (384ch, 16x16)
        self.cls_proj = nn.Sequential(
            nn.Conv2d(fc[2], 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
        )

        # Optional metadata encoder
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

        # Seg-guided attention
        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]  # (B, 1, H, W)
        attn = F.interpolate(seg_prob, size=f2.shape[2:], mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)

        f2_proj = self.cls_proj(f2)
        cls_feat = (f2_proj * attn).sum(dim=(2, 3))  # (B, 256)

        if self.meta_encoder is not None and metadata is not None:
            meta_feat = self.meta_encoder(metadata.float())
            cls_feat = torch.cat([cls_feat, meta_feat], dim=1)

        cls_logits = self.cls_mlp(cls_feat)
        return seg_logits, cls_logits


class MaxViTSegClsModel(nn.Module):
    """MaxViT-Tiny@320 + 4ch + UNet seg decoder + cls head.

    MaxViT features (5 stages): [64@160, 64@80, 128@40, 256@20, 512@10]
    UNet uses stages 1-4 (skip stage0 which is 160x160).
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=2, cls_dropout=0.3,
                 img_size=320, pretrained=True):
        super().__init__()
        self.img_size = img_size

        self.encoder = timm.create_model(
            "maxvit_tiny_tf_384.in1k", pretrained=pretrained,
            features_only=True, img_size=img_size,
        )
        adapt_model_to_4ch(self.encoder)
        fc = [info["num_chs"] for info in self.encoder.feature_info]
        # [64, 64, 128, 256, 512]

        # UNet decoder: stages 4→3→2→1 with skip connections
        self.dec4 = UNetDecoderBlock(fc[4], fc[3], fc[3])  # 512+256→256
        self.dec3 = UNetDecoderBlock(fc[3], fc[2], fc[2])  # 256+128→128
        self.dec2 = UNetDecoderBlock(fc[2], fc[1], fc[1])  # 128+64→64

        # 80x80 → 320x320 (4x upsample)
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(fc[1], 32, kernel_size=4, stride=4),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, num_seg_classes, 1),
        )

        self.cls_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(fc[4], 256), nn.GELU(), nn.Dropout(cls_dropout),
            nn.Linear(256, num_cls_classes),
        )
        self._fc = fc

    def forward(self, x, metadata=None):
        features = self.encoder(x)
        f0, f1, f2, f3, f4 = features
        # f0: 64@160, f1: 64@80, f2: 128@40, f3: 256@20, f4: 512@10

        d4 = self.dec4(f4, f3)   # → 256@20
        d3 = self.dec3(d4, f2)   # → 128@40
        d2 = self.dec2(d3, f1)   # → 64@80
        seg_logits = self.seg_final(d2)  # → 2@320

        cls_logits = self.cls_head(f4)
        return seg_logits, cls_logits


class MaxViTSegGuidedClsModel(nn.Module):
    """MaxViT-Tiny@320 + 4ch + UNet seg + seg-guided attention cls."""

    def __init__(self, num_seg_classes=2, num_cls_classes=2, cls_dropout=0.4,
                 img_size=320, pretrained=True):
        super().__init__()
        self.img_size = img_size

        self.encoder = timm.create_model(
            "maxvit_tiny_tf_384.in1k", pretrained=pretrained,
            features_only=True, img_size=img_size,
        )
        adapt_model_to_4ch(self.encoder)
        fc = [info["num_chs"] for info in self.encoder.feature_info]

        self.dec4 = UNetDecoderBlock(fc[4], fc[3], fc[3])
        self.dec3 = UNetDecoderBlock(fc[3], fc[2], fc[2])
        self.dec2 = UNetDecoderBlock(fc[2], fc[1], fc[1])
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(fc[1], 32, kernel_size=4, stride=4),
            nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, num_seg_classes, 1),
        )

        # Seg-guided cls on f3 (256ch, 20x20)
        self.cls_proj = nn.Sequential(
            nn.Conv2d(fc[3], 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
        )
        self.cls_mlp = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(cls_dropout),
            nn.Linear(128, num_cls_classes),
        )

    def forward(self, x, metadata=None):
        features = self.encoder(x)
        f0, f1, f2, f3, f4 = features

        d4 = self.dec4(f4, f3)
        d3 = self.dec3(d4, f2)
        d2 = self.dec2(d3, f1)
        seg_logits = self.seg_final(d2)

        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]
        attn = F.interpolate(seg_prob, size=f3.shape[2:], mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)

        f3_proj = self.cls_proj(f3)
        cls_feat = (f3_proj * attn).sum(dim=(2, 3))
        cls_logits = self.cls_mlp(cls_feat)
        return seg_logits, cls_logits


# ═══════════════════════════════════════════════════════════
#  v2 Training / Evaluation (metadata-aware)
# ═══════════════════════════════════════════════════════════

import atexit
import logging
import math
import shutil
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)


def _unpack_batch(batch):
    """Auto-detect batch format: 4-element or 5-element (with metadata)."""
    if len(batch) == 5:
        imgs, masks, metas, labels, has_masks = batch
        return imgs, masks, metas, labels, has_masks
    else:
        imgs, masks, labels, has_masks = batch
        return imgs, masks, None, labels, has_masks


def train_one_epoch_v2(model, dataloader, criterion, optimizer, device, scaler,
                       use_amp, grad_clip=None, num_seg_classes=2):
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


def evaluate_v2(model, dataloader, device, class_names, logger, phase="Test",
                num_seg_classes=2):
    model.eval()
    all_preds, all_labels = [], []
    all_seg_ious, all_seg_dices = [], []

    with torch.no_grad():
        for batch in dataloader:
            imgs, masks, metas, labels, has_masks = _unpack_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            has_masks = has_masks.to(device, non_blocking=True)
            if metas is not None:
                metas = metas.to(device, non_blocking=True)

            seg_logits, cls_logits = model(imgs, metadata=metas)
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


def find_optimal_threshold_v2(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            imgs, masks, metas, labels, has_masks = _unpack_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            if metas is not None:
                metas = metas.to(device, non_blocking=True)
            _, cls_logits = model(imgs, metadata=metas)
            probs = torch.softmax(cls_logits, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

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


def evaluate_with_threshold_v2(model, dataloader, device, class_names, logger,
                               threshold=0.5, phase="Test"):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            imgs, masks, metas, labels, has_masks = _unpack_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            if metas is not None:
                metas = metas.to(device, non_blocking=True)
            _, cls_logits = model(imgs, metadata=metas)
            probs = torch.softmax(cls_logits, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

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


def run_seg_cls_experiment_v2(cfg, build_model_fn, build_dataloaders_fn,
                              build_optimizer_fn, script_path):
    """v2 experiment runner — handles both 4ch and 4ch+meta dataloaders."""
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
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,}")
    logger.info(f"可训练参数量: {n_trainable:,}")

    cls_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    logger.info(f"分类类别权重: benign={cls_weights[0]:.4f}, no_tumor={cls_weights[1]:.4f}")

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

        train_metrics = train_one_epoch_v2(
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
            acc, prec, rec, f1, seg_iou, seg_dice = evaluate_v2(
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

    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )
    logger.info("=" * 70)
    logger.info("最终测试结果 (最优权重, threshold=0.5)")
    logger.info("=" * 70)
    evaluate_v2(model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Final Test", num_seg_classes=cfg.num_seg_classes)

    logger.info("\n" + "=" * 70)
    logger.info("阈值优化搜索")
    logger.info("=" * 70)
    best_thresh, best_thresh_f1 = find_optimal_threshold_v2(model, test_loader, cfg.device)
    logger.info(
        f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认0.5 F1: {best_f1:.4f})"
    )
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold_v2(
            model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=best_thresh, phase="Final Test (最优阈值)",
        )

    dst = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst):
        shutil.copy2(script_path, dst)
        logger.info(f"训练脚本已复制到: {dst}")
