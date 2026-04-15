"""
0414 Exp-B: 三分类 + 代价敏感损失 + 线性风险分数 + 均衡采样 (Baseline)

相比 Smoke-1 新增:
  1. CostSensitiveLoss: 非对称误分类代价矩阵 (对应 0414.txt 第1条)
  2. OrdinalScoreHead: 0~1 连续风险分数 + 可调阈值 (对应 0414.txt 第2条)
  3. malignant 分割 loss 自动置零 (对应 0414.txt 第3条)
  4. BalancedBatchSampler: 每 batch 2:2:4 (mal:ben:notumor) (对应 0414.txt 第5条)
  5. 患者级 val 切分 (从 train 中分出 15%)
  6. 双阈值 (t1, t2) 在 val 上搜索
  7. 临床指标: malignant sensitivity, NPV, 避免手术率
"""

import os
import sys
import time
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit

# ─── 路径设置 ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
SCRIPTS_0402 = os.path.join(ROOT_DIR, "0402", "scripts")
if SCRIPTS_0402 not in sys.path:
    sys.path.insert(0, SCRIPTS_0402)

from seg_cls_utils_v2 import (  # noqa: E402
    load_annotation, generate_lesion_mask, UNetDecoderBlock, DiceLoss,
    set_seed, setup_logger, acquire_run_lock, build_class_weights,
    set_epoch_lrs, build_optimizer_with_diff_lr, compute_seg_metrics,
)
from seg_cls_utils import cosine_warmup_factor  # noqa: E402

sys.path.insert(0, os.path.join(ROOT_DIR, "0323", "scripts"))
from test_yqh import adapt_model_to_4ch  # noqa: E402

import timm  # noqa: E402


# ═════════════════════════════════════════════════════════════
#  1. 辅助函数
# ═════════════════════════════════════════════════════════════

def extract_patient_id(image_path: str) -> str:
    """从 image_path 提取患者 ID.  格式: class/PATIENT_US_ImageXX.png"""
    fname = os.path.basename(image_path)
    return fname.split("_US_")[0] if "_US_" in fname else fname.rsplit("_", 1)[0]


def generate_gallbladder_mask(shapes, width, height):
    """从 LabelMe shapes 生成胆囊 ROI mask."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        if shape.get("label") != "gallbladder":
            continue
        points = shape.get("points", [])
        shape_type = shape.get("shape_type")
        if shape_type == "rectangle" and len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            draw.rectangle([x1, y1, x2, y2], fill=255)
        elif shape_type == "polygon" and len(points) >= 3:
            draw.polygon([(p[0], p[1]) for p in points], fill=255)
    return mask


# ═════════════════════════════════════════════════════════════
#  2. 代价敏感损失 + Ordinal 损失
# ═════════════════════════════════════════════════════════════

# 标签: malignant=0, benign=1, no_tumor=2
# 代价矩阵 (严格对应 0414.txt):
#   真实恶性->预测息肉: 最大(4.0)
#   真实良性->预测息肉: 中等偏大(2.0)
#   真实息肉->预测恶性: 中等偏小(1.5)
#   真实息肉->预测良性: 最小(1.0)
#   恶性<->良性: 中等偏小(1.0)
COST_MATRIX = torch.tensor([
    # pred_mal  pred_ben  pred_notumor
    [0.0,       1.0,      4.0],   # true = malignant
    [1.0,       0.0,      2.0],   # true = benign
    [1.5,       1.0,      0.0],   # true = no_tumor
], dtype=torch.float32)

# Ordinal 风险分数目标: malignant=1.0, benign=0.5, no_tumor=0.0
ORDINAL_TARGETS = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)


class CostSensitiveLoss(nn.Module):
    """
    代价敏感分类损失 = class-weighted CE + 期望误分类代价惩罚.

    - CE 部分: 使用代价矩阵行和作为 class weights
    - 惩罚部分: sum_c(cost[y,c] * p_c)  直接最小化期望误分类代价
    """

    def __init__(self, cost_matrix, penalty_weight=0.5, label_smoothing=0.1):
        super().__init__()
        self.register_buffer("cost_matrix", cost_matrix)
        self.penalty_weight = penalty_weight

        # class weights = 代价矩阵行和 (归一化)
        row_sums = cost_matrix.sum(dim=1)  # malignant=5, benign=3, no_tumor=2.5
        cls_weights = row_sums / row_sums.mean()
        self.register_buffer("cls_weights", cls_weights)
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        # CE with class weights
        ce_loss = F.cross_entropy(
            logits, labels, weight=self.cls_weights,
            label_smoothing=self.label_smoothing,
        )
        # 期望误分类代价惩罚
        probs = F.softmax(logits, dim=1)
        costs = self.cost_matrix[labels]     # (B, C)
        penalty = (costs * probs).sum(dim=1).mean()

        return ce_loss + self.penalty_weight * penalty


class OrdinalScoreLoss(nn.Module):
    """线性风险分数的 Smooth L1 损失."""

    def __init__(self, ordinal_targets):
        super().__init__()
        self.register_buffer("targets", ordinal_targets)

    def forward(self, score, labels):
        """score: (B,) sigmoid output 0~1, labels: (B,) long"""
        target = self.targets[labels]
        return F.smooth_l1_loss(score, target)


class SegClsOrdinalLoss(nn.Module):
    """
    总损失 = seg_loss + lambda_cls * cls_loss + lambda_ord * ord_loss

    - seg_loss: CE + Dice (仅 has_mask 样本)
    - cls_loss: CostSensitiveLoss
    - ord_loss: OrdinalScoreLoss
    """

    def __init__(self, cost_matrix, ordinal_targets,
                 lambda_cls=2.0, lambda_ord=0.5,
                 penalty_weight=0.5, label_smoothing=0.1,
                 seg_ce_weight=None):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ord = lambda_ord

        self.cls_loss_fn = CostSensitiveLoss(
            cost_matrix, penalty_weight=penalty_weight,
            label_smoothing=label_smoothing,
        )
        self.ord_loss_fn = OrdinalScoreLoss(ordinal_targets)
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, cls_logits, ordinal_score,
                seg_targets, cls_targets, has_mask):
        # 分类损失
        cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
        # Ordinal 损失
        ord_loss = self.ord_loss_fn(ordinal_score, cls_targets)
        # 分割损失 (仅有 mask 的样本)
        seg_loss = torch.tensor(0.0, device=seg_logits.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce + seg_dice

        total = seg_loss + self.lambda_cls * cls_loss + self.lambda_ord * ord_loss
        return total, seg_loss.item(), cls_loss.item(), ord_loss.item()


# ═════════════════════════════════════════════════════════════
#  3. BalancedBatchSampler
# ═════════════════════════════════════════════════════════════

class BalancedBatchSampler(Sampler):
    """
    每个 batch 固定包含各类指定数量的样本.
    默认: malignant=2, benign=2, no_tumor=4  (batch_size=8)
    """

    def __init__(self, labels, samples_per_class, shuffle=True):
        """
        labels: array-like, 每个样本的类别标签
        samples_per_class: dict {label: count_per_batch}
        """
        self.labels = np.array(labels)
        self.samples_per_class = samples_per_class
        self.shuffle = shuffle

        self.class_indices = {}
        for cls_label, count in samples_per_class.items():
            self.class_indices[cls_label] = np.where(self.labels == cls_label)[0]
            assert len(self.class_indices[cls_label]) > 0, \
                f"Class {cls_label} has no samples"

        self.batch_size = sum(samples_per_class.values())
        # 一个 epoch 的 batch 数 = 最大类的样本数 / 该类每 batch 数量
        max_batches = max(
            len(indices) // count
            for cls_label, (indices, count) in zip(
                self.class_indices.keys(),
                zip(self.class_indices.values(),
                    [samples_per_class[c] for c in self.class_indices.keys()])
            )
        )
        self.num_batches = max_batches

    def __iter__(self):
        # 为每个类创建 shuffled index 循环
        class_iters = {}
        for cls_label, indices in self.class_indices.items():
            idx = indices.copy()
            if self.shuffle:
                np.random.shuffle(idx)
            # 重复足够多次以覆盖所有 batches
            count = self.samples_per_class[cls_label]
            needed = self.num_batches * count
            repeats = (needed // len(idx)) + 2
            idx = np.tile(idx, repeats)
            class_iters[cls_label] = idx

        # 为每个类维护指针
        pointers = {cls_label: 0 for cls_label in self.class_indices}

        for _ in range(self.num_batches):
            batch = []
            for cls_label, count in self.samples_per_class.items():
                ptr = pointers[cls_label]
                batch.extend(class_iters[cls_label][ptr:ptr + count].tolist())
                pointers[cls_label] = ptr + count
            if self.shuffle:
                np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# ═════════════════════════════════════════════════════════════
#  4. 数据增强 & Dataset
# ═════════════════════════════════════════════════════════════

class SegCls0414SyncTransform:
    """同步变换: RGB + gallbladder mask + lesion mask."""

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, gb_mask, lesion_mask):
        size = [self.img_size, self.img_size]

        if self.is_train:
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=(0.7, 1.0), ratio=(0.85, 1.15))
            img = TF.resized_crop(img, i, j, h, w, size, InterpolationMode.BICUBIC)
            gb_mask = TF.resized_crop(gb_mask, i, j, h, w, size, InterpolationMode.NEAREST)
            lesion_mask = TF.resized_crop(lesion_mask, i, j, h, w, size, InterpolationMode.NEAREST)

            if np.random.rand() < 0.5:
                img = TF.hflip(img); gb_mask = TF.hflip(gb_mask); lesion_mask = TF.hflip(lesion_mask)
            if np.random.rand() < 0.3:
                img = TF.vflip(img); gb_mask = TF.vflip(gb_mask); lesion_mask = TF.vflip(lesion_mask)
            if np.random.rand() < 0.5:
                angle = float(np.random.uniform(-20, 20))
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BICUBIC, fill=0)
                gb_mask = TF.rotate(gb_mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
                lesion_mask = TF.rotate(lesion_mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
            if np.random.rand() < 0.5:
                angle = float(np.random.uniform(-5, 5))
                max_t = 0.06 * self.img_size
                translate = [int(np.random.uniform(-max_t, max_t)),
                             int(np.random.uniform(-max_t, max_t))]
                scale = float(np.random.uniform(0.9, 1.1))
                shear = [float(np.random.uniform(-5, 5))]
                img = TF.affine(img, angle, translate, scale, shear,
                                interpolation=InterpolationMode.BICUBIC, fill=0)
                gb_mask = TF.affine(gb_mask, angle, translate, scale, shear,
                                    interpolation=InterpolationMode.NEAREST, fill=0)
                lesion_mask = TF.affine(lesion_mask, angle, translate, scale, shear,
                                        interpolation=InterpolationMode.NEAREST, fill=0)
            if np.random.rand() < 0.6:
                img = TF.adjust_brightness(img, float(np.random.uniform(0.7, 1.3)))
                img = TF.adjust_contrast(img, float(np.random.uniform(0.7, 1.3)))
                img = TF.adjust_saturation(img, float(np.random.uniform(0.8, 1.2)))
            if np.random.rand() < 0.2:
                img = TF.gaussian_blur(img, kernel_size=3)
        else:
            img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
            gb_mask = TF.resize(gb_mask, size, interpolation=InterpolationMode.NEAREST)
            lesion_mask = TF.resize(lesion_mask, size, interpolation=InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)
        gb_t = TF.to_tensor(gb_mask)
        lesion_t = TF.to_tensor(lesion_mask)
        img_t = TF.normalize(img_t, self.mean, self.std)

        if self.is_train and np.random.rand() < 0.2:
            img_t = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))(img_t)
        if self.is_train and np.random.rand() < 0.3:
            img_t = img_t + torch.randn_like(img_t) * 0.03

        input_4ch = torch.cat([img_t, gb_t], dim=0)
        seg_target = (lesion_t.squeeze(0) > 0.5).long()
        return input_4ch, seg_target


class GBPDataset0414(Dataset):
    """0414 dataset: 4ch + lesion seg target + 3-class label."""

    def __init__(self, df, data_root, sync_transform=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.sync_transform = sync_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        json_path = img_path.replace(".png", ".json")
        label = int(row["label"])  # 0=malignant, 1=benign, 2=no_tumor

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        shapes = []
        has_lesion_mask = False
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])
            has_lesion_mask = any(
                s.get("label") != "gallbladder"
                and s.get("shape_type") == "polygon"
                and len(s.get("points", [])) >= 3
                for s in shapes
            )

        gb_mask = generate_gallbladder_mask(shapes, img_w, img_h)
        lesion_mask = generate_lesion_mask(shapes, img_w, img_h)

        if self.sync_transform:
            input_4ch, seg_target = self.sync_transform(img, gb_mask, lesion_mask)
        else:
            img_t = TF.to_tensor(img)
            gb_t = TF.to_tensor(gb_mask)
            lesion_t = TF.to_tensor(lesion_mask)
            input_4ch = torch.cat([img_t, gb_t], dim=0)
            seg_target = (lesion_t.squeeze(0) > 0.5).long()

        return input_4ch, seg_target, label, has_lesion_mask


def collate_fn(batch):
    inputs, masks, labels, has_masks = zip(*batch)
    return (torch.stack(inputs), torch.stack(masks),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(has_masks, dtype=torch.bool))


# ═════════════════════════════════════════════════════════════
#  5. 模型: SwinV2 + Ordinal Score Head
# ═════════════════════════════════════════════════════════════

class SwinV2SegCls4chOrdinalModel(nn.Module):
    """
    SwinV2-Tiny@256 + 4ch + UNet seg decoder
    + 3-class cls head + ordinal score head (0~1)
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=3,
                 cls_dropout=0.4, pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            "swinv2_tiny_window8_256", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
        )
        adapt_model_to_4ch(self.encoder)
        fc = [info["num_chs"] for info in self.encoder.feature_info]  # [96,192,384,768]

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

        # 共享 bottleneck: f3 -> 256D
        self.bottleneck = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(fc[3], 256), nn.GELU(), nn.Dropout(cls_dropout),
        )

        # 分类头: 256 -> 3
        self.cls_head = nn.Linear(256, num_cls_classes)

        # Ordinal score head: 256 -> 1 -> sigmoid
        self.ord_head = nn.Sequential(
            nn.Linear(256, 1),
        )

    def _to_bchw(self, x):
        if x.ndim == 4 and x.shape[1] != x.shape[3]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, metadata=None):
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        # Seg
        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        # Cls + Ordinal (共享 bottleneck)
        feat = self.bottleneck(f3)           # (B, 256)
        cls_logits = self.cls_head(feat)     # (B, 3)
        ord_score = torch.sigmoid(self.ord_head(feat).squeeze(-1))  # (B,)

        return seg_logits, cls_logits, ord_score


# ═════════════════════════════════════════════════════════════
#  6. 训练 & 评估
# ═════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    scaler, use_amp, grad_clip=None, num_seg_classes=2):
    model.train()
    stats = {"loss": 0, "seg": 0, "cls": 0, "ord": 0,
             "cls_correct": 0, "total": 0, "seg_dices": []}

    for imgs, masks, labels, has_masks in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            seg_logits, cls_logits, ord_score = model(imgs)
            loss, seg_l, cls_l, ord_l = criterion(
                seg_logits, cls_logits, ord_score, masks, labels, has_masks)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        stats["loss"] += loss.item() * bs
        stats["seg"] += seg_l * bs
        stats["cls"] += cls_l * bs
        stats["ord"] += ord_l * bs
        stats["cls_correct"] += (cls_logits.argmax(1) == labels).sum().item()
        stats["total"] += bs

        if has_masks.any():
            with torch.no_grad():
                idx = has_masks.nonzero(as_tuple=True)[0]
                m = compute_seg_metrics(seg_logits[idx], masks[idx], num_seg_classes)
                stats["seg_dices"].append(m["lesion_Dice"])

    n = stats["total"]
    return {
        "loss": stats["loss"] / n,
        "seg_loss": stats["seg"] / n,
        "cls_loss": stats["cls"] / n,
        "ord_loss": stats["ord"] / n,
        "cls_acc": stats["cls_correct"] / n,
        "seg_dice": np.mean(stats["seg_dices"]) if stats["seg_dices"] else 0.0,
    }


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    """收集所有预测结果: cls_logits, ordinal_scores, labels, seg metrics."""
    model.eval()
    all_logits, all_scores, all_labels = [], [], []
    all_seg_dices = []

    for imgs, masks, labels, has_masks in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        seg_logits, cls_logits, ord_score = model(imgs)
        all_logits.append(cls_logits.cpu())
        all_scores.append(ord_score.cpu())
        all_labels.append(labels)

        if has_masks.any():
            idx = has_masks.nonzero(as_tuple=True)[0]
            m = compute_seg_metrics(seg_logits[idx], masks[idx], 2)
            all_seg_dices.append(m["lesion_Dice"])

    return {
        "logits": torch.cat(all_logits),
        "scores": torch.cat(all_scores).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "seg_dice": np.mean(all_seg_dices) if all_seg_dices else 0.0,
    }


def search_thresholds(scores, labels, class_names, t1_range=(0.05, 0.45, 0.02),
                       t2_range=(0.45, 0.90, 0.02)):
    """在 val 上网格搜索最优 (t1, t2)."""
    best_f1, best_t1, best_t2 = 0.0, 0.33, 0.66
    for t1 in np.arange(*t1_range):
        for t2 in np.arange(*t2_range):
            if t2 <= t1:
                continue
            preds = np.where(scores >= t2, 0,        # malignant
                    np.where(scores >= t1, 1, 2))     # benign or no_tumor
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t1, best_t2 = f1, t1, t2
    return best_t1, best_t2, best_f1


def compute_clinical_metrics(scores, labels, t1, t2, logger, phase="Test"):
    """计算临床指标: sensitivity, NPV, 避免手术率."""
    # Ordinal score 分类
    preds = np.where(scores >= t2, 0,        # malignant
            np.where(scores >= t1, 1, 2))     # benign or no_tumor

    # 混淆矩阵
    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    logger.info(f"[{phase}] 混淆矩阵 (行=真实, 列=预测):")
    logger.info(f"         pred_mal  pred_ben  pred_notumor")
    names = ["malignant", "benign   ", "no_tumor "]
    for i, name in enumerate(names):
        logger.info(f"  {name}  {cm[i, 0]:>8d}  {cm[i, 1]:>8d}  {cm[i, 2]:>12d}")

    # Malignant sensitivity (recall)
    mal_total = (labels == 0).sum()
    mal_correct = cm[0, 0]
    mal_sensitivity = mal_correct / mal_total if mal_total > 0 else 0
    logger.info(f"[{phase}] Malignant Sensitivity: {mal_sensitivity:.4f} ({mal_correct}/{mal_total})")

    # 恶性漏诊为息肉的数量 (最危险的错误)
    mal_to_notumor = cm[0, 2]
    logger.info(f"[{phase}] Malignant->NoTumor 漏诊数: {mal_to_notumor} (临床安全红线)")

    # NPV: AI 说"息肉/随访" 时真正是息肉的概率
    pred_notumor_total = cm[:, 2].sum()
    true_notumor_in_pred = cm[2, 2]
    npv = true_notumor_in_pred / pred_notumor_total if pred_notumor_total > 0 else 0
    logger.info(f"[{phase}] NPV (预测息肉的准确率): {npv:.4f} ({true_notumor_in_pred}/{pred_notumor_total})")

    # 避免不必要手术率: 在保证当前 malignant sensitivity 的前提下,
    # 正确识别出多少比例的息肉患者
    notumor_total = (labels == 2).sum()
    notumor_correct = cm[2, 2]
    avoidable_rate = notumor_correct / notumor_total if notumor_total > 0 else 0
    logger.info(f"[{phase}] 避免不必要手术率: {avoidable_rate:.4f} ({notumor_correct}/{notumor_total})")

    return {
        "mal_sensitivity": mal_sensitivity,
        "mal_to_notumor": mal_to_notumor,
        "npv": npv,
        "avoidable_surgery_rate": avoidable_rate,
    }


def evaluate(model, dataloader, device, class_names, logger,
             phase="Test", t1=0.33, t2=0.66):
    """完整评估: softmax 分类 + ordinal 阈值分类 + 临床指标."""
    preds_data = collect_predictions(model, dataloader, device)
    logits = preds_data["logits"]
    scores = preds_data["scores"]
    labels = preds_data["labels"]
    seg_dice = preds_data["seg_dice"]

    # ── Softmax 分类结果 ──
    softmax_preds = logits.argmax(dim=1).numpy()
    acc = accuracy_score(labels, softmax_preds)
    prec = precision_score(labels, softmax_preds, average="macro", zero_division=0)
    rec = recall_score(labels, softmax_preds, average="macro", zero_division=0)
    f1 = f1_score(labels, softmax_preds, average="macro", zero_division=0)

    logger.info(f"[{phase}] Softmax — Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    report = classification_report(labels, softmax_preds,
                                   target_names=class_names, digits=4, zero_division=0)
    logger.info(f"[{phase}] Softmax Report:\n{report}")

    # ── Ordinal score 分类结果 ──
    ord_preds = np.where(scores >= t2, 0, np.where(scores >= t1, 1, 2))
    ord_f1 = f1_score(labels, ord_preds, average="macro", zero_division=0)
    ord_acc = accuracy_score(labels, ord_preds)
    logger.info(f"[{phase}] Ordinal (t1={t1:.2f}, t2={t2:.2f}) — Acc: {ord_acc:.4f} | F1: {ord_f1:.4f}")
    ord_report = classification_report(labels, ord_preds,
                                       target_names=class_names, digits=4, zero_division=0)
    logger.info(f"[{phase}] Ordinal Report:\n{ord_report}")

    # ── 分割 ──
    logger.info(f"[{phase}] Seg Dice: {seg_dice:.4f}")

    # ── 临床指标 ──
    compute_clinical_metrics(scores, labels, t1, t2, logger, phase)

    # ── 风险分数分布 ──
    for cls_idx, name in enumerate(class_names):
        cls_scores = scores[labels == cls_idx]
        if len(cls_scores) > 0:
            logger.info(f"[{phase}] Score分布 {name}: "
                        f"mean={cls_scores.mean():.3f} std={cls_scores.std():.3f} "
                        f"min={cls_scores.min():.3f} max={cls_scores.max():.3f}")

    return f1, ord_f1, seg_dice, scores, labels


# ═════════════════════════════════════════════════════════════
#  7. 配置
# ═════════════════════════════════════════════════════════════

class Config:
    project_root = ROOT_DIR
    data_root = os.path.join(project_root, "0414dataset")
    train_excel = os.path.join(data_root, "task_3class_train.xlsx")
    test_excel = os.path.join(data_root, "task_3class_test.xlsx")

    exp_name = "20260414_task3_SwinV2Tiny_segcls_2"
    log_dir = os.path.join(project_root, "0414", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 3
    cls_dropout = 0.4
    pretrained = True

    batch_size = 8
    num_epochs = 60
    warmup_epochs = 5
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 2
    seed = 42
    use_amp = True

    # 损失权重
    lambda_cls = 2.0
    lambda_ord = 0.5
    penalty_weight = 0.5          # 代价惩罚项权重
    label_smoothing = 0.1
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    # 验证集比例
    val_ratio = 0.15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["malignant", "benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + UNetSeg + 3ClsCostSensitive + OrdinalScore"
    modification = (
        "Exp-B baseline: 代价敏感CE(非对称代价矩阵) + "
        "ordinal score(0~1风险分数) + BalancedBatchSampler(2:2:4) + "
        "患者级val切分 + 双阈值搜索 + 临床指标"
    )


# ═════════════════════════════════════════════════════════════
#  8. 数据准备
# ═════════════════════════════════════════════════════════════

def patient_level_split(df, val_ratio=0.15, seed=42):
    """患者级分层切分 train -> train_inner + val."""
    df = df.copy()
    df["patient_id"] = df["image_path"].apply(extract_patient_id)

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(df, df["label"], groups=df["patient_id"]))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    return train_df, val_df


def build_dataloaders(cfg):
    full_train_df = pd.read_excel(cfg.train_excel)
    test_df = pd.read_excel(cfg.test_excel)

    # 患者级切分
    train_df, val_df = patient_level_split(full_train_df, cfg.val_ratio, cfg.seed)

    train_tf = SegCls0414SyncTransform(cfg.img_size, is_train=True)
    eval_tf = SegCls0414SyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDataset0414(train_df, cfg.data_root, train_tf)
    val_dataset = GBPDataset0414(val_df, cfg.data_root, eval_tf)
    test_dataset = GBPDataset0414(test_df, cfg.data_root, eval_tf)

    # BalancedBatchSampler: 2 mal + 2 ben + 4 notumor per batch
    sampler = BalancedBatchSampler(
        train_df["label"].values,
        samples_per_class={0: 2, 1: 2, 2: 4},
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, collate_fn=collate_fn,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════
#  9. 主流程
# ═════════════════════════════════════════════════════════════

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

    # ── 打印配置 ──
    logger.info("=" * 70)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"分类类别: {cfg.class_names}")
    logger.info(f"Batch Size: {cfg.batch_size} (2mal+2ben+4notumor)")
    logger.info(f"Backbone LR: {cfg.backbone_lr} | Head LR: {cfg.head_lr}")
    logger.info(f"Epochs: {cfg.num_epochs} | Warmup: {cfg.warmup_epochs}")
    logger.info(f"Lambda Cls: {cfg.lambda_cls} | Lambda Ord: {cfg.lambda_ord}")
    logger.info(f"代价惩罚权重: {cfg.penalty_weight}")
    logger.info(f"Val 比例: {cfg.val_ratio}")
    logger.info(f"设备: {cfg.device}")
    logger.info("代价矩阵:")
    logger.info(f"  malignant:  {COST_MATRIX[0].tolist()}")
    logger.info(f"  benign:     {COST_MATRIX[1].tolist()}")
    logger.info(f"  no_tumor:   {COST_MATRIX[2].tolist()}")
    logger.info("=" * 70)

    # ── 数据 ──
    (train_dataset, val_dataset, test_dataset,
     train_loader, val_loader, test_loader) = build_dataloaders(cfg)

    for name, ds in [("训练集", train_dataset), ("验证集", val_dataset), ("测试集", test_dataset)]:
        counts = ds.df["label"].value_counts().sort_index().to_dict()
        msg = ", ".join(f"{cfg.class_names[i]}={counts.get(i, 0)}" for i in range(3))
        logger.info(f"{name}: {len(ds)} 张 ({msg})")

    # ── 模型 ──
    try:
        model = SwinV2SegCls4chOrdinalModel(
            num_seg_classes=cfg.num_seg_classes,
            num_cls_classes=cfg.num_cls_classes,
            cls_dropout=cfg.cls_dropout,
            pretrained=cfg.pretrained,
        ).to(cfg.device)
    except Exception as exc:
        logger.warning(f"pretrained 加载失败, 使用随机初始化: {exc}")
        model = SwinV2SegCls4chOrdinalModel(
            num_seg_classes=cfg.num_seg_classes,
            num_cls_classes=cfg.num_cls_classes,
            cls_dropout=cfg.cls_dropout,
            pretrained=False,
        ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,}")
    logger.info(f"可训练参数量: {n_trainable:,}")

    # ── 损失 ──
    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight], dtype=torch.float32, device=cfg.device)
    criterion = SegClsOrdinalLoss(
        cost_matrix=COST_MATRIX.to(cfg.device),
        ordinal_targets=ORDINAL_TARGETS.to(cfg.device),
        lambda_cls=cfg.lambda_cls,
        lambda_ord=cfg.lambda_ord,
        penalty_weight=cfg.penalty_weight,
        label_smoothing=cfg.label_smoothing,
        seg_ce_weight=seg_ce_weight,
    )

    # ── 优化器 ──
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.")]
    optimizer = build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)

    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    # ── 训练循环 ──
    best_val_f1 = 0.0
    best_epoch = 0
    best_t1, best_t2 = 0.33, 0.66

    logger.info("=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f} cls={train_metrics['cls_loss']:.4f} "
            f"ord={train_metrics['ord_loss']:.4f}) "
            f"| Acc: {train_metrics['cls_acc']:.4f} "
            f"| Dice: {train_metrics['seg_dice']:.4f} "
            f"| {elapsed:.0f}s"
        )

        # ── 验证 (用 val 选模型, 非 test) ──
        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 50)

            # 先搜索最优阈值
            val_preds = collect_predictions(model, val_loader, cfg.device)
            t1, t2, val_ord_f1 = search_thresholds(
                val_preds["scores"], val_preds["labels"], cfg.class_names)

            # softmax F1
            val_softmax_preds = val_preds["logits"].argmax(1).numpy()
            val_softmax_f1 = f1_score(val_preds["labels"], val_softmax_preds,
                                      average="macro", zero_division=0)

            # 用更好的 F1 来选模型 (softmax or ordinal)
            val_f1 = max(val_softmax_f1, val_ord_f1)

            logger.info(
                f"[Val] Softmax F1: {val_softmax_f1:.4f} | "
                f"Ordinal F1: {val_ord_f1:.4f} (t1={t1:.2f}, t2={t2:.2f}) | "
                f"Dice: {val_preds['seg_dice']:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                best_t1, best_t2 = t1, t2
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优模型 (Val F1: {best_val_f1:.4f}, Epoch: {best_epoch}) ***")

            logger.info("-" * 50)

    logger.info("=" * 70)
    logger.info(f"训练完成! 最优 Epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}")
    logger.info(f"最优阈值: t1={best_t1:.3f}, t2={best_t2:.3f}")
    logger.info("=" * 70)

    # ── 加载最优权重, 最终测试 ──
    if os.path.exists(cfg.best_weight_path):
        try:
            state = torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
        except TypeError:
            state = torch.load(cfg.best_weight_path, map_location=cfg.device)
        model.load_state_dict(state)

        logger.info("\n" + "=" * 70)
        logger.info("最终测试 (最优权重)")
        logger.info("=" * 70)

        # 用 val 搜索到的阈值在 test 上评估
        evaluate(model, test_loader, cfg.device, cfg.class_names, logger,
                 phase="Final Test", t1=best_t1, t2=best_t2)

        # 同时报告默认阈值
        logger.info("-" * 50)
        evaluate(model, test_loader, cfg.device, cfg.class_names, logger,
                 phase="Final Test (默认阈值)", t1=0.33, t2=0.66)

    # ── 复制脚本 ──
    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        Path(dst).write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"训练脚本已复制到: {dst}")


if __name__ == "__main__":
    main()
