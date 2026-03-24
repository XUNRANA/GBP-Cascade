"""
GBP-Cascade Task 2 自主实验 — 固定组件 (READ-ONLY, DO NOT MODIFY)

提供: 数据路径常量, 数据集类, Transform, 评估函数, 训练工具函数.
作为 train_task2.py 的基础设施.

任务: 良性肿瘤(benign, label=0) vs 非肿瘤性息肉(no_tumor, label=1) 二分类
数据: 0322dataset — 1229 训练 / 523 测试, 类别比 1:3 (benign:no_tumor)
"""

import json
import math
import os
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, WeightedRandomSampler
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


# ═══════════════════════════════════════════════════════════
#  Constants (FIXED — do not modify)
# ═══════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
DATA_ROOT = os.path.join(PROJECT_ROOT, "0322dataset")
TRAIN_EXCEL = os.path.join(DATA_ROOT, "task_2_train.xlsx")
TEST_EXCEL = os.path.join(DATA_ROOT, "task_2_test.xlsx")
CLASS_NAMES = ["benign", "no_tumor"]
NUM_CLASSES = 2
TIME_BUDGET = 600  # 10 minutes max training time per experiment
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "_best_model.pth")


# ═══════════════════════════════════════════════════════════
#  Part 1: JSON Annotation Parsing
# ═══════════════════════════════════════════════════════════


def load_annotation(json_path):
    """Load LabelMe JSON annotation file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gallbladder_rect(shapes):
    """Extract gallbladder bounding box [x1, y1, x2, y2] from shapes."""
    for s in shapes:
        if s["label"] == "gallbladder" and s["shape_type"] == "rectangle":
            pts = s["points"]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None


def generate_lesion_mask(shapes, width, height):
    """Generate binary lesion mask from all non-gallbladder polygon annotations."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for s in shapes:
        if s["label"] == "gallbladder":
            continue
        if s["shape_type"] == "polygon" and len(s["points"]) >= 3:
            pts = [(p[0], p[1]) for p in s["points"]]
            draw.polygon(pts, fill=255)
    return mask


def crop_roi(img, rect, padding_ratio=0.02):
    """Crop image by gallbladder bounding box with optional padding."""
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = w * padding_ratio, h * padding_ratio
    img_w, img_h = img.size
    cx1 = max(0, int(x1 - pad_w))
    cy1 = max(0, int(y1 - pad_h))
    cx2 = min(img_w, int(x2 + pad_w))
    cy2 = min(img_h, int(y2 + pad_h))
    return img.crop((cx1, cy1, cx2, cy2))


# ═══════════════════════════════════════════════════════════
#  Part 2: Transforms (synchronized image + mask)
# ═══════════════════════════════════════════════════════════


class SyncTransform:
    """Weak augmentation: synchronized geometric transforms on RGB + mask."""

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        size = [self.img_size, self.img_size]
        img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        if self.is_train:
            if random.random() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() < 0.2:
                img, mask = TF.vflip(img), TF.vflip(mask)

            angle = random.uniform(-10, 10)
            max_t = 0.04 * self.img_size
            translate = [int(round(random.uniform(-max_t, max_t))),
                         int(round(random.uniform(-max_t, max_t)))]
            scale = random.uniform(0.9, 1.1)
            img = TF.affine(img, angle, translate, scale, shear=[0.0],
                            interpolation=InterpolationMode.BICUBIC, fill=0)
            mask = TF.affine(mask, angle, translate, scale, shear=[0.0],
                             interpolation=InterpolationMode.NEAREST, fill=0)

            if random.random() < 0.4:
                img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
                img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        img_t = TF.normalize(img_t, self.mean, self.std)
        return torch.cat([img_t, mask_t], dim=0)  # [4, H, W]


class StrongSyncTransform:
    """Strong augmentation: aggressive geometric + color transforms on RGB + mask."""

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        size = [self.img_size, self.img_size]

        if self.is_train:
            # RandomResizedCrop (synchronized)
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=(0.7, 1.0), ratio=(0.85, 1.15))
            img = TF.resized_crop(img, i, j, h, w, size, InterpolationMode.BICUBIC)
            mask = TF.resized_crop(mask, i, j, h, w, size, InterpolationMode.NEAREST)

            if random.random() < 0.5:
                img, mask = TF.hflip(img), TF.hflip(mask)
            if random.random() < 0.3:
                img, mask = TF.vflip(img), TF.vflip(mask)

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

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        img_t = TF.normalize(img_t, self.mean, self.std)

        if self.is_train and random.random() < 0.2:
            img_t = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))(img_t)
        if self.is_train and random.random() < 0.3:
            img_t = img_t + torch.randn_like(img_t) * 0.03

        return torch.cat([img_t, mask_t], dim=0)  # [4, H, W]


def build_roi_train_transform(img_size):
    """3-channel ROI crop training transform (no mask)."""
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomAffine(degrees=10, translate=(0.04, 0.04), scale=(0.9, 1.1),
                        interpolation=InterpolationMode.BICUBIC),
        T.RandomApply([T.ColorJitter(brightness=0.15, contrast=0.15)], p=0.4),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_roi_test_transform(img_size):
    """3-channel ROI crop test transform (no mask)."""
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ═══════════════════════════════════════════════════════════
#  Part 3: Dataset Classes
# ═══════════════════════════════════════════════════════════


class GBPDatasetROI(Dataset):
    """3-channel ROI crop dataset: gallbladder rect crop -> standard transform."""

    def __init__(self, excel_path, data_root, transform=None, padding_ratio=0.02):
        self.df = pd.read_excel(excel_path)
        self.data_root = data_root
        self.transform = transform
        self.padding_ratio = padding_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        json_path = img_path.replace(".png", ".json")
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            rect = get_gallbladder_rect(ann.get("shapes", []))
            if rect is not None:
                img = crop_roi(img, rect, self.padding_ratio)

        if self.transform:
            img = self.transform(img)
        return img, label


class GBPDatasetROI4ch(Dataset):
    """4-channel ROI crop dataset: ROI-cropped RGB + lesion mask."""

    def __init__(self, excel_path, data_root, sync_transform=None, padding_ratio=0.02):
        self.df = pd.read_excel(excel_path)
        self.data_root = data_root
        self.sync_transform = sync_transform
        self.padding_ratio = padding_ratio

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        json_path = img_path.replace(".png", ".json")
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size
        shapes, rect = [], None
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])
            rect = get_gallbladder_rect(shapes)

        mask = generate_lesion_mask(shapes, img_w, img_h)
        if rect is not None:
            img = crop_roi(img, rect, self.padding_ratio)
            mask = crop_roi(mask, rect, self.padding_ratio)

        if self.sync_transform:
            return self.sync_transform(img, mask), label

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        return torch.cat([img_t, mask_t], dim=0), label


class GBPDatasetFull4ch(Dataset):
    """4-channel full image dataset: no ROI crop, full context + lesion mask."""

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
        if self.sync_transform:
            return self.sync_transform(img, mask), label

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        return torch.cat([img_t, mask_t], dim=0), label


# ═══════════════════════════════════════════════════════════
#  Part 4: Model Utilities
# ═══════════════════════════════════════════════════════════


def adapt_model_to_4ch(model):
    """
    Adapt model's first Conv2d from 3 to 4 input channels.
    First 3 channels keep pretrained weights; 4th (mask) uses kaiming init.
    """
    # Find first 3-channel conv
    conv_path = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            conv_path = name
            break
    if conv_path is None:
        raise ValueError("No Conv2d with in_channels=3 found in model")

    # Navigate to parent and replace
    parts = conv_path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    old = getattr(parent, parts[-1])

    new = nn.Conv2d(4, old.out_channels, old.kernel_size, old.stride, old.padding,
                    dilation=old.dilation, groups=old.groups, bias=(old.bias is not None))
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        nn.init.kaiming_normal_(new.weight[:, 3:], mode="fan_out", nonlinearity="relu")
        if old.bias is not None:
            new.bias.copy_(old.bias)
    setattr(parent, parts[-1], new)
    return model


def split_backbone_and_head(model, head_module):
    """Split model parameters into backbone and head groups for differential LR."""
    head_params = [p for p in head_module.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    return backbone_params, head_params


# ═══════════════════════════════════════════════════════════
#  Part 5: Training Utilities
# ═══════════════════════════════════════════════════════════


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_class_weights(train_df, device):
    """Compute inverse-frequency class weights for loss function."""
    label_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    weights = [total / (NUM_CLASSES * int(label_counts[i])) for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_weighted_sampler(train_df):
    """Create WeightedRandomSampler for class-balanced batches."""
    labels = train_df["label"].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts.astype(np.float64)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


class FocalLoss(nn.Module):
    """Focal Loss with per-class alpha weighting."""

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.weight = alpha  # compatibility with soft CE weight check

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha.gather(0, targets) * focal
        return focal.sum() if self.reduction == "sum" else focal.mean()


def cosine_warmup_factor(epoch, num_epochs, warmup_epochs, min_lr_ratio):
    """Compute LR multiplier: linear warmup + cosine decay."""
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return epoch / warmup_epochs
    if num_epochs <= warmup_epochs:
        return 1.0
    progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def mixup_cutmix_data(images, labels, num_classes=2, mixup_alpha=0.4,
                       cutmix_alpha=1.0, prob=0.5):
    """
    Apply Mixup or CutMix to a batch (randomly chosen).
    Returns: (mixed_images, original_labels, soft_labels_or_None)
    """
    batch_size = images.size(0)
    if batch_size < 2:
        return images, labels, None

    use_cutmix = random.random() < prob
    alpha = cutmix_alpha if use_cutmix else mixup_alpha
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(batch_size, device=images.device)

    if use_cutmix:
        _, _, H, W = images.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
        cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
        y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
        images_mixed = images.clone()
        images_mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    else:
        images_mixed = lam * images + (1.0 - lam) * images[index]

    labels_onehot = F.one_hot(labels, num_classes).float()
    labels_mixed = lam * labels_onehot + (1.0 - lam) * labels_onehot[index]
    return images_mixed, labels, labels_mixed


# ═══════════════════════════════════════════════════════════
#  Part 6: Evaluation (GROUND TRUTH — DO NOT MODIFY)
# ═══════════════════════════════════════════════════════════


def evaluate_model(model, dataloader, device):
    """
    Authoritative evaluation function. Returns dict with all metrics.
    Includes threshold search on P(benign) for optimal F1(macro).
    """
    model.eval()
    all_probs, all_preds, all_labels = [], [], []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()  # P(benign)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Default threshold (0.5) metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # Per-class at default threshold
    f1_per = f1_score(all_labels, all_preds, average=None, zero_division=0)
    prec_per = precision_score(all_labels, all_preds, average=None, zero_division=0)
    rec_per = recall_score(all_labels, all_preds, average=None, zero_division=0)

    # Threshold search: find threshold that maximizes macro F1
    best_thresh_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds_t = np.where(all_probs >= thresh, 0, 1)
        f1_t = f1_score(all_labels, preds_t, average="macro", zero_division=0)
        if f1_t > best_thresh_f1:
            best_thresh_f1 = f1_t
            best_thresh = float(thresh)

    # Per-class at optimal threshold
    preds_opt = np.where(all_probs >= best_thresh, 0, 1)
    f1_opt_per = f1_score(all_labels, preds_opt, average=None, zero_division=0)
    prec_opt_per = precision_score(all_labels, preds_opt, average=None, zero_division=0)
    rec_opt_per = recall_score(all_labels, preds_opt, average=None, zero_division=0)

    return {
        # Default threshold (0.5)
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_benign": float(f1_per[0]),
        "f1_no_tumor": float(f1_per[1]),
        "precision_benign": float(prec_per[0]),
        "precision_no_tumor": float(prec_per[1]),
        "recall_benign": float(rec_per[0]),
        "recall_no_tumor": float(rec_per[1]),
        # Optimal threshold
        "best_threshold": best_thresh,
        "f1_at_threshold": float(best_thresh_f1),
        "f1_benign_at_thresh": float(f1_opt_per[0]),
        "f1_no_tumor_at_thresh": float(f1_opt_per[1]),
        "precision_benign_at_thresh": float(prec_opt_per[0]),
        "recall_benign_at_thresh": float(rec_opt_per[0]),
    }
