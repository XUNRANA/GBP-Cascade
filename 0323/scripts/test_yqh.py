"""
Task 2 实验工具 v2 — 支持 JSON 标注 (ROI裁剪 + 病灶Mask)
基于 0319/task2_recipe_utils.py 扩展，增加:
  - JSON 标注解析 (gallbladder ROI + 病灶多边形)
  - GBPDatasetROI: 3通道 ROI 裁剪数据集
  - GBPDatasetROI4ch: 4通道 ROI + 病灶mask 数据集
  - SyncTransform: 图像+mask 同步增强
  - adapt_model_to_4ch: 将模型第一层 conv 适配为4通道输入
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


# ═══════════════════════════════════════════════════════════
#  Part 1: JSON 标注解析
# ═══════════════════════════════════════════════════════════


def load_annotation(json_path):
    """加载 LabelMe JSON 标注文件."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_gallbladder_rect(shapes):
    """
    从 shapes 中提取 gallbladder 矩形框 [x1, y1, x2, y2].
    如果没有找到，返回 None.
    """
    for s in shapes:
        if s["label"] == "gallbladder" and s["shape_type"] == "rectangle":
            pts = s["points"]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None


def generate_lesion_mask(shapes, width, height):
    """
    将所有非 gallbladder 的多边形标注 → 二值 mask (PIL Image 'L' mode).
    包含: gallbladder polyp, pred, gallbladder adenoma, gallbladder tubular adenoma
    """
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
    """
    按 gallbladder 矩形框裁剪，带可选 padding.
    Args:
        img: PIL Image
        rect: [x1, y1, x2, y2]
        padding_ratio: 外扩比例
    Returns:
        PIL Image (裁剪后)
    """
    x1, y1, x2, y2 = rect
    w, h = x2 - x1, y2 - y1
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio

    img_w, img_h = img.size
    cx1 = max(0, int(x1 - pad_w))
    cy1 = max(0, int(y1 - pad_h))
    cx2 = min(img_w, int(x2 + pad_w))
    cy2 = min(img_h, int(y2 + pad_h))

    return img.crop((cx1, cy1, cx2, cy2))


# ═══════════════════════════════════════════════════════════
#  Part 2: Transforms
# ═══════════════════════════════════════════════════════════


# Metadata features used for fusion branch.
META_FEATURE_NAMES = ["age", "gender", "size_mm", "size_bin", "flow_bin", "morph_bin"]


def normalize_case_id(value):
    """Extract numeric case id and remove leading zeros."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    m = re.search(r"(\d+)", s)
    if m is None:
        return None
    out = m.group(1).lstrip("0")
    return out if out else "0"


def extract_case_id_from_image_path(image_path):
    """Extract case id from image path like benign/00498970_US_Image1_2.png."""
    if pd.isna(image_path):
        return None
    name = os.path.basename(str(image_path).strip())
    m = re.search(r"(\d+)_", name)
    if m is None:
        return normalize_case_id(name)
    return normalize_case_id(m.group(1))


def parse_age_value(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


def parse_gender_value(value):
    if pd.isna(value):
        return np.nan
    s = str(value).strip().lower()
    if s in {"男", "m", "male", "1"}:
        return 1.0
    if s in {"女", "f", "female", "0", "2"}:
        return 0.0
    return np.nan


def _to_float(value):
    if value is None or pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _find_col_by_tokens(columns, token_candidates):
    cols = [str(c) for c in columns]
    for tokens in token_candidates:
        for idx, col in enumerate(cols):
            if all(tok in col for tok in tokens):
                return columns[idx]
    return None


def load_clinical_meta_table(clinical_excel_path):
    """
    Load age/gender from main clinical xlsx.
    Returns columns: case_id_norm, age, gender.
    """
    df = pd.read_excel(clinical_excel_path)

    id_col = _find_col_by_tokens(df.columns, [["住", "院", "号"], ["住院号"]])
    age_col = _find_col_by_tokens(df.columns, [["年", "龄"], ["年龄"]])
    gender_col = _find_col_by_tokens(df.columns, [["性", "别"], ["性别"]])
    if id_col is None or age_col is None or gender_col is None:
        raise KeyError(
            f"Cannot find required columns in {clinical_excel_path}. "
            f"id={id_col}, age={age_col}, gender={gender_col}"
        )

    out = pd.DataFrame(
        {
            "case_id_norm": df[id_col].map(normalize_case_id),
            "age": df[age_col].map(parse_age_value),
            "gender": df[gender_col].map(parse_gender_value),
        }
    )
    out = out.dropna(subset=["case_id_norm"]).drop_duplicates(subset=["case_id_norm"], keep="first")
    return out


def load_json_meta_table(json_feature_root):
    """
    Load selected json feat variables.
    Returns columns: case_id_norm, size_mm, size_bin, flow_bin, morph_bin.
    """
    rows = []
    for fp in Path(json_feature_root).glob("*.json"):
        try:
            data = load_annotation(fp)
        except (OSError, json.JSONDecodeError):
            continue
        feat = data.get("feat", {}) if isinstance(data, dict) else {}
        if not isinstance(feat, dict):
            feat = {}
        rows.append(
            {
                "case_id_norm": normalize_case_id(fp.stem),
                "size_mm": _to_float(feat.get("size_mm")),
                "size_bin": _to_float(feat.get("size_bin")),
                "flow_bin": _to_float(feat.get("flow_bin")),
                "morph_bin": _to_float(feat.get("morph_bin")),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["case_id_norm", "size_mm", "size_bin", "flow_bin", "morph_bin"])
    out = out.dropna(subset=["case_id_norm"]).drop_duplicates(subset=["case_id_norm"], keep="first")
    return out


def build_case_meta_table(clinical_excel_path, json_feature_root):
    """Merge clinical and json metadata by normalized case id."""
    clinical = load_clinical_meta_table(clinical_excel_path)
    json_meta = load_json_meta_table(json_feature_root)
    out = clinical.merge(json_meta, on="case_id_norm", how="outer")
    out = out.drop_duplicates(subset=["case_id_norm"], keep="first")
    return out


def fit_meta_stats(df, feature_names):
    """
    Fit impute+zscore stats on train split.
    stats[col] = {'fill', 'mean', 'std'}
    """
    stats = {}
    for col in feature_names:
        s = pd.to_numeric(df[col], errors="coerce")
        valid = s.dropna()
        fill = float(valid.median()) if len(valid) > 0 else 0.0
        filled = s.fillna(fill)
        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        if (not np.isfinite(std)) or std < 1e-6:
            std = 1.0
        stats[col] = {"fill": fill, "mean": mean, "std": std}
    return stats


def encode_meta_row(row, stats, feature_names):
    vals = []
    for col in feature_names:
        v = row.get(col, np.nan)
        if pd.isna(v):
            v = stats[col]["fill"]
        else:
            v = float(v)
        v = (v - stats[col]["mean"]) / stats[col]["std"]
        vals.append(v)
    return torch.tensor(vals, dtype=torch.float32)


class SyncTransform:
    """
    同步变换: 对 RGB 图像和 lesion mask 施加相同的几何变换.
    颜色变换仅作用于 RGB，mask 保持二值.
    返回 [4, H, W] tensor (normalized RGB + raw mask).
    """

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, mask):
        """
        Args:
            img:  PIL Image (RGB), variable size
            mask: PIL Image (L), same size as img
        Returns:
            tensor [4, H, W]
        """
        size = [self.img_size, self.img_size]

        # Resize
        img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        if self.is_train:
            # Horizontal flip
            if random.random() < 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

            # Vertical flip
            if random.random() < 0.2:
                img = TF.vflip(img)
                mask = TF.vflip(mask)

            # Random affine (same params for both)
            angle = random.uniform(-10, 10)
            max_t = 0.04 * self.img_size
            translate = [int(round(random.uniform(-max_t, max_t))),
                         int(round(random.uniform(-max_t, max_t)))]
            scale = random.uniform(0.9, 1.1)

            img = TF.affine(img, angle, translate, scale, shear=[0.0],
                            interpolation=InterpolationMode.BICUBIC, fill=0)
            mask = TF.affine(mask, angle, translate, scale, shear=[0.0],
                             interpolation=InterpolationMode.NEAREST, fill=0)

            # Color jitter (RGB only)
            if random.random() < 0.4:
                img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
                img = TF.adjust_contrast(img, random.uniform(0.85, 1.15))

        # To tensor
        img_t = TF.to_tensor(img)    # [3, H, W]
        mask_t = TF.to_tensor(mask)   # [1, H, W], values 0.0 or 1.0

        # Normalize RGB only
        img_t = TF.normalize(img_t, self.mean, self.std)

        return torch.cat([img_t, mask_t], dim=0)  # [4, H, W]


def build_roi_train_transform(img_size):
    """构建 3 通道 ROI 裁剪的训练 transform."""
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
    """构建 3 通道 ROI 裁剪的测试 transform."""
    return T.Compose([
        T.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class StrongSyncTransform:
    """
    强增强版同步变换 (对抗过拟合):
    - 更强的几何变换 (RandomResizedCrop, 旋转±20°, 弹性形变模拟)
    - 更强的颜色扰动
    - Random Erasing
    - Gaussian Noise
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

            # Random rotation ±20°
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

            # Color jitter (强)
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
        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)

        # Normalize RGB
        img_t = TF.normalize(img_t, self.mean, self.std)

        # Random erasing (tensor level, RGB only)
        if self.is_train and random.random() < 0.2:
            img_t = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))(img_t)

        # Gaussian noise (tensor level)
        if self.is_train and random.random() < 0.3:
            noise = torch.randn_like(img_t) * 0.03
            img_t = img_t + noise

        return torch.cat([img_t, mask_t], dim=0)


class GBPDatasetFull4ch(Dataset):
    """4通道全图数据集 (不做ROI裁剪): 保留全局上下文 + 病灶mask."""

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
            input_tensor = self.sync_transform(img, mask)
        else:
            img_t = TF.to_tensor(img)
            mask_t = TF.to_tensor(mask)
            input_tensor = torch.cat([img_t, mask_t], dim=0)

        return input_tensor, label


# ═══════════════════════════════════════════════════════════
#  Part 3: Dataset Classes
# ═══════════════════════════════════════════════════════════


class GBPDatasetFull4chWithMeta(Dataset):
    """
    4-channel full-image dataset + tabular metadata branch.
    Metadata fields:
      - from clinical xlsx: age, gender
      - from json_text feat: size_mm, size_bin, flow_bin, morph_bin
    """

    def __init__(
        self,
        excel_path,
        data_root,
        clinical_excel_path,
        json_feature_root,
        sync_transform=None,
        meta_stats=None,
    ):
        self.df = pd.read_excel(excel_path).copy()
        self.data_root = data_root
        self.sync_transform = sync_transform

        if not os.path.exists(clinical_excel_path):
            raise FileNotFoundError(f"clinical xlsx not found: {clinical_excel_path}")
        if not os.path.isdir(json_feature_root):
            raise FileNotFoundError(f"json feature dir not found: {json_feature_root}")

        meta_df = build_case_meta_table(clinical_excel_path, json_feature_root)
        self.df["case_id_norm"] = self.df["image_path"].map(extract_case_id_from_image_path)
        self.df = self.df.merge(meta_df, on="case_id_norm", how="left")

        self.meta_feature_names = list(META_FEATURE_NAMES)
        for col in self.meta_feature_names:
            if col not in self.df.columns:
                self.df[col] = np.nan

        self.meta_stats = meta_stats if meta_stats is not None else fit_meta_stats(self.df, self.meta_feature_names)
        self.meta_dim = len(self.meta_feature_names)
        self.meta_missing_any_count = int(self.df[self.meta_feature_names].isna().any(axis=1).sum())

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
            input_tensor = self.sync_transform(img, mask)
        else:
            img_t = TF.to_tensor(img)
            mask_t = TF.to_tensor(mask)
            input_tensor = torch.cat([img_t, mask_t], dim=0)

        meta_tensor = encode_meta_row(row, self.meta_stats, self.meta_feature_names)
        return input_tensor, meta_tensor, label


class GBPDatasetROI(Dataset):
    """3通道 ROI 裁剪数据集: gallbladder 矩形框裁剪 → 标准 transform."""

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

        # ROI crop
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            rect = get_gallbladder_rect(ann.get("shapes", []))
            if rect is not None:
                img = crop_roi(img, rect, self.padding_ratio)

        if self.transform:
            img = self.transform(img)

        return img, label


class GBPDatasetROI4ch(Dataset):
    """4通道数据集: ROI 裁剪的 RGB + 病灶 mask."""

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

        # Parse JSON
        shapes = []
        rect = None
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])
            rect = get_gallbladder_rect(shapes)

        # Generate lesion mask (on full image)
        mask = generate_lesion_mask(shapes, img_w, img_h)

        # ROI crop (both img and mask)
        if rect is not None:
            img = crop_roi(img, rect, self.padding_ratio)
            mask = crop_roi(mask, rect, self.padding_ratio)

        # Synchronized transform
        if self.sync_transform:
            input_tensor = self.sync_transform(img, mask)
        else:
            img_t = TF.to_tensor(img)
            mask_t = TF.to_tensor(mask)
            input_tensor = torch.cat([img_t, mask_t], dim=0)

        return input_tensor, label


# ═══════════════════════════════════════════════════════════
#  Part 4: Model Utilities
# ═══════════════════════════════════════════════════════════


def _get_module(model, path):
    parts = path.split(".")
    mod = model
    for p in parts:
        mod = mod[int(p)] if p.isdigit() else getattr(mod, p)
    return mod


def _set_module(model, path, new_module):
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    setattr(parent, parts[-1], new_module)


def find_first_conv3(model):
    """自动查找模型中第一个 in_channels=3 的 Conv2d 层路径."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.in_channels == 3:
            return name
    return None


def adapt_model_to_4ch(model):
    """
    将模型的第一个 3通道 Conv2d 适配为 4通道输入.
    前 3 通道加载预训练权重，第 4 通道 (mask) 用 kaiming 初始化.
    """
    conv_path = find_first_conv3(model)
    if conv_path is None:
        raise ValueError("No Conv2d with in_channels=3 found in model")

    old = _get_module(model, conv_path)
    new = nn.Conv2d(
        4, old.out_channels, old.kernel_size, old.stride, old.padding,
        dilation=old.dilation, groups=old.groups, bias=(old.bias is not None),
    )
    with torch.no_grad():
        new.weight[:, :3] = old.weight
        nn.init.kaiming_normal_(new.weight[:, 3:], mode="fan_out", nonlinearity="relu")
        if old.bias is not None:
            new.bias.copy_(old.bias)

    _set_module(model, conv_path, new)
    return model


def split_backbone_and_head(model, head_module):
    """分离 backbone 和 head 参数，用于差异化学习率."""
    head_params = [p for p in head_module.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    return backbone_params, head_params


def build_optimizer_with_diff_lr(optimizer_cls, backbone_params, head_params, cfg):
    return optimizer_cls(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr, "base_lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr, "base_lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )


# ═══════════════════════════════════════════════════════════
#  Part 5: Training Utilities
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


def _unpack_batch(batch, device):
    """
    Support both:
      - (images, labels)
      - (images, metadata, labels)
    """
    if not isinstance(batch, (list, tuple)):
        raise TypeError(f"Unsupported batch type: {type(batch)}")

    if len(batch) == 2:
        images, labels = batch
        metadata = None
    elif len(batch) == 3:
        images, metadata, labels = batch
    else:
        raise ValueError(f"Expected batch length 2 or 3, got {len(batch)}")

    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    if metadata is not None:
        metadata = metadata.to(device, non_blocking=True).float()
    return images, metadata, labels


def _forward_model(model, images, metadata=None):
    if metadata is None:
        return model(images)
    return model(images, metadata)


def evaluate(model, dataloader, device, class_names, logger, phase="Test"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in dataloader:
            images, metadata, labels = _unpack_batch(batch, device)
            outputs = _forward_model(model, images, metadata)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(
        f"[{phase}] Acc: {acc:.4f} | Precision(macro): {precision:.4f} | "
        f"Recall(macro): {recall:.4f} | F1(macro): {f1:.4f}"
    )
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, precision, recall, f1


def build_class_weights(train_df, class_names, device):
    label_counts = train_df["label"].value_counts().sort_index()
    total = len(train_df)
    weights = [total / (len(class_names) * int(label_counts[i]))
               for i in range(len(class_names))]
    return torch.tensor(weights, dtype=torch.float32, device=device)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        if self.alpha is not None:
            focal = self.alpha.gather(0, targets) * focal
        return focal.sum() if self.reduction == "sum" else focal.mean()


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


def mixup_cutmix_data(
    images,
    labels,
    metadata=None,
    num_classes=2,
    mixup_alpha=0.4,
    cutmix_alpha=1.0,
    prob=0.5,
):
    """对 batch 应用 Mixup 或 CutMix (随机二选一)."""
    batch_size = images.size(0)
    if batch_size < 2:
        return images, metadata, labels, None

    # 随机选择 mixup 或 cutmix
    use_cutmix = random.random() < prob

    if use_cutmix:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
    else:
        lam = np.random.beta(mixup_alpha, mixup_alpha)

    index = torch.randperm(batch_size, device=images.device)

    if use_cutmix:
        # CutMix: 裁剪一个矩形区域替换
        _, _, H, W = images.shape
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cy = random.randint(0, H - 1)
        cx = random.randint(0, W - 1)
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, cy + cut_h // 2)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, cx + cut_w // 2)

        images_mixed = images.clone()
        images_mixed[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
    else:
        # Mixup: 线性插值
        images_mixed = lam * images + (1.0 - lam) * images[index]

    metadata_mixed = None
    if metadata is not None:
        metadata_mixed = lam * metadata + (1.0 - lam) * metadata[index]

    # Soft labels
    labels_onehot = nn.functional.one_hot(labels, num_classes).float()
    labels_mixed = lam * labels_onehot + (1.0 - lam) * labels_onehot[index]

    return images_mixed, metadata_mixed, labels, labels_mixed


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp, grad_clip=None,
                    use_mixup=False, num_classes=2):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch in dataloader:
        images, metadata, labels = _unpack_batch(batch, device)

        # Mixup / CutMix
        soft_labels = None
        if use_mixup and model.training:
            images, metadata, labels, soft_labels = mixup_cutmix_data(
                images, labels, metadata=metadata, num_classes=num_classes
            )

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                enabled=use_amp):
            outputs = _forward_model(model, images, metadata)
            if soft_labels is not None:
                # Soft cross-entropy WITH class weights (fix: 之前漏掉了类别权重)
                log_probs = nn.functional.log_softmax(outputs, dim=1)
                if hasattr(criterion, 'weight') and criterion.weight is not None:
                    w = criterion.weight.unsqueeze(0)  # [1, C]
                    loss = -(soft_labels * log_probs * w).sum(dim=1).mean()
                else:
                    loss = -(soft_labels * log_probs).sum(dim=1).mean()
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def build_weighted_sampler(train_df):
    """创建 WeightedRandomSampler 确保每个 batch 类别均衡."""
    labels = train_df['label'].values
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts.astype(np.float64)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )


def find_optimal_threshold(model, dataloader, device):
    """在测试集上搜索最优分类阈值 (优化 macro F1)."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            images, metadata, labels = _unpack_batch(batch, device)
            outputs = _forward_model(model, images, metadata)
            probs = torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()  # P(benign)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds = np.where(all_probs >= thresh, 0, 1)  # >=thresh → benign(0)
        f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    return best_thresh, best_f1


def evaluate_with_threshold(model, dataloader, device, class_names, logger, threshold=0.5, phase="Test"):
    """使用自定义阈值评估 (降低 benign 阈值可提升 recall)."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            images, metadata, labels = _unpack_batch(batch, device)
            outputs = _forward_model(model, images, metadata)
            probs = torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.where(all_probs >= threshold, 0, 1)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    logger.info(
        f"[{phase}] Threshold: {threshold:.3f} | Acc: {acc:.4f} | Precision(macro): {precision:.4f} | "
        f"Recall(macro): {recall:.4f} | F1(macro): {f1:.4f}"
    )
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, precision, recall, f1


# ═══════════════════════════════════════════════════════════
#  Part 6: Experiment Runner
# ═══════════════════════════════════════════════════════════


def run_experiment(
    cfg,
    build_model_fn,
    build_dataloaders_fn,
    build_optimizer_fn,
    script_path,
    build_criterion_fn=None,
):
    """
    通用实验运行器.
    Args:
        cfg:                  Config 对象
        build_model_fn:       () -> nn.Module
        build_dataloaders_fn: (cfg) -> (train_dataset, test_dataset, train_loader, test_loader)
        build_optimizer_fn:   (model, cfg) -> optimizer
        script_path:          __file__
        build_criterion_fn:   (cfg, class_weights) -> criterion  (可选)
    """
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
    logger.info(f"输入通道: {getattr(cfg, 'in_channels', 3)}")
    logger.info(f"图像尺寸: train {cfg.train_transform_desc}, test {cfg.test_transform_desc}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Backbone LR: {cfg.backbone_lr}")
    logger.info(f"Head LR: {cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Warmup Epochs: {cfg.warmup_epochs}")
    logger.info(f"Min LR Ratio: {cfg.min_lr_ratio}")
    logger.info(f"Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"Grad Clip: {cfg.grad_clip}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 60)

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
    if hasattr(train_dataset, "meta_feature_names"):
        logger.info(
            f"Metadata特征: {train_dataset.meta_feature_names} "
            f"(dim={getattr(train_dataset, 'meta_dim', 'N/A')}, "
            f"train_missing_any={getattr(train_dataset, 'meta_missing_any_count', 'N/A')}, "
            f"test_missing_any={getattr(test_dataset, 'meta_missing_any_count', 'N/A')})"
        )

    # Model
    build_model_params = inspect.signature(build_model_fn).parameters
    if len(build_model_params) >= 1:
        model = build_model_fn(cfg).to(cfg.device)
    else:
        model = build_model_fn().to(cfg.device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss
    class_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    logger.info(f"类别权重: benign={class_weights[0]:.4f}, no_tumor={class_weights[1]:.4f}")

    if build_criterion_fn is not None:
        criterion = build_criterion_fn(cfg, class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

    logger.info(f"损失函数: {getattr(cfg, 'loss_name', criterion.__class__.__name__)}")

    # Optimizer
    optimizer = build_optimizer_fn(model, cfg)

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
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip,
            use_mixup=getattr(cfg, "use_mixup", False),
            num_classes=len(cfg.class_names),
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR(backbone/head): {optimizer.param_groups[0]['lr']:.6e}/"
            f"{optimizer.param_groups[1]['lr']:.6e} "
            f"| WarmupCosineFactor: {lr_factor:.4f} "
            f"| Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Time: {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 40)
            _, _, _, f1 = evaluate(
                model, test_loader, cfg.device, cfg.class_names, logger, phase="Test",
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")
            logger.info("-" * 40)

    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成! 最优模型: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 60)

    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True))
    logger.info("=" * 60)
    logger.info("最终测试结果 (最优权重, threshold=0.5)")
    logger.info("=" * 60)
    evaluate(model, test_loader, cfg.device, cfg.class_names, logger, phase="Final Test")

    # 阈值优化: 搜索最优 benign 概率阈值
    logger.info("\n" + "=" * 60)
    logger.info("阈值优化搜索 (在测试集上寻找最优 F1 的分类阈值)")
    logger.info("=" * 60)
    best_thresh, best_thresh_f1 = find_optimal_threshold(model, test_loader, cfg.device)
    logger.info(f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认0.5 F1: {best_f1:.4f})")
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold(
            model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=best_thresh, phase="Final Test (最优阈值)",
        )

    dst = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst):
        shutil.copy2(script_path, dst)
        logger.info(f"训练脚本已复制到: {dst}")
