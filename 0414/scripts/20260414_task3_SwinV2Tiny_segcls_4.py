"""
0414 Exp-D: 针对性优化 — SupCon + Focal + 调整代价矩阵 + 均衡采样优化

核心问题: benign recall 极低 (Exp-B 仅 24~37%), benign/no_tumor 特征空间高度重叠

针对性改进 (在 Exp-C 基础上):
  1. Supervised Contrastive Loss (SupCon): 在 cls_feat 上施加对比损失,
     直接拉开 benign vs no_tumor 的特征距离 (治本)
  2. Focal CostSensitive Loss: focal weighting (gamma=2) 聚焦难分样本,
     benign 是最难分的类 → 自动获得更大梯度
  3. 调整代价矩阵:
     - benign→no_tumor: 2.0→3.0 (漏掉良性肿瘤代价更大)
     - no_tumor→benign: 1.0→0.5 (息肉被诊断为良性, 临床可接受)
     - malignant→no_tumor: 4.0→5.0 (进一步加强安全底线)
  4. BalancedBatchSampler 2:3:3 (增加 benign 每 batch 占比)
  5. Ordinal 目标微调: benign 0.5→0.35 (拉大 benign 与 no_tumor 的分数间距)
  6. Label smoothing 0.1→0.05 (focal 已处理过度自信, 减少平滑)

保留 Exp-C 机制:
  - Seg-Guided Attention + 10D metadata
  - Ordinal score 0~1 + 双阈值搜索
  - malignant seg loss 自动置零
  - 患者级 val 切分 + 临床指标
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
SCRIPTS_0408 = os.path.join(ROOT_DIR, "0408", "scripts")
if SCRIPTS_0402 not in sys.path:
    sys.path.insert(0, SCRIPTS_0402)
if SCRIPTS_0408 not in sys.path:
    sys.path.insert(0, SCRIPTS_0408)

from seg_cls_utils_v2 import (  # noqa: E402
    load_annotation, generate_lesion_mask, UNetDecoderBlock, DiceLoss,
    set_seed, setup_logger, acquire_run_lock, build_class_weights,
    set_epoch_lrs, build_optimizer_with_diff_lr, compute_seg_metrics,
)
from seg_cls_utils import cosine_warmup_factor  # noqa: E402
from seg_cls_utils_v5 import (  # noqa: E402
    build_ext_case_meta_table, EXT_CLINICAL_FEATURE_NAMES,
)

sys.path.insert(0, os.path.join(ROOT_DIR, "0323", "scripts"))
from test_yqh import (  # noqa: E402
    adapt_model_to_4ch, fit_meta_stats, encode_meta_row,
    extract_case_id_from_image_path,
)

import timm  # noqa: E402


# ═════════════════════════════════════════════════════════════
#  1. 辅助函数
# ═════════════════════════════════════════════════════════════

def extract_patient_id(image_path: str) -> str:
    fname = os.path.basename(image_path)
    return fname.split("_US_")[0] if "_US_" in fname else fname.rsplit("_", 1)[0]


def generate_gallbladder_mask(shapes, width, height):
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
#  2. 损失函数 (Exp-D 针对性改进)
# ═════════════════════════════════════════════════════════════

# ── 调整后的代价矩阵 ──
# vs Exp-B/C:
#   malignant→no_tumor: 4→5 (加强安全底线)
#   benign→no_tumor:    2→3 (漏掉良性肿瘤代价更大)
#   benign→malignant:   1→2 (不必要激进手术)
#   no_tumor→benign:    1→0.5 (息肉诊断为良性, 临床可接受)
COST_MATRIX = torch.tensor([
    # pred_mal  pred_ben  pred_notumor
    [0.0,       1.0,      5.0],   # true = malignant
    [2.0,       0.0,      3.0],   # true = benign
    [1.5,       0.5,      0.0],   # true = no_tumor
], dtype=torch.float32)

# ── 调整后的 Ordinal 目标 ──
# benign 0.5→0.35: 拉大 benign 与 no_tumor 的分数间距 (0.35 vs 0.0)
# 同时保持 malignant 与 benign 间距 (1.0 vs 0.35 = 0.65)
ORDINAL_TARGETS = torch.tensor([1.0, 0.35, 0.0], dtype=torch.float32)


class FocalCostSensitiveLoss(nn.Module):
    """
    Focal + 代价敏感分类损失.

    改进: 加入 focal weighting (1-pt)^gamma, 自动聚焦难分样本.
    benign 作为最难分的类, 会自动获得更大的梯度权重.
    """

    def __init__(self, cost_matrix, gamma=2.0, penalty_weight=0.5,
                 label_smoothing=0.05):
        super().__init__()
        self.register_buffer("cost_matrix", cost_matrix)
        self.gamma = gamma
        self.penalty_weight = penalty_weight

        row_sums = cost_matrix.sum(dim=1)
        cls_weights = row_sums / row_sums.mean()
        self.register_buffer("cls_weights", cls_weights)
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        # Per-sample CE (不做 reduction)
        ce_per_sample = F.cross_entropy(
            logits, labels, weight=self.cls_weights,
            label_smoothing=self.label_smoothing, reduction="none",
        )
        # Focal weighting
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma
        focal_ce = (focal_weight * ce_per_sample).mean()

        # 期望误分类代价惩罚
        costs = self.cost_matrix[labels]
        penalty = (costs * probs).sum(dim=1).mean()

        return focal_ce + self.penalty_weight * penalty


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss.

    拉近同类样本的嵌入, 推远不同类样本.
    核心作用: 强制 benign 和 no_tumor 在特征空间中分离.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: (B, D) — L2 normalized embeddings
        labels: (B,) — class labels
        """
        device = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        # 相似度矩阵
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # 同类 mask (排除对角线)
        labels_col = labels.unsqueeze(0)  # (1, B)
        labels_row = labels.unsqueeze(1)  # (B, 1)
        pos_mask = (labels_row == labels_col).float()  # (B, B)
        diag_mask = 1.0 - torch.eye(B, device=device)
        pos_mask = pos_mask * diag_mask

        # 数值稳定: 减去每行最大值
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max

        # log-softmax over all non-self pairs
        exp_logits = torch.exp(logits) * diag_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        # 每个样本的正样本对数量
        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)

        # 对正样本对取平均 log-prob
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / pos_count

        loss = -mean_log_prob.mean()
        return loss


class OrdinalScoreLoss(nn.Module):
    def __init__(self, ordinal_targets):
        super().__init__()
        self.register_buffer("targets", ordinal_targets)

    def forward(self, score, labels):
        target = self.targets[labels]
        return F.smooth_l1_loss(score, target)


class SegClsOrdinalSupConLoss(nn.Module):
    """
    总损失 = seg + λ_cls * focal_cls + λ_ord * ord + λ_con * supcon

    新增 SupCon 项: 在 cls_feat embedding 上施加对比损失.
    """

    def __init__(self, cost_matrix, ordinal_targets,
                 lambda_cls=2.0, lambda_ord=0.5, lambda_con=0.3,
                 gamma=2.0, penalty_weight=0.5, label_smoothing=0.05,
                 seg_ce_weight=None, temperature=0.1):
        super().__init__()
        self.lambda_cls = lambda_cls
        self.lambda_ord = lambda_ord
        self.lambda_con = lambda_con

        self.cls_loss_fn = FocalCostSensitiveLoss(
            cost_matrix, gamma=gamma, penalty_weight=penalty_weight,
            label_smoothing=label_smoothing,
        )
        self.ord_loss_fn = OrdinalScoreLoss(ordinal_targets)
        self.con_loss_fn = SupConLoss(temperature=temperature)
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, cls_logits, ordinal_score, proj_feat,
                seg_targets, cls_targets, has_mask):
        # 分类损失 (Focal + Cost-sensitive)
        cls_loss = self.cls_loss_fn(cls_logits, cls_targets)
        # Ordinal 损失
        ord_loss = self.ord_loss_fn(ordinal_score, cls_targets)
        # Supervised Contrastive 损失
        con_loss = self.con_loss_fn(proj_feat, cls_targets)
        # 分割损失
        seg_loss = torch.tensor(0.0, device=seg_logits.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce + seg_dice

        total = (seg_loss
                 + self.lambda_cls * cls_loss
                 + self.lambda_ord * ord_loss
                 + self.lambda_con * con_loss)
        return total, seg_loss.item(), cls_loss.item(), ord_loss.item(), con_loss.item()


# ═════════════════════════════════════════════════════════════
#  3. BalancedBatchSampler (调整为 2:3:3)
# ═════════════════════════════════════════════════════════════

class BalancedBatchSampler(Sampler):
    """每个 batch 固定各类数量. Exp-D: 2:3:3 (mal:ben:notumor)."""

    def __init__(self, labels, samples_per_class, shuffle=True):
        self.labels = np.array(labels)
        self.samples_per_class = samples_per_class
        self.shuffle = shuffle

        self.class_indices = {}
        for cls_label, count in samples_per_class.items():
            self.class_indices[cls_label] = np.where(self.labels == cls_label)[0]
            assert len(self.class_indices[cls_label]) > 0, \
                f"Class {cls_label} has no samples"

        self.batch_size = sum(samples_per_class.values())
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
        class_iters = {}
        for cls_label, indices in self.class_indices.items():
            idx = indices.copy()
            if self.shuffle:
                np.random.shuffle(idx)
            count = self.samples_per_class[cls_label]
            needed = self.num_batches * count
            repeats = (needed // len(idx)) + 2
            idx = np.tile(idx, repeats)
            class_iters[cls_label] = idx

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
#  4. 数据增强 & Dataset (同 Exp-C)
# ═════════════════════════════════════════════════════════════

class SegCls0414SyncTransform:
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


class GBPDataset0414WithMeta(Dataset):
    """0414 dataset: 4ch + seg + label + 10D metadata."""

    def __init__(self, df, data_root, clinical_excel_path, json_feature_root,
                 sync_transform=None, meta_stats=None):
        self.df = df.copy().reset_index(drop=True)
        self.data_root = data_root
        self.sync_transform = sync_transform

        meta_df = build_ext_case_meta_table(clinical_excel_path, json_feature_root)
        self.df["case_id_norm"] = self.df["image_path"].map(extract_case_id_from_image_path)
        self.df = self.df.merge(meta_df, on="case_id_norm", how="left")

        self.meta_feature_names = list(EXT_CLINICAL_FEATURE_NAMES)
        for col in self.meta_feature_names:
            if col not in self.df.columns:
                self.df[col] = np.nan

        self.meta_stats = (
            meta_stats if meta_stats is not None
            else fit_meta_stats(self.df, self.meta_feature_names)
        )
        self.meta_dim = len(self.meta_feature_names)
        self.meta_missing_count = int(
            self.df[self.meta_feature_names].isna().any(axis=1).sum()
        )

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

        meta_tensor = encode_meta_row(row, self.meta_stats, self.meta_feature_names)
        return input_4ch, seg_target, label, has_lesion_mask, meta_tensor


def collate_fn(batch):
    inputs, masks, labels, has_masks, metas = zip(*batch)
    return (torch.stack(inputs), torch.stack(masks),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(has_masks, dtype=torch.bool),
            torch.stack(metas))


# ═════════════════════════════════════════════════════════════
#  5. 模型: SegGuided + Meta + SupCon Projection Head
# ═════════════════════════════════════════════════════════════

class SwinV2SegGuidedOrdinalMetaSupConModel(nn.Module):
    """
    Exp-D 模型: Exp-C 架构 + SupCon Projection Head

    新增:
      - proj_head: 256D cls_feat → 128D → 128D (L2 normalized)
        专用于 SupConLoss, 不影响分类/ordinal head
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=3, meta_dim=10,
                 meta_hidden=64, meta_dropout=0.2, cls_dropout=0.4,
                 proj_dim=128, pretrained=True):
        super().__init__()
        self.meta_dim = meta_dim

        # ── Encoder ──
        self.encoder = timm.create_model(
            "swinv2_tiny_window8_256", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
        )
        adapt_model_to_4ch(self.encoder)
        fc = [info["num_chs"] for info in self.encoder.feature_info]

        # ── Seg decoder ──
        self.dec3 = UNetDecoderBlock(fc[3], fc[2], fc[2])
        self.dec2 = UNetDecoderBlock(fc[2], fc[1], fc[1])
        self.dec1 = UNetDecoderBlock(fc[1], fc[0], fc[0])
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(fc[0], 48, kernel_size=4, stride=4),
            nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, num_seg_classes, 1),
        )

        # ── Seg-Guided Attention ──
        self.cls_proj = nn.Sequential(
            nn.Conv2d(fc[2], 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
        )

        # ── Metadata encoder ──
        fusion_in = 256
        self.meta_encoder = None
        if meta_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_dim, meta_hidden), nn.LayerNorm(meta_hidden),
                nn.GELU(), nn.Dropout(meta_dropout),
                nn.Linear(meta_hidden, meta_hidden), nn.GELU(), nn.Dropout(meta_dropout),
            )
            fusion_in = 256 + meta_hidden

        # ── 分类头 ──
        self.cls_head = nn.Sequential(
            nn.Linear(fusion_in, 128), nn.GELU(), nn.Dropout(cls_dropout),
            nn.Linear(128, num_cls_classes),
        )

        # ── Ordinal score head ──
        self.ord_head = nn.Linear(fusion_in, 1)

        # ── SupCon Projection Head (新增) ──
        # 使用 cls_feat (256D, 纯图像特征) 做对比学习
        self.proj_head = nn.Sequential(
            nn.Linear(256, proj_dim), nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def _to_bchw(self, x):
        if x.ndim == 4 and x.shape[1] != x.shape[3]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, metadata=None):
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        # ── Seg ──
        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        # ── Seg-Guided Attention ──
        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]
        attn = F.interpolate(seg_prob, size=f2.shape[2:], mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)

        f2_proj = self.cls_proj(f2)
        cls_feat = (f2_proj * attn).sum(dim=(2, 3))  # (B, 256)

        # ── SupCon Projection (L2 normalized) ──
        proj_feat = self.proj_head(cls_feat)
        proj_feat = F.normalize(proj_feat, dim=1)  # (B, 128)

        # ── Metadata fusion ──
        fused_feat = cls_feat
        if self.meta_encoder is not None and metadata is not None:
            meta_feat = self.meta_encoder(metadata.float())
            fused_feat = torch.cat([cls_feat, meta_feat], dim=1)  # (B, 320)

        # ── Cls + Ordinal ──
        cls_logits = self.cls_head(fused_feat)
        ord_score = torch.sigmoid(self.ord_head(fused_feat).squeeze(-1))

        return seg_logits, cls_logits, ord_score, proj_feat


# ═════════════════════════════════════════════════════════════
#  6. 训练 & 评估
# ═════════════════════════════════════════════════════════════

def train_one_epoch(model, dataloader, criterion, optimizer, device,
                    scaler, use_amp, grad_clip=None, num_seg_classes=2):
    model.train()
    stats = {"loss": 0, "seg": 0, "cls": 0, "ord": 0, "con": 0,
             "cls_correct": 0, "total": 0, "seg_dices": []}

    for imgs, masks, labels, has_masks, metas in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            seg_logits, cls_logits, ord_score, proj_feat = model(imgs, metadata=metas)
            loss, seg_l, cls_l, ord_l, con_l = criterion(
                seg_logits, cls_logits, ord_score, proj_feat,
                masks, labels, has_masks)

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
        stats["con"] += con_l * bs
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
        "con_loss": stats["con"] / n,
        "cls_acc": stats["cls_correct"] / n,
        "seg_dice": np.mean(stats["seg_dices"]) if stats["seg_dices"] else 0.0,
    }


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_logits, all_scores, all_labels = [], [], []
    all_seg_dices = []

    for imgs, masks, labels, has_masks, metas in dataloader:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)

        seg_logits, cls_logits, ord_score, _ = model(imgs, metadata=metas)
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
    best_f1, best_t1, best_t2 = 0.0, 0.33, 0.66
    for t1 in np.arange(*t1_range):
        for t2 in np.arange(*t2_range):
            if t2 <= t1:
                continue
            preds = np.where(scores >= t2, 0,
                    np.where(scores >= t1, 1, 2))
            f1 = f1_score(labels, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t1, best_t2 = f1, t1, t2
    return best_t1, best_t2, best_f1


def compute_clinical_metrics(scores, labels, t1, t2, logger, phase="Test"):
    preds = np.where(scores >= t2, 0,
            np.where(scores >= t1, 1, 2))

    cm = confusion_matrix(labels, preds, labels=[0, 1, 2])
    logger.info(f"[{phase}] 混淆矩阵 (行=真实, 列=预测):")
    logger.info(f"         pred_mal  pred_ben  pred_notumor")
    names = ["malignant", "benign   ", "no_tumor "]
    for i, name in enumerate(names):
        logger.info(f"  {name}  {cm[i, 0]:>8d}  {cm[i, 1]:>8d}  {cm[i, 2]:>12d}")

    mal_total = (labels == 0).sum()
    mal_correct = cm[0, 0]
    mal_sensitivity = mal_correct / mal_total if mal_total > 0 else 0
    logger.info(f"[{phase}] Malignant Sensitivity: {mal_sensitivity:.4f} ({mal_correct}/{mal_total})")

    mal_to_notumor = cm[0, 2]
    logger.info(f"[{phase}] Malignant->NoTumor 漏诊数: {mal_to_notumor} (临床安全红线)")

    pred_notumor_total = cm[:, 2].sum()
    true_notumor_in_pred = cm[2, 2]
    npv = true_notumor_in_pred / pred_notumor_total if pred_notumor_total > 0 else 0
    logger.info(f"[{phase}] NPV (预测息肉的准确率): {npv:.4f} ({true_notumor_in_pred}/{pred_notumor_total})")

    notumor_total = (labels == 2).sum()
    notumor_correct = cm[2, 2]
    avoidable_rate = notumor_correct / notumor_total if notumor_total > 0 else 0
    logger.info(f"[{phase}] 避免不必要手术率: {avoidable_rate:.4f} ({notumor_correct}/{notumor_total})")

    # 新增: benign 识别率
    ben_total = (labels == 1).sum()
    ben_correct = cm[1, 1]
    ben_recall = ben_correct / ben_total if ben_total > 0 else 0
    logger.info(f"[{phase}] Benign 识别率: {ben_recall:.4f} ({ben_correct}/{ben_total})")

    return {
        "mal_sensitivity": mal_sensitivity,
        "mal_to_notumor": mal_to_notumor,
        "npv": npv,
        "avoidable_surgery_rate": avoidable_rate,
        "benign_recall": ben_recall,
    }


def evaluate(model, dataloader, device, class_names, logger,
             phase="Test", t1=0.33, t2=0.66):
    preds_data = collect_predictions(model, dataloader, device)
    logits = preds_data["logits"]
    scores = preds_data["scores"]
    labels = preds_data["labels"]
    seg_dice = preds_data["seg_dice"]

    # Softmax
    softmax_preds = logits.argmax(dim=1).numpy()
    acc = accuracy_score(labels, softmax_preds)
    prec = precision_score(labels, softmax_preds, average="macro", zero_division=0)
    rec = recall_score(labels, softmax_preds, average="macro", zero_division=0)
    f1 = f1_score(labels, softmax_preds, average="macro", zero_division=0)

    logger.info(f"[{phase}] Softmax — Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    report = classification_report(labels, softmax_preds,
                                   target_names=class_names, digits=4, zero_division=0)
    logger.info(f"[{phase}] Softmax Report:\n{report}")

    # Ordinal
    ord_preds = np.where(scores >= t2, 0, np.where(scores >= t1, 1, 2))
    ord_f1 = f1_score(labels, ord_preds, average="macro", zero_division=0)
    ord_acc = accuracy_score(labels, ord_preds)
    logger.info(f"[{phase}] Ordinal (t1={t1:.2f}, t2={t2:.2f}) — Acc: {ord_acc:.4f} | F1: {ord_f1:.4f}")
    ord_report = classification_report(labels, ord_preds,
                                       target_names=class_names, digits=4, zero_division=0)
    logger.info(f"[{phase}] Ordinal Report:\n{ord_report}")

    # Seg
    logger.info(f"[{phase}] Seg Dice: {seg_dice:.4f}")

    # 临床指标
    compute_clinical_metrics(scores, labels, t1, t2, logger, phase)

    # 风险分数分布
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
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260414_task3_SwinV2Tiny_segcls_4"
    log_dir = os.path.join(project_root, "0414", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 3
    meta_dim = 10
    meta_hidden = 64
    meta_dropout = 0.2
    cls_dropout = 0.4
    proj_dim = 128       # SupCon projection dimension
    pretrained = True

    batch_size = 8        # 2 mal + 3 ben + 3 notumor = 8
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
    lambda_con = 0.3      # SupCon loss weight
    focal_gamma = 2.0     # Focal loss gamma
    penalty_weight = 0.5
    label_smoothing = 0.05  # 降低 (focal 已处理过度自信)
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0
    temperature = 0.1     # SupCon temperature

    val_ratio = 0.15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["malignant", "benign", "no_tumor"]
    model_name = (
        "SwinV2-Tiny@256 + 4ch + UNetSeg + SegGuidedAttn + "
        "10D_Meta + FocalCostSensitive + OrdinalScore + SupCon"
    )
    modification = (
        "Exp-D: 针对性优化 benign recall — "
        "SupCon对比损失(拉开benign/no_tumor特征距离) + "
        "Focal代价敏感损失(聚焦难分样本) + "
        "调整代价矩阵(ben→notumor:3.0) + "
        "采样2:3:3(增加benign占比) + "
        "ordinal目标微调(ben:0.5→0.35)"
    )


# ═════════════════════════════════════════════════════════════
#  8. 数据准备
# ═════════════════════════════════════════════════════════════

def patient_level_split(df, val_ratio=0.15, seed=42):
    df = df.copy()
    df["patient_id"] = df["image_path"].apply(extract_patient_id)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(df, df["label"], groups=df["patient_id"]))
    return df.iloc[train_idx].copy(), df.iloc[val_idx].copy()


def build_dataloaders(cfg):
    full_train_df = pd.read_excel(cfg.train_excel)
    test_df = pd.read_excel(cfg.test_excel)

    train_df, val_df = patient_level_split(full_train_df, cfg.val_ratio, cfg.seed)

    train_tf = SegCls0414SyncTransform(cfg.img_size, is_train=True)
    eval_tf = SegCls0414SyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDataset0414WithMeta(
        train_df, cfg.data_root, cfg.clinical_excel, cfg.json_feature_root,
        sync_transform=train_tf, meta_stats=None,
    )
    meta_stats = train_dataset.meta_stats

    val_dataset = GBPDataset0414WithMeta(
        val_df, cfg.data_root, cfg.clinical_excel, cfg.json_feature_root,
        sync_transform=eval_tf, meta_stats=meta_stats,
    )
    test_dataset = GBPDataset0414WithMeta(
        test_df, cfg.data_root, cfg.clinical_excel, cfg.json_feature_root,
        sync_transform=eval_tf, meta_stats=meta_stats,
    )

    # Exp-D: 2:3:3 (增加 benign 每 batch 占比)
    sampler = BalancedBatchSampler(
        train_dataset.df["label"].values,
        samples_per_class={0: 2, 1: 3, 2: 3},
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

    logger.info("=" * 70)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"分类类别: {cfg.class_names}")
    logger.info(f"Metadata: {cfg.meta_dim}D → {cfg.meta_hidden}D MLP")
    logger.info(f"SupCon: proj_dim={cfg.proj_dim}, temperature={cfg.temperature}, lambda={cfg.lambda_con}")
    logger.info(f"Focal: gamma={cfg.focal_gamma}")
    logger.info(f"Batch Size: {cfg.batch_size} (2mal+3ben+3notumor)")
    logger.info(f"Backbone LR: {cfg.backbone_lr} | Head LR: {cfg.head_lr}")
    logger.info(f"Epochs: {cfg.num_epochs} | Warmup: {cfg.warmup_epochs}")
    logger.info(f"Lambda Cls: {cfg.lambda_cls} | Lambda Ord: {cfg.lambda_ord} | Lambda Con: {cfg.lambda_con}")
    logger.info(f"代价惩罚权重: {cfg.penalty_weight}")
    logger.info(f"Val 比例: {cfg.val_ratio}")
    logger.info(f"设备: {cfg.device}")
    logger.info("代价矩阵 (Exp-D 调整后):")
    logger.info(f"  malignant:  {COST_MATRIX[0].tolist()}")
    logger.info(f"  benign:     {COST_MATRIX[1].tolist()}")
    logger.info(f"  no_tumor:   {COST_MATRIX[2].tolist()}")
    logger.info(f"Ordinal 目标: {ORDINAL_TARGETS.tolist()}")
    logger.info("=" * 70)

    # ── 数据 ──
    (train_dataset, val_dataset, test_dataset,
     train_loader, val_loader, test_loader) = build_dataloaders(cfg)

    for name, ds in [("训练集", train_dataset), ("验证集", val_dataset), ("测试集", test_dataset)]:
        counts = ds.df["label"].value_counts().sort_index().to_dict()
        msg = ", ".join(f"{cfg.class_names[i]}={counts.get(i, 0)}" for i in range(3))
        logger.info(f"{name}: {len(ds)} 张 ({msg})")
    logger.info(f"训练集 metadata 缺失样本数: {train_dataset.meta_missing_count}/{len(train_dataset)}")

    # ── 模型 ──
    try:
        model = SwinV2SegGuidedOrdinalMetaSupConModel(
            num_seg_classes=cfg.num_seg_classes,
            num_cls_classes=cfg.num_cls_classes,
            meta_dim=cfg.meta_dim,
            meta_hidden=cfg.meta_hidden,
            meta_dropout=cfg.meta_dropout,
            cls_dropout=cfg.cls_dropout,
            proj_dim=cfg.proj_dim,
            pretrained=cfg.pretrained,
        ).to(cfg.device)
    except Exception as exc:
        logger.warning(f"pretrained 加载失败, 使用随机初始化: {exc}")
        model = SwinV2SegGuidedOrdinalMetaSupConModel(
            num_seg_classes=cfg.num_seg_classes,
            num_cls_classes=cfg.num_cls_classes,
            meta_dim=cfg.meta_dim,
            meta_hidden=cfg.meta_hidden,
            meta_dropout=cfg.meta_dropout,
            cls_dropout=cfg.cls_dropout,
            proj_dim=cfg.proj_dim,
            pretrained=False,
        ).to(cfg.device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,}")
    logger.info(f"可训练参数量: {n_trainable:,}")

    # ── 损失 ──
    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight], dtype=torch.float32, device=cfg.device)
    criterion = SegClsOrdinalSupConLoss(
        cost_matrix=COST_MATRIX.to(cfg.device),
        ordinal_targets=ORDINAL_TARGETS.to(cfg.device),
        lambda_cls=cfg.lambda_cls,
        lambda_ord=cfg.lambda_ord,
        lambda_con=cfg.lambda_con,
        gamma=cfg.focal_gamma,
        penalty_weight=cfg.penalty_weight,
        label_smoothing=cfg.label_smoothing,
        seg_ce_weight=seg_ce_weight,
        temperature=cfg.temperature,
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
            f"ord={train_metrics['ord_loss']:.4f} con={train_metrics['con_loss']:.4f}) "
            f"| Acc: {train_metrics['cls_acc']:.4f} "
            f"| Dice: {train_metrics['seg_dice']:.4f} "
            f"| {elapsed:.0f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 50)

            val_preds = collect_predictions(model, val_loader, cfg.device)
            t1, t2, val_ord_f1 = search_thresholds(
                val_preds["scores"], val_preds["labels"], cfg.class_names)

            val_softmax_preds = val_preds["logits"].argmax(1).numpy()
            val_softmax_f1 = f1_score(val_preds["labels"], val_softmax_preds,
                                      average="macro", zero_division=0)
            val_f1 = max(val_softmax_f1, val_ord_f1)

            # 额外打印 benign recall
            val_ben_recall = recall_score(
                val_preds["labels"], val_softmax_preds,
                labels=[1], average=None, zero_division=0)[0]

            logger.info(
                f"[Val] Softmax F1: {val_softmax_f1:.4f} | "
                f"Ordinal F1: {val_ord_f1:.4f} (t1={t1:.2f}, t2={t2:.2f}) | "
                f"Dice: {val_preds['seg_dice']:.4f} | "
                f"Benign Recall: {val_ben_recall:.4f}"
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

    # ── 最终测试 ──
    if os.path.exists(cfg.best_weight_path):
        try:
            state = torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
        except TypeError:
            state = torch.load(cfg.best_weight_path, map_location=cfg.device)
        model.load_state_dict(state)

        logger.info("\n" + "=" * 70)
        logger.info("最终测试 (最优权重)")
        logger.info("=" * 70)

        evaluate(model, test_loader, cfg.device, cfg.class_names, logger,
                 phase="Final Test", t1=best_t1, t2=best_t2)

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
