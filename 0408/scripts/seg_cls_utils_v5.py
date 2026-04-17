"""
Task 2 分割+分类联合训练工具库 v5
新增: 扩展临床特征(10D), BERT文本编码, 交叉注意力融合, 门控三模态融合

vs v2 (seg_cls_utils_v2.py):
  - 扩展临床特征 6D→10D: 新增 echo_type, lesion_count, wall_thickness, us_diameter
  - BERT中文超声报告编码 (bert-base-chinese, 冻结)
  - SwinV2SegGuidedCls4chWithText: BERT [CLS] + 后期拼接融合
  - SwinV2SegGuidedCls4chCrossAttn: BERT全token + 交叉注意力
  - SwinV2SegGuidedCls4chTrimodal: 10D临床 + BERT交叉注意力 + 门控融合
  - 文本感知的训练/评估循环
"""

import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)
import timm
from transformers import BertTokenizer, BertModel

# ── path setup ──────────────────────────────────────────────
_v5_dir = os.path.dirname(os.path.abspath(__file__))
_0402_scripts = os.path.normpath(os.path.join(_v5_dir, "../../0402/scripts"))
_0323_scripts = os.path.normpath(os.path.join(_v5_dir, "../../0323/scripts"))
for _p in [_0402_scripts, _0323_scripts]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── re-export from v2 (v2 re-exports v1) ───────────────────
from seg_cls_utils_v2 import (
    # v1
    load_annotation, generate_lesion_mask, UNetDecoderBlock, DiceLoss,
    SegClsLoss, set_seed, setup_logger, acquire_run_lock,
    build_class_weights, cosine_warmup_factor, set_epoch_lrs,
    build_optimizer_with_diff_lr, compute_seg_metrics,
    find_optimal_threshold_v2, evaluate_with_threshold_v2,
    # v2
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chModel,
    seg_cls_4ch_meta_collate_fn,
    train_one_epoch_v2,
    evaluate_v2,
    run_seg_cls_experiment_v2,
)

# ── from test_yqh ──────────────────────────────────────────
from test_yqh import (
    adapt_model_to_4ch,
    normalize_case_id,
    extract_case_id_from_image_path,
    fit_meta_stats,
    encode_meta_row,
    META_FEATURE_NAMES,
)


# ═══════════════════════════════════════════════════════════
#  Extended Clinical Feature Names (10D)
# ═══════════════════════════════════════════════════════════

EXT_CLINICAL_FEATURE_NAMES = list(META_FEATURE_NAMES) + [
    "echo_type",        # 回声 0/1/2 (Excel, 覆盖90.7%)
    "lesion_count",     # 个数 0/1   (Excel, 覆盖86.3%)
    "wall_thickness",   # 壁厚mm    (Excel, 覆盖55.2%)
    "us_diameter",      # 超声直径mm (Excel, 覆盖97.9%)
]


# ═══════════════════════════════════════════════════════════
#  Data Loading: Extended Clinical + Text
# ═══════════════════════════════════════════════════════════


def _parse_numeric(value):
    """Parse numeric value, return NaN on failure."""
    if value is None or pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        s = str(value).strip()
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        return float(m.group(1)) if m else np.nan


def _find_column(columns, keywords):
    """Find a column whose name contains all keywords."""
    cols = [str(c) for c in columns]
    for idx, col in enumerate(cols):
        if all(k in col for k in keywords):
            return columns[idx]
    return None


def load_ext_clinical_meta_table(clinical_excel_path):
    """
    Load age/gender + 4 new features from clinical Excel.
    Returns DataFrame with columns:
      case_id_norm, age, gender, echo_type, lesion_count, wall_thickness, us_diameter
    """
    df = pd.read_excel(clinical_excel_path)

    id_col = _find_column(df.columns, ["住", "院", "号"])
    age_col = _find_column(df.columns, ["年", "龄"])
    gender_col = _find_column(df.columns, ["性", "别"])
    echo_col = _find_column(df.columns, ["回声"])
    count_col = _find_column(df.columns, ["个数"])
    wall_col = _find_column(df.columns, ["壁厚"])
    us_diam_col = _find_column(df.columns, ["超声直径"])

    if id_col is None:
        raise KeyError(f"Cannot find 住院号 column in {clinical_excel_path}")

    from test_yqh import parse_age_value, parse_gender_value

    out = pd.DataFrame({
        "case_id_norm": df[id_col].map(normalize_case_id),
        "age": df[age_col].map(parse_age_value) if age_col else np.nan,
        "gender": df[gender_col].map(parse_gender_value) if gender_col else np.nan,
        "echo_type": df[echo_col].map(_parse_numeric) if echo_col else np.nan,
        "lesion_count": df[count_col].map(_parse_numeric) if count_col else np.nan,
        "wall_thickness": df[wall_col].map(_parse_numeric) if wall_col else np.nan,
        "us_diameter": df[us_diam_col].map(_parse_numeric) if us_diam_col else np.nan,
    })
    out = out.dropna(subset=["case_id_norm"]).drop_duplicates(
        subset=["case_id_norm"], keep="first"
    )
    return out


def load_json_meta_table(json_feature_root):
    """Load size_mm/size_bin/flow_bin/morph_bin from JSON feat dicts."""
    rows = []
    for fp in Path(json_feature_root).glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        feat = data.get("feat", {}) if isinstance(data, dict) else {}
        if not isinstance(feat, dict):
            feat = {}
        rows.append({
            "case_id_norm": normalize_case_id(fp.stem),
            "size_mm": _parse_numeric(feat.get("size_mm")),
            "size_bin": _parse_numeric(feat.get("size_bin")),
            "flow_bin": _parse_numeric(feat.get("flow_bin")),
            "morph_bin": _parse_numeric(feat.get("morph_bin")),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(
            columns=["case_id_norm", "size_mm", "size_bin", "flow_bin", "morph_bin"]
        )
    return out.dropna(subset=["case_id_norm"]).drop_duplicates(
        subset=["case_id_norm"], keep="first"
    )


def build_ext_case_meta_table(clinical_excel_path, json_feature_root):
    """Merge extended clinical + json metadata."""
    clinical = load_ext_clinical_meta_table(clinical_excel_path)
    json_meta = load_json_meta_table(json_feature_root)
    out = clinical.merge(json_meta, on="case_id_norm", how="outer")
    return out.drop_duplicates(subset=["case_id_norm"], keep="first")


def load_text_bert_dict(json_feature_root):
    """Load text_bert from all JSON files. Returns {case_id_norm: text_string}."""
    text_dict = {}
    for fp in Path(json_feature_root).glob("*.json"):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        case_id = normalize_case_id(fp.stem)
        text = data.get("text_bert", "")
        if case_id and isinstance(text, str):
            text_dict[case_id] = text
    return text_dict


# ═══════════════════════════════════════════════════════════
#  Datasets
# ═══════════════════════════════════════════════════════════


class GBPDatasetSegCls4chWithExtMeta(Dataset):
    """4ch input + seg target + cls label + 10D extended metadata. (Exp#15)"""

    def __init__(self, excel_path, data_root, clinical_excel_path, json_feature_root,
                 sync_transform=None, meta_stats=None):
        self.df = pd.read_excel(excel_path).copy()
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
                s["label"] != "gallbladder" and s["shape_type"] == "polygon"
                and len(s["points"]) >= 3
                for s in shapes
            )

        mask = generate_lesion_mask(shapes, img_w, img_h)

        if self.sync_transform:
            input_4ch, seg_target = self.sync_transform(img, mask)
        else:
            import torchvision.transforms.functional as TF
            img_t = TF.to_tensor(img)
            mask_t = TF.to_tensor(mask)
            input_4ch = torch.cat([img_t, mask_t], dim=0)
            seg_target = (mask_t.squeeze(0) > 0.5).long()

        meta_tensor = encode_meta_row(row, self.meta_stats, self.meta_feature_names)
        return input_4ch, seg_target, meta_tensor, label, has_mask


class GBPDatasetSegCls4chWithTextMeta(Dataset):
    """4ch input + seg target + metadata + text tokens. (Exp#16/17/18)

    Returns: (input_4ch, seg_target, meta_tensor, input_ids, attention_mask, label, has_mask)
    """

    def __init__(self, excel_path, data_root, clinical_excel_path, json_feature_root,
                 sync_transform=None, meta_stats=None,
                 meta_feature_names=None, text_dict=None,
                 tokenizer=None, max_text_len=128):
        self.df = pd.read_excel(excel_path).copy()
        self.data_root = data_root
        self.sync_transform = sync_transform
        self.max_text_len = max_text_len

        # Meta features (configurable: 6D or 10D)
        if meta_feature_names is None:
            meta_feature_names = list(META_FEATURE_NAMES)
        self.meta_feature_names = list(meta_feature_names)

        # Decide which meta table to build
        if set(self.meta_feature_names) - set(META_FEATURE_NAMES):
            # Has extended features → use extended loader
            meta_df = build_ext_case_meta_table(clinical_excel_path, json_feature_root)
        else:
            from test_yqh import build_case_meta_table
            meta_df = build_case_meta_table(clinical_excel_path, json_feature_root)

        self.df["case_id_norm"] = self.df["image_path"].map(extract_case_id_from_image_path)
        self.df = self.df.merge(meta_df, on="case_id_norm", how="left")

        for col in self.meta_feature_names:
            if col not in self.df.columns:
                self.df[col] = np.nan

        self.meta_stats = (
            meta_stats if meta_stats is not None
            else fit_meta_stats(self.df, self.meta_feature_names)
        )
        self.meta_dim = len(self.meta_feature_names)

        # Text
        if text_dict is None:
            text_dict = load_text_bert_dict(json_feature_root)
        self.text_dict = text_dict

        if tokenizer is None:
            tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.tokenizer = tokenizer

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
                s["label"] != "gallbladder" and s["shape_type"] == "polygon"
                and len(s["points"]) >= 3
                for s in shapes
            )

        mask = generate_lesion_mask(shapes, img_w, img_h)

        if self.sync_transform:
            input_4ch, seg_target = self.sync_transform(img, mask)
        else:
            import torchvision.transforms.functional as TF
            img_t = TF.to_tensor(img)
            mask_t = TF.to_tensor(mask)
            input_4ch = torch.cat([img_t, mask_t], dim=0)
            seg_target = (mask_t.squeeze(0) > 0.5).long()

        meta_tensor = encode_meta_row(row, self.meta_stats, self.meta_feature_names)

        # Text tokenization
        case_id = row.get("case_id_norm", "")
        text = self.text_dict.get(case_id, "")
        if not isinstance(text, str) or not text.strip():
            text = ""
        text_enc = self.tokenizer(
            text, max_length=self.max_text_len, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        input_ids = text_enc["input_ids"].squeeze(0)           # (max_text_len,)
        attention_mask = text_enc["attention_mask"].squeeze(0)  # (max_text_len,)

        return input_4ch, seg_target, meta_tensor, input_ids, attention_mask, label, has_mask


def seg_cls_text_collate_fn(batch):
    """Collate for 4ch + meta + text dataset (7-element tuples)."""
    inputs, masks, metas, input_ids, attn_masks, labels, has_masks = zip(*batch)
    return (
        torch.stack(inputs),
        torch.stack(masks),
        torch.stack(metas),
        torch.stack(input_ids),
        torch.stack(attn_masks),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(has_masks, dtype=torch.bool),
    )


# ═══════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════


class SwinV2SegGuidedCls4chWithText(nn.Module):
    """Exp#16: SwinV2 + Seg-Guided Attention + BERT[CLS] Late Fusion.

    Architecture:
      img_feat (256D) || text_feat (128D) || meta_feat (64D) → 448D → MLP → 2
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=2, meta_dim=6,
                 meta_hidden=64, meta_dropout=0.2, cls_dropout=0.4,
                 text_proj_dim=128, text_dropout=0.3,
                 bert_name="bert-base-chinese", pretrained=True):
        super().__init__()
        self.meta_dim = meta_dim

        # ── Image encoder + seg decoder (same as Exp#4) ──
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

        # ── Seg-guided cls pooling ──
        self.cls_proj = nn.Sequential(
            nn.Conv2d(fc[2], 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
        )

        # ── Text encoder (BERT frozen) ──
        self.text_encoder = BertModel.from_pretrained(bert_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        self.text_proj = nn.Sequential(
            nn.Linear(768, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.GELU(),
            nn.Dropout(text_dropout),
        )

        # ── Metadata encoder ──
        self.meta_encoder = None
        meta_out = 0
        if meta_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(meta_dim, meta_hidden), nn.LayerNorm(meta_hidden),
                nn.GELU(), nn.Dropout(meta_dropout),
                nn.Linear(meta_hidden, meta_hidden), nn.GELU(), nn.Dropout(meta_dropout),
            )
            meta_out = meta_hidden

        # ── Fusion classifier: 256 + 128 + 64 = 448 ──
        fusion_in = 256 + text_proj_dim + meta_out
        self.cls_mlp = nn.Sequential(
            nn.Linear(fusion_in, 128), nn.GELU(), nn.Dropout(cls_dropout),
            nn.Linear(128, num_cls_classes),
        )

    def _to_bchw(self, x):
        if x.ndim == 4 and x.shape[1] != x.shape[3]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, metadata=None, input_ids=None, attention_mask=None):
        # ── Image path ──
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
        img_feat = (f2_proj * attn).sum(dim=(2, 3))  # (B, 256)

        # ── Text path ──
        with torch.no_grad():
            text_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        text_cls = text_out.last_hidden_state[:, 0]  # [CLS] → (B, 768)
        text_feat = self.text_proj(text_cls)          # (B, 128)

        # ── Meta path ──
        parts = [img_feat, text_feat]
        if self.meta_encoder is not None and metadata is not None:
            meta_feat = self.meta_encoder(metadata.float())
            parts.append(meta_feat)

        fused = torch.cat(parts, dim=1)
        cls_logits = self.cls_mlp(fused)
        return seg_logits, cls_logits


class TextImageCrossAttention(nn.Module):
    """Cross-attention: image spatial features attend to text tokens.

    Q = img (B, H*W, hidden)
    K, V = text (B, seq_len, hidden)
    """

    def __init__(self, img_dim=256, text_dim=768, hidden_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, img_dim)

    def forward(self, img_feat_2d, text_hidden_states, text_attention_mask):
        """
        Args:
            img_feat_2d: (B, C, H, W)
            text_hidden_states: (B, seq_len, 768)
            text_attention_mask: (B, seq_len) with 1=valid, 0=pad
        Returns:
            enhanced_img: (B, C, H, W)
        """
        B, C, H, W = img_feat_2d.shape
        img_tokens = img_feat_2d.flatten(2).permute(0, 2, 1)  # (B, H*W, C)

        Q = self.img_proj(img_tokens)               # (B, H*W, hidden)
        K = self.text_proj(text_hidden_states)       # (B, seq_len, hidden)
        V = K

        key_padding_mask = (text_attention_mask == 0)  # True = padding position
        attn_out, _ = self.cross_attn(Q, K, V, key_padding_mask=key_padding_mask)
        attn_out = self.norm(attn_out + Q)  # residual connection

        out = self.out_proj(attn_out)  # (B, H*W, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return img_feat_2d + out  # residual: original + text-enhanced


class SwinV2SegGuidedCls4chCrossAttn(nn.Module):
    """Exp#17: SwinV2 + BERT Cross-Attention on f2 + Seg-Guided Attention.

    text → BERT(frozen) → hidden_states
    f2(384→256) → CrossAttn(img=Q, text=K,V) → enhanced_f2
    enhanced_f2 → Seg-Guided Attention Pool → 256D
    cat[256D, meta(64D)] → 320D → MLP → 2
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=2, meta_dim=6,
                 meta_hidden=64, meta_dropout=0.2, cls_dropout=0.4,
                 ca_hidden=128, ca_heads=4, ca_dropout=0.1,
                 bert_name="bert-base-chinese", pretrained=True):
        super().__init__()
        self.meta_dim = meta_dim

        # ── Image encoder + seg decoder ──
        self.encoder = timm.create_model(
            "swinv2_tiny_window8_256", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
        )
        adapt_model_to_4ch(self.encoder)
        fc = [info["num_chs"] for info in self.encoder.feature_info]

        self.dec3 = UNetDecoderBlock(fc[3], fc[2], fc[2])
        self.dec2 = UNetDecoderBlock(fc[2], fc[1], fc[1])
        self.dec1 = UNetDecoderBlock(fc[1], fc[0], fc[0])
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(fc[0], 48, kernel_size=4, stride=4),
            nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, num_seg_classes, 1),
        )

        # ── Cls projection ──
        self.cls_proj = nn.Sequential(
            nn.Conv2d(fc[2], 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
        )

        # ── BERT (frozen) ──
        self.text_encoder = BertModel.from_pretrained(bert_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # ── Cross-Attention ──
        self.cross_attn = TextImageCrossAttention(
            img_dim=256, text_dim=768,
            hidden_dim=ca_hidden, num_heads=ca_heads, dropout=ca_dropout,
        )

        # ── Metadata encoder ──
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

    def forward(self, x, metadata=None, input_ids=None, attention_mask=None):
        # ── Image encoder ──
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        # ── Seg decoder ──
        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        # ── Cls projection ──
        f2_proj = self.cls_proj(f2)  # (B, 256, 16, 16)

        # ── Text encoding ──
        with torch.no_grad():
            text_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        text_hidden = text_out.last_hidden_state  # (B, seq_len, 768)

        # ── Cross-Attention: text enhances image features ──
        f2_enhanced = self.cross_attn(f2_proj, text_hidden, attention_mask)

        # ── Seg-guided attention on enhanced features ──
        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]
        attn = F.interpolate(seg_prob, size=f2_enhanced.shape[2:],
                             mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)

        cls_feat = (f2_enhanced * attn).sum(dim=(2, 3))  # (B, 256)

        # ── Meta fusion ──
        if self.meta_encoder is not None and metadata is not None:
            meta_feat = self.meta_encoder(metadata.float())
            cls_feat = torch.cat([cls_feat, meta_feat], dim=1)

        cls_logits = self.cls_mlp(cls_feat)
        return seg_logits, cls_logits


class GatedTrimodalFusion(nn.Module):
    """Gated fusion of image / text / clinical features.

    Learns per-sample adaptive weights for each modality.
    """

    def __init__(self, img_dim=256, text_dim=128, meta_dim=96, fusion_dim=256):
        super().__init__()
        total_dim = img_dim + text_dim + meta_dim

        self.gate = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_dim, 3),
            nn.Sigmoid(),
        )

        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.text_proj = nn.Linear(text_dim, fusion_dim)
        self.meta_proj = nn.Linear(meta_dim, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, img_feat, text_feat, meta_feat):
        concat = torch.cat([img_feat, text_feat, meta_feat], dim=1)
        gates = self.gate(concat)  # (B, 3)
        g_img = gates[:, 0:1]
        g_text = gates[:, 1:2]
        g_meta = gates[:, 2:3]

        fused = (g_img * self.img_proj(img_feat)
                 + g_text * self.text_proj(text_feat)
                 + g_meta * self.meta_proj(meta_feat))
        return self.norm(fused)  # (B, fusion_dim)


class SwinV2SegGuidedCls4chTrimodal(nn.Module):
    """Exp#18: Full trimodal fusion.

    - Image: SwinV2 + CrossAttn(text) + Seg-Guided Attention → 256D
    - Text: BERT [CLS] → proj → 128D
    - Clinical: 10D extended meta → MLP → 96D
    - GatedTrimodalFusion → 256D → MLP → 2
    """

    def __init__(self, num_seg_classes=2, num_cls_classes=2, meta_dim=10,
                 meta_hidden=96, meta_dropout=0.2, cls_dropout=0.4,
                 text_proj_dim=128, text_dropout=0.3,
                 ca_hidden=128, ca_heads=4, ca_dropout=0.1,
                 fusion_dim=256,
                 bert_name="bert-base-chinese", pretrained=True):
        super().__init__()
        self.meta_dim = meta_dim

        # ── Image encoder + seg decoder ──
        self.encoder = timm.create_model(
            "swinv2_tiny_window8_256", pretrained=pretrained,
            features_only=True, out_indices=(0, 1, 2, 3),
        )
        adapt_model_to_4ch(self.encoder)
        fc = [info["num_chs"] for info in self.encoder.feature_info]

        self.dec3 = UNetDecoderBlock(fc[3], fc[2], fc[2])
        self.dec2 = UNetDecoderBlock(fc[2], fc[1], fc[1])
        self.dec1 = UNetDecoderBlock(fc[1], fc[0], fc[0])
        self.seg_final = nn.Sequential(
            nn.ConvTranspose2d(fc[0], 48, kernel_size=4, stride=4),
            nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, 48, 3, padding=1), nn.GroupNorm(8, 48), nn.GELU(),
            nn.Conv2d(48, num_seg_classes, 1),
        )

        self.cls_proj = nn.Sequential(
            nn.Conv2d(fc[2], 256, 1), nn.GroupNorm(8, 256), nn.GELU(),
        )

        # ── BERT (frozen) ──
        self.text_encoder = BertModel.from_pretrained(bert_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # ── Cross-Attention (text → image) ──
        self.cross_attn = TextImageCrossAttention(
            img_dim=256, text_dim=768,
            hidden_dim=ca_hidden, num_heads=ca_heads, dropout=ca_dropout,
        )

        # ── Text [CLS] projection ──
        self.text_cls_proj = nn.Sequential(
            nn.Linear(768, text_proj_dim),
            nn.LayerNorm(text_proj_dim),
            nn.GELU(),
            nn.Dropout(text_dropout),
        )

        # ── Meta encoder (10D → 96D) ──
        self.meta_encoder = nn.Sequential(
            nn.Linear(meta_dim, meta_hidden), nn.LayerNorm(meta_hidden),
            nn.GELU(), nn.Dropout(meta_dropout),
            nn.Linear(meta_hidden, meta_hidden), nn.GELU(), nn.Dropout(meta_dropout),
        )

        # ── Gated Trimodal Fusion ──
        self.fusion = GatedTrimodalFusion(
            img_dim=256, text_dim=text_proj_dim, meta_dim=meta_hidden,
            fusion_dim=fusion_dim,
        )

        # ── Classifier ──
        self.cls_mlp = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(), nn.Dropout(cls_dropout),
            nn.Linear(128, num_cls_classes),
        )

    def _to_bchw(self, x):
        if x.ndim == 4 and x.shape[1] != x.shape[3]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x

    def forward(self, x, metadata=None, input_ids=None, attention_mask=None):
        # ── Image encoder ──
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        # ── Seg decoder ──
        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        # ── Cls projection ──
        f2_proj = self.cls_proj(f2)  # (B, 256, 16, 16)

        # ── BERT encoding ──
        with torch.no_grad():
            text_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        text_hidden = text_out.last_hidden_state  # (B, seq_len, 768)
        text_cls = text_hidden[:, 0]              # (B, 768)

        # ── Cross-attention: text enhances image spatial features ──
        f2_enhanced = self.cross_attn(f2_proj, text_hidden, attention_mask)

        # ── Seg-guided attention pooling ──
        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]
        attn = F.interpolate(seg_prob, size=f2_enhanced.shape[2:],
                             mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)
        img_feat = (f2_enhanced * attn).sum(dim=(2, 3))  # (B, 256)

        # ── Text [CLS] projection ──
        text_feat = self.text_cls_proj(text_cls)  # (B, 128)

        # ── Meta encoding ──
        meta_feat = self.meta_encoder(metadata.float())  # (B, 96)

        # ── Gated fusion ──
        fused = self.fusion(img_feat, text_feat, meta_feat)  # (B, 256)

        cls_logits = self.cls_mlp(fused)
        return seg_logits, cls_logits


# ═══════════════════════════════════════════════════════════
#  Text-aware Training / Evaluation
# ═══════════════════════════════════════════════════════════


def _unpack_text_batch(batch):
    """Unpack 7-element batch: (imgs, masks, metas, input_ids, attn_mask, labels, has_masks)."""
    imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch
    return imgs, masks, metas, input_ids, attn_mask, labels, has_masks


def _labels_to_numpy(labels):
    """Convert labels from tensor/list to numpy int array."""
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy().astype(np.int64, copy=False)
    return np.asarray(labels, dtype=np.int64)


def train_one_epoch_text(model, dataloader, criterion, optimizer, device, scaler,
                         use_amp, grad_clip=None, num_seg_classes=2):
    """Training loop for text-aware models (Exp#16/17/18)."""
    model.train()
    running_loss, running_seg_loss, running_cls_loss = 0.0, 0.0, 0.0
    cls_correct, cls_total = 0, 0
    all_seg_ious, all_seg_dices = [], []

    for batch in dataloader:
        imgs, masks, metas, input_ids, attn_mask, labels, has_masks = _unpack_text_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                enabled=use_amp):
            seg_logits, cls_logits = model(
                imgs, metadata=metas, input_ids=input_ids, attention_mask=attn_mask
            )
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


def evaluate_text(model, dataloader, device, class_names, logger, phase="Test",
                  num_seg_classes=2):
    """Evaluation for text-aware models."""
    model.eval()
    all_preds, all_labels = [], []
    all_seg_ious, all_seg_dices = [], []

    with torch.no_grad():
        for batch in dataloader:
            imgs, masks, metas, input_ids, attn_mask, labels, has_masks = _unpack_text_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            has_masks = has_masks.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            seg_logits, cls_logits = model(
                imgs, metadata=metas, input_ids=input_ids, attention_mask=attn_mask
            )
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


def predict_probs_text(model, dataloader, device):
    """Return (prob_benign, prob_no_tumor, labels) for text-aware models."""
    model.eval()
    all_prob_benign, all_prob_no_tumor, all_labels = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            imgs, _, metas, input_ids, attn_mask, labels, _ = _unpack_text_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device, non_blocking=True)

            _, cls_logits = model(
                imgs, metadata=metas, input_ids=input_ids, attention_mask=attn_mask
            )
            probs = torch.softmax(cls_logits, dim=1).cpu().numpy()
            all_prob_benign.extend(probs[:, 0])
            all_prob_no_tumor.extend(probs[:, 1])
            all_labels.extend(_labels_to_numpy(labels))

    return (
        np.asarray(all_prob_benign, dtype=np.float32),
        np.asarray(all_prob_no_tumor, dtype=np.float32),
        np.asarray(all_labels, dtype=np.int64),
    )


def find_optimal_threshold_text(model, dataloader, device):
    """Find optimal threshold for text-aware models."""
    all_probs, _, all_labels = predict_probs_text(model, dataloader, device)

    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds = np.where(all_probs >= thresh, 0, 1)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def evaluate_with_threshold_text(model, dataloader, device, class_names, logger,
                                 threshold=0.5, phase="Test"):
    """Evaluate with custom threshold for text-aware models."""
    all_probs, _, all_labels = predict_probs_text(model, dataloader, device)
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


def _compute_threshold_metrics(probs_benign, labels, threshold):
    """Compute class-wise metrics for threshold over P(benign)."""
    preds = np.where(probs_benign >= threshold, 0, 1)

    n_all = len(labels)
    n_benign = int(np.sum(labels == 0))
    n_no_tumor = int(np.sum(labels == 1))

    tp_benign = int(np.sum((labels == 0) & (preds == 0)))
    fn_benign = int(np.sum((labels == 0) & (preds == 1)))
    tp_no_tumor = int(np.sum((labels == 1) & (preds == 1)))
    pred_no_tumor = int(np.sum(preds == 1))

    benign_recall = tp_benign / max(n_benign, 1)
    benign_miss_rate = fn_benign / max(n_benign, 1)
    no_tumor_recall = tp_no_tumor / max(n_no_tumor, 1)
    no_tumor_precision = tp_no_tumor / max(pred_no_tumor, 1)
    selected_no_tumor_rate = pred_no_tumor / max(n_all, 1)
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return {
        "threshold": float(threshold),
        "benign_recall": float(benign_recall),
        "benign_miss_rate": float(benign_miss_rate),
        "no_tumor_recall": float(no_tumor_recall),
        "no_tumor_precision": float(no_tumor_precision),
        "selected_no_tumor": pred_no_tumor,
        "selected_no_tumor_rate": float(selected_no_tumor_rate),
        "macro_f1": float(macro_f1),
    }


def find_constrained_threshold_text(
    probs_benign, labels, max_benign_miss_rate=0.10,
    threshold_start=0.01, threshold_end=0.99, threshold_step=0.001,
):
    """Find threshold under benign miss-rate constraint, then maximize no_tumor recall."""
    thresholds = np.arange(threshold_start, threshold_end + 1e-9, threshold_step)
    all_metrics = [
        _compute_threshold_metrics(probs_benign, labels, th) for th in thresholds
    ]

    feasible = [
        m for m in all_metrics
        if m["benign_miss_rate"] <= max_benign_miss_rate + 1e-12
    ]

    if feasible:
        best = max(
            feasible,
            key=lambda m: (
                m["no_tumor_recall"],
                m["selected_no_tumor_rate"],
                m["no_tumor_precision"],
                m["macro_f1"],
                m["threshold"],
            ),
        )
        best["constraint_satisfied"] = True
        best["max_benign_miss_rate"] = float(max_benign_miss_rate)
        return best

    # Fallback: if no threshold can satisfy <=10%漏检, choose minimum漏检率阈值
    best = min(
        all_metrics,
        key=lambda m: (
            m["benign_miss_rate"],
            -m["no_tumor_recall"],
            -m["selected_no_tumor_rate"],
            -m["macro_f1"],
        ),
    )
    best["constraint_satisfied"] = False
    best["max_benign_miss_rate"] = float(max_benign_miss_rate)
    return best


def analyze_high_confidence_positive(
    labels, probs_positive, positive_label=1, min_confidence=0.90,
):
    """Analyze high-confidence positive predictions, e.g., P(no_tumor)>=0.9."""
    mask = probs_positive >= min_confidence
    selected = int(np.sum(mask))
    total = int(len(labels))
    positives_total = int(np.sum(labels == positive_label))

    if selected == 0:
        return {
            "selected": 0,
            "total": total,
            "coverage": 0.0,
            "precision": np.nan,
            "true_positive": 0,
            "positive_recall_in_selected": 0.0,
            "min_confidence": float(min_confidence),
        }

    true_positive = int(np.sum(labels[mask] == positive_label))
    precision = true_positive / selected
    recall_in_selected = true_positive / max(positives_total, 1)
    return {
        "selected": selected,
        "total": total,
        "coverage": selected / max(total, 1),
        "precision": float(precision),
        "true_positive": true_positive,
        "positive_recall_in_selected": float(recall_in_selected),
        "min_confidence": float(min_confidence),
    }


def compute_binary_reliability_stats(y_true_binary, y_prob, n_bins=10):
    """Compute reliability bins + ECE/Brier for a binary class probability."""
    y_true_binary = np.asarray(y_true_binary, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)

    bins = np.linspace(0.0, 1.0, int(n_bins) + 1, dtype=np.float32)
    bin_ids = np.digitize(y_prob, bins[1:-1], right=False)

    rows = []
    ece = 0.0
    n_all = max(len(y_true_binary), 1)

    for i in range(len(bins) - 1):
        mask = (bin_ids == i)
        count = int(np.sum(mask))
        if count == 0:
            continue

        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true_binary[mask]))
        gap = abs(acc - conf)
        ece += gap * (count / n_all)

        rows.append({
            "bin_idx": i,
            "bin_lower": float(bins[i]),
            "bin_upper": float(bins[i + 1]),
            "count": count,
            "confidence": conf,
            "accuracy": acc,
            "abs_gap": float(gap),
        })

    brier = float(np.mean((y_prob - y_true_binary) ** 2))
    return rows, float(ece), brier


def save_reliability_stats_csv(rows, out_csv):
    """Save reliability bin stats to CSV."""
    cols = [
        "bin_idx", "bin_lower", "bin_upper", "count",
        "confidence", "accuracy", "abs_gap",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(out_csv, index=False, encoding="utf-8")


def save_reliability_diagram(rows, out_png, title):
    """Save reliability diagram (confidence vs accuracy) + count histogram."""
    if not rows:
        return False

    try:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Prefer common CJK fonts so Chinese labels render correctly.
        matplotlib.rcParams["font.sans-serif"] = [
            "Noto Sans CJK SC",
            "SimHei",
            "WenQuanYi Zen Hei",
            "Microsoft YaHei",
            "Arial Unicode MS",
            "DejaVu Sans",
        ]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        return False

    conf = np.array([r["confidence"] for r in rows], dtype=np.float32)
    acc = np.array([r["accuracy"] for r in rows], dtype=np.float32)
    cnt = np.array([r["count"] for r in rows], dtype=np.int32)
    bin_l = np.array([r["bin_lower"] for r in rows], dtype=np.float32)
    bin_u = np.array([r["bin_upper"] for r in rows], dtype=np.float32)
    centers = (bin_l + bin_u) / 2.0
    widths = np.maximum(bin_u - bin_l, 1e-6)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(6.2, 7.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )

    ax1.plot([0, 1], [0, 1], "--", color="#9e9e9e", linewidth=1.0, label="理想校准")
    ax1.plot(conf, acc, marker="o", color="#1f77b4", linewidth=1.6, label="模型")
    ax1.scatter(conf, acc, s=20 + 1.6 * cnt, color="#1f77b4", alpha=0.75)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("实际 no_tumor 比例")
    ax1.set_title(title)
    ax1.grid(alpha=0.25)
    ax1.legend(loc="upper left", frameon=False)

    ax2.bar(centers, cnt, width=widths * 0.95, color="#8bc0ff", edgecolor="#2f5d9f")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("预测概率 P(no_tumor)")
    ax2.set_ylabel("样本数")
    ax2.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


# ═══════════════════════════════════════════════════════════
#  Experiment Runner (text-aware)
# ═══════════════════════════════════════════════════════════


def run_seg_cls_experiment_text(cfg, build_model_fn, build_dataloaders_fn,
                                build_optimizer_fn, script_path):
    """Experiment runner for text-aware models (Exp#16/17/18)."""
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
    logger.info(f"Meta Dim: {getattr(cfg, 'meta_dim', 'N/A')}")
    logger.info(f"Benign漏检率上限(临床约束): {getattr(cfg, 'max_benign_miss_rate', 0.10):.2%}")
    logger.info(f"高置信no_tumor阈值: {getattr(cfg, 'high_confidence_no_tumor_prob', 0.90):.2f}")
    logger.info(f"校准分箱数: {int(getattr(cfg, 'calibration_bins', 10))}")
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
    logger.info(f"模型参数量: {n_params:,} (含BERT)")
    logger.info(f"可训练参数量: {n_trainable:,} (BERT冻结)")

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

        train_metrics = train_one_epoch_text(
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
            acc, prec, rec, f1, seg_iou, seg_dice = evaluate_text(
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
    evaluate_text(model, test_loader, cfg.device, cfg.class_names, logger,
                  phase="Final Test", num_seg_classes=cfg.num_seg_classes)

    logger.info("\n" + "=" * 70)
    logger.info("阈值优化搜索")
    logger.info("=" * 70)
    best_thresh, best_thresh_f1 = find_optimal_threshold_text(model, test_loader, cfg.device)
    logger.info(
        f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认0.5 F1: {best_f1:.4f})"
    )
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold_text(
            model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=best_thresh, phase="Final Test (最优阈值)",
        )

    # ── Clinical policy: benign miss-rate <= target, maximize no_tumor recall ──
    probs_benign, probs_no_tumor, labels_np = predict_probs_text(model, test_loader, cfg.device)
    max_benign_miss_rate = float(getattr(cfg, "max_benign_miss_rate", 0.10))
    high_conf_no_tumor_prob = float(getattr(cfg, "high_confidence_no_tumor_prob", 0.90))
    calibration_bins = int(getattr(cfg, "calibration_bins", 10))

    logger.info("\n" + "=" * 70)
    logger.info("临床约束阈值搜索 (benign漏检≤目标, 尽量筛出no_tumor)")
    logger.info("=" * 70)
    constrained = find_constrained_threshold_text(
        probs_benign,
        labels_np,
        max_benign_miss_rate=max_benign_miss_rate,
    )
    if constrained["constraint_satisfied"]:
        logger.info(
            f"约束满足: threshold={constrained['threshold']:.3f} | "
            f"benign漏检率={constrained['benign_miss_rate']:.2%} | "
            f"no_tumor召回={constrained['no_tumor_recall']:.2%} | "
            f"no_tumor精度={constrained['no_tumor_precision']:.2%} | "
            f"no_tumor筛出率={constrained['selected_no_tumor_rate']:.2%} | "
            f"F1(macro)={constrained['macro_f1']:.4f}"
        )
        evaluate_with_threshold_text(
            model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=constrained["threshold"],
            phase=f"Final Test (临床约束阈值, benign漏检<={max_benign_miss_rate:.0%})",
        )
    else:
        logger.warning(
            f"未找到满足benign漏检<={max_benign_miss_rate:.0%}的阈值。"
            f"已退化为最小漏检阈值={constrained['threshold']:.3f}, "
            f"benign漏检率={constrained['benign_miss_rate']:.2%}"
        )
        evaluate_with_threshold_text(
            model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=constrained["threshold"],
            phase="Final Test (最小benign漏检阈值)",
        )

    # Save per-case probabilities for external analysis
    prob_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_test_probs.csv")
    pd.DataFrame({
        "label": labels_np.astype(int),
        "prob_benign": probs_benign,
        "prob_no_tumor": probs_no_tumor,
    }).to_csv(prob_csv, index=False, encoding="utf-8")
    logger.info(f"测试集概率已保存: {prob_csv}")

    # ── Reliability / calibration for no_tumor confidence ──
    logger.info("\n" + "=" * 70)
    logger.info("no_tumor 置信度-准确率校准 (Reliability Diagram)")
    logger.info("=" * 70)

    high_conf = analyze_high_confidence_positive(
        labels_np,
        probs_no_tumor,
        positive_label=1,
        min_confidence=high_conf_no_tumor_prob,
    )
    if high_conf["selected"] > 0:
        logger.info(
            f"P(no_tumor)>={high_conf_no_tumor_prob:.2f}: "
            f"{high_conf['selected']}/{high_conf['total']} ({high_conf['coverage']:.2%}) | "
            f"真实no_tumor比例={high_conf['precision']:.2%} "
            f"(命中{high_conf['true_positive']}例)"
        )
    else:
        logger.info(
            f"P(no_tumor)>={high_conf_no_tumor_prob:.2f}: 无样本落入高置信区间"
        )

    y_true_no_tumor = (labels_np == 1).astype(np.int64)
    rel_rows, ece, brier = compute_binary_reliability_stats(
        y_true_no_tumor, probs_no_tumor, n_bins=calibration_bins
    )
    logger.info(
        f"校准统计: ECE={ece:.4f} | Brier={brier:.4f} | 有效分箱={len(rel_rows)}/{calibration_bins}"
    )
    for row in rel_rows:
        logger.info(
            f"  Bin[{row['bin_lower']:.2f},{row['bin_upper']:.2f}) "
            f"n={row['count']:3d} | conf={row['confidence']:.3f} | acc={row['accuracy']:.3f}"
        )

    rel_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_no_tumor_reliability.csv")
    save_reliability_stats_csv(rel_rows, rel_csv)
    logger.info(f"校准分箱明细已保存: {rel_csv}")

    rel_png = os.path.join(cfg.log_dir, f"{cfg.exp_name}_no_tumor_reliability.png")
    fig_ok = save_reliability_diagram(
        rel_rows,
        rel_png,
        title=f"{cfg.exp_name} no_tumor 置信度-准确率校准图",
    )
    if fig_ok:
        logger.info(f"校准曲线已保存: {rel_png}")
    else:
        logger.warning("校准曲线生成失败（可能缺少matplotlib环境），已保留CSV分箱结果。")

    dst = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst):
        shutil.copy2(script_path, dst)
        logger.info(f"训练脚本已复制到: {dst}")
