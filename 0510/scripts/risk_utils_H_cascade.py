"""实验 H 工具库 —— Stage1 (tumor vs no_tumor) → Stage2 (mal vs ben | tumor)。

修正 0502 实验 C 的失败:
  C 的 stage1=risk(ben=0.5),stage2=ben/mal —— 强相关,stage2 学不动 (AUC 0.66)。
  H 改为:
    stage1: tumor vs no_tumor (二分类,差异显著,AUC 应可达 0.99+)
    stage2: mal vs ben         (在 tumor 上的精细分类)
  stage2 BCE 用 y_tumor.float() 加权,no_tumor 样本对 stage2 不贡献梯度。

决策规则:
  if p_tumor < tau1:            → low      (no_tumor 直接走 low)
  elif p_mal > tau2:            → high     (mal 直接走 high)
  else:                         → medium   (ben 强制 medium,绝不进 low!)

物理意义: stage1 干净分离 nt → 低 (目标 1+2),stage2 在 tumor 池内分 mal/ben,
         ben 不可能跑到 low band (目标 3)。

阈值搜索:
  tau1 在 val 上需满足 tumor_recall >= 0.98 (硬底线,不漏 tumor)
  tau2 grid 搜在所有满足 (硬+软底线) 的组合里取 obj 最大
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
for _sub in ("0408/scripts", "0402/scripts", "0502/scripts"):
    _p = os.path.join(_PROJECT_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from risk_utils import SwinV2SegGuidedRiskTrimodal  # noqa: E402
from seg_cls_utils_v2 import DiceLoss  # noqa: E402
from risk_utils_E_aux_cls import _band_metrics  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════


class SwinV2SegGuidedRiskCascade(SwinV2SegGuidedRiskTrimodal):
    """删 risk_head, 加 tumor_head + mal_head。

    forward 返回 (seg_logits, tumor_logit, mal_logit)
      tumor_logit: tumor=1 vs no_tumor=0
      mal_logit:   mal=1 vs ben=0 (no_tumor 也输出但不参与 loss)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fusion_dim = kwargs.get("fusion_dim", 256)

        def _head():
            return nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
        self.tumor_head = _head()
        self.mal_head = _head()

    def forward(self, x, metadata=None, input_ids=None, attention_mask=None):
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        f2_proj = self.cls_proj(f2)

        with torch.no_grad():
            text_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        text_hidden = text_out.last_hidden_state
        text_cls = text_hidden[:, 0]

        f2_enhanced = self.cross_attn(f2_proj, text_hidden, attention_mask)

        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]
        attn = F.interpolate(seg_prob, size=f2_enhanced.shape[2:],
                             mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)
        img_feat = (f2_enhanced * attn).sum(dim=(2, 3))

        text_feat = self.text_cls_proj(text_cls)
        meta_feat = self.meta_encoder(metadata.float())

        fused = self.fusion(img_feat, text_feat, meta_feat)

        tumor_logit = self.tumor_head(fused).squeeze(-1)
        mal_logit = self.mal_head(fused).squeeze(-1)
        return seg_logits, tumor_logit, mal_logit


# ═══════════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════════


class CascadeLoss(nn.Module):
    """L = BCE(p_tumor, y_tumor) + lambda_mal * BCE_weighted(p_mal | tumor)
         + (CE + Dice)(seg)
    label 顺序: 0=mal, 1=ben, 2=nt
    y_tumor = (label != 2)
    y_mal   = (label == 0)
    mal head 的 loss 用 y_tumor.float() 加权 (no_tumor 样本梯度=0)
    """

    def __init__(self, lambda_mal: float = 1.0,
                 seg_ce_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.lambda_mal = lambda_mal
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, tumor_logit, mal_logit,
                seg_targets, cls_targets, has_mask):
        y_tumor = (cls_targets != 2).to(tumor_logit.dtype)
        y_mal = (cls_targets == 0).to(tumor_logit.dtype)
        loss_tumor = F.binary_cross_entropy_with_logits(tumor_logit, y_tumor)

        # mal head: 只在真 tumor 样本上算 loss
        mal_loss_per = F.binary_cross_entropy_with_logits(
            mal_logit, y_mal, reduction='none')
        loss_mal = ((mal_loss_per * y_tumor).sum() /
                    (y_tumor.sum() + 1e-6))

        seg_loss = torch.tensor(0.0, device=tumor_logit.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce_v = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice_v = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce_v + seg_dice_v

        total = seg_loss + loss_tumor + self.lambda_mal * loss_mal
        return total, seg_loss.item(), float(loss_tumor), float(loss_mal)


# ═══════════════════════════════════════════════════════════════════════════
#  决策规则
# ═══════════════════════════════════════════════════════════════════════════


def decide_bands_cascade(p_tumor: np.ndarray, p_mal: np.ndarray,
                         tau1: float, tau2: float) -> np.ndarray:
    """tau1: tumor 阈值; tau2: mal 阈值 (在 tumor 池中)
    p_tumor < tau1   → low (no_tumor)
    p_mal   > tau2   → high (mal)
    else             → medium (ben)
    """
    bands = np.full(len(p_tumor), 1, dtype=np.int64)  # 默认 medium
    bands = np.where(p_tumor < tau1, 2, bands)
    bands = np.where((p_tumor >= tau1) & (p_mal > tau2), 0, bands)
    return bands


# ═══════════════════════════════════════════════════════════════════════════
#  阈值搜索 (cascade)
# ═══════════════════════════════════════════════════════════════════════════


def search_thresholds_cascade(
    p_tumor: np.ndarray, p_mal: np.ndarray, labels: np.ndarray,
    tau1_range: Tuple[float, float, float] = (0.10, 0.71, 0.02),
    tau2_range: Tuple[float, float, float] = (0.30, 0.91, 0.02),
    min_tumor_recall: float = 0.98,
    max_ben_to_low: float = 0.05,
    max_nt_to_high: float = 0.05,
) -> Dict:
    """先按 tumor_recall >= min_tumor_recall 缩 tau1,再 grid search tau2 + 软底线。

    硬底线: tumor_recall >= 0.98 (不漏 tumor),mal_to_low = 0,high_recall >= 0.95
    软底线: ben_to_low <= max_ben_to_low,nt_to_high <= max_nt_to_high
    无解 → 软底线 0.05 → 0.10 → 完全去掉(只硬底线) 逐级 fallback。
    """
    y_tumor = (labels != 2).astype(np.int64)
    n_tumor = int(y_tumor.sum())

    def _try(max_ben_low, max_nt_high):
        best = None
        for tau1 in np.arange(*tau1_range):
            tau1_f = float(tau1)
            tumor_recall = (((y_tumor == 1) & (p_tumor >= tau1_f)).sum() /
                            max(n_tumor, 1))
            if tumor_recall < min_tumor_recall:
                continue
            for tau2 in np.arange(*tau2_range):
                tau2_f = float(tau2)
                bands = decide_bands_cascade(p_tumor, p_mal, tau1_f, tau2_f)
                m = _band_metrics(bands, labels)
                if m["mal_to_low"] != 0:           continue
                if m["high_recall"] < 0.95:        continue
                if m["ben_to_low_share"] > max_ben_low: continue
                if m["nt_to_high_share"] > max_nt_high: continue
                obj = (m["high_precision"] + m["low_precision_nt"]) / 2
                if best is None or obj > best["objective"]:
                    best = {**m, "tau1": tau1_f, "tau2": tau2_f,
                            "tumor_recall": float(tumor_recall),
                            "objective": float(obj)}
        return best

    best = _try(max_ben_to_low, max_nt_to_high)
    if best is not None:
        best["fallback_used"] = None
        return best
    best = _try(0.10, 0.10)
    if best is not None:
        best["fallback_used"] = "soft_relaxed_0.10"
        return best
    best = _try(1.0, 1.0)
    if best is not None:
        best["fallback_used"] = "no_soft_floor"
    return best
