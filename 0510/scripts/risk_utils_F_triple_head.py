"""实验 F 工具库 —— 三档独立 BCE head + argmax 决策(放弃 ordinal 回归)。

设计目标
  替换 baseline 的 risk_head 为 3 个独立二分类 head:
    head_high (mal=1 vs others=0)
    head_med  (ben=1 vs others=0)
    head_low  (nt=1 vs others=0)
  决策直接 argmax (p_high, p_med, p_low),用 tau_safe 做 high_recall fallback。

  注意: 失去 ordinal 性质,温度校准退化为对每个 head 单独 sigmoid scaling。
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
from risk_utils_E_aux_cls import _band_metrics  # noqa: E402  共享 metric 工具


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════


class SwinV2SegGuidedRiskTripleHead(SwinV2SegGuidedRiskTrimodal):
    """删 risk_head, 加 3 个独立二分类 head。

    forward 返回 (seg_logits, logit_high, logit_med, logit_low)
      logit_high: mal=1 vs others=0
      logit_med:  ben=1 vs others=0
      logit_low:  nt=1 vs others=0
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fusion_dim = kwargs.get("fusion_dim", 256)
        # 父类的 risk_head 我们不再用,但留着不影响(也省得 state_dict 兼容麻烦)。
        # 实际 forward 不调用 self.risk_head,只调 3 个新 head。
        def _head():
            return nn.Sequential(
                nn.Linear(fusion_dim, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
            )
        self.head_high = _head()
        self.head_med = _head()
        self.head_low = _head()

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

        logit_high = self.head_high(fused).squeeze(-1)
        logit_med = self.head_med(fused).squeeze(-1)
        logit_low = self.head_low(fused).squeeze(-1)
        return seg_logits, logit_high, logit_med, logit_low


# ═══════════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════════


class TripleBCELoss(nn.Module):
    """L = BCE(p_high, y_high) + BCE(p_med, y_med) + BCE(p_low, y_low)
         + (CE + Dice)(seg)
    label 顺序: 0=mal, 1=ben, 2=nt
    y_high = (label==0); y_med = (label==1); y_low = (label==2)
    """

    def __init__(self, seg_ce_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, logit_high, logit_med, logit_low,
                seg_targets, cls_targets, has_mask):
        y_high = (cls_targets == 0).to(logit_high.dtype)
        y_med = (cls_targets == 1).to(logit_high.dtype)
        y_low = (cls_targets == 2).to(logit_high.dtype)
        bce_high = F.binary_cross_entropy_with_logits(logit_high, y_high)
        bce_med = F.binary_cross_entropy_with_logits(logit_med, y_med)
        bce_low = F.binary_cross_entropy_with_logits(logit_low, y_low)

        seg_loss = torch.tensor(0.0, device=logit_high.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce_v = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice_v = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce_v + seg_dice_v

        total = seg_loss + bce_high + bce_med + bce_low
        return total, seg_loss.item(), float(bce_high), float(bce_med), float(bce_low)


# ═══════════════════════════════════════════════════════════════════════════
#  决策规则
# ═══════════════════════════════════════════════════════════════════════════


def decide_bands_triple(p_high: np.ndarray, p_med: np.ndarray, p_low: np.ndarray,
                        tau_safe: Optional[float] = None) -> np.ndarray:
    """argmax(p_high, p_med, p_low) + 安全 fallback。
    p_*: shape (N,) post-sigmoid in [0, 1]
    返回 bands: 0=high, 1=med, 2=low
    """
    bands = np.argmax(np.stack([p_high, p_med, p_low], axis=1), axis=1)
    if tau_safe is not None:
        # 即使 argmax 不是 high,只要 p_high > tau_safe 就保守归 high
        bands = np.where(p_high > tau_safe, 0, bands)
    return bands.astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════
#  tau_safe 搜索
# ═══════════════════════════════════════════════════════════════════════════


def search_tau_safe(p_high: np.ndarray, p_med: np.ndarray, p_low: np.ndarray,
                    labels: np.ndarray,
                    tau_range: Tuple[float, float, float] = (0.10, 0.51, 0.02),
                    max_ben_to_low: float = 0.10,
                    max_nt_to_high: float = 0.10) -> Dict:
    """在 val 上搜最小 tau_safe 满足全部 5 条底线 (硬+软)。
      tau_safe 越小,安全 fallback 越频繁,可能伤精度;
      因此偏好「能满足全部底线的最大 tau_safe」(即:依赖 fallback 越少越好,
      但 high_recall 不能掉)。

    硬底线: mal_to_low=0, high_recall>=0.95
    软底线: ben_to_low<=max_ben_to_low, nt_to_high<=max_nt_to_high
    无解 → fallback (放宽到 0.20 / 0.20) → 最差 fallback 仅硬底线。
    """

    def _try(max_ben_low, max_nt_high):
        best = None
        for tau in np.arange(*tau_range):
            tau_f = float(tau)
            bands = decide_bands_triple(p_high, p_med, p_low, tau_safe=tau_f)
            m = _band_metrics(bands, labels)
            if m["mal_to_low"] != 0:        continue
            if m["high_recall"] < 0.95:     continue
            if m["ben_to_low_share"] > max_ben_low:  continue
            if m["nt_to_high_share"] > max_nt_high:  continue
            obj = (m["high_precision"] + m["low_precision_nt"]) / 2
            # 偏好更大的 tau (依赖 fallback 越少越好) + 更高的 obj
            score = (tau_f, obj)
            if best is None or score > (best["tau_safe"], best["objective"]):
                best = {**m, "tau_safe": tau_f, "objective": float(obj)}
        return best

    best = _try(max_ben_to_low, max_nt_to_high)
    if best is not None:
        best["fallback_used"] = None
        return best
    best = _try(0.20, 0.20)
    if best is not None:
        best["fallback_used"] = "soft_relaxed_0.20"
        return best
    # 最差 fallback: 只看硬底线
    best = _try(1.0, 1.0)
    if best is not None:
        best["fallback_used"] = "no_soft_floor"
    return best
