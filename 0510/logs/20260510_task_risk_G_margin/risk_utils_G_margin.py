"""实验 G 工具库 —— Ordinal Margin Ranking Loss。

设计目标
  baseline 在 score 空间下 benign / no_tumor 高度重叠 (post-sigmoid 都在 [0.1, 0.4])。
  在 BCE 之上加 pairwise margin loss,显式约束:
      sigmoid(score_mal)  >= sigmoid(score_ben) + margin
      sigmoid(score_ben)  >= sigmoid(score_nt)  + margin

  margin 操作的是 post-sigmoid score (in [0,1]),不是 logit。
  目的: ben 不再混入 low band,nt 不再混入 high band。

只新增 loss,不新增模型类 —— 复用 baseline 的 SwinV2SegGuidedRiskTrimodal。

阈值搜索
  在 baseline 双阈值基础上加 2 条新软底线:
    benign_to_low  <= 5%   (本轮目标 3)
    no_tumor_to_high <= 5% (本轮目标 1)
  目标: maximize (high_precision + low_precision_nt) / 2
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
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# 复用 0408 的 DiceLoss
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
for _sub in ("0408/scripts", "0402/scripts"):
    _p = os.path.join(_PROJECT_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from seg_cls_utils_v2 import DiceLoss  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════════


class RiskOrdinalMarginLoss(nn.Module):
    """L = lambda_ord * BCE_logits(risk_logit, ord_target)
         + lambda_rank * margin_ranking_loss(sigmoid(logit), label)
         + (CE + Dice)(seg)

    label 顺序: 0=mal, 1=ben, 2=nt (与 ORDINAL_TARGETS_DEFAULT 对齐)
    margin 在 post-sigmoid score 空间度量 (in [0, 1])。
    """

    def __init__(self, ordinal_targets: torch.Tensor,
                 lambda_ord: float = 2.0,
                 lambda_rank: float = 0.3,
                 margin: float = 0.2,
                 seg_ce_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("targets", ordinal_targets)
        self.lambda_ord = lambda_ord
        self.lambda_rank = lambda_rank
        self.margin = margin
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, risk_logit, seg_targets, cls_targets, has_mask):
        # ── BCE ord loss (AMP-safe) ──
        ord_target = self.targets[cls_targets].to(risk_logit.dtype)
        ord_loss = F.binary_cross_entropy_with_logits(risk_logit, ord_target)

        # ── Margin ranking loss (post-sigmoid space, fp32 for AMP safety) ──
        score_fp32 = torch.sigmoid(risk_logit.float())
        rank_loss = self._margin_ranking(score_fp32, cls_targets)

        # ── Seg loss (only on has_mask samples) ──
        seg_loss = torch.tensor(0.0, device=risk_logit.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce_v = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice_v = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce_v + seg_dice_v

        total = (seg_loss
                 + self.lambda_ord * ord_loss
                 + self.lambda_rank * rank_loss.to(risk_logit.dtype))
        return total, seg_loss.item(), ord_loss.item(), float(rank_loss.item())

    def _margin_ranking(self, score: torch.Tensor,
                        label: torch.Tensor) -> torch.Tensor:
        """label: 0=mal, 1=ben, 2=nt. score in [0, 1].
        要求 score(mal) > score(ben) + margin AND score(ben) > score(nt) + margin
        """
        s_mal = score[label == 0]
        s_ben = score[label == 1]
        s_nt = score[label == 2]
        loss = score.new_zeros(())
        # (mal, ben): 期望 s_mal - s_ben > margin
        if s_mal.numel() > 0 and s_ben.numel() > 0:
            diff = s_mal.unsqueeze(1) - s_ben.unsqueeze(0)  # (n_mal, n_ben)
            loss = loss + F.relu(self.margin - diff).mean()
        # (ben, nt): 期望 s_ben - s_nt > margin
        if s_ben.numel() > 0 and s_nt.numel() > 0:
            diff = s_ben.unsqueeze(1) - s_nt.unsqueeze(0)
            loss = loss + F.relu(self.margin - diff).mean()
        return loss


# ═══════════════════════════════════════════════════════════════════════════
#  阈值搜索 + 新软底线
# ═══════════════════════════════════════════════════════════════════════════


def _g_eval(scores: np.ndarray, labels: np.ndarray,
            t_low: float, t_high: float) -> Dict[str, float]:
    """单个阈值组合的 5 底线 + key metrics。"""
    bands = np.where(scores >= t_high, 0,
                     np.where(scores <= t_low, 2, 1))
    n_mal = int((labels == 0).sum())
    n_ben = int((labels == 1).sum())
    n_nt = int((labels == 2).sum())
    mal_to_low = int(((labels == 0) & (bands == 2)).sum())
    ben_to_low = int(((labels == 1) & (bands == 2)).sum())
    nt_to_high = int(((labels == 2) & (bands == 0)).sum())
    high_recall = ((labels == 0) & (bands == 0)).sum() / max(n_mal, 1)
    n_high = int((bands == 0).sum())
    n_low = int((bands == 2).sum())
    high_precision = (((labels == 0) & (bands == 0)).sum() /
                      n_high) if n_high > 0 else 0.0
    low_precision_nt = (((labels == 2) & (bands == 2)).sum() /
                        n_low) if n_low > 0 else 0.0
    return dict(
        t_low=float(t_low), t_high=float(t_high),
        mal_to_low=mal_to_low,
        high_recall=float(high_recall),
        high_precision=float(high_precision),
        low_precision_nt=float(low_precision_nt),
        ben_to_low_share=ben_to_low / max(n_ben, 1),
        nt_to_high_share=nt_to_high / max(n_nt, 1),
        n_high_pred=n_high,
        n_low_pred=n_low,
        n_med_pred=int((bands == 1).sum()),
    )


def search_thresholds_g(
    scores: np.ndarray,
    labels: np.ndarray,
    t_low_range: Tuple[float, float, float] = (0.05, 0.45, 0.02),
    t_high_range: Tuple[float, float, float] = (0.50, 0.95, 0.02),
    max_benign_to_low: float = 0.05,
    max_nt_to_high: float = 0.05,
    min_gap: float = 0.05,
) -> Dict:
    """G 的阈值搜索: baseline 双阈值 + 新软底线。

    硬底线: mal_to_low = 0, high_recall >= 0.95
    软底线: benign_to_low <= max_benign_to_low, no_tumor_to_high <= max_nt_to_high
    目标: maximize (high_precision + low_precision_nt) / 2
    无解时: 软底线 0.05 → 0.10 → 完全去掉(只硬底线)逐级 fallback。
    """
    def _search(max_ben_low, max_nt_high):
        best = None
        for t_low in np.arange(*t_low_range):
            for t_high in np.arange(*t_high_range):
                if t_high - t_low < min_gap:
                    continue
                m = _g_eval(scores, labels, float(t_low), float(t_high))
                if m["mal_to_low"] != 0:
                    continue
                if m["high_recall"] < 0.95:
                    continue
                if m["ben_to_low_share"] > max_ben_low:
                    continue
                if m["nt_to_high_share"] > max_nt_high:
                    continue
                obj = (m["high_precision"] + m["low_precision_nt"]) / 2
                m["objective"] = obj
                if best is None or obj > best["objective"]:
                    best = m
        return best

    best = _search(max_benign_to_low, max_nt_to_high)
    if best is None:
        # fallback 1: 软底线放宽到 0.10
        best = _search(0.10, 0.10)
        if best is not None:
            best["fallback_used"] = "soft_relaxed_0.10"
    if best is None:
        # fallback 2: 完全去掉软底线,只看硬底线
        best = _search(1.0, 1.0)
        if best is not None:
            best["fallback_used"] = "no_soft_floor"
    return best
