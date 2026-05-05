"""实验 C 工具库 —— 双阶段决策 (Two-Stage Cascade)。

设计目标
  把"安全筛查 (恶性 vs 非恶性)"和"细分 (benign vs no_tumor)"完全解耦。
    - 阶段 1: 沿用 baseline 的 risk_logit (BCE + ordinal target)
    - 阶段 2: 新增 stage2_head,只在非恶性样本上学 benign vs no_tumor
              recall 取信号

推理:
    if score >= t_high            → high
    elif p_benign >= t_med        → medium     (在非高样本里再切)
    else                           → low

新增组件
  1. SwinV2SegGuidedRiskTwoStage   —— 加 stage2_head, forward 返回 3-tuple
  2. RiskTwoStageLoss              —— BCE_risk + (CE+Dice) seg
                                       + λ_stage2 * BCE_stage2 (掩掉恶性样本)
  3. search_two_stage_thresholds   —— 先 stage1 锁 t_high (保安全), 然后在
                                       non-high 样本里 grid 搜 t_med 最大化
                                       low_precision
  4. evaluate_two_stage_bands      —— 三档输出 + 同时报告 stage2 head 的 b/n AUC
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
for _sub in ("0408/scripts", "0402/scripts", "0323/scripts"):
    _p = os.path.join(_PROJECT_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


from risk_utils import (  # noqa: E402
    SwinV2SegGuidedRiskTrimodal,
    BAND_HIGH, BAND_MED, BAND_LOW, BAND_NAMES, CLASS_NAMES,
    ConstraintProfile, PRIMARY_PROFILE, FALLBACK_PROFILES,
    _calibration_curve,
)
from seg_cls_utils_v2 import DiceLoss  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  Model: 在 baseline 上加 stage2_head
# ═══════════════════════════════════════════════════════════════════════════


class SwinV2SegGuidedRiskTwoStage(SwinV2SegGuidedRiskTrimodal):
    """baseline 模型 + 第二个 BCE head (stage2_head)。

    forward 返回 (seg_logits, risk_logit, stage2_logit),
    stage2_logit 用于 benign(=0) vs no_tumor(=1) 二分类。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fusion_dim = kwargs.get("fusion_dim", 256)
        self.stage2_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_dim // 2, 1),
        )

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
                input_ids=input_ids, attention_mask=attention_mask)
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

        risk_logit = self.risk_head(fused).squeeze(-1)
        stage2_logit = self.stage2_head(fused).squeeze(-1)
        return seg_logits, risk_logit, stage2_logit


# ═══════════════════════════════════════════════════════════════════════════
#  Loss: BCE_risk + (CE+Dice) seg + λ_stage2 * BCE_stage2 (mask 掉恶性)
# ═══════════════════════════════════════════════════════════════════════════


class RiskTwoStageLoss(nn.Module):
    """L = lambda_ord * BCE-with-logits(risk_logit, ord_target)
            + (CE+Dice)(seg)
            + lambda_stage2 * BCE-with-logits(stage2_logit, is_benign)[on non-malignant]

    - is_benign: label==1 → 1, label==2 (no_tumor) → 0, label==0 (mal) → 不参与
    - 恶性样本的 stage2 BCE 通过 mask 置零,不污染 stage2 的判别
    """

    def __init__(self, ordinal_targets: torch.Tensor,
                 lambda_ord: float = 2.0,
                 lambda_stage2: float = 1.0,
                 seg_ce_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("targets", ordinal_targets)
        self.lambda_ord = lambda_ord
        self.lambda_stage2 = lambda_stage2
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, risk_logit, stage2_logit,
                seg_targets, cls_targets, has_mask):
        # ── stage1: BCE on continuous ordinal target ──
        ord_target = self.targets[cls_targets].to(risk_logit.dtype)
        ord_loss = F.binary_cross_entropy_with_logits(risk_logit, ord_target)

        # ── seg ──
        seg_loss = torch.tensor(0.0, device=risk_logit.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce + seg_dice

        # ── stage2: 仅在 non-malignant 样本上算 BCE ──
        non_mal_mask = (cls_targets != 0)
        stage2_loss = torch.tensor(0.0, device=risk_logit.device,
                                    dtype=risk_logit.dtype)
        if non_mal_mask.any():
            idx = non_mal_mask.nonzero(as_tuple=True)[0]
            s2_logit = stage2_logit[idx]
            # benign(label=1) → 1, no_tumor(label=2) → 0
            s2_target = (cls_targets[idx] == 1).to(s2_logit.dtype)
            stage2_loss = F.binary_cross_entropy_with_logits(s2_logit, s2_target)

        total = (seg_loss
                 + self.lambda_ord * ord_loss
                 + self.lambda_stage2 * stage2_loss)
        return (total,
                seg_loss.item(),
                ord_loss.item(),
                float(stage2_loss.item()))


# ═══════════════════════════════════════════════════════════════════════════
#  双阶段阈值搜索
# ═══════════════════════════════════════════════════════════════════════════


def _two_stage_bands(scores_stage1: np.ndarray,
                     p_benign: np.ndarray,
                     t_high: float, t_med: float) -> np.ndarray:
    """
        if score >= t_high            → high (0)
        elif p_benign >= t_med        → medium (1)
        else                           → low (2)
    """
    bands = np.full_like(scores_stage1, fill_value=BAND_LOW, dtype=np.int64)
    is_high = scores_stage1 >= t_high
    bands[is_high] = BAND_HIGH
    is_med = (~is_high) & (p_benign >= t_med)
    bands[is_med] = BAND_MED
    return bands


def _two_stage_metrics(scores_stage1: np.ndarray, p_benign: np.ndarray,
                        labels: np.ndarray, t_high: float, t_med: float
                        ) -> Dict[str, float]:
    bands = _two_stage_bands(scores_stage1, p_benign, t_high, t_med)
    n = len(labels)
    n_mal = int((labels == 0).sum())
    mal_to_low = int(((labels == 0) & (bands == BAND_LOW)).sum())
    high_recall = (((labels == 0) & (bands == BAND_HIGH)).sum() / max(n_mal, 1))
    medium_share = (bands == BAND_MED).sum() / n

    n_low_pred = int((bands == BAND_LOW).sum())
    n_low_true_nt = int(((labels == 2) & (bands == BAND_LOW)).sum())
    low_precision = (n_low_true_nt / n_low_pred) if n_low_pred > 0 else 0.0

    n_high_pred = int((bands == BAND_HIGH).sum())
    n_high_true_mal = int(((labels == 0) & (bands == BAND_HIGH)).sum())
    high_precision = (n_high_true_mal / n_high_pred) if n_high_pred > 0 else 0.0

    return {
        "t_high": float(t_high),
        "t_med": float(t_med),
        "mal_to_low": mal_to_low,
        "high_recall": float(high_recall),
        "high_precision": float(high_precision),
        "medium_share": float(medium_share),
        "low_precision": float(low_precision),
        "n_high_pred": n_high_pred,
        "n_med_pred": int((bands == BAND_MED).sum()),
        "n_low_pred": n_low_pred,
    }


def search_two_stage_thresholds(
    scores_stage1: np.ndarray,
    p_benign: np.ndarray,
    labels: np.ndarray,
    t_high_range: Tuple[float, float, float] = (0.40, 0.85, 0.02),
    t_med_range: Tuple[float, float, float] = (0.10, 0.90, 0.02),
    profile: ConstraintProfile = PRIMARY_PROFILE,
    fallback_profiles: Sequence[ConstraintProfile] = FALLBACK_PROFILES,
) -> Dict:
    """先用 stage1 锁 t_high (保安全),再在 non-high 样本里 grid 搜 t_med
    最大化 low_precision。

    硬底线 (profile):
      - mal_to_low ≤ profile.max_mal_to_low
      - high_recall ≥ profile.min_high_recall
      - medium_share ≤ profile.max_medium_share
    """
    profiles_to_try = [profile] + list(fallback_profiles)

    for prof in profiles_to_try:
        candidates: List[Dict] = []
        for t_high in np.arange(*t_high_range):
            for t_med in np.arange(*t_med_range):
                m = _two_stage_metrics(
                    scores_stage1, p_benign, labels, float(t_high), float(t_med))
                if m["mal_to_low"] > prof.max_mal_to_low:
                    continue
                if m["high_recall"] < prof.min_high_recall:
                    continue
                if m["medium_share"] > prof.max_medium_share:
                    continue
                candidates.append(m)
        if candidates:
            candidates.sort(key=lambda c: (-c["low_precision"], c["medium_share"],
                                            -c["high_recall"]))
            best = candidates[0]
            return {
                "profile_used": prof.name,
                "searched_profile": prof,
                "t_high": best["t_high"], "t_med": best["t_med"],
                "metrics": best,
                "all_candidates": candidates,
            }

    # 所有 profile 都没解 → 退化到无约束 max(low_precision)
    fb = []
    for t_high in np.arange(*t_high_range):
        for t_med in np.arange(*t_med_range):
            m = _two_stage_metrics(
                scores_stage1, p_benign, labels, float(t_high), float(t_med))
            fb.append(m)
    fb.sort(key=lambda c: (-c["low_precision"], -c["high_recall"], c["medium_share"]))
    best = fb[0]
    return {
        "profile_used": "unconstrained",
        "searched_profile": ConstraintProfile("unconstrained", 999, 0.0, 1.0),
        "t_high": best["t_high"], "t_med": best["t_med"],
        "metrics": best,
        "all_candidates": [best],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  双阶段评估
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_two_stage_bands(
    scores_stage1: np.ndarray,
    p_benign: np.ndarray,
    labels: np.ndarray,
    t_high: float, t_med: float,
    logger=None, phase: str = "Test",
) -> Dict:
    bands = _two_stage_bands(scores_stage1, p_benign, t_high, t_med)
    n = len(labels)

    # ── 3-band 混淆矩阵 ──
    cm_3band = np.zeros((3, 3), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            cm_3band[i, j] = int(((labels == i) & (bands == j)).sum())

    # ── 二值安全口径 ──
    is_mal = (labels == 0).astype(np.int64)
    is_high = (bands == BAND_HIGH).astype(np.int64)
    cm_safety = confusion_matrix(is_mal, is_high, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_safety.ravel()
    high_recall = tp_s / max(tp_s + fn_s, 1)
    high_precision = tp_s / max(tp_s + fp_s, 1)

    # ── 每档统计 ──
    per_band = {}
    for b, name in enumerate(BAND_NAMES):
        n_pred = int((bands == b).sum())
        if b == BAND_HIGH:
            ideal = 0
        elif b == BAND_LOW:
            ideal = 2
        else:
            ideal = 1
        n_true_ideal = int((labels == ideal).sum())
        n_correct = int(((bands == b) & (labels == ideal)).sum())
        precision = (n_correct / n_pred) if n_pred > 0 else 0.0
        recall = (n_correct / n_true_ideal) if n_true_ideal > 0 else 0.0
        per_band[name] = {
            "n_pred": n_pred,
            "n_true_ideal_label": n_true_ideal,
            "ideal_label_name": CLASS_NAMES[ideal],
            "precision": precision, "recall": recall,
            "share_of_test": n_pred / n,
        }

    mal_to_low = int(cm_3band[0, BAND_LOW])
    medium_share = float((bands == BAND_MED).sum() / n)
    try:
        binary_auc = float(roc_auc_score(is_mal, scores_stage1))
    except ValueError:
        binary_auc = 0.0

    # ── stage2 内部诊断: benign vs no_tumor (在 non-malignant 上) ──
    nm_mask = (labels != 0)
    stage2_auc = 0.0
    if nm_mask.sum() > 1 and len(set(labels[nm_mask])) >= 2:
        is_ben = (labels[nm_mask] == 1).astype(np.int64)
        try:
            stage2_auc = float(roc_auc_score(is_ben, p_benign[nm_mask]))
        except ValueError:
            stage2_auc = 0.0

    # ── 分数分布 (stage1 + stage2 各类别) ──
    distribution_stage1 = {}
    distribution_stage2 = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        s1 = scores_stage1[labels == cls_idx]
        s2 = p_benign[labels == cls_idx]
        if len(s1) > 0:
            distribution_stage1[cls_name] = {
                "n": int(len(s1)), "mean": float(s1.mean()),
                "std": float(s1.std()),
                "min": float(s1.min()), "max": float(s1.max()),
            }
        if len(s2) > 0:
            distribution_stage2[cls_name] = {
                "n": int(len(s2)), "mean": float(s2.mean()),
                "std": float(s2.std()),
                "min": float(s2.min()), "max": float(s2.max()),
            }

    calib = _calibration_curve(scores_stage1, is_mal, n_bins=10)

    result = {
        "phase": phase,
        "thresholds": {"t_high": float(t_high), "t_med": float(t_med)},
        "n_total": int(n),
        "confusion_3band": cm_3band.tolist(),
        "confusion_binary_safety": cm_safety.tolist(),
        "per_band": per_band,
        "safety": {
            "mal_to_low": mal_to_low,
            "high_recall": float(high_recall),
            "high_precision": float(high_precision),
            "medium_share": medium_share,
            "binary_roc_auc": binary_auc,
            "stage2_roc_auc_b_vs_n": stage2_auc,
        },
        "distribution_stage1": distribution_stage1,
        "distribution_stage2": distribution_stage2,
        "calibration": calib,
    }

    if logger is not None:
        _log_two_stage(result, logger)
    return result


def _log_two_stage(result: Dict, logger) -> None:
    phase = result["phase"]
    thr = result["thresholds"]
    cm = result["confusion_3band"]
    safety = result["safety"]
    per_band = result["per_band"]

    logger.info("=" * 72)
    logger.info(f"[{phase}] 双阶段风险分层评估 "
                f"(t_high={thr['t_high']:.3f}, t_med={thr['t_med']:.3f})")
    logger.info("=" * 72)
    logger.info(f"[{phase}] 3-band 混淆矩阵 (行=真实, 列=预测档):")
    logger.info(f"               band_high  band_med   band_low")
    for i, name in enumerate(["malignant", "benign   ", "no_tumor "]):
        row = cm[i]
        logger.info(f"  {name}      {row[0]:>8d}  {row[1]:>8d}  {row[2]:>9d}")
    logger.info(f"[{phase}] 关键指标:")
    logger.info(f"  M → 低风险 漏诊: {safety['mal_to_low']}")
    logger.info(f"  高风险召回    : {safety['high_recall']:.4f}")
    logger.info(f"  高风险精确率  : {safety['high_precision']:.4f}")
    logger.info(f"  中风险占比    : {safety['medium_share']:.4f}")
    logger.info(f"  二值 ROC-AUC  : {safety['binary_roc_auc']:.4f}")
    logger.info(f"  Stage2 b/n AUC: {safety['stage2_roc_auc_b_vs_n']:.4f}")
    logger.info(f"[{phase}] 每档统计:")
    for band_name in BAND_NAMES:
        b = per_band[band_name]
        logger.info(
            f"  band_{band_name:6s} | n_pred={b['n_pred']:4d} | "
            f"share={b['share_of_test']:.3f} | "
            f"P(vs {b['ideal_label_name']})={b['precision']:.4f} | "
            f"R(vs {b['ideal_label_name']})={b['recall']:.4f}"
        )
    logger.info("=" * 72)
