"""
0502 风险预测工具库 (Risk Stratification Utilities)

设计目标
  把"恶性 vs 良性 vs 普通息肉"的三分类，改成"高/中/低风险"三档分层。
  - 训练时：模型输出连续风险分数 ∈ [0, 1]
              malignant → 1.0  (高风险目标)
              benign    → 0.5  (中风险目标)
              no_tumor  → 0.0  (低风险目标)
  - 推理时：用双阈值 (t_low, t_high) 将分数切成三档：
              risk_score >= t_high  → 高风险 (建议手术)
              risk_score <= t_low   → 低风险 (建议随访)
              其余                  → 中风险 (可手术或密切观察)

主要组件
  1. SwinV2SegGuidedRiskTrimodal：在 0408 Exp#18 主干上把 cls_mlp 换成 1D sigmoid
  2. RiskOrdinalLoss：BCE(risk_score, target) + (CE + Dice)(seg_logits, mask)
  3. BalancedBatchSampler：每 batch 各类等比例
  4. search_risk_thresholds_constrained：带"M→低 ≤ 1, 高风险召回 ≥ 0.95,
        中风险占比 ≤ 0.35"约束的双阈值网格搜索，目标最大化低风险精确率
  5. evaluate_risk_bands：3-band 混淆矩阵 + 二值安全口径 + 校准 + 分数分布
  6. plot_risk_confusion_matrices：导出风险分层混淆矩阵图

只依赖 0402/0408 现有工具库，不再需要 0414 全文件。
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import Sampler


# ─── 路径设置：把 0408/0402/0323 的 scripts/ 加进 sys.path，便于 import ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
for _sub in ("0408/scripts", "0402/scripts", "0323/scripts"):
    _p = os.path.join(_PROJECT_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


from seg_cls_utils_v5 import (  # noqa: E402
    SwinV2SegGuidedCls4chTrimodal,
)
from seg_cls_utils_v2 import (  # noqa: E402
    DiceLoss,
    compute_seg_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════
#  1. 模型：复用 0408 Exp#18 主干，替换分类头为 1D sigmoid 风险分数头
# ═══════════════════════════════════════════════════════════════════════════


class SwinV2SegGuidedRiskTrimodal(SwinV2SegGuidedCls4chTrimodal):
    """0408 Exp#18 主干 (SwinV2 + Seg-Guided + BERT CrossAttn + 10D 临床 + Gated Fusion)
    + 1D sigmoid 风险分数头。

    forward 返回 (seg_logits, risk_score)，risk_score ∈ [0, 1]。
    """

    def __init__(self, **kwargs):
        # 父类需要一个合法的 num_cls_classes，我们传 2 占位，后面把 cls_mlp 替换掉
        kwargs.setdefault("num_cls_classes", 2)
        super().__init__(**kwargs)
        fusion_dim = kwargs.get("fusion_dim", 256)
        # 替换 cls_mlp 为 nn.Identity()，节省参数；新建 risk_head
        self.cls_mlp = nn.Identity()
        self.risk_head = nn.Linear(fusion_dim, 1)

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

        # ── BERT encoding (frozen) ──
        with torch.no_grad():
            text_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        text_hidden = text_out.last_hidden_state  # (B, seq_len, 768)
        text_cls = text_hidden[:, 0]              # (B, 768)

        # ── Cross-attention (text → image) ──
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

        # ── Gated trimodal fusion ──
        fused = self.fusion(img_feat, text_feat, meta_feat)  # (B, 256)

        # ── 风险分数头 (输出 raw logit, sigmoid 留到 loss / 推理时) ──
        # 原因: AMP 不允许 F.binary_cross_entropy, 必须用 BCE-with-logits
        risk_logit = self.risk_head(fused).squeeze(-1)  # (B,)
        return seg_logits, risk_logit


# ═══════════════════════════════════════════════════════════════════════════
#  2. 损失函数：BCE(risk) + 加权 (CE + Dice)(seg)
# ═══════════════════════════════════════════════════════════════════════════


# malignant=0, benign=1, no_tumor=2 → ordinal 目标 (1.0, 0.5, 0.0)
ORDINAL_TARGETS_DEFAULT = torch.tensor([1.0, 0.5, 0.0], dtype=torch.float32)


class RiskOrdinalLoss(nn.Module):
    """L_total = lambda_ord * BCE-with-logits(risk_logit, ord_target) + (CE + Dice)(seg)

    - risk_logit: 模型 forward 输出的 raw logit (任意实数)
    - ord_target = ORDINAL_TARGETS[label]，在 [0, 1] 之间
    - seg loss 仅在 has_mask=True 的样本上计算（malignant 通常无病灶 mask，自动跳过）
    - 用 BCE-with-logits 而不是 sigmoid + BCE，是为了让 AMP 安全；二者数学等价
    """

    def __init__(self, ordinal_targets: torch.Tensor,
                 lambda_ord: float = 2.0,
                 seg_ce_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("targets", ordinal_targets)
        self.lambda_ord = lambda_ord
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, risk_logit, seg_targets, cls_targets, has_mask):
        ord_target = self.targets[cls_targets].to(risk_logit.dtype)
        # BCE-with-logits on continuous [0, 1] target —— AMP-safe
        ord_loss = F.binary_cross_entropy_with_logits(risk_logit, ord_target)

        seg_loss = torch.tensor(0.0, device=risk_logit.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce + seg_dice

        total = seg_loss + self.lambda_ord * ord_loss
        return total, seg_loss.item(), ord_loss.item()


# ═══════════════════════════════════════════════════════════════════════════
#  3. BalancedBatchSampler：每个 batch 固定各类样本数
# ═══════════════════════════════════════════════════════════════════════════


class BalancedBatchSampler(Sampler):
    """每个 batch 固定各类数量。例如 samples_per_class={0: 3, 1: 3, 2: 2}。"""

    def __init__(self, labels: Sequence[int], samples_per_class: Dict[int, int],
                 shuffle: bool = True):
        self.labels = np.asarray(labels)
        self.samples_per_class = dict(samples_per_class)
        self.shuffle = shuffle

        self.class_indices: Dict[int, np.ndarray] = {}
        for cls_label in self.samples_per_class:
            idx = np.where(self.labels == cls_label)[0]
            assert len(idx) > 0, f"Class {cls_label} has zero samples"
            self.class_indices[cls_label] = idx

        self.batch_size = sum(self.samples_per_class.values())
        # 以"最常见的类能够无重复跑完"为 num_batches 上界
        self.num_batches = max(
            len(self.class_indices[cls]) // count
            for cls, count in self.samples_per_class.items()
        )

    def __iter__(self):
        class_pools = {}
        for cls_label, indices in self.class_indices.items():
            idx = indices.copy()
            if self.shuffle:
                np.random.shuffle(idx)
            count = self.samples_per_class[cls_label]
            needed = self.num_batches * count
            repeats = (needed // max(len(idx), 1)) + 2
            class_pools[cls_label] = np.tile(idx, repeats)

        pointers = {cls: 0 for cls in self.samples_per_class}
        for _ in range(self.num_batches):
            batch = []
            for cls, count in self.samples_per_class.items():
                ptr = pointers[cls]
                batch.extend(class_pools[cls][ptr:ptr + count].tolist())
                pointers[cls] = ptr + count
            if self.shuffle:
                np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# ═══════════════════════════════════════════════════════════════════════════
#  4. 双阈值约束搜索
# ═══════════════════════════════════════════════════════════════════════════


# 风险档位常量：和"标签"区分，专门表示预测的风险档
BAND_HIGH = 0   # 高风险
BAND_MED = 1    # 中风险
BAND_LOW = 2    # 低风险

# 真实标签 → 该样本"应该"落到的档位（仅用于注释；评估时不直接用）
LABEL_TO_BAND = {0: BAND_HIGH, 1: BAND_MED, 2: BAND_LOW}


def _bands_from_scores(scores: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """风险分数 → 档位编号 (0=高 / 1=中 / 2=低)。"""
    return np.where(scores >= t_high, BAND_HIGH,
            np.where(scores <= t_low, BAND_LOW, BAND_MED))


def _candidate_metrics(scores: np.ndarray, labels: np.ndarray,
                        t_low: float, t_high: float) -> Dict[str, float]:
    """在给定阈值下计算关键指标。labels: 0=mal, 1=ben, 2=no_tumor。"""
    bands = _bands_from_scores(scores, t_low, t_high)
    n = len(labels)
    n_mal = int((labels == 0).sum())
    mal_to_low = int(((labels == 0) & (bands == BAND_LOW)).sum())
    high_recall = (((labels == 0) & (bands == BAND_HIGH)).sum() /
                   max(n_mal, 1))
    medium_share = (bands == BAND_MED).sum() / n

    n_low_pred = int((bands == BAND_LOW).sum())
    n_low_true_notumor = int(((labels == 2) & (bands == BAND_LOW)).sum())
    low_precision = (n_low_true_notumor / n_low_pred) if n_low_pred > 0 else 0.0

    n_high_pred = int((bands == BAND_HIGH).sum())
    n_high_true_mal = int(((labels == 0) & (bands == BAND_HIGH)).sum())
    high_precision = (n_high_true_mal / n_high_pred) if n_high_pred > 0 else 0.0

    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "mal_to_low": mal_to_low,
        "high_recall": float(high_recall),
        "high_precision": float(high_precision),
        "medium_share": float(medium_share),
        "low_precision": float(low_precision),
        "n_high_pred": n_high_pred,
        "n_med_pred": int((bands == BAND_MED).sum()),
        "n_low_pred": n_low_pred,
    }


@dataclass
class ConstraintProfile:
    """阈值搜索的硬底线 / 约束。可以传不同 profile 实现 fallback。"""
    name: str = "primary"
    max_mal_to_low: int = 1
    min_high_recall: float = 0.95
    max_medium_share: float = 0.35


PRIMARY_PROFILE = ConstraintProfile(
    name="primary", max_mal_to_low=1, min_high_recall=0.95, max_medium_share=0.35,
)
FALLBACK_PROFILES: List[ConstraintProfile] = [
    ConstraintProfile("relax_medium", 1, 0.95, 0.50),
    ConstraintProfile("relax_recall", 1, 0.93, 0.50),
    ConstraintProfile("relax_safety", 2, 0.90, 0.60),
]


def search_risk_thresholds_constrained(
    scores: np.ndarray,
    labels: np.ndarray,
    t_low_range: Tuple[float, float, float] = (0.05, 0.45, 0.02),
    t_high_range: Tuple[float, float, float] = (0.50, 0.95, 0.02),
    profile: ConstraintProfile = PRIMARY_PROFILE,
    fallback_profiles: Sequence[ConstraintProfile] = FALLBACK_PROFILES,
    min_gap: float = 0.05,
) -> Dict:
    """带约束的双阈值网格搜索。

    硬底线 (profile)：
      - mal_to_low ≤ profile.max_mal_to_low
      - high_recall ≥ profile.min_high_recall
      - medium_share ≤ profile.max_medium_share

    若 primary profile 无解，自动按 fallback_profiles 顺序回退。
    主目标：maximize low_precision；次目标：minimize medium_share。

    返回 dict 包含：
      - profile_used: "primary" / "relax_medium" / ...
      - t_low, t_high: 选定阈值
      - metrics: 完整指标字典
      - all_candidates: 所有满足约束的候选（包含 metrics）
      - searched_profile: 实际生效的 ConstraintProfile
    """
    profiles_to_try = [profile] + list(fallback_profiles)

    for prof in profiles_to_try:
        candidates: List[Dict] = []
        for t_low in np.arange(*t_low_range):
            for t_high in np.arange(*t_high_range):
                if t_high <= t_low + min_gap:
                    continue
                m = _candidate_metrics(scores, labels, float(t_low), float(t_high))
                if m["mal_to_low"] > prof.max_mal_to_low:
                    continue
                if m["high_recall"] < prof.min_high_recall:
                    continue
                if m["medium_share"] > prof.max_medium_share:
                    continue
                candidates.append(m)

        if candidates:
            # 主目标：低风险精确率最大；次目标：中风险占比最小；
            # 若仍并列，挑 high_recall 更大的。
            candidates.sort(
                key=lambda c: (-c["low_precision"], c["medium_share"], -c["high_recall"])
            )
            best = candidates[0]
            return {
                "profile_used": prof.name,
                "searched_profile": prof,
                "t_low": best["t_low"],
                "t_high": best["t_high"],
                "metrics": best,
                "all_candidates": candidates,
            }

    # 所有 profile 都没有解 → 退化到无约束 max(low_precision)
    fallback_candidates = []
    for t_low in np.arange(*t_low_range):
        for t_high in np.arange(*t_high_range):
            if t_high <= t_low + min_gap:
                continue
            m = _candidate_metrics(scores, labels, float(t_low), float(t_high))
            fallback_candidates.append(m)
    fallback_candidates.sort(
        key=lambda c: (-c["low_precision"], -c["high_recall"], c["medium_share"])
    )
    best = fallback_candidates[0] if fallback_candidates else _candidate_metrics(
        scores, labels, 0.33, 0.66)
    return {
        "profile_used": "unconstrained",
        "searched_profile": ConstraintProfile(
            "unconstrained", 999, 0.0, 1.0,
        ),
        "t_low": best["t_low"],
        "t_high": best["t_high"],
        "metrics": best,
        "all_candidates": [best],
    }


# ═══════════════════════════════════════════════════════════════════════════
#  5. 评估：3-band 混淆矩阵 + 二值安全口径 + 校准 + 分数分布
# ═══════════════════════════════════════════════════════════════════════════


CLASS_NAMES = ["malignant", "benign", "no_tumor"]
BAND_NAMES = ["high", "medium", "low"]


def _calibration_curve(scores: np.ndarray, binary_labels: np.ndarray,
                        n_bins: int = 10) -> Dict:
    """Reliability curve + Brier score。binary_labels: 1=malignant, 0=others."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(scores, bins) - 1, 0, n_bins - 1)
    bin_centers = []
    bin_pred = []
    bin_true = []
    bin_count = []
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        bin_centers.append((bins[b] + bins[b + 1]) / 2.0)
        bin_pred.append(float(scores[mask].mean()))
        bin_true.append(float(binary_labels[mask].mean()))
        bin_count.append(int(mask.sum()))
    brier = float(np.mean((scores - binary_labels) ** 2))
    return {
        "bin_centers": bin_centers,
        "bin_mean_pred": bin_pred,
        "bin_mean_true": bin_true,
        "bin_count": bin_count,
        "brier_score": brier,
    }


def evaluate_risk_bands(
    scores: np.ndarray,
    labels: np.ndarray,
    t_low: float,
    t_high: float,
    logger=None,
    phase: str = "Test",
) -> Dict:
    """对一组 (scores, labels, t_low, t_high) 跑全套评估。

    返回字典包含：
      - confusion_3band: 3x3 矩阵 (行=true label, 列=predicted band)
      - confusion_binary_safety: 2x2 (恶性 vs 非恶性) - 高风险 vs 非高风险
      - per_band: 每档的样本数 / 精确率 / 召回率
      - safety: M→低 漏诊数、高风险召回、二值 ROC-AUC
      - distribution: 各类的 risk_score 均值 / 标准差 / 范围
      - calibration: 校准曲线 + Brier
      - thresholds: 选用的 (t_low, t_high)
    """
    bands = _bands_from_scores(scores, t_low, t_high)
    n = len(labels)

    # ── 3-band 混淆矩阵 (true label × predicted band) ──
    cm_3band = np.zeros((3, 3), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            cm_3band[i, j] = int(((labels == i) & (bands == j)).sum())

    # ── 二值安全口径：true=is_malignant, pred=is_high_risk ──
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
        # 这里"每档的精确率/召回率"用对应"理想真实标签"来定义
        # 高风险档 ↔ malignant (label 0)
        # 低风险档 ↔ no_tumor (label 2)
        # 中风险档 ↔ benign (label 1) —— 但中风险本身允许混合，所以也额外报告
        if b == BAND_HIGH:
            ideal_label = 0
        elif b == BAND_LOW:
            ideal_label = 2
        else:
            ideal_label = 1
        n_true_ideal = int((labels == ideal_label).sum())
        n_correct = int(((bands == b) & (labels == ideal_label)).sum())
        precision = (n_correct / n_pred) if n_pred > 0 else 0.0
        recall = (n_correct / n_true_ideal) if n_true_ideal > 0 else 0.0
        per_band[name] = {
            "n_pred": n_pred,
            "n_true_ideal_label": n_true_ideal,
            "ideal_label_name": CLASS_NAMES[ideal_label],
            "precision": precision,
            "recall": recall,
            "share_of_test": n_pred / n,
        }

    # ── 整体安全指标 ──
    mal_to_low = int(cm_3band[0, BAND_LOW])
    medium_share = float((bands == BAND_MED).sum() / n)
    try:
        binary_auc = float(roc_auc_score(is_mal, scores))
    except ValueError:
        binary_auc = 0.0

    # ── 分数分布 ──
    distribution = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_scores = scores[labels == cls_idx]
        if len(cls_scores) > 0:
            distribution[cls_name] = {
                "n": int(len(cls_scores)),
                "mean": float(cls_scores.mean()),
                "std": float(cls_scores.std()),
                "min": float(cls_scores.min()),
                "max": float(cls_scores.max()),
            }
        else:
            distribution[cls_name] = {"n": 0}

    calib = _calibration_curve(scores, is_mal, n_bins=10)

    result = {
        "phase": phase,
        "thresholds": {"t_low": float(t_low), "t_high": float(t_high)},
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
        },
        "distribution": distribution,
        "calibration": calib,
    }

    if logger is not None:
        _log_evaluation(result, logger)

    return result


def _log_evaluation(result: Dict, logger) -> None:
    """以人友好的格式打印一份完整的评估报告。"""
    phase = result["phase"]
    thr = result["thresholds"]
    cm = result["confusion_3band"]
    safety = result["safety"]
    per_band = result["per_band"]

    logger.info("=" * 72)
    logger.info(f"[{phase}] 风险分层评估 (t_low={thr['t_low']:.3f}, "
                f"t_high={thr['t_high']:.3f})")
    logger.info("=" * 72)

    logger.info(f"[{phase}] 3-band 混淆矩阵 (行=真实, 列=预测档)：")
    logger.info(f"               band_high  band_med   band_low")
    for i, name in enumerate(["malignant", "benign   ", "no_tumor "]):
        row = cm[i]
        logger.info(f"  {name}      {row[0]:>8d}  {row[1]:>8d}  {row[2]:>9d}")

    logger.info(f"[{phase}] 二值安全口径 (高风险 vs 非高风险)：")
    cm_s = result["confusion_binary_safety"]
    logger.info(f"               pred_high  pred_other")
    logger.info(f"  is_malignant  {cm_s[1][1]:>8d}  {cm_s[1][0]:>9d}")
    logger.info(f"  not_mal       {cm_s[0][1]:>8d}  {cm_s[0][0]:>9d}")

    logger.info(f"[{phase}] 关键安全指标：")
    logger.info(f"  M → 低风险 漏诊 (硬底线 ≤ 1)：{safety['mal_to_low']}")
    logger.info(f"  高风险召回 (硬底线 ≥ 0.95) ：{safety['high_recall']:.4f}")
    logger.info(f"  高风险精确率              ：{safety['high_precision']:.4f}")
    logger.info(f"  中风险占比 (约束 ≤ 0.35)  ：{safety['medium_share']:.4f}")
    logger.info(f"  二值 ROC-AUC (恶性 vs 其他)：{safety['binary_roc_auc']:.4f}")

    logger.info(f"[{phase}] 每档统计：")
    for band_name in BAND_NAMES:
        b = per_band[band_name]
        logger.info(
            f"  band_{band_name:6s} | n_pred={b['n_pred']:4d} | "
            f"share={b['share_of_test']:.3f} | "
            f"P(vs {b['ideal_label_name']})={b['precision']:.4f} | "
            f"R(vs {b['ideal_label_name']})={b['recall']:.4f}"
        )

    dist = result["distribution"]
    logger.info(f"[{phase}] 风险分数分布：")
    for cls_name in CLASS_NAMES:
        d = dist[cls_name]
        if d.get("n", 0) > 0:
            logger.info(
                f"  {cls_name:10s} n={d['n']:4d} "
                f"mean={d['mean']:.4f} std={d['std']:.4f} "
                f"[{d['min']:.4f}, {d['max']:.4f}]"
            )

    calib = result["calibration"]
    logger.info(f"[{phase}] 校准 (Brier={calib['brier_score']:.4f})；"
                f"分箱样本数: {calib['bin_count']}")
    logger.info("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════
#  6. 可视化：3-band & 二值安全混淆矩阵
# ═══════════════════════════════════════════════════════════════════════════


def plot_risk_confusion_matrices(results_by_phase: Dict[str, Dict],
                                  out_path: str) -> None:
    """把每个 phase 的 3-band 和 二值安全混淆矩阵画成一张图保存。

    results_by_phase: {"111-test": eval_result, "112-test": eval_result, ...}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_phases = len(results_by_phase)
    fig, axes = plt.subplots(n_phases, 2, figsize=(10, 4 * n_phases),
                             squeeze=False)

    for row, (phase, res) in enumerate(results_by_phase.items()):
        cm3 = np.array(res["confusion_3band"], dtype=np.int64)
        cm2 = np.array(res["confusion_binary_safety"], dtype=np.int64)

        ax3 = axes[row, 0]
        im = ax3.imshow(cm3, cmap="Blues")
        ax3.set_title(f"[{phase}] 3-band CM (true × pred-band)")
        ax3.set_xticks(range(3)); ax3.set_xticklabels(BAND_NAMES)
        ax3.set_yticks(range(3)); ax3.set_yticklabels(CLASS_NAMES)
        ax3.set_xlabel("predicted band"); ax3.set_ylabel("true label")
        for i in range(3):
            for j in range(3):
                ax3.text(j, i, str(cm3[i, j]),
                         ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax3, fraction=0.046)

        ax2 = axes[row, 1]
        im = ax2.imshow(cm2, cmap="Reds")
        ax2.set_title(f"[{phase}] Binary safety CM (mal × pred-high)")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["pred_other", "pred_high"])
        ax2.set_yticks([0, 1]); ax2.set_yticklabels(["not_mal", "is_mal"])
        for i in range(2):
            for j in range(2):
                ax2.text(j, i, str(cm2[i, j]),
                         ha="center", va="center", color="black")
        plt.colorbar(im, ax=ax2, fraction=0.046)

    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
#  7. 患者级 split 工具（沿用 0414 的命名规则）
# ═══════════════════════════════════════════════════════════════════════════


def extract_patient_id(image_path: str) -> str:
    fname = os.path.basename(image_path)
    return fname.split("_US_")[0] if "_US_" in fname else fname.rsplit("_", 1)[0]


# ═══════════════════════════════════════════════════════════════════════════
#  8. JSON / 字典保存助手
# ═══════════════════════════════════════════════════════════════════════════


def dump_json(obj, path: str) -> None:
    """支持 numpy 类型的 JSON 落盘。"""

    def _convert(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, ConstraintProfile):
            return asdict(o)
        return str(o)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_convert)


# ═══════════════════════════════════════════════════════════════════════════
#  9. 自测：阈值搜索单元测试
# ═══════════════════════════════════════════════════════════════════════════


def _selftest_threshold_search() -> None:
    """构造一组干净的合成数据，验证阈值搜索能找到正确的解。"""
    rng = np.random.RandomState(42)
    n_each = 100
    # 恶性：分数中心 0.85
    mal_scores = np.clip(rng.normal(0.85, 0.05, n_each), 0, 1)
    # 良性：分数中心 0.5
    ben_scores = np.clip(rng.normal(0.50, 0.10, n_each), 0, 1)
    # 普通息肉：分数中心 0.10
    nt_scores = np.clip(rng.normal(0.10, 0.05, n_each), 0, 1)

    scores = np.concatenate([mal_scores, ben_scores, nt_scores])
    labels = np.concatenate([
        np.full(n_each, 0), np.full(n_each, 1), np.full(n_each, 2)
    ])

    res = search_risk_thresholds_constrained(scores, labels)
    assert res["profile_used"] in ("primary", "relax_medium", "relax_recall",
                                    "relax_safety", "unconstrained"), \
        f"unexpected profile: {res['profile_used']}"
    assert 0.05 <= res["t_low"] <= 0.45
    assert 0.50 <= res["t_high"] <= 0.95
    assert res["t_high"] > res["t_low"]
    print(f"[selftest] OK profile={res['profile_used']} "
          f"t_low={res['t_low']:.3f} t_high={res['t_high']:.3f} "
          f"low_p={res['metrics']['low_precision']:.4f} "
          f"high_recall={res['metrics']['high_recall']:.4f}")


if __name__ == "__main__":
    _selftest_threshold_search()
