"""实验 E 工具库 —— 三分类辅助 head + 类别感知决策。

设计目标
  baseline 的 risk score 在 ben/nt 区间高度重叠 (post-sigmoid 都在 [0.1, 0.4]),
  无法仅靠阈值分开。在 baseline 模型上额外接一个 3-class 辅助 head:
      cls_head: 256 → 128 → 3   (输出 [malignant, benign, no_tumor] logits)
  与 risk_head 并行训练。决策时按 (p_nt, p_mal, p_ben) 优先级硬路由,
  bypass risk score 在 ben/nt 重叠区的不可分性。

决策规则 (val 上 grid search 确定 tau_nt/tau_mal/tau_ben):
  if p_nt  > tau_nt:  → low
  elif p_mal > tau_mal: → high
  elif p_ben > tau_ben: → medium  (绝不进 low!)
  else: 按 risk 双阈值 (t_low, t_high)
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


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════


class SwinV2SegGuidedRiskAuxCls(SwinV2SegGuidedRiskTrimodal):
    """baseline + 3-class 辅助 head (并行 risk_head)。

    forward 返回 (seg_logits, risk_logit, cls_logits)
      cls_logits 顺序: [malignant=0, benign=1, no_tumor=2] (与数据集 label 对齐)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        fusion_dim = kwargs.get("fusion_dim", 256)
        self.cls_head = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    def forward(self, x, metadata=None, input_ids=None, attention_mask=None):
        # ── 完整复制父类 forward 直到 fused,然后在 risk_head 之外多接 cls_head ──
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

        risk_logit = self.risk_head(fused).squeeze(-1)
        cls_logits = self.cls_head(fused)
        return seg_logits, risk_logit, cls_logits


# ═══════════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════════


class RiskOrdinalCEL(nn.Module):
    """L = lambda_ord * BCE_logits(risk, ord)
         + lambda_cls * CE(cls_logits, label)
         + (CE + Dice)(seg)

    label 顺序: 0=mal, 1=ben, 2=nt (与数据集 / cls_head output 对齐)
    """

    def __init__(self, ordinal_targets: torch.Tensor,
                 lambda_ord: float = 2.0,
                 lambda_cls: float = 0.5,
                 seg_ce_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("targets", ordinal_targets)
        self.lambda_ord = lambda_ord
        self.lambda_cls = lambda_cls
        self.cls_ce = nn.CrossEntropyLoss()
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, risk_logit, cls_logits,
                seg_targets, cls_targets, has_mask):
        ord_target = self.targets[cls_targets].to(risk_logit.dtype)
        ord_loss = F.binary_cross_entropy_with_logits(risk_logit, ord_target)
        cls_loss = self.cls_ce(cls_logits, cls_targets)

        seg_loss = torch.tensor(0.0, device=risk_logit.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce_v = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice_v = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce_v + seg_dice_v

        total = (seg_loss
                 + self.lambda_ord * ord_loss
                 + self.lambda_cls * cls_loss)
        return total, seg_loss.item(), ord_loss.item(), cls_loss.item()


# ═══════════════════════════════════════════════════════════════════════════
#  决策规则 (类别感知)
# ═══════════════════════════════════════════════════════════════════════════


def decide_bands_aux_cls_vec(risk_scores: np.ndarray,
                             cls_probs: np.ndarray,
                             t_low: float, t_high: float,
                             tau_nt: float,
                             tau_mal: float,
                             tau_ben: float) -> np.ndarray:
    """向量化决策。
    risk_scores: (N,) post-sigmoid in [0, 1]
    cls_probs:   (N, 3) softmax 后,顺序 [mal, ben, nt]
    返回 bands: (N,) int,0=high, 1=medium, 2=low

    优先级 (与 spec §3.4 if/elif 等价):
      1. p_nt  > tau_nt   → low
      2. p_mal > tau_mal  → high  (覆盖前一步,因为 high > low 优先级)
      3. p_ben > tau_ben  → medium  (ben 强制 medium,绝不进 low)
      4. 否则按 risk 双阈值
    """
    p_mal = cls_probs[:, 0]
    p_ben = cls_probs[:, 1]
    p_nt = cls_probs[:, 2]

    bands = np.full(len(risk_scores), 1, dtype=np.int64)  # 默认 medium

    # Priority 1: nt route to low
    nt_route = p_nt > tau_nt

    # Priority 2: mal route to high (会覆盖 nt route 的样本——但实际上
    # spec 写的是 if/elif,nt 优先于 mal)。这里保持 spec 语义:nt 优先。
    mal_route = (p_mal > tau_mal) & ~nt_route

    # Priority 3: ben route to medium (前 2 个都没触发时才生效)
    ben_route = (p_ben > tau_ben) & ~nt_route & ~mal_route

    # Priority 4: risk 双阈值 (前 3 个都没触发时)
    no_route = ~nt_route & ~mal_route & ~ben_route

    bands = np.where(nt_route, 2, bands)
    bands = np.where(mal_route, 0, bands)
    bands = np.where(ben_route, 1, bands)
    bands = np.where(no_route & (risk_scores >= t_high), 0, bands)
    bands = np.where(no_route & (risk_scores <= t_low), 2, bands)
    # no_route 且 risk 在中间区间的样本保持默认 medium

    return bands


# ═══════════════════════════════════════════════════════════════════════════
#  阈值 + 类别先验 grid search
# ═══════════════════════════════════════════════════════════════════════════


def _band_metrics(bands: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """5 底线 + 关键指标。共享给 F/H 实验也能用。"""
    n_mal = int((labels == 0).sum())
    n_ben = int((labels == 1).sum())
    n_nt = int((labels == 2).sum())
    n_high = int((bands == 0).sum())
    n_low = int((bands == 2).sum())
    n_med = int((bands == 1).sum())

    mal_to_low = int(((labels == 0) & (bands == 2)).sum())
    high_recall = ((labels == 0) & (bands == 0)).sum() / max(n_mal, 1)
    high_precision = (((labels == 0) & (bands == 0)).sum() /
                      n_high) if n_high > 0 else 0.0
    low_precision_nt = (((labels == 2) & (bands == 2)).sum() /
                        n_low) if n_low > 0 else 0.0
    ben_to_low_share = ((labels == 1) & (bands == 2)).sum() / max(n_ben, 1)
    nt_to_high_share = ((labels == 2) & (bands == 0)).sum() / max(n_nt, 1)
    nt_to_low_share = ((labels == 2) & (bands == 2)).sum() / max(n_nt, 1)

    return dict(
        mal_to_low=mal_to_low,
        high_recall=float(high_recall),
        high_precision=float(high_precision),
        low_precision_nt=float(low_precision_nt),
        ben_to_low_share=float(ben_to_low_share),
        nt_to_high_share=float(nt_to_high_share),
        nt_to_low_share=float(nt_to_low_share),
        n_high_pred=n_high, n_med_pred=n_med, n_low_pred=n_low,
    )


def _passes_constraints(m: Dict[str, float],
                        max_ben_low: float,
                        max_nt_high: float) -> bool:
    if m["mal_to_low"] != 0:
        return False
    if m["high_recall"] < 0.95:
        return False
    if m["ben_to_low_share"] > max_ben_low:
        return False
    if m["nt_to_high_share"] > max_nt_high:
        return False
    return True


def search_thresholds_e(
    risk_scores: np.ndarray,
    cls_probs: np.ndarray,
    labels: np.ndarray,
    t_low_range: Tuple[float, float, float] = (0.05, 0.45, 0.10),
    t_high_range: Tuple[float, float, float] = (0.50, 0.95, 0.10),
    tau_range: Tuple[float, float, float] = (0.30, 0.71, 0.05),
    max_ben_to_low: float = 0.05,
    max_nt_to_high: float = 0.05,
) -> Dict:
    """5D grid: (t_low, t_high, tau_nt, tau_mal, tau_ben)。
    硬底线: mal_to_low=0, high_recall>=0.95
    软底线: ben_to_low<=max_ben_to_low, nt_to_high<=max_nt_to_high
    无解 → 软底线 0.05 → 0.10 → no_soft 逐级 fallback。
    """

    def _search(max_ben_low: float, max_nt_high: float):
        best = None
        for t_low in np.arange(*t_low_range):
            for t_high in np.arange(*t_high_range):
                if t_high - t_low < 0.05:
                    continue
                for tau_nt in np.arange(*tau_range):
                    for tau_mal in np.arange(*tau_range):
                        for tau_ben in np.arange(*tau_range):
                            bands = decide_bands_aux_cls_vec(
                                risk_scores, cls_probs,
                                float(t_low), float(t_high),
                                float(tau_nt), float(tau_mal), float(tau_ben))
                            m = _band_metrics(bands, labels)
                            if not _passes_constraints(m, max_ben_low,
                                                        max_nt_high):
                                continue
                            obj = (m["high_precision"]
                                   + m["low_precision_nt"]) / 2
                            if best is None or obj > best["objective"]:
                                best = {
                                    **m,
                                    "t_low": float(t_low),
                                    "t_high": float(t_high),
                                    "tau_nt": float(tau_nt),
                                    "tau_mal": float(tau_mal),
                                    "tau_ben": float(tau_ben),
                                    "objective": float(obj),
                                }
        return best

    # 严格软底线
    best = _search(max_ben_to_low, max_nt_to_high)
    if best is not None:
        best["fallback_used"] = None
        return best
    # fallback 1: 0.10
    best = _search(0.10, 0.10)
    if best is not None:
        best["fallback_used"] = "soft_relaxed_0.10"
        return best
    # fallback 2: 完全去掉软底线
    best = _search(1.0, 1.0)
    if best is not None:
        best["fallback_used"] = "no_soft_floor"
    return best


# ═══════════════════════════════════════════════════════════════════════════
#  通用 evaluator —— 接受外部计算好的 bands (供 E/F/H 共享)
# ═══════════════════════════════════════════════════════════════════════════


def evaluate_bands_external(scores: np.ndarray,
                            bands: np.ndarray,
                            labels: np.ndarray,
                            phase: str = "Test",
                            extra_meta: Optional[Dict] = None) -> Dict:
    """与 0502 risk_utils.evaluate_risk_bands 输出格式对齐的通用 evaluator,
    但接受外部预先计算好的 bands(适配 E/F/H 的非双阈值决策)。

    scores: (N,) post-sigmoid risk score (用于 distribution/calibration/AUC)。
            对 F (无单一 risk score) 可传 p_high 作为 score 占位。
    bands:  (N,) int (0=high, 1=med, 2=low) —— 由实验自己的决策器算
    labels: (N,) int (0=mal, 1=ben, 2=nt)
    """
    from sklearn.metrics import confusion_matrix, roc_auc_score
    n = len(labels)
    BAND_NAMES = ("high", "medium", "low")
    CLS_NAMES = ("malignant", "benign", "no_tumor")

    cm_3band = np.zeros((3, 3), dtype=np.int64)
    for i in range(3):
        for j in range(3):
            cm_3band[i, j] = int(((labels == i) & (bands == j)).sum())

    is_mal = (labels == 0).astype(np.int64)
    is_high = (bands == 0).astype(np.int64)
    cm_safety = confusion_matrix(is_mal, is_high, labels=[0, 1])
    tn_s, fp_s, fn_s, tp_s = cm_safety.ravel()
    high_recall = tp_s / max(tp_s + fn_s, 1)
    high_precision = tp_s / max(tp_s + fp_s, 1)

    per_band = {}
    for b, name in enumerate(BAND_NAMES):
        n_pred = int((bands == b).sum())
        ideal = {0: 0, 1: 1, 2: 2}[b]  # high↔mal, med↔ben, low↔nt
        n_correct = int(((bands == b) & (labels == ideal)).sum())
        n_true_ideal = int((labels == ideal).sum())
        per_band[name] = dict(
            n_pred=n_pred,
            n_true_ideal_label=n_true_ideal,
            ideal_label_name=CLS_NAMES[ideal],
            precision=(n_correct / n_pred) if n_pred > 0 else 0.0,
            recall=(n_correct / n_true_ideal) if n_true_ideal > 0 else 0.0,
            share_of_test=n_pred / n,
        )

    mal_to_low = int(cm_3band[0, 2])
    medium_share = float((bands == 1).sum() / n)
    try:
        binary_auc = float(roc_auc_score(is_mal, scores))
    except ValueError:
        binary_auc = 0.0

    distribution = {}
    for cls_idx, cls_name in enumerate(CLS_NAMES):
        cls_scores = scores[labels == cls_idx]
        if len(cls_scores) > 0:
            distribution[cls_name] = dict(
                n=int(len(cls_scores)),
                mean=float(cls_scores.mean()),
                std=float(cls_scores.std()),
                min=float(cls_scores.min()),
                max=float(cls_scores.max()),
            )

    # 校准曲线 (10-bin)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_mean_pred, bin_mean_true, bin_count = [], [], []
    for i in range(10):
        m = (scores >= bin_edges[i]) & (scores < bin_edges[i + 1] + 1e-9)
        if m.sum() > 0:
            bin_mean_pred.append(float(scores[m].mean()))
            bin_mean_true.append(float(((labels == 0) & m).sum() / m.sum()))
            bin_count.append(int(m.sum()))
        else:
            bin_mean_pred.append(0.0); bin_mean_true.append(0.0); bin_count.append(0)
    brier = float(np.mean((scores - is_mal.astype(np.float64)) ** 2))

    out = dict(
        phase=phase,
        n_total=n,
        confusion_3band=cm_3band.tolist(),
        confusion_binary_safety=cm_safety.tolist(),
        per_band=per_band,
        safety=dict(
            mal_to_low=mal_to_low,
            high_recall=float(high_recall),
            high_precision=float(high_precision),
            medium_share=medium_share,
            binary_roc_auc=binary_auc,
        ),
        distribution=distribution,
        calibration=dict(
            bin_centers=bin_centers.tolist(),
            bin_mean_pred=bin_mean_pred,
            bin_mean_true=bin_mean_true,
            bin_count=bin_count,
            brier_score=brier,
        ),
    )
    if extra_meta:
        out.update(extra_meta)
    return out
