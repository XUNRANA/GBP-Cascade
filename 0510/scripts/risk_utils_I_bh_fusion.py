"""实验 I 工具库 —— B + H 融合: ordinal 监督 + cascade 决策。

设计动机
  H 的失败根源: stage1 tumor_head val AUC 仅 0.85 (期望 ≥0.99),
  导致 test 上大量 nt 的 p_tumor >= tau1 → 走 medium,nt→low 仅 7.5%。

  H 的优势: ben→low 仅 2.6% (B 的 74.6% 灾难,baseline 30%),cascade 思路
  对 ben 隔离绝对成功。

  I 在 H 基础上加一个 risk_head,用 B 的非对称 ordinal target (1.0/0.35/0.0)
  做辅助监督。ordinal 损失给 fusion 表示一个跨类别连续顺序约束 (mal > ben > nt),
  期望间接让 tumor_head 学到的 nt 表示更紧凑、与 polyp 表示分离更开。
  推理决策完全走 H 的 cascade (不用 risk_logit 决策)。

模型
  3 个并行 head 共享 fused (256D):
    risk_head  (1D, ordinal 监督 by 1.0/0.35/0.0,B 的 target)
    tumor_head (1D, BCE on y_tumor=mal+ben vs nt)
    mal_head   (1D, BCE on y_mal=mal vs ben,只在 tumor 子集上算)

损失
  L = lambda_ord * BCE(risk_logit, ord_target)
    + BCE(tumor_logit, y_tumor)
    + lambda_mal * BCE_weighted(mal_logit | tumor)
    + (CE + Dice)(seg)

推理
  双温度校准: T_tumor (全 val 上 fit) + T_mal (只在 tumor 子集上 fit)
  cascade 决策: 同 H (tau1 + tau2)
  risk_logit 仅用于诊断 (不参与决策)

预期
  stage1 tumor_AUC 从 H 的 0.85 提升到 0.95+;
  ben→low 保持 H 的 ~3%;
  nt→low 从 H 的 7.5% 提升到 80%+ → 5 底线全过。
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
# 复用 H 的 cascade 决策 + 阈值搜索
from risk_utils_H_cascade import (  # noqa: E402
    decide_bands_cascade, search_thresholds_cascade,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════════════════════


class SwinV2SegGuidedRiskBHFusion(SwinV2SegGuidedRiskTrimodal):
    """3 个并行 head: risk (ordinal supervision) + tumor + mal (cascade decision)。

    forward 返回 (seg_logits, risk_logit, tumor_logit, mal_logit)
      - risk_logit: 1D,ordinal supervision (B 的 1.0/0.35/0.0)
      - tumor_logit: 1D,tumor vs no_tumor 二分类
      - mal_logit: 1D,malignant vs benign 二分类 (只在 tumor 上算 loss)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # 父类已建 risk_head
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

        risk_logit = self.risk_head(fused).squeeze(-1)
        tumor_logit = self.tumor_head(fused).squeeze(-1)
        mal_logit = self.mal_head(fused).squeeze(-1)
        return seg_logits, risk_logit, tumor_logit, mal_logit


# ═══════════════════════════════════════════════════════════════════════════
#  Loss
# ═══════════════════════════════════════════════════════════════════════════


class BHFusionLoss(nn.Module):
    """L = lambda_ord * BCE(risk_logit, ord_target)
         + BCE(tumor_logit, y_tumor)
         + lambda_mal * BCE_weighted(mal_logit | tumor)
         + (CE + Dice)(seg)
    label 顺序: 0=mal, 1=ben, 2=nt (与 ordinal targets 对齐)
    ordinal targets 期望: tensor([1.0, 0.35, 0.0]) (B 的非对称版)
    """

    def __init__(self, ordinal_targets: torch.Tensor,
                 lambda_ord: float = 2.0,
                 lambda_mal: float = 1.0,
                 seg_ce_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("targets", ordinal_targets)
        self.lambda_ord = lambda_ord
        self.lambda_mal = lambda_mal
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, risk_logit, tumor_logit, mal_logit,
                seg_targets, cls_targets, has_mask):
        # ── ordinal BCE (B 的非对称 target) ──
        ord_target = self.targets[cls_targets].to(risk_logit.dtype)
        ord_loss = F.binary_cross_entropy_with_logits(risk_logit, ord_target)

        # ── stage1: tumor vs nt BCE ──
        y_tumor = (cls_targets != 2).to(tumor_logit.dtype)
        y_mal = (cls_targets == 0).to(tumor_logit.dtype)
        loss_tumor = F.binary_cross_entropy_with_logits(tumor_logit, y_tumor)

        # ── stage2: mal head 加权 BCE (只在 tumor 上算) ──
        mal_loss_per = F.binary_cross_entropy_with_logits(
            mal_logit, y_mal, reduction='none')
        loss_mal = ((mal_loss_per * y_tumor).sum() /
                    (y_tumor.sum() + 1e-6))

        # ── seg loss ──
        seg_loss = torch.tensor(0.0, device=risk_logit.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce_v = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice_v = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce_v + seg_dice_v

        total = (seg_loss
                 + self.lambda_ord * ord_loss
                 + loss_tumor
                 + self.lambda_mal * loss_mal)
        return (total, seg_loss.item(), ord_loss.item(),
                float(loss_tumor), float(loss_mal))
