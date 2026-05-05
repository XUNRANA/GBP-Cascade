"""实验 A 工具库 —— Supervised Contrastive 边界对比学习。

设计目标
  在 baseline (单 BCE-with-logits ordinal head) 基础上,新增 projection head
  并加入 SupConLoss,把 benign / no_tumor 在特征空间显式拉开 ——
  直接攻击 baseline 中 benign 仅 53.5% 识别率的瓶颈。

新增组件
  1. SupConLoss             —— 复制自 0414/.../20260414_task3_SwinV2Tiny_segcls_4.py:167-214
  2. SwinV2SegGuidedRiskSupCon —— baseline 模型的子类,新增 proj_head,
                                  forward 返回 (seg_logits, risk_logit, proj_feat)
  3. RiskOrdinalSupConLoss  —— L = lambda_ord * BCE + (CE+Dice) seg + lambda_supcon * SupCon

注意
  - proj_feat 已 L2 归一化,直接喂给 SupConLoss
  - SupCon 只在 batch 中类别多样性 ≥ 2 时才贡献梯度,否则返回 0
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
for _sub in ("0408/scripts", "0402/scripts", "0323/scripts"):
    _p = os.path.join(_PROJECT_ROOT, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


from risk_utils import SwinV2SegGuidedRiskTrimodal  # noqa: E402
from seg_cls_utils_v2 import DiceLoss              # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  SupConLoss (来源: 0414/scripts/20260414_task3_SwinV2Tiny_segcls_4.py:167-214)
# ═══════════════════════════════════════════════════════════════════════════


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    拉近同类样本的嵌入, 推远异类样本.
    核心作用: 强制 benign 和 no_tumor 在特征空间中分离.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = features.device
        B = features.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        labels_col = labels.unsqueeze(0)
        labels_row = labels.unsqueeze(1)
        pos_mask = (labels_row == labels_col).float()
        diag_mask = 1.0 - torch.eye(B, device=device)
        pos_mask = pos_mask * diag_mask

        # 数值稳定
        logits_max, _ = sim.detach().max(dim=1, keepdim=True)
        logits = sim - logits_max

        exp_logits = torch.exp(logits) * diag_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-6)

        pos_count = pos_mask.sum(dim=1).clamp(min=1.0)
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / pos_count

        return -mean_log_prob.mean()


# ═══════════════════════════════════════════════════════════════════════════
#  SwinV2SegGuidedRiskSupCon —— 在 baseline 上加 projection head
# ═══════════════════════════════════════════════════════════════════════════


class SwinV2SegGuidedRiskSupCon(SwinV2SegGuidedRiskTrimodal):
    """baseline 模型 + proj_head(L2 normalized).

    forward 返回 (seg_logits, risk_logit, proj_feat),
    其中 proj_feat 已 L2 归一化,可直接喂给 SupConLoss。
    """

    def __init__(self, proj_dim: int = 128, **kwargs):
        super().__init__(**kwargs)
        fusion_dim = kwargs.get("fusion_dim", 256)
        self.proj_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, proj_dim),
        )

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
        f2_proj = self.cls_proj(f2)

        # ── BERT (frozen) ──
        with torch.no_grad():
            text_out = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
        text_hidden = text_out.last_hidden_state
        text_cls = text_hidden[:, 0]

        # ── Cross-attention ──
        f2_enhanced = self.cross_attn(f2_proj, text_hidden, attention_mask)

        # ── Seg-guided attention pooling ──
        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]
        attn = F.interpolate(seg_prob, size=f2_enhanced.shape[2:],
                             mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)
        img_feat = (f2_enhanced * attn).sum(dim=(2, 3))

        # ── Text [CLS] / Meta encoding ──
        text_feat = self.text_cls_proj(text_cls)
        meta_feat = self.meta_encoder(metadata.float())

        # ── Gated trimodal fusion ──
        fused = self.fusion(img_feat, text_feat, meta_feat)

        # ── Risk head + Projection head ──
        risk_logit = self.risk_head(fused).squeeze(-1)
        proj_raw = self.proj_head(fused)
        # 用 fp32 做 L2 归一化,避免 AMP 下 fp16 的数值不稳
        proj_feat = F.normalize(proj_raw.float(), dim=1)
        return seg_logits, risk_logit, proj_feat


# ═══════════════════════════════════════════════════════════════════════════
#  Loss: BCE-with-logits + (CE+Dice) seg + lambda_supcon * SupCon
# ═══════════════════════════════════════════════════════════════════════════


class RiskOrdinalSupConLoss(nn.Module):
    """总损失 = seg(CE+Dice) + lambda_ord * BCE-with-logits + lambda_supcon * SupCon.

    - risk_logit: raw logit (AMP-safe BCE-with-logits 输入)
    - proj_feat:  已 L2 归一化的 projection 向量
    - SupCon temperature 默认 0.1
    """

    def __init__(self, ordinal_targets: torch.Tensor,
                 lambda_ord: float = 2.0,
                 lambda_supcon: float = 0.5,
                 supcon_temperature: float = 0.1,
                 seg_ce_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("targets", ordinal_targets)
        self.lambda_ord = lambda_ord
        self.lambda_supcon = lambda_supcon
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()
        self.supcon = SupConLoss(temperature=supcon_temperature)

    def forward(self, seg_logits, risk_logit, proj_feat,
                seg_targets, cls_targets, has_mask):
        ord_target = self.targets[cls_targets].to(risk_logit.dtype)
        ord_loss = F.binary_cross_entropy_with_logits(risk_logit, ord_target)

        seg_loss = torch.tensor(0.0, device=risk_logit.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce + seg_dice

        # SupCon: 在已 L2 归一化的 proj_feat 上算
        supcon_loss = self.supcon(proj_feat, cls_targets)

        total = (seg_loss
                 + self.lambda_ord * ord_loss
                 + self.lambda_supcon * supcon_loss)
        return (total,
                seg_loss.item(),
                ord_loss.item(),
                float(supcon_loss.item()))
