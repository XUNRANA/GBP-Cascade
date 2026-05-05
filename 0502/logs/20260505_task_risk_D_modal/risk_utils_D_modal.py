"""实验 D 工具库 —— Modality Dropout + Gate 熵正则 + 文本/临床增强。

设计目标
  baseline 在 Test-112 上 high_precision 比 Test-111 掉 7pt,说明决策边界对
  样本分布敏感。当前可能 image 模态过强,benign vs no_tumor 影像本身极相似,
  必须让 BERT 文本和 10D 临床特征发挥更大作用。

新增组件
  1. SwinV2SegGuidedRiskModalDropout —— 训练时随机置零 text / meta 模态;
                                         同时拆开 fusion 内部抓 gate weights
  2. RiskOrdinalGateEntropyLoss      —— BCE + (CE+Dice) seg
                                         - lambda_gate * H(gate weights)
                                           (鼓励三模态权重均衡)
  3. mlm_mask_input_ids              —— 训练时对 input_ids 做 BERT-style 15%
                                         token masking
  4. add_meta_noise                   —— 训练时给 metadata 加 N(0, sigma) 噪声
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Tuple

import numpy as np
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
#  Model: 模态 dropout + gate 暴露
# ═══════════════════════════════════════════════════════════════════════════


class SwinV2SegGuidedRiskModalDropout(SwinV2SegGuidedRiskTrimodal):
    """baseline 模型 + 训练态模态 dropout + gate weights 暴露。

    forward 返回 (seg_logits, risk_logit, gate_weights),
    gate_weights shape (B, 3) 表示 (image, text, meta) 三模态门控权重。

    训练态行为 (model.training=True):
      - 以概率 modal_dropout_p 把整个 batch 的 text_feat 置零
        (同时把 attention_mask 置零,使 BERT 输出也是 0)
      - 以独立概率把整个 batch 的 meta_feat 置零
        (注意: 是 batch-level dropout —— 整个 batch 同时被 drop,
         不是 per-sample,这样不破坏 BalancedBatchSampler 的多样性)
    推理态: 不做任何 dropout。
    """

    def __init__(self, modal_dropout_p: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.modal_dropout_p = modal_dropout_p

    def forward(self, x, metadata=None, input_ids=None, attention_mask=None):
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        f2_proj = self.cls_proj(f2)

        # ── 训练态: 决定本 batch 是否 drop text / meta ──
        drop_text = (self.training
                     and torch.rand((), device=x.device).item() < self.modal_dropout_p)
        drop_meta = (self.training
                     and torch.rand((), device=x.device).item() < self.modal_dropout_p)

        # ── BERT (frozen) ──
        # 注意: 不要把 attention_mask 全置零, 否则 BERT 内部 softmax(全 mask)→NaN。
        # drop_text 的语义在 fusion 阶段实现(把 text_feat 置零即可)。
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
        if drop_text:
            text_feat = text_feat * 0.0  # 不影响梯度结构

        meta_feat = self.meta_encoder(metadata.float())
        if drop_meta:
            meta_feat = meta_feat * 0.0

        # ── 不再调用 self.fusion(...),拆开手算以便抓 gate weights ──
        concat = torch.cat([img_feat, text_feat, meta_feat], dim=1)
        gates = self.fusion.gate(concat)  # (B, 3)
        g_img = gates[:, 0:1]
        g_text = gates[:, 1:2]
        g_meta = gates[:, 2:3]
        fused_raw = (g_img * self.fusion.img_proj(img_feat)
                     + g_text * self.fusion.text_proj(text_feat)
                     + g_meta * self.fusion.meta_proj(meta_feat))
        fused = self.fusion.norm(fused_raw)

        risk_logit = self.risk_head(fused).squeeze(-1)
        return seg_logits, risk_logit, gates


# ═══════════════════════════════════════════════════════════════════════════
#  Loss: BCE + (CE+Dice) seg - lambda_gate * H(gate)
# ═══════════════════════════════════════════════════════════════════════════


class RiskOrdinalGateEntropyLoss(nn.Module):
    """L = lambda_ord * BCE-with-logits(risk_logit, ord_target)
            + (CE+Dice)(seg)
            - lambda_gate * H(gate)

    H(gate): 把 (g_img, g_text, g_meta) 当成离散概率分布 (先 normalize 到和为 1),
    然后取香农熵。熵越大 (即权重越均衡) 损失越小。
    """

    def __init__(self, ordinal_targets: torch.Tensor,
                 lambda_ord: float = 2.0,
                 lambda_gate: float = 0.05,
                 seg_ce_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("targets", ordinal_targets)
        self.lambda_ord = lambda_ord
        self.lambda_gate = lambda_gate
        self.seg_ce = nn.CrossEntropyLoss(weight=seg_ce_weight)
        self.seg_dice = DiceLoss()

    def forward(self, seg_logits, risk_logit, gates,
                seg_targets, cls_targets, has_mask):
        ord_target = self.targets[cls_targets].to(risk_logit.dtype)
        ord_loss = F.binary_cross_entropy_with_logits(risk_logit, ord_target)

        seg_loss = torch.tensor(0.0, device=risk_logit.device)
        if has_mask.any():
            idx = has_mask.nonzero(as_tuple=True)[0]
            seg_ce = self.seg_ce(seg_logits[idx], seg_targets[idx])
            seg_dice = self.seg_dice(seg_logits[idx], seg_targets[idx])
            seg_loss = seg_ce + seg_dice

        # 熵正则: 让三门控权重不要塌缩到单一模态
        # 强制 fp32 + clamp,避免 AMP/fp16 下 log(0) → NaN
        gates_f32 = gates.float()
        gate_sum = gates_f32.sum(dim=1, keepdim=True).clamp(min=1e-3)
        gate_p = (gates_f32 / gate_sum).clamp(min=1e-4, max=1.0)
        gate_entropy = -(gate_p * gate_p.log()).sum(dim=1).mean()
        # 熵越大 (越均衡), 损失越小; 所以用减号
        gate_term = -self.lambda_gate * gate_entropy.to(risk_logit.dtype)

        total = seg_loss + self.lambda_ord * ord_loss + gate_term
        return (total,
                seg_loss.item(),
                ord_loss.item(),
                float(gate_entropy.item()))


# ═══════════════════════════════════════════════════════════════════════════
#  Token masking: BERT-style 15% 随机 mask
# ═══════════════════════════════════════════════════════════════════════════


def mlm_mask_input_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                       mask_token_id: int,
                       cls_token_id: int = 101, sep_token_id: int = 102,
                       pad_token_id: int = 0,
                       mask_prob: float = 0.15) -> torch.Tensor:
    """对 input_ids 做 BERT-style 15% 随机 mask,返回新 input_ids。
    不修改 [CLS]/[SEP]/[PAD] 位置。

    用法 (训练 collate 之后, 上 device 之前):
        input_ids = mlm_mask_input_ids(input_ids, attention_mask,
                                        mask_token_id=tokenizer.mask_token_id)
    """
    if mask_prob <= 0:
        return input_ids
    # 哪些位置允许被 mask: 非 pad / 非 cls / 非 sep
    maskable = (input_ids != cls_token_id) & (input_ids != sep_token_id) \
               & (input_ids != pad_token_id) & (attention_mask != 0)
    rand = torch.rand_like(input_ids, dtype=torch.float32)
    do_mask = (rand < mask_prob) & maskable
    out = input_ids.clone()
    out[do_mask] = mask_token_id
    return out


def add_meta_noise(metadata: torch.Tensor, sigma: float = 0.05) -> torch.Tensor:
    """训练时对 metadata 加 N(0, sigma) 高斯噪声。"""
    if sigma <= 0:
        return metadata
    noise = torch.randn_like(metadata) * sigma
    return metadata + noise
