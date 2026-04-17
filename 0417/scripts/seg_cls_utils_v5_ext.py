"""
seg_cls_utils_v5_ext.py — Exp#19 系列新增组件

导入原工具库中全部内容，再新增：
  - SubsetDataset / stratified_dev_split / save|load_dev_split_json
  - PAUCSurrogateLoss
  - SAM optimizer
  - TemperatureScaler + predict_logits_text
  - train_one_epoch_text_sam (SAM 无 Mixup)
  - train_one_epoch_text_sam_mixup (SAM + fused-space between-class Mixup)
  - SwinV2SegGuidedCls4chTrimodalMixup (subclass, exposes forward_with_fused)
  - between_class_mixup_fused / soft_cross_entropy
  - tta_predict_probs / apply_tta_op
  - mc_dropout_predict
  - simple_ensemble / rank_ensemble
  - compute_pauc_at_fpr
  - build_swa_model / update_swa_bn
"""

import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

# 注入 0408/scripts 路径以导入 v5 工具库
_V5_SCRIPTS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "0408", "scripts",
)
sys.path.insert(0, os.path.abspath(_V5_SCRIPTS))

from seg_cls_utils_v5 import (  # noqa: F401  (re-export everything)
    # --- 数据 ---
    GBPDatasetSegCls4chWithTextMeta,
    SegCls4chSyncTransform,
    seg_cls_text_collate_fn,
    load_text_bert_dict,
    EXT_CLINICAL_FEATURE_NAMES,
    # --- 模型 ---
    SwinV2SegGuidedCls4chTrimodal,
    # --- 损失 ---
    SegClsLoss,
    # --- 优化 ---
    build_optimizer_with_diff_lr,
    set_epoch_lrs,
    # --- 训练 ---
    train_one_epoch_text,
    evaluate_text,
    find_optimal_threshold_text,
    evaluate_with_threshold_text,
    predict_probs_text,
    find_constrained_threshold_text,
    analyze_high_confidence_positive,
    compute_binary_reliability_stats,
    save_reliability_stats_csv,
    save_reliability_diagram,
    # --- 工具 ---
    set_seed,
    setup_logger,
    acquire_run_lock,
    _unpack_text_batch,
    compute_seg_metrics,
)


# ═══════════════════════════════════════════════════════════════════════
#  1. Dev Split 工具
# ═══════════════════════════════════════════════════════════════════════

def stratified_dev_split(df, dev_frac=0.10, label_col="label", seed=42):
    """分层抽取 dev 集。返回 (train_indices, dev_indices)，均为 numpy int 数组。"""
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=dev_frac, random_state=seed)
    train_idx, dev_idx = next(sss.split(df, df[label_col]))
    return train_idx.astype(np.int64), dev_idx.astype(np.int64)


def save_dev_split_json(train_idx, dev_idx, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"train": train_idx.tolist(), "dev": dev_idx.tolist()}, f)


def load_dev_split_json(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return np.array(d["train"], dtype=np.int64), np.array(d["dev"], dtype=np.int64)


class SubsetDataset(Dataset):
    """将 GBPDatasetSegCls4chWithTextMeta 按下标子集包装。
    暴露 .df, .meta_stats, .meta_dim, .tokenizer 保持与 build_train_loader_with_ratio 兼容。
    """

    def __init__(self, base_dataset, indices):
        self.base = base_dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.df = base_dataset.df.iloc[self.indices].reset_index(drop=True)
        self.meta_stats = base_dataset.meta_stats
        self.meta_dim = base_dataset.meta_dim
        self.tokenizer = base_dataset.tokenizer

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return self.base[int(self.indices[i])]


# ═══════════════════════════════════════════════════════════════════════
#  2. pAUC Surrogate Loss
# ═══════════════════════════════════════════════════════════════════════

class PAUCSurrogateLoss(nn.Module):
    """Top-K pairwise hinge on P(benign).
    Penalizes 'hard benign samples being ranked below no_tumor'.
    """

    def __init__(self, K=4, margin=0.1):
        super().__init__()
        self.K = K
        self.margin = margin

    def forward(self, cls_logits, labels):
        """
        cls_logits: (B, 2)  — raw logits, class 0=benign, class 1=no_tumor
        labels:     (B,)    — 0=benign, 1=no_tumor
        """
        p_benign = cls_logits.softmax(-1)[:, 0]
        b_mask = labels == 0
        nt_mask = labels == 1

        if b_mask.sum() == 0 or nt_mask.sum() == 0:
            return cls_logits.new_zeros(())

        p_b = p_benign[b_mask]
        p_nt = p_benign[nt_mask]
        K = min(self.K, p_b.numel())
        hard_b = p_b.topk(K, largest=False).values  # K 个最难 benign（prob 最低）

        # (K, |nt|) hinge: margin - (p_b_i - p_nt_j)
        diff = self.margin - (hard_b[:, None] - p_nt[None, :])
        return diff.clamp_min(0).mean()


# ═══════════════════════════════════════════════════════════════════════
#  3. SAM Optimizer
# ═══════════════════════════════════════════════════════════════════════

class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization，包装任意 base optimizer。
    使用方式：
        sam = SAM(param_groups, AdamW, rho=0.05, lr=lr, weight_decay=wd)
        # first forward
        loss.backward()
        sam.first_step(zero_grad=True)
        # second forward (perturbed weights)
        loss2.backward()
        sam.second_step(zero_grad=True)

    注意：与 GradScaler 不兼容，使用时关闭 scaler（only autocast）。
    set_epoch_lrs 可直接作用于 sam 因为 self.param_groups 指向 base_optimizer.param_groups。
    """

    def __init__(self, params, base_optimizer_cls, rho=0.05, **kwargs):
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
        # 透传同一引用，set_epoch_lrs 可直接作用
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        norms = [
            p.grad.norm(2).to(p.device)
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not norms:
            return torch.tensor(1e-12)
        return torch.stack(norms).norm(2)

    def step(self, closure=None):
        raise NotImplementedError("SAM: use first_step() / second_step() instead of step().")

    def zero_grad(self, set_to_none=True):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)


# ═══════════════════════════════════════════════════════════════════════
#  4. Temperature Scaler + predict_logits_text
# ═══════════════════════════════════════════════════════════════════════

class TemperatureScaler(nn.Module):
    """单参数温度缩放。在 dev set 上用 LBFGS 拟合，推理时 logits / T。"""

    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.T.clamp(min=0.1)  # 防止 T→0

    def fit(self, logits, labels, lr=0.01, max_iter=500):
        """
        logits: (N, 2) CPU tensor  （raw logits，不是 softmax）
        labels: (N,)  CPU long tensor
        返回拟合后的 T 值 (float)。
        """
        self.T.data.fill_(1.5)  # warm start，大多数模型过拟合置信度 >1
        opt = torch.optim.LBFGS([self.T], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / self.T.clamp(min=0.1), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return float(self.T.item())


def predict_logits_text(model, dataloader, device):
    """返回 (logits_tensor (N,2), labels_tensor (N,)) on CPU，用于温度缩放。"""
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            imgs, _, metas, iids, amask, labels, _ = _unpack_text_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)
            iids = iids.to(device, non_blocking=True)
            amask = amask.to(device, non_blocking=True)
            _, cls_logits = model(
                imgs, metadata=metas, input_ids=iids, attention_mask=amask
            )
            all_logits.append(cls_logits.cpu())
            all_labels.append(labels.cpu())
    return torch.cat(all_logits), torch.cat(all_labels)


# ═══════════════════════════════════════════════════════════════════════
#  5. train_one_epoch_text_sam（SAM，无 Mixup）
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch_text_sam(
    model, dataloader, criterion, pauc_loss_fn, pauc_coeff,
    sam_optimizer, device, use_amp=True, grad_clip=None, num_seg_classes=2,
):
    """SAM double-forward 训练。无 GradScaler（与 SAM 不兼容），仅 autocast。
    pauc_loss_fn: PAUCSurrogateLoss 实例；pauc_coeff=0 时不加。
    """
    model.train()
    running_loss = running_seg = running_cls = 0.0
    cls_correct = cls_total = 0
    all_ious, all_dices = [], []

    for batch in dataloader:
        imgs, masks, metas, iids, amask, labels, has_masks = _unpack_text_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        iids = iids.to(device, non_blocking=True)
        amask = amask.to(device, non_blocking=True)

        # ── SAM first forward ──────────────────────────
        sam_optimizer.zero_grad()
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            seg_l, cls_l = model(imgs, metadata=metas, input_ids=iids, attention_mask=amask)
            loss, seg_v, cls_v = criterion(seg_l, cls_l, masks, labels, has_masks)
            if pauc_coeff > 0:
                loss = loss + pauc_coeff * pauc_loss_fn(cls_l, labels)

        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        sam_optimizer.first_step(zero_grad=True)

        # ── SAM second forward ─────────────────────────
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            seg_l2, cls_l2 = model(imgs, metadata=metas, input_ids=iids, attention_mask=amask)
            loss2, _, _ = criterion(seg_l2, cls_l2, masks, labels, has_masks)
            if pauc_coeff > 0:
                loss2 = loss2 + pauc_coeff * pauc_loss_fn(cls_l2, labels)

        loss2.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        sam_optimizer.second_step(zero_grad=True)

        # ── Metrics ────────────────────────────────────
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_seg += seg_v * bs
        running_cls += cls_v * bs
        cls_correct += (cls_l.argmax(dim=1) == labels).sum().item()
        cls_total += bs

        if has_masks.any():
            with torch.no_grad():
                idx = has_masks.nonzero(as_tuple=True)[0]
                m = compute_seg_metrics(seg_l[idx], masks[idx], num_seg_classes)
                all_ious.append(m["lesion_IoU"])
                all_dices.append(m["lesion_Dice"])

    n = max(cls_total, 1)
    return {
        "loss": running_loss / n,
        "seg_loss": running_seg / n,
        "cls_loss": running_cls / n,
        "cls_acc": cls_correct / n,
        "seg_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "seg_dice": float(np.mean(all_dices)) if all_dices else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  6. SwinV2SegGuidedCls4chTrimodalMixup + Mixup 工具
# ═══════════════════════════════════════════════════════════════════════

class SwinV2SegGuidedCls4chTrimodalMixup(SwinV2SegGuidedCls4chTrimodal):
    """继承父模型，新增 forward_with_fused() 用于 fused-space between-class Mixup。"""

    def forward_with_fused(self, x, metadata=None, input_ids=None, attention_mask=None):
        """与 forward() 完全相同，但额外返回 fused 256D 向量（用于 Mixup）。
        返回: (seg_logits, cls_logits, fused)
        """
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
        attn = F.interpolate(
            seg_prob, size=f2_enhanced.shape[2:], mode="bilinear", align_corners=False
        )
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)
        img_feat = (f2_enhanced * attn).sum(dim=(2, 3))

        text_feat = self.text_cls_proj(text_cls)
        meta_feat = self.meta_encoder(metadata.float())

        fused = self.fusion(img_feat, text_feat, meta_feat)
        cls_logits = self.cls_mlp(fused)
        return seg_logits, cls_logits, fused


def between_class_mixup_fused(fused, labels, alpha=2.0, beta_param=0.5):
    """在 fused(256D) 空间做 between-class Mixup。
    每个样本随机配对一个对立类样本，λ ~ Beta(alpha, beta_param)（偏向 1 以避免过强扰动）。
    返回: (fused_mix (B,256), soft_labels (B,2))
    """
    B = fused.size(0)
    lam = torch.distributions.Beta(
        torch.tensor(alpha), torch.tensor(beta_param)
    ).sample((B,)).to(fused.device).view(-1, 1)  # (B, 1)

    j = torch.zeros(B, dtype=torch.long, device=fused.device)
    for i in range(B):
        opp = (labels != labels[i]).nonzero(as_tuple=True)[0]
        if len(opp) == 0:
            j[i] = i
        else:
            j[i] = opp[torch.randint(len(opp), (1,), device=opp.device)]

    fused_mix = lam * fused + (1.0 - lam) * fused[j]

    one_hot = F.one_hot(labels, num_classes=2).float()  # (B, 2)
    soft_labels = lam * one_hot + (1.0 - lam) * one_hot[j]
    return fused_mix, soft_labels


def soft_cross_entropy(logits, soft_labels):
    """软标签交叉熵。logits: (B,C), soft_labels: (B,C)。"""
    return -(soft_labels * F.log_softmax(logits, dim=-1)).sum(dim=-1).mean()


# ═══════════════════════════════════════════════════════════════════════
#  7. train_one_epoch_text_sam_mixup（SAM + Fused Mixup，Stage 2）
# ═══════════════════════════════════════════════════════════════════════

def train_one_epoch_text_sam_mixup(
    model, dataloader, criterion, pauc_loss_fn, pauc_coeff,
    sam_optimizer, device, lambda_cls=2.0, use_amp=True, grad_clip=None,
    mixup_alpha=2.0, mixup_beta=0.5, num_seg_classes=2,
):
    """SAM double-forward + between-class fused Mixup（Stage 2 专用）。
    model 必须是 SwinV2SegGuidedCls4chTrimodalMixup。
    Seg loss 用原始样本，cls loss 用 Mixup 软标签，pAUC loss 用原始硬标签。
    """
    model.train()
    running_loss = running_seg = running_cls = 0.0
    cls_correct = cls_total = 0
    all_ious, all_dices = [], []

    for batch in dataloader:
        imgs, masks, metas, iids, amask, labels, has_masks = _unpack_text_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        iids = iids.to(device, non_blocking=True)
        amask = amask.to(device, non_blocking=True)

        # 预生成 Mixup 参数（两次 forward 保持一致）
        B = imgs.size(0)
        lam = torch.distributions.Beta(
            torch.tensor(float(mixup_alpha)), torch.tensor(float(mixup_beta))
        ).sample((B,)).to(device).view(-1, 1)
        j_idx = torch.zeros(B, dtype=torch.long, device=device)
        for i in range(B):
            opp = (labels != labels[i]).nonzero(as_tuple=True)[0]
            j_idx[i] = opp[torch.randint(len(opp), (1,))] if len(opp) > 0 else i
        one_hot = F.one_hot(labels, num_classes=2).float()
        soft_lbl = lam * one_hot + (1.0 - lam) * one_hot[j_idx]  # (B, 2)

        def _compute_loss(seg_l, fused, masks, labels, has_masks):
            # Seg loss（原始样本，不 Mixup）
            if has_masks.any():
                idx = has_masks.nonzero(as_tuple=True)[0]
                seg_loss_t = (
                    criterion.seg_ce(seg_l[idx], masks[idx])
                    + criterion.seg_dice(seg_l[idx], masks[idx])
                )
            else:
                seg_loss_t = seg_l.new_zeros(())
            # Cls loss（Mixup 软标签）
            fused_mix = lam * fused + (1.0 - lam) * fused[j_idx]
            cls_mix = model.cls_mlp(fused_mix)
            cls_loss_mix = soft_cross_entropy(cls_mix, soft_lbl)
            total = seg_loss_t + lambda_cls * cls_loss_mix
            # pAUC loss（原始硬标签）
            if pauc_coeff > 0:
                cls_hard = model.cls_mlp(fused)
                total = total + pauc_coeff * pauc_loss_fn(cls_hard, labels)
            return total, float(seg_loss_t.item() if seg_loss_t.requires_grad else 0.0), float(cls_loss_mix.item())

        # ── SAM first forward ──
        sam_optimizer.zero_grad()
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            seg_l, cls_l, fused = model.forward_with_fused(
                imgs, metadata=metas, input_ids=iids, attention_mask=amask
            )
            loss, seg_v, cls_v = _compute_loss(seg_l, fused, masks, labels, has_masks)

        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        sam_optimizer.first_step(zero_grad=True)

        # ── SAM second forward ──
        with torch.amp.autocast("cuda" if device.type == "cuda" else "cpu", enabled=use_amp):
            seg_l2, _, fused2 = model.forward_with_fused(
                imgs, metadata=metas, input_ids=iids, attention_mask=amask
            )
            loss2, _, _ = _compute_loss(seg_l2, fused2, masks, labels, has_masks)

        loss2.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        sam_optimizer.second_step(zero_grad=True)

        # ── Metrics ──
        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_seg += seg_v * bs
        running_cls += cls_v * bs
        # 用原始 cls_l 计算准确率（不 Mixup）
        cls_correct += (cls_l.argmax(dim=1) == labels).sum().item()
        cls_total += bs

        if has_masks.any():
            with torch.no_grad():
                idx = has_masks.nonzero(as_tuple=True)[0]
                m = compute_seg_metrics(seg_l[idx], masks[idx], num_seg_classes)
                all_ious.append(m["lesion_IoU"])
                all_dices.append(m["lesion_Dice"])

    n = max(cls_total, 1)
    return {
        "loss": running_loss / n,
        "seg_loss": running_seg / n,
        "cls_loss": running_cls / n,
        "cls_acc": cls_correct / n,
        "seg_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "seg_dice": float(np.mean(all_dices)) if all_dices else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  8. TTA 推理
# ═══════════════════════════════════════════════════════════════════════

def apply_tta_op(imgs, op):
    """对 (B, 4, H, W) tensor 应用 TTA 变换（就地不修改原 tensor）。"""
    if op == "orig":
        return imgs
    if op == "hflip":
        return imgs.flip(-1)
    if op == "rot5":
        return TF.rotate(imgs, 5, interpolation=TF.InterpolationMode.BILINEAR)
    if op == "rot_neg5":
        return TF.rotate(imgs, -5, interpolation=TF.InterpolationMode.BILINEAR)
    if op == "scale95":
        B, C, H, W = imgs.shape
        h95, w95 = max(1, int(H * 0.95)), max(1, int(W * 0.95))
        cropped = TF.center_crop(imgs, [h95, w95])
        return TF.resize(cropped, [H, W], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
    return imgs


def tta_predict_probs(
    model, test_loader, device,
    tta_ops=("orig", "hflip", "rot5", "rot_neg5", "scale95"),
):
    """对 test_loader 做多组 TTA，返回 (mean_prob_benign, mean_prob_no_tumor, labels)。
    test_loader 需要可重复迭代（即 dataset 确定、不 shuffle）。
    """
    model.eval()
    all_op_probs = []

    for op in tta_ops:
        probs_b_op, probs_nt_op, labels_op = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                imgs, _, metas, iids, amask, labels, _ = _unpack_text_batch(batch)
                imgs = imgs.to(device, non_blocking=True)
                imgs_aug = apply_tta_op(imgs, op)
                metas = metas.to(device, non_blocking=True)
                iids = iids.to(device, non_blocking=True)
                amask = amask.to(device, non_blocking=True)

                _, cls_logits = model(
                    imgs_aug, metadata=metas, input_ids=iids, attention_mask=amask
                )
                probs = cls_logits.softmax(-1).cpu().numpy()
                probs_b_op.extend(probs[:, 0])
                probs_nt_op.extend(probs[:, 1])
                labels_op.extend(labels.numpy())

        all_op_probs.append(np.array(probs_b_op, dtype=np.float32))

    # 只需返回最终平均，labels 从最后一次 op 取（所有 op 相同）
    mean_pb = np.mean(np.stack(all_op_probs, axis=0), axis=0)
    labels_np = np.array(labels_op, dtype=np.int64)
    return mean_pb, 1.0 - mean_pb, labels_np


def mc_dropout_predict(model, test_loader, device, T=30):
    """MC Dropout 推理。对 nn.Dropout 层保持 train() 其余保持 eval()。
    返回 (mean_prob_benign (N,), std_prob_benign (N,), labels (N,))。
    """
    # 仅对 Dropout 启用 train
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    all_probs = []  # T × (N,) list
    labels_all = None

    for _ in range(T):
        probs_b, labs = [], []
        with torch.no_grad():
            for batch in test_loader:
                imgs, _, metas, iids, amask, labels, _ = _unpack_text_batch(batch)
                imgs = imgs.to(device, non_blocking=True)
                metas = metas.to(device, non_blocking=True)
                iids = iids.to(device, non_blocking=True)
                amask = amask.to(device, non_blocking=True)

                _, cls_logits = model(
                    imgs, metadata=metas, input_ids=iids, attention_mask=amask
                )
                p = cls_logits.softmax(-1).cpu().numpy()[:, 0]
                probs_b.extend(p)
                labs.extend(labels.numpy())
        all_probs.append(np.array(probs_b, dtype=np.float32))
        if labels_all is None:
            labels_all = np.array(labs, dtype=np.int64)

    probs_stack = np.stack(all_probs, axis=0)  # (T, N)
    mean_pb = probs_stack.mean(axis=0)
    std_pb = probs_stack.std(axis=0)

    # 恢复 eval
    model.eval()
    return mean_pb, std_pb, labels_all


# ═══════════════════════════════════════════════════════════════════════
#  9. Ensemble 工具
# ═══════════════════════════════════════════════════════════════════════

def simple_ensemble(list_of_probs_benign):
    """简单平均。list_of_probs_benign: list of (N,) arrays。"""
    return np.mean(np.stack(list_of_probs_benign, axis=0), axis=0)


def rank_ensemble(list_of_probs_benign):
    """排名平均。每个模型内部排名后归一化，再平均。"""
    ranks = []
    for p in list_of_probs_benign:
        r = np.argsort(np.argsort(p)).astype(np.float32)
        r /= max(len(p) - 1, 1)
        ranks.append(r)
    return np.mean(np.stack(ranks, axis=0), axis=0)


def compute_pauc_at_fpr(labels, probs_benign, max_fpr=0.05):
    """计算 pAUC@FPR≤max_fpr（正类=benign，score=P(benign)）。"""
    from sklearn.metrics import roc_auc_score
    try:
        pos = (labels == 0).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            return 0.0
        return float(roc_auc_score(pos, probs_benign, max_fpr=max_fpr))
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════
#  10. SWA 工具
# ═══════════════════════════════════════════════════════════════════════

def build_swa_model(model):
    from torch.optim.swa_utils import AveragedModel
    return AveragedModel(model)


def update_swa_bn(swa_model, train_loader, device):
    from torch.optim.swa_utils import update_bn
    update_bn(train_loader, swa_model, device=device)


# ═══════════════════════════════════════════════════════════════════════
#  11. Post-training 工具：保存 probs / thresholds
# ═══════════════════════════════════════════════════════════════════════

def save_probs_csv(path, labels, prob_benign, prob_no_tumor):
    import pandas as pd
    pd.DataFrame({
        "label": labels.astype(int),
        "prob_benign": prob_benign.astype(np.float32),
        "prob_no_tumor": prob_no_tumor.astype(np.float32),
    }).to_csv(path, index=False, encoding="utf-8")


def compute_and_save_thresholds(probs_benign, labels, log_dir, exp_name, T_val,
                                 miss_rates=(0.02, 0.05, 0.10)):
    """在给定概率上搜索各 miss_rate 对应阈值，保存 JSON。"""
    result = {"T": float(T_val)}
    for mr in miss_rates:
        policy = find_constrained_threshold_text(probs_benign, labels, max_benign_miss_rate=mr)
        key = f"miss_{mr:.2f}".replace(".", "_")
        result[key] = {
            "threshold": float(policy["threshold"]),
            "benign_recall": float(policy["benign_recall"]),
            "no_tumor_hits": int(policy["selected_no_tumor"]),
            "no_tumor_recall": float(policy["no_tumor_recall"]),
            "constraint_satisfied": bool(policy["constraint_satisfied"]),
        }
    thr_path = os.path.join(log_dir, f"{exp_name}_thresholds.json")
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result, thr_path


def log_threshold_results(logger, result, phase="Test (calibrated)"):
    logger.info(f"\n{'='*60}")
    logger.info(f"[{phase}] 临床阈值结果 (T={result['T']:.4f})")
    logger.info(f"{'='*60}")
    for key, val in result.items():
        if key == "T":
            continue
        mr_str = key.replace("miss_0_", "miss_rate=0.").replace("miss_", "miss_rate=0.")
        logger.info(
            f"  {mr_str}: thr={val['threshold']:.3f} | "
            f"benign召回={val['benign_recall']:.2%} | "
            f"no_tumor命中={val['no_tumor_hits']}/394 | "
            f"no_tumor召回={val['no_tumor_recall']:.2%} | "
            f"约束满足={val['constraint_satisfied']}"
        )
