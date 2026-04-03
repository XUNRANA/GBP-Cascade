"""
Exp #9: Exp#4 + Supervised Contrastive Learning

核心思路: 在 Exp#4 基础上加 SupConLoss, 拉开 benign 和 no_tumor 的特征距离.
  - 模型在 seg-guided attention 之后额外输出 projection features
  - Loss = seg_loss + 2.0 * cls_loss + 0.5 * supcon_loss
  - SupCon 促使同类样本特征靠近, 异类特征远离

vs Exp#4:
  + Supervised Contrastive Loss (temperature=0.07)
  + Projection head (256→128→64)
  - 只训练 50 epoch
"""

import os
import sys
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v3 import (
    GBPDatasetSegCls4chWithMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chModel,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    adapt_model_to_4ch,
    META_FEATURE_NAMES,
    SupConLoss, SegClsLoss,
    set_seed, setup_logger, acquire_run_lock,
    build_class_weights, set_epoch_lrs, compute_seg_metrics,
    evaluate_v2, find_optimal_threshold_v2, evaluate_with_threshold_v2,
    _unpack_batch,
)
from sklearn.metrics import f1_score


# ─── Model with projection head ───

class SwinV2SegGuidedCls4chWithProj(SwinV2SegGuidedCls4chModel):
    """Exp#4 model + projection head for SupCon loss."""

    def __init__(self, proj_dim=64, **kwargs):
        super().__init__(**kwargs)
        # proj_head operates on the 256d cls_feat (before metadata fusion)
        self.proj_head = nn.Sequential(
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, proj_dim),
        )

    def forward(self, x, metadata=None, return_proj=False):
        features = self.encoder(x)
        f0, f1, f2, f3 = [self._to_bchw(f) for f in features]

        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)
        seg_logits = self.seg_final(d1)

        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]
        attn = F.interpolate(seg_prob, size=f2.shape[2:], mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)

        f2_proj = self.cls_proj(f2)
        cls_feat = (f2_proj * attn).sum(dim=(2, 3))  # (B, 256)

        proj_feat = self.proj_head(cls_feat) if return_proj else None

        if self.meta_encoder is not None and metadata is not None:
            meta_feat = self.meta_encoder(metadata.float())
            cls_in = torch.cat([cls_feat, meta_feat], dim=1)
        else:
            cls_in = cls_feat

        cls_logits = self.cls_mlp(cls_in)

        if return_proj:
            return seg_logits, cls_logits, proj_feat
        return seg_logits, cls_logits


# ─── Config ───

class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260402_task2_SwinV2Tiny_segcls_9"
    log_dir = os.path.join(project_root, "0402", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(META_FEATURE_NAMES)
    meta_hidden = 64
    meta_dropout = 0.2
    proj_dim = 64
    supcon_temperature = 0.07
    lambda_cls = 2.0
    lambda_con = 0.5

    batch_size = 8
    num_epochs = 50
    warmup_epochs = 5
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 2
    seed = 42
    use_amp = True

    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + SegAttn + Meta + SupConLoss"
    modification = (
        "Exp#4基础 + SupConLoss(temp=0.07, lambda=0.5) "
        "+ proj head(256→128→64) + 50ep"
    )


# ─── Training loop (handles 3-output model) ───

def train_one_epoch_supcon(model, dataloader, seg_cls_criterion, supcon_criterion,
                           lambda_con, optimizer, device, scaler, use_amp,
                           grad_clip=None, num_seg_classes=2):
    model.train()
    running_loss, running_seg, running_cls, running_con = 0.0, 0.0, 0.0, 0.0
    cls_correct, cls_total = 0, 0
    all_seg_dices = []

    for batch in dataloader:
        imgs, masks, metas, labels, has_masks = _unpack_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        if metas is not None:
            metas = metas.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                enabled=use_amp):
            seg_logits, cls_logits, proj_feat = model(imgs, metadata=metas, return_proj=True)
            seg_cls_loss, seg_l, cls_l = seg_cls_criterion(
                seg_logits, cls_logits, masks, labels, has_masks
            )
            con_loss = supcon_criterion(proj_feat, labels)
            loss = seg_cls_loss + lambda_con * con_loss

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_seg += seg_l * bs
        running_cls += cls_l * bs
        running_con += con_loss.item() * bs
        cls_correct += (cls_logits.argmax(dim=1) == labels).sum().item()
        cls_total += bs

        if has_masks.any():
            with torch.no_grad():
                mask_idx = has_masks.nonzero(as_tuple=True)[0]
                m = compute_seg_metrics(seg_logits[mask_idx], masks[mask_idx], num_seg_classes)
                all_seg_dices.append(m["lesion_Dice"])

    n = cls_total
    return {
        "loss": running_loss / n, "seg_loss": running_seg / n,
        "cls_loss": running_cls / n, "con_loss": running_con / n,
        "cls_acc": cls_correct / n,
        "seg_dice": np.mean(all_seg_dices) if all_seg_dices else 0.0,
    }


def main():
    cfg = Config()
    os.makedirs(cfg.log_dir, exist_ok=True)
    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_file, cfg.exp_name)
    logger.info("=" * 70)
    logger.info(f"实验: {cfg.exp_name}")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"SupCon: temp={cfg.supcon_temperature}, lambda_con={cfg.lambda_con}")
    logger.info(f"lambda_cls={cfg.lambda_cls}, epochs={cfg.num_epochs}")
    logger.info("=" * 70)

    # Data
    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)
    train_dataset = GBPDatasetSegCls4chWithMeta(
        cfg.train_excel, cfg.data_root, cfg.clinical_excel, cfg.json_feature_root,
        sync_transform=train_sync,
    )
    test_dataset = GBPDatasetSegCls4chWithMeta(
        cfg.test_excel, cfg.data_root, cfg.clinical_excel, cfg.json_feature_root,
        sync_transform=test_sync, meta_stats=train_dataset.meta_stats,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )
    logger.info(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")

    # Model
    model = SwinV2SegGuidedCls4chWithProj(
        proj_dim=cfg.proj_dim,
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        pretrained=True,
    ).to(cfg.device)
    logger.info(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    cls_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    seg_ce_weight = torch.tensor([cfg.seg_bg_weight, cfg.seg_lesion_weight],
                                 dtype=torch.float32, device=cfg.device)
    seg_cls_criterion = SegClsLoss(
        cls_weights=cls_weights, lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing, seg_ce_weight=seg_ce_weight,
    )
    supcon_criterion = SupConLoss(temperature=cfg.supcon_temperature)

    # Optimizer
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.")]
    optimizer = build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)
    scaler = torch.amp.GradScaler(device=cfg.device.type,
                                  enabled=(cfg.device.type == "cuda" and cfg.use_amp))

    best_f1, best_epoch = 0.0, 0

    logger.info("\n开始训练")
    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        m = train_one_epoch_supcon(
            model, train_loader, seg_cls_criterion, supcon_criterion,
            cfg.lambda_con, optimizer, cfg.device, scaler,
            use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
        )
        elapsed = time.time() - t0
        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"Loss: {m['loss']:.4f} (seg={m['seg_loss']:.4f}, cls={m['cls_loss']:.4f}, "
            f"con={m['con_loss']:.4f}) | Acc: {m['cls_acc']:.4f} | "
            f"Dice: {m['seg_dice']:.4f} | {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 50)
            # eval uses 2-output mode (return_proj=False is default)
            acc, prec, rec, f1, seg_iou, seg_dice = evaluate_v2(
                model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Test", num_seg_classes=cfg.num_seg_classes,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")
            logger.info("-" * 50)

    logger.info(f"\n训练完成! 最优: Epoch {best_epoch}, F1: {best_f1:.4f}")

    # Final eval with best weights
    model.load_state_dict(torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True))
    logger.info("=" * 70)
    evaluate_v2(model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Final Test", num_seg_classes=cfg.num_seg_classes)
    best_thresh, best_thresh_f1 = find_optimal_threshold_v2(model, test_loader, cfg.device)
    logger.info(f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f})")
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold_v2(model, test_loader, cfg.device, cfg.class_names, logger,
                                   threshold=best_thresh, phase="Final Test (最优阈值)")

    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)


if __name__ == "__main__":
    main()
