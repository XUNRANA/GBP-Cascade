"""
Exp #10: Exp#4 × 3 Seed Ensemble

核心思路: 用 3 个不同 seed 训练 Exp#4 架构, 每个 30 epochs,
  推理时 3 模型概率平均, 减少随机性, 稳定提升 F1.

训练: seed 42 → 30ep → save best → seed 123 → 30ep → save best → seed 456 → 30ep → save best
推理: 加载 3 个 best 模型 → 概率平均 → 分类

总训练时间约: 3 × ~8 min = ~25 min (在单 GPU 上串行)
"""

import os
import sys
import time
import shutil

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v3 import (
    GBPDatasetSegCls4chWithMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chModel,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    META_FEATURE_NAMES,
    SegClsLoss,
    set_seed, setup_logger, acquire_run_lock,
    build_class_weights, set_epoch_lrs,
    train_one_epoch_v2, evaluate_v2,
    _unpack_batch,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260402_task2_SwinV2Tiny_segcls_10"
    log_dir = os.path.join(project_root, "0402", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")

    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(META_FEATURE_NAMES)
    meta_hidden = 64
    meta_dropout = 0.2

    # Per-seed training
    batch_size = 8
    num_epochs = 30       # 每个 seed 只训 30 epoch
    warmup_epochs = 5
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 2
    use_amp = True

    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    # 3 seeds for ensemble
    seeds = [42, 123, 456]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + SegAttn + Meta (3-seed Ensemble)"
    modification = "Exp#4 × 3 seeds (42/123/456), 每 seed 30ep, 概率平均 ensemble"


def build_model(cfg):
    return SwinV2SegGuidedCls4chModel(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        pretrained=True,
    )


def build_dataloaders(cfg):
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
    return train_dataset, test_dataset, train_loader, test_loader


def train_single_seed(cfg, seed, train_dataset, train_loader, test_loader, logger):
    """Train one model with given seed, return path to best checkpoint."""
    set_seed(seed)
    weight_path = os.path.join(cfg.log_dir, f"seed{seed}_best.pth")

    model = build_model(cfg).to(cfg.device)
    cls_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    seg_ce_weight = torch.tensor([cfg.seg_bg_weight, cfg.seg_lesion_weight],
                                 dtype=torch.float32, device=cfg.device)
    criterion = SegClsLoss(
        cls_weights=cls_weights, lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing, seg_ce_weight=seg_ce_weight,
    )

    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.")]
    optimizer = build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)
    scaler = torch.amp.GradScaler(device=cfg.device.type,
                                  enabled=(cfg.device.type == "cuda" and cfg.use_amp))

    best_f1, best_epoch = 0.0, 0
    logger.info(f"\n--- Seed {seed}: 开始训练 ({cfg.num_epochs} epochs) ---")

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()
        m = train_one_epoch_v2(
            model, train_loader, criterion, optimizer, cfg.device, scaler,
            use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
        )
        elapsed = time.time() - t0
        logger.info(
            f"[Seed {seed}] Epoch [{epoch}/{cfg.num_epochs}] "
            f"Loss: {m['loss']:.4f} | Acc: {m['cls_acc']:.4f} | "
            f"Dice: {m['seg_dice']:.4f} | {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            acc, prec, rec, f1, seg_iou, seg_dice = evaluate_v2(
                model, test_loader, cfg.device, cfg.class_names, logger,
                phase=f"Test(seed{seed})", num_seg_classes=cfg.num_seg_classes,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), weight_path)
                logger.info(f"*** Seed {seed} 最优: F1={best_f1:.4f} @ Epoch {best_epoch} ***")

    logger.info(f"--- Seed {seed}: 完成, 最优 F1={best_f1:.4f} @ Epoch {best_epoch} ---\n")
    return weight_path, best_f1, best_epoch


def ensemble_evaluate(models, dataloader, device, class_names, logger, phase="Ensemble"):
    """Evaluate ensemble of models by averaging probabilities."""
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            imgs, masks, metas, labels, has_masks = _unpack_batch(batch)
            imgs = imgs.to(device, non_blocking=True)
            if metas is not None:
                metas = metas.to(device, non_blocking=True)

            batch_probs = []
            for model in models:
                model.eval()
                _, cls_logits = model(imgs, metadata=metas)
                probs = torch.softmax(cls_logits, dim=1).cpu()
                batch_probs.append(probs)

            # Average probabilities
            avg_probs = torch.stack(batch_probs, dim=0).mean(dim=0)  # (B, 2)
            all_probs.append(avg_probs)
            all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

    all_probs = torch.cat(all_probs, dim=0).numpy()  # (N, 2)
    all_labels = np.array(all_labels)

    # Default threshold
    all_preds = all_probs.argmax(axis=1)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(
        f"[{phase}] Acc: {acc:.4f} | P(macro): {precision:.4f} | "
        f"R(macro): {recall:.4f} | F1(macro): {f1:.4f}"
    )
    report = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")

    # Threshold optimization
    benign_probs = all_probs[:, 0]
    best_f1_t, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds_t = np.where(benign_probs >= thresh, 0, 1)
        f1_t = f1_score(all_labels, preds_t, average="macro", zero_division=0)
        if f1_t > best_f1_t:
            best_f1_t = f1_t
            best_thresh = thresh

    logger.info(f"[{phase}] 阈值优化: thresh={best_thresh:.3f}, F1={best_f1_t:.4f}")

    if abs(best_thresh - 0.5) > 0.01:
        preds_opt = np.where(benign_probs >= best_thresh, 0, 1)
        acc_opt = accuracy_score(all_labels, preds_opt)
        f1_opt = f1_score(all_labels, preds_opt, average="macro", zero_division=0)
        report_opt = classification_report(
            all_labels, preds_opt, target_names=class_names, digits=4, zero_division=0,
        )
        logger.info(
            f"[{phase} 最优阈值] Acc: {acc_opt:.4f} | F1: {f1_opt:.4f}\n{report_opt}"
        )

    return f1, best_f1_t, best_thresh


def main():
    cfg = Config()
    os.makedirs(cfg.log_dir, exist_ok=True)
    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    logger = setup_logger(cfg.log_file, cfg.exp_name)
    logger.info("=" * 70)
    logger.info(f"实验: {cfg.exp_name}")
    logger.info(f"策略: 3-Seed Ensemble (seeds={cfg.seeds})")
    logger.info(f"每 seed: {cfg.num_epochs} epochs")
    logger.info("=" * 70)

    # Build dataloaders (shared across seeds — need fresh loader per seed for shuffling)
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
    logger.info(f"训练集: {len(train_dataset)} | 测试集: {len(test_dataset)}")

    # Train 3 seeds sequentially
    weight_paths = []
    seed_results = []
    for seed in cfg.seeds:
        # Rebuild train_loader with new seed for proper shuffling
        set_seed(seed)
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
        wp, bf1, bep = train_single_seed(cfg, seed, train_dataset, train_loader, test_loader, logger)
        weight_paths.append(wp)
        seed_results.append((seed, bf1, bep))

    # Summary of individual seeds
    logger.info("\n" + "=" * 70)
    logger.info("单模型结果汇总:")
    for seed, bf1, bep in seed_results:
        logger.info(f"  Seed {seed}: F1={bf1:.4f} @ Epoch {bep}")
    logger.info("=" * 70)

    # Ensemble evaluation
    logger.info("\n" + "=" * 70)
    logger.info("Ensemble 评估 (3模型概率平均)")
    logger.info("=" * 70)

    models = []
    for wp in weight_paths:
        m = build_model(cfg).to(cfg.device)
        m.load_state_dict(torch.load(wp, map_location=cfg.device, weights_only=True))
        m.eval()
        models.append(m)

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )
    ens_f1, ens_f1_thresh, ens_thresh = ensemble_evaluate(
        models, test_loader, cfg.device, cfg.class_names, logger,
        phase="Ensemble (3-seed)",
    )

    logger.info("\n" + "=" * 70)
    logger.info(f"最终结果: Ensemble F1={ens_f1:.4f}, 阈值优化 F1={ens_f1_thresh:.4f}")
    avg_single = np.mean([r[1] for r in seed_results])
    logger.info(f"单模型平均 F1={avg_single:.4f}, Ensemble 提升={ens_f1 - avg_single:+.4f}")
    logger.info("=" * 70)

    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)


if __name__ == "__main__":
    main()
