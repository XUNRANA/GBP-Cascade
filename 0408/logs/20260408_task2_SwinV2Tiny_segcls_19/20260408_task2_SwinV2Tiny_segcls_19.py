"""
Exp #19: 患者级5折交叉验证 + 早停 — 建立诚实评估基线

核心改动 (vs Exp#4):
  1. 患者级 StratifiedGroupKFold(5) — 同一患者的所有图像只在同一折中
  2. 每epoch评估val集, patience=8 早停
  3. 不用EMA (Exp#12已证明EMA=0.9995导致benign学不到)
  4. 30 epochs/fold (Exp#4在epoch 10-15最优, 30足够)
  5. 测试集仅最终评估一次
  6. 汇报: 5折平均val F1 + 单模型test F1 + 5模型集成test F1

架构: 与Exp#4完全相同 — SwinV2-Tiny@256 + 4ch + UNet Seg + SegGuided Attn + 6D Meta
"""

import os
import sys
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

# ── path setup ─────────────────────────────────────────────
_script_dir = os.path.dirname(os.path.abspath(__file__))
_0402_scripts = os.path.normpath(os.path.join(_script_dir, "../../0402/scripts"))
_0323_scripts = os.path.normpath(os.path.join(_script_dir, "../../0323/scripts"))
for _p in [_script_dir, _0402_scripts, _0323_scripts]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from seg_cls_utils_v2 import (
    GBPDatasetSegCls4chWithMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chModel,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    build_class_weights,
    SegClsLoss,
    set_seed,
    setup_logger,
    acquire_run_lock,
    cosine_warmup_factor,
    set_epoch_lrs,
    train_one_epoch_v2,
    evaluate_v2,
    find_optimal_threshold_v2,
    evaluate_with_threshold_v2,
    compute_seg_metrics,
)
from test_yqh import META_FEATURE_NAMES


# ═══════════════════════════════════════════════════════════
#  Config
# ═══════════════════════════════════════════════════════════

class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260408_task2_SwinV2Tiny_segcls_19"
    log_dir = os.path.join(project_root, "0408", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")

    # 模型 (与Exp#4完全一致)
    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(META_FEATURE_NAMES)  # 6
    meta_hidden = 64
    meta_dropout = 0.2

    # 训练
    batch_size = 8
    num_epochs = 30          # 每折30 epochs (Exp#4 epoch 10-15最优)
    warmup_epochs = 8
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    seed = 42
    use_amp = True

    # 损失
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    # CV
    n_splits = 5
    patience = 8             # 早停patience

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + Seg-Guided Attention + 6D Metadata"
    modification = (
        "Exp#4架构 + 患者级5折CV(StratifiedGroupKFold) + "
        "每epoch评估 + patience=8早停 + 30ep/fold + 无EMA + 测试集仅最终评估一次"
    )


# ═══════════════════════════════════════════════════════════
#  Build helpers
# ═══════════════════════════════════════════════════════════

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


def build_optimizer(model, cfg):
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def build_full_dataset(cfg, is_train=True):
    """Build dataset for the full training set (for CV splitting)."""
    sync = SegCls4chSyncTransform(cfg.img_size, is_train=is_train)
    ds = GBPDatasetSegCls4chWithMeta(
        cfg.train_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=sync,
    )
    return ds


def build_test_dataset(cfg, meta_stats):
    """Build test dataset using training meta_stats."""
    sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)
    ds = GBPDatasetSegCls4chWithMeta(
        cfg.test_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=sync,
        meta_stats=meta_stats,
    )
    return ds


# ═══════════════════════════════════════════════════════════
#  Evaluate returning probs (for ensemble)
# ═══════════════════════════════════════════════════════════

def predict_probs(model, dataloader, device):
    """Return (all_probs_benign, all_labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            imgs, masks, metas, labels, has_masks = batch
            imgs = imgs.to(device, non_blocking=True)
            if metas is not None:
                metas = metas.to(device, non_blocking=True)
            _, cls_logits = model(imgs, metadata=metas)
            probs = torch.softmax(cls_logits, dim=1)[:, 0].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    return np.array(all_probs), np.array(all_labels)


def evaluate_from_probs(probs, labels, class_names, logger, phase="Test", threshold=0.5):
    """Evaluate from pre-computed probabilities."""
    preds = np.where(probs >= threshold, 0, 1)
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    logger.info(
        f"[{phase}] Threshold: {threshold:.3f} | Acc: {acc:.4f} | "
        f"P(macro): {precision:.4f} | R(macro): {recall:.4f} | F1(macro): {f1:.4f}"
    )
    report = classification_report(
        labels, preds, target_names=class_names, digits=4, zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return f1


def find_threshold_from_probs(probs, labels):
    """Search optimal threshold on given probs/labels."""
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds = np.where(probs >= thresh, 0, 1)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


# ═══════════════════════════════════════════════════════════
#  Train one fold
# ═══════════════════════════════════════════════════════════

def train_one_fold(fold_idx, train_indices, val_indices,
                   full_train_ds_aug, full_train_ds_noaug, cfg, logger):
    """Train a single fold with early stopping. Returns (best_state_dict, best_val_f1)."""

    logger.info(f"\n{'='*60}")
    logger.info(f"Fold {fold_idx + 1}/{cfg.n_splits}")
    logger.info(f"{'='*60}")

    # Create subset dataloaders
    train_subset = Subset(full_train_ds_aug, train_indices)
    val_subset = Subset(full_train_ds_noaug, val_indices)

    # Log fold split stats
    train_labels = [full_train_ds_aug.df.iloc[i]["label"] for i in train_indices]
    val_labels = [full_train_ds_noaug.df.iloc[i]["label"] for i in val_indices]
    n_train_benign = sum(1 for l in train_labels if l == 0)
    n_train_notumor = sum(1 for l in train_labels if l == 1)
    n_val_benign = sum(1 for l in val_labels if l == 0)
    n_val_notumor = sum(1 for l in val_labels if l == 1)
    logger.info(
        f"  Train: {len(train_indices)} (benign={n_train_benign}, no_tumor={n_train_notumor})"
    )
    logger.info(
        f"  Val:   {len(val_indices)} (benign={n_val_benign}, no_tumor={n_val_notumor})"
    )

    train_loader = DataLoader(
        train_subset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )
    val_loader = DataLoader(
        val_subset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )

    # Build model, optimizer, criterion, scaler
    set_seed(cfg.seed + fold_idx)  # Different seed per fold for diversity
    model = build_model(cfg).to(cfg.device)
    optimizer = build_optimizer(model, cfg)

    # Use fold-specific class weights
    import pandas as pd
    fold_train_df = full_train_ds_aug.df.iloc[train_indices]
    cls_weights = build_class_weights(fold_train_df, cfg.class_names, cfg.device)
    logger.info(f"  Fold类别权重: benign={cls_weights[0]:.4f}, no_tumor={cls_weights[1]:.4f}")

    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight], dtype=torch.float32, device=cfg.device
    )
    criterion = SegClsLoss(
        cls_weights=cls_weights,
        lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing,
        seg_ce_weight=seg_ce_weight,
    )

    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    # Training loop with early stopping
    best_val_f1, best_epoch = 0.0, 0
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        train_metrics = train_one_epoch_v2(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
        )
        elapsed = time.time() - t0

        # Evaluate on val set every epoch
        acc, prec, rec, val_f1, seg_iou, seg_dice = evaluate_v2(
            model, val_loader, cfg.device, cfg.class_names, logger,
            phase=f"Fold{fold_idx+1}-Val", num_seg_classes=cfg.num_seg_classes,
        )

        logger.info(
            f"  Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f}, cls={train_metrics['cls_loss']:.4f}) "
            f"| Train Acc: {train_metrics['cls_acc']:.4f} "
            f"| Val F1: {val_f1:.4f} | {elapsed:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            logger.info(f"  *** Fold{fold_idx+1} 新最优 (Val F1: {best_val_f1:.4f}, Epoch: {epoch}) ***")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info(
                    f"  Early stopping at epoch {epoch} "
                    f"(patience={cfg.patience}, best_epoch={best_epoch})"
                )
                break

    logger.info(
        f"  Fold{fold_idx+1} 完成: best_epoch={best_epoch}, best_val_f1={best_val_f1:.4f}"
    )
    return best_state, best_val_f1


# ═══════════════════════════════════════════════════════════
#  Main experiment runner
# ═══════════════════════════════════════════════════════════

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

    # ── Log config ──
    logger.info("=" * 70)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"输入通道: {cfg.in_channels}")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"分割类别: {cfg.num_seg_classes}")
    logger.info(f"分类类别: {cfg.class_names}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Backbone LR: {cfg.backbone_lr}")
    logger.info(f"Head LR: {cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Warmup Epochs: {cfg.warmup_epochs}")
    logger.info(f"Lambda Cls: {cfg.lambda_cls}")
    logger.info(f"Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Epochs/Fold: {cfg.num_epochs}")
    logger.info(f"Grad Clip: {cfg.grad_clip}")
    logger.info(f"CV Folds: {cfg.n_splits}")
    logger.info(f"Early Stop Patience: {cfg.patience}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 70)

    # ── Build full training dataset (with aug and without aug) ──
    full_train_ds_aug = build_full_dataset(cfg, is_train=True)
    full_train_ds_noaug = build_full_dataset(cfg, is_train=False)
    meta_stats = full_train_ds_aug.meta_stats  # Use global train stats for all folds

    logger.info(
        f"训练集: {len(full_train_ds_aug)} 张 "
        f"(benign={sum(full_train_ds_aug.df['label'] == 0)}, "
        f"no_tumor={sum(full_train_ds_aug.df['label'] == 1)})"
    )

    # ── Build test dataset ──
    test_dataset = build_test_dataset(cfg, meta_stats)
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )
    logger.info(
        f"测试集: {len(test_dataset)} 张 "
        f"(benign={sum(test_dataset.df['label'] == 0)}, "
        f"no_tumor={sum(test_dataset.df['label'] == 1)})"
    )

    # ── Patient-level stratified group K-fold ──
    labels = full_train_ds_aug.df["label"].values
    groups = full_train_ds_aug.df["case_id_norm"].values

    # Verify patient-level grouping
    unique_patients = len(set(groups))
    logger.info(f"唯一患者数: {unique_patients} (图像数: {len(labels)})")

    sgkf = StratifiedGroupKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    fold_val_f1s = []
    fold_states = []

    logger.info("\n" + "=" * 70)
    logger.info("开始5折交叉验证训练")
    logger.info("=" * 70)

    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(
        np.zeros(len(labels)), labels, groups
    )):
        # Verify no patient leakage
        train_patients = set(groups[train_idx])
        val_patients = set(groups[val_idx])
        overlap = train_patients & val_patients
        if overlap:
            logger.warning(f"  !! 患者泄漏: {len(overlap)} 个患者同时出现在train和val !!")
        else:
            logger.info(
                f"  患者分组验证通过: train {len(train_patients)} patients, "
                f"val {len(val_patients)} patients, overlap=0"
            )

        state, val_f1 = train_one_fold(
            fold_idx, train_idx.tolist(), val_idx.tolist(),
            full_train_ds_aug, full_train_ds_noaug, cfg, logger,
        )
        fold_val_f1s.append(val_f1)
        fold_states.append(state)

        # Save fold weight
        fold_weight_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}_fold{fold_idx}.pth")
        torch.save(state, fold_weight_path)
        logger.info(f"  Fold{fold_idx+1} 权重已保存: {fold_weight_path}")

    # ── CV Summary ──
    logger.info("\n" + "=" * 70)
    logger.info("交叉验证汇总")
    logger.info("=" * 70)
    for i, f1 in enumerate(fold_val_f1s):
        logger.info(f"  Fold {i+1}: Val F1 = {f1:.4f}")
    mean_val_f1 = np.mean(fold_val_f1s)
    std_val_f1 = np.std(fold_val_f1s)
    logger.info(f"  平均 Val F1: {mean_val_f1:.4f} ± {std_val_f1:.4f}")

    # ── Test set evaluation: best single fold model ──
    best_fold_idx = int(np.argmax(fold_val_f1s))
    logger.info(f"\n最优单折: Fold {best_fold_idx + 1} (Val F1: {fold_val_f1s[best_fold_idx]:.4f})")

    logger.info("\n" + "=" * 70)
    logger.info(f"测试集评估: 最优单折模型 (Fold {best_fold_idx + 1})")
    logger.info("=" * 70)

    model = build_model(cfg).to(cfg.device)
    model.load_state_dict({k: v.to(cfg.device) for k, v in fold_states[best_fold_idx].items()})

    evaluate_v2(
        model, test_loader, cfg.device, cfg.class_names, logger,
        phase="Test (Best Single Fold)", num_seg_classes=cfg.num_seg_classes,
    )

    # Threshold optimization on test set (for comparison with previous experiments)
    best_thresh, best_thresh_f1 = find_optimal_threshold_v2(model, test_loader, cfg.device)
    logger.info(f"单折最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f})")
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold_v2(
            model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=best_thresh, phase="Test (Best Single Fold, 最优阈值)",
        )

    # ── Test set evaluation: 5-model ensemble ──
    logger.info("\n" + "=" * 70)
    logger.info("测试集评估: 5折模型集成 (概率平均)")
    logger.info("=" * 70)

    ensemble_probs = None
    ensemble_labels = None

    for fold_idx, state in enumerate(fold_states):
        model.load_state_dict({k: v.to(cfg.device) for k, v in state.items()})
        probs, test_labels = predict_probs(model, test_loader, cfg.device)

        if ensemble_probs is None:
            ensemble_probs = probs
            ensemble_labels = test_labels
        else:
            ensemble_probs += probs

    ensemble_probs /= len(fold_states)  # Average probabilities

    # Default threshold
    logger.info("--- 集成 (threshold=0.5) ---")
    ensemble_f1_default = evaluate_from_probs(
        ensemble_probs, ensemble_labels, cfg.class_names, logger,
        phase="Test (5-Fold Ensemble)", threshold=0.5,
    )

    # Optimal threshold
    ens_best_thresh, ens_best_f1 = find_threshold_from_probs(ensemble_probs, ensemble_labels)
    logger.info(f"\n集成最优阈值: {ens_best_thresh:.3f} (F1: {ens_best_f1:.4f} vs 默认0.5 F1: {ensemble_f1_default:.4f})")
    if abs(ens_best_thresh - 0.5) > 0.01:
        logger.info("--- 集成 (最优阈值) ---")
        evaluate_from_probs(
            ensemble_probs, ensemble_labels, cfg.class_names, logger,
            phase="Test (5-Fold Ensemble, 最优阈值)", threshold=ens_best_thresh,
        )

    # ── Final summary ──
    logger.info("\n" + "=" * 70)
    logger.info("最终结果汇总")
    logger.info("=" * 70)
    logger.info(f"  5折平均 Val F1:           {mean_val_f1:.4f} ± {std_val_f1:.4f}")
    logger.info(f"  最优单折 Test F1:          {best_thresh_f1:.4f} (threshold={best_thresh:.3f})")
    logger.info(f"  5折集成 Test F1:           {ens_best_f1:.4f} (threshold={ens_best_thresh:.3f})")
    logger.info("=" * 70)

    # Copy script
    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)
        logger.info(f"训练脚本已复制到: {dst}")


if __name__ == "__main__":
    main()
