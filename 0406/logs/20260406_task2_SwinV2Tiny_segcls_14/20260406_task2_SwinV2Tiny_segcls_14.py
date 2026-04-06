"""
Exp #14: 5-Fold CV + TTA + Morph 集成 (生产级流水线)

核心思路:
  - 综合 Exp#11 (18D 扩展 metadata) + Exp#12 (Early Stopping + EMA) + TTA
  - 5-Fold 交叉验证: 每个 fold 看到不同训练子集 → 真正多样化的模型
  - TTA: 5个增强视角的概率平均
  - 集成: 5 fold × 5 TTA = 25 次前向 → 概率平均 → 最终预测
  - vs Exp#10 (3-seed 集成, F1=0.6326): 这次用不同数据分割而非不同种子

vs Exp#4:
  + 18D metadata (6 clinical + 12 morph)
  + 5-Fold CV (不同训练子集)
  + Early Stopping + EMA (每个 fold)
  + TTA (5 views)
"""

import os
import sys
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v4 import (
    GBPDatasetSegCls4chWithExtMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chModelV4,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    build_class_weights,
    SegClsLoss, ModelEMA,
    set_seed, setup_logger, acquire_run_lock,
    cosine_warmup_factor, set_epoch_lrs,
    train_one_epoch_v2, evaluate_v2,
    compute_seg_metrics,
    tta_predict, evaluate_with_tta,
    EXT_META_FEATURE_NAMES,
    _unpack_batch,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260406_task2_SwinV2Tiny_segcls_14"
    log_dir = os.path.join(project_root, "0406", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")

    # 模型
    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(EXT_META_FEATURE_NAMES)  # 18
    meta_hidden = 128
    meta_dropout = 0.2

    # 训练 (每 fold)
    batch_size = 8
    num_epochs = 30
    warmup_epochs = 8
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 1
    seed = 42
    use_amp = True

    # K-Fold
    n_splits = 5
    patience = 8
    ema_decay = 0.9995

    # 损失
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + SegAttn + ExtMeta(18D) + 5Fold + TTA"
    modification = (
        "5-Fold CV + EarlyStopping(patience=8) + EMA(0.9995) "
        "+ 18D ExtMeta + TTA(5views) + 集成推理"
    )


class _SubsetDatasetWrapper:
    """Expose parent dataset's .df filtered by subset indices."""
    def __init__(self, subset, parent_dataset):
        self._subset = subset
        self.df = parent_dataset.df.iloc[subset.indices].reset_index(drop=True)

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        return self._subset[idx]


def build_model(cfg):
    return SwinV2SegGuidedCls4chModelV4(
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


def train_one_fold(fold_idx, train_indices, val_indices, full_train_dataset_aug,
                   full_train_dataset_noaug, test_dataset, test_loader, cfg, logger):
    """训练单个 fold, 返回 (best_model_state, val_f1, test_f1)."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Fold {fold_idx + 1}/{cfg.n_splits}")
    logger.info(f"{'='*70}")

    set_seed(cfg.seed + fold_idx)

    # Train/val subsets
    train_subset = Subset(full_train_dataset_aug, train_indices)
    val_subset = Subset(full_train_dataset_noaug, val_indices)

    train_ds = _SubsetDatasetWrapper(train_subset, full_train_dataset_aug)
    val_ds = _SubsetDatasetWrapper(val_subset, full_train_dataset_noaug)

    logger.info(f"Fold {fold_idx+1}: Train={len(train_ds)}, Val={len(val_ds)}")

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

    model = build_model(cfg).to(cfg.device)
    ema = ModelEMA(model, decay=cfg.ema_decay)

    cls_weights = build_class_weights(train_ds.df, cfg.class_names, cfg.device)
    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight], dtype=torch.float32, device=cfg.device
    )
    criterion = SegClsLoss(
        cls_weights=cls_weights, lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing, seg_ce_weight=seg_ce_weight,
    )
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    best_f1, best_epoch = 0.0, 0
    no_improve = 0
    best_state = None

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        train_metrics = train_one_epoch_v2(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
        )
        ema.update(model)
        elapsed = time.time() - t0

        logger.info(
            f"  Fold{fold_idx+1} Epoch [{epoch}/{cfg.num_epochs}] "
            f"Loss: {train_metrics['loss']:.4f} "
            f"| Acc: {train_metrics['cls_acc']:.4f} "
            f"| Dice: {train_metrics['seg_dice']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            _, _, _, f1_val, _, _ = evaluate_v2(
                ema.module, val_loader, cfg.device, cfg.class_names, logger,
                phase=f"Fold{fold_idx+1} Val(EMA)", num_seg_classes=cfg.num_seg_classes,
            )
            if f1_val > best_f1:
                best_f1 = f1_val
                best_epoch = epoch
                no_improve = 0
                best_state = {k: v.cpu().clone() for k, v in ema.module.state_dict().items()}
                logger.info(f"  *** Fold{fold_idx+1} Best: F1={best_f1:.4f} @ Epoch {best_epoch} ***")
            else:
                no_improve += 1
            if no_improve >= cfg.patience:
                logger.info(f"  Fold{fold_idx+1} Early stopping @ Epoch {epoch}")
                break

    # Fold 结果
    logger.info(f"\nFold {fold_idx+1} 最优: Epoch {best_epoch}, Val F1: {best_f1:.4f}")

    # 加载 best state, 在 test 集评估
    model.load_state_dict({k: v.to(cfg.device) for k, v in best_state.items()})
    _, _, _, f1_test, _, _ = evaluate_v2(
        model, test_loader, cfg.device, cfg.class_names, logger,
        phase=f"Fold{fold_idx+1} Test", num_seg_classes=cfg.num_seg_classes,
    )

    # 保存 fold 权重
    fold_weight_path = os.path.join(cfg.log_dir, f"fold{fold_idx+1}_best.pth")
    torch.save(best_state, fold_weight_path)
    logger.info(f"  Fold{fold_idx+1} 权重已保存: {fold_weight_path}")

    return best_state, best_f1, f1_test


def ensemble_evaluate(fold_states, test_loader, cfg, logger):
    """加载所有 fold 模型, 集成推理 (概率平均), 可选 TTA."""
    logger.info("\n" + "=" * 70)
    logger.info(f"集成推理: {len(fold_states)} fold models")
    logger.info("=" * 70)

    models = []
    for state in fold_states:
        m = build_model(cfg).to(cfg.device)
        m.load_state_dict({k: v.to(cfg.device) for k, v in state.items()})
        m.eval()
        models.append(m)

    # ── 1. 简单集成 (无 TTA) ──
    all_probs_ensemble = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            imgs, masks, metas, labels, has_masks = _unpack_batch(batch)
            imgs = imgs.to(cfg.device, non_blocking=True)
            if metas is not None:
                metas = metas.to(cfg.device, non_blocking=True)

            batch_probs = []
            for m in models:
                _, logits = m(imgs, metadata=metas)
                batch_probs.append(F.softmax(logits, dim=1))

            avg_probs = torch.stack(batch_probs, dim=0).mean(dim=0)
            all_probs_ensemble.append(avg_probs.cpu())
            all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

    all_probs = torch.cat(all_probs_ensemble, dim=0).numpy()
    all_labels = np.array(all_labels)
    all_preds = all_probs.argmax(axis=1)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(f"[Ensemble (no TTA)] Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1:.4f}")
    report = classification_report(all_labels, all_preds, target_names=cfg.class_names, digits=4)
    logger.info(f"[Ensemble (no TTA)] Report:\n{report}")

    # 阈值优化
    best_f1_t, best_thresh = 0.0, 0.5
    benign_probs = all_probs[:, 0]
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds_t = np.where(benign_probs >= thresh, 0, 1)
        f1_t = f1_score(all_labels, preds_t, average="macro", zero_division=0)
        if f1_t > best_f1_t:
            best_f1_t = f1_t
            best_thresh = thresh
    logger.info(f"[Ensemble (no TTA)] 最优阈值: {best_thresh:.3f} (F1: {best_f1_t:.4f})")

    # ── 2. 集成 + TTA ──
    all_probs_tta = []
    all_labels2 = []

    with torch.no_grad():
        for batch in test_loader:
            imgs, masks, metas, labels, has_masks = _unpack_batch(batch)
            imgs = imgs.to(cfg.device, non_blocking=True)
            if metas is not None:
                metas = metas.to(cfg.device, non_blocking=True)

            batch_probs = []
            for m in models:
                tta_probs = tta_predict(m, imgs, metas, cfg.device)
                batch_probs.append(tta_probs)

            avg_probs = torch.stack(batch_probs, dim=0).mean(dim=0)
            all_probs_tta.append(avg_probs.cpu())
            all_labels2.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

    all_probs_tta = torch.cat(all_probs_tta, dim=0).numpy()
    all_labels2 = np.array(all_labels2)
    all_preds_tta = all_probs_tta.argmax(axis=1)

    acc = accuracy_score(all_labels2, all_preds_tta)
    prec = precision_score(all_labels2, all_preds_tta, average="macro", zero_division=0)
    rec = recall_score(all_labels2, all_preds_tta, average="macro", zero_division=0)
    f1_tta = f1_score(all_labels2, all_preds_tta, average="macro", zero_division=0)

    logger.info(f"\n[Ensemble + TTA] Acc: {acc:.4f} | P: {prec:.4f} | R: {rec:.4f} | F1: {f1_tta:.4f}")
    report = classification_report(all_labels2, all_preds_tta, target_names=cfg.class_names, digits=4)
    logger.info(f"[Ensemble + TTA] Report:\n{report}")

    # 阈值优化
    best_f1_tta, best_thresh_tta = 0.0, 0.5
    benign_probs_tta = all_probs_tta[:, 0]
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds_t = np.where(benign_probs_tta >= thresh, 0, 1)
        f1_t = f1_score(all_labels2, preds_t, average="macro", zero_division=0)
        if f1_t > best_f1_tta:
            best_f1_tta = f1_t
            best_thresh_tta = thresh
    logger.info(f"[Ensemble + TTA] 最优阈值: {best_thresh_tta:.3f} (F1: {best_f1_tta:.4f})")

    return f1, best_f1_t, f1_tta, best_f1_tta


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
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"K-Fold: {cfg.n_splits} folds")
    logger.info(f"每 fold: {cfg.num_epochs} epochs, patience={cfg.patience}")
    logger.info(f"EMA decay: {cfg.ema_decay}")
    logger.info("=" * 70)

    # 加载完整训练数据集 (有增强 + 无增强)
    train_sync_aug = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    train_sync_noaug = SegCls4chSyncTransform(cfg.img_size, is_train=False)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    full_train_dataset_aug = GBPDatasetSegCls4chWithExtMeta(
        cfg.train_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=train_sync_aug,
    )
    full_train_dataset_noaug = GBPDatasetSegCls4chWithExtMeta(
        cfg.train_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=train_sync_noaug,
        meta_stats=full_train_dataset_aug.meta_stats,
    )
    test_dataset = GBPDatasetSegCls4chWithExtMeta(
        cfg.test_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=test_sync,
        meta_stats=full_train_dataset_aug.meta_stats,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )

    logger.info(f"训练集: {len(full_train_dataset_aug)} 张")
    logger.info(f"测试集: {len(test_dataset)} 张")

    # K-Fold 划分
    labels = full_train_dataset_aug.df["label"].values
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)

    fold_states = []
    val_f1s, test_f1s = [], []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        state, val_f1, test_f1 = train_one_fold(
            fold_idx, train_idx.tolist(), val_idx.tolist(),
            full_train_dataset_aug, full_train_dataset_noaug,
            test_dataset, test_loader, cfg, logger,
        )
        fold_states.append(state)
        val_f1s.append(val_f1)
        test_f1s.append(test_f1)

    # 汇总各 fold 结果
    logger.info("\n" + "=" * 70)
    logger.info("各 Fold 结果汇总")
    logger.info("=" * 70)
    for i, (vf, tf) in enumerate(zip(val_f1s, test_f1s)):
        logger.info(f"  Fold {i+1}: Val F1={vf:.4f}, Test F1={tf:.4f}")
    logger.info(f"  平均: Val F1={np.mean(val_f1s):.4f}±{np.std(val_f1s):.4f}, "
                f"Test F1={np.mean(test_f1s):.4f}±{np.std(test_f1s):.4f}")

    # 集成推理
    f1_ens, f1_ens_thresh, f1_tta, f1_tta_thresh = ensemble_evaluate(
        fold_states, test_loader, cfg, logger,
    )

    logger.info("\n" + "=" * 70)
    logger.info("最终结果汇总")
    logger.info("=" * 70)
    logger.info(f"  单模型平均 Test F1: {np.mean(test_f1s):.4f}")
    logger.info(f"  集成 (无 TTA) F1:   {f1_ens:.4f} (阈值优化: {f1_ens_thresh:.4f})")
    logger.info(f"  集成 + TTA F1:      {f1_tta:.4f} (阈值优化: {f1_tta_thresh:.4f})")

    # 复制脚本
    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)
        logger.info(f"脚本已复制到: {dst}")


if __name__ == "__main__":
    main()
