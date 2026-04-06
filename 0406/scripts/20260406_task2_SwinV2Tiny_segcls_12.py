"""
Exp #12: Exp#4 + 验证集划分 + Early Stopping + EMA

核心思路:
  - 架构与 Exp#4 完全相同 (SwinV2-Tiny + 4ch + SegAttn + 6D Meta)
  - 训练集 85/15 分层划分 → train/val
  - Early Stopping: 监控 val F1, patience=10
  - EMA decay=0.9995
  - 解决 Exp#4 的核心问题: 无验证集 → 测试集选模型有乐观偏差

vs Exp#4:
  + 分层验证集 (85/15)
  + Early Stopping (patience=10)
  + EMA (decay=0.9995)
  + eval_interval=1 (每轮评估)
"""

import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v4 import (
    GBPDatasetSegCls4chWithMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chModel,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    run_experiment_with_early_stopping,
    META_FEATURE_NAMES,
)

from sklearn.model_selection import StratifiedShuffleSplit


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260406_task2_SwinV2Tiny_segcls_12"
    log_dir = os.path.join(project_root, "0406", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型 — 与 Exp#4 完全相同
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
    num_epochs = 50
    warmup_epochs = 8
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 1  # 每轮评估 (用于精确 early stopping)
    seed = 42
    use_amp = True

    # Early stopping + EMA
    val_ratio = 0.15
    patience = 10
    use_ema = True
    ema_decay = 0.9995

    # 损失
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + SegAttn + Meta (同 Exp#4)"
    modification = (
        "Exp#4架构 + 85/15分层验证集 + EarlyStopping(patience=10) "
        "+ EMA(decay=0.9995) + eval每轮"
    )


class _SubsetDatasetWrapper:
    """Wrap a Subset to expose parent dataset's .df (filtered by indices)."""
    def __init__(self, subset, parent_dataset):
        self._subset = subset
        self.df = parent_dataset.df.iloc[subset.indices].reset_index(drop=True)
        self.meta_stats = parent_dataset.meta_stats

    def __len__(self):
        return len(self._subset)

    def __getitem__(self, idx):
        return self._subset[idx]


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
    """返回 6-tuple: (train_ds, val_ds, test_ds, train_loader, val_loader, test_loader)."""
    # 先加载完整训练集
    full_train_dataset = GBPDatasetSegCls4chWithMeta(
        cfg.train_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=SegCls4chSyncTransform(cfg.img_size, is_train=True),
    )

    # 分层划分 train/val
    labels = full_train_dataset.df["label"].values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=cfg.val_ratio, random_state=cfg.seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    train_subset = Subset(full_train_dataset, train_idx.tolist())
    train_dataset = _SubsetDatasetWrapper(train_subset, full_train_dataset)

    # Val dataset: 无增强
    val_full = GBPDatasetSegCls4chWithMeta(
        cfg.train_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=SegCls4chSyncTransform(cfg.img_size, is_train=False),
        meta_stats=full_train_dataset.meta_stats,
    )
    val_subset = Subset(val_full, val_idx.tolist())
    val_dataset = _SubsetDatasetWrapper(val_subset, val_full)

    # Test dataset
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)
    test_dataset = GBPDatasetSegCls4chWithMeta(
        cfg.test_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=test_sync,
        meta_stats=full_train_dataset.meta_stats,
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
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )
    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def build_optimizer(model, cfg):
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def main():
    run_experiment_with_early_stopping(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
