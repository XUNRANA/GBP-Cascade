"""
Exp #13: Exp#4 + Feature-Level Mixup

核心思路:
  - 输入级 Mixup 在 Exp#2 中失败 (F1=0.5905) 因为混合破坏分割 mask
  - 本实验在 seg-guided attention 输出后的 feature space 做 Mixup
  - 分割分支完全不受影响, 仅分类分支获得 Mixup 正则化
  - 直接对抗 Exp#4 的核心问题: epoch 10 过拟合

vs Exp#4:
  + Feature-Level Mixup (p=0.5, alpha=0.4)
  + 使用 SwinV2SegGuidedCls4chModelV4 (支持 forward_features/forward_cls 拆分)
  + 6D metadata (同 Exp#4)
"""

import os
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v4 import (
    GBPDatasetSegCls4chWithMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chModelV4,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    run_experiment_with_early_stopping,
    train_one_epoch_feature_mixup,
    META_FEATURE_NAMES,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260406_task2_SwinV2Tiny_segcls_13"
    log_dir = os.path.join(project_root, "0406", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型 — 同 Exp#4 但用 V4 模型类
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
    num_epochs = 50  # Mixup 正则化可能允许训练更久
    warmup_epochs = 8
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

    # Feature Mixup
    mixup_prob = 0.5
    mixup_alpha = 0.4

    # Early stopping
    patience = 12  # Mixup 可能需要更久收敛
    use_ema = False  # 不用 EMA, 单独验证 Feature Mixup 效果

    # 损失
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + SegAttn + Meta + FeatureMixup"
    modification = (
        "Exp#4架构 + Feature-Level Mixup(p=0.5, alpha=0.4) "
        "+ 50ep + patience=12"
    )


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


def build_dataloaders(cfg):
    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDatasetSegCls4chWithMeta(
        cfg.train_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=train_sync,
    )
    test_dataset = GBPDatasetSegCls4chWithMeta(
        cfg.test_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=test_sync,
        meta_stats=train_dataset.meta_stats,
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
        train_fn=train_one_epoch_feature_mixup,
    )


if __name__ == "__main__":
    main()
