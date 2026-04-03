"""
Exp #7: Exp#4 + 强正则化 + EMA (最有希望)

核心思路: 在 Exp#4 (F1=0.6707) 基础上, 解决过拟合问题:
  - drop_path_rate=0.3 (Stochastic Depth)
  - weight_decay=0.1 (加倍)
  - label_smoothing=0.15
  - cls_dropout=0.5, meta_dropout=0.3
  - 更强增强 (旋转±30°, 更大scale/shear/色彩变化)
  - EMA (decay=0.999) 平滑权重
  - 50 epochs (Exp#4 在 Epoch 10 就到峰值, 100 太多)
  - eval_interval=2 (更频繁评估)

vs Exp#4:
  + DropPath 0.3, 更大 WD, 更大 dropout
  + 更强数据增强
  + EMA
  - 只训练 50 epoch
"""

import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v3 import (
    GBPDatasetSegCls4chWithMeta,
    SegCls4chStrongSyncTransform,
    SwinV2SegGuidedCls4chModel,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    run_seg_cls_experiment_v3,
    adapt_model_to_4ch,
    META_FEATURE_NAMES,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260402_task2_SwinV2Tiny_segcls_7"
    log_dir = os.path.join(project_root, "0402", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.5       # 0.4 → 0.5
    meta_dim = len(META_FEATURE_NAMES)
    meta_hidden = 64
    meta_dropout = 0.3      # 0.2 → 0.3
    drop_path_rate = 0.3    # NEW

    # 训练
    batch_size = 8
    num_epochs = 50          # 100 → 50
    warmup_epochs = 5
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 0.1       # 0.05 → 0.1
    min_lr_ratio = 0.01
    label_smoothing = 0.15   # 0.1 → 0.15
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 2        # 5 → 2 (更频繁评估)
    seed = 42
    use_amp = True

    # EMA
    use_ema = True
    ema_decay = 0.999

    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + SegAttn + Meta + 强正则化 + EMA"
    modification = (
        "Exp#4基础 + drop_path=0.3 + WD=0.1 + LS=0.15 + dropout=0.5/0.3 "
        "+ 强增强(±30°/scale0.5-1.0) + EMA(0.999) + 50ep"
    )
    train_transform_desc = "StrongSync+(RRC0.5+Rot30+Shear10+StrongColor+MoreErase+MoreNoise)"
    test_transform_desc = "Resize256 4ch"


def build_model(cfg):
    model = SwinV2SegGuidedCls4chModel(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        pretrained=True,
    )
    # Replace encoder with drop_path version
    model.encoder = timm.create_model(
        "swinv2_tiny_window8_256", pretrained=True,
        features_only=True, out_indices=(0, 1, 2, 3),
        drop_path_rate=cfg.drop_path_rate,
    )
    adapt_model_to_4ch(model.encoder)
    return model


def build_dataloaders(cfg):
    train_sync = SegCls4chStrongSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chStrongSyncTransform(cfg.img_size, is_train=False)

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
    run_seg_cls_experiment_v3(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
