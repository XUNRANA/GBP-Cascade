"""
Exp #8: Exp#4 + 多尺度 Seg-Guided Attention

核心思路: 用 f2 (16x16, 细粒度) + f3 (8x8, 粗粒度) 双尺度 seg-guided attention,
  捕获不同分辨率的病灶特征, 拼接后做分类.

vs Exp#4 (单尺度 f2):
  + 额外用 f3 的粗粒度特征
  + 两尺度特征拼接 (512d vs 256d)
  + 加 drop_path_rate=0.2 (轻度正则化)
  - 只训练 50 epoch
"""

import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v3 import (
    GBPDatasetSegCls4chWithMeta,
    SegCls4chSyncTransform,
    SwinV2MultiScaleSegGuidedCls4chModel,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    run_seg_cls_experiment_v3,
    META_FEATURE_NAMES,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260402_task2_SwinV2Tiny_segcls_8"
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
    drop_path_rate = 0.2

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
    use_ema = False

    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + MultiScale SegAttn (f2+f3) + Meta"
    modification = (
        "双尺度seg-guided attention: f2(16x16)+f3(8x8) → 512d "
        "+ metadata融合 + drop_path=0.2 + 50ep"
    )
    train_transform_desc = "StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise) 4ch"
    test_transform_desc = "Resize256 4ch"


def build_model(cfg):
    return SwinV2MultiScaleSegGuidedCls4chModel(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        drop_path_rate=cfg.drop_path_rate,
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
    run_seg_cls_experiment_v3(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
