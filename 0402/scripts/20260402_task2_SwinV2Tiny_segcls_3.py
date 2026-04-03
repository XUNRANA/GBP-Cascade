"""
Exp #3: SwinV2-Tiny@256 + 4ch输入 + 分割辅助正则化

核心思路:
  - 保留基线的 4ch 输入 (RGB + lesion_mask)
  - 额外加 UNet 分割解码器作为辅助任务/正则化器
  - 分割 loss 权重低 (lambda_cls=2.0 → cls 是主任务)
  - 等于 "基线 + 免费的分割正则化"

vs 基线 (F1=0.6371):
  - 新增 UNet 分割解码器 (辅助正则化)
  - 无 Mixup (分割与 Mixup 冲突)
  - 无 metadata (先验证分割辅助效果)

vs Exp#1 (F1=0.6157):
  - 恢复 4ch 输入 (不再丢失 mask 信息!)
  - lambda_cls=2.0 (cls 为主, seg 为辅)
"""

import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v2 import (
    GBPDatasetSegCls4ch,
    SegCls4chSyncTransform,
    SwinV2SegCls4chModel,
    seg_cls_4ch_collate_fn,
    build_optimizer_with_diff_lr,
    run_seg_cls_experiment_v2,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260402_task2_SwinV2Tiny_segcls_3"
    log_dir = os.path.join(project_root, "0402", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型
    img_size = 256
    in_channels = 4  # RGB + mask
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.3

    # 训练
    batch_size = 8
    num_epochs = 100
    warmup_epochs = 8
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 5
    seed = 42
    use_amp = True

    # 损失: total = seg_loss + lambda_cls * cls_loss
    # lambda_cls=2.0 → cls 是主任务 (等价于 cls_loss + 0.5*seg_loss)
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + UNet seg aux regularizer"
    modification = (
        "4ch(RGB+mask)输入 + UNet分割解码器(辅助正则化, lambda_cls=2.0) "
        "+ GAP分类头 + 无Mixup + 无metadata + 100ep"
    )
    train_transform_desc = "StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise) 4ch"
    test_transform_desc = "Resize256 4ch"


def build_model(cfg):
    return SwinV2SegCls4chModel(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        cls_dropout=cfg.cls_dropout,
        pretrained=True,
    )


def build_dataloaders(cfg):
    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDatasetSegCls4ch(cfg.train_excel, cfg.data_root, sync_transform=train_sync)
    test_dataset = GBPDatasetSegCls4ch(cfg.test_excel, cfg.data_root, sync_transform=test_sync)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=seg_cls_4ch_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_4ch_collate_fn,
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
    run_seg_cls_experiment_v2(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
