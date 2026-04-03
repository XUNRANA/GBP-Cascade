"""
Exp #1: SwinV2-Tiny 分割+分类联合训练 (模仿 unet-valjustclass 思路)

核心思路:
  - 输入 3ch RGB (不把 mask 当输入, 而是当分割目标)
  - SwinV2-Tiny 编码器提取4级层次特征 (features_only=True)
  - UNet 解码器用 skip connections 做病灶分割 (binary: bg/lesion)
  - 分类头从最深特征做 benign/no_tumor 分类
  - 联合训练: seg_loss(CE+Dice) + lambda_cls * cls_loss(CE)
  - 分割迫使模型学习病灶空间特征, 辅助提升分类

vs 之前的 Exp#7 (best F1=0.6371):
  - 去掉第4通道 mask 输入 (mask 变成分割目标)
  - 新增 UNet 分割解码器
  - 新增分割损失 (CE + Dice)
  - 模型参数增加约15M (解码器部分)
"""

import os
import sys

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

# 加入当前脚本目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils import (
    GBPDatasetSegCls,
    SegClsSyncTransform,
    SwinV2SegClsModel,
    build_optimizer_with_diff_lr,
    run_seg_cls_experiment,
    seg_cls_collate_fn,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260402_task2_SwinV2Tiny_segcls_1"
    log_dir = os.path.join(project_root, "0402", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型
    img_size = 256
    in_channels = 3  # 3ch RGB (mask 是分割目标, 不是输入)
    num_seg_classes = 2  # background + lesion
    num_cls_classes = 2  # benign + no_tumor
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

    # 损失权重
    lambda_cls = 1.0       # 分类损失权重 (相对于分割损失)
    seg_bg_weight = 1.0    # 分割: 背景权重
    seg_lesion_weight = 5.0  # 分割: 病灶权重 (病灶面积小, 给更高权重)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny + UNet Decoder + Cls Head (segcls joint training)"
    modification = (
        "3ch RGB输入 + UNet分割解码器(skip connections) + 分类头 "
        "+ 分割目标=病灶mask(from JSON) + 联合训练(seg CE+Dice + cls CE) "
        "+ 强增强 + 100ep"
    )
    train_transform_desc = "StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise)"
    test_transform_desc = "Resize256"


def build_model(cfg):
    return SwinV2SegClsModel(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        cls_dropout=cfg.cls_dropout,
        pretrained=True,
    )


def build_dataloaders(cfg):
    train_sync = SegClsSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegClsSyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDatasetSegCls(cfg.train_excel, cfg.data_root, sync_transform=train_sync)
    test_dataset = GBPDatasetSegCls(cfg.test_excel, cfg.data_root, sync_transform=test_sync)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=seg_cls_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_collate_fn,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    # 分离 backbone 和 head 参数
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def main():
    run_seg_cls_experiment(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
