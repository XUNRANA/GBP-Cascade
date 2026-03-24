"""
方案五: SwinV2-Tiny @ 384×384 + 200 epochs + CosineRestart
============================================================
更大分辨率 → 更多超声细节, 更长训练 → 更好收敛
使用 swinv2_tiny_window16_256 (window_size=16, 可适配384)
"""

import os
import sys
import math

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from task2_json_utils import (
    GBPDatasetFull4ch,
    StrongSyncTransform,
    SyncTransform,
    adapt_model_to_4ch,
    build_optimizer_with_diff_lr,
    build_weighted_sampler,
    run_experiment,
    split_backbone_and_head,
)


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    img_size = 384       # ← 更大分辨率
    in_channels = 4
    batch_size = 8        # swinv2_cr_tiny_384 原生支持 384
    num_epochs = 200      # ← 更长训练
    warmup_epochs = 10
    backbone_lr = 1e-5    # 稍微更小的学习率
    head_lr = 1e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 5
    use_amp = True
    use_mixup = True
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "ConvNeXt-Tiny (384×384, 4ch, drop=0.3)"
    train_transform_desc = "Full_img → StrongSync384 + mask"
    test_transform_desc = "Full_img → Resize384 + mask"

    exp_name = "20260323_task2_SwinV2_384_200ep"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")
    modification = "全图4ch + BalancedSampler + 修复Mixup + 强增强 + 384分辨率 + 200ep"
    loss_name = "CE(class_weight+LS=0.1) + Weighted Mixup + BalancedSampler"


def build_model():
    # ConvNeXt-Tiny: CNN 架构无输入尺寸限制, 适合 384 分辨率
    model = timm.create_model("convnext_tiny.fb_in1k", pretrained=True,
                              num_classes=2, drop_rate=0.3)
    adapt_model_to_4ch(model)
    return model


def build_dataloaders(cfg):
    train_sync = StrongSyncTransform(cfg.img_size, is_train=True)
    test_sync = SyncTransform(cfg.img_size, is_train=False)
    train_dataset = GBPDatasetFull4ch(cfg.train_excel, cfg.data_root, sync_transform=train_sync)
    test_dataset = GBPDatasetFull4ch(cfg.test_excel, cfg.data_root, sync_transform=test_sync)
    sampler = build_weighted_sampler(train_dataset.df)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    head = model.head.fc if hasattr(model.head, "fc") else model.head
    backbone_params, head_params = split_backbone_and_head(model, head)
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def main():
    cfg = Config()
    run_experiment(
        cfg=cfg,
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
