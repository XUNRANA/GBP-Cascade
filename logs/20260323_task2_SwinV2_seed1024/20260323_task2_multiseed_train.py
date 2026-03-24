"""
方案三-Step1: 多种子训练 — 用 5 个不同种子训练 SwinV2 最优配置
训练完毕后 Step2 用 feature_xgboost_v2.py 提取多种子特征 + XGBoost
"""

import os
import sys

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
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

SEEDS = [42, 123, 456, 789, 1024]


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    img_size = 256
    in_channels = 4
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
    use_amp = True
    use_mixup = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "Swin-V2-Tiny (ImageNet, 4ch, drop=0.3)"
    train_transform_desc = "Full_img → StrongSync + mask"
    test_transform_desc = "Full_img → Resize256 + mask"


def build_model():
    model = timm.create_model("swinv2_tiny_window8_256", pretrained=True, num_classes=2, drop_rate=0.3)
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
    seed_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    seed = SEEDS[seed_idx]

    cfg = Config()
    cfg.seed = seed
    cfg.exp_name = f"20260323_task2_SwinV2_seed{seed}"
    cfg.log_dir = os.path.join(cfg.project_root, "logs", cfg.exp_name)
    cfg.log_file = os.path.join(cfg.log_dir, f"{cfg.exp_name}.log")
    cfg.best_weight_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}_best.pth")
    cfg.modification = f"全图4ch + BalancedSampler + 修复Mixup + 强增强 + seed={seed}"
    cfg.loss_name = f"CE(class_weight+LS=0.1) + Weighted Mixup + BalancedSampler [seed={seed}]"

    run_experiment(
        cfg=cfg,
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
