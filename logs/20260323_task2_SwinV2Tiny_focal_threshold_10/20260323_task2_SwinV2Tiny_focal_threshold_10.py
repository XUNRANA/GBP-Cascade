"""
Exp #10: SwinV2-Tiny + Focal Loss + Balanced Sampler + 阈值优化 ★★★★★
核心改进:
  1. Focal Loss (gamma=2): 聚焦难分样本 (benign 被误判为 no_tumor 的情况)
  2. WeightedRandomSampler: batch 类别均衡
  3. 不用 Mixup: Focal Loss 和 Mixup 有冲突 (Focal需要hard label)
  4. 训练后阈值优化: 搜索最优 benign 概率阈值 (替代默认0.5)
  5. 强增强
"""

import os

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from task2_json_utils import (
    FocalLoss,
    GBPDatasetFull4ch,
    StrongSyncTransform,
    SyncTransform,
    adapt_model_to_4ch,
    build_class_weights,
    build_optimizer_with_diff_lr,
    build_weighted_sampler,
    evaluate_with_threshold,
    find_optimal_threshold,
    run_experiment,
    split_backbone_and_head,
)


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260323_task2_SwinV2Tiny_focal_threshold_10"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    batch_size = 8
    num_epochs = 100
    warmup_epochs = 8
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.0  # Focal Loss 不需要 label smoothing
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 5
    seed = 42
    use_amp = True
    use_mixup = False  # Focal Loss 不搭配 Mixup
    loss_name = "FocalLoss(gamma=2, class_weight) + BalancedSampler + ThresholdTuning"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "Swin-V2-Tiny (ImageNet, 4ch, head dropout=0.3)"
    modification = "全图4ch + FocalLoss + WeightedSampler + 强增强 + 阈值优化 + 100ep"
    train_transform_desc = "Full_img → StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise) + mask"
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
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_criterion(cfg, class_weights):
    return FocalLoss(alpha=class_weights, gamma=2.0)


def build_optimizer(model, cfg):
    head = model.head.fc if hasattr(model.head, "fc") else model.head
    backbone_params, head_params = split_backbone_and_head(model, head)
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def main():
    run_experiment(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
        build_criterion_fn=build_criterion,
    )


if __name__ == "__main__":
    main()
