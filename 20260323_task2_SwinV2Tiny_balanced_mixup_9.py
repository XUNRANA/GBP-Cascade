"""
Exp #9: SwinV2-Tiny + Full4ch + WeightedSampler + 修复ClassWeight Mixup ★★★★
核心改进:
  1. 修复 BUG: Mixup soft CE 现在包含类别权重 (之前遗漏导致 benign recall 极低)
  2. WeightedRandomSampler: 每个 batch 类别均衡 (不再靠 shuffle 随机)
  3. 强增强 + Mixup/CutMix
  4. 100 epochs, dropout 0.3
"""

import os

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


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260323_task2_SwinV2Tiny_balanced_mixup_9"
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
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 5
    seed = 42
    use_amp = True
    use_mixup = True
    loss_name = "CrossEntropyLoss(class_weight + LS=0.1) + Weighted Mixup/CutMix + BalancedSampler"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "Swin-V2-Tiny (ImageNet, 4ch, head dropout=0.3)"
    modification = "全图4ch + WeightedSampler均衡 + 修复Mixup类权重Bug + 强增强 + 100ep"
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

    # WeightedRandomSampler 确保 batch 内类别均衡
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
    )


if __name__ == "__main__":
    main()
