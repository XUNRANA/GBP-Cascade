"""
Exp #8: ConvNeXt-Tiny + 全图4ch + 强增强 + Mixup/CutMix
- 和 #7 相同策略，但用 ConvNeXt (CNN对比)
- 目的：验证强增强+Mixup对CNN是否同样有效
"""

import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import models

from task2_json_utils import (
    GBPDatasetFull4ch,
    StrongSyncTransform,
    SyncTransform,
    adapt_model_to_4ch,
    build_optimizer_with_diff_lr,
    run_experiment,
    split_backbone_and_head,
)


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260323_task2_ConvNeXtTiny_full4ch_strongaug_8"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 320
    in_channels = 4
    batch_size = 8
    num_epochs = 100
    warmup_epochs = 8
    backbone_lr = 5e-5
    head_lr = 5e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 5
    seed = 42
    use_amp = True
    use_mixup = True
    loss_name = "CrossEntropyLoss(class_weight + LS=0.1) + Mixup/CutMix"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "ConvNeXt-Tiny (ImageNet, 4ch, dropout=0.3)"
    modification = "全图(不裁ROI) + 病灶mask 4ch + 强增强 + Mixup/CutMix + 100ep"
    train_transform_desc = "Full_img → StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise) + mask"
    test_transform_desc = "Full_img → Resize320 + mask"


def build_model():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    adapt_model_to_4ch(model)
    in_feat = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_feat, 2),
    )
    return model


def build_dataloaders(cfg):
    train_sync = StrongSyncTransform(cfg.img_size, is_train=True)
    test_sync = SyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDatasetFull4ch(cfg.train_excel, cfg.data_root, sync_transform=train_sync)
    test_dataset = GBPDatasetFull4ch(cfg.test_excel, cfg.data_root, sync_transform=test_sync)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    backbone_params, head_params = split_backbone_and_head(model, model.classifier[2])
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
