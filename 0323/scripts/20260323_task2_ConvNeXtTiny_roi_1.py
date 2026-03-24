"""
Exp #1: ConvNeXt-Tiny + ROI裁剪 (CNN Baseline)
- 仅用 gallbladder 矩形框裁剪，3通道输入
- 对照组：量化 ROI 裁剪 vs 0319 全图直接 resize 的增益
"""

import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import models

from task2_json_utils import (
    GBPDatasetROI,
    build_optimizer_with_diff_lr,
    build_roi_test_transform,
    build_roi_train_transform,
    run_experiment,
    split_backbone_and_head,
)


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260323_task2_ConvNeXtTiny_roi_1"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 320
    in_channels = 3
    batch_size = 8
    num_epochs = 60
    warmup_epochs = 5
    backbone_lr = 5e-5
    head_lr = 5e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.05
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 3
    seed = 42
    use_amp = True
    loss_name = "CrossEntropyLoss(class_weight + label_smoothing=0.1)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "ConvNeXt-Tiny (ImageNet pretrained)"
    modification = "ROI crop + AdamW + diff lr + warmup cosine + class weight + label smoothing"
    train_transform_desc = "ROI_crop → Resize320 + HFlip + VFlip + Affine + ColorJitter"
    test_transform_desc = "ROI_crop → Resize320"


def build_model():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    return model


def build_dataloaders(cfg):
    train_transform = build_roi_train_transform(cfg.img_size)
    test_transform = build_roi_test_transform(cfg.img_size)

    train_dataset = GBPDatasetROI(cfg.train_excel, cfg.data_root, transform=train_transform)
    test_dataset = GBPDatasetROI(cfg.test_excel, cfg.data_root, transform=test_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True,
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
