"""
Task 2 Recipe-Tuned: Swin-Tiny 二分类 (良性肿瘤 vs 非肿瘤性息肉)
- 基于当前可用 recipe
- 将损失函数替换为 class-aware focal loss
- 目标是继续提升 benign recall / benign f1
"""

import os

import torch
from torch.optim import AdamW
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from task2_recipe_utils import (
    FocalLoss,
    build_optimizer_with_diff_lr,
    run_experiment,
    split_backbone_and_head,
)


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "..", "0316dataset", "Processed")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260319_task2_SwinTiny_adamw_warmupcosine_focalloss_3"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 224
    eval_resize = 232
    batch_size = 16
    num_epochs = 60
    warmup_epochs = 5
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.1
    label_smoothing = 0.0
    focal_gamma = 2.0
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 3
    seed = 42
    use_amp = True
    loss_name = "FocalLoss(alpha=class_weight, gamma=2.0)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "Swin-Tiny (ImageNet预训练)"
    modification = "AdamW + diff lr + warmup cosine + weakaug + focal loss + grad clip"
    train_transform_desc = "RandomResizedCrop224 + HFlip + LightColorJitter"
    test_transform_desc = "Resize232 + CenterCrop224"


def build_model():
    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
    model.head = torch.nn.Linear(model.head.in_features, 2)
    return model


def build_train_transform(cfg):
    return transforms.Compose([
        transforms.RandomResizedCrop(
            cfg.img_size,
            scale=(0.80, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.10, contrast=0.10),
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_test_transform(cfg):
    return transforms.Compose([
        transforms.Resize(cfg.eval_resize, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_optimizer(model, cfg):
    backbone_params, head_params = split_backbone_and_head(model, model.head)
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def build_criterion(cfg, class_weights):
    return FocalLoss(alpha=class_weights, gamma=cfg.focal_gamma)


def main():
    run_experiment(
        cfg=Config(),
        build_model_fn=build_model,
        build_train_transform_fn=build_train_transform,
        build_test_transform_fn=build_test_transform,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
        build_criterion_fn=build_criterion,
    )


if __name__ == "__main__":
    main()
