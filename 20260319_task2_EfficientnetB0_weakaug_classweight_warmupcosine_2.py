"""
Task 2 Optimized: EfficientNet-B0 二分类 (良性肿瘤 vs 非肿瘤性息肉)
- 使用 AdamW + 差分学习率
- 使用 warmup cosine 学习率调度
- 使用 weak augmentation + class weight + label smoothing
- 针对 baseline 过早过拟合做节奏优化
"""

import os

import torch
from torch.optim import AdamW
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from task2_recipe_utils import build_optimizer_with_diff_lr, run_experiment, split_backbone_and_head


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "dataset", "Processed")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260319_task2_EfficientnetB0_weakaug_classweight_warmupcosine_2"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 224
    train_resize = 256
    eval_resize = 256
    batch_size = 32
    num_epochs = 50
    warmup_epochs = 3
    backbone_lr = 5e-5
    head_lr = 5e-4
    weight_decay = 1e-3
    min_lr_ratio = 0.05
    label_smoothing = 0.05
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 2
    seed = 42
    use_amp = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "EfficientNet-B0 (ImageNet预训练)"
    modification = "AdamW + diff lr + warmup cosine + weakaug + class weight + label smoothing"
    train_transform_desc = "Resize256 + HFlip + RandomAffine + CenterCrop224 + ColorJitter"
    test_transform_desc = "Resize256 + CenterCrop224"


def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    return model


def build_train_transform(cfg):
    return transforms.Compose([
        transforms.Resize((cfg.train_resize, cfg.train_resize), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            interpolation=InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(cfg.img_size),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ], p=0.5),
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
    backbone_params, head_params = split_backbone_and_head(model, model.classifier[1])
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def main():
    run_experiment(
        cfg=Config(),
        build_model_fn=build_model,
        build_train_transform_fn=build_train_transform,
        build_test_transform_fn=build_test_transform,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
