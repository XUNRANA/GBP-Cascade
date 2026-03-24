"""
Exp #3: Swin-V2-Tiny + ROI裁剪 + 病灶Mask 4通道 ★ 核心实验
- 4通道输入: RGB(ROI裁剪) + lesion_mask
- Swin-V2 窗口注意力，适合细粒度特征
- 第4通道 kaiming init，前3通道加载 ImageNet 预训练
"""

import os

import timm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from task2_json_utils import (
    GBPDatasetROI4ch,
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

    exp_name = "20260323_task2_SwinV2Tiny_roi4ch_3"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    batch_size = 8
    num_epochs = 60
    warmup_epochs = 5
    backbone_lr = 2e-5
    head_lr = 2e-4
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
    model_name = "Swin-V2-Tiny (ImageNet pretrained, 4ch adapted)"
    modification = "ROI crop + lesion mask 4ch + AdamW + diff lr + warmup cosine + class weight"
    train_transform_desc = "ROI_crop → SyncResize256 + HFlip + VFlip + Affine + ColorJitter + mask"
    test_transform_desc = "ROI_crop → SyncResize256 + mask"


def build_model():
    model = timm.create_model("swinv2_tiny_window8_256", pretrained=True, num_classes=2)
    adapt_model_to_4ch(model)
    return model


def build_dataloaders(cfg):
    train_sync = SyncTransform(cfg.img_size, is_train=True)
    test_sync = SyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDatasetROI4ch(cfg.train_excel, cfg.data_root, sync_transform=train_sync)
    test_dataset = GBPDatasetROI4ch(cfg.test_excel, cfg.data_root, sync_transform=test_sync)

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
