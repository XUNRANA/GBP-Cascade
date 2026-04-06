"""
Exp #11: Exp#4 + 12D 形态学特征扩展

核心思路:
  - 在 Exp#4 基础上, 将 metadata 从 6D (临床) 扩展到 18D (6D 临床 + 12D 形态学)
  - 12D 形态学特征从原始 mask 提取: area_ratio, circularity, eccentricity, solidity, ...
  - meta_hidden 从 64 扩大到 128, 容纳更丰富的特征

vs Exp#4:
  + 12D 形态学特征 (from task2_lat_v1.compute_morph_features)
  + meta_dim=18, meta_hidden=128
  + 30 epochs (Exp#4 peak@10, 无需 100)
"""

import os
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v4 import (
    GBPDatasetSegCls4chWithExtMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chModelV4,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    run_experiment_with_early_stopping,
    EXT_META_FEATURE_NAMES,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260406_task2_SwinV2Tiny_segcls_11"
    log_dir = os.path.join(project_root, "0406", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型
    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(EXT_META_FEATURE_NAMES)  # 18 = 6 clinical + 12 morph
    meta_hidden = 128  # 更大容量 (原 64)
    meta_dropout = 0.2

    # 训练
    batch_size = 8
    num_epochs = 30  # Exp#4 peak@10, 无需 100
    warmup_epochs = 8
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 2
    seed = 42
    use_amp = True

    # Early stopping
    patience = 10
    use_ema = False  # 本实验不用 EMA, 单独验证形态学特征的效果

    # 损失
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + SegAttn + ExtMeta(18D=6clinical+12morph)"
    modification = (
        "Exp#4 + 12D形态学特征(area_ratio/circularity/eccentricity/solidity/...) "
        "+ meta_dim=18 + meta_hidden=128 + 30ep"
    )


def build_model(cfg):
    return SwinV2SegGuidedCls4chModelV4(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        pretrained=True,
    )


def build_dataloaders(cfg):
    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDatasetSegCls4chWithExtMeta(
        cfg.train_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=train_sync,
    )
    test_dataset = GBPDatasetSegCls4chWithExtMeta(
        cfg.test_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=test_sync,
        meta_stats=train_dataset.meta_stats,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_4ch_meta_collate_fn,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def main():
    run_experiment_with_early_stopping(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
