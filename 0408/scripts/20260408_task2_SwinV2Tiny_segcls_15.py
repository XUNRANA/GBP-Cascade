"""
Exp #15: Exp#4 + 扩展结构化临床特征 (6D → 10D)

核心思路:
  - 在 Exp#4 基础上, 从 Excel 新增 4 个临床特征:
    echo_type (回声0/1/2, 覆盖90.7%)
    lesion_count (个数0/1, 覆盖86.3%)
    wall_thickness (壁厚mm, 覆盖55.2%)
    us_diameter (超声直径mm, 覆盖97.9%)
  - metadata 维度: 6D → 10D
  - meta_hidden: 64 → 96 (适度增大隐藏层)
  - 其余架构、超参数完全同 Exp#4

vs Exp#4:
  + 4 个新的结构化临床特征 (覆盖率>50%)
  + meta_dim=10, meta_hidden=96
  目的: 验证结构化特征的增量价值, 作为后续实验的对照基线
"""

import os
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v5 import (
    GBPDatasetSegCls4chWithExtMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chModel,
    seg_cls_4ch_meta_collate_fn,
    build_optimizer_with_diff_lr,
    run_seg_cls_experiment_v2,
    EXT_CLINICAL_FEATURE_NAMES,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260408_task2_SwinV2Tiny_segcls_15"
    log_dir = os.path.join(project_root, "0408", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型
    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(EXT_CLINICAL_FEATURE_NAMES)  # 10
    meta_hidden = 96   # 64→96 (适配10D)
    meta_dropout = 0.2

    # 训练 (同Exp#4)
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

    # 损失
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + Seg-Guided Attention + 10D Extended Metadata"
    modification = (
        "Exp#4基础上扩展临床特征 6D→10D: "
        "新增 echo_type(回声), lesion_count(个数), wall_thickness(壁厚), us_diameter(超声直径) "
        "+ meta_hidden=96 + 其余同Exp#4"
    )
    train_transform_desc = "StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise) 4ch"
    test_transform_desc = "Resize256 4ch"


def build_model(cfg):
    return SwinV2SegGuidedCls4chModel(
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

    cfg.meta_dim = train_dataset.meta_dim  # should be 10

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
    run_seg_cls_experiment_v2(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
