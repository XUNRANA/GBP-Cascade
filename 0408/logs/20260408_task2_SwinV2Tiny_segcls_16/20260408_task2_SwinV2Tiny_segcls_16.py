"""
Exp #16: Exp#4 + BERT超声报告编码 + 后期拼接融合 (Late Fusion)

核心思路:
  - 首次引入超声报告文本 (text_bert), 验证文本语义对分类的增益
  - BERT: bert-base-chinese (冻结全部参数, 仅训练投影层)
  - 文本编码: text_bert → BERT → [CLS] 768D → Linear投影 → 128D
  - 融合方式: 后期拼接 img_feat(256D) || text_feat(128D) || meta_feat(64D) = 448D
  - 保留6D元数据 (不扩展到10D), 独立验证文本的增量价值

vs Exp#4:
  + BERT文本编码分支 (冻结, [CLS] → 128D)
  + 融合维度 320D → 448D
  + 文本感知的训练循环
  目的: 验证超声报告文本的增量价值
"""

import os
import sys

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v5 import (
    GBPDatasetSegCls4chWithTextMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chWithText,
    seg_cls_text_collate_fn,
    build_optimizer_with_diff_lr,
    run_seg_cls_experiment_text,
    load_text_bert_dict,
    META_FEATURE_NAMES,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260408_task2_SwinV2Tiny_segcls_16"
    log_dir = os.path.join(project_root, "0408", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型
    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(META_FEATURE_NAMES)  # 6 (保持原始, 独立验证文本增益)
    meta_hidden = 64
    meta_dropout = 0.2
    text_proj_dim = 128
    text_dropout = 0.3
    max_text_len = 128

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
    model_name = "SwinV2-Tiny@256 + 4ch + Seg-Guided Attn + BERT[CLS] Late Fusion + 6D Meta"
    modification = (
        "Exp#4 + BERT-base-chinese(冻结)编码超声报告 "
        "→ [CLS] 768D → proj 128D + img 256D + meta 64D = 448D late fusion"
    )
    train_transform_desc = "StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise) 4ch"
    test_transform_desc = "Resize256 4ch"


def build_model(cfg):
    return SwinV2SegGuidedCls4chWithText(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        text_proj_dim=cfg.text_proj_dim,
        text_dropout=cfg.text_dropout,
        bert_name="bert-base-chinese",
        pretrained=True,
    )


def build_dataloaders(cfg):
    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    # 预加载文本字典 (共享, 避免重复IO)
    text_dict = load_text_bert_dict(cfg.json_feature_root)

    train_dataset = GBPDatasetSegCls4chWithTextMeta(
        cfg.train_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=train_sync,
        meta_feature_names=list(META_FEATURE_NAMES),
        text_dict=text_dict,
        max_text_len=cfg.max_text_len,
    )
    test_dataset = GBPDatasetSegCls4chWithTextMeta(
        cfg.test_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=test_sync,
        meta_stats=train_dataset.meta_stats,
        meta_feature_names=list(META_FEATURE_NAMES),
        text_dict=text_dict,
        tokenizer=train_dataset.tokenizer,
        max_text_len=cfg.max_text_len,
    )

    cfg.meta_dim = train_dataset.meta_dim

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=seg_cls_text_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_text_collate_fn,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    # BERT frozen → only update encoder (backbone) + non-encoder/non-BERT params
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.") and not name.startswith("text_encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def main():
    run_seg_cls_experiment_text(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
