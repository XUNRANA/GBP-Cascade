"""
Exp #17: Exp#4 + BERT文本 + 交叉注意力融合 (Cross-Attention)

核心思路:
  - BERT输出不仅用[CLS], 而是用全部token的hidden states (seq_len × 768D)
  - 图像空间特征 f2 (B, 256, 16, 16) 作为 Query
  - 文本token特征作为 Key 和 Value
  - Cross-Attention让图像每个空间位置"查询"最相关的文本描述
  - 交叉注意力放在seg-guided attention之前, 让文本增强图像空间特征
  - 保留残差连接: 即使文本无用, 模型也能退化为原始Exp#4

vs Exp#16:
  - Exp#16: 仅用[CLS]做后期拼接 (late fusion)
  - Exp#17: 用全部token做交叉注意力 (deep fusion), 建模文本-图像空间对应
  目的: 验证深层文本-图像交互的价值
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
    SwinV2SegGuidedCls4chCrossAttn,
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

    exp_name = "20260408_task2_SwinV2Tiny_segcls_17"
    log_dir = os.path.join(project_root, "0408", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型
    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(META_FEATURE_NAMES)  # 6
    meta_hidden = 64
    meta_dropout = 0.2
    max_text_len = 128

    # Cross-Attention 参数
    ca_hidden = 128
    ca_heads = 4
    ca_dropout = 0.1

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
    model_name = "SwinV2-Tiny@256 + 4ch + BERT CrossAttn + Seg-Guided Attn + 6D Meta"
    modification = (
        "Exp#4 + BERT(冻结)全token交叉注意力: "
        "f2(256,16×16)=Q, text_hidden(seq,768)=K,V "
        "→ enhanced_f2 → seg-guided attention → 256D + meta 64D = 320D"
    )
    train_transform_desc = "StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise) 4ch"
    test_transform_desc = "Resize256 4ch"


def build_model(cfg):
    return SwinV2SegGuidedCls4chCrossAttn(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        ca_hidden=cfg.ca_hidden,
        ca_heads=cfg.ca_heads,
        ca_dropout=cfg.ca_dropout,
        bert_name="bert-base-chinese",
        pretrained=True,
    )


def build_dataloaders(cfg):
    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)

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
