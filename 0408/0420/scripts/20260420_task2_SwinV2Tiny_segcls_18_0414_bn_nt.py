"""
Exp #18 (0420版): 全模态融合, 数据改为 0414dataset 的二分类子集

目标:
  - 基于原 Exp#18 模型结构 (SwinV2 + BERT CrossAttn + BERT[CLS] + 10D临床 + 门控融合)
  - 数据来源: 0414dataset/task_3class_{train,test}.xlsx
  - 仅保留 benign/no_tumor:
      原始标签映射: {'malignant': 0, 'benign': 1, 'no_tumor': 2}
      训练标签重映射: benign(1)->0, no_tumor(2)->1
  - 在 0408/0420 目录下产出本次训练日志与权重
"""

import os
import sys

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_0408_SCRIPTS = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "scripts"))
if BASE_0408_SCRIPTS not in sys.path:
    sys.path.insert(0, BASE_0408_SCRIPTS)

from seg_cls_utils_v5 import (
    GBPDatasetSegCls4chWithTextMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chTrimodal,
    seg_cls_text_collate_fn,
    build_optimizer_with_diff_lr,
    run_seg_cls_experiment_text,
    load_text_bert_dict,
    EXT_CLINICAL_FEATURE_NAMES,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0414dataset")

    raw_train_excel = os.path.join(data_root, "task_3class_train.xlsx")
    raw_test_excel = os.path.join(data_root, "task_3class_test.xlsx")

    prepared_excel_dir = os.path.join(project_root, "0408", "0420", "prepared_excels")
    train_excel = os.path.join(prepared_excel_dir, "task_2_bn_notumor_train.xlsx")
    test_excel = os.path.join(prepared_excel_dir, "task_2_bn_notumor_test.xlsx")

    # 0414dataset 标签映射: malignant=0, benign=1, no_tumor=2
    keep_label_map = {1: 0, 2: 1}  # benign->0, no_tumor->1

    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260420_task2_SwinV2Tiny_segcls_18_0414_bn_nt"
    log_dir = os.path.join(project_root, "0408", "0420", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型
    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(EXT_CLINICAL_FEATURE_NAMES)  # 10
    meta_hidden = 96
    meta_dropout = 0.2
    text_proj_dim = 128
    text_dropout = 0.3
    max_text_len = 128
    fusion_dim = 256

    # Cross-Attention 参数
    ca_hidden = 128
    ca_heads = 4
    ca_dropout = 0.1

    # 训练
    batch_size = 8
    num_epochs = 100
    warmup_epochs = 10
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

    # 临床策略与校准分析
    max_benign_miss_rate = 0.10
    high_confidence_no_tumor_prob = 0.90
    calibration_bins = 10

    # 损失
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = (
        "SwinV2-Tiny@256 + 4ch + BERT CrossAttn + BERT[CLS] + "
        "10D ExtMeta + Gated Trimodal Fusion"
    )
    modification = (
        "全模态融合 + 0414dataset二分类过滤: "
        "保留benign/no_tumor, 映射 benign(1)->0, no_tumor(2)->1; "
        "img(CrossAttn+SegAttn→256D) + text(BERT[CLS]→128D) + clinical(10D→96D) "
        "→ GatedTrimodalFusion → 256D → MLP → 2类 + warmup=10ep"
    )
    train_transform_desc = "StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise) 4ch"
    test_transform_desc = "Resize256 4ch"


def build_model(cfg):
    return SwinV2SegGuidedCls4chTrimodal(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        text_proj_dim=cfg.text_proj_dim,
        text_dropout=cfg.text_dropout,
        ca_hidden=cfg.ca_hidden,
        ca_heads=cfg.ca_heads,
        ca_dropout=cfg.ca_dropout,
        fusion_dim=cfg.fusion_dim,
        bert_name="bert-base-chinese",
        pretrained=True,
    )


def _prepare_binary_excel(src_excel, dst_excel, label_map, split_name):
    df = pd.read_excel(src_excel).copy()
    if "label" not in df.columns:
        raise KeyError(f"{src_excel} 缺少 label 列")

    keep_labels = set(label_map.keys())
    df = df[df["label"].isin(keep_labels)].copy()
    if df.empty:
        raise ValueError(f"{split_name} 过滤后为空: {src_excel}")

    df["label"] = df["label"].map(label_map)
    if df["label"].isna().any():
        raise ValueError(f"{split_name} 标签映射失败: {src_excel}")

    df["label"] = df["label"].astype(int)
    df = df.reset_index(drop=True)
    df.to_excel(dst_excel, index=False)

    n_benign = int((df["label"] == 0).sum())
    n_no_tumor = int((df["label"] == 1).sum())
    print(f"[{split_name}] 已生成二分类Excel: {dst_excel}")
    print(f"[{split_name}] 样本数={len(df)} (benign={n_benign}, no_tumor={n_no_tumor})")


def prepare_binary_excels(cfg):
    os.makedirs(cfg.prepared_excel_dir, exist_ok=True)
    _prepare_binary_excel(cfg.raw_train_excel, cfg.train_excel, cfg.keep_label_map, "Train")
    _prepare_binary_excel(cfg.raw_test_excel, cfg.test_excel, cfg.keep_label_map, "Test")


def build_dataloaders(cfg):
    prepare_binary_excels(cfg)

    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    text_dict = load_text_bert_dict(cfg.json_feature_root)

    train_dataset = GBPDatasetSegCls4chWithTextMeta(
        cfg.train_excel,
        cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=train_sync,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        max_text_len=cfg.max_text_len,
    )
    test_dataset = GBPDatasetSegCls4chWithTextMeta(
        cfg.test_excel,
        cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=test_sync,
        meta_stats=train_dataset.meta_stats,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        tokenizer=train_dataset.tokenizer,
        max_text_len=cfg.max_text_len,
    )

    cfg.meta_dim = train_dataset.meta_dim

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_cls_text_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=seg_cls_text_collate_fn,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad
        and not name.startswith("encoder.")
        and not name.startswith("text_encoder.")
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
