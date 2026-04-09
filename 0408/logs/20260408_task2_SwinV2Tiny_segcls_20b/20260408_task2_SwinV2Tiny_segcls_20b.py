"""
Exp #20b: 物理欠采样 1:1 + Exp#18 全模态门控融合

核心思路:
  - 与 Exp#18 完全相同的模型架构
  - 物理欠采样: 训练集 309 benign + 309 no_tumor = 618 张
  - 测试集也平衡: 129 benign + 129 no_tumor = 258 张
  - class_weight 设为 [1.0, 1.0] (数据已平衡)

vs Exp#18:
  - 训练集: 1229 → 618 (欠采样 no_tumor)
  - 测试集: 523 → 258 (欠采样 no_tumor)
  - cls class_weight: [1.986, 0.668] → [1.0, 1.0]
"""

import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from seg_cls_utils_v5 import (
    GBPDatasetSegCls4chWithTextMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chTrimodal,
    seg_cls_text_collate_fn,
    SegClsLoss,
    build_optimizer_with_diff_lr,
    load_text_bert_dict,
    EXT_CLINICAL_FEATURE_NAMES,
    set_seed,
    setup_logger,
    acquire_run_lock,
    set_epoch_lrs,
    train_one_epoch_text,
    evaluate_text,
    find_optimal_threshold_text,
    evaluate_with_threshold_text,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260408_task2_SwinV2Tiny_segcls_20b"
    log_dir = os.path.join(project_root, "0408", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型 (与 Exp#18 完全相同)
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

    # 训练 (与 Exp#18 完全相同)
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

    # 损失
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = (
        "SwinV2-Tiny@256 + 4ch + BERT CrossAttn + BERT[CLS] + "
        "10D ExtMeta + Gated Trimodal Fusion + 1:1 Physical Undersampling"
    )
    modification = (
        "与Exp#18相同架构, "
        "物理欠采样: train 309:309=618, test 129:129=258, "
        "cls class_weight=[1.0,1.0]"
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


def _balance_df(df, seed=42):
    """欠采样多数类, 使两类数量与少数类一致."""
    rng = np.random.RandomState(seed)
    benign = df[df["label"] == 0]
    notumor = df[df["label"] == 1]
    n_min = min(len(benign), len(notumor))
    if len(benign) > n_min:
        benign = benign.sample(n=n_min, random_state=rng)
    if len(notumor) > n_min:
        notumor = notumor.sample(n=n_min, random_state=rng)
    return benign._append(notumor).reset_index(drop=True)


def build_dataloaders(cfg):
    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    text_dict = load_text_bert_dict(cfg.json_feature_root)

    # 先创建完整 dataset (为了加载 meta 表和 tokenizer)
    train_dataset = GBPDatasetSegCls4chWithTextMeta(
        cfg.train_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=train_sync,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        max_text_len=cfg.max_text_len,
    )
    test_dataset = GBPDatasetSegCls4chWithTextMeta(
        cfg.test_excel, cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=test_sync,
        meta_stats=train_dataset.meta_stats,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        tokenizer=train_dataset.tokenizer,
        max_text_len=cfg.max_text_len,
    )

    # ── 物理欠采样: 训练集 309:309, 测试集 129:129 ──
    train_dataset.df = _balance_df(train_dataset.df, seed=cfg.seed)
    test_dataset.df = _balance_df(test_dataset.df, seed=cfg.seed)

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
    cfg = Config()
    os.makedirs(cfg.log_dir, exist_ok=True)
    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_file, cfg.exp_name)

    logger.info("=" * 70)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"输入通道: {cfg.in_channels}")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"分割类别: {cfg.num_seg_classes}")
    logger.info(f"分类类别: {cfg.class_names}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Backbone LR: {cfg.backbone_lr}")
    logger.info(f"Head LR: {cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Warmup Epochs: {cfg.warmup_epochs}")
    logger.info(f"Lambda Cls: {cfg.lambda_cls}")
    logger.info(f"Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"Grad Clip: {cfg.grad_clip}")
    logger.info(f"Meta Dim: {cfg.meta_dim}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 70)

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(cfg)

    n_benign_train = sum(train_dataset.df["label"] == 0)
    n_notumor_train = sum(train_dataset.df["label"] == 1)
    n_benign_test = sum(test_dataset.df["label"] == 0)
    n_notumor_test = sum(test_dataset.df["label"] == 1)
    logger.info(
        f"训练集 (欠采样后): {len(train_dataset)} 张 "
        f"(benign={n_benign_train}, no_tumor={n_notumor_train})"
    )
    logger.info(
        f"测试集 (欠采样后): {len(test_dataset)} 张 "
        f"(benign={n_benign_test}, no_tumor={n_notumor_test})"
    )

    model = build_model(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,} (含BERT)")
    logger.info(f"可训练参数量: {n_trainable:,} (BERT冻结)")

    # ── class_weight = [1.0, 1.0] (数据已平衡) ──
    cls_weights = torch.tensor([1.0, 1.0], dtype=torch.float32, device=cfg.device)
    logger.info(f"分类类别权重: benign={cls_weights[0]:.4f}, no_tumor={cls_weights[1]:.4f} (数据已平衡)")

    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight], dtype=torch.float32, device=cfg.device
    )
    logger.info(f"分割类别权重: bg={cfg.seg_bg_weight}, lesion={cfg.seg_lesion_weight}")

    criterion = SegClsLoss(
        cls_weights=cls_weights,
        lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing,
        seg_ce_weight=seg_ce_weight,
    )

    optimizer = build_optimizer(model, cfg)

    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    best_f1, best_epoch = 0.0, 0

    logger.info("\n" + "=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        lr_factor = set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        train_metrics = train_one_epoch_text(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f}, cls={train_metrics['cls_loss']:.4f}) "
            f"| Cls Acc: {train_metrics['cls_acc']:.4f} "
            f"| Seg IoU: {train_metrics['seg_iou']:.4f} "
            f"| Seg Dice: {train_metrics['seg_dice']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 50)
            acc, prec, rec, f1, seg_iou, seg_dice = evaluate_text(
                model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Test", num_seg_classes=cfg.num_seg_classes,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(
                    f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***"
                )
            logger.info("-" * 50)

    logger.info("\n" + "=" * 70)
    logger.info(f"训练完成! 最优模型: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 70)

    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )
    logger.info("=" * 70)
    logger.info("最终测试结果 (最优权重, threshold=0.5)")
    logger.info("=" * 70)
    evaluate_text(model, test_loader, cfg.device, cfg.class_names, logger,
                  phase="Final Test", num_seg_classes=cfg.num_seg_classes)

    logger.info("\n" + "=" * 70)
    logger.info("阈值优化搜索")
    logger.info("=" * 70)
    best_thresh, best_thresh_f1 = find_optimal_threshold_text(model, test_loader, cfg.device)
    logger.info(
        f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认0.5 F1: {best_f1:.4f})"
    )
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold_text(
            model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=best_thresh, phase="Final Test (最优阈值)",
        )

    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)
        logger.info(f"训练脚本已复制到: {dst}")


if __name__ == "__main__":
    main()
