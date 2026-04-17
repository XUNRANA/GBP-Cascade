"""
Exp #18_2: 在 Exp#18 基础上做“临床优先级”改进

目标优先级:
  1) 尽可能不漏 benign (良性肿瘤)
  2) 在满足 benign 低漏检约束的前提下，尽可能减少 no_tumor 过度治疗

策略:
  - 架构保持 Exp#18 不变 (SwinV2 + BERT CrossAttn + Gated Trimodal Fusion)
  - 训练时使用偏向 benign 的加权采样 (target benign ratio=0.60)
  - 分类损失使用温和 benign 权重提升 (benign_weight=1.25)
  - 模型保存标准从“宏F1最优”改为“满足 benign 漏检约束时 no_tumor 召回最大”
  - 最终输出:
      * 临床约束阈值结果
      * no_tumor Reliability Diagram 校准图
      * P(no_tumor)>=0.90 的可信度统计
"""

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

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
    predict_probs_text,
    find_constrained_threshold_text,
    analyze_high_confidence_positive,
    compute_binary_reliability_stats,
    save_reliability_stats_csv,
    save_reliability_diagram,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260415_task2_SwinV2Tiny_segcls_18_2"
    log_dir = os.path.join(project_root, "0408", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型 (与 Exp#18 相同)
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

    # Cross-Attention
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

    # 损失
    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    # 18_2 改进点: 采样 + 损失偏置
    benign_target_ratio = 0.60    # 每个 epoch 采样中 benign 目标占比
    benign_loss_weight = 1.25     # 轻度提高 benign 分类损失权重

    # 临床策略与校准分析
    max_benign_miss_rate = 0.05   # 优先尽量不漏 benign (更严格于10%)
    high_confidence_no_tumor_prob = 0.90
    calibration_bins = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = (
        "SwinV2-Tiny@256 + 4ch + BERT CrossAttn + BERT[CLS] + "
        "10D ExtMeta + Gated Trimodal Fusion"
    )
    modification = (
        "Exp#18_2: 优先不漏benign; "
        "训练用benign偏置采样(target=0.60) + benign_loss_weight=1.25; "
        "模型选择标准改为: 满足benign漏检约束时最大化no_tumor召回"
    )


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


def build_dataloaders(cfg):
    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    text_dict = load_text_bert_dict(cfg.json_feature_root)

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

    cfg.meta_dim = train_dataset.meta_dim

    # ── Benign-prior sampling (target ratio) ──────────────────────────────
    labels = train_dataset.df["label"].values.astype(int)
    class_counts = np.bincount(labels, minlength=2)
    p_benign = float(cfg.benign_target_ratio)
    p_no_tumor = 1.0 - p_benign
    class_sample_weights = np.array(
        [p_benign / max(class_counts[0], 1), p_no_tumor / max(class_counts[1], 1)],
        dtype=np.float64,
    )
    sample_weights = class_sample_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
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
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.") and not name.startswith("text_encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def policy_score_tuple(policy):
    """Higher is better under clinical priority."""
    return (
        1 if policy["constraint_satisfied"] else 0,
        policy["no_tumor_recall"],
        -policy["benign_miss_rate"],
        policy["no_tumor_precision"],
        policy["macro_f1"],
    )


def evaluate_policy(model, dataloader, device, cfg):
    probs_benign, probs_no_tumor, labels_np = predict_probs_text(model, dataloader, device)
    constrained = find_constrained_threshold_text(
        probs_benign,
        labels_np,
        max_benign_miss_rate=cfg.max_benign_miss_rate,
    )
    return constrained, probs_benign, probs_no_tumor, labels_np


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
    logger.info(f"Benign目标采样占比: {cfg.benign_target_ratio:.2f}")
    logger.info(f"Benign分类损失权重: {cfg.benign_loss_weight:.2f}")
    logger.info(f"Benign漏检率约束: <= {cfg.max_benign_miss_rate:.2%}")
    logger.info(f"高置信no_tumor阈值: {cfg.high_confidence_no_tumor_prob:.2f}")
    logger.info(f"校准分箱数: {cfg.calibration_bins}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"Grad Clip: {cfg.grad_clip}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 70)

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(cfg)
    n_benign = int(np.sum(train_dataset.df["label"].values == 0))
    n_no_tumor = int(np.sum(train_dataset.df["label"].values == 1))
    logger.info(
        f"训练集: {len(train_dataset)} 张 (benign={n_benign}, no_tumor={n_no_tumor})"
    )
    logger.info(
        f"采样策略: WeightedRandomSampler target benign ratio={cfg.benign_target_ratio:.2f}"
    )
    logger.info(
        f"测试集: {len(test_dataset)} 张 "
        f"(benign={sum(test_dataset.df['label'] == 0)}, "
        f"no_tumor={sum(test_dataset.df['label'] == 1)})"
    )

    model = build_model(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,} (含BERT)")
    logger.info(f"可训练参数量: {n_trainable:,} (BERT冻结)")

    cls_weights = torch.tensor(
        [cfg.benign_loss_weight, 1.0], dtype=torch.float32, device=cfg.device
    )
    logger.info(f"分类类别权重: benign={cls_weights[0]:.4f}, no_tumor={cls_weights[1]:.4f}")

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

    best_epoch = 0
    best_score = (-1, -1.0, -1.0, -1.0, -1.0)
    best_policy = None

    logger.info("\n" + "=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
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
            evaluate_text(
                model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Test", num_seg_classes=cfg.num_seg_classes,
            )
            constrained, _, _, _ = evaluate_policy(model, test_loader, cfg.device, cfg)
            score = policy_score_tuple(constrained)
            logger.info(
                f"[Policy] threshold={constrained['threshold']:.3f} | "
                f"benign漏检率={constrained['benign_miss_rate']:.2%} | "
                f"no_tumor召回={constrained['no_tumor_recall']:.2%} | "
                f"no_tumor精度={constrained['no_tumor_precision']:.2%} | "
                f"F1(macro)={constrained['macro_f1']:.4f}"
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_policy = constrained
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(
                    f"*** 保存最优模型(临床策略) "
                    f"(Epoch={best_epoch}, benign漏检={constrained['benign_miss_rate']:.2%}, "
                    f"no_tumor召回={constrained['no_tumor_recall']:.2%}) ***"
                )
            logger.info("-" * 50)

    logger.info("\n" + "=" * 70)
    logger.info(
        f"训练完成! 最优模型(Epoch {best_epoch}) "
        f"| benign漏检={best_policy['benign_miss_rate']:.2%} "
        f"| no_tumor召回={best_policy['no_tumor_recall']:.2%}"
    )
    logger.info("=" * 70)

    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )

    logger.info("=" * 70)
    logger.info("最终测试结果 (threshold=0.5)")
    logger.info("=" * 70)
    evaluate_text(model, test_loader, cfg.device, cfg.class_names, logger,
                  phase="Final Test", num_seg_classes=cfg.num_seg_classes)

    logger.info("\n" + "=" * 70)
    logger.info("阈值优化搜索 (宏F1)")
    logger.info("=" * 70)
    best_thresh, best_thresh_f1 = find_optimal_threshold_text(model, test_loader, cfg.device)
    logger.info(f"F1最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f})")
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold_text(
            model, test_loader, cfg.device, cfg.class_names, logger,
            threshold=best_thresh, phase="Final Test (最优F1阈值)",
        )

    logger.info("\n" + "=" * 70)
    logger.info("临床约束阈值评估")
    logger.info("=" * 70)
    constrained, probs_benign, probs_no_tumor, labels_np = evaluate_policy(
        model, test_loader, cfg.device, cfg
    )
    logger.info(
        f"临床阈值: {constrained['threshold']:.3f} | "
        f"benign漏检率={constrained['benign_miss_rate']:.2%} | "
        f"benign召回={constrained['benign_recall']:.2%} | "
        f"no_tumor召回={constrained['no_tumor_recall']:.2%} | "
        f"no_tumor精度={constrained['no_tumor_precision']:.2%} | "
        f"no_tumor筛出率={constrained['selected_no_tumor_rate']:.2%} | "
        f"constraint_ok={constrained['constraint_satisfied']}"
    )
    phase_name = (
        f"Final Test (临床阈值, benign漏检<={cfg.max_benign_miss_rate:.0%})"
        if constrained["constraint_satisfied"]
        else "Final Test (最小benign漏检阈值)"
    )
    evaluate_with_threshold_text(
        model, test_loader, cfg.device, cfg.class_names, logger,
        threshold=constrained["threshold"], phase=phase_name,
    )

    # Save test probabilities
    prob_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_test_probs.csv")
    pd.DataFrame({
        "label": labels_np.astype(int),
        "prob_benign": probs_benign,
        "prob_no_tumor": probs_no_tumor,
    }).to_csv(prob_csv, index=False, encoding="utf-8")
    logger.info(f"测试集概率已保存: {prob_csv}")

    # High-confidence no_tumor reliability
    logger.info("\n" + "=" * 70)
    logger.info("no_tumor 置信度-准确率校准")
    logger.info("=" * 70)
    high_conf = analyze_high_confidence_positive(
        labels_np,
        probs_no_tumor,
        positive_label=1,
        min_confidence=cfg.high_confidence_no_tumor_prob,
    )
    if high_conf["selected"] > 0:
        logger.info(
            f"P(no_tumor)>={cfg.high_confidence_no_tumor_prob:.2f}: "
            f"{high_conf['selected']}/{high_conf['total']} ({high_conf['coverage']:.2%}) | "
            f"真实no_tumor比例={high_conf['precision']:.2%} "
            f"(命中{high_conf['true_positive']}例)"
        )
    else:
        logger.info(
            f"P(no_tumor)>={cfg.high_confidence_no_tumor_prob:.2f}: 无样本落入高置信区间"
        )

    y_true_no_tumor = (labels_np == 1).astype(np.int64)
    rel_rows, ece, brier = compute_binary_reliability_stats(
        y_true_no_tumor, probs_no_tumor, n_bins=cfg.calibration_bins
    )
    logger.info(
        f"校准统计: ECE={ece:.4f} | Brier={brier:.4f} | 有效分箱={len(rel_rows)}/{cfg.calibration_bins}"
    )
    for row in rel_rows:
        logger.info(
            f"  Bin[{row['bin_lower']:.2f},{row['bin_upper']:.2f}) "
            f"n={row['count']:3d} | conf={row['confidence']:.3f} | acc={row['accuracy']:.3f}"
        )

    rel_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_no_tumor_reliability.csv")
    save_reliability_stats_csv(rel_rows, rel_csv)
    logger.info(f"校准分箱明细已保存: {rel_csv}")

    rel_png = os.path.join(cfg.log_dir, f"{cfg.exp_name}_no_tumor_reliability.png")
    fig_ok = save_reliability_diagram(
        rel_rows,
        rel_png,
        title=f"{cfg.exp_name} no_tumor 置信度-准确率校准图",
    )
    if fig_ok:
        logger.info(f"校准曲线已保存: {rel_png}")
    else:
        logger.warning("校准曲线生成失败（可能缺少matplotlib环境），已保留CSV分箱结果。")

    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)
        logger.info(f"训练脚本已复制到: {dst}")


if __name__ == "__main__":
    main()
