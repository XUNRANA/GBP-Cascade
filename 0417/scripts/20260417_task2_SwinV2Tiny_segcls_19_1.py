"""
Exp #19-1: Exp#18-3 骨架 + pAUC Surrogate Loss + Dev Split

改进点：
  1. 新增 10% 分层 dev split（全程盲盒 test）
  2. pAUC Surrogate Loss（staged coeff: stage1=0, stage2=0.5, stage3=1.0）
  3. 模型选择改为 dev pAUC@FPR≤0.05（非 test F1）
  4. 训练后: dev 上温标定，保存 uncal/calibrated probs + thresholds.json

GPU: 0  预计墙钟 ~2.5h
"""

import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler

_ROOT = "/data1/ouyangxinglong/GBP-Cascade"
sys.path.insert(0, os.path.join(_ROOT, "0408", "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "0417", "scripts"))

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
    _unpack_text_batch,
    compute_seg_metrics,
)
from seg_cls_utils_v5_ext import (
    stratified_dev_split,
    save_dev_split_json,
    load_dev_split_json,
    SubsetDataset,
    PAUCSurrogateLoss,
    TemperatureScaler,
    predict_logits_text,
    compute_pauc_at_fpr,
    save_probs_csv,
    compute_and_save_thresholds,
    log_threshold_results,
)


class Config:
    project_root = _ROOT
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260417_task2_SwinV2Tiny_segcls_19_1"
    log_dir = os.path.join(project_root, "0417", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # Dev split
    dev_split_json = os.path.join(project_root, "0417", "logs", "shared", "dev_split.json")
    dev_frac = 0.10

    # 模型（与 Exp#18 相同）
    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(EXT_CLINICAL_FEATURE_NAMES)
    meta_hidden = 96
    meta_dropout = 0.2
    text_proj_dim = 128
    text_dropout = 0.3
    max_text_len = 128
    fusion_dim = 256
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

    # 三阶段 curriculum（与 18-3 完全相同）
    stage2_start_epoch = 41
    stage3_start_epoch = 81
    stage1_benign_target_ratio = 0.72
    stage1_benign_loss_weight = 1.40
    stage2_benign_target_ratio = 0.62
    stage2_benign_loss_weight = 1.22
    stage3_benign_target_ratio = 0.55
    stage3_benign_loss_weight = 1.08
    stage3_backbone_lr_scale = 0.80
    stage3_head_lr_scale = 0.90

    # pAUC Loss（Exp#19-1 新增）
    pauc_K = 4
    pauc_margin = 0.1
    pauc_coeff_stage1 = 0.0
    pauc_coeff_stage2 = 0.5
    pauc_coeff_stage3 = 1.0

    # 临床策略与校准
    max_benign_miss_rate = 0.05
    high_confidence_no_tumor_prob = 0.90
    calibration_bins = 10

    # 早停（按 dev pAUC）
    policy_patience = 8
    min_epochs_for_early_stop = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = (
        "SwinV2-Tiny@256 + 4ch + BERT CrossAttn + BERT[CLS] + "
        "10D ExtMeta + Gated Trimodal Fusion"
    )
    modification = (
        "Exp#19-1: 18-3 三阶段curriculum + pAUC SurrogateLoss(staged) "
        "+ dev split 10% + 模型选择→dev pAUC@0.05 + 温标定"
    )


# ═══════════════════════════════════════════════════════════════════════
#  Helper functions（与 18-3 相同部分）
# ═══════════════════════════════════════════════════════════════════════

def get_stage_params(epoch, cfg):
    if epoch < cfg.stage2_start_epoch:
        return "stage1", cfg.stage1_benign_target_ratio, cfg.stage1_benign_loss_weight, cfg.pauc_coeff_stage1
    if epoch < cfg.stage3_start_epoch:
        return "stage2", cfg.stage2_benign_target_ratio, cfg.stage2_benign_loss_weight, cfg.pauc_coeff_stage2
    return "stage3", cfg.stage3_benign_target_ratio, cfg.stage3_benign_loss_weight, cfg.pauc_coeff_stage3


def build_train_loader_with_ratio(train_subset, cfg, benign_target_ratio):
    labels = train_subset.df["label"].values.astype(int)
    class_counts = np.bincount(labels, minlength=2)
    p_b = float(benign_target_ratio)
    p_nt = 1.0 - p_b
    class_sample_weights = np.array(
        [p_b / max(class_counts[0], 1), p_nt / max(class_counts[1], 1)],
        dtype=np.float64,
    )
    sample_weights = class_sample_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(train_subset), replacement=True
    )
    return DataLoader(
        train_subset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_cls_text_collate_fn,
    )


def build_datasets(cfg):
    train_sync = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)
    text_dict = load_text_bert_dict(cfg.json_feature_root)

    full_train_dataset = GBPDatasetSegCls4chWithTextMeta(
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
        meta_stats=full_train_dataset.meta_stats,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        tokenizer=full_train_dataset.tokenizer,
        max_text_len=cfg.max_text_len,
    )
    cfg.meta_dim = full_train_dataset.meta_dim
    return full_train_dataset, test_dataset


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


def build_optimizer(model, cfg):
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.") and not name.startswith("text_encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def build_criterion(cfg, benign_loss_weight):
    cls_weights = torch.tensor(
        [float(benign_loss_weight), 1.0], dtype=torch.float32, device=cfg.device
    )
    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight], dtype=torch.float32, device=cfg.device
    )
    return SegClsLoss(
        cls_weights=cls_weights,
        lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing,
        seg_ce_weight=seg_ce_weight,
    )


def train_epoch_with_pauc(
    model, dataloader, criterion, pauc_fn, pauc_coeff,
    optimizer, device, scaler, use_amp, grad_clip, num_seg_classes,
):
    """与 train_one_epoch_text 相同，额外加 pAUC 项。pauc_coeff=0 退化为原版。"""
    model.train()
    running_loss = running_seg = running_cls = 0.0
    cls_correct = cls_total = 0
    all_ious, all_dices = [], []

    for batch in dataloader:
        imgs, masks, metas, iids, amask, labels, has_masks = _unpack_text_batch(batch)
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        iids = iids.to(device, non_blocking=True)
        amask = amask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu", enabled=use_amp
        ):
            seg_l, cls_l = model(imgs, metadata=metas, input_ids=iids, attention_mask=amask)
            loss, seg_v, cls_v = criterion(seg_l, cls_l, masks, labels, has_masks)
            if pauc_coeff > 0:
                loss = loss + pauc_coeff * pauc_fn(cls_l, labels)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        running_seg += seg_v * bs
        running_cls += cls_v * bs
        cls_correct += (cls_l.argmax(dim=1) == labels).sum().item()
        cls_total += bs

        if has_masks.any():
            with torch.no_grad():
                idx = has_masks.nonzero(as_tuple=True)[0]
                m = compute_seg_metrics(seg_l[idx], masks[idx], num_seg_classes)
                all_ious.append(m["lesion_IoU"])
                all_dices.append(m["lesion_Dice"])

    n = max(cls_total, 1)
    return {
        "loss": running_loss / n,
        "seg_loss": running_seg / n,
        "cls_loss": running_cls / n,
        "cls_acc": cls_correct / n,
        "seg_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "seg_dice": float(np.mean(all_dices)) if all_dices else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════

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
    logger.info(f"Seed: {cfg.seed} | Device: {cfg.device} | AMP: {cfg.use_amp}")
    logger.info(f"Stage1/2/3 benign采样: {cfg.stage1_benign_target_ratio}/{cfg.stage2_benign_target_ratio}/{cfg.stage3_benign_target_ratio}")
    logger.info(f"Stage1/2/3 benign权重: {cfg.stage1_benign_loss_weight}/{cfg.stage2_benign_loss_weight}/{cfg.stage3_benign_loss_weight}")
    logger.info(f"pAUC coeff Stage1/2/3: {cfg.pauc_coeff_stage1}/{cfg.pauc_coeff_stage2}/{cfg.pauc_coeff_stage3}")
    logger.info(f"Dev frac: {cfg.dev_frac} | split JSON: {cfg.dev_split_json}")
    logger.info("=" * 70)

    # ── 数据 ──────────────────────────────────────────────────────
    full_train_dataset, test_dataset = build_datasets(cfg)

    if os.path.exists(cfg.dev_split_json):
        train_idx, dev_idx = load_dev_split_json(cfg.dev_split_json)
        logger.info(f"加载已有 dev split: {cfg.dev_split_json} (train={len(train_idx)}, dev={len(dev_idx)})")
    else:
        train_idx, dev_idx = stratified_dev_split(
            full_train_dataset.df, dev_frac=cfg.dev_frac, seed=cfg.seed
        )
        save_dev_split_json(train_idx, dev_idx, cfg.dev_split_json)
        logger.info(f"新建 dev split 并保存: {cfg.dev_split_json} (train={len(train_idx)}, dev={len(dev_idx)})")

    train_subset = SubsetDataset(full_train_dataset, train_idx)
    dev_subset = SubsetDataset(full_train_dataset, dev_idx)

    dev_loader = DataLoader(
        dev_subset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
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

    n_benign_tr = int(np.sum(train_subset.df["label"].values == 0))
    n_nt_tr = int(np.sum(train_subset.df["label"].values == 1))
    n_benign_dev = int(np.sum(dev_subset.df["label"].values == 0))
    n_nt_dev = int(np.sum(dev_subset.df["label"].values == 1))
    logger.info(f"训练子集: {len(train_subset)} 张 (benign={n_benign_tr}, no_tumor={n_nt_tr})")
    logger.info(f"Dev  子集: {len(dev_subset)} 张  (benign={n_benign_dev}, no_tumor={n_nt_dev})")
    logger.info(f"测试集:    {len(test_dataset)} 张  (benign={sum(test_dataset.df['label']==0)}, no_tumor={sum(test_dataset.df['label']==1)})")

    # ── 模型 & 优化器 ──────────────────────────────────────────────
    model = build_model(cfg).to(cfg.device)
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type, enabled=(cfg.device.type == "cuda" and cfg.use_amp)
    )
    pauc_fn = PAUCSurrogateLoss(K=cfg.pauc_K, margin=cfg.pauc_margin)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,} | 可训练: {n_trainable:,} (BERT冻结)")

    # ── 训练循环 ───────────────────────────────────────────────────
    best_dev_pauc = -1.0
    best_epoch = 0
    no_improve_evals = 0

    logger.info("\n" + "=" * 70)
    logger.info("开始训练（模型选择: dev pAUC@0.05）")
    logger.info("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        stage_name, benign_ratio, benign_weight, pauc_coeff = get_stage_params(epoch, cfg)
        train_loader = build_train_loader_with_ratio(train_subset, cfg, benign_ratio)
        criterion = build_criterion(cfg, benign_weight)

        set_epoch_lrs(optimizer, epoch, cfg)
        if stage_name == "stage3":
            optimizer.param_groups[0]["lr"] *= cfg.stage3_backbone_lr_scale
            optimizer.param_groups[1]["lr"] *= cfg.stage3_head_lr_scale

        t0 = time.time()
        metrics = train_epoch_with_pauc(
            model, train_loader, criterion, pauc_fn, pauc_coeff,
            optimizer, cfg.device, scaler,
            use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip,
            num_seg_classes=cfg.num_seg_classes,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] {stage_name} "
            f"(ratio={benign_ratio:.2f}, w={benign_weight:.2f}, pAUC_coeff={pauc_coeff:.1f}) "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {metrics['loss']:.4f} "
            f"(seg={metrics['seg_loss']:.4f}, cls={metrics['cls_loss']:.4f}) "
            f"| Acc: {metrics['cls_acc']:.4f} | Seg IoU: {metrics['seg_iou']:.4f} "
            f"| {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 50)

            # Dev pAUC 评估（模型选择）
            dev_logits, dev_labels = predict_logits_text(model, dev_loader, cfg.device)
            dev_probs_b = dev_logits.softmax(-1)[:, 0].numpy()
            dev_pauc = compute_pauc_at_fpr(dev_labels.numpy(), dev_probs_b, max_fpr=0.05)

            # Dev 临床指标
            dev_policy = find_constrained_threshold_text(
                dev_probs_b, dev_labels.numpy(), max_benign_miss_rate=cfg.max_benign_miss_rate
            )
            logger.info(
                f"[Dev] pAUC@0.05={dev_pauc:.4f} | "
                f"no_tumor召回={dev_policy['no_tumor_recall']:.2%} | "
                f"benign漏检={dev_policy['benign_miss_rate']:.2%}"
            )

            # Test 快速评估（仅参考，不用于选模型）
            evaluate_text(
                model, test_loader, cfg.device, cfg.class_names, logger,
                phase="Test(ref)", num_seg_classes=cfg.num_seg_classes,
            )

            if dev_pauc > best_dev_pauc:
                best_dev_pauc = dev_pauc
                best_epoch = epoch
                no_improve_evals = 0
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(
                    f"*** 保存最优模型 (Epoch={best_epoch}, dev_pAUC@0.05={best_dev_pauc:.4f}) ***"
                )
            else:
                no_improve_evals += 1

            logger.info("-" * 50)

            if (
                epoch >= cfg.min_epochs_for_early_stop
                and no_improve_evals >= cfg.policy_patience
            ):
                logger.info(
                    f"触发Early Stop: 连续{no_improve_evals}次无提升, best_epoch={best_epoch}"
                )
                break

    logger.info("\n" + "=" * 70)
    logger.info(f"训练完成! 最优模型 Epoch={best_epoch}, dev_pAUC@0.05={best_dev_pauc:.4f}")
    logger.info("=" * 70)

    # ── 加载最优权重 ───────────────────────────────────────────────
    model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )

    # ── 温标定（在 dev 上拟合）──────────────────────────────────────
    logger.info("\n[温标定] 在 dev 上拟合 T...")
    dev_logits, dev_labels = predict_logits_text(model, dev_loader, cfg.device)
    T_scaler = TemperatureScaler()
    T_val = T_scaler.fit(dev_logits, dev_labels.long())
    logger.info(f"[温标定] T = {T_val:.4f}")

    # ── 保存 dev probs ────────────────────────────────────────────
    dev_probs_b_final = dev_logits.softmax(-1)[:, 0].numpy()
    dev_probs_nt_final = dev_logits.softmax(-1)[:, 1].numpy()
    dev_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_dev_probs.csv")
    save_probs_csv(dev_csv, dev_labels.numpy(), dev_probs_b_final, dev_probs_nt_final)
    logger.info(f"Dev probs 已保存: {dev_csv}")

    # ── Test: uncalibrated probs ─────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("最终 Test 评估 (threshold=0.5, uncalibrated)")
    logger.info("=" * 70)
    evaluate_text(model, test_loader, cfg.device, cfg.class_names, logger,
                  phase="Final Test (uncal)", num_seg_classes=cfg.num_seg_classes)

    test_probs_b, test_probs_nt, test_labels = predict_probs_text(model, test_loader, cfg.device)
    uncal_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_test_probs_uncal.csv")
    save_probs_csv(uncal_csv, test_labels, test_probs_b, test_probs_nt)
    logger.info(f"Test uncal probs 已保存: {uncal_csv}")

    # dev 上阈值搜索（uncalibrated）
    dev_thr_result, thr_uncal_path = compute_and_save_thresholds(
        dev_probs_b_final, dev_labels.numpy(),
        cfg.log_dir, cfg.exp_name + "_uncal", T_val=1.0,
    )
    log_threshold_results(logger, dev_thr_result, phase="Dev (uncalibrated)")

    # ── Test: calibrated probs ───────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"最终 Test 评估 (calibrated, T={T_val:.4f})")
    logger.info("=" * 70)
    test_logits, _ = predict_logits_text(model, test_loader, cfg.device)
    cal_logits = test_logits / max(T_val, 0.1)
    cal_probs_b = cal_logits.softmax(-1)[:, 0].numpy()
    cal_probs_nt = cal_logits.softmax(-1)[:, 1].numpy()

    cal_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_test_probs_calibrated.csv")
    save_probs_csv(cal_csv, test_labels, cal_probs_b, cal_probs_nt)
    logger.info(f"Test cal probs 已保存: {cal_csv}")

    # dev 上阈值搜索（calibrated）
    dev_logits_cal = dev_logits / max(T_val, 0.1)
    dev_probs_b_cal = dev_logits_cal.softmax(-1)[:, 0].numpy()
    dev_thr_cal, thr_cal_path = compute_and_save_thresholds(
        dev_probs_b_cal, dev_labels.numpy(),
        cfg.log_dir, cfg.exp_name, T_val=T_val,
    )
    log_threshold_results(logger, dev_thr_cal, phase="Dev (calibrated) → 最终推荐阈值")
    logger.info(f"Thresholds JSON 已保存: {thr_cal_path}")

    # ── Reliability 分析 ──────────────────────────────────────────
    for suffix, probs_b_plot, label_str in [
        ("uncal", test_probs_b, "未校准"),
        ("cal", cal_probs_b, f"校准(T={T_val:.2f})"),
    ]:
        y_true_nt = (test_labels == 1).astype(int)
        probs_nt_plot = 1.0 - probs_b_plot
        rel_rows, ece, brier = compute_binary_reliability_stats(
            y_true_nt, probs_nt_plot, n_bins=cfg.calibration_bins
        )
        logger.info(f"[Reliability {label_str}] ECE={ece:.4f} | Brier={brier:.4f}")
        save_reliability_stats_csv(
            rel_rows, os.path.join(cfg.log_dir, f"{cfg.exp_name}_reliability_{suffix}.csv")
        )
        save_reliability_diagram(
            rel_rows,
            os.path.join(cfg.log_dir, f"{cfg.exp_name}_reliability_{suffix}.png"),
            title=f"{cfg.exp_name} [{label_str}]",
        )

    # ── 复制脚本 ─────────────────────────────────────────────────
    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)
        logger.info(f"训练脚本已复制到: {dst}")

    logger.info("\n[Done] Exp#19-1 完成.")


if __name__ == "__main__":
    main()
