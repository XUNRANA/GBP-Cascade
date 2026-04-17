"""
Exp #19-3: 19-2 + Between-class Mixup + TTA + MC Dropout

改进点（在 19-2 基础上）：
  1. 模型改为 SwinV2SegGuidedCls4chTrimodalMixup（暴露 forward_with_fused）
  2. Stage 2 [41-80]: SAM + fused-space between-class Mixup + pAUC=0.5
  3. Stage 1/3: SAM（无 Mixup）
  4. 推理阶段: TTA (5组) + MC Dropout (T=30) 各自产出 probs CSV
  5. 其余同 19-2（SWA, dev split, 温标定）

GPU: 2  预计墙钟 ~3.5h + TTA/MC ~20min
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
    GBPDatasetSegCls4chWithTextMeta, SegCls4chSyncTransform,
    seg_cls_text_collate_fn, SegClsLoss, build_optimizer_with_diff_lr,
    load_text_bert_dict, EXT_CLINICAL_FEATURE_NAMES,
    set_seed, setup_logger, acquire_run_lock, set_epoch_lrs,
    evaluate_text, find_constrained_threshold_text,
    compute_binary_reliability_stats, save_reliability_stats_csv, save_reliability_diagram,
    _unpack_text_batch, compute_seg_metrics,
)
from seg_cls_utils_v5_ext import (
    stratified_dev_split, save_dev_split_json, load_dev_split_json, SubsetDataset,
    PAUCSurrogateLoss, SAM, TemperatureScaler, predict_logits_text,
    SwinV2SegGuidedCls4chTrimodalMixup,
    train_one_epoch_text_sam, train_one_epoch_text_sam_mixup,
    tta_predict_probs, mc_dropout_predict,
    compute_pauc_at_fpr, save_probs_csv, compute_and_save_thresholds, log_threshold_results,
    build_swa_model, update_swa_bn,
)


class Config:
    project_root = _ROOT
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260417_task2_SwinV2Tiny_segcls_19_3"
    log_dir = os.path.join(project_root, "0417", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")
    swa_weight_path = os.path.join(log_dir, f"{exp_name}_swa.pth")

    dev_split_json = os.path.join(project_root, "0417", "logs", "shared", "dev_split.json")
    dev_frac = 0.10

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
    use_amp = True  # autocast only (no scaler)

    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    # 三阶段 curriculum
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

    # pAUC
    pauc_K = 4
    pauc_margin = 0.1
    pauc_coeff_stage1 = 0.0
    pauc_coeff_stage2 = 0.5
    pauc_coeff_stage3 = 1.0

    # SAM
    sam_rho = 0.05

    # SWA
    swa_start_epoch = 81

    # Mixup (Stage 2 only)
    mixup_alpha = 2.0
    mixup_beta = 0.5

    # TTA
    tta_ops = ("orig", "hflip", "rot5", "rot_neg5", "scale95")

    # MC Dropout
    mc_dropout_T = 30

    max_benign_miss_rate = 0.05
    calibration_bins = 10
    policy_patience = 8
    min_epochs_for_early_stop = 30

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch + BERT + 10D Meta + Trimodal + Mixup"
    modification = (
        "Exp#19-3: 19-2 + fused-space between-class Mixup(stage2) "
        "+ TTA(5-aug) + MC Dropout(T=30)"
    )


def get_stage_params(epoch, cfg):
    if epoch < cfg.stage2_start_epoch:
        return "stage1", cfg.stage1_benign_target_ratio, cfg.stage1_benign_loss_weight, cfg.pauc_coeff_stage1
    if epoch < cfg.stage3_start_epoch:
        return "stage2", cfg.stage2_benign_target_ratio, cfg.stage2_benign_loss_weight, cfg.pauc_coeff_stage2
    return "stage3", cfg.stage3_benign_target_ratio, cfg.stage3_benign_loss_weight, cfg.pauc_coeff_stage3


def build_train_loader_with_ratio(train_sub, cfg, benign_target_ratio):
    labels = train_sub.df["label"].values.astype(int)
    cc = np.bincount(labels, minlength=2)
    p_b = float(benign_target_ratio); p_nt = 1.0 - p_b
    w = np.array([p_b / max(cc[0], 1), p_nt / max(cc[1], 1)], dtype=np.float64)
    sampler = WeightedRandomSampler(weights=w[labels], num_samples=len(train_sub), replacement=True)
    return DataLoader(
        train_sub, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=seg_cls_text_collate_fn,
    )


def build_datasets(cfg):
    text_dict = load_text_bert_dict(cfg.json_feature_root)
    full_train = GBPDatasetSegCls4chWithTextMeta(
        cfg.train_excel, cfg.data_root, clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=SegCls4chSyncTransform(cfg.img_size, is_train=True),
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict, max_text_len=cfg.max_text_len,
    )
    test_ds = GBPDatasetSegCls4chWithTextMeta(
        cfg.test_excel, cfg.data_root, clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=SegCls4chSyncTransform(cfg.img_size, is_train=False),
        meta_stats=full_train.meta_stats, meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict, tokenizer=full_train.tokenizer, max_text_len=cfg.max_text_len,
    )
    cfg.meta_dim = full_train.meta_dim
    return full_train, test_ds


def build_model(cfg):
    return SwinV2SegGuidedCls4chTrimodalMixup(
        num_seg_classes=cfg.num_seg_classes, num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim, meta_hidden=cfg.meta_hidden, meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout, text_proj_dim=cfg.text_proj_dim, text_dropout=cfg.text_dropout,
        ca_hidden=cfg.ca_hidden, ca_heads=cfg.ca_heads, ca_dropout=cfg.ca_dropout,
        fusion_dim=cfg.fusion_dim, bert_name="bert-base-chinese", pretrained=True,
    )


def build_criterion(cfg, benign_loss_weight):
    cls_w = torch.tensor([float(benign_loss_weight), 1.0], dtype=torch.float32, device=cfg.device)
    seg_w = torch.tensor([cfg.seg_bg_weight, cfg.seg_lesion_weight], dtype=torch.float32, device=cfg.device)
    return SegClsLoss(cls_weights=cls_w, lambda_cls=cfg.lambda_cls,
                      label_smoothing=cfg.label_smoothing, seg_ce_weight=seg_w)


def post_training_analysis(model_or_swa, model_tag, cfg, dev_loader, test_loader, logger):
    dev_logits, dev_labels = predict_logits_text(model_or_swa, dev_loader, cfg.device)
    T_scaler = TemperatureScaler()
    T_val = T_scaler.fit(dev_logits, dev_labels.long())
    logger.info(f"[{model_tag}] T = {T_val:.4f}")

    dev_pb = dev_logits.softmax(-1)[:, 0].numpy()
    save_probs_csv(os.path.join(cfg.log_dir, f"{cfg.exp_name}_{model_tag}_dev_probs.csv"),
                   dev_labels.numpy(), dev_pb, 1.0 - dev_pb)

    test_logits, test_labels_t = predict_logits_text(model_or_swa, test_loader, cfg.device)
    test_pb = test_logits.softmax(-1)[:, 0].numpy()
    test_labels_np = test_labels_t.numpy()
    save_probs_csv(os.path.join(cfg.log_dir, f"{cfg.exp_name}_{model_tag}_test_probs_uncal.csv"),
                   test_labels_np, test_pb, 1.0 - test_pb)

    cal_pb = (test_logits / max(T_val, 0.1)).softmax(-1)[:, 0].numpy()
    save_probs_csv(os.path.join(cfg.log_dir, f"{cfg.exp_name}_{model_tag}_test_probs_calibrated.csv"),
                   test_labels_np, cal_pb, 1.0 - cal_pb)

    dev_pb_cal = (dev_logits / max(T_val, 0.1)).softmax(-1)[:, 0].numpy()
    thr_result, thr_path = compute_and_save_thresholds(
        dev_pb_cal, dev_labels.numpy(), cfg.log_dir, f"{cfg.exp_name}_{model_tag}", T_val=T_val,
    )
    log_threshold_results(logger, thr_result, phase=f"{model_tag} Dev (cal)")

    for suffix, pb in [("uncal", test_pb), ("cal", cal_pb)]:
        y_nt = (test_labels_np == 1).astype(int)
        rows, ece, brier = compute_binary_reliability_stats(y_nt, 1.0 - pb, n_bins=cfg.calibration_bins)
        logger.info(f"[{model_tag} {suffix}] ECE={ece:.4f} | Brier={brier:.4f}")
        save_reliability_stats_csv(rows, os.path.join(cfg.log_dir, f"{cfg.exp_name}_{model_tag}_reliability_{suffix}.csv"))
        save_reliability_diagram(rows, os.path.join(cfg.log_dir, f"{cfg.exp_name}_{model_tag}_reliability_{suffix}.png"),
                                  title=f"{cfg.exp_name} [{model_tag}] [{suffix}]")

    return compute_pauc_at_fpr(dev_labels.numpy(), dev_pb, max_fpr=0.05), T_val


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
    logger.info(f"实验: {cfg.exp_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"Mixup alpha={cfg.mixup_alpha} beta={cfg.mixup_beta} (Stage2 only)")
    logger.info(f"TTA ops: {cfg.tta_ops}")
    logger.info(f"MC Dropout T={cfg.mc_dropout_T}")
    logger.info("=" * 70)

    # ── 数据 ──────────────────────────────────────────────────────
    full_train, test_ds = build_datasets(cfg)

    if os.path.exists(cfg.dev_split_json):
        train_idx, dev_idx = load_dev_split_json(cfg.dev_split_json)
    else:
        train_idx, dev_idx = stratified_dev_split(full_train.df, dev_frac=cfg.dev_frac, seed=cfg.seed)
        save_dev_split_json(train_idx, dev_idx, cfg.dev_split_json)

    train_sub = SubsetDataset(full_train, train_idx)
    dev_sub = SubsetDataset(full_train, dev_idx)
    logger.info(f"Train: {len(train_sub)} | Dev: {len(dev_sub)} | Test: {len(test_ds)}")

    dev_loader = DataLoader(dev_sub, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True,
                            collate_fn=seg_cls_text_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True,
                             collate_fn=seg_cls_text_collate_fn)

    # ── 模型 & SAM ─────────────────────────────────────────────────
    model = build_model(cfg).to(cfg.device)
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.") and not name.startswith("text_encoder.")
    ]
    base_optim = build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)
    sam_optimizer = SAM(base_optim.param_groups, AdamW, rho=cfg.sam_rho,
                        lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    pauc_fn = PAUCSurrogateLoss(K=cfg.pauc_K, margin=cfg.pauc_margin)

    logger.info(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

    best_dev_pauc = -1.0
    best_epoch = 0
    no_improve_evals = 0
    swa_model = None

    logger.info("\n开始训练 (SAM + Mixup[stage2] + SWA)")
    logger.info("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        stage_name, benign_ratio, benign_weight, pauc_coeff = get_stage_params(epoch, cfg)
        train_loader = build_train_loader_with_ratio(train_sub, cfg, benign_ratio)
        criterion = build_criterion(cfg, benign_weight)

        set_epoch_lrs(sam_optimizer, epoch, cfg)
        if stage_name == "stage3":
            sam_optimizer.param_groups[0]["lr"] *= cfg.stage3_backbone_lr_scale
            sam_optimizer.param_groups[1]["lr"] *= cfg.stage3_head_lr_scale

        t0 = time.time()
        use_mixup = (stage_name == "stage2")
        if use_mixup:
            metrics = train_one_epoch_text_sam_mixup(
                model, train_loader, criterion, pauc_fn, pauc_coeff,
                sam_optimizer, cfg.device,
                lambda_cls=cfg.lambda_cls,
                use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
                grad_clip=cfg.grad_clip,
                mixup_alpha=cfg.mixup_alpha,
                mixup_beta=cfg.mixup_beta,
                num_seg_classes=cfg.num_seg_classes,
            )
        else:
            metrics = train_one_epoch_text_sam(
                model, train_loader, criterion, pauc_fn, pauc_coeff,
                sam_optimizer, cfg.device,
                use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
                grad_clip=cfg.grad_clip,
                num_seg_classes=cfg.num_seg_classes,
            )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] {stage_name}{'[Mixup]' if use_mixup else ''} "
            f"(ratio={benign_ratio:.2f}, w={benign_weight:.2f}, pAUC={pauc_coeff:.1f}) "
            f"LR: {sam_optimizer.param_groups[0]['lr']:.2e}/{sam_optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {metrics['loss']:.4f} | Acc: {metrics['cls_acc']:.4f} "
            f"| IoU: {metrics['seg_iou']:.4f} | {elapsed:.1f}s"
        )

        if epoch >= cfg.swa_start_epoch:
            if swa_model is None:
                swa_model = build_swa_model(model)
                logger.info(f"[SWA] 开始 (epoch {epoch})")
            swa_model.update_parameters(model)

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 50)
            dev_logits, dev_labels = predict_logits_text(model, dev_loader, cfg.device)
            dev_pb = dev_logits.softmax(-1)[:, 0].numpy()
            dev_pauc = compute_pauc_at_fpr(dev_labels.numpy(), dev_pb, max_fpr=0.05)
            dev_policy = find_constrained_threshold_text(dev_pb, dev_labels.numpy(),
                                                         max_benign_miss_rate=cfg.max_benign_miss_rate)
            logger.info(
                f"[Dev] pAUC@0.05={dev_pauc:.4f} | "
                f"no_tumor召回={dev_policy['no_tumor_recall']:.2%} | "
                f"benign漏检={dev_policy['benign_miss_rate']:.2%}"
            )
            evaluate_text(model, test_loader, cfg.device, cfg.class_names, logger,
                          phase="Test(ref)", num_seg_classes=cfg.num_seg_classes)

            if dev_pauc > best_dev_pauc:
                best_dev_pauc = dev_pauc
                best_epoch = epoch
                no_improve_evals = 0
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 最优 (Epoch={best_epoch}, dev_pAUC={best_dev_pauc:.4f}) ***")
            else:
                no_improve_evals += 1
            logger.info("-" * 50)

            if epoch >= cfg.min_epochs_for_early_stop and no_improve_evals >= cfg.policy_patience:
                logger.info(f"Early Stop (best={best_epoch})")
                break

    logger.info(f"\n训练完成 best_epoch={best_epoch}")

    # SWA BN 更新
    if swa_model is not None:
        logger.info("[SWA] 更新 BN...")
        train_loader_noaug = DataLoader(train_sub, batch_size=cfg.batch_size, shuffle=True,
                                        num_workers=cfg.num_workers, pin_memory=True,
                                        collate_fn=seg_cls_text_collate_fn)
        update_swa_bn(swa_model, train_loader_noaug, cfg.device)
        torch.save(swa_model.state_dict(), cfg.swa_weight_path)

    # ── Best checkpoint 分析 ───────────────────────────────────────
    model.load_state_dict(
        torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    )
    logger.info("\n" + "=" * 70)
    logger.info("Best Checkpoint 分析")
    best_dev_pauc_f, T_best = post_training_analysis(
        model, "best", cfg, dev_loader, test_loader, logger
    )

    # ── SWA 分析 ──────────────────────────────────────────────────
    winner_model = model
    winner_tag = "best"
    if swa_model is not None:
        logger.info("\nSWA 分析")
        swa_dev_pauc, T_swa = post_training_analysis(
            swa_model, "swa", cfg, dev_loader, test_loader, logger
        )
        if swa_dev_pauc > best_dev_pauc_f:
            winner_model = swa_model
            winner_tag = "swa"
        logger.info(f"best pAUC={best_dev_pauc_f:.4f} | SWA pAUC={swa_dev_pauc:.4f} → 使用 {winner_tag}")

    # 复制 winner 的标准输出文件
    for suffix in ["test_probs_calibrated", "test_probs_uncal"]:
        src = os.path.join(cfg.log_dir, f"{cfg.exp_name}_{winner_tag}_{suffix}.csv")
        dst = os.path.join(cfg.log_dir, f"{cfg.exp_name}_{suffix}.csv")
        if os.path.exists(src):
            shutil.copy2(src, dst)

    # ── TTA 推理 ──────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"TTA 推理 (ops={cfg.tta_ops})")
    logger.info("=" * 70)
    tta_pb, tta_pnt, tta_labels = tta_predict_probs(
        winner_model, test_loader, cfg.device, tta_ops=cfg.tta_ops
    )
    tta_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_tta_probs.csv")
    save_probs_csv(tta_csv, tta_labels, tta_pb, tta_pnt)
    logger.info(f"TTA probs 已保存: {tta_csv}")

    tta_policy = find_constrained_threshold_text(tta_pb, tta_labels, max_benign_miss_rate=0.05)
    logger.info(
        f"[TTA] 良≥95% no_tumor={tta_policy['selected_no_tumor']}/394 "
        f"({tta_policy['no_tumor_recall']:.2%}) | "
        f"benign召回={tta_policy['benign_recall']:.2%}"
    )

    # ── MC Dropout 推理 ───────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info(f"MC Dropout 推理 (T={cfg.mc_dropout_T})")
    logger.info("=" * 70)
    mc_pb_mean, mc_pb_std, mc_labels = mc_dropout_predict(
        winner_model, test_loader, cfg.device, T=cfg.mc_dropout_T
    )
    mc_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_mc_dropout.csv")
    pd.DataFrame({
        "label": mc_labels.astype(int),
        "prob_benign_mean": mc_pb_mean.astype(np.float32),
        "prob_benign_std": mc_pb_std.astype(np.float32),
        "prob_no_tumor_mean": (1.0 - mc_pb_mean).astype(np.float32),
    }).to_csv(mc_csv, index=False, encoding="utf-8")
    logger.info(f"MC Dropout probs 已保存: {mc_csv}")

    mc_policy = find_constrained_threshold_text(mc_pb_mean, mc_labels, max_benign_miss_rate=0.05)
    logger.info(
        f"[MC mean] 良≥95% no_tumor={mc_policy['selected_no_tumor']}/394 | "
        f"benign召回={mc_policy['benign_recall']:.2%}"
    )
    logger.info(
        f"[MC std] mean±std of P(benign): "
        f"{mc_pb_mean.mean():.3f} ± {mc_pb_std.mean():.3f}"
    )

    # 复制脚本
    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)

    logger.info("\n[Done] Exp#19-3 完成.")


if __name__ == "__main__":
    main()
