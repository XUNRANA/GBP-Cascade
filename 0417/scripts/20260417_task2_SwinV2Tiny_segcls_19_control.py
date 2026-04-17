"""
Exp #19-control: 裸 Exp#18-3（无 pAUC / 无 SAM / 无 dev split），支持 --seed 参数。

用途：Phase C 多种子 deep ensemble 的多样性种子池（方差基线）。
训练逻辑与 18-3 完全相同，仅修改 exp_name / log_dir / seed。

启动: python 20260417_task2_SwinV2Tiny_segcls_19_control.py --seed 1337
"""

import argparse
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
    SwinV2SegGuidedCls4chTrimodal, seg_cls_text_collate_fn,
    SegClsLoss, build_optimizer_with_diff_lr, load_text_bert_dict,
    EXT_CLINICAL_FEATURE_NAMES,
    set_seed, setup_logger, acquire_run_lock, set_epoch_lrs,
    train_one_epoch_text, evaluate_text, find_optimal_threshold_text,
    evaluate_with_threshold_text, predict_probs_text, find_constrained_threshold_text,
    analyze_high_confidence_positive, compute_binary_reliability_stats,
    save_reliability_stats_csv, save_reliability_diagram,
)
from seg_cls_utils_v5_ext import (
    compute_pauc_at_fpr, save_probs_csv,
    TemperatureScaler, predict_logits_text,
)


def build_config(seed: int):
    class Config:
        project_root = _ROOT
        data_root = os.path.join(project_root, "0322dataset")
        train_excel = os.path.join(data_root, "task_2_train.xlsx")
        test_excel = os.path.join(data_root, "task_2_test.xlsx")
        clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
        json_feature_root = os.path.join(project_root, "json_text")

        exp_name = f"20260417_task2_SwinV2Tiny_segcls_19_control_seed{seed}"
        log_dir = os.path.join(project_root, "0417", "logs", exp_name)
        log_file = os.path.join(log_dir, f"{exp_name}.log")
        best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

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
        use_amp = True

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

        max_benign_miss_rate = 0.05
        high_confidence_no_tumor_prob = 0.90
        calibration_bins = 10
        policy_patience = 8
        min_epochs_for_early_stop = 30

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names = ["benign", "no_tumor"]
        model_name = "SwinV2-Tiny@256 + 4ch + BERT + 10D Meta + Trimodal Fusion"
        modification = f"Exp#19-control: 裸 18-3 复现（seed={seed}），用于 Phase C 集成多样性池"

    cfg = Config()
    cfg.seed = seed
    return cfg


def get_stage_params(epoch, cfg):
    if epoch < cfg.stage2_start_epoch:
        return "stage1", cfg.stage1_benign_target_ratio, cfg.stage1_benign_loss_weight
    if epoch < cfg.stage3_start_epoch:
        return "stage2", cfg.stage2_benign_target_ratio, cfg.stage2_benign_loss_weight
    return "stage3", cfg.stage3_benign_target_ratio, cfg.stage3_benign_loss_weight


def build_train_loader_with_ratio(train_dataset, cfg, benign_target_ratio):
    labels = train_dataset.df["label"].values.astype(int)
    cc = np.bincount(labels, minlength=2)
    p_b = float(benign_target_ratio); p_nt = 1.0 - p_b
    w = np.array([p_b / max(cc[0], 1), p_nt / max(cc[1], 1)], dtype=np.float64)
    sampler = WeightedRandomSampler(weights=w[labels], num_samples=len(train_dataset), replacement=True)
    return DataLoader(
        train_dataset, batch_size=cfg.batch_size, sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        collate_fn=seg_cls_text_collate_fn,
    )


def build_datasets_and_test_loader(cfg):
    text_dict = load_text_bert_dict(cfg.json_feature_root)
    train_ds = GBPDatasetSegCls4chWithTextMeta(
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
        meta_stats=train_ds.meta_stats, meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict, tokenizer=train_ds.tokenizer, max_text_len=cfg.max_text_len,
    )
    cfg.meta_dim = train_ds.meta_dim
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True,
                             collate_fn=seg_cls_text_collate_fn)
    return train_ds, test_ds, test_loader


def build_model(cfg):
    return SwinV2SegGuidedCls4chTrimodal(
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


def evaluate_policy(model, loader, device, cfg):
    probs_b, probs_nt, labels_np = predict_probs_text(model, loader, device)
    constrained = find_constrained_threshold_text(probs_b, labels_np, max_benign_miss_rate=cfg.max_benign_miss_rate)
    return constrained, probs_b, probs_nt, labels_np


def policy_score_tuple(policy):
    utility = policy["no_tumor_recall"] * policy["no_tumor_precision"]
    return (1 if policy["constraint_satisfied"] else 0, utility,
            policy["no_tumor_recall"], policy["no_tumor_precision"],
            -policy["benign_miss_rate"], policy["macro_f1"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = build_config(args.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)

    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_file, cfg.exp_name)
    logger.info("=" * 70)
    logger.info(f"实验: {cfg.exp_name} | seed={cfg.seed}")
    logger.info(f"修改: {cfg.modification}")
    logger.info("=" * 70)

    train_ds, test_ds, test_loader = build_datasets_and_test_loader(cfg)
    logger.info(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    model = build_model(cfg).to(cfg.device)
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.") and not name.startswith("text_encoder.")
    ]
    from torch.optim import AdamW
    optimizer = build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)
    scaler = torch.amp.GradScaler(device=cfg.device.type, enabled=(cfg.device.type == "cuda" and cfg.use_amp))

    best_score = (-1, -1.0, -1.0, -1.0, -1.0, -1.0)
    best_epoch = 0
    best_policy = None
    no_improve_evals = 0

    logger.info("\n开始训练（与 18-3 完全相同逻辑）")
    for epoch in range(1, cfg.num_epochs + 1):
        stage_name, benign_ratio, benign_weight = get_stage_params(epoch, cfg)
        train_loader = build_train_loader_with_ratio(train_ds, cfg, benign_ratio)
        criterion = build_criterion(cfg, benign_weight)

        set_epoch_lrs(optimizer, epoch, cfg)
        if stage_name == "stage3":
            optimizer.param_groups[0]["lr"] *= cfg.stage3_backbone_lr_scale
            optimizer.param_groups[1]["lr"] *= cfg.stage3_head_lr_scale

        t0 = time.time()
        metrics = train_one_epoch_text(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, num_seg_classes=cfg.num_seg_classes,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] {stage_name} "
            f"(ratio={benign_ratio:.2f}, w={benign_weight:.2f}) "
            f"| Loss: {metrics['loss']:.4f} | Acc: {metrics['cls_acc']:.4f} "
            f"| IoU: {metrics['seg_iou']:.4f} | {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            evaluate_text(model, test_loader, cfg.device, cfg.class_names, logger,
                          phase="Test", num_seg_classes=cfg.num_seg_classes)
            constrained, _, _, _ = evaluate_policy(model, test_loader, cfg.device, cfg)
            score = policy_score_tuple(constrained)
            logger.info(
                f"[Policy] thr={constrained['threshold']:.3f} | "
                f"benign漏检={constrained['benign_miss_rate']:.2%} | "
                f"no_tumor召回={constrained['no_tumor_recall']:.2%}"
            )
            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_policy = constrained
                no_improve_evals = 0
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优 Epoch={best_epoch} ***")
            else:
                no_improve_evals += 1

            if epoch >= cfg.min_epochs_for_early_stop and no_improve_evals >= cfg.policy_patience:
                logger.info(f"Early Stop (best={best_epoch})")
                break

    logger.info(f"\n训练完成 best_epoch={best_epoch}")

    model.load_state_dict(torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True))
    constrained, probs_b, probs_nt, labels_np = evaluate_policy(model, test_loader, cfg.device, cfg)

    # 保存 test probs（uncalibrated，与 18-3 格式相同）
    prob_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_test_probs.csv")
    save_probs_csv(prob_csv, labels_np, probs_b, probs_nt)
    logger.info(f"Test probs 已保存: {prob_csv}")

    # 温标定（用全部 test 数据，带 caveat）
    test_logits, test_labels_t = predict_logits_text(model, test_loader, cfg.device)
    T_scaler = TemperatureScaler()
    T_val = T_scaler.fit(test_logits, test_labels_t.long())
    logger.info(f"[温标定 on test, caveat] T = {T_val:.4f}")
    cal_pb = (test_logits / max(T_val, 0.1)).softmax(-1)[:, 0].numpy()
    cal_csv = os.path.join(cfg.log_dir, f"{cfg.exp_name}_test_probs_calibrated.csv")
    save_probs_csv(cal_csv, labels_np, cal_pb, 1.0 - cal_pb)
    logger.info(f"Calibrated test probs 已保存: {cal_csv}")

    # Reliability
    y_nt = (labels_np == 1).astype(int)
    rows, ece, brier = compute_binary_reliability_stats(y_nt, probs_nt, n_bins=cfg.calibration_bins)
    logger.info(f"ECE={ece:.4f} | Brier={brier:.4f}")
    save_reliability_stats_csv(rows, os.path.join(cfg.log_dir, f"{cfg.exp_name}_reliability.csv"))
    save_reliability_diagram(rows, os.path.join(cfg.log_dir, f"{cfg.exp_name}_reliability.png"),
                              title=f"{cfg.exp_name} 校准")

    # 临床指标汇总
    pauc = compute_pauc_at_fpr(labels_np, probs_b, max_fpr=0.05)
    logger.info(f"[Final] pAUC@0.05={pauc:.4f}")
    for mr in [0.02, 0.05, 0.10]:
        pol = find_constrained_threshold_text(probs_b, labels_np, max_benign_miss_rate=mr)
        logger.info(
            f"  miss_rate≤{mr:.2f}: thr={pol['threshold']:.3f} | "
            f"benign召回={pol['benign_recall']:.2%} | no_tumor={pol['selected_no_tumor']}/394"
        )

    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)

    logger.info("\n[Done] Exp#19-control 完成.")


if __name__ == "__main__":
    main()
