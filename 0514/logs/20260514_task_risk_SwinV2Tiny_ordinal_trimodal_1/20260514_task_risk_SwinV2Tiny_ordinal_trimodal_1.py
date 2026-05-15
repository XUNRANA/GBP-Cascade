"""
0514 Risk #1: 胆囊息肉风险分层模型 (高/中/低风险)

相比 0502 版本的主要变化
  - 数据集: 0514dataset_flat（完整原始档案库，1791 患者 / 3902 图像）
            比 0502dataset111 多 57 患者（+50 benign / +7 malignant）
            benign 从 438 图增至 682 图，直接攻克 benign 识别率瓶颈
  - 文本:   json_text_0514/（扩展了 104 名 0514 独有患者的超声报告）
  - 测试集: 仅 0514dataset_flat 内的 30% 测试集，无二级稳定性集
  - 其余:   模型架构、损失、超参数与 0502-exp#1 完全一致，便于对比

运行前请先执行:
  python prepare_0514_dataset.py
"""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

# ─── 路径设置 ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
for _p in [
    os.path.join(ROOT_DIR, "0408", "scripts"),
    os.path.join(ROOT_DIR, "0402", "scripts"),
    os.path.join(ROOT_DIR, "0323", "scripts"),
    os.path.join(ROOT_DIR, "0502", "scripts"),
    SCRIPT_DIR,
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from risk_utils import (  # noqa: E402
    SwinV2SegGuidedRiskTrimodal,
    RiskOrdinalLoss,
    BalancedBatchSampler,
    ORDINAL_TARGETS_DEFAULT,
    search_risk_thresholds_constrained,
    evaluate_risk_bands,
    plot_risk_confusion_matrices,
    extract_patient_id,
    dump_json,
    PRIMARY_PROFILE,
    CLASS_NAMES,
)
from seg_cls_utils_v5 import (  # noqa: E402
    GBPDatasetSegCls4chWithTextMeta,
    SegCls4chSyncTransform,
    seg_cls_text_collate_fn,
    EXT_CLINICAL_FEATURE_NAMES,
    load_text_bert_dict,
)
from seg_cls_utils_v2 import (  # noqa: E402
    set_seed,
    setup_logger,
    acquire_run_lock,
    set_epoch_lrs,
    build_optimizer_with_diff_lr,
    compute_seg_metrics,
)


# ═════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════


class Config:
    project_root = ROOT_DIR

    # 0514dataset_flat: 由 prepare_0514_dataset.py 生成
    data_root       = os.path.join(project_root, "0514dataset_flat")
    train_excel     = os.path.join(data_root, "task_3class_train.xlsx")
    test_excel      = os.path.join(data_root, "task_3class_test.xlsx")

    # 临床特征表（与 0502 共用同一份）
    clinical_excel  = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    # 扩展的超声报告 json（含 0514 独有患者）
    json_feature_root = os.path.join(project_root, "json_text_0514")

    exp_name = "20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1"
    log_dir  = os.path.join(project_root, "0514", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 模型（与 0502 一致）
    img_size        = 256
    in_channels     = 4
    num_seg_classes = 2
    cls_dropout     = 0.4
    meta_dim        = len(EXT_CLINICAL_FEATURE_NAMES)  # 10
    meta_hidden     = 96
    meta_dropout    = 0.2
    text_proj_dim   = 128
    text_dropout    = 0.3
    max_text_len    = 128
    fusion_dim      = 256
    ca_hidden       = 128
    ca_heads        = 4
    ca_dropout      = 0.1

    # 训练（与 0502 一致）
    batch_size      = 8
    num_epochs      = 60
    warmup_epochs   = 5
    backbone_lr     = 2e-5
    head_lr         = 2e-4
    weight_decay    = 5e-2
    min_lr_ratio    = 0.01
    grad_clip       = 1.0
    num_workers     = 4
    eval_interval   = 2
    seed            = 42
    use_amp         = True

    # 损失
    lambda_ord       = 2.0
    seg_bg_weight    = 1.0
    seg_lesion_weight = 5.0

    # 采样 3:3:2 (mal:ben:notumor) per batch
    samples_per_class = {0: 3, 1: 3, 2: 2}

    # Val split: 从 train 里 patient-level 切 15%
    val_ratio = 0.15

    ordinal_targets = ORDINAL_TARGETS_DEFAULT  # (1.0, 0.5, 0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = CLASS_NAMES
    model_name = (
        "SwinV2-Tiny@256 + 4ch + Seg-Guided + BERT CrossAttn + BERT[CLS] + "
        "10D ExtMeta + Gated Trimodal Fusion + 1D Risk Head (sigmoid) "
        "[0514dataset]"
    )


# ═════════════════════════════════════════════════════════════
#  Data
# ═════════════════════════════════════════════════════════════


def patient_level_train_val_split(train_excel: str, val_ratio: float, seed: int):
    df = pd.read_excel(train_excel)
    df["patient_id"] = df["image_path"].apply(extract_patient_id)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(df, df["label"], groups=df["patient_id"]))
    return (
        df.iloc[train_idx].copy().reset_index(drop=True),
        df.iloc[val_idx].copy().reset_index(drop=True),
    )


def build_dataloaders(cfg: Config, logger):
    logger.info("[1/4] 切分 train/val (patient-level 15%) ...")
    t0 = time.time()
    train_df, val_df = patient_level_train_val_split(
        cfg.train_excel, cfg.val_ratio, cfg.seed
    )
    train_excel_split = os.path.join(cfg.log_dir, "train_split.xlsx")
    val_excel_split   = os.path.join(cfg.log_dir, "val_split.xlsx")
    train_df.to_excel(train_excel_split, index=False)
    val_df.to_excel(val_excel_split, index=False)
    logger.info(f"[1/4] 完成: train={len(train_df)} val={len(val_df)} ({time.time()-t0:.1f}s)")

    train_tf = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    eval_tf  = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    logger.info("[2/4] 装载 text_dict ...")
    t0 = time.time()
    text_dict = load_text_bert_dict(cfg.json_feature_root)
    logger.info(f"[2/4] 完成: {len(text_dict)} 条 ({time.time()-t0:.1f}s)")

    logger.info("[3/4] 构造 train / val / test Dataset ...")
    t0 = time.time()

    common_kwargs = dict(
        data_root=cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        max_text_len=cfg.max_text_len,
    )

    train_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=train_excel_split,
        sync_transform=train_tf,
        **common_kwargs,
    )
    meta_stats = train_dataset.meta_stats
    tokenizer  = train_dataset.tokenizer
    logger.info(f"  train OK ({time.time()-t0:.1f}s)")

    val_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=val_excel_split,
        sync_transform=eval_tf,
        meta_stats=meta_stats,
        tokenizer=tokenizer,
        **common_kwargs,
    )
    test_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=cfg.test_excel,
        sync_transform=eval_tf,
        meta_stats=meta_stats,
        tokenizer=tokenizer,
        **common_kwargs,
    )

    for name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        counts = ds.df["label"].value_counts().sort_index().to_dict()
        msg = ", ".join(f"{cfg.class_names[i]}={counts.get(i, 0)}"
                        for i in range(len(cfg.class_names)))
        logger.info(f"  {name}: {len(ds)} 张 ({msg})")

    logger.info("[4/4] 构建 BalancedBatchSampler + DataLoader ...")
    sampler = BalancedBatchSampler(
        labels=train_dataset.df["label"].values,
        samples_per_class=cfg.samples_per_class,
        shuffle=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_text_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_text_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_text_collate_fn,
    )
    logger.info(f"DataLoaders OK | train={len(train_loader)} "
                f"val={len(val_loader)} test={len(test_loader)} batches")

    return train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════
#  Model / Optimizer
# ═════════════════════════════════════════════════════════════


def build_model(cfg: Config, logger):
    logger.info("构建模型 (BERT + SwinV2-Tiny 三模态融合, 离线加载) ...")
    t0 = time.time()
    model_kwargs = dict(
        num_seg_classes=cfg.num_seg_classes,
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
    )
    try:
        model = SwinV2SegGuidedRiskTrimodal(**model_kwargs, pretrained=True).to(cfg.device)
    except Exception as exc:
        logger.warning(f"pretrained 加载失败, 使用随机初始化: {exc}")
        model = SwinV2SegGuidedRiskTrimodal(**model_kwargs, pretrained=False).to(cfg.device)
    logger.info(f"模型构建完成 ({time.time()-t0:.1f}s)")
    return model


def build_optimizer(model, cfg: Config):
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad
        and not name.startswith("encoder.")
        and not name.startswith("text_encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


# ═════════════════════════════════════════════════════════════
#  Train / Eval loops
# ═════════════════════════════════════════════════════════════


def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    use_amp, grad_clip):
    model.train()
    stats = {"loss": 0.0, "seg": 0.0, "ord": 0.0, "total": 0, "seg_dices": []}

    for batch in loader:
        imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch
        imgs      = imgs.to(device, non_blocking=True)
        masks     = masks.to(device, non_blocking=True)
        metas     = metas.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=use_amp,
        ):
            seg_logits, risk_logit = model(
                imgs, metadata=metas,
                input_ids=input_ids, attention_mask=attn_mask,
            )
            loss, seg_l, ord_l = criterion(
                seg_logits, risk_logit, masks, labels, has_masks,
            )

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        stats["loss"] += loss.item() * bs
        stats["seg"]  += seg_l * bs
        stats["ord"]  += ord_l * bs
        stats["total"] += bs

        if has_masks.any():
            with torch.no_grad():
                idx = has_masks.nonzero(as_tuple=True)[0]
                m = compute_seg_metrics(seg_logits[idx], masks[idx], num_classes=2)
                stats["seg_dices"].append(m["lesion_Dice"])

    n = max(stats["total"], 1)
    return {
        "loss":     stats["loss"] / n,
        "seg_loss": stats["seg"] / n,
        "ord_loss": stats["ord"] / n,
        "seg_dice": np.mean(stats["seg_dices"]) if stats["seg_dices"] else 0.0,
    }


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_scores, all_labels, seg_dices = [], [], []

    for batch in loader:
        imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch
        imgs      = imgs.to(device, non_blocking=True)
        masks     = masks.to(device, non_blocking=True)
        metas     = metas.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        seg_logits, risk_logit = model(
            imgs, metadata=metas,
            input_ids=input_ids, attention_mask=attn_mask,
        )
        risk_score = torch.sigmoid(risk_logit)
        all_scores.append(risk_score.detach().float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        if has_masks.any():
            idx = has_masks.nonzero(as_tuple=True)[0]
            m = compute_seg_metrics(seg_logits[idx], masks[idx], num_classes=2)
            seg_dices.append(m["lesion_Dice"])

    return {
        "scores":   np.concatenate(all_scores).astype(np.float64),
        "labels":   np.concatenate(all_labels).astype(np.int64),
        "seg_dice": float(np.mean(seg_dices)) if seg_dices else 0.0,
    }


def _val_roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    is_mal = (labels == 0).astype(np.int64)
    if is_mal.sum() == 0 or is_mal.sum() == len(is_mal):
        return 0.0
    try:
        return float(roc_auc_score(is_mal, scores))
    except ValueError:
        return 0.0


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════


def main():
    cfg = Config()
    os.makedirs(cfg.log_dir, exist_ok=True)

    # 检查数据集已准备好
    if not os.path.exists(cfg.train_excel):
        raise FileNotFoundError(
            f"找不到 {cfg.train_excel}\n"
            "请先运行: python prepare_0514_dataset.py"
        )

    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_file, cfg.exp_name)

    logger.info("=" * 72)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"数据集: {cfg.data_root}")
    logger.info(f"图像尺寸: {cfg.img_size}, 4ch (RGB+gallbladder mask)")
    logger.info(f"风险目标: {cfg.ordinal_targets.tolist()} "
                f"(malignant=1.0, benign=0.5, no_tumor=0.0)")
    logger.info(f"约束 profile: max_mal_to_low={PRIMARY_PROFILE.max_mal_to_low}, "
                f"min_high_recall={PRIMARY_PROFILE.min_high_recall}, "
                f"max_medium_share={PRIMARY_PROFILE.max_medium_share}")
    logger.info(f"Batch sampler: {cfg.samples_per_class} = batch_size {cfg.batch_size}")
    logger.info(f"Backbone LR: {cfg.backbone_lr} | Head LR: {cfg.head_lr}")
    logger.info(f"Lambda Ord: {cfg.lambda_ord} | Seg lesion weight: {cfg.seg_lesion_weight}")
    logger.info(f"Epochs: {cfg.num_epochs} | Warmup: {cfg.warmup_epochs} "
                f"| Eval interval: {cfg.eval_interval}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 72)

    # ── 数据 ──
    train_loader, val_loader, test_loader = build_dataloaders(cfg, logger)

    # ── 模型 ──
    model = build_model(cfg, logger)
    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,} (可训练 {n_trainable:,})")

    # ── 损失 ──
    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight],
        dtype=torch.float32, device=cfg.device,
    )
    criterion = RiskOrdinalLoss(
        ordinal_targets=cfg.ordinal_targets.to(cfg.device),
        lambda_ord=cfg.lambda_ord,
        seg_ce_weight=seg_ce_weight,
    ).to(cfg.device)

    # ── 优化器 + AMP ──
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    # ── 训练 ──
    best_val_auc = 0.0
    best_epoch   = 0

    logger.info("=" * 72)
    logger.info("开始训练")
    logger.info("=" * 72)

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler,
            use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f} ord={train_metrics['ord_loss']:.4f}) "
            f"| Dice: {train_metrics['seg_dice']:.4f} | {elapsed:.0f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            val_pred = collect_predictions(model, val_loader, cfg.device)
            val_auc  = _val_roc_auc(val_pred["scores"], val_pred["labels"])
            val_search = search_risk_thresholds_constrained(
                val_pred["scores"], val_pred["labels"]
            )
            logger.info(
                f"[Val] ROC-AUC(mal vs others): {val_auc:.4f} | "
                f"profile={val_search['profile_used']} "
                f"t_low={val_search['t_low']:.3f} t_high={val_search['t_high']:.3f} "
                f"low_p={val_search['metrics']['low_precision']:.4f} "
                f"high_recall={val_search['metrics']['high_recall']:.4f} "
                f"M→低={val_search['metrics']['mal_to_low']} "
                f"med_share={val_search['metrics']['medium_share']:.3f} | "
                f"Seg Dice: {val_pred['seg_dice']:.4f}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch   = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(
                    f"*** 保存最优模型 (Val AUC: {best_val_auc:.4f}, Epoch: {best_epoch}) ***"
                )

    logger.info("=" * 72)
    logger.info(f"训练完成。最优 Epoch: {best_epoch}, Best Val AUC: {best_val_auc:.4f}")
    logger.info("=" * 72)

    # ── 加载最优权重 ──
    if not os.path.exists(cfg.best_weight_path):
        logger.error("best weight 不存在，跳过最终评估。")
        return

    try:
        state = torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    except TypeError:
        state = torch.load(cfg.best_weight_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()

    # ── 阈值搜索：在 val 上确定 (t_low, t_high) ──
    val_pred   = collect_predictions(model, val_loader, cfg.device)
    val_search = search_risk_thresholds_constrained(
        val_pred["scores"], val_pred["labels"]
    )
    t_low, t_high = val_search["t_low"], val_search["t_high"]

    logger.info("=" * 72)
    logger.info(f"在 val 上约束阈值搜索: profile_used={val_search['profile_used']}")
    logger.info(f"  → 选定阈值: t_low={t_low:.3f}, t_high={t_high:.3f}")
    logger.info(f"  val 指标: {val_search['metrics']}")
    if val_search["profile_used"] != "primary":
        logger.warning(
            f"⚠ primary profile 未满足，回退到 {val_search['profile_used']} —— "
            "硬底线 (M→低 ≤ 1, 高风险召回 ≥ 0.95) 在 val 上未达成。"
        )
    logger.info("=" * 72)

    thresholds_path = os.path.join(cfg.log_dir, "thresholds.json")
    dump_json({
        "selection_set": "val",
        "best_epoch":    best_epoch,
        "best_val_auc":  best_val_auc,
        "t_low":  t_low,
        "t_high": t_high,
        "profile_used": val_search["profile_used"],
        "val_metrics":  val_search["metrics"],
        "all_candidates_top10": val_search["all_candidates"][:10],
    }, thresholds_path)
    logger.info(f"阈值已保存: {thresholds_path}")

    # ── 在测试集上评估 ──
    pred = collect_predictions(model, test_loader, cfg.device)

    eval_main = evaluate_risk_bands(
        pred["scores"], pred["labels"], t_low, t_high,
        logger=logger, phase="Test-0514 (val-thr)",
    )
    eval_main["seg_dice"] = pred["seg_dice"]

    # 诊断: 若直接在测试集上搜阈值
    diag_search = search_risk_thresholds_constrained(pred["scores"], pred["labels"])
    logger.info(
        f"[Test] 诊断 — 若直接在该集上搜阈值: "
        f"profile={diag_search['profile_used']} "
        f"t_low={diag_search['t_low']:.3f} t_high={diag_search['t_high']:.3f} "
        f"low_p={diag_search['metrics']['low_precision']:.4f}"
    )
    eval_main["diagnostic_search_on_self"] = {
        "profile_used": diag_search["profile_used"],
        "t_low":   diag_search["t_low"],
        "t_high":  diag_search["t_high"],
        "metrics": diag_search["metrics"],
    }

    json_path = os.path.join(cfg.log_dir, "eval_test.json")
    dump_json(eval_main, json_path)
    logger.info(f"评估结果已保存: {json_path}")

    # ── 混淆矩阵图 ──
    fig_path = os.path.join(cfg.log_dir, "confusion_matrix.png")
    plot_risk_confusion_matrices({"Test-0514": eval_main}, fig_path)
    logger.info(f"混淆矩阵图: {fig_path}")

    # ── 保存脚本副本 ──
    for src_file in [__file__, os.path.join(SCRIPT_DIR, "prepare_0514_dataset.py")]:
        dst = os.path.join(cfg.log_dir, os.path.basename(src_file))
        if os.path.exists(src_file) and os.path.abspath(src_file) != os.path.abspath(dst):
            Path(dst).write_text(Path(src_file).read_text(encoding="utf-8"), encoding="utf-8")

    risk_utils_src = os.path.join(ROOT_DIR, "0502", "scripts", "risk_utils.py")
    if os.path.exists(risk_utils_src):
        Path(os.path.join(cfg.log_dir, "risk_utils.py")).write_text(
            Path(risk_utils_src).read_text(encoding="utf-8"), encoding="utf-8"
        )

    logger.info("=" * 72)
    logger.info("全部完成。关键产出:")
    logger.info(f"  best weight    : {cfg.best_weight_path}")
    logger.info(f"  thresholds.json: {thresholds_path}")
    logger.info(f"  eval_test.json : {json_path}")
    logger.info(f"  混淆矩阵图     : {fig_path}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
