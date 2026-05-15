"""
train_with_ratio.py — 通用训练入口，参数化 sampler 和 train_xlsx。

基于 20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1.py，关键修改:
  1. 命令行接收 --exp_label / --train_xlsx / --val_xlsx / --sampler / --num_epochs
  2. 不重新切 train/val（强制用 baseline val_split.xlsx）
  3. meta_stats 从 baseline train_split.xlsx 拟合（跨实验一致）
  4. sampler: 4 种模式
       1_1_1  → BalancedBatchSampler({0:2, 1:2, 2:2})
       3_3_2  → BalancedBatchSampler({0:3, 1:3, 2:2})  (current baseline)
       1_2_6  → BalancedBatchSampler({0:1, 1:2, 2:6})
       natural → RandomSampler (不平衡，自然分布)
  5. 输出: 0514/logs/ratio_ablation/{exp_label}/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

# ─── 路径 ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR   = (SCRIPT_DIR / ".." / ".." / "..").resolve()
for _p in [
    ROOT_DIR / "0408" / "scripts",
    ROOT_DIR / "0402" / "scripts",
    ROOT_DIR / "0502" / "scripts",
]:
    sys.path.insert(0, str(_p))

from risk_utils import (  # noqa: E402
    SwinV2SegGuidedRiskTrimodal,
    RiskOrdinalLoss,
    BalancedBatchSampler,
    ORDINAL_TARGETS_DEFAULT,
    search_risk_thresholds_constrained,
    evaluate_risk_bands,
    plot_risk_confusion_matrices,
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
    set_epoch_lrs,
    build_optimizer_with_diff_lr,
    compute_seg_metrics,
)


# ═════════════════════════════════════════════════════════════
#  Sampler 配置
# ═════════════════════════════════════════════════════════════

SAMPLER_CONFIGS = {
    "1_1_1": {0: 2, 1: 2, 2: 2},   # batch_size=6
    "3_3_2": {0: 3, 1: 3, 2: 2},   # batch_size=8 (baseline)
    "1_2_6": {0: 1, 1: 2, 2: 6},   # batch_size=9
    # natural 用 RandomSampler，由代码逻辑分支处理
}


# ═════════════════════════════════════════════════════════════
#  Args
# ═════════════════════════════════════════════════════════════


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_label",  required=True,
                   help="实验标签，输出目录名，例如 S1_1_1_1")
    p.add_argument("--train_xlsx", required=True,
                   help="训练集 xlsx 绝对路径")
    p.add_argument("--val_xlsx",   required=True,
                   help="验证集 xlsx 绝对路径（所有实验共用 baseline 那份）")
    p.add_argument("--sampler",    required=True,
                   choices=["1_1_1", "3_3_2", "1_2_6", "natural"])
    p.add_argument("--num_epochs", type=int, default=20)
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ═════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════


class Config:
    def __init__(self, args):
        self.project_root = str(ROOT_DIR)

        self.data_root         = str(ROOT_DIR / "0514dataset_flat")
        self.train_excel       = args.train_xlsx
        self.val_excel         = args.val_xlsx
        self.test_excel        = str(ROOT_DIR / "0514dataset_flat" / "task_3class_test.xlsx")
        # 用于拟合 meta_stats —— 跨实验一致，永远用 baseline train_split
        self.meta_fit_excel    = str(
            ROOT_DIR / "0514" / "logs" /
            "20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1" / "train_split.xlsx"
        )

        self.clinical_excel    = str(ROOT_DIR / "胆囊超声组学_分析.xlsx")
        self.json_feature_root = str(ROOT_DIR / "json_text_0514")

        self.exp_label = args.exp_label
        self.log_dir = ROOT_DIR / "0514" / "logs" / "ratio_ablation" / self.exp_label
        self.log_file = str(self.log_dir / f"{self.exp_label}.log")
        self.best_weight_path = str(self.log_dir / f"{self.exp_label}_best.pth")

        # 模型（与 baseline 一致）
        self.img_size        = 256
        self.in_channels     = 4
        self.num_seg_classes = 2
        self.cls_dropout     = 0.4
        self.meta_dim        = len(EXT_CLINICAL_FEATURE_NAMES)
        self.meta_hidden     = 96
        self.meta_dropout    = 0.2
        self.text_proj_dim   = 128
        self.text_dropout    = 0.3
        self.max_text_len    = 128
        self.fusion_dim      = 256
        self.ca_hidden       = 128
        self.ca_heads        = 4
        self.ca_dropout      = 0.1

        # 训练
        self.num_epochs    = args.num_epochs
        self.warmup_epochs = 5
        self.backbone_lr   = 2e-5
        self.head_lr       = 2e-4
        self.weight_decay  = 5e-2
        self.min_lr_ratio  = 0.01
        self.grad_clip     = 1.0
        self.num_workers   = 4
        self.eval_interval = 2
        self.seed          = args.seed
        self.use_amp       = True

        # 损失
        self.lambda_ord        = 2.0
        self.seg_bg_weight     = 1.0
        self.seg_lesion_weight = 5.0

        # Sampler 模式
        self.sampler_mode = args.sampler
        if self.sampler_mode == "natural":
            self.samples_per_class = None
            self.batch_size = 8
        else:
            self.samples_per_class = SAMPLER_CONFIGS[self.sampler_mode]
            self.batch_size = sum(self.samples_per_class.values())

        self.ordinal_targets = ORDINAL_TARGETS_DEFAULT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = CLASS_NAMES


# ═════════════════════════════════════════════════════════════
#  Data
# ═════════════════════════════════════════════════════════════


def build_dataloaders(cfg: Config, logger):
    logger.info(f"训练 xlsx: {cfg.train_excel}")
    logger.info(f"验证 xlsx: {cfg.val_excel}")
    logger.info(f"测试 xlsx: {cfg.test_excel}")
    logger.info(f"meta_stats fit xlsx: {cfg.meta_fit_excel}")

    train_tf = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    eval_tf  = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    logger.info("[1/3] 装载 text_dict ...")
    t0 = time.time()
    text_dict = load_text_bert_dict(cfg.json_feature_root)
    logger.info(f"  text_dict: {len(text_dict)} 条 ({time.time()-t0:.1f}s)")

    common_kwargs = dict(
        data_root=cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        max_text_len=cfg.max_text_len,
    )

    logger.info("[2/3] 拟合 meta_stats（用 baseline train_split，保证跨实验一致） ...")
    t0 = time.time()
    stats_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=cfg.meta_fit_excel,
        sync_transform=eval_tf,
        **common_kwargs,
    )
    meta_stats = stats_dataset.meta_stats
    tokenizer  = stats_dataset.tokenizer
    del stats_dataset
    logger.info(f"  meta_stats OK ({time.time()-t0:.1f}s)")

    logger.info("[3/3] 构造 train / val / test Dataset ...")
    t0 = time.time()
    train_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=cfg.train_excel,
        sync_transform=train_tf,
        meta_stats=meta_stats,
        tokenizer=tokenizer,
        **common_kwargs,
    )
    val_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=cfg.val_excel,
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
    logger.info(f"  Dataset OK ({time.time()-t0:.1f}s)")

    for name, ds in [("Train", train_dataset), ("Val", val_dataset), ("Test", test_dataset)]:
        counts = ds.df["label"].value_counts().sort_index().to_dict()
        msg = ", ".join(f"{cfg.class_names[i]}={counts.get(i,0)}"
                        for i in range(len(cfg.class_names)))
        logger.info(f"  {name}: {len(ds)} 张 ({msg})")

    # ── Sampler 分支 ──
    if cfg.sampler_mode == "natural":
        logger.info(f"Sampler: RandomSampler (natural, batch_size={cfg.batch_size})")
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True,
            collate_fn=seg_cls_text_collate_fn,
        )
    else:
        logger.info(f"Sampler: BalancedBatchSampler({cfg.samples_per_class}) "
                    f"batch_size={cfg.batch_size}")
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
    logger.info(f"DataLoaders OK | train_batches={len(train_loader)} "
                f"val={len(val_loader)} test={len(test_loader)}")
    return train_loader, val_loader, test_loader


# ═════════════════════════════════════════════════════════════
#  Model
# ═════════════════════════════════════════════════════════════


def build_model(cfg: Config, logger):
    logger.info("构建模型 ...")
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
        logger.warning(f"pretrained 加载失败, 随机初始化: {exc}")
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


def train_one_epoch(model, loader, criterion, optimizer, device, scaler, use_amp, grad_clip):
    model.train()
    stats = {"loss": 0.0, "seg": 0.0, "ord": 0.0, "total": 0, "seg_dices": []}
    for batch in loader:
        imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch
        imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True); input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True); labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type=="cuda" else "cpu", enabled=use_amp):
            seg_logits, risk_logit = model(imgs, metadata=metas, input_ids=input_ids, attention_mask=attn_mask)
            loss, seg_l, ord_l = criterion(seg_logits, risk_logit, masks, labels, has_masks)
        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer); scaler.update()
        bs = imgs.size(0)
        stats["loss"] += loss.item() * bs; stats["seg"] += seg_l * bs
        stats["ord"] += ord_l * bs; stats["total"] += bs
        if has_masks.any():
            with torch.no_grad():
                idx = has_masks.nonzero(as_tuple=True)[0]
                m = compute_seg_metrics(seg_logits[idx], masks[idx], num_classes=2)
                stats["seg_dices"].append(m["lesion_Dice"])
    n = max(stats["total"], 1)
    return {"loss": stats["loss"]/n, "seg_loss": stats["seg"]/n,
            "ord_loss": stats["ord"]/n,
            "seg_dice": np.mean(stats["seg_dices"]) if stats["seg_dices"] else 0.0}


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_scores, all_labels, seg_dices = [], [], []
    for batch in loader:
        imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch
        imgs = imgs.to(device, non_blocking=True); masks = masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True); input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True); has_masks = has_masks.to(device, non_blocking=True)
        seg_logits, risk_logit = model(imgs, metadata=metas, input_ids=input_ids, attention_mask=attn_mask)
        risk_score = torch.sigmoid(risk_logit)
        all_scores.append(risk_score.detach().float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        if has_masks.any():
            idx = has_masks.nonzero(as_tuple=True)[0]
            m = compute_seg_metrics(seg_logits[idx], masks[idx], num_classes=2)
            seg_dices.append(m["lesion_Dice"])
    return {"scores": np.concatenate(all_scores).astype(np.float64),
            "labels": np.concatenate(all_labels).astype(np.int64),
            "seg_dice": float(np.mean(seg_dices)) if seg_dices else 0.0}


def _val_roc_auc(scores, labels):
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
    args = parse_args()
    cfg = Config(args)

    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_file, cfg.exp_label)

    logger.info("=" * 72)
    logger.info(f"Ratio Ablation Experiment: {cfg.exp_label}")
    logger.info(f"Train: {cfg.train_excel}")
    logger.info(f"Val:   {cfg.val_excel}")
    logger.info(f"Test:  {cfg.test_excel}")
    logger.info(f"Sampler: {cfg.sampler_mode} → samples_per_class={cfg.samples_per_class} "
                f"batch_size={cfg.batch_size}")
    logger.info(f"Epochs: {cfg.num_epochs} | seed: {cfg.seed}")
    logger.info(f"Device: {cfg.device}")
    logger.info("=" * 72)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, logger)
    model = build_model(cfg, logger)

    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight],
        dtype=torch.float32, device=cfg.device,
    )
    criterion = RiskOrdinalLoss(
        ordinal_targets=cfg.ordinal_targets.to(cfg.device),
        lambda_ord=cfg.lambda_ord,
        seg_ce_weight=seg_ce_weight,
    ).to(cfg.device)

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    best_val_auc = 0.0
    best_epoch = 0

    logger.info("=" * 72); logger.info("开始训练"); logger.info("=" * 72)

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
            val_auc = _val_roc_auc(val_pred["scores"], val_pred["labels"])
            val_search = search_risk_thresholds_constrained(
                val_pred["scores"], val_pred["labels"]
            )
            logger.info(
                f"[Val] AUC: {val_auc:.4f} | "
                f"profile={val_search['profile_used']} "
                f"t_low={val_search['t_low']:.3f} t_high={val_search['t_high']:.3f} "
                f"low_p={val_search['metrics']['low_precision']:.4f} "
                f"high_recall={val_search['metrics']['high_recall']:.4f} "
                f"M→低={val_search['metrics']['mal_to_low']} "
                f"med_share={val_search['metrics']['medium_share']:.3f}"
            )
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优 (Val AUC: {best_val_auc:.4f}, Epoch {best_epoch}) ***")

    logger.info("=" * 72)
    logger.info(f"训练完成 | best_epoch={best_epoch} | best_val_auc={best_val_auc:.4f}")
    logger.info("=" * 72)

    # ── 最终评估 ──
    if not os.path.exists(cfg.best_weight_path):
        logger.error("best weight 不存在，跳过最终评估"); return

    state = torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
    model.load_state_dict(state); model.eval()

    val_pred = collect_predictions(model, val_loader, cfg.device)
    val_search = search_risk_thresholds_constrained(
        val_pred["scores"], val_pred["labels"]
    )
    t_low, t_high = val_search["t_low"], val_search["t_high"]

    thresholds_path = cfg.log_dir / "thresholds.json"
    dump_json({
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "t_low": t_low,
        "t_high": t_high,
        "profile_used": val_search["profile_used"],
        "val_metrics": val_search["metrics"],
    }, str(thresholds_path))

    pred = collect_predictions(model, test_loader, cfg.device)
    eval_main = evaluate_risk_bands(
        pred["scores"], pred["labels"], t_low, t_high,
        logger=logger, phase=f"Test ({cfg.exp_label})",
    )
    eval_main["seg_dice"] = pred["seg_dice"]
    eval_main["exp_label"] = cfg.exp_label
    eval_main["sampler_mode"] = cfg.sampler_mode
    eval_main["samples_per_class"] = cfg.samples_per_class
    eval_main["train_xlsx"] = cfg.train_excel
    eval_main["best_epoch"] = best_epoch
    eval_main["best_val_auc"] = best_val_auc

    # 保存 raw scores 给后续阈值搜索 / 比较使用
    np.savez(cfg.log_dir / "test_inference.npz",
             score=pred["scores"], label=pred["labels"])
    np.savez(cfg.log_dir / "val_inference.npz",
             score=val_pred["scores"], label=val_pred["labels"])

    json_path = cfg.log_dir / "eval_test.json"
    dump_json(eval_main, str(json_path))
    logger.info(f"评估结果已保存: {json_path}")

    fig_path = cfg.log_dir / "confusion_matrix.png"
    plot_risk_confusion_matrices({"Test": eval_main}, str(fig_path))
    logger.info(f"混淆矩阵图: {fig_path}")

    logger.info("=" * 72)
    logger.info(f"完成 {cfg.exp_label}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
