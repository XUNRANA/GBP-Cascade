"""0510 Risk 实验 I —— B + H 融合: ordinal 监督 + cascade 决策。

H 实验暴露 stage1 tumor_head val AUC 仅 0.85 (远低于期望的 0.99+),
导致 nt→low 仅 7.5%。但 H 的 cascade 决策在 ben 隔离上极为成功 (ben→low 2.6%)。

I 在 H 基础上加 risk_head + 非对称 ordinal 监督 (B 的 1.0/0.35/0.0):
  - ordinal loss 给 fusion 表示加跨类别连续约束 (mal>ben>nt),
    期望间接提升 stage1 表示质量,把 nt 与 polyp 拉开
  - tumor + mal head 同 H,推理决策也走 H 的 cascade
  - risk_logit 仅作诊断,不参与决策

3 个并行 head,3 个独立温度校准 (risk / tumor / mal,后者只在 tumor 子集 fit)。
预期目标: 5 底线全过 (B 的 nt→high/low + H 的 ben→low + 修补 nt→low)。
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
for _p in [
    os.path.join(ROOT_DIR, "0408", "scripts"),
    os.path.join(ROOT_DIR, "0402", "scripts"),
    os.path.join(ROOT_DIR, "0323", "scripts"),
    os.path.join(ROOT_DIR, "0502", "scripts"),  # 复用 risk_utils.py / risk_utils_B_calib.py
    SCRIPT_DIR,
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


from risk_utils import (  # noqa: E402
    BalancedBatchSampler,
    plot_risk_confusion_matrices,
    extract_patient_id,
    dump_json,
    CLASS_NAMES,
)
from risk_utils_B_calib import (  # noqa: E402
    EarlyStopper, TemperatureScaler, apply_temperature, quick_brier,
)
from risk_utils_E_aux_cls import evaluate_bands_external  # noqa: E402
from risk_utils_H_cascade import (  # noqa: E402
    decide_bands_cascade, search_thresholds_cascade,
)
from risk_utils_I_bh_fusion import (  # noqa: E402
    SwinV2SegGuidedRiskBHFusion, BHFusionLoss,
)
from seg_cls_utils_v5 import (  # noqa: E402
    GBPDatasetSegCls4chWithTextMeta,
    SegCls4chSyncTransform,
    seg_cls_text_collate_fn,
    EXT_CLINICAL_FEATURE_NAMES,
    load_text_bert_dict,
)
from seg_cls_utils_v2 import (  # noqa: E402
    set_seed, setup_logger, acquire_run_lock, set_epoch_lrs,
    build_optimizer_with_diff_lr, compute_seg_metrics,
)


# ═════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════


# I 用 B 的非对称 ordinal target
ORDINAL_TARGETS_I = torch.tensor([1.0, 0.35, 0.0], dtype=torch.float32)


class Config:
    project_root = ROOT_DIR
    primary_data_root = os.path.join(project_root, "0502dataset111")
    primary_train_excel = os.path.join(primary_data_root, "task_3class_train.xlsx")
    primary_test_excel = os.path.join(primary_data_root, "task_3class_test.xlsx")
    secondary_data_root = os.path.join(project_root, "0502dataset112")
    secondary_test_excel = os.path.join(secondary_data_root, "task_3class_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260510_task_risk_I_bh_fusion"
    log_dir = os.path.join(project_root, "0510", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    num_seg_classes = 2
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

    # 实验 B 训练相关 (改动)
    batch_size = 8
    num_epochs = 25            # baseline 60 → 25 (best 已在 epoch 6)
    warmup_epochs = 3
    patience = 5               # 早停
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 2
    seed = 42
    use_amp = True

    lambda_ord = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    samples_per_class = {0: 3, 1: 3, 2: 2}
    val_ratio = 0.15

    # I 用 B 的非对称 ordinal target (辅助监督 fusion 表示)
    ordinal_targets = ORDINAL_TARGETS_I

    # 实验 I 专属: 3 个 head 的 loss 权重
    lambda_mal = 1.0     # mal head 权重 (cascade stage2)
    # lambda_ord 沿用 cfg.lambda_ord = 2.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = CLASS_NAMES
    model_name = (
        "SwinV2-Tiny@256 + 4ch + Trimodal + 3-Heads "
        "(risk ordinal 1.0/0.35/0.0 + tumor BCE + mal BCE-weighted, "
        "triple temp scaling, cascade decision)"
    )


# ═════════════════════════════════════════════════════════════
#  Data (与 baseline 一致)
# ═════════════════════════════════════════════════════════════


def patient_level_train_val_split(train_excel: str, val_ratio: float, seed: int):
    df = pd.read_excel(train_excel)
    df["patient_id"] = df["image_path"].apply(extract_patient_id)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(gss.split(df, df["label"], groups=df["patient_id"]))
    return (df.iloc[train_idx].copy().reset_index(drop=True),
            df.iloc[val_idx].copy().reset_index(drop=True))


def build_dataloaders(cfg: Config, logger):
    logger.info("[1/4] 切分 train/val ...")
    t0 = time.time()
    train_df, val_df = patient_level_train_val_split(
        cfg.primary_train_excel, cfg.val_ratio, cfg.seed)
    train_excel_split = os.path.join(cfg.log_dir, "train_split.xlsx")
    val_excel_split = os.path.join(cfg.log_dir, "val_split.xlsx")
    train_df.to_excel(train_excel_split, index=False)
    val_df.to_excel(val_excel_split, index=False)
    logger.info(f"[1/4] train={len(train_df)} val={len(val_df)} ({time.time()-t0:.1f}s)")

    train_tf = SegCls4chSyncTransform(cfg.img_size, is_train=True)
    eval_tf = SegCls4chSyncTransform(cfg.img_size, is_train=False)

    logger.info("[2/4] 装载 text_dict ...")
    text_dict = load_text_bert_dict(cfg.json_feature_root)
    logger.info(f"[2/4] 共 {len(text_dict)} 条")

    logger.info("[3/4] 构造 train_dataset ...")
    t0 = time.time()
    train_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=train_excel_split, data_root=cfg.primary_data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=train_tf,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict, max_text_len=cfg.max_text_len)
    meta_stats = train_dataset.meta_stats
    tokenizer = train_dataset.tokenizer
    logger.info(f"[3/4] train_dataset OK ({time.time()-t0:.1f}s)")

    def _make_eval_ds(excel, root):
        return GBPDatasetSegCls4chWithTextMeta(
            excel_path=excel, data_root=root,
            clinical_excel_path=cfg.clinical_excel,
            json_feature_root=cfg.json_feature_root,
            sync_transform=eval_tf, meta_stats=meta_stats,
            meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
            text_dict=text_dict, tokenizer=tokenizer,
            max_text_len=cfg.max_text_len)

    val_dataset = _make_eval_ds(val_excel_split, cfg.primary_data_root)
    test111_dataset = _make_eval_ds(cfg.primary_test_excel, cfg.primary_data_root)
    test112_dataset = _make_eval_ds(cfg.secondary_test_excel, cfg.secondary_data_root)
    logger.info("[3/4] eval datasets OK")

    for name, ds in [("Train", train_dataset), ("Val", val_dataset),
                     ("Test-111", test111_dataset), ("Test-112", test112_dataset)]:
        counts = ds.df["label"].value_counts().sort_index().to_dict()
        msg = ", ".join(f"{cfg.class_names[i]}={counts.get(i, 0)}"
                        for i in range(len(cfg.class_names)))
        logger.info(f"{name}: {len(ds)} 张 ({msg})")

    sampler = BalancedBatchSampler(
        labels=train_dataset.df["label"].values,
        samples_per_class=cfg.samples_per_class, shuffle=True)

    train_loader = DataLoader(train_dataset, batch_sampler=sampler,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_text_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_text_collate_fn)
    test111_loader = DataLoader(test111_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_text_collate_fn)
    test112_loader = DataLoader(test112_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        collate_fn=seg_cls_text_collate_fn)
    logger.info(f"[4/4] DataLoaders OK | train_batches={len(train_loader)}")
    return train_loader, val_loader, test111_loader, test112_loader


# ═════════════════════════════════════════════════════════════
#  Model / Optimizer
# ═════════════════════════════════════════════════════════════


def build_model(cfg: Config, logger):
    logger.info("构建模型 (BERT + SwinV2-Tiny + Trimodal + BH-Fusion, 离线加载) ...")
    t0 = time.time()
    common = dict(
        num_seg_classes=cfg.num_seg_classes,
        meta_dim=cfg.meta_dim, meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout, cls_dropout=cfg.cls_dropout,
        text_proj_dim=cfg.text_proj_dim, text_dropout=cfg.text_dropout,
        ca_hidden=cfg.ca_hidden, ca_heads=cfg.ca_heads,
        ca_dropout=cfg.ca_dropout, fusion_dim=cfg.fusion_dim,
        bert_name="bert-base-chinese",
    )
    try:
        model = SwinV2SegGuidedRiskBHFusion(pretrained=True, **common).to(cfg.device)
    except Exception as exc:
        logger.warning(f"pretrained 加载失败: {exc}")
        model = SwinV2SegGuidedRiskBHFusion(pretrained=False, **common).to(cfg.device)
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
#  Train / Eval
# ═════════════════════════════════════════════════════════════


def train_one_epoch(model, loader, criterion, optimizer, device, scaler,
                    use_amp, grad_clip):
    model.train()
    stats = {"loss": 0.0, "seg": 0.0, "ord": 0.0, "tumor": 0.0, "mal": 0.0,
             "total": 0, "seg_dices": []}
    for batch in loader:
        imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type="cuda" if device.type == "cuda" else "cpu",
            enabled=use_amp,
        ):
            seg_logits, risk_logit, tumor_logit, mal_logit = model(
                imgs, metadata=metas,
                input_ids=input_ids, attention_mask=attn_mask)
            loss, seg_l, ord_l, l_tumor, l_mal = criterion(
                seg_logits, risk_logit, tumor_logit, mal_logit,
                masks, labels, has_masks)

        scaler.scale(loss).backward()
        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        bs = imgs.size(0)
        stats["loss"] += loss.item() * bs
        stats["seg"] += seg_l * bs
        stats["ord"] += ord_l * bs
        stats["tumor"] += l_tumor * bs
        stats["mal"] += l_mal * bs
        stats["total"] += bs
        if has_masks.any():
            with torch.no_grad():
                idx = has_masks.nonzero(as_tuple=True)[0]
                m = compute_seg_metrics(seg_logits[idx], masks[idx], num_classes=2)
                stats["seg_dices"].append(m["lesion_Dice"])

    n = max(stats["total"], 1)
    return {
        "loss": stats["loss"] / n,
        "seg_loss": stats["seg"] / n,
        "ord_loss": stats["ord"] / n,
        "tumor_loss": stats["tumor"] / n,
        "mal_loss": stats["mal"] / n,
        "seg_dice": np.mean(stats["seg_dices"]) if stats["seg_dices"] else 0.0,
    }


@torch.no_grad()
def collect_predictions_with_logits(model, loader, device):
    """I 版: 收集 risk / tumor / mal 三个 head 的 logits + sigmoid scores。"""
    model.eval()
    all_lr, all_lt, all_lm = [], [], []
    all_pr, all_pt, all_pm = [], [], []
    all_labels = []
    seg_dices = []
    for batch in loader:
        imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        seg_logits, risk_logit, tumor_logit, mal_logit = model(
            imgs, metadata=metas,
            input_ids=input_ids, attention_mask=attn_mask)
        all_lr.append(risk_logit.detach().float().cpu().numpy())
        all_lt.append(tumor_logit.detach().float().cpu().numpy())
        all_lm.append(mal_logit.detach().float().cpu().numpy())
        all_pr.append(torch.sigmoid(risk_logit).detach().float().cpu().numpy())
        all_pt.append(torch.sigmoid(tumor_logit).detach().float().cpu().numpy())
        all_pm.append(torch.sigmoid(mal_logit).detach().float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        if has_masks.any():
            idx = has_masks.nonzero(as_tuple=True)[0]
            m = compute_seg_metrics(seg_logits[idx], masks[idx], num_classes=2)
            seg_dices.append(m["lesion_Dice"])
    return {
        "risk_logit": np.concatenate(all_lr).astype(np.float64),
        "tumor_logit": np.concatenate(all_lt).astype(np.float64),
        "mal_logit": np.concatenate(all_lm).astype(np.float64),
        "p_risk": np.concatenate(all_pr).astype(np.float64),
        "p_tumor": np.concatenate(all_pt).astype(np.float64),
        "p_mal": np.concatenate(all_pm).astype(np.float64),
        "labels": np.concatenate(all_labels).astype(np.int64),
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

    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_file, cfg.exp_name)

    logger.info("=" * 72)
    logger.info(f"实验: {cfg.exp_name}  (实验 B — 非对称 target + 早停 + 温度校准)")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"Ordinal targets: {cfg.ordinal_targets.tolist()}  (asymmetric)")
    logger.info(f"Epochs: {cfg.num_epochs} | Patience: {cfg.patience}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 72)
    logger.info("初始化完成,开始构建数据流水线 ...")

    train_loader, val_loader, test111_loader, test112_loader = build_dataloaders(cfg, logger)

    model = build_model(cfg, logger)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,} (可训练 {n_trainable:,})")

    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight],
        dtype=torch.float32, device=cfg.device)
    criterion = BHFusionLoss(
        ordinal_targets=cfg.ordinal_targets.to(cfg.device),
        lambda_ord=cfg.lambda_ord,
        lambda_mal=cfg.lambda_mal,
        seg_ce_weight=seg_ce_weight,
    ).to(cfg.device)

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp))

    early_stopper = EarlyStopper(patience=cfg.patience, mode="max")
    best_val_auc = 0.0
    best_epoch = 0

    logger.info("=" * 72)
    logger.info("开始训练 (含早停)")
    logger.info("=" * 72)

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()
        tm = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {tm['loss']:.4f} "
            f"(seg={tm['seg_loss']:.4f} ord={tm['ord_loss']:.4f} "
            f"tumor={tm['tumor_loss']:.4f} mal={tm['mal_loss']:.4f}) "
            f"| Dice: {tm['seg_dice']:.4f} | {elapsed:.0f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            val_pred = collect_predictions_with_logits(model, val_loader, cfg.device)
            # I 用 risk AUC (mal vs others, 沿用 baseline AUC 口径) 做主 monitor
            # 但同时报 tumor/mal AUC 看 stage1 是否被 ordinal 监督拉好了
            y_mal = (val_pred["labels"] == 0).astype(np.int64)
            y_tumor = (val_pred["labels"] != 2).astype(np.int64)
            try:
                risk_auc = float(roc_auc_score(y_mal, val_pred["p_risk"]))
            except ValueError:
                risk_auc = 0.0
            try:
                tumor_auc = float(roc_auc_score(y_tumor, val_pred["p_tumor"]))
            except ValueError:
                tumor_auc = 0.0
            tumor_mask = y_tumor == 1
            try:
                mal_auc = float(roc_auc_score(
                    y_mal[tumor_mask], val_pred["p_mal"][tumor_mask]))
            except ValueError:
                mal_auc = 0.0
            val_auc = risk_auc  # 主 monitor 用 risk AUC (与 B 可比)
            logger.info(
                f"[Val] risk_AUC: {risk_auc:.4f} | tumor_AUC: {tumor_auc:.4f} | "
                f"mal_AUC(on tumor): {mal_auc:.4f}"
            )

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优 (Val AUC: {best_val_auc:.4f}, "
                            f"Epoch: {best_epoch}) ***")

            if early_stopper.step(val_auc):
                logger.info(f"⏹ Early stopping triggered at epoch {epoch} "
                            f"(best epoch={best_epoch}, best AUC={best_val_auc:.4f})")
                break

    logger.info("=" * 72)
    logger.info(f"训练完成。Best Epoch: {best_epoch}, Best Val AUC: {best_val_auc:.4f}")
    logger.info("=" * 72)

    if not os.path.exists(cfg.best_weight_path):
        logger.error("best weight 不存在,跳过最终评估。")
        return

    try:
        state = torch.load(cfg.best_weight_path, map_location=cfg.device,
                           weights_only=True)
    except TypeError:
        state = torch.load(cfg.best_weight_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()

    # ── Triple Temperature Scaling: risk / tumor / mal 各做一次 ──
    val_pred = collect_predictions_with_logits(model, val_loader, cfg.device)
    y_tumor_val = (val_pred["labels"] != 2).astype(np.int64)
    y_mal_val = (val_pred["labels"] == 0).astype(np.int64)
    tumor_mask_val = y_tumor_val == 1

    # risk head 校准 (作诊断,不参与决策,但记录便于对比)
    ts_risk = TemperatureScaler(init_T=1.0).to(cfg.device)
    T_risk = ts_risk.fit(val_pred["risk_logit"], y_mal_val,
                         max_iter=200, lr=0.05)

    # tumor head 校准
    ts_tumor = TemperatureScaler(init_T=1.0).to(cfg.device)
    T_tumor = ts_tumor.fit(val_pred["tumor_logit"], y_tumor_val,
                           max_iter=200, lr=0.05)

    # mal head 校准: 只在 tumor 子集上 fit
    ts_mal = TemperatureScaler(init_T=1.0).to(cfg.device)
    if tumor_mask_val.sum() > 0:
        T_mal = ts_mal.fit(val_pred["mal_logit"][tumor_mask_val],
                           y_mal_val[tumor_mask_val],
                           max_iter=200, lr=0.05)
    else:
        T_mal = 1.0

    p_tumor_calib_val = apply_temperature(val_pred["tumor_logit"], T_tumor)
    p_mal_calib_val = apply_temperature(val_pred["mal_logit"], T_mal)
    p_risk_calib_val = apply_temperature(val_pred["risk_logit"], T_risk)

    val_brier_risk_raw = quick_brier(val_pred["p_risk"], val_pred["labels"])
    val_brier_risk_calib = quick_brier(p_risk_calib_val, val_pred["labels"])

    logger.info("=" * 72)
    logger.info(f"[Temperature Scaling] T_risk = {T_risk:.4f} "
                f"T_tumor = {T_tumor:.4f} T_mal = {T_mal:.4f}")
    logger.info(f"  val Brier(risk): raw={val_brier_risk_raw:.4f} "
                f"calib={val_brier_risk_calib:.4f}")
    logger.info("=" * 72)

    # ── Cascade 阈值搜索 (用校准后的 p_tumor, p_mal) ──
    i_best = search_thresholds_cascade(
        p_tumor_calib_val, p_mal_calib_val, val_pred["labels"])

    if i_best is None:
        logger.error("I cascade 阈值搜索完全失败,使用 (tau1=0.5, tau2=0.5) 作为兜底。")
        tau1, tau2 = 0.5, 0.5
        chosen_method = "hard_default"
    else:
        tau1, tau2 = i_best["tau1"], i_best["tau2"]
        chosen_method = (f"I_strict" if i_best.get("fallback_used") is None
                         else f"I_{i_best['fallback_used']}")
        logger.info(f"采用 I cascade 阈值 ({chosen_method}): "
                    f"tau1={tau1:.3f} tau2={tau2:.3f}")
        logger.info(f"  → tumor_recall={i_best['tumor_recall']:.4f} "
                    f"high_p={i_best['high_precision']:.4f} "
                    f"low_p_nt={i_best['low_precision_nt']:.4f} "
                    f"ben_to_low={i_best['ben_to_low_share']:.3f} "
                    f"nt_to_high={i_best['nt_to_high_share']:.3f} "
                    f"obj={i_best['objective']:.4f}")
    logger.info("=" * 72)

    thresholds_path = os.path.join(cfg.log_dir, "thresholds.json")
    dump_json({
        "selection_set": "val",
        "best_epoch": best_epoch,
        "best_val_auc": best_val_auc,
        "ordinal_targets": cfg.ordinal_targets.tolist(),
        "T_risk": T_risk, "T_tumor": T_tumor, "T_mal": T_mal,
        "lambda_ord": cfg.lambda_ord, "lambda_mal": cfg.lambda_mal,
        "val_brier_risk_raw": val_brier_risk_raw,
        "val_brier_risk_calib": val_brier_risk_calib,
        "tau1": tau1, "tau2": tau2,
        "chosen_method": chosen_method,
        "i_search_result": i_best,
    }, thresholds_path)
    logger.info(f"阈值已保存: {thresholds_path}")

    # ── 在测试集上用校准后的 (p_tumor, p_mal) + cascade 阈值评估 ──
    eval_results = {}
    for phase, loader, tag in [
        ("Test-111", test111_loader, "111"),
        ("Test-112", test112_loader, "112"),
    ]:
        pred = collect_predictions_with_logits(model, loader, cfg.device)
        p_tumor_calib = apply_temperature(pred["tumor_logit"], T_tumor)
        p_mal_calib = apply_temperature(pred["mal_logit"], T_mal)
        p_risk_calib = apply_temperature(pred["risk_logit"], T_risk)

        bands = decide_bands_cascade(p_tumor_calib, p_mal_calib, tau1, tau2)

        # 主决策用 p_mal_calib 作 score 占位 (做 distribution / AUC),
        # 但额外保存 p_risk_calib 的 brier 作诊断 (与 B 可比)
        ev = evaluate_bands_external(
            p_mal_calib, bands, pred["labels"],
            phase=f"{phase} (I cascade val-thr)",
            extra_meta=dict(
                thresholds=dict(tau1=tau1, tau2=tau2,
                                T_risk=T_risk, T_tumor=T_tumor, T_mal=T_mal),
                seg_dice=pred["seg_dice"],
                p_risk_raw_brier=quick_brier(pred["p_risk"], pred["labels"]),
                p_risk_calib_brier=quick_brier(p_risk_calib, pred["labels"]),
                p_tumor_calib_brier=quick_brier(p_tumor_calib, pred["labels"]),
                p_mal_calib_brier=quick_brier(p_mal_calib, pred["labels"]),
            ))
        logger.info(f"[{phase}] safety: {ev['safety']}")
        logger.info(f"[{phase}] Brier(risk): raw={ev['p_risk_raw_brier']:.4f} "
                    f"calib={ev['p_risk_calib_brier']:.4f}")

        # 诊断 — self-search cascade 阈值
        diag = search_thresholds_cascade(p_tumor_calib, p_mal_calib, pred["labels"])
        ev["diagnostic_search_on_self"] = diag
        if diag is not None:
            logger.info(f"[{phase}] 诊断 self-search: tau1={diag['tau1']:.3f} "
                        f"tau2={diag['tau2']:.3f} "
                        f"obj={diag.get('objective', 0):.4f} "
                        f"ben_to_low={diag.get('ben_to_low_share', 0):.3f} "
                        f"nt_to_high={diag.get('nt_to_high_share', 0):.3f}")

        json_path = os.path.join(cfg.log_dir, f"eval_{tag}.json")
        dump_json(ev, json_path)
        logger.info(f"评估已保存: {json_path}")
        eval_results[phase] = ev

    fig_path = os.path.join(cfg.log_dir, "confusion_matrices.png")
    plot_risk_confusion_matrices(eval_results, fig_path)
    logger.info(f"混淆矩阵图: {fig_path}")

    code_snapshots = [
        __file__,
        os.path.join(SCRIPT_DIR, "risk_utils_I_bh_fusion.py"),
        os.path.join(SCRIPT_DIR, "risk_utils_H_cascade.py"),
        os.path.join(SCRIPT_DIR, "risk_utils_E_aux_cls.py"),
        os.path.join(ROOT_DIR, "0502", "scripts", "risk_utils.py"),
        os.path.join(ROOT_DIR, "0502", "scripts", "risk_utils_B_calib.py"),
    ]
    for src in code_snapshots:
        if not os.path.exists(src):
            continue
        dst = os.path.join(cfg.log_dir, os.path.basename(src))
        if os.path.abspath(src) != os.path.abspath(dst):
            Path(dst).write_text(Path(src).read_text(encoding="utf-8"),
                                 encoding="utf-8")

    logger.info("=" * 72)
    logger.info("全部完成。")
    logger.info(f"  best weight    : {cfg.best_weight_path}")
    logger.info(f"  thresholds.json: {thresholds_path}")
    logger.info(f"  eval_111.json  : {os.path.join(cfg.log_dir, 'eval_111.json')}")
    logger.info(f"  eval_112.json  : {os.path.join(cfg.log_dir, 'eval_112.json')}")
    logger.info(f"  混淆矩阵图     : {fig_path}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
