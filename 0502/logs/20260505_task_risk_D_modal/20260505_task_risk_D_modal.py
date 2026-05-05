"""0502 Risk 实验 D —— Modality Dropout + Gate 熵正则 + 文本/临床增强。

核心改动 (相对 baseline)
  1. 模型: SwinV2SegGuidedRiskModalDropout —— 训练态 batch-level modality
     dropout (text/meta 独立 p=0.15),forward 返回 (seg_logits, risk_logit, gates)
  2. 损失: RiskOrdinalGateEntropyLoss —— 在 BCE+seg 基础上加 - λ_gate * H(gates)
  3. 训练增强:
       - 训练时对 input_ids 做 BERT-style 15% token masking
       - 训练时给 metadata 加 N(0, 0.05) 高斯噪声
  4. 训练日志多记录 epoch 平均 gate weights (image / text / meta)
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
    SCRIPT_DIR,
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


from risk_utils import (  # noqa: E402
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
from risk_utils_D_modal import (  # noqa: E402
    SwinV2SegGuidedRiskModalDropout,
    RiskOrdinalGateEntropyLoss,
    mlm_mask_input_ids,
    add_meta_noise,
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


class Config:
    project_root = ROOT_DIR
    primary_data_root = os.path.join(project_root, "0502dataset111")
    primary_train_excel = os.path.join(primary_data_root, "task_3class_train.xlsx")
    primary_test_excel = os.path.join(primary_data_root, "task_3class_test.xlsx")
    secondary_data_root = os.path.join(project_root, "0502dataset112")
    secondary_test_excel = os.path.join(secondary_data_root, "task_3class_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260505_task_risk_D_modal"
    log_dir = os.path.join(project_root, "0502", "logs", exp_name)
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

    # 实验 D 专属
    modal_dropout_p = 0.15      # text / meta 各自的 batch-level drop 概率
    lambda_gate = 0.05           # gate 熵正则权重
    text_mlm_prob = 0.15         # 训练时对 token 的 15% mask
    meta_noise_sigma = 0.05      # 训练时对临床特征加噪

    # 训练
    batch_size = 8
    num_epochs = 60
    warmup_epochs = 5
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
    ordinal_targets = ORDINAL_TARGETS_DEFAULT

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = CLASS_NAMES
    model_name = (
        "SwinV2-Tiny@256 + 4ch + Trimodal + 1D Risk Head "
        "(modal-dropout, gate-entropy, text-mlm-aug, meta-noise)"
    )


# ═════════════════════════════════════════════════════════════
#  Data
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
    return train_loader, val_loader, test111_loader, test112_loader, tokenizer


# ═════════════════════════════════════════════════════════════
#  Model / Optimizer
# ═════════════════════════════════════════════════════════════


def build_model(cfg: Config, logger):
    logger.info("构建 ModalDropout 模型 ...")
    t0 = time.time()
    common = dict(
        num_seg_classes=cfg.num_seg_classes,
        meta_dim=cfg.meta_dim, meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout, cls_dropout=cfg.cls_dropout,
        text_proj_dim=cfg.text_proj_dim, text_dropout=cfg.text_dropout,
        ca_hidden=cfg.ca_hidden, ca_heads=cfg.ca_heads,
        ca_dropout=cfg.ca_dropout, fusion_dim=cfg.fusion_dim,
        bert_name="bert-base-chinese", modal_dropout_p=cfg.modal_dropout_p,
    )
    try:
        model = SwinV2SegGuidedRiskModalDropout(pretrained=True, **common).to(cfg.device)
    except Exception as exc:
        logger.warning(f"pretrained 加载失败: {exc}")
        model = SwinV2SegGuidedRiskModalDropout(pretrained=False, **common).to(cfg.device)
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
                    use_amp, grad_clip, cfg, mask_token_id):
    model.train()
    stats = {"loss": 0.0, "seg": 0.0, "ord": 0.0, "gate_H": 0.0,
             "total": 0, "seg_dices": [],
             "gate_img_sum": 0.0, "gate_text_sum": 0.0, "gate_meta_sum": 0.0}

    for batch in loader:
        imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch

        # ── 训练增强: token mask + meta noise (在 to-device 之前更便宜) ──
        input_ids = mlm_mask_input_ids(
            input_ids, attn_mask,
            mask_token_id=mask_token_id,
            mask_prob=cfg.text_mlm_prob,
        )
        metas = add_meta_noise(metas, sigma=cfg.meta_noise_sigma)

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
            seg_logits, risk_logit, gates = model(
                imgs, metadata=metas,
                input_ids=input_ids, attention_mask=attn_mask)
            loss, seg_l, ord_l, gate_h = criterion(
                seg_logits, risk_logit, gates,
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
        stats["gate_H"] += gate_h * bs
        stats["total"] += bs
        # 记录三模态 gate 平均权重
        with torch.no_grad():
            stats["gate_img_sum"] += float(gates[:, 0].float().sum().item())
            stats["gate_text_sum"] += float(gates[:, 1].float().sum().item())
            stats["gate_meta_sum"] += float(gates[:, 2].float().sum().item())
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
        "gate_entropy": stats["gate_H"] / n,
        "seg_dice": np.mean(stats["seg_dices"]) if stats["seg_dices"] else 0.0,
        "gate_img": stats["gate_img_sum"] / n,
        "gate_text": stats["gate_text_sum"] / n,
        "gate_meta": stats["gate_meta_sum"] / n,
    }


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_scores, all_labels = [], []
    seg_dices = []
    for batch in loader:
        imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        input_ids = input_ids.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)
        has_masks = has_masks.to(device, non_blocking=True)

        seg_logits, risk_logit, _gates = model(
            imgs, metadata=metas,
            input_ids=input_ids, attention_mask=attn_mask)
        risk_score = torch.sigmoid(risk_logit)
        all_scores.append(risk_score.detach().float().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
        if has_masks.any():
            idx = has_masks.nonzero(as_tuple=True)[0]
            m = compute_seg_metrics(seg_logits[idx], masks[idx], num_classes=2)
            seg_dices.append(m["lesion_Dice"])

    return {
        "scores": np.concatenate(all_scores).astype(np.float64),
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
    logger.info(f"实验: {cfg.exp_name}  (实验 D — 模态平衡)")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"Modal dropout p: {cfg.modal_dropout_p} (text/meta 各自)")
    logger.info(f"Lambda gate: {cfg.lambda_gate} (- λ * H(gates))")
    logger.info(f"Text MLM prob: {cfg.text_mlm_prob} | Meta noise sigma: {cfg.meta_noise_sigma}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 72)
    logger.info("初始化完成,开始构建数据流水线 ...")

    train_loader, val_loader, test111_loader, test112_loader, tokenizer = \
        build_dataloaders(cfg, logger)

    mask_token_id = int(tokenizer.mask_token_id)
    logger.info(f"BERT mask_token_id = {mask_token_id}")

    model = build_model(cfg, logger)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,} (可训练 {n_trainable:,})")

    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight],
        dtype=torch.float32, device=cfg.device)
    criterion = RiskOrdinalGateEntropyLoss(
        ordinal_targets=cfg.ordinal_targets.to(cfg.device),
        lambda_ord=cfg.lambda_ord,
        lambda_gate=cfg.lambda_gate,
        seg_ce_weight=seg_ce_weight,
    ).to(cfg.device)

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp))

    best_val_auc = 0.0
    best_epoch = 0

    logger.info("=" * 72)
    logger.info("开始训练")
    logger.info("=" * 72)

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()
        tm = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip, cfg=cfg, mask_token_id=mask_token_id)
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {tm['loss']:.4f} (seg={tm['seg_loss']:.4f} ord={tm['ord_loss']:.4f}) "
            f"| Dice: {tm['seg_dice']:.4f} "
            f"| Gate img/text/meta: {tm['gate_img']:.3f}/{tm['gate_text']:.3f}/"
            f"{tm['gate_meta']:.3f} (H={tm['gate_entropy']:.3f}) "
            f"| {elapsed:.0f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            val_pred = collect_predictions(model, val_loader, cfg.device)
            val_auc = _val_roc_auc(val_pred["scores"], val_pred["labels"])
            val_search = search_risk_thresholds_constrained(
                val_pred["scores"], val_pred["labels"])
            logger.info(
                f"[Val] ROC-AUC: {val_auc:.4f} | "
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
                logger.info(f"*** 保存最优 (Val AUC: {best_val_auc:.4f}, "
                            f"Epoch: {best_epoch}) ***")

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

    val_pred = collect_predictions(model, val_loader, cfg.device)
    val_search = search_risk_thresholds_constrained(
        val_pred["scores"], val_pred["labels"])
    t_low, t_high = val_search["t_low"], val_search["t_high"]
    logger.info("=" * 72)
    logger.info(f"在 val 上做约束阈值搜索: profile={val_search['profile_used']}")
    logger.info(f"  → t_low={t_low:.3f}, t_high={t_high:.3f}")
    logger.info(f"  val: {val_search['metrics']}")
    logger.info("=" * 72)

    thresholds_path = os.path.join(cfg.log_dir, "thresholds.json")
    dump_json({
        "selection_set": "val", "best_epoch": best_epoch,
        "best_val_auc": best_val_auc, "t_low": t_low, "t_high": t_high,
        "profile_used": val_search["profile_used"],
        "val_metrics": val_search["metrics"],
        "all_candidates_top10": val_search["all_candidates"][:10],
    }, thresholds_path)
    logger.info(f"阈值已保存: {thresholds_path}")

    eval_results = {}
    for phase, loader, tag in [
        ("Test-111", test111_loader, "111"),
        ("Test-112", test112_loader, "112"),
    ]:
        pred = collect_predictions(model, loader, cfg.device)
        ev = evaluate_risk_bands(pred["scores"], pred["labels"], t_low, t_high,
                                  logger=logger, phase=f"{phase} (val-thr)")
        ev["seg_dice"] = pred["seg_dice"]
        diag = search_risk_thresholds_constrained(pred["scores"], pred["labels"])
        logger.info(
            f"[{phase}] 诊断 — 直接在该集上搜阈值: profile={diag['profile_used']} "
            f"t_low={diag['t_low']:.3f} t_high={diag['t_high']:.3f} "
            f"low_p={diag['metrics']['low_precision']:.4f}")
        ev["diagnostic_search_on_self"] = {
            "profile_used": diag["profile_used"],
            "t_low": diag["t_low"], "t_high": diag["t_high"],
            "metrics": diag["metrics"],
        }
        json_path = os.path.join(cfg.log_dir, f"eval_{tag}.json")
        dump_json(ev, json_path)
        logger.info(f"评估已保存: {json_path}")
        eval_results[phase] = ev

    fig_path = os.path.join(cfg.log_dir, "confusion_matrices.png")
    plot_risk_confusion_matrices(eval_results, fig_path)
    logger.info(f"混淆矩阵图: {fig_path}")

    for src in [__file__,
                os.path.join(SCRIPT_DIR, "risk_utils.py"),
                os.path.join(SCRIPT_DIR, "risk_utils_D_modal.py")]:
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
