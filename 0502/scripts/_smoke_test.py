"""Smoke test (不会落盘训练): 跑一个 batch 的前向+反向，确认整条链路 OK。

跑法：
  CUDA_VISIBLE_DEVICES=0 python /data1/ouyangxinglong/GBP-Cascade/0502/scripts/_smoke_test.py
"""

import os
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", ".."))
for p in [
    os.path.join(ROOT_DIR, "0408", "scripts"),
    os.path.join(ROOT_DIR, "0402", "scripts"),
    os.path.join(ROOT_DIR, "0323", "scripts"),
    THIS_DIR,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util

# 直接 import 主脚本 (文件名带数字开头)
SCRIPT_PATH = os.path.join(
    THIS_DIR, "20260502_task_risk_SwinV2Tiny_ordinal_trimodal_1.py"
)
spec = importlib.util.spec_from_file_location("risk_main", SCRIPT_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from risk_utils import (
    SwinV2SegGuidedRiskTrimodal, RiskOrdinalLoss,
    ORDINAL_TARGETS_DEFAULT, search_risk_thresholds_constrained,
    evaluate_risk_bands, BalancedBatchSampler,
)


class _StubLogger:
    def info(self, *a): print("[info]", *a)
    def warning(self, *a): print("[warn]", *a)
    def error(self, *a): print("[err]", *a)


def main():
    print("=" * 60)
    print("0502 smoke test")
    print("=" * 60)

    cfg = mod.Config()
    # 缩短：smoke 不需要全量训练
    cfg.num_epochs = 1
    cfg.warmup_epochs = 0
    cfg.eval_interval = 1
    cfg.num_workers = 2
    cfg.log_dir = os.path.join(cfg.log_dir + "_smoke")
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = _StubLogger()

    print(f"Device: {cfg.device}")
    print(f"Primary data root: {cfg.primary_data_root}")
    print(f"Secondary data root: {cfg.secondary_data_root}")

    # 1) DataLoaders
    print("\n[1/5] Building dataloaders ...")
    t0 = time.time()
    train_loader, val_loader, test111_loader, test112_loader = \
        mod.build_dataloaders(cfg, logger)
    print(f"    train batches/epoch: {len(train_loader)}")
    print(f"    val batches: {len(val_loader)}")
    print(f"    test-111 batches: {len(test111_loader)}")
    print(f"    test-112 batches: {len(test112_loader)}")
    print(f"    [{time.time()-t0:.1f}s]")

    # 2) 模型
    print("\n[2/5] Building model ...")
    t0 = time.time()
    model = mod.build_model(cfg, logger)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    params total: {n_params:,} | trainable: {n_train:,}")
    print(f"    [{time.time()-t0:.1f}s]")

    # 3) 一个 batch 前向 + 反向
    print("\n[3/5] One-batch forward + backward ...")
    seg_ce_w = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight],
        dtype=torch.float32, device=cfg.device)
    criterion = RiskOrdinalLoss(
        ordinal_targets=cfg.ordinal_targets.to(cfg.device),
        lambda_ord=cfg.lambda_ord,
        seg_ce_weight=seg_ce_w,
    ).to(cfg.device)
    optimizer = mod.build_optimizer(model, cfg)

    batch = next(iter(train_loader))
    imgs, masks, metas, input_ids, attn_mask, labels, has_masks = batch
    print(f"    imgs: {imgs.shape}, masks: {masks.shape}, "
          f"metas: {metas.shape}, input_ids: {input_ids.shape}, "
          f"labels: {labels.tolist()}, has_masks: {has_masks.tolist()}")

    imgs = imgs.to(cfg.device)
    masks = masks.to(cfg.device)
    metas = metas.to(cfg.device)
    input_ids = input_ids.to(cfg.device)
    attn_mask = attn_mask.to(cfg.device)
    labels = labels.to(cfg.device)
    has_masks = has_masks.to(cfg.device)

    model.train()
    optimizer.zero_grad()
    with torch.amp.autocast(device_type="cuda", enabled=True):
        seg_logits, risk_logit = model(
            imgs, metadata=metas,
            input_ids=input_ids, attention_mask=attn_mask,
        )
        loss, seg_l, ord_l = criterion(seg_logits, risk_logit, masks, labels, has_masks)
    risk_score = torch.sigmoid(risk_logit)
    print(f"    seg_logits: {seg_logits.shape}  risk_logit: {risk_logit.shape} "
          f"sigmoid(min/max)={risk_score.min().item():.4f}/{risk_score.max().item():.4f}")
    assert risk_logit.dim() == 1 and risk_logit.size(0) == imgs.size(0)
    assert (risk_score >= 0).all() and (risk_score <= 1).all()
    print(f"    loss={loss.item():.4f} (seg={seg_l:.4f}, ord={ord_l:.4f})")
    assert torch.isfinite(loss).item(), "loss not finite!"
    loss.backward()
    optimizer.step()
    print("    backward + step OK")

    # 4) 验证集采集 + 阈值搜索
    print("\n[4/5] Collect val predictions + threshold search ...")
    t0 = time.time()
    val_pred = mod.collect_predictions(model, val_loader, cfg.device)
    print(f"    val: scores shape={val_pred['scores'].shape} "
          f"labels shape={val_pred['labels'].shape} "
          f"score range=[{val_pred['scores'].min():.3f}, "
          f"{val_pred['scores'].max():.3f}] "
          f"seg_dice={val_pred['seg_dice']:.4f}")
    auc = mod._val_roc_auc(val_pred["scores"], val_pred["labels"])
    print(f"    val ROC-AUC (mal vs others): {auc:.4f}")
    res = search_risk_thresholds_constrained(
        val_pred["scores"], val_pred["labels"])
    print(f"    threshold search: profile={res['profile_used']} "
          f"t_low={res['t_low']:.3f} t_high={res['t_high']:.3f}")
    print(f"    metrics: {res['metrics']}")
    print(f"    [{time.time()-t0:.1f}s]")

    # 5) 评估 + 画图
    print("\n[5/5] Evaluate on val (using its own thresholds) ...")
    t0 = time.time()
    ev = evaluate_risk_bands(
        val_pred["scores"], val_pred["labels"],
        res["t_low"], res["t_high"],
        logger=_StubLogger(), phase="smoke-val",
    )
    print(f"    confusion_3band: {ev['confusion_3band']}")
    print(f"    safety: {ev['safety']}")
    print(f"    [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 60)
    print("Smoke test PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
