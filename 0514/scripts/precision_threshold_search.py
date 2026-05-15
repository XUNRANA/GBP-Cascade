"""
precision_threshold_search.py

目标：在已训练好的 0514 best 模型基础上，**仅做后处理调阈值**，找到一组
(t_low, t_high) 使得 high band 和 low band 的 precision 同时尽可能高
（medium band 当不确定区，可任意膨胀）。

硬底线（不可破）:
  mal_to_low == 0
  high_recall >= 0.95

策略:
  1. 用 best.pth 对 val / test 各跑一次推理，保存 raw logit + sigmoid(score) + label。
  2. 在 val 上做二维网格搜索：t_low ∈ [0.005, 0.45]、t_high ∈ [0.50, 0.99]。
  3. 对每个 precision 档位 ∈ {0.60, 0.70, 0.80, 0.90, 0.95, 0.99}:
       筛 high_p >= target AND low_p >= target AND mal_to_low==0 AND high_recall>=0.95
       挑 medium_share 最小的（即覆盖率最大、最有用的那一对阈值）
  4. 把选出的阈值应用到 test，得到 test 上的实际 high_p / low_p / 覆盖率。
  5. 额外做 "在 test 上直接搜阈值" 的诊断对比，判断 val→test 是否过拟合。

输出（写入 0514/logs/precision_search/）:
  - inference_val.npz / inference_test.npz: 每样本的 (logit, score, label)
  - precision_search_results.json: 每档位的完整 metrics
  - precision_coverage_curve.png: precision-coverage Pareto 曲线
  - confusion_matrices_by_target.png: 6 档位的混淆矩阵
  - search_log.txt: 人类可读摘要表
"""

from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

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
    evaluate_risk_bands,
    CLASS_NAMES,
)
from seg_cls_utils_v5 import (  # noqa: E402
    GBPDatasetSegCls4chWithTextMeta,
    SegCls4chSyncTransform,
    seg_cls_text_collate_fn,
    EXT_CLINICAL_FEATURE_NAMES,
    load_text_bert_dict,
)
from seg_cls_utils_v2 import set_seed  # noqa: E402


# ═════════════════════════════════════════════════════════════
#  Config
# ═════════════════════════════════════════════════════════════


PROJECT_ROOT = ROOT_DIR
TRAINED_DIR = os.path.join(
    PROJECT_ROOT, "0514", "logs",
    "20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1",
)
BEST_WEIGHT = os.path.join(
    TRAINED_DIR, "20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1_best.pth",
)
TRAIN_SPLIT_XLSX = os.path.join(TRAINED_DIR, "train_split.xlsx")
VAL_SPLIT_XLSX   = os.path.join(TRAINED_DIR, "val_split.xlsx")

DATA_ROOT      = os.path.join(PROJECT_ROOT, "0514dataset_flat")
TEST_XLSX      = os.path.join(DATA_ROOT, "task_3class_test.xlsx")
CLINICAL_XLSX  = os.path.join(PROJECT_ROOT, "胆囊超声组学_分析.xlsx")
JSON_TEXT_ROOT = os.path.join(PROJECT_ROOT, "json_text_0514")

OUT_DIR = os.path.join(PROJECT_ROOT, "0514", "logs", "precision_search")
os.makedirs(OUT_DIR, exist_ok=True)

# 模型 / 推理超参（必须与训练时一致）
IMG_SIZE        = 256
NUM_SEG_CLASSES = 2
CLS_DROPOUT     = 0.4
META_HIDDEN     = 96
META_DROPOUT    = 0.2
TEXT_PROJ_DIM   = 128
TEXT_DROPOUT    = 0.3
MAX_TEXT_LEN    = 128
FUSION_DIM      = 256
CA_HIDDEN       = 128
CA_HEADS        = 4
CA_DROPOUT      = 0.1
BATCH_SIZE      = 8
NUM_WORKERS     = 4
SEED            = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 阈值搜索网格
T_LOW_GRID  = np.round(np.arange(0.005, 0.45 + 1e-9, 0.005), 4)   # 90 点
T_HIGH_GRID = np.round(np.arange(0.50, 0.99 + 1e-9, 0.01), 4)    # 50 点

# Precision 档位
PRECISION_TARGETS = [0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

# 安全硬底线（永远不破）
MAL_TO_LOW_MAX = 0
HIGH_RECALL_MIN = 0.95


# ═════════════════════════════════════════════════════════════
#  Data & Model
# ═════════════════════════════════════════════════════════════


def build_datasets():
    """返回 (val_dataset, test_dataset)。meta_stats 从 train_split 拟合。"""
    eval_tf = SegCls4chSyncTransform(IMG_SIZE, is_train=False)

    text_dict = load_text_bert_dict(JSON_TEXT_ROOT)
    print(f"  text_dict 加载: {len(text_dict)} 条")

    common_kwargs = dict(
        data_root=DATA_ROOT,
        clinical_excel_path=CLINICAL_XLSX,
        json_feature_root=JSON_TEXT_ROOT,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        max_text_len=MAX_TEXT_LEN,
    )

    # 用 train_split 拟合 meta_stats（与训练完全一致）
    train_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=TRAIN_SPLIT_XLSX,
        sync_transform=eval_tf,   # 拟合用，不做增强
        **common_kwargs,
    )
    meta_stats = train_dataset.meta_stats
    tokenizer  = train_dataset.tokenizer

    val_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=VAL_SPLIT_XLSX,
        sync_transform=eval_tf,
        meta_stats=meta_stats,
        tokenizer=tokenizer,
        **common_kwargs,
    )
    test_dataset = GBPDatasetSegCls4chWithTextMeta(
        excel_path=TEST_XLSX,
        sync_transform=eval_tf,
        meta_stats=meta_stats,
        tokenizer=tokenizer,
        **common_kwargs,
    )
    return val_dataset, test_dataset


def build_model():
    model = SwinV2SegGuidedRiskTrimodal(
        num_seg_classes=NUM_SEG_CLASSES,
        meta_dim=len(EXT_CLINICAL_FEATURE_NAMES),
        meta_hidden=META_HIDDEN,
        meta_dropout=META_DROPOUT,
        cls_dropout=CLS_DROPOUT,
        text_proj_dim=TEXT_PROJ_DIM,
        text_dropout=TEXT_DROPOUT,
        ca_hidden=CA_HIDDEN,
        ca_heads=CA_HEADS,
        ca_dropout=CA_DROPOUT,
        fusion_dim=FUSION_DIM,
        bert_name="bert-base-chinese",
        pretrained=False,   # 立刻 load_state_dict 覆盖
    ).to(DEVICE)
    try:
        state = torch.load(BEST_WEIGHT, map_location=DEVICE, weights_only=True)
    except TypeError:
        state = torch.load(BEST_WEIGHT, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def run_inference(model, dataset, name: str) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
        collate_fn=seg_cls_text_collate_fn,
    )
    logits, scores, labels = [], [], []
    t0 = time.time()
    for batch in loader:
        imgs, masks, metas, input_ids, attn_mask, lbls, _ = batch
        imgs      = imgs.to(DEVICE, non_blocking=True)
        metas     = metas.to(DEVICE, non_blocking=True)
        input_ids = input_ids.to(DEVICE, non_blocking=True)
        attn_mask = attn_mask.to(DEVICE, non_blocking=True)
        _, risk_logit = model(
            imgs, metadata=metas,
            input_ids=input_ids, attention_mask=attn_mask,
        )
        score = torch.sigmoid(risk_logit)
        logits.append(risk_logit.detach().float().cpu().numpy())
        scores.append(score.detach().float().cpu().numpy())
        labels.append(lbls.detach().cpu().numpy())
    out = {
        "logit": np.concatenate(logits).astype(np.float64),
        "score": np.concatenate(scores).astype(np.float64),
        "label": np.concatenate(labels).astype(np.int64),
    }
    print(f"  [{name}] 推理 {len(out['label'])} 张, {time.time()-t0:.1f}s")
    return out


# ═════════════════════════════════════════════════════════════
#  Threshold grid metrics
# ═════════════════════════════════════════════════════════════


def compute_metrics_for_thresholds(
    scores: np.ndarray, labels: np.ndarray,
    t_low: float, t_high: float,
) -> Dict[str, float]:
    """单组 (t_low, t_high) 在 (scores, labels) 上的核心指标。
    保持与 evaluate_risk_bands 一致的语义。
    """
    n = len(labels)
    bands = np.full(n, 1, dtype=np.int64)   # 默认 medium
    bands[scores >= t_high] = 0             # high
    bands[scores <= t_low]  = 2             # low

    is_mal  = (labels == 0)
    is_nt   = (labels == 2)
    is_ben  = (labels == 1)
    pred_h  = (bands == 0)
    pred_m  = (bands == 1)
    pred_l  = (bands == 2)

    n_h = int(pred_h.sum())
    n_m = int(pred_m.sum())
    n_l = int(pred_l.sum())

    tp_h = int((pred_h & is_mal).sum())
    tp_l = int((pred_l & is_nt).sum())
    tp_m = int((pred_m & is_ben).sum())

    high_p = (tp_h / n_h) if n_h > 0 else 0.0
    low_p  = (tp_l / n_l) if n_l > 0 else 0.0
    med_p  = (tp_m / n_m) if n_m > 0 else 0.0

    mal_total = int(is_mal.sum())
    high_recall = (tp_h / mal_total) if mal_total > 0 else 0.0
    mal_to_low  = int(((bands == 2) & is_mal).sum())

    return {
        "t_low":          float(t_low),
        "t_high":         float(t_high),
        "high_precision": float(high_p),
        "low_precision":  float(low_p),
        "med_precision":  float(med_p),
        "high_recall":    float(high_recall),
        "mal_to_low":     int(mal_to_low),
        "n_high":         n_h,
        "n_med":          n_m,
        "n_low":          n_l,
        "high_share":     n_h / n,
        "medium_share":   n_m / n,
        "low_share":      n_l / n,
    }


def grid_search(scores: np.ndarray, labels: np.ndarray) -> List[Dict]:
    """对所有 (t_low, t_high) 组合算指标。"""
    rows: List[Dict] = []
    for t_low in T_LOW_GRID:
        for t_high in T_HIGH_GRID:
            if t_high <= t_low:
                continue
            rows.append(compute_metrics_for_thresholds(scores, labels, t_low, t_high))
    return rows


def select_for_target(
    candidates: List[Dict], target: float,
    enforce_safety: bool = True,
) -> Optional[Dict]:
    """在所有候选里挑：high_p >= target AND low_p >= target，
    并满足硬底线，最后取 medium_share 最小的（覆盖率最高）。"""
    feasible = []
    for c in candidates:
        if enforce_safety:
            if c["mal_to_low"] > MAL_TO_LOW_MAX:
                continue
            if c["high_recall"] < HIGH_RECALL_MIN:
                continue
        if c["high_precision"] < target:
            continue
        if c["low_precision"] < target:
            continue
        feasible.append(c)
    if not feasible:
        return None
    # 关键多目标：min medium_share，然后 max(high_p + low_p) 作 tiebreaker
    feasible.sort(key=lambda x: (
        x["medium_share"],
        -(x["high_precision"] + x["low_precision"]),
    ))
    return feasible[0]


# ═════════════════════════════════════════════════════════════
#  Plotting
# ═════════════════════════════════════════════════════════════


def plot_precision_coverage(
    val_candidates: List[Dict],
    selected_by_target: Dict[float, Dict],
    out_path: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 只画硬底线满足的点
    safe = [c for c in val_candidates
            if c["mal_to_low"] <= MAL_TO_LOW_MAX
            and c["high_recall"] >= HIGH_RECALL_MIN]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── 左图：high_precision vs high_share ──
    ax = axes[0]
    xs = [c["high_share"] for c in safe]
    ys = [c["high_precision"] for c in safe]
    ax.scatter(xs, ys, s=6, alpha=0.25, color="#888")
    for target, sel in selected_by_target.items():
        if sel is None:
            continue
        ax.scatter([sel["high_share"]], [sel["high_precision"]],
                   s=80, marker="*", edgecolor="black", linewidth=0.7,
                   label=f"target={target:.2f}  (n_high={sel['n_high']})")
    ax.set_xlabel("High band coverage (fraction of test)")
    ax.set_ylabel("High precision")
    ax.set_title("High Band: Precision vs Coverage (val grid)")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(0, max(xs) * 1.05 if xs else 1)
    ax.axhline(0.9, ls="--", color="red", alpha=0.5, lw=0.8)
    ax.axhline(0.95, ls="--", color="darkred", alpha=0.5, lw=0.8)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    # ── 右图：low_precision vs low_share ──
    ax = axes[1]
    xs = [c["low_share"] for c in safe]
    ys = [c["low_precision"] for c in safe]
    ax.scatter(xs, ys, s=6, alpha=0.25, color="#888")
    for target, sel in selected_by_target.items():
        if sel is None:
            continue
        ax.scatter([sel["low_share"]], [sel["low_precision"]],
                   s=80, marker="*", edgecolor="black", linewidth=0.7,
                   label=f"target={target:.2f}  (n_low={sel['n_low']})")
    ax.set_xlabel("Low band coverage (fraction of test)")
    ax.set_ylabel("Low precision")
    ax.set_title("Low Band: Precision vs Coverage (val grid)")
    ax.set_ylim(0.4, 1.02)
    ax.set_xlim(0, max(xs) * 1.05 if xs else 1)
    ax.axhline(0.9, ls="--", color="red", alpha=0.5, lw=0.8)
    ax.axhline(0.95, ls="--", color="darkred", alpha=0.5, lw=0.8)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.2)

    fig.suptitle(
        "Precision-Coverage trade-off (硬底线: mal→low=0 + high_recall≥0.95)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrices(
    results_by_target: Dict[float, Dict],
    out_path: str,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = sorted(results_by_target.keys())
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.6))
    if n == 1:
        axes = [axes]

    band_names = ["high", "med", "low"]
    cls_names_short = ["mal", "ben", "nt"]

    for ax, target in zip(axes, targets):
        entry = results_by_target[target]
        if entry["selection"] is None:
            ax.text(0.5, 0.5, "不可达", ha="center", va="center", fontsize=14)
            ax.set_title(f"target={target:.2f}")
            ax.set_xticks([]); ax.set_yticks([])
            continue
        cm = np.array(entry["test_eval"]["confusion_3band"])
        im = ax.imshow(cm, cmap="Blues")
        for i in range(3):
            for j in range(3):
                v = cm[i, j]
                ax.text(j, i, f"{v}", ha="center", va="center",
                        color="white" if v > cm.max() / 2 else "black",
                        fontsize=10, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_xticklabels(band_names)
        ax.set_yticks(range(3)); ax.set_yticklabels(cls_names_short)
        sel = entry["selection"]
        ax.set_title(
            f"target={target:.2f}\n"
            f"t=({sel['t_low']:.3f}, {sel['t_high']:.2f})\n"
            f"test high_p={entry['test_metrics']['high_precision']:.3f} "
            f"low_p={entry['test_metrics']['low_precision']:.3f}\n"
            f"med_share={entry['test_metrics']['medium_share']:.2f}",
            fontsize=9,
        )
        ax.set_xlabel("predicted band")
        if ax is axes[0]:
            ax.set_ylabel("true class")

    fig.suptitle(
        "Test 集 3-band 混淆矩阵 by precision target",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ═════════════════════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════════════════════


def main():
    set_seed(SEED)
    print("=" * 72)
    print("0514 模型 precision 后处理阈值搜索")
    print(f"  权重: {BEST_WEIGHT}")
    print(f"  输出: {OUT_DIR}")
    print(f"  设备: {DEVICE}")
    print("=" * 72)

    # ── 推理 ──
    print("\n[1/5] 加载数据集 & 模型 ...")
    val_ds, test_ds = build_datasets()
    print(f"  val={len(val_ds)} test={len(test_ds)}")
    model = build_model()

    val_npz_path  = os.path.join(OUT_DIR, "inference_val.npz")
    test_npz_path = os.path.join(OUT_DIR, "inference_test.npz")

    print("\n[2/5] 跑推理 ...")
    val_pred  = run_inference(model, val_ds,  "val")
    test_pred = run_inference(model, test_ds, "test")
    np.savez(val_npz_path,  **val_pred)
    np.savez(test_npz_path, **test_pred)
    print(f"  保存 → {val_npz_path}")
    print(f"  保存 → {test_npz_path}")

    # ── 网格搜索 ──
    print("\n[3/5] 在 val 上做 2D 网格搜索 ...")
    t0 = time.time()
    val_candidates = grid_search(val_pred["score"], val_pred["label"])
    safe_pool = [c for c in val_candidates
                 if c["mal_to_low"] <= MAL_TO_LOW_MAX
                 and c["high_recall"] >= HIGH_RECALL_MIN]
    print(f"  {len(val_candidates)} 个候选, "
          f"{len(safe_pool)} 个满足硬底线 ({time.time()-t0:.1f}s)")

    if len(safe_pool) == 0:
        print("❌ 没有任何 (t_low, t_high) 满足硬底线！请检查模型/数据。")
        return

    # ── 多目标筛选 ──
    print("\n[4/5] 选阈值 + 在 test 上评估 ...")
    results_by_target: Dict[float, Dict] = {}

    for target in PRECISION_TARGETS:
        sel = select_for_target(val_candidates, target, enforce_safety=True)
        if sel is None:
            print(f"  target={target:.2f}: ❌ val 上不可达")
            results_by_target[target] = {
                "selection": None,
                "val_metrics": None,
                "test_metrics": None,
                "test_eval": None,
                "diag_search_on_test": None,
            }
            continue

        # 应用到 test
        test_metrics = compute_metrics_for_thresholds(
            test_pred["score"], test_pred["label"],
            sel["t_low"], sel["t_high"],
        )
        # 用 evaluate_risk_bands 得到完整 confusion / distribution / calibration
        test_eval = evaluate_risk_bands(
            test_pred["score"], test_pred["label"],
            sel["t_low"], sel["t_high"],
            phase=f"Test (target={target:.2f})",
        )

        # 诊断：若直接在 test 上对同一 target 搜
        diag = select_for_target(
            grid_search(test_pred["score"], test_pred["label"]),
            target, enforce_safety=True,
        )

        results_by_target[target] = {
            "selection":  sel,
            "val_metrics":  sel,
            "test_metrics": test_metrics,
            "test_eval":    test_eval,
            "diag_search_on_test": diag,
        }
        print(
            f"  target={target:.2f}: "
            f"val (t_l={sel['t_low']:.3f}, t_h={sel['t_high']:.2f}) "
            f"val_h_p={sel['high_precision']:.3f} val_l_p={sel['low_precision']:.3f} "
            f"val_med={sel['medium_share']:.2f}  →  "
            f"test_h_p={test_metrics['high_precision']:.3f} "
            f"test_l_p={test_metrics['low_precision']:.3f} "
            f"test_med={test_metrics['medium_share']:.2f}"
        )

    # ── 绘图 + 保存 ──
    print("\n[5/5] 生成产物 ...")
    selected_by_target = {t: r["selection"] for t, r in results_by_target.items()}

    curve_path = os.path.join(OUT_DIR, "precision_coverage_curve.png")
    plot_precision_coverage(val_candidates, selected_by_target, curve_path)
    print(f"  → {curve_path}")

    cm_path = os.path.join(OUT_DIR, "confusion_matrices_by_target.png")
    plot_confusion_matrices(results_by_target, cm_path)
    print(f"  → {cm_path}")

    # ── 摘要表 ──
    log_lines = [
        "=" * 100,
        "0514 模型 — High/Low precision 后处理阈值搜索结果",
        "=" * 100,
        f"模型权重: {BEST_WEIGHT}",
        f"硬底线: mal_to_low == 0  AND  high_recall >= {HIGH_RECALL_MIN}",
        f"val={len(val_pred['label'])} test={len(test_pred['label'])}",
        "",
        "选阈值来源: val   阈值挑选规则: high_p>=t AND low_p>=t, 取 medium_share 最小者",
        "",
        f"{'target':>8} | {'t_low':>6} {'t_high':>7} | "
        f"{'val_h_p':>8} {'val_l_p':>8} {'val_med':>8} | "
        f"{'test_h_p':>9} {'test_l_p':>9} {'test_med':>9} | "
        f"{'n_high':>7} {'n_low':>7} {'mal→low':>8} {'high_R':>7}",
        "-" * 130,
    ]
    for target in PRECISION_TARGETS:
        entry = results_by_target[target]
        if entry["selection"] is None:
            log_lines.append(
                f"{target:>8.2f} | val 上不可达"
            )
            continue
        sel = entry["selection"]
        t = entry["test_metrics"]
        log_lines.append(
            f"{target:>8.2f} | {sel['t_low']:>6.3f} {sel['t_high']:>7.3f} | "
            f"{sel['high_precision']:>8.4f} {sel['low_precision']:>8.4f} "
            f"{sel['medium_share']:>8.4f} | "
            f"{t['high_precision']:>9.4f} {t['low_precision']:>9.4f} "
            f"{t['medium_share']:>9.4f} | "
            f"{t['n_high']:>7d} {t['n_low']:>7d} "
            f"{t['mal_to_low']:>8d} {t['high_recall']:>7.4f}"
        )

    # 诊断对比
    log_lines += ["", "─" * 100,
                  "诊断 — 若直接在 test 上搜阈值（无防泛化）:",
                  ""]
    log_lines.append(
        f"{'target':>8} | {'val→test':>30} | {'test 直接搜':>30} | gap"
    )
    log_lines.append("-" * 100)
    for target in PRECISION_TARGETS:
        entry = results_by_target[target]
        if entry["selection"] is None:
            continue
        sel = entry["selection"]
        t = entry["test_metrics"]
        diag = entry["diag_search_on_test"]
        if diag is None:
            log_lines.append(
                f"{target:>8.2f} | "
                f"h_p={t['high_precision']:.3f} l_p={t['low_precision']:.3f} med={t['medium_share']:.2f} | "
                f"test 上同样不可达"
            )
        else:
            log_lines.append(
                f"{target:>8.2f} | "
                f"h_p={t['high_precision']:.3f} l_p={t['low_precision']:.3f} med={t['medium_share']:.2f} | "
                f"h_p={diag['high_precision']:.3f} l_p={diag['low_precision']:.3f} med={diag['medium_share']:.2f} | "
                f"med Δ={t['medium_share']-diag['medium_share']:+.3f}"
            )

    log_lines += ["", "=" * 100]
    log_text = "\n".join(log_lines)
    log_path = os.path.join(OUT_DIR, "search_log.txt")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(log_text)
    print(f"  → {log_path}")
    print()
    print(log_text)

    # ── 完整 JSON ──
    out_json = {
        "trained_dir": TRAINED_DIR,
        "best_weight": BEST_WEIGHT,
        "hard_floor":  {"mal_to_low_max": MAL_TO_LOW_MAX,
                         "high_recall_min": HIGH_RECALL_MIN},
        "val_n":  int(len(val_pred["label"])),
        "test_n": int(len(test_pred["label"])),
        "grid": {"t_low": T_LOW_GRID.tolist(),
                  "t_high": T_HIGH_GRID.tolist()},
        "precision_targets": PRECISION_TARGETS,
        "by_target": {
            str(target): {
                "selection":         results_by_target[target]["selection"],
                "val_metrics":       results_by_target[target]["val_metrics"],
                "test_metrics":      results_by_target[target]["test_metrics"],
                "test_confusion_3band":
                    results_by_target[target]["test_eval"]["confusion_3band"]
                    if results_by_target[target]["test_eval"] is not None else None,
                "diag_search_on_test": results_by_target[target]["diag_search_on_test"],
            } for target in PRECISION_TARGETS
        },
    }
    json_path = os.path.join(OUT_DIR, "precision_search_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)
    print(f"  → {json_path}")

    print("\n✅ 完成。")


if __name__ == "__main__":
    main()
