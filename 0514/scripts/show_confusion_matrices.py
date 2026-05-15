"""
show_confusion_matrices.py

直接展示 0514 模型在 90%/95% precision 目标下的混淆矩阵。
同时探索：在保 base 尽可能大的前提下，能否同时达到 90/95% 双精度。

输入：inference_val.npz / inference_test.npz
输出：cm_target_90_95.png + 终端打印
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---- 中文字体（已安装的备选）----
for fam in ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "SimHei", "DejaVu Sans"]:
    mpl.rcParams["font.sans-serif"] = [fam]
    break
mpl.rcParams["axes.unicode_minus"] = False

LOG_DIR = Path("/data1/ouyangxinglong/GBP-Cascade/0514/logs/precision_search")
CLASS_NAMES = ["Malignant", "Benign", "No_Tumor"]
BAND_NAMES  = ["High", "Medium", "Low"]


def decide_bands(scores: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """0=high, 1=medium, 2=low"""
    b = np.full_like(scores, 1, dtype=np.int64)
    b[scores >= t_high] = 0
    b[scores <= t_low]  = 2
    return b


def metrics(labels: np.ndarray, bands: np.ndarray) -> dict:
    cm = np.zeros((3, 3), dtype=np.int64)  # rows: true class, cols: band
    for y, b in zip(labels, bands):
        cm[y, b] += 1
    n = len(labels)
    n_high = int((bands == 0).sum())
    n_med  = int((bands == 1).sum())
    n_low  = int((bands == 2).sum())
    # high_precision = P(y==malignant | band==high)
    high_p = cm[0, 0] / max(n_high, 1)
    # low_precision = P(y==no_tumor | band==low)
    low_p  = cm[2, 2] / max(n_low, 1)
    mal_to_low = int(cm[0, 2])
    high_recall = cm[0, 0] / max(cm[0, :].sum(), 1)
    return dict(
        cm=cm, n_high=n_high, n_med=n_med, n_low=n_low,
        high_p=high_p, low_p=low_p,
        mal_to_low=mal_to_low, high_recall=high_recall,
        coverage=(n_high + n_low) / n,
    )


def grid_search(val_scores, val_labels, *,
                target_high_p, target_low_p,
                hard_safety=True):
    """在 val 上找满足精度约束 + 安全底线下 medium_share 最小的 (t_low, t_high)."""
    best = None
    t_lows  = np.arange(0.005, 0.451, 0.005)
    t_highs = np.arange(0.500, 0.991, 0.010)
    for t_low in t_lows:
        for t_high in t_highs:
            if t_low >= t_high: continue
            b = decide_bands(val_scores, t_low, t_high)
            m = metrics(val_labels, b)
            if m["n_high"] == 0 or m["n_low"] == 0: continue
            if hard_safety and (m["mal_to_low"] > 0 or m["high_recall"] < 0.95):
                continue
            if target_high_p is not None and m["high_p"] < target_high_p: continue
            if target_low_p  is not None and m["low_p"]  < target_low_p:  continue
            cand = (m["n_med"], -m["coverage"], t_low, t_high, m)
            if best is None or cand < best:
                best = cand
    return best


def plot_cm(ax, cm, title, n_high, n_med, n_low):
    n_total = cm.sum()
    band_totals = [n_high, n_med, n_low]
    # 用百分比上色 (列归一化: 该 band 内各真实类占比)
    col_pct = cm / np.maximum(np.array(band_totals)[None, :], 1) * 100
    im = ax.imshow(col_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(BAND_NAMES, fontsize=11)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted band", fontsize=11)
    ax.set_ylabel("True class", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(3):
        for j in range(3):
            c = cm[i, j]
            p = col_pct[i, j]
            color = "white" if p > 55 else "black"
            ax.text(j, i, f"{c}\n({p:.1f}%)", ha="center", va="center",
                    color=color, fontsize=10)
    # column band counts
    for j, t in enumerate(band_totals):
        ax.text(j, 3.05, f"n={t}", ha="center", va="top", fontsize=9, color="navy")


def render_target(ax_test, ax_val, label, t_low, t_high,
                  val_scores, val_labels, test_scores, test_labels):
    mv = metrics(val_labels,  decide_bands(val_scores,  t_low, t_high))
    mt = metrics(test_labels, decide_bands(test_scores, t_low, t_high))
    plot_cm(ax_val, mv["cm"],
            f"{label}\nVAL  (n={len(val_labels)})  t_low={t_low:.3f}  t_high={t_high:.2f}\n"
            f"high_p={mv['high_p']:.3f}  low_p={mv['low_p']:.3f}  "
            f"mal→low={mv['mal_to_low']}  high_R={mv['high_recall']:.3f}",
            mv["n_high"], mv["n_med"], mv["n_low"])
    plot_cm(ax_test, mt["cm"],
            f"{label}\nTEST (n={len(test_labels)})  t_low={t_low:.3f}  t_high={t_high:.2f}\n"
            f"high_p={mt['high_p']:.3f}  low_p={mt['low_p']:.3f}  "
            f"mal→low={mt['mal_to_low']}  high_R={mt['high_recall']:.3f}",
            mt["n_high"], mt["n_med"], mt["n_low"])
    return mv, mt


def fmt_metrics(tag, m, n_total):
    return (f"  {tag:30s} | high {m['n_high']:>4d} (p={m['high_p']:.3f}) | "
            f"low {m['n_low']:>4d} (p={m['low_p']:.3f}) | "
            f"med {m['n_med']:>4d} ({m['n_med']/n_total*100:5.1f}%) | "
            f"mal→low={m['mal_to_low']} | high_R={m['high_recall']:.3f} | "
            f"base={m['n_high']+m['n_low']:>4d} ({m['coverage']*100:5.1f}%)")


def main():
    val  = np.load(LOG_DIR / "inference_val.npz")
    test = np.load(LOG_DIR / "inference_test.npz")
    vs, vl = val["score"], val["label"]
    ts, tl = test["score"], test["label"]
    nt = len(tl)

    print("=" * 110)
    print(f"VAL  n={len(vl)}   |   TEST n={nt}")
    print("=" * 110)

    # ─── 方案 A: 双约束 (high_p≥t AND low_p≥t)，medium 最小化 ───
    print("\n[方案 A] 双约束：high_p ≥ target AND low_p ≥ target，medium 最小化\n")
    plans_A = []
    for tgt in [0.90, 0.95]:
        r = grid_search(vs, vl, target_high_p=tgt, target_low_p=tgt)
        if r:
            _, _, t_low, t_high, _ = r
            plans_A.append((f"A·{int(tgt*100)}% (both)", t_low, t_high))
            print(f"  target={tgt}: t_low={t_low:.3f}  t_high={t_high:.2f}")
        else:
            print(f"  target={tgt}: 不可达")

    # ─── 方案 B: 只锁 high_p；t_low 选让 low_p 也 ≥ target 的最宽 ───
    print("\n[方案 B] 只锁 high_p ≥ target；low band 尽量大（low_p 不强求达 target）\n")
    plans_B = []
    for tgt in [0.90, 0.95]:
        # 固定 t_high 让 val 上 high_p ≥ tgt; t_low 设到 val 上 mal→low 仍 = 0 的最高值
        # 即扫所有 (t_low, t_high)，约束 high_p≥tgt, mal→low=0, high_R≥0.95，挑 n_high+n_low 最大
        best = None
        for t_low in np.arange(0.005, 0.451, 0.005):
            for t_high in np.arange(0.500, 0.991, 0.010):
                if t_low >= t_high: continue
                m = metrics(vl, decide_bands(vs, t_low, t_high))
                if m["n_high"] == 0 or m["n_low"] == 0: continue
                if m["mal_to_low"] > 0 or m["high_recall"] < 0.95: continue
                if m["high_p"] < tgt: continue
                key = (-m["coverage"], m["n_med"], t_low, t_high, m)
                if best is None or key < best:
                    best = key
        if best:
            _, _, t_low, t_high, _ = best
            plans_B.append((f"B·high_p≥{int(tgt*100)}%", t_low, t_high))
            print(f"  target_high_p={tgt}: t_low={t_low:.3f}  t_high={t_high:.2f}")

    # ─── 把所有方案画大图 ───
    plans = plans_A + plans_B
    fig, axes = plt.subplots(len(plans), 2, figsize=(13, 4.5 * len(plans)))
    if len(plans) == 1: axes = axes[None, :]
    fig.suptitle("0514 模型 — High/Low precision 各方案的混淆矩阵（左 VAL · 右 TEST）",
                 fontsize=13, fontweight="bold", y=1.0)

    print("\n" + "=" * 110)
    print("结果汇总（test）")
    print("=" * 110)
    for i, (label, t_low, t_high) in enumerate(plans):
        mv, mt = render_target(axes[i, 1], axes[i, 0], label, t_low, t_high,
                               vs, vl, ts, tl)
        print(fmt_metrics(label + f" (t_l={t_low:.3f}, t_h={t_high:.2f})", mt, nt))

    plt.tight_layout()
    out = LOG_DIR / "cm_target_90_95.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"\n图已保存: {out}")


if __name__ == "__main__":
    main()
