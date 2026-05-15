"""
compare_all.py — 汇总 7 个比例消融实验的 test 指标。

输入: 0514/logs/ratio_ablation/{exp_label}/eval_test.json
      + baseline (S2): 0514/logs/20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1/eval_test.json

输出: 0514/logs/ratio_ablation/_summary/
      - results_table.csv
      - trends.png
      - confusion_matrices_grid.png
      - ratio_ablation_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["axes.unicode_minus"] = False

ROOT = Path("/data1/ouyangxinglong/GBP-Cascade")
LOG_RA = ROOT / "0514" / "logs" / "ratio_ablation"
SUMMARY = LOG_RA / "_summary"
BASELINE_LOG = ROOT / "0514" / "logs" / "20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1"

# 实验注册表（顺序决定行/列展示顺序）
EXPERIMENTS = [
    # Sampler 轴 (A组) — 固定数据集 = 全量 train_split
    {"label": "S1_1_1_1",   "group": "Sampler", "ratio_str": "1:1:1",      "nt_exposure": 1.00, "is_baseline": False},
    {"label": "S2_3_3_2",   "group": "Sampler", "ratio_str": "3:3:2",      "nt_exposure": 0.67, "is_baseline": True},
    {"label": "S3_1_2_6",   "group": "Sampler", "ratio_str": "1:2:6",      "nt_exposure": 6.00, "is_baseline": False},
    {"label": "S4_natural", "group": "Sampler", "ratio_str": "1:1.9:6.65", "nt_exposure": 6.65, "is_baseline": False},
    # Dataset 轴 (B组) — 固定 sampler = RandomSampler
    {"label": "D1_1_1_1",   "group": "Dataset", "ratio_str": "1:1:1",      "n_train": 728,  "nt_pct": 33.2},
    {"label": "D2_1_1_2",   "group": "Dataset", "ratio_str": "1:1:2",      "n_train": 973,  "nt_pct": 50.1},
    {"label": "D3_1_1.9_3", "group": "Dataset", "ratio_str": "1:1.9:3",    "n_train": 1429, "nt_pct": 50.9},
    # D4 ≡ S4_natural (同一实验，不重复行；仅在 dataset 轴 plot 时用 S4 数据)
]


def load_eval(exp_label: str) -> dict:
    """加载某个实验的 eval_test.json。S2 直接读 baseline。"""
    if exp_label == "S2_3_3_2":
        path = BASELINE_LOG / "eval_test.json"
    else:
        path = LOG_RA / exp_label / "eval_test.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(d: dict) -> dict:
    """从 eval_test.json 抽出关键指标。"""
    if d is None:
        return {k: None for k in ["high_p", "low_p", "mal_to_low", "high_recall",
                                    "medium_share", "binary_auc", "best_epoch",
                                    "t_low", "t_high", "high_n", "low_n", "med_n"]}
    safety = d.get("safety", {})
    per = d.get("per_band", {})
    thr = d.get("thresholds", {})
    return {
        "high_p": safety.get("high_precision"),
        "low_p":  per.get("low", {}).get("precision"),
        "mal_to_low": safety.get("mal_to_low"),
        "high_recall": safety.get("high_recall"),
        "medium_share": safety.get("medium_share"),
        "binary_auc": safety.get("binary_roc_auc"),
        "best_epoch": d.get("best_epoch"),
        "t_low": thr.get("t_low"),
        "t_high": thr.get("t_high"),
        "high_n": per.get("high", {}).get("n_pred"),
        "low_n":  per.get("low",  {}).get("n_pred"),
        "med_n":  per.get("medium", {}).get("n_pred"),
        "confusion_3band": d.get("confusion_3band"),
    }


def build_results_table():
    """构建 results table DataFrame。"""
    rows = []
    for exp in EXPERIMENTS:
        d = load_eval(exp["label"])
        m = extract_metrics(d)
        row = {**exp, **m, "found": d is not None}
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


def save_csv(df: pd.DataFrame):
    SUMMARY.mkdir(parents=True, exist_ok=True)
    cols = ["label", "group", "ratio_str", "found",
            "best_epoch", "binary_auc",
            "t_low", "t_high",
            "high_n", "high_p", "low_n", "low_p",
            "med_n", "medium_share",
            "mal_to_low", "high_recall"]
    df_out = df[cols].copy()
    df_out.to_csv(SUMMARY / "results_table.csv", index=False)
    print(f"  CSV: {SUMMARY / 'results_table.csv'}")


def plot_trends(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # ── 左: Sampler 轴 ──
    sdf = df[df["group"] == "Sampler"].sort_values("nt_exposure")
    ax = axes[0]
    x = sdf["nt_exposure"].values
    labels = sdf["ratio_str"].values
    for col, color, marker in [("high_p", "tab:red", "o"),
                                ("low_p", "tab:blue", "s"),
                                ("high_recall", "tab:orange", "^"),
                                ("binary_auc", "tab:green", "v")]:
        y = sdf[col].astype(float).values
        ax.plot(x, y, marker=marker, color=color, label=col, lw=2, ms=8)
    ax.set_xscale("log")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Sampler ratio (NT exposure relative to mal)")
    ax.set_ylabel("Metric value")
    ax.set_title("A group: Sampler ablation (dataset fixed)")
    ax.set_ylim(0.5, 1.02); ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── 右: Dataset 轴 ──
    # D 轴包含 D1/D2/D3 + S4 (=D4 natural)
    ddf = df[df["group"] == "Dataset"].copy()
    s4 = df[df["label"] == "S4_natural"].copy()
    if len(s4):
        s4 = s4.iloc[0]
        ddf = pd.concat([ddf, pd.DataFrame([{
            **s4.to_dict(),
            "label": "D4=S4_natural", "ratio_str": "1:1.9:6.65",
            "n_train": 2311, "nt_pct": 69.7, "group": "Dataset",
        }])], ignore_index=True)
    ddf = ddf.sort_values("n_train")
    ax = axes[1]
    x = ddf["n_train"].astype(int).values
    labels = ddf["ratio_str"].values
    for col, color, marker in [("high_p", "tab:red", "o"),
                                ("low_p", "tab:blue", "s"),
                                ("high_recall", "tab:orange", "^"),
                                ("binary_auc", "tab:green", "v")]:
        y = ddf[col].astype(float).values
        ax.plot(x, y, marker=marker, color=color, label=col, lw=2, ms=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=10)
    ax.set_xlabel("Physical training set ratio (train size)")
    ax.set_ylabel("Metric value")
    ax.set_title("B group: Dataset ablation (sampler=natural)")
    ax.set_ylim(0.5, 1.02); ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = SUMMARY / "trends.png"
    plt.savefig(out, dpi=140, bbox_inches="tight"); plt.close()
    print(f"  trends: {out}")


def plot_cm_grid(df: pd.DataFrame):
    """7 张混淆矩阵 2x4 网格（最后 1 格留空）。"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes_flat = axes.flatten()
    class_names = ["Mal", "Ben", "NT"]
    band_names  = ["High", "Med", "Low"]

    for i, row in df.iterrows():
        ax = axes_flat[i]
        cm_obj = row["confusion_3band"]
        if cm_obj is None:
            ax.set_visible(False); continue
        cm = np.array(cm_obj)
        # cm 是 3x3，行=真实类，列=预测 band (High=0, Med=1, Low=2)
        n_band = cm.sum(axis=0)
        col_pct = cm / np.maximum(n_band[None, :], 1) * 100
        im = ax.imshow(col_pct, cmap="Blues", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(band_names, fontsize=9)
        ax.set_yticks(range(3)); ax.set_yticklabels(class_names, fontsize=9)
        for a in range(3):
            for b in range(3):
                c = cm[a, b]; p = col_pct[a, b]
                color = "white" if p > 55 else "black"
                ax.text(b, a, f"{c}\n({p:.0f}%)", ha="center", va="center",
                        color=color, fontsize=8)
        title = (f"{row['label']} ({row['group']})\n"
                 f"high_p={row['high_p']:.3f} low_p={row['low_p']:.3f} "
                 f"M→low={int(row['mal_to_low'])}")
        ax.set_title(title, fontsize=9)

    # 隐藏多余的格子
    for j in range(len(df), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    out = SUMMARY / "confusion_matrices_grid.png"
    plt.savefig(out, dpi=130, bbox_inches="tight"); plt.close()
    print(f"  CM grid: {out}")


def write_markdown_report(df: pd.DataFrame):
    """自动生成 ratio_ablation_summary.md。"""
    lines = []
    lines.append("# 比例消融实验总结 (Ratio Ablation, 7 experiments)")
    lines.append("")
    lines.append("> 生成日期: 2026-05-14")
    lines.append("> 模型: SwinV2-Tiny ordinal trimodal (与 0514 baseline 同架构)")
    lines.append("> 测试集: 0514dataset_flat/task_3class_test.xlsx (1167 张)")
    lines.append("")
    lines.append("## 1. 实验设置")
    lines.append("")
    lines.append("### A组 Sampler 消融（固定数据集 = baseline train_split, 2311 张）")
    lines.append("| 实验 | sampler 配置 | 模型 batch 见到 |")
    lines.append("|---|---|---|")
    sdf = df[df["group"] == "Sampler"]
    for _, r in sdf.iterrows():
        lines.append(f"| {r['label']} | — | {r['ratio_str']} |")
    lines.append("")
    lines.append("### B组 Dataset 消融（固定 sampler = RandomSampler）")
    lines.append("| 实验 | 物理比例 | 训练量 |")
    lines.append("|---|---|---|")
    ddf = df[df["group"] == "Dataset"]
    for _, r in ddf.iterrows():
        lines.append(f"| {r['label']} | {r['ratio_str']} | {r.get('n_train','—')} |")
    lines.append("")
    lines.append("> 注: D4 ≡ S4_natural (同一训练任务，物理=自然分布且 sampler=natural)")
    lines.append("")

    # 主结果表
    lines.append("## 2. 主结果对比（Test n=1167）")
    lines.append("")
    lines.append("| 实验 | 组 | 比例 | best_ep | binary_AUC | high_n | high_p | low_n | low_p | medium% | mal→low | high_R |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in df.iterrows():
        if not r["found"]:
            lines.append(f"| {r['label']} | {r['group']} | {r['ratio_str']} | — | (missing) | | | | | | | |")
            continue
        be = int(r['best_epoch']) if pd.notna(r['best_epoch']) else '—'
        lines.append(
            f"| {r['label']} | {r['group']} | {r['ratio_str']} | "
            f"{be} | "
            f"{r['binary_auc']:.4f} | "
            f"{int(r['high_n'])} | {r['high_p']:.4f} | "
            f"{int(r['low_n'])} | {r['low_p']:.4f} | "
            f"{r['medium_share']*100:.1f}% | "
            f"{int(r['mal_to_low'])} | {r['high_recall']:.4f} |"
        )
    lines.append("")

    # 简单结论
    valid = df[df["found"]]
    if len(valid):
        best_high = valid.loc[valid["high_p"].idxmax()]
        best_low  = valid.loc[valid["low_p"].idxmax()]
        best_auc  = valid.loc[valid["binary_auc"].idxmax()]
        safest    = valid[valid["mal_to_low"] == 0].sort_values("high_recall", ascending=False)
        lines.append("## 3. 关键结论")
        lines.append("")
        lines.append(f"- **high_p 最高**：`{best_high['label']}` ({best_high['ratio_str']}) → {best_high['high_p']:.4f}")
        lines.append(f"- **low_p 最高**：`{best_low['label']}` ({best_low['ratio_str']}) → {best_low['low_p']:.4f}")
        lines.append(f"- **binary_AUC 最高**：`{best_auc['label']}` ({best_auc['ratio_str']}) → {best_auc['binary_auc']:.4f}")
        if len(safest):
            top = safest.iloc[0]
            lines.append(f"- **最安全**（mal→low=0 且 high_R 最高）：`{top['label']}` → high_R={top['high_recall']:.4f}")
        lines.append("")

    lines.append("## 4. 产物")
    lines.append("")
    lines.append("- `_summary/results_table.csv` — 完整指标 CSV")
    lines.append("- `_summary/trends.png` — A/B 两组趋势图")
    lines.append("- `_summary/confusion_matrices_grid.png` — 7 张混淆矩阵")
    lines.append(f"- 每个实验日志: `0514/logs/ratio_ablation/{{exp_label}}/`")
    lines.append("")

    out = SUMMARY / "ratio_ablation_summary.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"  Markdown: {out}")


def main():
    print("=" * 70)
    print("Aggregating ratio ablation results")
    print("=" * 70)
    SUMMARY.mkdir(parents=True, exist_ok=True)

    df = build_results_table()

    # 打印控制台
    print(f"\n{'label':<14} {'group':<8} {'ratio':<12} {'found':<6} "
          f"{'high_p':<8} {'low_p':<8} {'mal→low':<8} {'high_R':<8}")
    print("-" * 80)
    for _, r in df.iterrows():
        hp = f"{r['high_p']:.4f}" if pd.notna(r['high_p']) else "—"
        lp = f"{r['low_p']:.4f}"  if pd.notna(r['low_p'])  else "—"
        ml = f"{int(r['mal_to_low'])}" if pd.notna(r['mal_to_low']) else "—"
        hr = f"{r['high_recall']:.4f}" if pd.notna(r['high_recall']) else "—"
        print(f"{r['label']:<14} {r['group']:<8} {r['ratio_str']:<12} {str(r['found']):<6} "
              f"{hp:<8} {lp:<8} {ml:<8} {hr:<8}")
    print()

    save_csv(df)

    # 仅 found 行参与画图
    valid_df = df[df["found"]].reset_index(drop=True)
    if len(valid_df) >= 2:
        plot_trends(valid_df)
        plot_cm_grid(valid_df)
    else:
        print("  ⚠ 有效实验少于 2 个，跳过画图")

    write_markdown_report(df)
    print("\n✅ 完成")


if __name__ == "__main__":
    main()
