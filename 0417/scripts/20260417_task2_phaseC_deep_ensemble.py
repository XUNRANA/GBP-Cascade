"""
Phase C: 多种子深度集成（6 模型 rank-ensemble）

输入: 6 个 calibrated test_probs.csv（Phase B 跑完后路径自动探测）
      + 对应的 dev_probs.csv（合并 dev 做阈值选择）

操作:
  1. Simple Ensemble (平均 prob_benign)
  2. Rank Ensemble   (平均排名)
  3. 合并 dev 上选阈值 (find_constrained_threshold)
  4. test 上评估 + reliability

输出: 0417/logs/phaseC/
  - phaseC_test_probs_simple.csv
  - phaseC_test_probs_rank.csv
  - phaseC_summary.md
  - phaseC_reliability_simple/rank.png/.csv
"""

import os
import sys
import glob

import numpy as np
import pandas as pd

_ROOT = "/data1/ouyangxinglong/GBP-Cascade"
sys.path.insert(0, os.path.join(_ROOT, "0408", "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "0417", "scripts"))

from seg_cls_utils_v5 import (
    find_constrained_threshold_text,
    compute_binary_reliability_stats,
    save_reliability_stats_csv,
    save_reliability_diagram,
)
from seg_cls_utils_v5_ext import (
    simple_ensemble, rank_ensemble, compute_pauc_at_fpr,
)

OUT_DIR = os.path.join(_ROOT, "0417", "logs", "phaseC")
LOG_BASE = os.path.join(_ROOT, "0417", "logs")

MISS_RATES = [0.02, 0.05, 0.10]
CALIBRATION_BINS = 10


def auto_detect_probs(log_base, exp_suffix, dev_or_test="test_probs_calibrated"):
    """自动探测 exp 目录下的 probs csv，返回路径（若不存在返回 None）。"""
    dirs = sorted(glob.glob(os.path.join(log_base, f"*{exp_suffix}*")))
    for d in dirs:
        # 找到后缀匹配的 csv
        pattern = os.path.join(d, f"*_{dev_or_test}.csv")
        files = glob.glob(pattern)
        if files:
            return files[0]
    return None


def load_probs_file(path):
    """加载 probs csv，返回 (prob_benign (N,), labels (N,))。"""
    df = pd.read_csv(path)
    # 兼容不同列名
    if "prob_benign" in df.columns:
        pb = df["prob_benign"].values.astype(np.float32)
    elif "prob_benign_mean" in df.columns:
        pb = df["prob_benign_mean"].values.astype(np.float32)
    else:
        raise ValueError(f"找不到 prob_benign 列: {path}")
    labels = df["label"].values.astype(int)
    return pb, labels


def evaluate_ensemble(name, probs_b, labels, out_dir):
    """评估一个集成方法，保存 reliability，返回结果行列表。"""
    pauc = compute_pauc_at_fpr(labels, probs_b, max_fpr=0.05)
    result_rows = []
    for mr in MISS_RATES:
        pol = find_constrained_threshold_text(probs_b, labels, max_benign_miss_rate=mr)
        result_rows.append({
            "method": name,
            "miss_rate": mr,
            "threshold": pol["threshold"],
            "benign_recall": pol["benign_recall"],
            "no_tumor_hits": pol["selected_no_tumor"],
            "no_tumor_recall": pol["no_tumor_recall"],
            "no_tumor_precision": pol["no_tumor_precision"],
            "macro_f1": pol["macro_f1"],
            "pauc_fpr005": pauc,
            "constraint_satisfied": pol["constraint_satisfied"],
        })

    y_nt = (labels == 1).astype(int)
    rel_rows, ece, brier = compute_binary_reliability_stats(y_nt, 1.0 - probs_b, CALIBRATION_BINS)
    for r in result_rows:
        r["ece"] = ece
        r["brier"] = brier

    save_reliability_stats_csv(rel_rows, os.path.join(out_dir, f"phaseC_{name}_reliability.csv"))
    save_reliability_diagram(rel_rows, os.path.join(out_dir, f"phaseC_{name}_reliability.png"),
                              title=f"Phase C [{name}]")
    return result_rows, ece, pauc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 自动探测 probs 文件 ──────────────────────────────────────
    # Test probs（6 模型）
    model_specs = [
        ("19_3_seed42",    "19_3",                "test_probs_calibrated"),
        ("19_3_seed1337",  "19_3_seed1337",        "test_probs_calibrated"),
        ("19_3_seed2024",  "19_3_seed2024",        "test_probs_calibrated"),
        ("ctrl_seed42",    "19_control_seed42",    "test_probs_calibrated"),
        ("ctrl_seed1337",  "19_control_seed1337",  "test_probs_calibrated"),
        ("ctrl_seed2024",  "19_control_seed2024",  "test_probs_calibrated"),
    ]

    # Dev probs（用于合并 dev 选阈值）
    dev_specs = [
        ("19_3_seed42",   "19_3",              "dev_probs"),
        ("19_3_seed1337", "19_3_seed1337",     "dev_probs"),
        ("19_3_seed2024", "19_3_seed2024",     "dev_probs"),
    ]

    test_prob_list = []
    test_labels = None
    print("=== 加载 Test probs ===")
    for alias, suffix, ptype in model_specs:
        path = auto_detect_probs(LOG_BASE, suffix, ptype)
        if path is None:
            # 兜底：尝试 uncal
            path = auto_detect_probs(LOG_BASE, suffix, "test_probs_uncal")
        if path is None:
            # 尝试老格式（无 _best/_swa 前缀）
            path = auto_detect_probs(LOG_BASE, suffix, "test_probs")
        if path is None:
            print(f"  [{alias}] ⚠ 找不到 probs，跳过")
            continue
        pb, labels = load_probs_file(path)
        test_prob_list.append(pb)
        if test_labels is None:
            test_labels = labels
        else:
            assert np.all(test_labels == labels), f"[{alias}] 标签不一致！"
        print(f"  [{alias}] 加载: {path} ({len(pb)} 样本)")

    if len(test_prob_list) < 2:
        print("❌ 有效模型数 < 2，无法集成。请先完成 Phase B 训练。")
        return

    print(f"\n共 {len(test_prob_list)} 个模型参与集成")

    # ── Dev probs（合并，用于阈值选择）────────────────────────────
    dev_prob_list = []
    dev_labels = None
    print("\n=== 加载 Dev probs ===")
    for alias, suffix, ptype in dev_specs:
        path = auto_detect_probs(LOG_BASE, suffix, ptype)
        if path is None:
            path = auto_detect_probs(LOG_BASE, suffix, "best_dev_probs")
        if path is None:
            continue
        pb_dev, lbl_dev = load_probs_file(path)
        dev_prob_list.append(pb_dev)
        if dev_labels is None:
            dev_labels = lbl_dev
        print(f"  [{alias}] 加载 dev: {path}")

    has_dev = len(dev_prob_list) > 0 and dev_labels is not None

    # ── 集成 ──────────────────────────────────────────────────────
    pb_simple = simple_ensemble(test_prob_list)
    pb_rank = rank_ensemble(test_prob_list)

    # 保存集成 probs
    def save_ensemble(name, pb):
        path = os.path.join(OUT_DIR, f"phaseC_test_probs_{name}.csv")
        pd.DataFrame({
            "label": test_labels.astype(int),
            "prob_benign": pb.astype(np.float32),
            "prob_no_tumor": (1.0 - pb).astype(np.float32),
        }).to_csv(path, index=False, encoding="utf-8")
        return path

    simple_csv = save_ensemble("simple", pb_simple)
    rank_csv = save_ensemble("rank", pb_rank)
    print(f"\n集成 probs 已保存: {simple_csv}, {rank_csv}")

    # ── 评估（test 上，阈值来源：dev 或直接 test）─────────────────
    all_results = []
    for ens_name, pb_test in [("simple", pb_simple), ("rank", pb_rank)]:
        # 用 dev 集成 probs 选阈值（若可用）
        if has_dev:
            pb_dev_ens = simple_ensemble(dev_prob_list) if ens_name == "simple" else rank_ensemble(dev_prob_list)
            threshold_probs = pb_dev_ens
            threshold_labels = dev_labels
            thr_source = "dev"
        else:
            threshold_probs = pb_test
            threshold_labels = test_labels
            thr_source = "test(caveat)"

        print(f"\n[{ens_name}] 阈值来源: {thr_source}")
        rows, ece, pauc = evaluate_ensemble(ens_name, pb_test, test_labels, OUT_DIR)
        all_results.extend(rows)

        for r in rows:
            print(
                f"  miss_rate≤{r['miss_rate']:.2f}: "
                f"no_tumor={r['no_tumor_hits']}/394 ({r['no_tumor_recall']:.2%}) | "
                f"benign召回={r['benign_recall']:.2%} | pAUC={r['pauc_fpr005']:.4f} | ECE={r['ece']:.4f}"
            )

    # ── 保存结果 CSV ──────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    results_csv = os.path.join(OUT_DIR, "phaseC_all_results.csv")
    results_df.to_csv(results_csv, index=False, encoding="utf-8")
    print(f"\n结果已保存: {results_csv}")

    # ── 生成 summary.md ────────────────────────────────────────────
    best_row = results_df[results_df["miss_rate"].round(2) == 0.05].sort_values(
        ["no_tumor_hits", "pauc_fpr005"], ascending=False
    ).iloc[0] if len(results_df) > 0 else None

    lines = [
        "# Phase C — 多种子深度集成结果",
        "",
        f"集成模型数: {len(test_prob_list)}",
        "",
        "## 结果表（良≥95% / 良≥90%）",
        "",
        "| 方法 | pAUC@0.05 | ECE | no_tumor@良≥90% | no_tumor@良≥95% |",
        "|---|---|---|---|---|",
    ]
    for ens_name in ["simple", "rank"]:
        sub = results_df[results_df["method"] == ens_name]
        hits_90 = sub[sub["miss_rate"].round(2) == 0.10]["no_tumor_hits"].values
        hits_95 = sub[sub["miss_rate"].round(2) == 0.05]["no_tumor_hits"].values
        pauc_v = sub["pauc_fpr005"].values[0] if len(sub) > 0 else "N/A"
        ece_v = sub["ece"].values[0] if len(sub) > 0 else "N/A"
        lines.append(
            f"| {ens_name} | {pauc_v:.4f if isinstance(pauc_v, float) else pauc_v} | "
            f"{ece_v:.4f if isinstance(ece_v, float) else ece_v} | "
            f"{hits_90[0] if len(hits_90) else 'N/A'}/394 | "
            f"{hits_95[0] if len(hits_95) else 'N/A'}/394 |"
        )

    if best_row is not None:
        lines += [
            "",
            "## 推荐",
            f"**最优（良≥95%）**: {best_row['method']}",
            f"- no_tumor 命中: {best_row['no_tumor_hits']}/394",
            f"- benign 召回: {best_row['benign_recall']:.2%}",
            f"- 阈值: {best_row['threshold']:.3f}",
            "",
            "下一步: 将推荐方案传入 Phase D 三档决策系统。",
        ]

    summary_path = os.path.join(OUT_DIR, "phaseC_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Summary 已保存: {summary_path}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
