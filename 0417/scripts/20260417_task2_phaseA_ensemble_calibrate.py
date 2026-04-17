"""
Phase A: 零训练 quick win — 集成已有三模型 + 温度缩放校准

输入: 三份已有 test_probs.csv
输出: 0417/logs/phaseA/
  - phaseA_all_methods.csv    方法×miss_rate×指标长表
  - phaseA_summary.md         最优推荐
  - *_reliability.png/.csv    校准曲线 (4 张)

注意：温标定在 test 上做 5-fold CV 仅作诊断，存在少量泄漏，
      不作为最终阈值决策依据（Phase B/C 有干净 dev 后替换）。
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ── 路径设置 ────────────────────────────────────────────────────────
_ROOT = "/data1/ouyangxinglong/GBP-Cascade"
_SCRIPTS_V5 = os.path.join(_ROOT, "0408", "scripts")
_SCRIPTS_EXT = os.path.join(_ROOT, "0417", "scripts")
sys.path.insert(0, _SCRIPTS_V5)
sys.path.insert(0, _SCRIPTS_EXT)

from seg_cls_utils_v5 import (
    find_constrained_threshold_text,
    compute_binary_reliability_stats,
    save_reliability_stats_csv,
    save_reliability_diagram,
)
from seg_cls_utils_v5_ext import (
    simple_ensemble,
    rank_ensemble,
    compute_pauc_at_fpr,
    TemperatureScaler,
)

# ── 输入文件 ─────────────────────────────────────────────────────────
PROB_FILES = {
    "Exp18": os.path.join(
        _ROOT, "0408", "logs",
        "20260408_task2_SwinV2Tiny_segcls_18",
        "20260408_task2_SwinV2Tiny_segcls_18_test_probs.csv",
    ),
    "Exp18_2": os.path.join(
        _ROOT, "0408", "logs",
        "20260415_task2_SwinV2Tiny_segcls_18_2",
        "20260415_task2_SwinV2Tiny_segcls_18_2_test_probs.csv",
    ),
    "Exp18_3": os.path.join(
        _ROOT, "0408", "logs",
        "20260415_task2_SwinV2Tiny_segcls_18_3",
        "20260415_task2_SwinV2Tiny_segcls_18_3_test_probs.csv",
    ),
}

OUT_DIR = os.path.join(_ROOT, "0417", "logs", "phaseA")
MISS_RATES = [0.02, 0.05, 0.10]
CALIBRATION_BINS = 10


# ── 5-fold 温标定（在 test 上，仅作诊断）────────────────────────────

def kfold_temperature_calibrate(probs_benign, labels, k=5, seed=42):
    """在 test 上做 K-fold 交叉温标定，返回校准后的 prob_benign (N,)。
    注意: test 上 CV 有少量泄漏，结果仅供诊断，不用于最终决策。
    """
    from sklearn.model_selection import KFold

    N = len(probs_benign)
    calibrated = np.zeros(N, dtype=np.float32)
    logits = np.stack([probs_benign, 1.0 - probs_benign], axis=1)  # (N, 2) 近似 logits

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    T_values = []

    for fold, (fit_idx, apply_idx) in enumerate(kf.split(np.arange(N))):
        logits_fit = torch.tensor(logits[fit_idx], dtype=torch.float32)
        labels_fit = torch.tensor(labels[fit_idx], dtype=torch.long)

        scaler = TemperatureScaler()
        T_val = scaler.fit(logits_fit, labels_fit)
        T_values.append(T_val)

        logits_apply = torch.tensor(logits[apply_idx], dtype=torch.float32)
        probs_cal = F.softmax(logits_apply / max(T_val, 0.1), dim=-1).numpy()
        calibrated[apply_idx] = probs_cal[:, 0]

    return calibrated, float(np.mean(T_values))


def evaluate_method(name, probs_benign, labels, out_dir):
    """对一个方法计算所有 miss_rate 下的阈值指标 + reliability 统计。"""
    rows = []
    pauc = compute_pauc_at_fpr(labels, probs_benign, max_fpr=0.05)

    for mr in MISS_RATES:
        policy = find_constrained_threshold_text(
            probs_benign, labels, max_benign_miss_rate=mr
        )
        rows.append({
            "method": name,
            "miss_rate": mr,
            "threshold": policy["threshold"],
            "benign_recall": policy["benign_recall"],
            "benign_miss_rate_actual": policy["benign_miss_rate"],
            "no_tumor_hits": policy["selected_no_tumor"],
            "no_tumor_recall": policy["no_tumor_recall"],
            "no_tumor_precision": policy["no_tumor_precision"],
            "macro_f1": policy["macro_f1"],
            "pauc_fpr005": pauc,
            "constraint_satisfied": policy["constraint_satisfied"],
        })

    # reliability
    y_true_nt = (labels == 1).astype(int)
    prob_nt = 1.0 - probs_benign
    rel_rows, ece, brier = compute_binary_reliability_stats(y_true_nt, prob_nt, n_bins=CALIBRATION_BINS)

    for r in rows:
        r["ece"] = ece
        r["brier"] = brier

    rel_csv = os.path.join(out_dir, f"{name}_reliability.csv")
    rel_png = os.path.join(out_dir, f"{name}_reliability.png")
    save_reliability_stats_csv(rel_rows, rel_csv)
    save_reliability_diagram(rel_rows, rel_png, title=f"{name} no_tumor 置信度-准确率校准")

    return rows, ece, brier, pauc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 加载三模型 probs ───────────────────────────────────────────
    dfs = {}
    for name, path in PROB_FILES.items():
        df = pd.read_csv(path)
        dfs[name] = df
        print(f"  [{name}] loaded {len(df)} rows from {path}")

    labels = dfs["Exp18"]["label"].values.astype(int)

    # 验证所有标签一致
    for name, df in dfs.items():
        assert np.all(df["label"].values == labels), f"{name} 标签与 Exp18 不一致！"

    probs = {name: df["prob_benign"].values.astype(np.float32) for name, df in dfs.items()}

    # ── 计算各方法 ─────────────────────────────────────────────────
    all_methods = {}

    # 三个单模型
    for name in probs:
        all_methods[name] = probs[name]

    # Simple Ensemble
    all_methods["Simple_Ensemble"] = simple_ensemble(list(probs.values()))

    # Rank Ensemble
    all_methods["Rank_Ensemble"] = rank_ensemble(list(probs.values()))

    # 5-fold 温标定（对 Simple Ensemble）
    cal_pb, mean_T = kfold_temperature_calibrate(all_methods["Simple_Ensemble"], labels)
    all_methods[f"Simple_Ensemble_Cal(T={mean_T:.2f})"] = cal_pb
    print(f"  [Simple Ensemble 5-fold Cal] 平均 T = {mean_T:.3f}")

    # 5-fold 温标定（对 Rank Ensemble）
    cal_pb_rank, mean_T_rank = kfold_temperature_calibrate(all_methods["Rank_Ensemble"], labels)
    all_methods[f"Rank_Ensemble_Cal(T={mean_T_rank:.2f})"] = cal_pb_rank
    print(f"  [Rank Ensemble 5-fold Cal] 平均 T = {mean_T_rank:.3f}")

    # ── 评估所有方法 ───────────────────────────────────────────────
    all_result_rows = []
    summary_lines = [
        "# Phase A — 集成 + 温标定 结果摘要",
        "",
        "> 注意：温标定在 test 上做 5-fold CV，有少量泄漏，结果仅供诊断。",
        "> Phase B/C 的阈值均来自干净 dev 划分。",
        "",
        "## 各方法指标（良≥90% / 良≥95%）",
        "",
        "| 方法 | pAUC@0.05 | ECE | Brier | no_tumor@良≥90% | no_tumor@良≥95% |",
        "|---|---|---|---|---|---|",
    ]

    for name, pb in all_methods.items():
        print(f"\n[{name}] 评估中...")
        rows, ece, brier, pauc = evaluate_method(name, pb, labels, OUT_DIR)
        all_result_rows.extend(rows)

        hits_90 = next((r["no_tumor_hits"] for r in rows if abs(r["miss_rate"] - 0.10) < 1e-6), "N/A")
        hits_95 = next((r["no_tumor_hits"] for r in rows if abs(r["miss_rate"] - 0.05) < 1e-6), "N/A")
        summary_lines.append(
            f"| {name} | {pauc:.4f} | {ece:.4f} | {brier:.4f} | {hits_90}/394 | {hits_95}/394 |"
        )
        print(f"  pAUC@0.05={pauc:.4f} | ECE={ece:.4f} | 良≥90%→{hits_90} | 良≥95%→{hits_95}")

    # ── 保存 all_methods.csv ───────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "phaseA_all_methods.csv")
    pd.DataFrame(all_result_rows).to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n已保存: {csv_path}")

    # ── 最优方法推荐 ───────────────────────────────────────────────
    df_res = pd.DataFrame(all_result_rows)
    # 在 miss_rate=0.05 下找 no_tumor 命中最多的方法
    df_95 = df_res[df_res["miss_rate"].round(2) == 0.05].sort_values(
        ["no_tumor_hits", "pauc_fpr005"], ascending=False
    )
    best_row = df_95.iloc[0] if len(df_95) > 0 else None

    summary_lines += [
        "",
        "## 推荐方案",
        "",
    ]
    if best_row is not None:
        summary_lines += [
            f"**最优（良≥95%）**：{best_row['method']}",
            f"- no_tumor 命中: {best_row['no_tumor_hits']}/394 ({best_row['no_tumor_recall']:.2%})",
            f"- 阈值: {best_row['threshold']:.3f}",
            f"- benign 召回: {best_row['benign_recall']:.2%}",
            f"- pAUC@0.05: {best_row['pauc_fpr005']:.4f}",
            f"- ECE: {best_row['ece']:.4f} (校准前 Exp18-3 ECE ≈ 0.32)",
            "",
            "**下一步**：Phase B 消融实验完成后，用干净 dev 集重新校准并集成。",
        ]
    else:
        summary_lines.append("（未找到满足约束的方案）")

    summary_path = os.path.join(OUT_DIR, "phaseA_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    print(f"已保存: {summary_path}")
    print("\n" + "\n".join(summary_lines))


if __name__ == "__main__":
    main()
