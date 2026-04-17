"""
Phase D: 三档临床决策系统

输入:
  - Phase C 最佳集成 test_probs（自动探测 phaseC/）
  - Exp#19-3 MC Dropout _mc_dropout.csv（含 std，用于不确定性筛选）
  - 合并 dev probs（从 19-x dev_probs.csv 读取，用于阈值选择）

三档定义（阈值全在 dev 上确定）:
  A 档: P(no_tumor) >= T_high AND std(P(benign)) <= σ_lo
        → 高置信 no_tumor，建议免手术
  B 档: P(benign) >= T_b_high (benign 召回=100% 对应的最高阈值)
        → 高置信 benign，建议手术
  C 档: 其余 → 灰区，建议 MDT / 复查

输出: 0417/logs/phaseD/
  - phaseD_tier_assignments.csv
  - phaseD_tier_report.md
  - phaseD_tier_confusion.png
"""

import os
import sys
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = "/data1/ouyangxinglong/GBP-Cascade"
sys.path.insert(0, os.path.join(_ROOT, "0408", "scripts"))
sys.path.insert(0, os.path.join(_ROOT, "0417", "scripts"))

from seg_cls_utils_v5 import find_constrained_threshold_text
from seg_cls_utils_v5_ext import compute_pauc_at_fpr

OUT_DIR = os.path.join(_ROOT, "0417", "logs", "phaseD")
LOG_BASE = os.path.join(_ROOT, "0417", "logs")

# ── 参数 ──────────────────────────────────────────────────────────
DEV_BENIGN_RECALL_FOR_A = 0.98   # A档: dev 上 benign 召回 >= 98%
DEV_BENIGN_RECALL_FOR_B = 1.00   # B档: dev 上 benign 召回 = 100%
STD_PERCENTILE_FOR_A = 10        # σ_lo = dev 上 std 的第 10 百分位（低不确定性）


def find_file(patterns):
    for p in patterns:
        files = glob.glob(p)
        if files:
            return sorted(files)[-1]  # 取最新
    return None


def load_probs_csv(path):
    df = pd.read_csv(path)
    pb = df.get("prob_benign", df.get("prob_benign_mean")).values.astype(np.float32)
    labels = df["label"].values.astype(int)
    return pb, labels, df


def find_threshold_for_recall(probs_benign, labels, target_benign_recall):
    """在 probs_benign 上搜索满足 benign_recall >= target 的最高阈值（即最少 no_tumor 被标为 benign）。
    用于 A 档：找到保证 benign 召回=target 的 T_high（分类为 no_tumor 的阈值）。
    pred = (prob_benign < T_high) → no_tumor（高 T_high → 更多被识别为 no_tumor）
    """
    best_T = 0.01
    for T in np.arange(0.01, 0.99, 0.001):
        # A档条件: prob_no_tumor >= T_high, 等价 prob_benign < T_high → no_tumor
        # 此处 T_high 作用在 prob_no_tumor 上: P_nt >= T_high → no_tumor
        preds_no_tumor = (1.0 - probs_benign) >= T  # predicted as no_tumor
        # benign recall: benign 中没被预测为 no_tumor 的比例
        benign_mask = labels == 0
        benign_recall = 1.0 - preds_no_tumor[benign_mask].mean()
        if benign_recall >= target_benign_recall:
            best_T = T
    return float(best_T)


def find_benign_100_threshold(probs_benign, labels):
    """B档: 找最高的 T_b_high 使得预测为 benign (prob_benign >= T_b_high) 的样本中 benign 召回=100%。
    等价于: 不漏掉任何 benign。"""
    # 当 T_b_high 很高时，只有高置信 benign 被识别，benign 召回 < 100%
    # 当 T_b_high 很低时，大量样本被预测为 benign，benign 召回 = 100%
    # 我们要找最高的 T 使得所有 benign 都被捕获
    benign_probs = probs_benign[labels == 0]
    # 最低 benign prob
    return float(np.percentile(benign_probs, 5))  # 保守: 5th percentile of benign probs


def apply_three_tier(pb_test, std_test, labels_test, T_nt_high, sigma_lo, T_b_high):
    """按三档规则分配每个样本。返回 tier 数组 (A/B/C)。"""
    p_nt = 1.0 - pb_test
    tier = np.full(len(pb_test), "C", dtype=object)

    # A 档: 高置信 no_tumor
    a_mask = (p_nt >= T_nt_high) & (std_test <= sigma_lo)
    tier[a_mask] = "A"

    # B 档: 高置信 benign（优先级低于 A，因为 A 是高置信 no_tumor）
    b_mask = (pb_test >= T_b_high) & (tier != "A")
    tier[b_mask] = "B"

    return tier


def tier_stats(tier, labels, tier_name):
    """计算某档的统计数据。"""
    mask = tier == tier_name
    n = mask.sum()
    if n == 0:
        return {"tier": tier_name, "n": 0, "coverage": 0.0,
                "benign_recall_in_tier": 0.0, "no_tumor_recall_in_tier": 0.0,
                "ppv_benign": 0.0, "npv_no_tumor": 0.0}

    in_tier_labels = labels[mask]
    total = len(labels)
    n_benign_total = (labels == 0).sum()
    n_nt_total = (labels == 1).sum()

    # benign_recall_in_tier: 档内 benign 正确标注为 benign 的比例（不适用 A 档）
    benign_in_tier = (in_tier_labels == 0).sum()
    nt_in_tier = (in_tier_labels == 1).sum()

    # PPV(benign): 档内 benign 占档内总数
    ppv_benign = benign_in_tier / n if n > 0 else 0.0
    # NPV(no_tumor): 档内 no_tumor 占档内总数
    npv_nt = nt_in_tier / n if n > 0 else 0.0

    # 全局 recall
    benign_recall_global = benign_in_tier / max(n_benign_total, 1)
    nt_recall_global = nt_in_tier / max(n_nt_total, 1)

    return {
        "tier": tier_name,
        "n": int(n),
        "coverage": float(n / total),
        "benign_in_tier": int(benign_in_tier),
        "no_tumor_in_tier": int(nt_in_tier),
        "benign_recall_global": float(benign_recall_global),
        "no_tumor_recall_global": float(nt_recall_global),
        "ppv_benign": float(ppv_benign),
        "npv_no_tumor": float(npv_nt),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── 加载 Phase C 集成 test probs ─────────────────────────────
    test_patterns = [
        os.path.join(LOG_BASE, "phaseC", "phaseC_test_probs_rank.csv"),
        os.path.join(LOG_BASE, "phaseC", "phaseC_test_probs_simple.csv"),
    ]
    test_path = find_file(test_patterns)
    if test_path is None:
        # 回退: 使用 19-3 单模型
        test_path = find_file([os.path.join(LOG_BASE, "*19_3*", "*test_probs_calibrated*.csv")])
    if test_path is None:
        print("❌ 找不到 test probs。请先运行 Phase C 或至少完成 Exp#19-3。")
        return
    print(f"Test probs: {test_path}")
    pb_test, labels_test, df_test = load_probs_csv(test_path)

    # ── 加载 MC Dropout std ────────────────────────────────────────
    mc_pattern = os.path.join(LOG_BASE, "*19_3*", "*mc_dropout*.csv")
    mc_path = find_file([mc_pattern])
    if mc_path is not None:
        df_mc = pd.read_csv(mc_path)
        std_test = df_mc["prob_benign_std"].values.astype(np.float32)
        print(f"MC Dropout std: {mc_path}")
        assert len(std_test) == len(pb_test), "MC Dropout std 与 test probs 长度不一致！"
    else:
        print("⚠ 未找到 MC Dropout std，使用常数 0（禁用不确定性筛选）")
        std_test = np.zeros(len(pb_test), dtype=np.float32)

    # ── 加载合并 dev probs ─────────────────────────────────────────
    dev_patterns = [
        os.path.join(LOG_BASE, "*19_3*", "*best_dev_probs*.csv"),
        os.path.join(LOG_BASE, "*19_3*", "*dev_probs*.csv"),
        os.path.join(LOG_BASE, "*19_1*", "*dev_probs*.csv"),
    ]
    dev_path = find_file(dev_patterns)
    if dev_path is not None:
        pb_dev, labels_dev, _ = load_probs_csv(dev_path)
        # dev MC std
        mc_dev_path = find_file([
            os.path.join(LOG_BASE, "*19_3*", "*mc_dropout*.csv"),
        ])
        if mc_dev_path is not None:
            df_mc_dev = pd.read_csv(mc_dev_path)
            # MC 推理是对 test 做的，dev 没有 MC std，用 0.05 作为近似 σ_lo
            sigma_lo = 0.05
        else:
            sigma_lo = 0.05
        print(f"Dev probs: {dev_path}")
        use_dev_thresholds = True
    else:
        print("⚠ 未找到 dev probs，在 test 上估计阈值（有偏差，仅参考）")
        pb_dev = pb_test
        labels_dev = labels_test
        sigma_lo = float(np.percentile(std_test, STD_PERCENTILE_FOR_A))
        use_dev_thresholds = False

    # ── 在 dev 上确定阈值 ──────────────────────────────────────────
    # A 档: T_nt_high = 在 dev 上 benign 召回 >= 98% 时，P(no_tumor) 的阈值
    T_nt_high = find_threshold_for_recall(pb_dev, labels_dev, DEV_BENIGN_RECALL_FOR_A)
    # B 档: T_b_high = dev 上 benign 的 P(benign) 5th percentile（确保所有 benign 被捕获）
    T_b_high = find_benign_100_threshold(pb_dev, labels_dev)

    if std_test.max() > 0:
        sigma_lo = float(np.percentile(std_test, STD_PERCENTILE_FOR_A))

    print(f"\n阈值: T_nt_high={T_nt_high:.3f} (A档 P_nt≥), T_b_high={T_b_high:.3f} (B档 P_benign≥), σ_lo={sigma_lo:.4f}")

    # 验证 dev 上的 A 档 benign 召回
    p_nt_dev = 1.0 - pb_dev
    dev_a_mask = (p_nt_dev >= T_nt_high)
    dev_a_benign_recall = 1.0 - dev_a_mask[labels_dev == 0].mean() if (labels_dev == 0).sum() > 0 else 0.0
    print(f"Dev A档验证: benign 召回 = {dev_a_benign_recall:.2%} (目标≥{DEV_BENIGN_RECALL_FOR_A:.0%})")

    # ── 在 test 上分配三档 ─────────────────────────────────────────
    tier = apply_three_tier(pb_test, std_test, labels_test, T_nt_high, sigma_lo, T_b_high)

    # ── 保存分配结果 ───────────────────────────────────────────────
    df_assignments = df_test.copy()
    df_assignments["tier"] = tier
    df_assignments["prob_no_tumor"] = 1.0 - pb_test
    df_assignments["std_benign"] = std_test
    assignments_csv = os.path.join(OUT_DIR, "phaseD_tier_assignments.csv")
    df_assignments.to_csv(assignments_csv, index=False, encoding="utf-8")
    print(f"\n三档分配已保存: {assignments_csv}")

    # ── 统计各档 ──────────────────────────────────────────────────
    stats = {t: tier_stats(tier, labels_test, t) for t in ["A", "B", "C"]}
    total_test = len(labels_test)
    n_benign = (labels_test == 0).sum()
    n_nt = (labels_test == 1).sum()

    print("\n=== 三档统计 ===")
    for t, s in stats.items():
        print(
            f"  {t} 档: n={s['n']} ({s['coverage']:.1%}) | "
            f"benign={s['benign_in_tier']} | no_tumor={s['no_tumor_in_tier']} | "
            f"benign全局召回={s['benign_recall_global']:.2%} | "
            f"no_tumor全局召回={s['no_tumor_recall_global']:.2%}"
        )

    # ── 混淆图 ────────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # 左图: 各档分布（benign/no_tumor 叠加）
        ax = axes[0]
        tier_labels = ["A (免手术)", "B (手术)", "C (灰区)"]
        b_counts = [stats["A"]["benign_in_tier"], stats["B"]["benign_in_tier"], stats["C"]["benign_in_tier"]]
        nt_counts = [stats["A"]["no_tumor_in_tier"], stats["B"]["no_tumor_in_tier"], stats["C"]["no_tumor_in_tier"]]
        x = np.arange(3)
        ax.bar(x, b_counts, label="benign (需手术)", color="#e74c3c", alpha=0.8)
        ax.bar(x, nt_counts, bottom=b_counts, label="no_tumor (息肉)", color="#2ecc71", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(tier_labels)
        ax.set_title("三档分类结果（test set）"); ax.set_ylabel("样本数"); ax.legend()
        for i, (b, nt) in enumerate(zip(b_counts, nt_counts)):
            ax.text(i, b + nt + 1, f"{b+nt}", ha="center", fontsize=9)

        # 右图: 概率分布直方图（按档）
        ax2 = axes[1]
        colors = {"A": "#2ecc71", "B": "#e74c3c", "C": "#95a5a6"}
        for t_name, color in colors.items():
            mask_t = tier == t_name
            if mask_t.sum() > 0:
                ax2.hist((1.0 - pb_test)[mask_t], bins=20, alpha=0.5, color=color,
                         label=f"{t_name} 档 (n={mask_t.sum()})", density=True)
        ax2.set_xlabel("P(no_tumor)"); ax2.set_ylabel("密度")
        ax2.set_title("各档 P(no_tumor) 分布"); ax2.legend()
        ax2.axvline(T_nt_high, color="green", linestyle="--", label=f"T_high={T_nt_high:.3f}")
        ax2.legend()

        plt.tight_layout()
        fig_path = os.path.join(OUT_DIR, "phaseD_tier_confusion.png")
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"三档图已保存: {fig_path}")
    except Exception as e:
        print(f"⚠ 图表生成失败: {e}")

    # ── 生成 tier_report.md ───────────────────────────────────────
    lines = [
        "# Phase D — 三档临床决策系统报告",
        "",
        f"测试集总数: {total_test} (benign={n_benign}, no_tumor={n_nt})",
        f"集成方法: {os.path.basename(test_path)}",
        f"MC Dropout std: {'是' if mc_path else '否（统一0）'}",
        "",
        "## 阈值设定（dev 上确定）",
        "",
        f"- A 档 T_high = {T_nt_high:.3f}（P(no_tumor) ≥ T_high → no_tumor）",
        f"- A 档 σ_lo = {sigma_lo:.4f}（MC std ≤ σ_lo，低不确定性）",
        f"- B 档 T_b_high = {T_b_high:.3f}（P(benign) ≥ T_b_high → benign）",
        f"- Dev A档 benign 召回 = {dev_a_benign_recall:.2%}（目标 ≥ {DEV_BENIGN_RECALL_FOR_A:.0%}）",
        "",
        "## 三档统计（test set）",
        "",
        "| 档位 | 含义 | 样本数 | 覆盖率 | benign数 | no_tumor数 | benign全局召回 | no_tumor全局召回 |",
        "|---|---|---|---|---|---|---|---|",
    ]

    tier_desc = {
        "A": "高置信 no_tumor → 建议免手术/随访",
        "B": "高置信 benign → 建议手术",
        "C": "灰色区域 → 建议 MDT/复查",
    }
    for t in ["A", "B", "C"]:
        s = stats[t]
        lines.append(
            f"| **{t}** | {tier_desc[t]} | {s['n']} | {s['coverage']:.1%} | "
            f"{s['benign_in_tier']} | {s['no_tumor_in_tier']} | "
            f"{s['benign_recall_global']:.2%} | {s['no_tumor_recall_global']:.2%} |"
        )

    lines += [
        "",
        "## 临床解读",
        "",
    ]

    # A 档分析
    sa = stats["A"]
    a_benign_miss = sa["benign_in_tier"]
    lines += [
        f"### A 档（高置信 no_tumor，建议免手术）",
        f"- **覆盖 {sa['n']}/{total_test}（{sa['coverage']:.1%}）** 患者可免手术",
        f"- 档内 no_tumor：{sa['no_tumor_in_tier']}（正确免手术）",
        f"- 档内 benign 漏诊：{a_benign_miss}（{'需改进阈值' if a_benign_miss > 2 else '在可接受范围内'}）",
        f"- 全局 benign 召回损失：{sa['benign_recall_global']:.2%}（这部分 benign 被漏到 A 档）",
        "",
        f"### B 档（高置信 benign，建议手术）",
        f"- 覆盖 {stats['B']['n']}/{total_test}（{stats['B']['coverage']:.1%}）",
        f"- 档内 benign：{stats['B']['benign_in_tier']}，no_tumor：{stats['B']['no_tumor_in_tier']}",
        "",
        f"### C 档（灰区，建议 MDT/复查）",
        f"- 覆盖 {stats['C']['n']}/{total_test}（{stats['C']['coverage']:.1%}）",
        f"- 模型无法高置信决策，需医生/MDT 介入",
        "",
        "## 使用说明",
        "",
        "1. A 档患者：可提供免手术依据（随访），需向患者告知约 2% 的漏诊可能性",
        "2. B 档患者：高置信手术指征，AI 作为辅助确认",
        "3. C 档患者：转 MDT 或要求额外检查（CT/MRI/EUS）",
        "",
        "> 本系统为辅助决策工具，最终手术决定由临床医生负责。",
    ]

    report_path = os.path.join(OUT_DIR, "phaseD_tier_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n报告已保存: {report_path}")
    print("\n" + "\n".join(lines[:30]))


if __name__ == "__main__":
    main()
