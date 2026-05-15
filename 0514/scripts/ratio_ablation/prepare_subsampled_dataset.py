"""
prepare_subsampled_dataset.py

从 baseline 的 train_split.xlsx (2311 张) 中按 patient-level 随机抽取，
生成 D1/D2/D3 三份不同物理比例的子集 xlsx。

D4（自然分布）= baseline train_split.xlsx 本身，不重复生成。
val_split.xlsx 永远是 baseline 的那份（424 张），所有实验共用。

Baseline train_split 类别分布:
  mal = 242, ben = 459, nt = 1610  (1 : 1.9 : 6.65)

输出比例（用 mal=242 作锚点）:
  D1 (1:1:1)    -> mal=242, ben=242, nt=242   (n=726)
  D2 (1:1:2)    -> mal=242, ben=242, nt=484   (n=968)
  D3 (1:1.9:3)  -> mal=242, ben=459, nt=726   (n=1427)
"""

from __future__ import annotations

import os
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

# ─── 路径 ─────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
ROOT_DIR   = (SCRIPT_DIR / ".." / ".." / "..").resolve()
sys.path.insert(0, str(ROOT_DIR / "0502" / "scripts"))

from risk_utils import extract_patient_id  # noqa: E402

BASE_LOG = ROOT_DIR / "0514" / "logs" / "20260514_task_risk_SwinV2Tiny_ordinal_trimodal_1"
TRAIN_SPLIT_BASELINE = BASE_LOG / "train_split.xlsx"
OUT_DIR = ROOT_DIR / "0514dataset_flat" / "ratio_ablation_splits"

SEED = 42
CLASS_NAMES = {0: "Malignant", 1: "Benign", 2: "No_Tumor"}

# 目标每类 image 数（mal 全保留作为锚点）
CONFIGS = {
    "D1_1_1_1":   {0: 242, 1: 242, 2: 242},
    "D2_1_1_2":   {0: 242, 1: 242, 2: 484},
    "D3_1_1.9_3": {0: 242, 1: 459, 2: 726},
}


def patient_level_subsample(df: pd.DataFrame, target_per_class: dict[int, int],
                            seed: int) -> pd.DataFrame:
    """对每个类，按 patient-level 随机抽到 target image 数为止。

    确定性: 同一 seed 同一 df → 同一结果。
    若 patient 抽完后图像数超过 target，多出的部分接受（不裁单张图，保留整患者）。
    """
    rng = random.Random(seed)
    selected_rows = []

    for label, target_n in target_per_class.items():
        cls_df = df[df["label"] == label].copy()
        # 按 patient_id 分组，得到 {pid: [row_indices]}
        cls_df["__pid"] = cls_df["image_path"].apply(extract_patient_id)
        pid_groups = cls_df.groupby("__pid").groups  # {pid: Index([row_idx,...])}
        pids = sorted(pid_groups.keys())  # 排序保证确定性
        rng.shuffle(pids)

        accum = 0
        chosen_rows = []
        for pid in pids:
            row_idx = list(pid_groups[pid])
            chosen_rows.extend(row_idx)
            accum += len(row_idx)
            if accum >= target_n:
                break

        sub = cls_df.loc[chosen_rows].drop(columns=["__pid"])
        selected_rows.append(sub)

    out = pd.concat(selected_rows, ignore_index=True)
    return out


def main():
    print("=" * 70)
    print(f"读取 baseline train_split: {TRAIN_SPLIT_BASELINE}")
    df = pd.read_excel(TRAIN_SPLIT_BASELINE)
    counts = df["label"].value_counts().sort_index().to_dict()
    print(f"原始: {len(df)} 张 | "
          f"{', '.join(f'{CLASS_NAMES[k]}={v}' for k,v in counts.items())}")
    print(f"输出目录: {OUT_DIR}")
    print("=" * 70)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    report_lines = [
        "=== Ratio Ablation Subsampling Report ===",
        f"Source: {TRAIN_SPLIT_BASELINE}",
        f"Seed:   {SEED}",
        f"Baseline counts: {counts}",
        "",
    ]

    for cfg_name, target in CONFIGS.items():
        sub_df = patient_level_subsample(df, target, SEED)
        out_path = OUT_DIR / f"train_{cfg_name}.xlsx"
        # 仅保留训练脚本需要的列
        sub_df[["image_path", "label"]].to_excel(out_path, index=False)

        actual = sub_df["label"].value_counts().sort_index().to_dict()
        actual_str = ", ".join(f"{CLASS_NAMES[k]}={actual.get(k,0)}" for k in [0,1,2])
        # patient 校验
        n_patients = sub_df["image_path"].apply(extract_patient_id).nunique()

        line = (f"  [{cfg_name}] target={target} | actual={actual} "
                f"| total={len(sub_df)} | patients={n_patients}")
        print(line)
        report_lines.append(line)

    # patient 泄露校验: 子集 patient 必须全部 ⊆ baseline train_split
    base_pids = set(df["image_path"].apply(extract_patient_id))
    for cfg_name in CONFIGS:
        sub = pd.read_excel(OUT_DIR / f"train_{cfg_name}.xlsx")
        sub_pids = set(sub["image_path"].apply(extract_patient_id))
        leak = sub_pids - base_pids
        assert not leak, f"{cfg_name} 含 baseline 之外的患者: {leak}"
        report_lines.append(f"  [{cfg_name}] patient 子集校验 ✅")

    report_path = OUT_DIR / "subsample_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告: {report_path}")
    print("✅ 完成")


if __name__ == "__main__":
    main()
