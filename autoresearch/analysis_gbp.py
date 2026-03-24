"""
GBP-Cascade 自主实验分析脚本.
用法: python analysis_gbp.py
"""

import os
import sys

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TSV_PATH = os.path.join(SCRIPT_DIR, "results.tsv")


def main():
    if not os.path.exists(TSV_PATH):
        print(f"No results.tsv found at {TSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(TSV_PATH, sep="\t")
    n = len(df)
    n_keep = (df["status"] == "keep").sum()
    n_discard = (df["status"] == "discard").sum()
    n_crash = (df["status"] == "crash").sum()

    print(f"{'=' * 50}")
    print(f"GBP-Cascade Task 2 AutoResearch Results")
    print(f"{'=' * 50}")
    print(f"Total experiments:  {n}")
    print(f"  Kept:             {n_keep} ({100*n_keep/max(n,1):.0f}%)")
    print(f"  Discarded:        {n_discard} ({100*n_discard/max(n,1):.0f}%)")
    print(f"  Crashed:          {n_crash} ({100*n_crash/max(n,1):.0f}%)")

    kept = df[df["status"] == "keep"]
    if len(kept) > 0:
        best_idx = kept["f1_at_threshold"].idxmax()
        best_row = kept.loc[best_idx]
        print(f"\nBest f1_at_threshold: {best_row['f1_at_threshold']:.6f}")
        print(f"  Commit:           {best_row['commit']}")
        print(f"  VRAM:             {best_row['peak_vram_mb']:.1f} MB")
        print(f"  Description:      {best_row['description']}")

        # Progression
        print(f"\nProgression (kept experiments):")
        running_best = 0.0
        for _, row in kept.iterrows():
            improved = ""
            if row["f1_at_threshold"] > running_best:
                running_best = row["f1_at_threshold"]
                improved = " <-- NEW BEST"
            print(f"  {row['commit']}  F1={row['f1_at_threshold']:.6f}  "
                  f"VRAM={row['peak_vram_mb']:.0f}MB  {row['description']}{improved}")

    # Plot
    if n == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # F1 progression
    ax1 = axes[0]
    colors = {"keep": "#2ecc71", "discard": "#e74c3c", "crash": "#95a5a6"}
    for i, row in df.iterrows():
        c = colors.get(row["status"], "#3498db")
        marker = "o" if row["status"] == "keep" else ("x" if row["status"] == "crash" else "^")
        ax1.scatter(i, row["f1_at_threshold"], c=c, s=60, marker=marker, zorder=3, alpha=0.8)

    if len(kept) > 0:
        cummax = kept["f1_at_threshold"].cummax()
        ax1.plot(kept.index, cummax, "g-", linewidth=2.5, label="Best so far", zorder=2)
        ax1.axhline(y=cummax.iloc[-1], color="g", linestyle="--", alpha=0.3)

    ax1.axhline(y=0.624, color="orange", linestyle=":", alpha=0.6, label="Previous best (0.624)")
    ax1.set_ylabel("F1 at threshold (macro)", fontsize=12)
    ax1.set_title("GBP-Cascade Task 2 AutoResearch Progress", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.5, n - 0.5)

    # Memory usage
    ax2 = axes[1]
    valid = df[df["peak_vram_mb"] > 0]
    for i, row in valid.iterrows():
        c = colors.get(row["status"], "#3498db")
        ax2.bar(i, row["peak_vram_mb"] / 1024, color=c, alpha=0.7, width=0.8)
    ax2.set_xlabel("Experiment #", fontsize=12)
    ax2.set_ylabel("Peak VRAM (GB)", fontsize=12)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xlim(-0.5, n - 0.5)

    plt.tight_layout()
    out_path = os.path.join(SCRIPT_DIR, "progress.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot: {out_path}")


if __name__ == "__main__":
    main()
