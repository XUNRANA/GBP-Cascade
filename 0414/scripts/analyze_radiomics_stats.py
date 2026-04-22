#!/usr/bin/env python3
"""
Phase 2.3 ~ 2.6 — 经典影像组学特征分析.

读入:
  logs/feature_analysis_best_softmax_segcls3/radiomics/features_test_gb_roi.csv
  (可选) logs/feature_analysis_best_softmax_segcls3/radiomics/features_test_lesion_roi.csv

输出 (全部写入 radiomics/ 目录):
  2.3_pca_scree.png           - PCA 累计方差
  2.3_pca_2d.png              - PC1 vs PC2 散点 (按类着色)
  2.3_tsne.png                - t-SNE 2D 散点
  2.3_umap.png                - UMAP 2D 散点
  2.5_volcano_3panels.png     - mal-ben / mal-no / ben-no 三幅 volcano
  2.5_top10_boxplot_grid.png  - top-10 显著特征箱线 (按 q_anova)
  2.5_top5_violin_ben_vs_no.png - top-5 benign-vs-no_tumor 最显著特征 violin
  2.5_top20_corr_heatmap.png  - top-20 特征相关矩阵
  2.5_class_radar.png         - 三类在 top-6 特征上的 z-score 雷达图
  2.6_gb_vs_lesion_compare.png - (可选) gb ROI vs lesion ROI 对比
  biomarker_table.csv
  report.json
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path("/data1/ouyangxinglong/GBP-Cascade/0414")
RAD_DIR = ROOT / "logs/feature_analysis_best_softmax_segcls3/radiomics"
GB_CSV = RAD_DIR / "features_test_gb_roi.csv"
LESION_CSV = RAD_DIR / "features_test_lesion_roi.csv"

CLASS_NAMES = ["malignant", "benign", "no_tumor"]
CLASS_COLORS = {0: "#d73027", 1: "#4575b4", 2: "#1a9850"}
META_COLS = {"image_path", "true_class", "true_class_name",
             "pred_class", "ordinal_score"}


# ─────────────────────────────────────────────────────────────
def load_features(csv_path: Path):
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    # drop non-numeric / constant columns
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    bad_cols = X.columns[(X.isna().any()) | (X.nunique() <= 1)].tolist()
    if bad_cols:
        X = X.drop(columns=bad_cols)
        feat_cols = [c for c in feat_cols if c not in bad_cols]
    return df, X.values.astype(np.float64), feat_cols, bad_cols


def kmeans_ari(X_std, y_true, n_clusters=3):
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = km.fit_predict(X_std)
    return float(adjusted_rand_score(y_true, clusters))


# ─────────────────────────────────────────────────────────────
# Phase 2.3 — 降维可视化
# ─────────────────────────────────────────────────────────────
def plot_pca_scree(X_std, out_path):
    pca = PCA().fit(X_std)
    cum = np.cumsum(pca.explained_variance_ratio_)
    idx80 = int(np.searchsorted(cum, 0.80) + 1)
    idx90 = int(np.searchsorted(cum, 0.90) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(range(1, len(cum) + 1), cum, marker="o", ms=3)
    ax.axhline(0.80, color="orange", ls="--", lw=1, label=f"80% @ PC{idx80}")
    ax.axhline(0.90, color="red", ls="--", lw=1, label=f"90% @ PC{idx90}")
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title(f"PCA Scree (n_feat={X_std.shape[1]}, n={X_std.shape[0]})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return idx80, idx90, cum.tolist()


def scatter_by_class(emb2d, y, class_names, title, out_path, sil):
    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=140)
    for c in sorted(np.unique(y)):
        idx = y == c
        ax.scatter(
            emb2d[idx, 0], emb2d[idx, 1], s=22, alpha=0.75,
            c=CLASS_COLORS[int(c)],
            label=f"{class_names[int(c)]} (n={int(idx.sum())})",
            edgecolors="white", linewidths=0.5,
        )
    ax.set_title(f"{title}\nsilhouette={sil:.3f}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_pca_2d(X_std, y, class_names, out_path):
    pca = PCA(n_components=2, random_state=42).fit(X_std)
    Z = pca.transform(X_std)
    sil = float(silhouette_score(Z, y, metric="euclidean"))
    scatter_by_class(Z, y, class_names,
                     f"Radiomics PCA 2D   (expl. var = "
                     f"{pca.explained_variance_ratio_[0]*100:.1f}% / "
                     f"{pca.explained_variance_ratio_[1]*100:.1f}%)",
                     out_path, sil)
    return sil, pca.explained_variance_ratio_.tolist()


def run_tsne_2d(X_std, y, class_names, out_path):
    n = X_std.shape[0]
    perp = max(5, min(30, (n - 1) // 3))
    ts = TSNE(n_components=2, perplexity=perp, learning_rate="auto",
              init="pca", random_state=42)
    Z = ts.fit_transform(X_std)
    sil = float(silhouette_score(Z, y, metric="euclidean"))
    scatter_by_class(Z, y, class_names,
                     f"Radiomics t-SNE 2D  (perplexity={perp})",
                     out_path, sil)
    return sil, Z


def run_umap_2d(X_std, y, class_names, out_path):
    import umap
    um = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   random_state=42)
    Z = um.fit_transform(X_std)
    sil = float(silhouette_score(Z, y, metric="euclidean"))
    scatter_by_class(Z, y, class_names,
                     "Radiomics UMAP 2D  (n_neighbors=15, min_dist=0.1)",
                     out_path, sil)
    return sil, Z


# ─────────────────────────────────────────────────────────────
# Phase 2.4 — 单特征统计
# ─────────────────────────────────────────────────────────────
def cliff_delta(a, b):
    """Vectorized Cliff's delta: P(A>B) - P(A<B)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    # rank-based O((na+nb) log(na+nb))
    both = np.concatenate([a, b])
    ranks = stats.rankdata(both)
    ra = ranks[:na]
    # U1 (Mann-Whitney) = sum(ra) - na*(na+1)/2
    u1 = ra.sum() - na * (na + 1) / 2.0
    u2 = na * nb - u1
    return float((u1 - u2) / (na * nb))


def fdr_bh(pvals):
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def per_feature_stats(df, X, feat_cols):
    y = df["true_class"].values
    rows = []
    for j, feat in enumerate(feat_cols):
        col = X[:, j]
        groups = [col[y == c] for c in (0, 1, 2)]
        try:
            f_stat, p_an = stats.f_oneway(*groups)
        except Exception:
            f_stat, p_an = (np.nan, 1.0)
        try:
            _, p_mb = stats.mannwhitneyu(groups[0], groups[1], alternative="two-sided")
        except ValueError:
            p_mb = 1.0
        try:
            _, p_mn = stats.mannwhitneyu(groups[0], groups[2], alternative="two-sided")
        except ValueError:
            p_mn = 1.0
        try:
            _, p_bn = stats.mannwhitneyu(groups[1], groups[2], alternative="two-sided")
        except ValueError:
            p_bn = 1.0

        rows.append({
            "feature": feat,
            "family": feat.split("_")[1] if "_" in feat else "other",
            "f_stat": float(f_stat) if not math.isnan(f_stat) else 0.0,
            "p_anova": float(p_an) if not math.isnan(p_an) else 1.0,
            "p_mal_ben": float(p_mb),
            "p_mal_no": float(p_mn),
            "p_ben_no": float(p_bn),
            "cliff_delta_mal_ben": cliff_delta(groups[0], groups[1]),
            "cliff_delta_mal_no": cliff_delta(groups[0], groups[2]),
            "cliff_delta_ben_no": cliff_delta(groups[1], groups[2]),
            "mean_mal": float(np.mean(groups[0])),
            "mean_ben": float(np.mean(groups[1])),
            "mean_no":  float(np.mean(groups[2])),
            "std_mal":  float(np.std(groups[0])),
            "std_ben":  float(np.std(groups[1])),
            "std_no":   float(np.std(groups[2])),
        })
    res = pd.DataFrame(rows)
    res["q_anova"]    = fdr_bh(res["p_anova"].values)
    res["q_mal_ben"]  = fdr_bh(res["p_mal_ben"].values)
    res["q_mal_no"]   = fdr_bh(res["p_mal_no"].values)
    res["q_ben_no"]   = fdr_bh(res["p_ben_no"].values)
    res = res.sort_values("q_ben_no").reset_index(drop=True)
    res.insert(0, "rank_ben_no", res.index + 1)
    return res


# ─────────────────────────────────────────────────────────────
# Phase 2.5 — biomarker 可视化
# ─────────────────────────────────────────────────────────────
def short_feat(name, max_len=28):
    s = name.replace("original_", "")
    return s if len(s) <= max_len else s[:max_len - 1] + "…"


def plot_volcano(stats_df, out_path):
    pairs = [
        ("mal_ben", "malignant vs benign"),
        ("mal_no",  "malignant vs no_tumor"),
        ("ben_no",  "benign vs no_tumor"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=140)
    for ax, (key, ttl) in zip(axes, pairs):
        x = stats_df[f"cliff_delta_{key}"].values
        y = -np.log10(np.clip(stats_df[f"q_{key}"].values, 1e-300, 1.0))
        sig = stats_df[f"q_{key}"].values < 0.05
        ax.scatter(x[~sig], y[~sig], s=18, c="lightgray", alpha=0.6, label="ns")
        ax.scatter(x[sig], y[sig], s=22, c="crimson", alpha=0.85, label="q<0.05")
        ax.axhline(-math.log10(0.05), ls="--", color="gray", lw=0.8)
        ax.axvline(0, ls="--", color="gray", lw=0.8)
        # label top-10 by |delta| among sig
        sub = stats_df[sig].copy()
        sub["abs_delta"] = sub[f"cliff_delta_{key}"].abs()
        for _, r in sub.sort_values("abs_delta", ascending=False).head(8).iterrows():
            ax.annotate(short_feat(r["feature"]),
                        (r[f"cliff_delta_{key}"], -math.log10(max(r[f"q_{key}"], 1e-300))),
                        fontsize=7, alpha=0.85)
        ax.set_title(ttl)
        ax.set_xlabel("Cliff's delta")
        ax.set_ylabel("-log10(q)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.25)
    fig.suptitle("Volcano — Radiomics biomarkers (FDR-BH per pair)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top10_box(stats_df, df, X, feat_cols, out_path, key="q_anova"):
    col_to_idx = {c: i for i, c in enumerate(feat_cols)}
    top = stats_df.sort_values(key).head(10)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), dpi=140)
    y = df["true_class"].values
    for ax, (_, row) in zip(axes.flat, top.iterrows()):
        feat = row["feature"]
        if feat not in col_to_idx:
            continue
        vals = X[:, col_to_idx[feat]]
        data_by_class = [vals[y == c] for c in (0, 1, 2)]
        parts = ax.boxplot(data_by_class, labels=["mal", "ben", "no"],
                           patch_artist=True, showfliers=False, widths=0.55)
        for patch, c in zip(parts["boxes"], [0, 1, 2]):
            patch.set_facecolor(CLASS_COLORS[c])
            patch.set_alpha(0.55)
        for ci, (arr, c) in enumerate(zip(data_by_class, [0, 1, 2])):
            jitter = (np.random.RandomState(ci).rand(len(arr)) - 0.5) * 0.25
            ax.scatter(np.full_like(arr, ci + 1) + jitter, arr,
                       s=8, c=CLASS_COLORS[c], alpha=0.65,
                       edgecolors="black", linewidths=0.2)
        ax.set_title(short_feat(feat, 34), fontsize=9)
        ax.set_xlabel(f"q_anova={row['q_anova']:.2e}", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(f"Top-10 Biomarker Boxplots (by {key})", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top5_violin_bvn(stats_df, df, X, feat_cols, out_path):
    col_to_idx = {c: i for i, c in enumerate(feat_cols)}
    top = stats_df.sort_values("q_ben_no").head(5)
    fig, axes = plt.subplots(1, 5, figsize=(20, 5), dpi=140)
    y = df["true_class"].values
    for ax, (_, row) in zip(axes.flat, top.iterrows()):
        feat = row["feature"]
        if feat not in col_to_idx:
            continue
        vals = X[:, col_to_idx[feat]]
        ben = vals[y == 1]
        no = vals[y == 2]
        sns.violinplot(data=[ben, no], ax=ax, inner="quartile",
                       palette=[CLASS_COLORS[1], CLASS_COLORS[2]])
        ax.scatter(np.zeros(len(ben)) + (np.random.RandomState(1).rand(len(ben)) - 0.5) * 0.25,
                   ben, s=8, c="white", edgecolors=CLASS_COLORS[1], alpha=0.7)
        ax.scatter(np.ones(len(no)) + (np.random.RandomState(2).rand(len(no)) - 0.5) * 0.25,
                   no, s=8, c="white", edgecolors=CLASS_COLORS[2], alpha=0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["benign", "no_tumor"])
        ax.set_title(short_feat(feat, 34), fontsize=9)
        ax.set_xlabel(f"q={row['q_ben_no']:.2e}  δ={row['cliff_delta_ben_no']:.2f}",
                      fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Top-5 benign vs no_tumor biomarkers (violin + jitter)", y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top20_corr(stats_df, df, X, feat_cols, out_path):
    col_to_idx = {c: i for i, c in enumerate(feat_cols)}
    top20 = stats_df.sort_values("q_anova").head(20)["feature"].tolist()
    top20 = [f for f in top20 if f in col_to_idx]
    idx = [col_to_idx[f] for f in top20]
    sub = X[:, idx]
    corr = np.corrcoef(sub.T)
    fig, ax = plt.subplots(figsize=(11, 10), dpi=140)
    short_names = [short_feat(f, 26) for f in top20]
    sns.heatmap(corr, xticklabels=short_names, yticklabels=short_names,
                cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=False, ax=ax, cbar_kws={"label": "Pearson r"})
    ax.set_title("Top-20 Biomarker Pearson Correlation")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_class_radar(stats_df, df, X, feat_cols, out_path):
    col_to_idx = {c: i for i, c in enumerate(feat_cols)}
    top6 = stats_df.sort_values("q_anova").head(6)["feature"].tolist()
    top6 = [f for f in top6 if f in col_to_idx]
    idx = [col_to_idx[f] for f in top6]
    sub = X[:, idx]
    sub_std = (sub - sub.mean(axis=0)) / (sub.std(axis=0) + 1e-9)

    y = df["true_class"].values
    means = np.stack([sub_std[y == c].mean(axis=0) for c in (0, 1, 2)])

    theta = np.linspace(0, 2 * np.pi, len(top6), endpoint=False).tolist()
    theta += theta[:1]

    fig, ax = plt.subplots(figsize=(8, 8), dpi=140, subplot_kw=dict(polar=True))
    for c in (0, 1, 2):
        vals = means[c].tolist()
        vals += vals[:1]
        ax.plot(theta, vals, color=CLASS_COLORS[c],
                label=CLASS_NAMES[c], lw=2)
        ax.fill(theta, vals, color=CLASS_COLORS[c], alpha=0.12)
    ax.set_xticks(theta[:-1])
    ax.set_xticklabels([short_feat(f, 20) for f in top6], fontsize=8)
    ax.set_title("Top-6 Biomarker Radar (z-score means)", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────
# Phase 2.6 — gb vs lesion ROI
# ─────────────────────────────────────────────────────────────
def roi_compare(out_path):
    if not LESION_CSV.exists():
        return None
    df_gb, X_gb, feat_gb, _ = load_features(GB_CSV)
    df_le, X_le, feat_le, _ = load_features(LESION_CSV)

    # 限定到同样的 ben+no_tumor 子集 (lesion 本来就是)
    mask_gb_subset = df_gb["true_class"].isin([1, 2])
    df_gb_sub = df_gb[mask_gb_subset].reset_index(drop=True)
    X_gb_sub = X_gb[mask_gb_subset.values]

    def eval_roi(X, df, name):
        X_std = StandardScaler().fit_transform(X)
        y = df["true_class"].values
        # t-SNE 2D silhouette
        n = len(y)
        perp = max(5, min(30, (n - 1) // 3))
        Z = TSNE(n_components=2, perplexity=perp, learning_rate="auto",
                 init="pca", random_state=42).fit_transform(X_std)
        sil_tsne = float(silhouette_score(Z, y))
        sil_raw = float(silhouette_score(X_std, y))
        return dict(name=name, Z=Z, y=y, sil_tsne=sil_tsne,
                    sil_raw=sil_raw, n=n, d=X.shape[1])

    r_gb = eval_roi(X_gb_sub, df_gb_sub, "gallbladder ROI (ben+no)")
    r_le = eval_roi(X_le, df_le, "lesion ROI (ben+no)")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=140)
    for ax, r in zip(axes, [r_gb, r_le]):
        for c in (1, 2):
            idx = r["y"] == c
            ax.scatter(r["Z"][idx, 0], r["Z"][idx, 1], s=22, alpha=0.8,
                       c=CLASS_COLORS[c],
                       label=f"{CLASS_NAMES[c]} (n={int(idx.sum())})",
                       edgecolors="white", linewidths=0.5)
        ax.set_title(f"{r['name']}  (d={r['d']})\n"
                     f"sil(t-SNE)={r['sil_tsne']:.3f}  sil(raw)={r['sil_raw']:.3f}")
        ax.legend()
        ax.grid(alpha=0.2)
    fig.suptitle("Phase 2.6 — gallbladder ROI vs lesion ROI  (benign vs no_tumor)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

    return {"gb_sub": {k: v for k, v in r_gb.items() if k not in ("Z", "y")},
            "lesion":  {k: v for k, v in r_le.items() if k not in ("Z", "y")}}


# ─────────────────────────────────────────────────────────────
def main():
    RAD_DIR.mkdir(parents=True, exist_ok=True)

    df, X, feat_cols, bad_cols = load_features(GB_CSV)
    y = df["true_class"].values
    print(f"Loaded {len(df)} samples × {len(feat_cols)} features  "
          f"(dropped {len(bad_cols)} NaN/constant cols)")

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 2.3 降维
    idx80, idx90, cum = plot_pca_scree(X_std, RAD_DIR / "2.3_pca_scree.png")
    sil_pca, evr = run_pca_2d(X_std, y, CLASS_NAMES, RAD_DIR / "2.3_pca_2d.png")
    sil_tsne, _ = run_tsne_2d(X_std, y, CLASS_NAMES, RAD_DIR / "2.3_tsne.png")
    sil_umap, _ = run_umap_2d(X_std, y, CLASS_NAMES, RAD_DIR / "2.3_umap.png")
    ari = kmeans_ari(X_std, y, n_clusters=3)
    sil_raw = float(silhouette_score(X_std, y))
    print(f"sil(raw)={sil_raw:.3f}  sil(pca)={sil_pca:.3f}  "
          f"sil(tsne)={sil_tsne:.3f}  sil(umap)={sil_umap:.3f}  KMeans ARI={ari:.3f}")

    # 2.4 统计
    stats_df = per_feature_stats(df, X, feat_cols)
    stats_df.to_csv(RAD_DIR / "biomarker_table.csv", index=False)
    n_sig_anova = int((stats_df["q_anova"] < 0.05).sum())
    n_sig_bn = int((stats_df["q_ben_no"] < 0.05).sum())
    print(f"significant (q<0.05): anova={n_sig_anova}/{len(feat_cols)}  "
          f"ben_vs_no={n_sig_bn}")

    # 2.5 图
    plot_volcano(stats_df, RAD_DIR / "2.5_volcano_3panels.png")
    plot_top10_box(stats_df, df, X, feat_cols,
                   RAD_DIR / "2.5_top10_boxplot_grid.png")
    plot_top5_violin_bvn(stats_df, df, X, feat_cols,
                         RAD_DIR / "2.5_top5_violin_ben_vs_no.png")
    plot_top20_corr(stats_df, df, X, feat_cols,
                    RAD_DIR / "2.5_top20_corr_heatmap.png")
    plot_class_radar(stats_df, df, X, feat_cols,
                     RAD_DIR / "2.5_class_radar.png")

    # 2.6 gb vs lesion
    roi_cmp = roi_compare(RAD_DIR / "2.6_gb_vs_lesion_compare.png")

    # top biomarkers summary
    top_anova = stats_df.sort_values("q_anova").head(10)[
        ["feature", "f_stat", "p_anova", "q_anova", "cliff_delta_ben_no", "q_ben_no"]
    ].to_dict(orient="records")
    top_bn = stats_df.sort_values("q_ben_no").head(10)[
        ["feature", "p_ben_no", "q_ben_no", "cliff_delta_ben_no",
         "mean_ben", "mean_no"]
    ].to_dict(orient="records")

    report = {
        "csv_source": str(GB_CSV),
        "n_samples": int(len(df)),
        "n_features_total": int(len(feat_cols) + len(bad_cols)),
        "n_features_used": int(len(feat_cols)),
        "n_features_dropped_nan_or_const": int(len(bad_cols)),
        "dropped_cols_sample": bad_cols[:10],
        "class_counts": df["true_class_name"].value_counts().to_dict(),
        "pca": {
            "pc1_var": evr[0], "pc2_var": evr[1],
            "pc80_index": idx80, "pc90_index": idx90,
        },
        "silhouette": {
            "raw_standardized": sil_raw,
            "pca2d": sil_pca,
            "tsne2d": sil_tsne,
            "umap2d": sil_umap,
        },
        "kmeans3_ari": ari,
        "significance_q005": {
            "anova": n_sig_anova,
            "mal_vs_ben": int((stats_df["q_mal_ben"] < 0.05).sum()),
            "mal_vs_no":  int((stats_df["q_mal_no"]  < 0.05).sum()),
            "ben_vs_no":  n_sig_bn,
        },
        "top10_by_anova": top_anova,
        "top10_by_ben_vs_no": top_bn,
        "roi_compare": roi_cmp,
    }
    with (RAD_DIR / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
