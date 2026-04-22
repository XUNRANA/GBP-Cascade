#!/usr/bin/env python3
"""Phase 1 — Deep feature multiview analysis for best 3-class model.

Outputs go to logs/feature_analysis_best_softmax_segcls3/deep/.
See 实验计划_特征可视化.md §3 for the plan.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="paper")

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
EXP_SCRIPT = SCRIPT_DIR / "20260414_task3_SwinV2Tiny_segcls_3.py"
OUT_ROOT = ROOT_DIR / "logs" / "feature_analysis_best_softmax_segcls3"
OUT_DIR = OUT_ROOT / "deep"
CACHE_NPZ = OUT_DIR / "features_by_layer.npz"

CLASS_COLORS = ["#d73027", "#4575b4", "#1a9850"]
TSNE_KWARGS = dict(n_components=2, perplexity=30, learning_rate="auto", init="pca",
                   random_state=42)


# ───────────────────────────────────────────────────────────────
#  Utilities
# ───────────────────────────────────────────────────────────────

def load_exp_module(exp_script: Path):
    spec = importlib.util.spec_from_file_location("exp3_module", str(exp_script))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def gap(x: torch.Tensor) -> torch.Tensor:
    """Global average pool NCHW → N,C."""
    return F.adaptive_avg_pool2d(x, 1).flatten(1)


def run_tsne(X: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    n = Xz.shape[0]
    perp = max(5, min(30, (n - 1) // 3))
    kw = dict(TSNE_KWARGS)
    kw["perplexity"] = perp
    return TSNE(**kw).fit_transform(Xz)


def run_pca(X: np.ndarray, n_components: int = 2):
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(Xz), pca.explained_variance_ratio_


def maybe_umap(X: np.ndarray):
    try:
        import umap  # type: ignore
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X)
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        return reducer.fit_transform(Xz), True
    except ImportError:
        return None, False


def silhouette_safe(X: np.ndarray, y: np.ndarray) -> float:
    try:
        return float(silhouette_score(X, y, metric="euclidean"))
    except Exception:
        return float("nan")


def kmeans_ari(X: np.ndarray, y: np.ndarray, k: int = 3) -> float:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    return float(adjusted_rand_score(y, km.labels_))


# ───────────────────────────────────────────────────────────────
#  Feature extraction with custom forward
# ───────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all_features(module):
    cfg = module.Config()
    cfg.num_workers = 4
    cfg.batch_size = 16
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_dataset, _, _, test_loader = module.build_dataloaders(cfg)

    model = module.SwinV2SegGuidedOrdinalMetaModel(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        pretrained=False,
    ).to(cfg.device)
    model.eval()

    ckpt = Path(cfg.best_weight_path)
    try:
        state = torch.load(str(ckpt), map_location=cfg.device, weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt), map_location=cfg.device)
    model.load_state_dict(state)

    store = {k: [] for k in [
        "f0", "f1", "f2", "f3",
        "cls_proj", "cls_feat_img", "cls_feat_320", "penult_128",
        "cls_logits", "ord_score",
    ]}
    all_labels, all_metas = [], []
    has_lesion = []

    for imgs, _masks, labels, has_masks, metas in test_loader:
        imgs = imgs.to(cfg.device, non_blocking=True)
        metas_t = metas.to(cfg.device, non_blocking=True)

        feats = model.encoder(imgs)
        f0, f1, f2, f3 = [model._to_bchw(f) for f in feats]

        d3 = model.dec3(f3, f2)
        d2 = model.dec2(d3, f1)
        d1 = model.dec1(d2, f0)
        seg_logits = model.seg_final(d1)

        seg_prob = F.softmax(seg_logits, dim=1)[:, 1:2]
        attn = F.interpolate(seg_prob, size=f2.shape[2:], mode="bilinear", align_corners=False)
        attn = attn + 0.1
        attn = attn / (attn.sum(dim=(2, 3), keepdim=True) + 1e-6)

        f2_proj = model.cls_proj(f2)
        cls_feat_img = (f2_proj * attn).sum(dim=(2, 3))  # (B, 256)
        meta_feat = model.meta_encoder(metas_t.float())
        cls_feat_320 = torch.cat([cls_feat_img, meta_feat], dim=1)  # (B, 320)

        # penultimate = Linear(320→128) + GELU
        h0 = model.cls_head[0](cls_feat_320)
        penult = model.cls_head[1](h0)
        cls_logits = model.cls_head(cls_feat_320)
        ord_score = torch.sigmoid(model.ord_head(cls_feat_320).squeeze(-1))

        store["f0"].append(gap(f0).cpu().numpy())
        store["f1"].append(gap(f1).cpu().numpy())
        store["f2"].append(gap(f2).cpu().numpy())
        store["f3"].append(gap(f3).cpu().numpy())
        store["cls_proj"].append(gap(f2_proj).cpu().numpy())
        store["cls_feat_img"].append(cls_feat_img.cpu().numpy())
        store["cls_feat_320"].append(cls_feat_320.cpu().numpy())
        store["penult_128"].append(penult.cpu().numpy())
        store["cls_logits"].append(cls_logits.cpu().numpy())
        store["ord_score"].append(ord_score.cpu().numpy())
        all_labels.append(labels.numpy())
        all_metas.append(metas.numpy())
        has_lesion.append(has_masks.numpy())

    feat_dict = {k: np.concatenate(v, axis=0) for k, v in store.items()}
    labels = np.concatenate(all_labels, axis=0)
    metas = np.concatenate(all_metas, axis=0)
    has_lesion_arr = np.concatenate(has_lesion, axis=0)

    # Raw meta bins from dataset.df
    df = test_dataset.df
    raw_size_bin = df.get("size_bin", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
    raw_flow_bin = df.get("flow_bin", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)
    raw_morph_bin = df.get("morph_bin", pd.Series([np.nan] * len(df))).to_numpy(dtype=float)

    return dict(
        features=feat_dict,
        labels=labels,
        metas=metas,
        raw_size_bin=raw_size_bin,
        raw_flow_bin=raw_flow_bin,
        raw_morph_bin=raw_morph_bin,
        has_lesion=has_lesion_arr,
        class_names=list(cfg.class_names),
        ckpt=str(ckpt),
        image_paths=df["image_path"].tolist(),
    )


# ───────────────────────────────────────────────────────────────
#  Phase 1.1  multilayer t-SNE grid
# ───────────────────────────────────────────────────────────────

LAYER_ORDER = [
    ("f0",            "f0  (96, GAP)"),
    ("f1",            "f1 (192, GAP)"),
    ("f2",            "f2 (384, GAP)"),
    ("f3",            "f3 (768, GAP)"),
    ("cls_proj",      "cls_proj (256, pre-attn)"),
    ("cls_feat_img",  "cls_feat_img (256, seg-attn)"),
    ("cls_feat_320",  "cls_feat_320 (fused)"),
    ("penult_128",    "penultimate (128)"),
]


def phase_1_1_multilayer(data, out_dir: Path):
    labels = data["labels"]
    names = data["class_names"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), dpi=140)
    stats_per_layer = {}

    for ax, (key, title) in zip(axes.flat, LAYER_ORDER):
        X = data["features"][key]
        emb = run_tsne(X)
        sil = silhouette_safe(X, labels)
        # inter-centroid in original space
        centroids = np.stack([X[labels == c].mean(0) for c in sorted(np.unique(labels))])
        bn = float(np.linalg.norm(centroids[1] - centroids[2]))  # benign=1, no_tumor=2
        for c in sorted(np.unique(labels)):
            idx = labels == c
            ax.scatter(emb[idx, 0], emb[idx, 1], s=14, alpha=0.7,
                       c=CLASS_COLORS[c], label=f"{names[c]} (n={idx.sum()})",
                       edgecolors="none")
        ax.set_title(f"{title}\nsilhouette={sil:.3f}, ben↔no L2={bn:.2f}", fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, frameon=True)
        stats_per_layer[key] = dict(silhouette=sil, bn_centroid_l2=bn, dim=int(X.shape[1]))

    fig.suptitle("Phase 1.1  Multilayer t-SNE of best 3-class model", fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "1.1_multilayer_tsne_grid.png", bbox_inches="tight")
    plt.close(fig)
    return stats_per_layer


# ───────────────────────────────────────────────────────────────
#  Phase 1.2  PCA / t-SNE / UMAP comparison on 320D
# ───────────────────────────────────────────────────────────────

def phase_1_2_projection(data, out_dir: Path):
    X = data["features"]["cls_feat_320"]
    labels = data["labels"]
    names = data["class_names"]

    pca2, var_ratio = run_pca(X, 2)
    tsne2 = run_tsne(X)
    umap2, ok = maybe_umap(X)

    ncols = 3 if ok else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6), dpi=140)

    def _draw(ax, emb, method: str, extra: str = ""):
        sil = silhouette_safe(emb, labels)
        ari = kmeans_ari(emb, labels, k=3)
        for c in sorted(np.unique(labels)):
            idx = labels == c
            ax.scatter(emb[idx, 0], emb[idx, 1], s=16, alpha=0.7,
                       c=CLASS_COLORS[c], label=f"{names[c]} (n={idx.sum()})",
                       edgecolors="none")
        ax.set_title(f"{method}{extra}\nsilhouette={sil:.3f} | KMeans ARI={ari:.3f}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=8)

    _draw(axes[0], pca2, "PCA", f" (var={var_ratio[0]+var_ratio[1]:.2f})")
    _draw(axes[1], tsne2, "t-SNE")
    if ok:
        _draw(axes[2], umap2, "UMAP")

    fig.suptitle("Phase 1.2  Projection comparison on 320D cls_feat", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "1.2_pca_tsne_umap.png", bbox_inches="tight")
    plt.close(fig)

    return dict(
        pca_var_ratio_pc12=float(var_ratio[0] + var_ratio[1]),
        sil_pca=silhouette_safe(pca2, labels),
        sil_tsne=silhouette_safe(tsne2, labels),
        sil_umap=silhouette_safe(umap2, labels) if ok else None,
        ari_tsne=kmeans_ari(tsne2, labels),
        umap_available=ok,
    ), tsne2


# ───────────────────────────────────────────────────────────────
#  Phase 1.3  9-attribute coloring grid on shared t-SNE
# ───────────────────────────────────────────────────────────────

def phase_1_3_attribute_grid(data, tsne2: np.ndarray, out_dir: Path):
    labels = data["labels"]
    names = data["class_names"]
    logits = data["features"]["cls_logits"]
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)
    entropy = -(probs * np.log(probs + 1e-12)).sum(axis=1)
    correct = (preds == labels).astype(int)
    ord_score = data["features"]["ord_score"]

    fig, axes = plt.subplots(3, 3, figsize=(18, 16), dpi=140)

    def _disc(ax, vals, title, cats_labels=None):
        uniq = sorted(np.unique(vals[~np.isnan(vals)]).tolist()) if np.issubdtype(vals.dtype, np.floating) else sorted(np.unique(vals).tolist())
        cmap = ListedColormap(CLASS_COLORS[: len(uniq)])
        for i, v in enumerate(uniq):
            idx = vals == v if not np.issubdtype(vals.dtype, np.floating) else np.isclose(vals, v)
            lab = cats_labels[int(v)] if cats_labels else str(int(v)) if float(v).is_integer() else f"{v:.2f}"
            ax.scatter(tsne2[idx, 0], tsne2[idx, 1], s=12, alpha=0.7,
                       c=[CLASS_COLORS[i % 3]], label=f"{lab} (n={idx.sum()})", edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=8)

    def _cont(ax, vals, title, cmap="viridis"):
        sc = ax.scatter(tsne2[:, 0], tsne2[:, 1], s=12, c=vals, alpha=0.8,
                        cmap=cmap, edgecolors="none")
        plt.colorbar(sc, ax=ax, fraction=0.046)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])

    _disc(axes[0, 0], labels.astype(int), "True class", names)
    _disc(axes[0, 1], preds.astype(int), "Predicted class", names)
    _disc(axes[0, 2], correct.astype(int), "Correct (1) vs Wrong (0)", ["wrong", "correct"])

    _cont(axes[1, 0], ord_score, "Ordinal score s (0~1)", "RdYlGn_r")
    _cont(axes[1, 1], max_prob, "Softmax max prob (confidence)", "viridis")
    _cont(axes[1, 2], entropy, "Predictive entropy", "magma")

    def _bin_attr(ax, vals, title):
        # NaN handling: missing values → grey
        mask_nan = np.isnan(vals)
        if (~mask_nan).sum() == 0:
            ax.set_title(title + " (all NaN)"); ax.set_xticks([]); ax.set_yticks([])
            return
        uniq = sorted(np.unique(vals[~mask_nan]).tolist())
        palette = sns.color_palette("tab10", n_colors=max(3, len(uniq)))
        for i, v in enumerate(uniq):
            idx = np.isclose(vals, v) & ~mask_nan
            lbl = f"{int(v)}" if float(v).is_integer() else f"{v:.2f}"
            ax.scatter(tsne2[idx, 0], tsne2[idx, 1], s=12, alpha=0.7,
                       c=[palette[i]], label=f"{lbl} (n={idx.sum()})", edgecolors="none")
        if mask_nan.any():
            ax.scatter(tsne2[mask_nan, 0], tsne2[mask_nan, 1], s=10, alpha=0.3,
                       c="lightgrey", label=f"NaN (n={mask_nan.sum()})", edgecolors="none")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=8)

    _bin_attr(axes[2, 0], data["raw_size_bin"], "size_bin (raw)")
    _bin_attr(axes[2, 1], data["raw_flow_bin"], "flow_bin (raw)")
    _bin_attr(axes[2, 2], data["raw_morph_bin"], "morph_bin (raw)")

    fig.suptitle("Phase 1.3  Multi-attribute coloring on shared 320D t-SNE",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "1.3_attribute_color_grid.png", bbox_inches="tight")
    plt.close(fig)

    return dict(preds=preds, probs=probs, max_prob=max_prob, entropy=entropy,
                correct=correct, ord_score=ord_score)


# ───────────────────────────────────────────────────────────────
#  Phase 1.4  distance distributions
# ───────────────────────────────────────────────────────────────

def phase_1_4_distance(data, out_dir: Path):
    X = data["features"]["cls_feat_320"]
    labels = data["labels"]
    names = data["class_names"]
    classes = sorted(np.unique(labels).tolist())

    centroids = {c: X[labels == c].mean(axis=0) for c in classes}

    intra_rows = []
    for c in classes:
        feats = X[labels == c]
        d = np.linalg.norm(feats - centroids[c], axis=1)
        for v in d:
            intra_rows.append(dict(cls=names[c], dist=float(v)))
    intra_df = pd.DataFrame(intra_rows)

    pair_rows = []
    rng = np.random.default_rng(42)
    for i in classes:
        xi = X[labels == i]
        for j in classes:
            xj = X[labels == j]
            # 避免 O(N^2), 每对采样 300 个对
            nn = min(300, len(xi) * len(xj))
            ia = rng.integers(0, len(xi), nn)
            ib = rng.integers(0, len(xj), nn)
            d = np.linalg.norm(xi[ia] - xj[ib], axis=1)
            key = f"{names[i][:3]}-{names[j][:3]}"
            for v in d:
                pair_rows.append(dict(pair=key, dist=float(v)))
    pair_df = pd.DataFrame(pair_rows)

    # centroid L2 matrix
    K = len(classes)
    cdist = np.zeros((K, K))
    for i in classes:
        for j in classes:
            cdist[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    # per-sample silhouette
    from sklearn.metrics import silhouette_samples
    sil_vals = silhouette_samples(X, labels)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=140)

    sns.violinplot(data=intra_df, x="cls", y="dist", hue="cls", ax=axes[0, 0],
                   palette=CLASS_COLORS, inner="quartile", legend=False)
    axes[0, 0].set_title("Intra-class L2 distance to centroid")
    axes[0, 0].set_xlabel(""); axes[0, 0].set_ylabel("L2 distance")

    order = [f"{names[i][:3]}-{names[j][:3]}" for i in classes for j in classes if i <= j]
    sns.boxplot(data=pair_df, x="pair", y="dist", ax=axes[0, 1],
                order=order, color="#87aade", showfliers=False)
    axes[0, 1].set_title("Pairwise L2 distance distribution (300 random pairs / group)")
    axes[0, 1].set_xlabel(""); axes[0, 1].set_ylabel("L2 distance")

    sns.heatmap(cdist, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=names, yticklabels=names, ax=axes[1, 0],
                cbar_kws=dict(label="L2"))
    axes[1, 0].set_title("Class-centroid L2 matrix (320D)")

    for c in classes:
        axes[1, 1].hist(sil_vals[labels == c], bins=30, alpha=0.55,
                        color=CLASS_COLORS[c], label=names[c])
    axes[1, 1].axvline(0, color="grey", lw=0.8)
    axes[1, 1].set_title(f"Per-sample silhouette (mean={sil_vals.mean():.3f})")
    axes[1, 1].set_xlabel("silhouette"); axes[1, 1].set_ylabel("count")
    axes[1, 1].legend()

    fig.suptitle("Phase 1.4  Distance / silhouette distributions (320D cls_feat)",
                 fontsize=14, y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "1.4_distance_distributions.png", bbox_inches="tight")
    plt.close(fig)

    return dict(
        intra_mean={names[c]: float(intra_df[intra_df.cls == names[c]].dist.mean())
                    for c in classes},
        centroid_matrix={f"{names[i]}_vs_{names[j]}": float(cdist[i, j])
                         for i in classes for j in classes if i < j},
        mean_silhouette=float(sil_vals.mean()),
    )


# ───────────────────────────────────────────────────────────────
#  Phase 1.5  discriminative dim analysis on 320D
# ───────────────────────────────────────────────────────────────

def phase_1_5_discriminative(data, out_dir: Path):
    X = data["features"]["cls_feat_320"]
    labels = data["labels"]
    names = data["class_names"]
    D = X.shape[1]

    # per-dim ANOVA F
    groups = [X[labels == c] for c in sorted(np.unique(labels))]
    f_stats = np.zeros(D); p_vals = np.zeros(D)
    for d in range(D):
        res = stats.f_oneway(*[g[:, d] for g in groups])
        f_stats[d] = res.statistic
        p_vals[d] = res.pvalue
    order = np.argsort(-f_stats)
    top30 = order[:30]
    top6 = order[:6]
    top20 = order[:20]

    # Bar top-30 F
    fig, ax = plt.subplots(figsize=(12, 5), dpi=140)
    ax.bar(range(30), f_stats[top30], color="#4575b4")
    ax.set_xticks(range(30))
    ax.set_xticklabels([str(int(i)) for i in top30], rotation=60, fontsize=8)
    ax.set_xlabel("Dim index (top-30 by F-stat)")
    ax.set_ylabel("ANOVA F-statistic")
    ax.set_title("Phase 1.5  Top-30 discriminative dims of 320D cls_feat")
    fig.tight_layout()
    fig.savefig(out_dir / "1.5_top_dims_bar.png", bbox_inches="tight")
    plt.close(fig)

    # Top-6 density
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=140)
    for ax, d in zip(axes.flat, top6):
        for c in sorted(np.unique(labels)):
            sns.kdeplot(X[labels == c, d], ax=ax, fill=True, alpha=0.35,
                        color=CLASS_COLORS[c], label=names[c])
        ax.set_title(f"dim {int(d)}  F={f_stats[d]:.1f}, p={p_vals[d]:.1e}", fontsize=10)
        ax.legend(fontsize=8)
        ax.set_ylabel("density")
    fig.suptitle("Phase 1.5  Top-6 dim class-conditional KDE", fontsize=13, y=1.00)
    fig.tight_layout()
    fig.savefig(out_dir / "1.5_top6_density_kde.png", bbox_inches="tight")
    plt.close(fig)

    # Top-20 correlation heatmap
    corr = np.corrcoef(X[:, top20].T)
    fig, ax = plt.subplots(figsize=(10, 9), dpi=140)
    sns.heatmap(corr, ax=ax, cmap="coolwarm", vmin=-1, vmax=1, square=True,
                xticklabels=[str(int(i)) for i in top20],
                yticklabels=[str(int(i)) for i in top20],
                cbar_kws=dict(label="Pearson r"))
    ax.set_title("Phase 1.5  Top-20 discriminative dim correlation")
    fig.tight_layout()
    fig.savefig(out_dir / "1.5_top20_corr_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    return dict(
        top30_dim=top30.tolist(),
        top30_F=f_stats[top30].tolist(),
        top30_p=p_vals[top30].tolist(),
        top20_mean_abs_corr=float(np.abs(corr - np.eye(len(top20))).mean()),
    )


# ───────────────────────────────────────────────────────────────
#  Phase 1.6  clinical gray-zone overlay
# ───────────────────────────────────────────────────────────────

def phase_1_6_gray_zone(data, tsne2, extras, out_dir: Path, t1=0.33, t2=0.66):
    labels = data["labels"]
    names = data["class_names"]
    preds = extras["preds"]; max_prob = extras["max_prob"]
    ord_score = extras["ord_score"]

    # malignant=0, no_tumor=2; miss = true malignant → predicted no_tumor
    mal_miss = (labels == 0) & (preds == 2)
    hi_conf_wrong = (max_prob > 0.9) & (preds != labels)

    fig, ax = plt.subplots(figsize=(11, 9), dpi=140)
    sc = ax.scatter(tsne2[:, 0], tsne2[:, 1], s=22, c=ord_score, cmap="RdYlGn_r",
                    alpha=0.8, edgecolors="none")
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046)
    cbar.set_label("ordinal score s (0 = low risk, 1 = high risk)")

    if mal_miss.any():
        ax.scatter(tsne2[mal_miss, 0], tsne2[mal_miss, 1], s=220, marker="*",
                   facecolor="none", edgecolor="black", linewidth=1.8,
                   label=f"mal→no_tumor miss (n={mal_miss.sum()})")
    if hi_conf_wrong.any():
        ax.scatter(tsne2[hi_conf_wrong, 0], tsne2[hi_conf_wrong, 1], s=90, marker="x",
                   c="black", linewidth=1.5,
                   label=f"high-conf wrong (n={hi_conf_wrong.sum()})")

    # add class label marker outlines
    for c in sorted(np.unique(labels)):
        idx = labels == c
        ax.scatter(tsne2[idx, 0], tsne2[idx, 1], s=24, facecolor="none",
                   edgecolor=CLASS_COLORS[c], linewidth=0.5,
                   label=f"{names[c]} (n={idx.sum()})")

    ax.set_title(f"Phase 1.6  Clinical gray-zone overlay (t1={t1}, t2={t2})")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "1.6_confusion_zone_overlay.png", bbox_inches="tight")
    plt.close(fig)

    return dict(
        mal_miss_count=int(mal_miss.sum()),
        hi_conf_wrong_count=int(hi_conf_wrong.sum()),
        t1=t1, t2=t2,
    )


# ───────────────────────────────────────────────────────────────
#  Main
# ───────────────────────────────────────────────────────────────

def main(use_cache: bool = True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    module = load_exp_module(EXP_SCRIPT)

    if use_cache and CACHE_NPZ.exists():
        cache = np.load(CACHE_NPZ, allow_pickle=True)
        data = dict(
            features={k: cache[f"feat_{k}"] for k in [
                "f0", "f1", "f2", "f3", "cls_proj", "cls_feat_img",
                "cls_feat_320", "penult_128", "cls_logits", "ord_score"]},
            labels=cache["labels"],
            metas=cache["metas"],
            raw_size_bin=cache["raw_size_bin"],
            raw_flow_bin=cache["raw_flow_bin"],
            raw_morph_bin=cache["raw_morph_bin"],
            has_lesion=cache["has_lesion"],
            class_names=cache["class_names"].tolist(),
            ckpt=str(cache["ckpt"]),
            image_paths=cache["image_paths"].tolist(),
        )
        print(f"[cache] Loaded from {CACHE_NPZ}")
    else:
        print("[extract] forward pass on test set ...")
        data = extract_all_features(module)
        to_save = {f"feat_{k}": v for k, v in data["features"].items()}
        to_save.update(dict(
            labels=data["labels"], metas=data["metas"],
            raw_size_bin=data["raw_size_bin"],
            raw_flow_bin=data["raw_flow_bin"],
            raw_morph_bin=data["raw_morph_bin"],
            has_lesion=data["has_lesion"],
            class_names=np.array(data["class_names"]),
            ckpt=np.array(data["ckpt"]),
            image_paths=np.array(data["image_paths"]),
        ))
        np.savez(CACHE_NPZ, **to_save)
        print(f"[cache] Saved to {CACHE_NPZ}")

    report = dict(ckpt=data["ckpt"], num_samples=int(len(data["labels"])),
                  class_names=data["class_names"])

    print("[1.1] multilayer t-SNE ...")
    report["phase_1_1_layer_stats"] = phase_1_1_multilayer(data, OUT_DIR)

    print("[1.2] PCA / t-SNE / UMAP comparison ...")
    r12, tsne320 = phase_1_2_projection(data, OUT_DIR)
    report["phase_1_2_projection"] = r12

    print("[1.3] attribute coloring grid ...")
    extras = phase_1_3_attribute_grid(data, tsne320, OUT_DIR)

    print("[1.4] distance distributions ...")
    report["phase_1_4_distance"] = phase_1_4_distance(data, OUT_DIR)

    print("[1.5] discriminative dim analysis ...")
    report["phase_1_5_discriminative"] = phase_1_5_discriminative(data, OUT_DIR)

    print("[1.6] clinical gray zone ...")
    report["phase_1_6_gray_zone"] = phase_1_6_gray_zone(
        data, tsne320, extras, OUT_DIR)

    # Sanity: 320D silhouette reproducing existing
    sil320 = silhouette_safe(data["features"]["cls_feat_320"], data["labels"])
    report["sanity_cls_feat_320_silhouette"] = sil320

    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] report → {OUT_DIR / 'report.json'}")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--no-cache", action="store_true",
                   help="force re-extract features")
    args = p.parse_args()
    main(use_cache=not args.no_cache)
