#!/usr/bin/env python3
"""
经典图像特征差异分析（3类）:
- 全图(img_) + 胆囊矩形ROI(gb_) 提取经典特征
- 统计检验: Kruskal(三组) + Mann-Whitney(两两) + FDR-BH + Cliff's delta
- 输出: CSV + 可视化 + 中文总结

输出目录:
  logs/dataset_stats_0414/classic_features/
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import font_manager
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

ROOT = Path("/data1/ouyangxinglong/GBP-Cascade/0414")
DATA_ROOT = Path("/data1/ouyangxinglong/GBP-Cascade/0414dataset")
OUT_DIR = ROOT / "logs/dataset_stats_0414/classic_features"

CLASSES = ["malignant", "benign", "no_tumor"]
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
CLASS_ZH = {"malignant": "恶性", "benign": "良性", "no_tumor": "无肿瘤"}
CLASS_COLOR = {"malignant": "#d73027", "benign": "#4575b4", "no_tumor": "#1a9850"}


def configure_chinese_plotting():
    candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Noto Sans CJK TC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Microsoft YaHei",
        "PingFang SC",
        "SimHei",
        "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen:
        mpl.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    else:
        mpl.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    mpl.rcParams["axes.unicode_minus"] = False


def fdr_bh(pvals: np.ndarray) -> np.ndarray:
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


def cliff_delta(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return float("nan")
    both = np.concatenate([a, b])
    ranks = stats.rankdata(both)
    ra = ranks[:na]
    u1 = ra.sum() - na * (na + 1) / 2.0
    u2 = na * nb - u1
    return float((u1 - u2) / (na * nb))


def entropy_u8(arr: np.ndarray) -> float:
    hist = np.bincount(arr.ravel(), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def high_freq_ratio(arr: np.ndarray) -> float:
    f = np.fft.fftshift(np.fft.fft2(arr.astype(np.float32)))
    power = np.abs(f) ** 2
    h, w = arr.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    r0 = min(h, w) * 0.12
    mask_high = r2 > (r0 * r0)
    den = float(power.sum())
    if den <= 1e-12:
        return 0.0
    return float(power[mask_high].sum() / den)


def glcm_features(arr: np.ndarray) -> dict:
    small = cv2.resize(arr, (128, 128), interpolation=cv2.INTER_AREA)
    q = (small // 8).astype(np.uint8)  # 32 灰度级
    glcm = graycomatrix(
        q,
        distances=[1, 2],
        angles=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=32,
        symmetric=True,
        normed=True,
    )
    out = {}
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]:
        out[prop] = float(graycoprops(glcm, prop).mean())
    return out


def lbp_features(arr: np.ndarray) -> dict:
    small = cv2.resize(arr, (128, 128), interpolation=cv2.INTER_AREA)
    P, R = 8, 1
    lbp = local_binary_pattern(small, P=P, R=R, method="uniform")
    bins = P + 2  # 10
    hist, _ = np.histogram(lbp, bins=np.arange(0, bins + 1), range=(0, bins))
    hist = hist.astype(np.float64)
    hist /= max(hist.sum(), 1.0)
    p = hist[hist > 0]
    return {
        "lbp_entropy": float(-(p * np.log2(p)).sum()) if p.size else 0.0,
        "lbp_uniform_ratio": float(hist[:-1].sum()),
        "lbp_nonuniform_ratio": float(hist[-1]),
    }


def extract_classic(arr: np.ndarray, prefix: str) -> dict:
    arr = np.asarray(arr, dtype=np.uint8)
    if arr.ndim != 2 or arr.size < 16:
        keys = [
            "mean",
            "std",
            "p10",
            "p50",
            "p90",
            "entropy",
            "skew",
            "kurtosis",
            "lap_var",
            "grad_mean",
            "grad_std",
            "edge_density",
            "hf_ratio",
            "glcm_contrast",
            "glcm_dissimilarity",
            "glcm_homogeneity",
            "glcm_energy",
            "glcm_correlation",
            "glcm_asm",
            "lbp_entropy",
            "lbp_uniform_ratio",
            "lbp_nonuniform_ratio",
        ]
        return {f"{prefix}_{k}": np.nan for k in keys}

    x = arr.astype(np.float64).ravel()
    feat = {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_p10": float(np.percentile(x, 10)),
        f"{prefix}_p50": float(np.percentile(x, 50)),
        f"{prefix}_p90": float(np.percentile(x, 90)),
        f"{prefix}_entropy": entropy_u8(arr),
        f"{prefix}_skew": float(stats.skew(x)) if x.size > 2 else np.nan,
        f"{prefix}_kurtosis": float(stats.kurtosis(x)) if x.size > 3 else np.nan,
    }

    lap = cv2.Laplacian(arr, cv2.CV_64F, ksize=3)
    feat[f"{prefix}_lap_var"] = float(lap.var())
    sx = cv2.Sobel(arr, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(arr, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(sx * sx + sy * sy)
    feat[f"{prefix}_grad_mean"] = float(np.mean(grad))
    feat[f"{prefix}_grad_std"] = float(np.std(grad))
    edges = cv2.Canny(arr, 50, 150)
    feat[f"{prefix}_edge_density"] = float(np.mean(edges > 0))
    feat[f"{prefix}_hf_ratio"] = high_freq_ratio(arr)

    gf = glcm_features(arr)
    feat[f"{prefix}_glcm_contrast"] = gf["contrast"]
    feat[f"{prefix}_glcm_dissimilarity"] = gf["dissimilarity"]
    feat[f"{prefix}_glcm_homogeneity"] = gf["homogeneity"]
    feat[f"{prefix}_glcm_energy"] = gf["energy"]
    feat[f"{prefix}_glcm_correlation"] = gf["correlation"]
    feat[f"{prefix}_glcm_asm"] = gf["ASM"]

    lf = lbp_features(arr)
    feat[f"{prefix}_lbp_entropy"] = lf["lbp_entropy"]
    feat[f"{prefix}_lbp_uniform_ratio"] = lf["lbp_uniform_ratio"]
    feat[f"{prefix}_lbp_nonuniform_ratio"] = lf["lbp_nonuniform_ratio"]
    return feat


def load_one(img_path: Path):
    ann_path = img_path.with_suffix(".json")
    if not ann_path.exists():
        return None
    ann = json.loads(ann_path.read_text(encoding="utf-8"))
    img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    h, w = img.shape

    rect = None
    for s in ann.get("shapes", []):
        if s.get("label") == "gallbladder" and s.get("shape_type") == "rectangle":
            pts = np.asarray(s.get("points", []), dtype=np.float64)
            if pts.shape[0] >= 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                xmin = int(np.floor(max(0.0, min(x1, x2))))
                xmax = int(np.ceil(min(float(w), max(x1, x2))))
                ymin = int(np.floor(max(0.0, min(y1, y2))))
                ymax = int(np.ceil(min(float(h), max(y1, y2))))
                if xmax > xmin and ymax > ymin:
                    rect = (xmin, ymin, xmax, ymax)
                    break
    if rect is None:
        crop = img
        gb_area_ratio = np.nan
    else:
        xmin, ymin, xmax, ymax = rect
        crop = img[ymin:ymax, xmin:xmax]
        gb_area_ratio = float((xmax - xmin) * (ymax - ymin) / (h * w))

    return img, crop, gb_area_ratio


def build_feature_table() -> pd.DataFrame:
    rows = []
    all_imgs = []
    for cls in CLASSES:
        all_imgs.extend(sorted((DATA_ROOT / cls).glob("*.png")))
    for i, p in enumerate(all_imgs, start=1):
        cls = p.parent.name
        loaded = load_one(p)
        if loaded is None:
            continue
        img, crop, gb_area_ratio = loaded
        rec = {
            "class_name": cls,
            "class_id": CLASS_ID[cls],
            "image_path": str(p.relative_to(DATA_ROOT)),
            "image_h": int(img.shape[0]),
            "image_w": int(img.shape[1]),
            "gb_area_ratio": gb_area_ratio,
        }
        rec.update(extract_classic(img, "img"))
        rec.update(extract_classic(crop, "gb"))
        rows.append(rec)
        if i % 200 == 0:
            print(f"[{i}/{len(all_imgs)}] processed", flush=True)
    return pd.DataFrame(rows)


def compute_stats(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for f in feat_cols:
        vals = [
            pd.to_numeric(df[df["class_name"] == c][f], errors="coerce").dropna().values
            for c in CLASSES
        ]
        if any(len(v) < 3 for v in vals):
            p_kw = 1.0
            h_kw = np.nan
        else:
            h_kw, p_kw = stats.kruskal(*vals)

        pair_map = [("malignant", "benign"), ("malignant", "no_tumor"), ("benign", "no_tumor")]
        pair_stat = {}
        for a, b in pair_map:
            xa = pd.to_numeric(df[df["class_name"] == a][f], errors="coerce").dropna().values
            xb = pd.to_numeric(df[df["class_name"] == b][f], errors="coerce").dropna().values
            if len(xa) < 3 or len(xb) < 3:
                p = 1.0
            else:
                _, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
            key = f"{a}_vs_{b}"
            pair_stat[f"p_{key}"] = float(p)
            pair_stat[f"delta_{key}"] = cliff_delta(xa, xb)
            pair_stat[f"mean_{a}"] = float(np.nanmean(xa)) if len(xa) else np.nan
            pair_stat[f"mean_{b}"] = float(np.nanmean(xb)) if len(xb) else np.nan

        rows.append({"feature": f, "kruskal_h": float(h_kw) if not np.isnan(h_kw) else np.nan, "p_kruskal": float(p_kw), **pair_stat})

    out = pd.DataFrame(rows)
    out["q_kruskal"] = fdr_bh(out["p_kruskal"].values)
    for pair in ["malignant_vs_benign", "malignant_vs_no_tumor", "benign_vs_no_tumor"]:
        out[f"q_{pair}"] = fdr_bh(out[f"p_{pair}"].values)
    out = out.sort_values("q_kruskal").reset_index(drop=True)
    return out


def zh_feat(name: str) -> str:
    rep = {
        "img_": "全图-",
        "gb_": "胆囊ROI-",
        "mean": "均值",
        "std": "标准差",
        "p10": "10分位",
        "p50": "50分位",
        "p90": "90分位",
        "entropy": "熵",
        "skew": "偏度",
        "kurtosis": "峰度",
        "lap_var": "拉普拉斯方差",
        "grad_mean": "梯度均值",
        "grad_std": "梯度标准差",
        "edge_density": "边缘密度",
        "hf_ratio": "高频能量比",
        "glcm_contrast": "GLCM对比度",
        "glcm_dissimilarity": "GLCM相异性",
        "glcm_homogeneity": "GLCM同质性",
        "glcm_energy": "GLCM能量",
        "glcm_correlation": "GLCM相关性",
        "glcm_asm": "GLCM二阶矩",
        "lbp_entropy": "LBP熵",
        "lbp_uniform_ratio": "LBP均匀模式比例",
        "lbp_nonuniform_ratio": "LBP非均匀模式比例",
        "gb_area_ratio": "胆囊框面积占比",
    }
    for k, v in rep.items():
        name = name.replace(k, v)
    return name


def plot_top8_box(df: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path):
    top = stats_df.head(8)["feature"].tolist()
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=140)
    for ax, feat in zip(axes.flat, top):
        vals = [pd.to_numeric(df[df["class_name"] == c][feat], errors="coerce").dropna().values for c in CLASSES]
        bp = ax.boxplot(vals, tick_labels=[CLASS_ZH[c] for c in CLASSES], showfliers=False, patch_artist=True)
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(CLASS_COLOR[c])
            patch.set_alpha(0.55)
        hit = stats_df[stats_df["feature"] == feat].iloc[0]
        ax.set_title(zh_feat(feat), fontsize=9)
        ax.set_xlabel(f"q={hit['q_kruskal']:.2e}", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("三分类差异最显著的前8个经典特征", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pair_volcano(stats_df: pd.DataFrame, pair: str, out_path: Path, title: str):
    pcol = f"q_{pair}"
    dcol = f"delta_{pair}"
    x = stats_df[dcol].values
    y = -np.log10(np.clip(stats_df[pcol].values, 1e-300, 1.0))
    sig = stats_df[pcol].values < 0.05
    fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=140)
    ax.scatter(x[~sig], y[~sig], s=16, c="lightgray", alpha=0.65, label="不显著")
    ax.scatter(x[sig], y[sig], s=18, c="crimson", alpha=0.85, label="显著(q<0.05)")
    ax.axvline(0, ls="--", c="gray", lw=0.8)
    ax.axhline(-math.log10(0.05), ls="--", c="gray", lw=0.8)
    sub = stats_df[sig].copy()
    sub["abs_delta"] = np.abs(sub[dcol].values)
    for _, r in sub.sort_values("abs_delta", ascending=False).head(10).iterrows():
        ax.annotate(zh_feat(r["feature"]), (r[dcol], -math.log10(max(r[pcol], 1e-300))), fontsize=7, alpha=0.85)
    ax.set_xlabel("效应量 Cliff's δ")
    ax.set_ylabel("-log10(q)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top10_heatmap(df: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path):
    top = stats_df.head(10)["feature"].tolist()
    mat = []
    for c in CLASSES:
        row = []
        for f in top:
            v = pd.to_numeric(df[df["class_name"] == c][f], errors="coerce").dropna().values
            row.append(float(np.nanmean(v)) if len(v) else np.nan)
        mat.append(row)
    mat = np.array(mat, dtype=np.float64)
    z = (mat - np.nanmean(mat, axis=0, keepdims=True)) / (np.nanstd(mat, axis=0, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(12, 4), dpi=140)
    sns.heatmap(
        z,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        xticklabels=[zh_feat(f) for f in top],
        yticklabels=[CLASS_ZH[c] for c in CLASSES],
        cbar_kws={"label": "按特征标准化后的类均值(z)"},
        ax=ax,
    )
    ax.set_title("前10个显著经典特征：三类别均值热图")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(df: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path):
    lines = []
    lines.append("# 0414dataset 经典特征差异总结")
    lines.append("")
    lines.append("## 1) 数据量")
    lines.append(f"- 总样本: {len(df)}")
    for c in CLASSES:
        lines.append(f"- {CLASS_ZH[c]}: {int((df['class_name']==c).sum())}")
    lines.append("")
    lines.append("## 2) 三分类总体差异（Kruskal + FDR）")
    sig3 = stats_df[stats_df["q_kruskal"] < 0.05]
    lines.append(f"- 显著特征数: {len(sig3)} / {len(stats_df)}")
    for _, r in sig3.head(12).iterrows():
        f = r["feature"]
        m0 = r.get("mean_malignant", np.nan)
        m1 = r.get("mean_benign", np.nan)
        m2 = r.get("mean_no_tumor", np.nan)
        lines.append(
            f"- {zh_feat(f)}: q={r['q_kruskal']:.2e}, 均值[恶/良/无]=[{m0:.3f}, {m1:.3f}, {m2:.3f}]"
        )
    lines.append("")
    lines.append("## 3) 经典可解释差异（重点）")
    lines.append("- 亮度相关（img_mean/gb_mean/img_p50）：无肿瘤整体偏亮，恶性偏暗。")
    lines.append("- 纹理相关（GLCM对比度、同质性、相关性）：类别间存在稳定差异，说明纹理组织模式不同。")
    lines.append("- 边缘与清晰度（edge_density、lap_var、grad_mean）：反映结构边界复杂度与局部细节强度。")
    lines.append("- 频域特征（hf_ratio）：反映高频细节占比，不同类别有统计差异时可作为补充判别信号。")
    lines.append("- LBP特征（lbp_uniform_ratio / lbp_entropy）：反映微纹理模式复杂度。")
    lines.append("")
    lines.append("## 4) 两两比较结论（q<0.05）")
    for pair, title in [
        ("malignant_vs_benign", "恶性 vs 良性"),
        ("malignant_vs_no_tumor", "恶性 vs 无肿瘤"),
        ("benign_vs_no_tumor", "良性 vs 无肿瘤"),
    ]:
        sig = stats_df[stats_df[f"q_{pair}"] < 0.05]
        lines.append(f"- {title}: 显著特征 {len(sig)} / {len(stats_df)}")
        for _, r in sig.reindex(sig[f"q_{pair}"].sort_values().index).head(6).iterrows():
            delta = r[f"delta_{pair}"]
            direction = "前者更大" if delta > 0 else "后者更大"
            lines.append(
                f"  - {zh_feat(r['feature'])}: q={r[f'q_{pair}']:.2e}, δ={delta:.2f}, {direction}"
            )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    configure_chinese_plotting()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] 提取经典特征...", flush=True)
    df = build_feature_table()
    df.to_csv(OUT_DIR / "classic_features_per_image.csv", index=False)

    feat_cols = [c for c in df.columns if c not in {"class_name", "class_id", "image_path", "image_h", "image_w"}]
    print(f"[2/4] 统计检验, n_feature={len(feat_cols)}", flush=True)
    stats_df = compute_stats(df, feat_cols)
    stats_df.to_csv(OUT_DIR / "classic_feature_stats_3class.csv", index=False)

    print("[3/4] 画图...", flush=True)
    plot_top8_box(df, stats_df, OUT_DIR / "01_top8_box_3class.png")
    plot_pair_volcano(stats_df, "malignant_vs_benign", OUT_DIR / "02_volcano_mal_vs_ben.png", "经典特征火山图：恶性 vs 良性")
    plot_pair_volcano(stats_df, "malignant_vs_no_tumor", OUT_DIR / "03_volcano_mal_vs_no.png", "经典特征火山图：恶性 vs 无肿瘤")
    plot_pair_volcano(stats_df, "benign_vs_no_tumor", OUT_DIR / "04_volcano_ben_vs_no.png", "经典特征火山图：良性 vs 无肿瘤")
    plot_top10_heatmap(df, stats_df, OUT_DIR / "05_top10_heatmap_3class.png")

    print("[4/4] 写总结...", flush=True)
    write_summary(df, stats_df, OUT_DIR / "SUMMARY_CLASSIC.md")

    result = {
        "n_samples": int(len(df)),
        "n_features": int(len(feat_cols)),
        "sig_kruskal_q05": int((stats_df["q_kruskal"] < 0.05).sum()),
        "sig_mal_vs_ben_q05": int((stats_df["q_malignant_vs_benign"] < 0.05).sum()),
        "sig_mal_vs_no_q05": int((stats_df["q_malignant_vs_no_tumor"] < 0.05).sum()),
        "sig_ben_vs_no_q05": int((stats_df["q_benign_vs_no_tumor"] < 0.05).sum()),
    }
    (OUT_DIR / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[out] {OUT_DIR}")


if __name__ == "__main__":
    from PIL import Image

    main()

