#!/usr/bin/env python3
"""
对 /0414dataset 做统计学特征分析：
1) 非矩形分割标签（polygon）最小外接矩形(MBR)特征
2) 图像级灰度统计特征（全图 + 胆囊矩形ROI）
3) benign vs no_tumor（二分类）与三分类差异检验

输出目录:
  logs/dataset_stats_0414/
    image_features.csv
    polygon_features.csv
    ben_vs_no_polygon_stats.csv
    ben_vs_no_image_stats.csv
    class3_image_stats.csv
    01_class_counts.png
    02_polygon_count_ben_vs_no.png
    03_top6_polygon_mbr_ben_vs_no.png
    04_top6_image_3class.png
    SUMMARY.md
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from matplotlib import font_manager
from PIL import Image
from scipy import stats

ROOT = Path("/data1/ouyangxinglong/GBP-Cascade/0414")
DATA_ROOT = Path("/data1/ouyangxinglong/GBP-Cascade/0414dataset")
OUT_DIR = ROOT / "logs/dataset_stats_0414"

CLASSES = ["malignant", "benign", "no_tumor"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
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


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def polygon_perimeter(points: np.ndarray) -> float:
    nxt = np.roll(points, -1, axis=0)
    return float(np.sqrt(((points - nxt) ** 2).sum(axis=1)).sum())


def gray_stats(arr: np.ndarray, prefix: str) -> dict:
    arr = arr.astype(np.float64).ravel()
    if arr.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_p10": np.nan,
            f"{prefix}_p50": np.nan,
            f"{prefix}_p90": np.nan,
            f"{prefix}_entropy": np.nan,
            f"{prefix}_skew": np.nan,
            f"{prefix}_kurtosis": np.nan,
            f"{prefix}_grad_mean": np.nan,
        }
    hist = np.bincount(arr.astype(np.uint8), minlength=256).astype(np.float64)
    p = hist / max(hist.sum(), 1.0)
    p = p[p > 0]
    entropy = float(-(p * np.log2(p)).sum())
    gx = np.diff(arr.reshape(-1, 1), axis=0)
    grad_mean = float(np.mean(np.abs(gx))) if gx.size else 0.0
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_min": float(np.min(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_p10": float(np.percentile(arr, 10)),
        f"{prefix}_p50": float(np.percentile(arr, 50)),
        f"{prefix}_p90": float(np.percentile(arr, 90)),
        f"{prefix}_entropy": entropy,
        f"{prefix}_skew": float(stats.skew(arr)) if arr.size > 2 else np.nan,
        f"{prefix}_kurtosis": float(stats.kurtosis(arr)) if arr.size > 3 else np.nan,
        f"{prefix}_grad_mean": grad_mean,
    }


def parse_one_image(cls_name: str, img_path: Path):
    json_path = img_path.with_suffix(".json")
    if not json_path.exists():
        return None, []
    ann = json.loads(json_path.read_text(encoding="utf-8"))
    shapes = ann.get("shapes", [])

    img = np.array(Image.open(img_path).convert("L"), dtype=np.uint8)
    h, w = img.shape
    img_area = float(h * w)
    rec = {
        "class_name": cls_name,
        "class_id": CLASS_TO_ID[cls_name],
        "image_path": str(img_path.relative_to(DATA_ROOT)),
        "image_h": h,
        "image_w": w,
    }
    rec.update(gray_stats(img, "img"))

    gb_rect = None
    for s in shapes:
        if s.get("label") == "gallbladder" and s.get("shape_type") == "rectangle":
            pts = np.asarray(s.get("points", []), dtype=np.float64)
            if pts.shape[0] >= 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                xmin = float(max(0.0, min(x1, x2)))
                xmax = float(min(w, max(x1, x2)))
                ymin = float(max(0.0, min(y1, y2)))
                ymax = float(min(h, max(y1, y2)))
                gb_rect = (xmin, ymin, xmax, ymax)
                break

    if gb_rect is not None:
        xmin, ymin, xmax, ymax = gb_rect
        gb_w = max(0.0, xmax - xmin)
        gb_h = max(0.0, ymax - ymin)
        gb_area = gb_w * gb_h
        rec["gb_rect_w"] = gb_w
        rec["gb_rect_h"] = gb_h
        rec["gb_rect_area"] = gb_area
        rec["gb_rect_aspect"] = gb_w / gb_h if gb_h > 0 else np.nan
        rec["gb_rect_area_ratio_img"] = gb_area / img_area if img_area > 0 else np.nan
        xs0 = int(np.floor(xmin))
        ys0 = int(np.floor(ymin))
        xs1 = int(np.ceil(xmax))
        ys1 = int(np.ceil(ymax))
        crop = img[ys0:ys1, xs0:xs1]
        rec.update(gray_stats(crop, "gb"))
    else:
        rec["gb_rect_w"] = np.nan
        rec["gb_rect_h"] = np.nan
        rec["gb_rect_area"] = np.nan
        rec["gb_rect_aspect"] = np.nan
        rec["gb_rect_area_ratio_img"] = np.nan
        rec.update(gray_stats(np.array([], dtype=np.uint8), "gb"))

    poly_rows = []
    for s in shapes:
        if s.get("shape_type") != "polygon":
            continue
        if s.get("label") == "gallbladder":
            continue
        pts = np.asarray(s.get("points", []), dtype=np.float64)
        if pts.shape[0] < 3:
            continue
        xs = pts[:, 0]
        ys = pts[:, 1]
        xmin = float(np.clip(xs.min(), 0, w))
        xmax = float(np.clip(xs.max(), 0, w))
        ymin = float(np.clip(ys.min(), 0, h))
        ymax = float(np.clip(ys.max(), 0, h))
        aabb_w = max(0.0, xmax - xmin)
        aabb_h = max(0.0, ymax - ymin)
        aabb_area = aabb_w * aabb_h
        area = polygon_area(pts)
        peri = polygon_perimeter(pts)
        compact = float(4 * math.pi * area / (peri * peri)) if peri > 1e-6 else np.nan
        rect = cv2.minAreaRect(pts.astype(np.float32))
        (rcx, rcy), (rw, rh), rangle = rect
        rw = float(abs(rw))
        rh = float(abs(rh))
        mbr_area = float(rw * rh)
        short = min(rw, rh)
        long_ = max(rw, rh)
        mbr_aspect = (long_ / short) if short > 0 else np.nan
        gb_area = rec.get("gb_rect_area", np.nan)
        poly_rows.append(
            {
                "class_name": cls_name,
                "class_id": CLASS_TO_ID[cls_name],
                "image_path": rec["image_path"],
                "poly_label": str(s.get("label", "")),
                "n_points": int(pts.shape[0]),
                "poly_area": area,
                "poly_perimeter": peri,
                "poly_compactness": compact,
                "aabb_xmin": xmin,
                "aabb_ymin": ymin,
                "aabb_xmax": xmax,
                "aabb_ymax": ymax,
                "aabb_w": aabb_w,
                "aabb_h": aabb_h,
                "aabb_area": aabb_area,
                "aabb_aspect": (aabb_w / aabb_h) if aabb_h > 0 else np.nan,
                "mbr_center_x": float(rcx),
                "mbr_center_y": float(rcy),
                "mbr_w": rw,
                "mbr_h": rh,
                "mbr_area": mbr_area,
                "mbr_aspect": mbr_aspect,
                "mbr_angle": float(rangle),
                "mbr_fill_ratio": (area / mbr_area) if mbr_area > 0 else np.nan,
                "mbr_center_x_norm": float(rcx) / w,
                "mbr_center_y_norm": float(rcy) / h,
                "poly_area_ratio_img": area / img_area if img_area > 0 else np.nan,
                "mbr_area_ratio_img": mbr_area / img_area if img_area > 0 else np.nan,
                "poly_area_ratio_gb": (area / gb_area) if gb_area and gb_area > 0 else np.nan,
                "mbr_area_ratio_gb": (mbr_area / gb_area) if gb_area and gb_area > 0 else np.nan,
            }
        )

    if len(poly_rows) == 0:
        rec["poly_count"] = 0
        rec["poly_area_total"] = 0.0
        rec["poly_area_mean"] = np.nan
        rec["poly_area_max"] = np.nan
        rec["poly_area_total_ratio_gb"] = 0.0
        rec["poly_mbr_area_mean"] = np.nan
        rec["poly_mbr_aspect_mean"] = np.nan
        rec["poly_mbr_fill_mean"] = np.nan
    else:
        areas = np.array([r["poly_area"] for r in poly_rows], dtype=np.float64)
        mbr_areas = np.array([r["mbr_area"] for r in poly_rows], dtype=np.float64)
        aspects = np.array([r["mbr_aspect"] for r in poly_rows], dtype=np.float64)
        fills = np.array([r["mbr_fill_ratio"] for r in poly_rows], dtype=np.float64)
        rec["poly_count"] = int(len(poly_rows))
        rec["poly_area_total"] = float(np.nansum(areas))
        rec["poly_area_mean"] = float(np.nanmean(areas))
        rec["poly_area_max"] = float(np.nanmax(areas))
        gb_area = rec.get("gb_rect_area", np.nan)
        rec["poly_area_total_ratio_gb"] = (
            float(np.nansum(areas) / gb_area) if gb_area and gb_area > 0 else np.nan
        )
        rec["poly_mbr_area_mean"] = float(np.nanmean(mbr_areas))
        rec["poly_mbr_aspect_mean"] = float(np.nanmean(aspects))
        rec["poly_mbr_fill_mean"] = float(np.nanmean(fills))

    return rec, poly_rows


def ben_vs_no_stats(df: pd.DataFrame, features: list[str], out_csv: Path) -> pd.DataFrame:
    a = df[df["class_name"] == "benign"]
    b = df[df["class_name"] == "no_tumor"]
    rows = []
    for f in features:
        xa = pd.to_numeric(a[f], errors="coerce").dropna().values
        xb = pd.to_numeric(b[f], errors="coerce").dropna().values
        if len(xa) < 3 or len(xb) < 3:
            p = 1.0
            u = np.nan
        else:
            u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        rows.append(
            {
                "feature": f,
                "n_benign": int(len(xa)),
                "n_no_tumor": int(len(xb)),
                "mean_benign": float(np.nanmean(xa)) if len(xa) else np.nan,
                "mean_no_tumor": float(np.nanmean(xb)) if len(xb) else np.nan,
                "median_benign": float(np.nanmedian(xa)) if len(xa) else np.nan,
                "median_no_tumor": float(np.nanmedian(xb)) if len(xb) else np.nan,
                "mw_u": float(u) if not np.isnan(u) else np.nan,
                "p_value": float(p),
                "cliff_delta": cliff_delta(xa, xb),
            }
        )
    out = pd.DataFrame(rows)
    out["q_value"] = fdr_bh(out["p_value"].values)
    out = out.sort_values(["q_value", "p_value"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out


def class3_stats(df: pd.DataFrame, features: list[str], out_csv: Path) -> pd.DataFrame:
    rows = []
    for f in features:
        vals = [pd.to_numeric(df[df["class_name"] == c][f], errors="coerce").dropna().values for c in CLASSES]
        if any(len(v) < 3 for v in vals):
            p_kw = 1.0
            kw_h = np.nan
            p_an = 1.0
            f_stat = np.nan
        else:
            kw_h, p_kw = stats.kruskal(*vals)
            f_stat, p_an = stats.f_oneway(*vals)
        rows.append(
            {
                "feature": f,
                "n_malignant": int(len(vals[0])),
                "n_benign": int(len(vals[1])),
                "n_no_tumor": int(len(vals[2])),
                "mean_malignant": float(np.nanmean(vals[0])) if len(vals[0]) else np.nan,
                "mean_benign": float(np.nanmean(vals[1])) if len(vals[1]) else np.nan,
                "mean_no_tumor": float(np.nanmean(vals[2])) if len(vals[2]) else np.nan,
                "kruskal_h": float(kw_h) if not np.isnan(kw_h) else np.nan,
                "p_kruskal": float(p_kw),
                "anova_f": float(f_stat) if not np.isnan(f_stat) else np.nan,
                "p_anova": float(p_an),
            }
        )
    out = pd.DataFrame(rows)
    out["q_kruskal"] = fdr_bh(out["p_kruskal"].values)
    out = out.sort_values(["q_kruskal", "p_kruskal"]).reset_index(drop=True)
    out.to_csv(out_csv, index=False)
    return out


def plot_class_counts(df_img: pd.DataFrame, out_path: Path):
    counts = df_img["class_name"].value_counts().reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
    ax.bar([CLASS_ZH[c] for c in CLASSES], counts.values, color=[CLASS_COLOR[c] for c in CLASSES], alpha=0.85)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=10)
    ax.set_title("数据集类别样本数")
    ax.set_ylabel("样本数")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_poly_count(df_img: pd.DataFrame, out_path: Path):
    sub = df_img[df_img["class_name"].isin(["benign", "no_tumor"])].copy()
    fig, ax = plt.subplots(figsize=(6.5, 4.2), dpi=140)
    data = [
        pd.to_numeric(sub[sub["class_name"] == "benign"]["poly_count"], errors="coerce").dropna().values,
        pd.to_numeric(sub[sub["class_name"] == "no_tumor"]["poly_count"], errors="coerce").dropna().values,
    ]
    bp = ax.boxplot(data, tick_labels=["良性", "无肿瘤"], patch_artist=True, showfliers=False)
    for patch, c in zip(bp["boxes"], ["benign", "no_tumor"]):
        patch.set_facecolor(CLASS_COLOR[c])
        patch.set_alpha(0.5)
    ax.set_title("每张图的多边形分割数量（良性 vs 无肿瘤）")
    ax.set_ylabel("polygon 数")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top6_two_group(df: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path, title: str):
    top = stats_df.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), dpi=140)
    sub = df[df["class_name"].isin(["benign", "no_tumor"])].copy()
    for ax, feat in zip(axes.flat, top):
        ben = pd.to_numeric(sub[sub["class_name"] == "benign"][feat], errors="coerce").dropna().values
        no = pd.to_numeric(sub[sub["class_name"] == "no_tumor"][feat], errors="coerce").dropna().values
        bp = ax.boxplot([ben, no], tick_labels=["良性", "无肿瘤"], patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], ["benign", "no_tumor"]):
            patch.set_facecolor(CLASS_COLOR[c])
            patch.set_alpha(0.55)
        ax.set_title(feat, fontsize=9)
        hit = stats_df[stats_df["feature"] == feat].iloc[0]
        ax.set_xlabel(f"q={hit['q_value']:.2e}, δ={hit['cliff_delta']:.2f}", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_top6_three_class(df: pd.DataFrame, stats_df: pd.DataFrame, out_path: Path, title: str):
    top = stats_df.head(6)["feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.5), dpi=140)
    for ax, feat in zip(axes.flat, top):
        vals = [
            pd.to_numeric(df[df["class_name"] == c][feat], errors="coerce").dropna().values
            for c in CLASSES
        ]
        bp = ax.boxplot(vals, tick_labels=[CLASS_ZH[c] for c in CLASSES], patch_artist=True, showfliers=False)
        for patch, c in zip(bp["boxes"], CLASSES):
            patch.set_facecolor(CLASS_COLOR[c])
            patch.set_alpha(0.55)
        ax.set_title(feat, fontsize=9)
        hit = stats_df[stats_df["feature"] == feat].iloc[0]
        ax.set_xlabel(f"q={hit['q_kruskal']:.2e}", fontsize=8)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    df_img: pd.DataFrame,
    df_poly: pd.DataFrame,
    ben_poly_stats: pd.DataFrame,
    ben_img_stats: pd.DataFrame,
    cls3_stats_df: pd.DataFrame,
    out_path: Path,
):
    lines = []
    lines.append("# 0414dataset 统计学特征分析汇总")
    lines.append("")
    lines.append("## 1) 数据规模")
    for c in CLASSES:
        n = int((df_img["class_name"] == c).sum())
        lines.append(f"- {CLASS_ZH[c]} ({c}): {n} 张")
    lines.append(f"- 非矩形 polygon 总数: {len(df_poly)}")
    lines.append(
        f"- 含 polygon 图像数: {int((df_img['poly_count'] > 0).sum())} / {len(df_img)}"
    )
    lines.append("")
    lines.append("## 2) 良性 vs 无肿瘤：polygon 最小外接矩形/形状差异")
    sig_poly = ben_poly_stats[ben_poly_stats["q_value"] < 0.05]
    if len(sig_poly) == 0:
        lines.append("- 未发现 q<0.05 的显著差异特征。")
    else:
        lines.append(f"- 显著特征数 (q<0.05): {len(sig_poly)} / {len(ben_poly_stats)}")
        for _, r in sig_poly.head(8).iterrows():
            direction = "良性 > 无肿瘤" if r["cliff_delta"] > 0 else "无肿瘤 > 良性"
            lines.append(
                f"- {r['feature']}: q={r['q_value']:.2e}, δ={r['cliff_delta']:.2f}, {direction}"
            )
    lines.append("")
    lines.append("## 3) 良性 vs 无肿瘤：图像级统计差异")
    sig_img = ben_img_stats[ben_img_stats["q_value"] < 0.05]
    lines.append(f"- 显著特征数 (q<0.05): {len(sig_img)} / {len(ben_img_stats)}")
    for _, r in sig_img.head(8).iterrows():
        direction = "良性 > 无肿瘤" if r["cliff_delta"] > 0 else "无肿瘤 > 良性"
        lines.append(f"- {r['feature']}: q={r['q_value']:.2e}, δ={r['cliff_delta']:.2f}, {direction}")
    lines.append("")
    lines.append("## 4) 三分类：共有图像特征差异（Kruskal）")
    sig3 = cls3_stats_df[cls3_stats_df["q_kruskal"] < 0.05]
    lines.append(f"- 显著特征数 (q<0.05): {len(sig3)} / {len(cls3_stats_df)}")
    for _, r in sig3.head(10).iterrows():
        lines.append(
            f"- {r['feature']}: q={r['q_kruskal']:.2e}, 均值[恶/良/无]="
            f"[{r['mean_malignant']:.2f}, {r['mean_benign']:.2f}, {r['mean_no_tumor']:.2f}]"
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    configure_chinese_plotting()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_rows = []
    poly_rows_all = []
    for cls in CLASSES:
        for img_path in sorted((DATA_ROOT / cls).glob("*.png")):
            rec, poly_rows = parse_one_image(cls, img_path)
            if rec is None:
                continue
            image_rows.append(rec)
            poly_rows_all.extend(poly_rows)

    df_img = pd.DataFrame(image_rows)
    df_poly = pd.DataFrame(poly_rows_all)

    df_img.to_csv(OUT_DIR / "image_features.csv", index=False)
    df_poly.to_csv(OUT_DIR / "polygon_features.csv", index=False)

    poly_features = [
        "n_points",
        "poly_area",
        "poly_perimeter",
        "poly_compactness",
        "mbr_w",
        "mbr_h",
        "mbr_area",
        "mbr_aspect",
        "mbr_fill_ratio",
        "mbr_center_x_norm",
        "mbr_center_y_norm",
        "poly_area_ratio_img",
        "mbr_area_ratio_img",
        "poly_area_ratio_gb",
        "mbr_area_ratio_gb",
    ]
    ben_poly_stats = ben_vs_no_stats(
        df_poly[df_poly["class_name"].isin(["benign", "no_tumor"])].copy(),
        poly_features,
        OUT_DIR / "ben_vs_no_polygon_stats.csv",
    )

    ben_img_features = [
        "poly_count",
        "poly_area_total",
        "poly_area_mean",
        "poly_area_max",
        "poly_area_total_ratio_gb",
        "poly_mbr_area_mean",
        "poly_mbr_aspect_mean",
        "poly_mbr_fill_mean",
        "img_mean",
        "img_std",
        "img_entropy",
        "img_skew",
        "img_kurtosis",
        "gb_mean",
        "gb_std",
        "gb_entropy",
        "gb_rect_area_ratio_img",
    ]
    ben_img_stats = ben_vs_no_stats(
        df_img[df_img["class_name"].isin(["benign", "no_tumor"])].copy(),
        ben_img_features,
        OUT_DIR / "ben_vs_no_image_stats.csv",
    )

    class3_features = [
        "img_mean",
        "img_std",
        "img_p10",
        "img_p50",
        "img_p90",
        "img_entropy",
        "img_skew",
        "img_kurtosis",
        "gb_mean",
        "gb_std",
        "gb_p10",
        "gb_p50",
        "gb_p90",
        "gb_entropy",
        "gb_rect_w",
        "gb_rect_h",
        "gb_rect_area",
        "gb_rect_aspect",
        "gb_rect_area_ratio_img",
    ]
    cls3 = class3_stats(df_img, class3_features, OUT_DIR / "class3_image_stats.csv")

    plot_class_counts(df_img, OUT_DIR / "01_class_counts.png")
    plot_poly_count(df_img, OUT_DIR / "02_polygon_count_ben_vs_no.png")
    plot_top6_two_group(
        df_poly[df_poly["class_name"].isin(["benign", "no_tumor"])].copy(),
        ben_poly_stats,
        OUT_DIR / "03_top6_polygon_mbr_ben_vs_no.png",
        "良性 vs 无肿瘤：前6个 polygon/MBR 显著差异特征",
    )
    plot_top6_three_class(
        df_img.copy(),
        cls3,
        OUT_DIR / "04_top6_image_3class.png",
        "三分类：前6个图像统计显著差异特征",
    )

    write_summary(
        df_img=df_img,
        df_poly=df_poly,
        ben_poly_stats=ben_poly_stats,
        ben_img_stats=ben_img_stats,
        cls3_stats_df=cls3,
        out_path=OUT_DIR / "SUMMARY.md",
    )

    print(f"[done] image_rows={len(df_img)} polygon_rows={len(df_poly)}")
    print(f"[out] {OUT_DIR}")
    print(
        json.dumps(
            {
                "class_counts": df_img["class_name"].value_counts().to_dict(),
                "polygon_counts": df_poly["class_name"].value_counts().to_dict(),
                "ben_vs_no_polygon_sig_q05": int((ben_poly_stats["q_value"] < 0.05).sum()),
                "ben_vs_no_image_sig_q05": int((ben_img_stats["q_value"] < 0.05).sum()),
                "class3_image_sig_q05": int((cls3["q_kruskal"] < 0.05).sum()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
