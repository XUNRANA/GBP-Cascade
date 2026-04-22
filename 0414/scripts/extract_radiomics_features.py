#!/usr/bin/env python3
"""
Phase 2.2 — 从 test 集 456 张图提取 PyRadiomics 107 维经典影像组学特征.

输入:
  - 图像: /data1/ouyangxinglong/GBP-Cascade/0414dataset/<class>/<stem>.png
  - 标注: 同路径 <stem>.json (LabelMe 格式, 'gallbladder' rectangle + 可选 lesion polygon)
  - Phase 1 NPZ: 用来对齐 image_path / pred_class / ordinal_score

输出:
  - logs/feature_analysis_best_softmax_segcls3/radiomics/extractor_config.yaml
  - logs/feature_analysis_best_softmax_segcls3/radiomics/features_test_gb_roi.csv   (主 ROI = gallbladder)
  - logs/feature_analysis_best_softmax_segcls3/radiomics/features_test_lesion_roi.csv (次 ROI = lesion, 仅 benign+no_tumor)
  - logs/feature_analysis_best_softmax_segcls3/radiomics/extract_log.json

提取参数 (固定):
  binWidth=25, force2D=True, force2Ddimension=0, interpolator=sitkBSpline,
  normalize=True, normalizeScale=100, label=1
featureClass: firstorder / shape2D / glcm / glrlm / glszm / gldm / ngtdm
imageType:  Original only  (不加 wavelet/LoG)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import yaml
from PIL import Image, ImageDraw
from radiomics import featureextractor

logging.getLogger("radiomics").setLevel(logging.ERROR)

ROOT = Path("/data1/ouyangxinglong/GBP-Cascade/0414")
DATA_ROOT = Path("/data1/ouyangxinglong/GBP-Cascade/0414dataset")
TEST_XLSX = DATA_ROOT / "task_3class_test.xlsx"
PHASE1_NPZ = ROOT / "logs/feature_analysis_best_softmax_segcls3/deep/features_by_layer.npz"
OUT_DIR = ROOT / "logs/feature_analysis_best_softmax_segcls3/radiomics"

EXTRACTOR_SETTINGS = {
    "binWidth": 25,
    "force2D": True,
    "force2Ddimension": 0,
    "interpolator": "sitkBSpline",
    "normalize": True,
    "normalizeScale": 100,
    "label": 1,
    "minimumROIDimensions": 2,
}
FEATURE_CLASSES = ["firstorder", "shape2D", "glcm", "glrlm", "glszm", "gldm", "ngtdm"]


def build_extractor():
    ext = featureextractor.RadiomicsFeatureExtractor(**EXTRACTOR_SETTINGS)
    ext.disableAllFeatures()
    for fc in FEATURE_CLASSES:
        ext.enableFeatureClassByName(fc)
    ext.disableAllImageTypes()
    ext.enableImageTypeByName("Original")
    return ext


def load_annotation(json_path: Path):
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate_mask(shapes, width, height, *, roi: str):
    """roi: 'gallbladder' → label=='gallbladder'; 'lesion' → 其他标签的 polygon."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        label = shape.get("label")
        shape_type = shape.get("shape_type")
        points = shape.get("points", [])
        if roi == "gallbladder":
            if label != "gallbladder":
                continue
            if shape_type == "rectangle" and len(points) >= 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                draw.rectangle([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)], fill=1)
            elif shape_type == "polygon" and len(points) >= 3:
                draw.polygon([(p[0], p[1]) for p in points], fill=1)
        elif roi == "lesion":
            if label == "gallbladder":
                continue
            if shape_type == "polygon" and len(points) >= 3:
                draw.polygon([(p[0], p[1]) for p in points], fill=1)
    return np.array(mask, dtype=np.uint8)


def to_sitk_img_and_mask(img_np_gray: np.ndarray, mask_np: np.ndarray):
    """Wrap 2D uint8 arrays as 3D SITK images with z=1 for force2D."""
    img3d = img_np_gray[None, ...].astype(np.float32)  # (1, H, W)
    msk3d = mask_np[None, ...].astype(np.uint8)
    img_sitk = sitk.GetImageFromArray(img3d)
    msk_sitk = sitk.GetImageFromArray(msk3d)
    img_sitk.SetSpacing((1.0, 1.0, 1.0))
    msk_sitk.SetSpacing((1.0, 1.0, 1.0))
    return img_sitk, msk_sitk


def extract_one(extractor, img_path: Path, json_path: Path, roi: str):
    img_pil = Image.open(img_path).convert("L")
    W, H = img_pil.size
    img_np = np.array(img_pil, dtype=np.uint8)

    if not json_path.exists():
        return None, f"missing_json:{json_path.name}"
    ann = load_annotation(json_path)
    shapes = ann.get("shapes", [])
    mask_np = generate_mask(shapes, W, H, roi=roi)

    if mask_np.sum() < 16:
        return None, f"empty_mask:{roi}"

    img_sitk, msk_sitk = to_sitk_img_and_mask(img_np, mask_np)
    try:
        result = extractor.execute(img_sitk, msk_sitk)
    except Exception as exc:  # pyradiomics can raise on degenerate ROIs
        return None, f"extractor_error:{type(exc).__name__}:{exc}"

    feats = {}
    for k, v in result.items():
        if k.startswith("diagnostics_"):
            continue
        try:
            feats[k] = float(v)
        except (TypeError, ValueError):
            continue
    return feats, None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "extractor_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"settings": EXTRACTOR_SETTINGS, "feature_classes": FEATURE_CLASSES,
             "image_types": ["Original"]},
            f, sort_keys=False,
        )

    # load Phase 1 alignment
    npz = np.load(PHASE1_NPZ, allow_pickle=True)
    phase1_paths = [str(p) for p in npz["image_paths"]]
    phase1_cls_logits = npz["feat_cls_logits"]  # (N, 3)
    phase1_ord = npz["feat_ord_score"]
    phase1_labels = npz["labels"]
    preds = phase1_cls_logits.argmax(axis=1)
    path2info = {
        p: {"pred_class": int(preds[i]), "ordinal_score": float(phase1_ord[i]),
            "phase1_label": int(phase1_labels[i])}
        for i, p in enumerate(phase1_paths)
    }

    test_df = pd.read_excel(TEST_XLSX)
    class_names = ["malignant", "benign", "no_tumor"]

    extractor = build_extractor()
    rows_gb, rows_lesion = [], []
    errors = {"gb": [], "lesion": []}
    t0 = time.time()

    for i, row in test_df.iterrows():
        rel = row["image_path"]
        label = int(row["label"])
        img_path = DATA_ROOT / rel
        json_path = Path(str(img_path).replace(".png", ".json"))

        info = path2info.get(rel, {})
        assert info and info.get("phase1_label") == label, \
            f"phase1 alignment broken for {rel}"

        # gallbladder ROI  (all 3 classes)
        feats_gb, err = extract_one(extractor, img_path, json_path, roi="gallbladder")
        if feats_gb is None:
            errors["gb"].append({"image_path": rel, "reason": err})
        else:
            row_out = {
                "image_path": rel,
                "true_class": label,
                "true_class_name": class_names[label],
                "pred_class": info.get("pred_class", -1),
                "ordinal_score": info.get("ordinal_score", float("nan")),
                **feats_gb,
            }
            rows_gb.append(row_out)

        # lesion ROI  (only benign + no_tumor)
        if label in (1, 2):
            feats_le, err2 = extract_one(extractor, img_path, json_path, roi="lesion")
            if feats_le is None:
                errors["lesion"].append({"image_path": rel, "reason": err2})
            else:
                rows_lesion.append({
                    "image_path": rel,
                    "true_class": label,
                    "true_class_name": class_names[label],
                    "pred_class": info.get("pred_class", -1),
                    "ordinal_score": info.get("ordinal_score", float("nan")),
                    **feats_le,
                })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"[{i + 1}/{len(test_df)}] elapsed={elapsed:.1f}s  "
                  f"ok_gb={len(rows_gb)}  ok_lesion={len(rows_lesion)}  "
                  f"err_gb={len(errors['gb'])}  err_le={len(errors['lesion'])}",
                  flush=True)

    df_gb = pd.DataFrame(rows_gb)
    df_le = pd.DataFrame(rows_lesion)
    df_gb.to_csv(OUT_DIR / "features_test_gb_roi.csv", index=False)
    df_le.to_csv(OUT_DIR / "features_test_lesion_roi.csv", index=False)

    meta_cols = {"image_path", "true_class", "true_class_name", "pred_class", "ordinal_score"}
    feat_cols_gb = [c for c in df_gb.columns if c not in meta_cols]
    feat_cols_le = [c for c in df_le.columns if c not in meta_cols]

    summary = {
        "elapsed_sec": round(time.time() - t0, 2),
        "n_total_test": int(len(test_df)),
        "n_ok_gb": int(len(df_gb)),
        "n_ok_lesion": int(len(df_le)),
        "n_feat_cols_gb": len(feat_cols_gb),
        "n_feat_cols_lesion": len(feat_cols_le),
        "n_err_gb": len(errors["gb"]),
        "n_err_lesion": len(errors["lesion"]),
        "class_counts_gb": (
            df_gb["true_class_name"].value_counts().to_dict() if len(df_gb) else {}
        ),
        "class_counts_lesion": (
            df_le["true_class_name"].value_counts().to_dict() if len(df_le) else {}
        ),
        "errors_sample": {
            "gb": errors["gb"][:10],
            "lesion": errors["lesion"][:10],
        },
        "extractor_settings": EXTRACTOR_SETTINGS,
        "feature_classes": FEATURE_CLASSES,
    }
    with (OUT_DIR / "extract_log.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
