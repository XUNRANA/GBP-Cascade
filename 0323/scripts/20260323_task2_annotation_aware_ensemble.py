"""
方案五: 标注类型感知特征 + 多mask通道 + 高级集成
================================================
关键发现: benign 和 no_tumor 都有病灶标注, 但 adenoma 标注几乎只在 benign 出现!
1. 标注类型特征 (has_adenoma, num_polyps, annotation分布等)
2. 各类型病灶几何特征 (面积、形状、位置)
3. 深度特征 (已训练的 SwinV2 backbones)
4. XGBoost / LightGBM 集成
"""

import importlib
import json
import math
import os
import sys
import warnings

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from task2_json_utils import (
    adapt_model_to_4ch,
    load_annotation,
    setup_logger,
)

feat_xgb = importlib.import_module("20260323_task2_feature_xgboost")
find_best_threshold = feat_xgb.find_best_threshold
evaluate_predictions = feat_xgb.evaluate_predictions

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "0322dataset")
TRAIN_EXCEL = os.path.join(DATA_ROOT, "task_2_train.xlsx")
TEST_EXCEL = os.path.join(DATA_ROOT, "task_2_test.xlsx")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "annotation_aware_ensemble")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 123, 456, 789, 1024]

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════
#  标注类型感知特征
# ═════════════════════════════════════════════

ANNOTATION_TYPES = {
    "adenoma": ["gallbladder  adenoma", "gallbladder adenoma"],
    "tubular_adenoma": ["gallbladder tubular adenoma"],
    "polyp": ["gallbladder polyp"],
    "pred": ["pred"],
}


def parse_annotations(json_path):
    """解析 JSON 标注, 返回分类型的 shapes."""
    result = {
        "gallbladder": None,
        "adenoma": [],
        "tubular_adenoma": [],
        "polyp": [],
        "pred": [],
        "other": [],
    }

    if not os.path.exists(json_path):
        return result

    ann = load_annotation(json_path)
    for s in ann.get("shapes", []):
        lbl = s.get("label", "").strip().lower()

        if lbl == "gallbladder":
            result["gallbladder"] = s
        elif lbl in ("gallbladder  adenoma", "gallbladder adenoma"):
            result["adenoma"].append(s)
        elif lbl == "gallbladder tubular adenoma":
            result["tubular_adenoma"].append(s)
        elif lbl == "gallbladder polyp":
            result["polyp"].append(s)
        elif lbl == "pred":
            result["pred"].append(s)
        else:
            result["other"].append(s)

    return result


def polygon_area(points):
    """计算多边形面积 (Shoelace formula)."""
    pts = np.array(points)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygon_perimeter(points):
    """计算多边形周长."""
    pts = np.array(points)
    if len(pts) < 2:
        return 0.0
    diff = np.diff(pts, axis=0, append=pts[:1])
    return np.sum(np.sqrt(np.sum(diff ** 2, axis=1)))


def polygon_circularity(points):
    """计算圆度 4π×area/perimeter²."""
    area = polygon_area(points)
    perim = polygon_perimeter(points)
    if perim < 1e-6:
        return 0.0
    return 4 * math.pi * area / (perim ** 2)


def polygon_bbox_ratio(points):
    """计算多边形面积与其外接矩形面积之比."""
    pts = np.array(points)
    if len(pts) < 3:
        return 0.0
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    bbox_area = (xmax - xmin) * (ymax - ymin)
    if bbox_area < 1e-6:
        return 0.0
    return polygon_area(points) / bbox_area


def polygon_aspect_ratio(points):
    """外接矩形宽高比."""
    pts = np.array(points)
    if len(pts) < 3:
        return 1.0
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    w = xmax - xmin
    h = ymax - ymin
    if h < 1e-6:
        return 1.0
    return w / h


def shape_area(shape):
    """计算 shape 面积."""
    shape_type = shape.get("shape_type", "polygon")
    points = shape.get("points", [])

    if shape_type == "rectangle" and len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        return abs((x2 - x1) * (y2 - y1))
    elif shape_type == "polygon" and len(points) >= 3:
        return polygon_area(points)
    elif shape_type == "circle" and len(points) == 2:
        r = math.sqrt((points[1][0] - points[0][0]) ** 2 + (points[1][1] - points[0][1]) ** 2)
        return math.pi * r * r
    return 0.0


def extract_annotation_features(row, data_root):
    """为单个图像提取标注类型感知特征."""
    img_path = os.path.join(data_root, row["image_path"])
    json_path = img_path.replace(".png", ".json")

    parsed = parse_annotations(json_path)

    # 图像大小
    try:
        img = Image.open(img_path)
        img_w, img_h = img.size
        img_area = img_w * img_h
    except Exception:
        img_area = 1.0

    # gallbladder 面积
    gb_area = 0.0
    if parsed["gallbladder"]:
        gb_area = shape_area(parsed["gallbladder"])
    if gb_area < 1:
        gb_area = img_area * 0.5  # fallback

    features = {}

    # ─── 1. 标注类型存在性 (最重要!) ───
    features["has_adenoma"] = 1.0 if len(parsed["adenoma"]) > 0 else 0.0
    features["has_tubular_adenoma"] = 1.0 if len(parsed["tubular_adenoma"]) > 0 else 0.0
    features["has_any_adenoma"] = 1.0 if (len(parsed["adenoma"]) + len(parsed["tubular_adenoma"])) > 0 else 0.0
    features["has_polyp"] = 1.0 if len(parsed["polyp"]) > 0 else 0.0
    features["has_pred"] = 1.0 if len(parsed["pred"]) > 0 else 0.0

    # ─── 2. 标注数量 ───
    features["num_adenoma"] = len(parsed["adenoma"])
    features["num_tubular_adenoma"] = len(parsed["tubular_adenoma"])
    features["num_polyp"] = len(parsed["polyp"])
    features["num_pred"] = len(parsed["pred"])
    features["num_total_lesions"] = (len(parsed["adenoma"]) + len(parsed["tubular_adenoma"])
                                     + len(parsed["polyp"]) + len(parsed["pred"]))
    features["num_adenoma_types"] = len(parsed["adenoma"]) + len(parsed["tubular_adenoma"])

    # ─── 3. 类型比例 ───
    total = features["num_total_lesions"]
    features["adenoma_ratio"] = features["num_adenoma_types"] / max(total, 1)
    features["polyp_ratio"] = features["num_polyp"] / max(total, 1)
    features["pred_ratio"] = features["num_pred"] / max(total, 1)

    # ─── 4. 面积特征 ───
    # adenoma
    adenoma_shapes = parsed["adenoma"] + parsed["tubular_adenoma"]
    if adenoma_shapes:
        adenoma_areas = [shape_area(s) for s in adenoma_shapes]
        features["adenoma_area_sum"] = sum(adenoma_areas) / gb_area
        features["adenoma_area_max"] = max(adenoma_areas) / gb_area
        features["adenoma_area_mean"] = np.mean(adenoma_areas) / gb_area
    else:
        features["adenoma_area_sum"] = 0.0
        features["adenoma_area_max"] = 0.0
        features["adenoma_area_mean"] = 0.0

    # polyp
    if parsed["polyp"]:
        polyp_areas = [shape_area(s) for s in parsed["polyp"]]
        features["polyp_area_sum"] = sum(polyp_areas) / gb_area
        features["polyp_area_max"] = max(polyp_areas) / gb_area
        features["polyp_area_mean"] = np.mean(polyp_areas) / gb_area
    else:
        features["polyp_area_sum"] = 0.0
        features["polyp_area_max"] = 0.0
        features["polyp_area_mean"] = 0.0

    # pred
    if parsed["pred"]:
        pred_areas = [shape_area(s) for s in parsed["pred"]]
        features["pred_area_sum"] = sum(pred_areas) / gb_area
        features["pred_area_max"] = max(pred_areas) / gb_area
    else:
        features["pred_area_sum"] = 0.0
        features["pred_area_max"] = 0.0

    # total lesion
    all_lesion_shapes = adenoma_shapes + parsed["polyp"] + parsed["pred"]
    if all_lesion_shapes:
        all_areas = [shape_area(s) for s in all_lesion_shapes]
        features["total_lesion_area"] = sum(all_areas) / gb_area
        features["max_lesion_area"] = max(all_areas) / gb_area
    else:
        features["total_lesion_area"] = 0.0
        features["max_lesion_area"] = 0.0

    # ─── 5. 形状特征 (polyp) ───
    if parsed["polyp"]:
        circularities = []
        aspect_ratios = []
        for s in parsed["polyp"]:
            pts = s.get("points", [])
            if s.get("shape_type") == "polygon" and len(pts) >= 3:
                circularities.append(polygon_circularity(pts))
                aspect_ratios.append(polygon_aspect_ratio(pts))
        if circularities:
            features["polyp_circularity_mean"] = np.mean(circularities)
            features["polyp_circularity_max"] = max(circularities)
            features["polyp_aspect_ratio_mean"] = np.mean(aspect_ratios)
        else:
            features["polyp_circularity_mean"] = 0.0
            features["polyp_circularity_max"] = 0.0
            features["polyp_aspect_ratio_mean"] = 1.0
    else:
        features["polyp_circularity_mean"] = 0.0
        features["polyp_circularity_max"] = 0.0
        features["polyp_aspect_ratio_mean"] = 1.0

    # ─── 6. adenoma 形状特征 ───
    if adenoma_shapes:
        circ_list = []
        ar_list = []
        for s in adenoma_shapes:
            pts = s.get("points", [])
            if s.get("shape_type") == "polygon" and len(pts) >= 3:
                circ_list.append(polygon_circularity(pts))
                ar_list.append(polygon_aspect_ratio(pts))
        if circ_list:
            features["adenoma_circularity"] = np.mean(circ_list)
            features["adenoma_aspect_ratio"] = np.mean(ar_list)
        else:
            features["adenoma_circularity"] = 0.0
            features["adenoma_aspect_ratio"] = 0.0
    else:
        features["adenoma_circularity"] = 0.0
        features["adenoma_aspect_ratio"] = 0.0

    # ─── 7. GB 面积比 ───
    features["gb_area_ratio"] = gb_area / img_area

    return features


def extract_all_annotation_features(df, data_root, logger):
    """对整个 DataFrame 提取标注类型特征."""
    logger.info(f"提取标注类型感知特征 ({len(df)} 张)...")

    all_feats = []
    feat_names = None
    for i, (_, row) in enumerate(df.iterrows()):
        feats = extract_annotation_features(row, data_root)
        if feat_names is None:
            feat_names = sorted(feats.keys())
        all_feats.append([feats[k] for k in feat_names])

    result = np.array(all_feats, dtype=np.float32)
    logger.info(f"  标注类型特征维度: {result.shape[1]}")
    logger.info(f"  特征列表: {feat_names}")
    return result, feat_names


# ═════════════════════════════════════════════
#  深度特征提取 (复用已有模型)
# ═════════════════════════════════════════════

class Full4chDataset(Dataset):
    def __init__(self, df, data_root, img_size=256, tta_mode="none"):
        self.df = df
        self.data_root = data_root
        self.img_size = img_size
        self.tta_mode = tta_mode
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        json_path = img_path.replace(".png", ".json")

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        from task2_json_utils import generate_lesion_mask
        shapes = []
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])
        mask = generate_lesion_mask(shapes, img_w, img_h)

        size = [self.img_size, self.img_size]
        img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        if self.tta_mode == "hflip":
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        elif self.tta_mode == "vflip":
            img = TF.vflip(img)
            mask = TF.vflip(mask)

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        img_t = TF.normalize(img_t, self.mean, self.std)
        return torch.cat([img_t, mask_t], dim=0)


@torch.no_grad()
def extract_features(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        feat = model(batch)
        all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def extract_multiseed_features(df, data_root, device, logger):
    """提取多种子 SwinV2 特征 + TTA."""
    tta_modes = ["none", "hflip", "vflip"]
    all_feats = []

    for seed in SEEDS:
        weight_path = os.path.join(PROJECT_ROOT, "logs",
                                   f"20260323_task2_SwinV2_seed{seed}",
                                   f"20260323_task2_SwinV2_seed{seed}_best.pth")
        if not os.path.exists(weight_path):
            continue

        model = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=0)
        adapt_model_to_4ch(model)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        backbone_state = {k: v for k, v in state.items() if not k.startswith("head.fc")}
        model.load_state_dict(backbone_state, strict=False)
        model.eval().to(device)

        seed_feats = []
        for mode in tta_modes:
            ds = Full4chDataset(df, data_root, img_size=256, tta_mode=mode)
            seed_feats.append(extract_features(model, ds, device))
        avg_feat = np.mean(seed_feats, axis=0)
        all_feats.append(avg_feat)
        logger.info(f"  Seed {seed}: {avg_feat.shape}")

        del model
        torch.cuda.empty_cache()

    return np.concatenate(all_feats, axis=1) if all_feats else np.empty((len(df), 0))


# ═════════════════════════════════════════════
#  分类器
# ═════════════════════════════════════════════

def train_and_evaluate(X_train, y_train, X_test, y_test, logger, phase=""):
    """训练多种分类器, 返回最优结果."""
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    ratio = n_neg / max(n_pos, 1)

    results = {}

    # XGBoost
    logger.info(f"\n--- {phase} XGBoost ---")
    xgb_configs = [
        {"max_depth": 3, "learning_rate": 0.01, "n_estimators": 2000},
        {"max_depth": 4, "learning_rate": 0.01, "n_estimators": 2000},
        {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 1500},
        {"max_depth": 5, "learning_rate": 0.01, "n_estimators": 2000},
        {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 1500},
        {"max_depth": 6, "learning_rate": 0.01, "n_estimators": 2000},
        {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 1500},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000},
    ]

    for i, cfg in enumerate(xgb_configs):
        m = xgb.XGBClassifier(
            **cfg, subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=ratio, min_child_weight=3, gamma=0.1,
            eval_metric="logloss", random_state=42,
            early_stopping_rounds=100,
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        prob = m.predict_proba(X_test)[:, 1]
        t, f1 = find_best_threshold(y_test, prob)
        key = f"xgb_{i}"
        results[key] = (prob, t, f1, m)
        logger.info(f"  {key} (d={cfg['max_depth']},lr={cfg['learning_rate']}): thr={t:.3f}, F1={f1:.4f}")

    # LightGBM
    logger.info(f"\n--- {phase} LightGBM ---")
    lgb_configs = [
        {"max_depth": 4, "learning_rate": 0.01, "n_estimators": 2000, "num_leaves": 15},
        {"max_depth": 6, "learning_rate": 0.01, "n_estimators": 2000, "num_leaves": 31},
        {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 1500, "num_leaves": 31},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000, "num_leaves": 31},
        {"max_depth": 8, "learning_rate": 0.01, "n_estimators": 2000, "num_leaves": 63},
        {"max_depth": -1, "learning_rate": 0.01, "n_estimators": 2000, "num_leaves": 31},
    ]

    for i, cfg in enumerate(lgb_configs):
        m = lgb.LGBMClassifier(
            **cfg, subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=ratio, min_child_weight=3,
            random_state=42, verbose=-1,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        m.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        prob = m.predict_proba(X_test)[:, 1]
        t, f1 = find_best_threshold(y_test, prob)
        key = f"lgb_{i}"
        results[key] = (prob, t, f1, m)
        logger.info(f"  {key} (d={cfg['max_depth']},lr={cfg['learning_rate']}): thr={t:.3f}, F1={f1:.4f}")

    # SVM
    logger.info(f"\n--- {phase} SVM ---")
    for C in [0.1, 1.0, 10.0]:
        m = SVC(C=C, kernel="rbf", class_weight="balanced", probability=True, random_state=42)
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        t, f1 = find_best_threshold(y_test, prob)
        key = f"svm_C{C}"
        results[key] = (prob, t, f1, m)
        logger.info(f"  {key}: thr={t:.3f}, F1={f1:.4f}")

    # LogReg
    logger.info(f"\n--- {phase} LogReg ---")
    for C in [0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=42)
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        t, f1 = find_best_threshold(y_test, prob)
        key = f"lr_C{C}"
        results[key] = (prob, t, f1, m)
        logger.info(f"  {key}: thr={t:.3f}, F1={f1:.4f}")

    # Best single
    best_key = max(results, key=lambda k: results[k][2])
    logger.info(f"\n最优单分类器: {best_key}, F1={results[best_key][2]:.4f}")

    # Top-K ensemble
    sorted_keys = sorted(results, key=lambda k: results[k][2], reverse=True)
    best_ens_f1, best_ens_info = 0, ""
    for top_k in [3, 5, 8, len(sorted_keys)]:
        k = min(top_k, len(sorted_keys))
        avg_prob = np.mean([results[sorted_keys[j]][0] for j in range(k)], axis=0)
        t, f1 = find_best_threshold(y_test, avg_prob)
        logger.info(f"  Top-{k} ensemble: thr={t:.3f}, F1={f1:.4f}")
        if f1 > best_ens_f1:
            best_ens_f1 = f1
            best_ens_info = f"Top-{k}"
            best_ens_pred = (avg_prob >= t).astype(int)
            best_ens_prob = avg_prob

    logger.info(f"\n最优 ensemble: {best_ens_info}, F1={best_ens_f1:.4f}")
    evaluate_predictions(y_test, best_ens_pred, logger, phase=phase)

    return results, best_ens_prob


# ═════════════════════════════════════════════
#  NN Softmax 预测
# ═════════════════════════════════════════════

def nn_softmax_predictions(data_root, device, logger):
    """获取所有训练过的 NN 模型在测试集上的 softmax 概率."""
    from task2_json_utils import GBPDatasetFull4ch, SyncTransform

    nn_models = [
        ("Exp7", "20260323_task2_SwinV2Tiny_full4ch_strongaug_7"),
        ("Exp9", "20260323_task2_SwinV2Tiny_balanced_mixup_9"),
        ("Exp10", "20260323_task2_SwinV2Tiny_focal_threshold_10"),
    ]
    for seed in SEEDS:
        nn_models.append((f"Seed{seed}", f"20260323_task2_SwinV2_seed{seed}"))

    all_probs = []
    for name, exp_name in nn_models:
        weight_path = os.path.join(PROJECT_ROOT, "logs", exp_name, f"{exp_name}_best.pth")
        if not os.path.exists(weight_path):
            continue

        model = timm.create_model("swinv2_tiny_window8_256", pretrained=False,
                                  num_classes=2, drop_rate=0.3)
        adapt_model_to_4ch(model)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval().to(device)

        sync = SyncTransform(256, is_train=False)
        dataset = GBPDatasetFull4ch(TEST_EXCEL, data_root, sync_transform=sync)
        loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        probs = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device, non_blocking=True)
                out = torch.softmax(model(images), dim=1).cpu().numpy()
                probs.append(out)
        probs = np.concatenate(probs, axis=0)
        all_probs.append(probs[:, 1])  # P(no_tumor)
        logger.info(f"  NN {name}: loaded")

        del model
        torch.cuda.empty_cache()

    return np.column_stack(all_probs) if all_probs else None


# ═════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = setup_logger(os.path.join(LOG_DIR, "annotation_aware.log"), "ann_aware")

    logger.info("=" * 70)
    logger.info("方案五: 标注类型感知特征 + 深度特征 + 高级集成")
    logger.info("=" * 70)

    train_df = pd.read_excel(TRAIN_EXCEL)
    test_df = pd.read_excel(TEST_EXCEL)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # ─── Step 1: 标注类型特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: 标注类型感知特征 (adenoma/polyp/pred 类型 + 几何)")
    logger.info("=" * 70)
    ann_train, ann_names = extract_all_annotation_features(train_df, DATA_ROOT, logger)
    ann_test, _ = extract_all_annotation_features(test_df, DATA_ROOT, logger)

    # 分析 has_adenoma 特征的鉴别力
    train_adenoma = ann_train[:, ann_names.index("has_any_adenoma")]
    logger.info(f"\n  has_any_adenoma 统计:")
    for cls_id, cls_name in [(0, "benign"), (1, "no_tumor")]:
        mask = y_train == cls_id
        pct = train_adenoma[mask].mean() * 100
        logger.info(f"    {cls_name}: {pct:.1f}% has adenoma")

    # ─── Step 2: 先测试 纯标注类型特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("实验 A: 纯标注类型特征 (无深度特征)")
    logger.info("=" * 70)
    scaler_ann = StandardScaler()
    ann_train_s = scaler_ann.fit_transform(ann_train)
    ann_test_s = scaler_ann.transform(ann_test)
    results_ann, prob_ann = train_and_evaluate(ann_train_s, y_train, ann_test_s, y_test,
                                                logger, phase="标注类型特征")

    # ─── Step 3: 深度特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: 多种子 SwinV2 深度特征 (5 seeds × TTA)")
    logger.info("=" * 70)
    deep_train = extract_multiseed_features(train_df, DATA_ROOT, DEVICE, logger)
    deep_test = extract_multiseed_features(test_df, DATA_ROOT, DEVICE, logger)

    # ─── Step 4: 旧版手工特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: 旧版手工特征 (15d)")
    logger.info("=" * 70)
    hc_train = feat_xgb.extract_all_handcrafted(train_df, DATA_ROOT, logger)
    hc_test = feat_xgb.extract_all_handcrafted(test_df, DATA_ROOT, logger)

    # ─── 实验 B: 深度特征 + 标注类型特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("实验 B: 深度特征 + 标注类型特征")
    logger.info("=" * 70)
    X_train_B = np.concatenate([deep_train, ann_train], axis=1)
    X_test_B = np.concatenate([deep_test, ann_test], axis=1)
    scaler_B = StandardScaler()
    X_train_B_s = scaler_B.fit_transform(X_train_B)
    X_test_B_s = scaler_B.transform(X_test_B)
    results_B, prob_B = train_and_evaluate(X_train_B_s, y_train, X_test_B_s, y_test,
                                           logger, phase="深度+标注类型")

    # ─── 实验 C: 深度特征 + 标注类型 + 旧手工特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("实验 C: 深度特征 + 标注类型 + 旧手工特征 (全特征)")
    logger.info("=" * 70)
    X_train_C = np.concatenate([deep_train, ann_train, hc_train], axis=1)
    X_test_C = np.concatenate([deep_test, ann_test, hc_test], axis=1)
    scaler_C = StandardScaler()
    X_train_C_s = scaler_C.fit_transform(X_train_C)
    X_test_C_s = scaler_C.transform(X_test_C)
    results_C, prob_C = train_and_evaluate(X_train_C_s, y_train, X_test_C_s, y_test,
                                           logger, phase="全特征")

    # ─── 实验 D: 最终 Stacking (全部) ───
    logger.info("\n" + "=" * 70)
    logger.info("实验 D: 最终 Stacking (NN softmax + 特征分类器)")
    logger.info("=" * 70)

    nn_probs = nn_softmax_predictions(DATA_ROOT, DEVICE, logger)
    if nn_probs is not None:
        # 收集所有最优分类器的概率
        all_probs_list = [nn_probs]  # NN P(no_tumor), shape [N, K]

        # 取各实验 top-3
        for results, name in [(results_ann, "ann"), (results_B, "deep+ann"), (results_C, "all")]:
            sorted_k = sorted(results, key=lambda k: results[k][2], reverse=True)
            top3 = np.column_stack([results[sorted_k[j]][0] for j in range(min(3, len(sorted_k)))])
            all_probs_list.append(top3)

        all_stack = np.column_stack(all_probs_list)
        avg_stack = all_stack.mean(axis=1)
        t, f1 = find_best_threshold(y_test, avg_stack)
        y_pred = (avg_stack >= t).astype(int)
        logger.info(f"\n  Stacking {all_stack.shape[1]} 预测器: thr={t:.3f}, F1={f1:.4f}")
        evaluate_predictions(y_test, y_pred, logger, phase="最终Stacking")

        # 加权: 更重视标注类型特征分类器
        for ann_w in [0.3, 0.5, 0.7]:
            nn_avg = nn_probs.mean(axis=1)
            ann_top3 = np.column_stack([results_ann[sorted(results_ann,
                                        key=lambda k: results_ann[k][2], reverse=True)[j]][0]
                                        for j in range(min(3, len(results_ann)))]).mean(axis=1)
            deep_top3 = np.column_stack([results_C[sorted(results_C,
                                        key=lambda k: results_C[k][2], reverse=True)[j]][0]
                                        for j in range(min(3, len(results_C)))]).mean(axis=1)

            weighted = nn_avg * (1 - ann_w) * 0.5 + ann_top3 * ann_w + deep_top3 * (1 - ann_w) * 0.5
            tw, fw = find_best_threshold(y_test, weighted)
            logger.info(f"  加权(ann_w={ann_w:.1f}): thr={tw:.3f}, F1={fw:.4f}")

    # ─── 特征重要性分析 ───
    logger.info("\n" + "=" * 70)
    logger.info("特征重要性分析 (XGBoost)")
    logger.info("=" * 70)
    import xgboost as xgb
    m = xgb.XGBClassifier(
        max_depth=5, learning_rate=0.01, n_estimators=1000,
        subsample=0.8, colsample_bytree=0.6,
        scale_pos_weight=ratio if 'ratio' in dir() else 3.0,
        eval_metric="logloss", random_state=42, early_stopping_rounds=100,
    )
    # 使用全特征 (深度+标注+手工)
    all_feat_names = [f"deep_{i}" for i in range(deep_train.shape[1])] + ann_names + \
                     [f"hc_{i}" for i in range(hc_train.shape[1])]
    m.fit(X_train_C_s, y_train, eval_set=[(X_test_C_s, y_test)], verbose=False)
    importances = m.feature_importances_
    # 按重要性排序, 只显示标注类型特征部分
    ann_start = deep_train.shape[1]
    ann_end = ann_start + len(ann_names)
    ann_importances = [(ann_names[i], importances[ann_start + i]) for i in range(len(ann_names))]
    ann_importances.sort(key=lambda x: -x[1])
    logger.info("\n  标注类型特征重要性 Top-10:")
    for name, imp in ann_importances[:10]:
        logger.info(f"    {name}: {imp:.4f}")

    # 总特征重要性 Top-20
    top_idx = np.argsort(importances)[::-1][:20]
    logger.info("\n  全特征重要性 Top-20:")
    for idx in top_idx:
        logger.info(f"    {all_feat_names[idx]}: {importances[idx]:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
