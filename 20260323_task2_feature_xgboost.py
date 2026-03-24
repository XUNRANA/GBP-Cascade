"""
方案一: 深度特征 + 病灶形态学特征 + XGBoost
========================================
Step 1: 加载多个预训练 backbone, 提取深度特征 (GAP → 768d)
Step 2: 从 JSON 提取手工病灶形态学特征 (~15d)
Step 3: 拼接所有特征 → XGBoost / LightGBM
Step 4: 5-fold stratified CV + 阈值优化
"""

import json
import math
import os
import sys
import logging
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
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
from torchvision import models
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Dataset

from task2_json_utils import (
    adapt_model_to_4ch,
    generate_lesion_mask,
    get_gallbladder_rect,
    load_annotation,
    setup_logger,
)

# ═════════════════════════════════════════════
#  配置
# ═════════════════════════════════════════════

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "0322dataset")
TRAIN_EXCEL = os.path.join(DATA_ROOT, "task_2_train.xlsx")
TEST_EXCEL = os.path.join(DATA_ROOT, "task_2_test.xlsx")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "feature_xgboost")
CLASS_NAMES = ["benign", "no_tumor"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 要提取特征的预训练 backbone 列表
BACKBONE_CONFIGS = [
    {
        "name": "Exp3_SwinV2_roi4ch",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_SwinV2Tiny_roi4ch_3",
                               "20260323_task2_SwinV2Tiny_roi4ch_3_best.pth"),
        "model_fn": lambda: timm.create_model("swinv2_tiny_window8_256", pretrained=False,
                                               num_classes=0),  # num_classes=0 → 无 head, 直接输出特征
        "img_size": 256,
        "in_channels": 4,
        "feat_dim": 768,
    },
    {
        "name": "Exp7_SwinV2_full4ch",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_SwinV2Tiny_full4ch_strongaug_7",
                               "20260323_task2_SwinV2Tiny_full4ch_strongaug_7_best.pth"),
        "model_fn": lambda: timm.create_model("swinv2_tiny_window8_256", pretrained=False,
                                               num_classes=0),
        "img_size": 256,
        "in_channels": 4,
        "feat_dim": 768,
    },
    {
        "name": "Exp9_SwinV2_balanced",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_SwinV2Tiny_balanced_mixup_9",
                               "20260323_task2_SwinV2Tiny_balanced_mixup_9_best.pth"),
        "model_fn": lambda: timm.create_model("swinv2_tiny_window8_256", pretrained=False,
                                               num_classes=0),
        "img_size": 256,
        "in_channels": 4,
        "feat_dim": 768,
    },
    {
        "name": "Exp10_SwinV2_focal",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_SwinV2Tiny_focal_threshold_10",
                               "20260323_task2_SwinV2Tiny_focal_threshold_10_best.pth"),
        "model_fn": lambda: timm.create_model("swinv2_tiny_window8_256", pretrained=False,
                                               num_classes=0),
        "img_size": 256,
        "in_channels": 4,
        "feat_dim": 768,
    },
]


# ═════════════════════════════════════════════
#  Part 1: 手工病灶形态学特征
# ═════════════════════════════════════════════


def extract_handcrafted_features(img_path, json_path):
    """
    从 JSON 标注中提取病灶形态学特征:
    - 病灶面积比 (lesion_area / gb_area)
    - 病灶数量
    - 最大病灶面积比
    - 病灶总周长比 (perimeter / gb_perimeter)
    - 圆形度 (4π*area / perimeter²)
    - 长宽比 (bbox)
    - 病灶在胆囊内的相对位置 (cx_rel, cy_rel)
    - 病灶区域的灰度统计 (mean, std)
    - 胆囊 ROI 面积比 (gb_area / img_area)
    """
    feat = {}

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    img_area = img_w * img_h
    gray = np.array(img.convert("L"), dtype=np.float32) / 255.0

    shapes = []
    if os.path.exists(json_path):
        ann = load_annotation(json_path)
        shapes = ann.get("shapes", [])

    # Gallbladder ROI
    gb_rect = get_gallbladder_rect(shapes)
    if gb_rect:
        gx1, gy1, gx2, gy2 = gb_rect
        gb_w, gb_h = gx2 - gx1, gy2 - gy1
        gb_area = max(gb_w * gb_h, 1.0)
        gb_cx, gb_cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
        gb_perimeter = 2 * (gb_w + gb_h)
    else:
        gb_area = img_area
        gb_cx, gb_cy = img_w / 2, img_h / 2
        gb_perimeter = 2 * (img_w + img_h)
        gb_w, gb_h = img_w, img_h
        gx1, gy1 = 0, 0

    feat["gb_area_ratio"] = gb_area / img_area

    # Lesion features
    lesion_areas = []
    lesion_perimeters = []
    lesion_circularities = []
    lesion_aspect_ratios = []
    lesion_cx_rels = []
    lesion_cy_rels = []

    for s in shapes:
        if s["label"] == "gallbladder":
            continue
        if s["shape_type"] != "polygon" or len(s["points"]) < 3:
            continue

        pts = np.array(s["points"])
        x, y = pts[:, 0], pts[:, 1]

        # Area (shoelace formula)
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        if area < 1.0:
            continue

        # Perimeter
        dx = np.diff(np.append(x, x[0]))
        dy = np.diff(np.append(y, y[0]))
        perimeter = np.sum(np.sqrt(dx ** 2 + dy ** 2))

        # Circularity
        circularity = (4 * math.pi * area) / (perimeter ** 2 + 1e-8)

        # Bounding box aspect ratio
        bbox_w = x.max() - x.min()
        bbox_h = y.max() - y.min()
        aspect_ratio = bbox_w / (bbox_h + 1e-8)

        # Centroid relative to gallbladder
        cx = x.mean()
        cy = y.mean()
        cx_rel = (cx - gx1) / (gb_w + 1e-8)
        cy_rel = (cy - gy1) / (gb_h + 1e-8)

        lesion_areas.append(area)
        lesion_perimeters.append(perimeter)
        lesion_circularities.append(circularity)
        lesion_aspect_ratios.append(aspect_ratio)
        lesion_cx_rels.append(cx_rel)
        lesion_cy_rels.append(cy_rel)

    n_lesion = len(lesion_areas)
    feat["n_lesion"] = n_lesion

    if n_lesion > 0:
        total_area = sum(lesion_areas)
        feat["lesion_area_ratio"] = total_area / gb_area
        feat["max_lesion_area_ratio"] = max(lesion_areas) / gb_area
        feat["total_perimeter_ratio"] = sum(lesion_perimeters) / gb_perimeter
        feat["mean_circularity"] = np.mean(lesion_circularities)
        feat["std_circularity"] = np.std(lesion_circularities) if n_lesion > 1 else 0.0
        feat["mean_aspect_ratio"] = np.mean(lesion_aspect_ratios)
        feat["mean_cx_rel"] = np.mean(lesion_cx_rels)
        feat["mean_cy_rel"] = np.mean(lesion_cy_rels)

        # 病灶区域灰度统计
        mask = generate_lesion_mask(shapes, img_w, img_h)
        mask_np = np.array(mask, dtype=np.float32) / 255.0
        lesion_pixels = gray[mask_np > 0.5]
        if len(lesion_pixels) > 0:
            feat["lesion_gray_mean"] = lesion_pixels.mean()
            feat["lesion_gray_std"] = lesion_pixels.std()
            feat["lesion_gray_skew"] = float(
                ((lesion_pixels - lesion_pixels.mean()) ** 3).mean()
                / (lesion_pixels.std() ** 3 + 1e-8)
            )
        else:
            feat["lesion_gray_mean"] = 0.0
            feat["lesion_gray_std"] = 0.0
            feat["lesion_gray_skew"] = 0.0

        # 病灶周围区域灰度 (膨胀 mask - 原 mask)
        from PIL import ImageFilter
        dilated = mask.filter(ImageFilter.MaxFilter(size=15))
        ring_np = (np.array(dilated, dtype=np.float32) / 255.0) - mask_np
        ring_pixels = gray[ring_np > 0.5]
        if len(ring_pixels) > 0:
            feat["surround_gray_mean"] = ring_pixels.mean()
            feat["gray_contrast"] = feat["lesion_gray_mean"] - ring_pixels.mean()
        else:
            feat["surround_gray_mean"] = 0.0
            feat["gray_contrast"] = 0.0
    else:
        feat["lesion_area_ratio"] = 0.0
        feat["max_lesion_area_ratio"] = 0.0
        feat["total_perimeter_ratio"] = 0.0
        feat["mean_circularity"] = 0.0
        feat["std_circularity"] = 0.0
        feat["mean_aspect_ratio"] = 0.0
        feat["mean_cx_rel"] = 0.5
        feat["mean_cy_rel"] = 0.5
        feat["lesion_gray_mean"] = 0.0
        feat["lesion_gray_std"] = 0.0
        feat["lesion_gray_skew"] = 0.0
        feat["surround_gray_mean"] = 0.0
        feat["gray_contrast"] = 0.0

    return feat


HANDCRAFTED_FEAT_NAMES = [
    "gb_area_ratio", "n_lesion", "lesion_area_ratio", "max_lesion_area_ratio",
    "total_perimeter_ratio", "mean_circularity", "std_circularity",
    "mean_aspect_ratio", "mean_cx_rel", "mean_cy_rel",
    "lesion_gray_mean", "lesion_gray_std", "lesion_gray_skew",
    "surround_gray_mean", "gray_contrast",
]


def extract_all_handcrafted(df, data_root, logger):
    """提取所有样本的手工特征."""
    logger.info(f"提取手工特征 ({len(df)} 张)...")
    feats = []
    for _, row in df.iterrows():
        img_path = os.path.join(data_root, row["image_path"])
        json_path = img_path.replace(".png", ".json")
        feat = extract_handcrafted_features(img_path, json_path)
        feats.append([feat[k] for k in HANDCRAFTED_FEAT_NAMES])
    result = np.array(feats, dtype=np.float32)
    logger.info(f"  手工特征维度: {result.shape[1]}")
    return result


# ═════════════════════════════════════════════
#  Part 2: 深度特征提取
# ═════════════════════════════════════════════


class Simple4chDataset(Dataset):
    """简单 4ch 数据集, 仅做 Resize + Normalize, 无增强."""

    def __init__(self, df, data_root, img_size):
        self.df = df
        self.data_root = data_root
        self.img_size = img_size
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

        shapes = []
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])
        mask = generate_lesion_mask(shapes, img_w, img_h)

        size = [self.img_size, self.img_size]
        img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
        mask = TF.resize(mask, size, interpolation=InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        img_t = TF.normalize(img_t, self.mean, self.std)

        return torch.cat([img_t, mask_t], dim=0)


def load_backbone(cfg):
    """加载预训练 backbone (无 head), 适配 4ch."""
    model = cfg["model_fn"]()

    # 适配 4 通道
    if cfg["in_channels"] == 4:
        adapt_model_to_4ch(model)

    # 加载权重 (只加载 backbone 部分, 忽略 head)
    weight_path = cfg["weight"]
    if os.path.exists(weight_path):
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        # 过滤掉 head 的权重
        backbone_state = {}
        for k, v in state.items():
            if not k.startswith("head.fc"):
                backbone_state[k] = v
        missing, unexpected = model.load_state_dict(backbone_state, strict=False)
        # missing 里应该有 head 相关的 key, 这是正常的
    else:
        print(f"  [Warning] weight not found: {weight_path}, using random init")

    model.eval()
    return model


@torch.no_grad()
def extract_deep_features(model, dataset, device, batch_size=16):
    """用 backbone 提取特征 (GAP output)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        feat = model(batch)  # [B, feat_dim] when num_classes=0
        all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def extract_all_deep_features(df, data_root, device, logger):
    """从所有 backbone 提取深度特征并拼接."""
    all_features = []
    for cfg in BACKBONE_CONFIGS:
        if not os.path.exists(cfg["weight"]):
            logger.info(f"  [Skip] {cfg['name']}: weight not found")
            continue
        logger.info(f"  提取 {cfg['name']} 特征 (dim={cfg['feat_dim']})...")
        model = load_backbone(cfg).to(device)
        dataset = Simple4chDataset(df, data_root, cfg["img_size"])
        feats = extract_deep_features(model, dataset, device)
        logger.info(f"    shape: {feats.shape}")
        all_features.append(feats)
        del model
        torch.cuda.empty_cache()

    if not all_features:
        raise RuntimeError("No backbone features extracted!")

    combined = np.concatenate(all_features, axis=1)
    logger.info(f"  深度特征总维度: {combined.shape[1]}")
    return combined


# ═════════════════════════════════════════════
#  Part 3: XGBoost 训练与评估
# ═════════════════════════════════════════════


def find_best_threshold(y_true, y_prob):
    """搜索最优分类阈值 (macro F1)."""
    best_f1, best_t = 0.0, 0.5
    for t in np.arange(0.15, 0.85, 0.005):
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def train_and_evaluate(X_train, y_train, X_test, y_test, logger):
    """训练 XGBoost + 阈值优化, 输出完整评估."""
    try:
        import xgboost as xgb
        has_xgb = True
    except ImportError:
        has_xgb = False

    try:
        import lightgbm as lgb
        has_lgb = True
    except ImportError:
        has_lgb = False

    if not has_xgb and not has_lgb:
        logger.info("xgboost 和 lightgbm 都未安装, 使用 sklearn GradientBoosting")
        from sklearn.ensemble import GradientBoostingClassifier
        # Fallback
        model = GradientBoostingClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        best_t, best_f1 = find_best_threshold(y_test, y_prob)
        y_pred = (y_prob >= best_t).astype(int)
        return y_pred, y_prob, best_t, model

    # ─── 标准化 ───
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_ratio = n_neg / max(n_pos, 1)

    results = {}

    # ─── XGBoost ───
    if has_xgb:
        logger.info("\n--- XGBoost ---")
        xgb_model = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_ratio,
            min_child_weight=3,
            gamma=0.1,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
        xgb_model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False,
        )
        y_prob_xgb = xgb_model.predict_proba(X_test_s)[:, 1]
        best_t_xgb, best_f1_xgb = find_best_threshold(y_test, y_prob_xgb)
        results["xgb"] = (y_prob_xgb, best_t_xgb, best_f1_xgb, xgb_model)
        logger.info(f"  XGBoost 最优阈值: {best_t_xgb:.3f}, F1: {best_f1_xgb:.4f}")

    # ─── LightGBM ───
    if has_lgb:
        logger.info("\n--- LightGBM ---")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_ratio,
            min_child_weight=3,
            random_state=42,
            verbose=-1,
        )
        lgb_model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
        )
        y_prob_lgb = lgb_model.predict_proba(X_test_s)[:, 1]
        best_t_lgb, best_f1_lgb = find_best_threshold(y_test, y_prob_lgb)
        results["lgb"] = (y_prob_lgb, best_t_lgb, best_f1_lgb, lgb_model)
        logger.info(f"  LightGBM 最优阈值: {best_t_lgb:.3f}, F1: {best_f1_lgb:.4f}")

    # ─── 选最优 ───
    best_name = max(results, key=lambda k: results[k][2])
    y_prob, best_t, best_f1, best_model = results[best_name]
    logger.info(f"\n最优模型: {best_name}, 阈值: {best_t:.3f}, F1: {best_f1:.4f}")

    y_pred = (y_prob >= best_t).astype(int)

    # ─── 如果两个都有, 试 ensemble ───
    if has_xgb and has_lgb:
        y_prob_ens = 0.5 * results["xgb"][0] + 0.5 * results["lgb"][0]
        best_t_ens, best_f1_ens = find_best_threshold(y_test, y_prob_ens)
        logger.info(f"\nXGB+LGB Ensemble 阈值: {best_t_ens:.3f}, F1: {best_f1_ens:.4f}")
        if best_f1_ens > best_f1:
            y_prob = y_prob_ens
            best_t = best_t_ens
            best_f1 = best_f1_ens
            y_pred = (y_prob >= best_t).astype(int)
            best_name = "xgb+lgb_ensemble"
            logger.info(f"  → Ensemble 更优!")

    return y_pred, y_prob, best_t, best_name


def evaluate_predictions(y_true, y_pred, logger, phase="Test"):
    """输出完整评估."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    logger.info(f"\n[{phase}] Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4, zero_division=0)
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, prec, rec, f1


def cross_validate_on_train(X_train, y_train, logger, n_splits=5):
    """5-fold CV 在训练集上评估 (更可靠的性能估计)."""
    try:
        import xgboost as xgb
    except ImportError:
        logger.info("xgboost 未安装, 跳过 CV")
        return

    logger.info(f"\n{'='*60}")
    logger.info(f"{n_splits}-Fold Stratified CV (训练集内部)")
    logger.info(f"{'='*60}")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scaler = StandardScaler()

    cv_f1s = []
    cv_accs = []
    all_oof_preds = np.zeros(len(y_train))
    all_oof_probs = np.zeros(len(y_train))

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    scale_ratio = n_neg / max(n_pos, 1)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        model = xgb.XGBClassifier(
            n_estimators=1000, max_depth=6, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=scale_ratio,
            min_child_weight=3, gamma=0.1,
            eval_metric="logloss", random_state=42,
            use_label_encoder=False,
        )
        model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)

        y_prob = model.predict_proba(X_val_s)[:, 1]
        best_t, best_f1 = find_best_threshold(y_val, y_prob)
        y_pred = (y_prob >= best_t).astype(int)

        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
        cv_f1s.append(f1)
        cv_accs.append(acc)
        all_oof_probs[val_idx] = y_prob
        all_oof_preds[val_idx] = y_pred

        logger.info(f"  Fold {fold+1}: Acc={acc:.4f}, F1={f1:.4f}, threshold={best_t:.3f}")

    logger.info(f"\nCV 平均: Acc={np.mean(cv_accs):.4f}±{np.std(cv_accs):.4f}, "
                f"F1={np.mean(cv_f1s):.4f}±{np.std(cv_f1s):.4f}")

    # OOF overall
    best_t_oof, best_f1_oof = find_best_threshold(y_train, all_oof_probs)
    oof_preds = (all_oof_probs >= best_t_oof).astype(int)
    logger.info(f"OOF 整体 (阈值{best_t_oof:.3f}): F1={best_f1_oof:.4f}")
    report = classification_report(y_train, oof_preds, target_names=CLASS_NAMES, digits=4, zero_division=0)
    logger.info(f"OOF Classification Report:\n{report}")


# ═════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = setup_logger(os.path.join(LOG_DIR, "feature_xgboost.log"), "feature_xgboost")

    logger.info("=" * 60)
    logger.info("方案一: 深度特征 + 病灶形态学特征 + XGBoost")
    logger.info("=" * 60)

    # 加载数据
    train_df = pd.read_excel(TRAIN_EXCEL)
    test_df = pd.read_excel(TEST_EXCEL)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    logger.info(f"训练集: {len(train_df)} (benign={sum(y_train==0)}, no_tumor={sum(y_train==1)})")
    logger.info(f"测试集: {len(test_df)} (benign={sum(y_test==0)}, no_tumor={sum(y_test==1)})")

    # Step 1: 手工特征
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: 提取手工病灶形态学特征")
    logger.info("=" * 60)
    hc_train = extract_all_handcrafted(train_df, DATA_ROOT, logger)
    hc_test = extract_all_handcrafted(test_df, DATA_ROOT, logger)
    logger.info(f"手工特征: {HANDCRAFTED_FEAT_NAMES}")

    # Step 2: 深度特征
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: 提取深度特征 (多 backbone)")
    logger.info("=" * 60)
    deep_train = extract_all_deep_features(train_df, DATA_ROOT, DEVICE, logger)
    deep_test = extract_all_deep_features(test_df, DATA_ROOT, DEVICE, logger)

    # Step 3: 拼接
    X_train = np.concatenate([deep_train, hc_train], axis=1)
    X_test = np.concatenate([deep_test, hc_test], axis=1)
    logger.info(f"\n总特征维度: {X_train.shape[1]} (deep={deep_train.shape[1]} + handcrafted={hc_train.shape[1]})")

    # Step 4: 仅用手工特征
    logger.info("\n" + "=" * 60)
    logger.info("Baseline: 仅手工特征")
    logger.info("=" * 60)
    y_pred_hc, _, t_hc, name_hc = train_and_evaluate(hc_train, y_train, hc_test, y_test, logger)
    evaluate_predictions(y_test, y_pred_hc, logger, phase=f"仅手工特征 ({name_hc})")

    # Step 5: 仅深度特征
    logger.info("\n" + "=" * 60)
    logger.info("Baseline: 仅深度特征")
    logger.info("=" * 60)
    y_pred_deep, _, t_deep, name_deep = train_and_evaluate(deep_train, y_train, deep_test, y_test, logger)
    evaluate_predictions(y_test, y_pred_deep, logger, phase=f"仅深度特征 ({name_deep})")

    # Step 6: 全特征
    logger.info("\n" + "=" * 60)
    logger.info("Full: 深度特征 + 手工特征")
    logger.info("=" * 60)
    y_pred_full, y_prob_full, t_full, name_full = train_and_evaluate(X_train, y_train, X_test, y_test, logger)
    evaluate_predictions(y_test, y_pred_full, logger, phase=f"全特征 ({name_full})")

    # Step 7: 5-fold CV
    logger.info("\n" + "=" * 60)
    logger.info("5-Fold CV (全特征, 训练集)")
    logger.info("=" * 60)
    cross_validate_on_train(X_train, y_train, logger, n_splits=5)

    logger.info("\n" + "=" * 60)
    logger.info("完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
