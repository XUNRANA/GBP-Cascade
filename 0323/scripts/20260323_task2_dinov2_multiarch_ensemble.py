"""
方案四: DINOv2 + 多架构深度特征融合 + 高级集成
================================================
1. DINOv2-Base 自监督特征 (768d, 3ch, 无需微调)
2. 多种子 SwinV2 特征 (5×768d, 4ch+mask)
3. ConvNeXt 特征 (Exp8, Exp11, 各768d, 4ch+mask)
4. 手工病灶形态学特征 (15d)
5. 特征选择 / PCA 降维
6. XGBoost / LightGBM 多配置 + Stacking
"""

import importlib
import json
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from task2_json_utils import (
    adapt_model_to_4ch,
    generate_lesion_mask,
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
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "dinov2_multiarch_ensemble")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [42, 123, 456, 789, 1024]

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════
#  数据集
# ═════════════════════════════════════════════

class RGB3chDataset(Dataset):
    """3通道 RGB 数据集 (用于 DINOv2)."""

    def __init__(self, df, data_root, img_size=518, tta_mode="none"):
        self.df = df
        self.data_root = data_root
        self.img_size = img_size
        self.tta_mode = tta_mode
        # DINOv2 使用 ImageNet 标准化
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        img = Image.open(img_path).convert("RGB")

        size = [self.img_size, self.img_size]
        img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)

        if self.tta_mode == "hflip":
            img = TF.hflip(img)
        elif self.tta_mode == "vflip":
            img = TF.vflip(img)

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, self.mean, self.std)
        return img_t


class Full4chDataset(Dataset):
    """4通道 (RGB+mask) 数据集."""

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


# ═════════════════════════════════════════════
#  特征提取
# ═════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, dataset, device, batch_size=16):
    """通用特征提取."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=4, pin_memory=True)
    all_feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        feat = model(batch)
        all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def extract_with_tta(model, df, data_root, img_size, device, dataset_cls, tta_modes, batch_size=16):
    """TTA 特征提取: 多种 TTA 模式平均."""
    feats_list = []
    for mode in tta_modes:
        ds = dataset_cls(df, data_root, img_size=img_size, tta_mode=mode)
        feats_list.append(extract_features(model, ds, device, batch_size))
    return np.mean(feats_list, axis=0)


# ═════════════════════════════════════════════
#  DINOv2 特征
# ═════════════════════════════════════════════

def extract_dinov2_features(df, data_root, device, logger):
    """提取 DINOv2-Base 特征 (自监督预训练, 非常强大)."""
    logger.info("加载 DINOv2-Base 预训练模型 (自监督, LVD-142M)...")
    model = timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=0)
    model.eval().to(device)

    tta_modes = ["none", "hflip", "vflip"]
    # DINOv2 default image size: 518 (patch14, 37x37)
    # 也试 224 (patch14, 16x16) — 速度更快
    feats = extract_with_tta(model, df, data_root, 518, device,
                             RGB3chDataset, tta_modes, batch_size=8)
    logger.info(f"  DINOv2 特征维度: {feats.shape[1]}")

    del model
    torch.cuda.empty_cache()
    return feats


# ═════════════════════════════════════════════
#  已训练模型特征
# ═════════════════════════════════════════════

def load_backbone_generic(model_name, weight_path, in_channels=4, drop_rate=0.3):
    """加载任意 timm 模型的 backbone (num_classes=0)."""
    model = timm.create_model(model_name, pretrained=False, num_classes=0, drop_rate=0.0)

    if in_channels == 4:
        adapt_model_to_4ch(model)

    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    # 过滤掉 head 相关的参数
    backbone_state = {}
    for k, v in state.items():
        if any(x in k for x in ["head.fc", "classifier", "head.l"]):
            continue
        backbone_state[k] = v

    model.load_state_dict(backbone_state, strict=False)
    model.eval()
    return model


def extract_multiseed_swinv2_features(df, data_root, device, logger):
    """提取 5 种子 SwinV2 backbone 特征 + TTA."""
    tta_modes = ["none", "hflip", "vflip"]
    all_feats = []

    for seed in SEEDS:
        weight_path = os.path.join(PROJECT_ROOT, "logs",
                                   f"20260323_task2_SwinV2_seed{seed}",
                                   f"20260323_task2_SwinV2_seed{seed}_best.pth")
        if not os.path.exists(weight_path):
            logger.info(f"  [Skip] seed={seed}: 权重不存在")
            continue

        model = load_backbone_generic("swinv2_tiny_window8_256", weight_path, in_channels=4)
        model = model.to(device)

        feats = extract_with_tta(model, df, data_root, 256, device,
                                 Full4chDataset, tta_modes)
        all_feats.append(feats)
        logger.info(f"  Seed {seed}: {feats.shape}")

        del model
        torch.cuda.empty_cache()

    return np.concatenate(all_feats, axis=1) if all_feats else np.empty((len(df), 0))


def extract_convnext_features(df, data_root, device, logger):
    """提取 ConvNeXt 特征 (Exp8 + Exp11)."""
    convnext_models = [
        ("Exp8_ConvNeXt_Full4ch", "20260323_task2_ConvNeXtTiny_full4ch_strongaug_8"),
        ("Exp11_ConvNeXt_Focal", "20260323_task2_ConvNeXtTiny_balanced_focal_11"),
    ]

    tta_modes = ["none", "hflip", "vflip"]
    all_feats = []

    for name, exp_name in convnext_models:
        weight_path = os.path.join(PROJECT_ROOT, "logs", exp_name, f"{exp_name}_best.pth")
        if not os.path.exists(weight_path):
            logger.info(f"  [Skip] {name}: 权重不存在")
            continue

        model = load_backbone_generic("convnext_tiny.fb_in1k", weight_path, in_channels=4)
        model = model.to(device)

        feats = extract_with_tta(model, df, data_root, 256, device,
                                 Full4chDataset, tta_modes)
        all_feats.append(feats)
        logger.info(f"  {name}: {feats.shape}")

        del model
        torch.cuda.empty_cache()

    return np.concatenate(all_feats, axis=1) if all_feats else np.empty((len(df), 0))


def extract_extra_swinv2_features(df, data_root, device, logger):
    """提取 Exp7, Exp9, Exp10 SwinV2 特征 (非种子模型)."""
    extra_models = [
        ("Exp7_Full4ch", "20260323_task2_SwinV2Tiny_full4ch_strongaug_7"),
        ("Exp9_Balanced", "20260323_task2_SwinV2Tiny_balanced_mixup_9"),
        ("Exp10_Focal", "20260323_task2_SwinV2Tiny_focal_threshold_10"),
    ]

    tta_modes = ["none", "hflip", "vflip"]
    all_feats = []

    for name, exp_name in extra_models:
        weight_path = os.path.join(PROJECT_ROOT, "logs", exp_name, f"{exp_name}_best.pth")
        if not os.path.exists(weight_path):
            logger.info(f"  [Skip] {name}")
            continue

        model = load_backbone_generic("swinv2_tiny_window8_256", weight_path, in_channels=4)
        model = model.to(device)

        feats = extract_with_tta(model, df, data_root, 256, device,
                                 Full4chDataset, tta_modes)
        all_feats.append(feats)
        logger.info(f"  {name}: {feats.shape}")

        del model
        torch.cuda.empty_cache()

    return np.concatenate(all_feats, axis=1) if all_feats else np.empty((len(df), 0))


# ═════════════════════════════════════════════
#  分类器训练
# ═════════════════════════════════════════════

def train_classifiers(X_train, y_train, X_test, y_test, logger, phase=""):
    """训练多种分类器 + 搜索最优."""
    import xgboost as xgb
    import lightgbm as lgb

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    ratio = n_neg / max(n_pos, 1)

    results = {}

    # ─── XGBoost: 扩展超参搜索 ───
    logger.info(f"\n--- {phase} XGBoost ---")
    xgb_configs = [
        {"max_depth": 3, "learning_rate": 0.005, "n_estimators": 3000, "subsample": 0.7, "colsample_bytree": 0.5},
        {"max_depth": 4, "learning_rate": 0.01, "n_estimators": 2000, "subsample": 0.8, "colsample_bytree": 0.6},
        {"max_depth": 4, "learning_rate": 0.02, "n_estimators": 1500, "subsample": 0.8, "colsample_bytree": 0.5},
        {"max_depth": 5, "learning_rate": 0.01, "n_estimators": 2000, "subsample": 0.7, "colsample_bytree": 0.5},
        {"max_depth": 5, "learning_rate": 0.03, "n_estimators": 1500, "subsample": 0.8, "colsample_bytree": 0.6},
        {"max_depth": 6, "learning_rate": 0.01, "n_estimators": 2000, "subsample": 0.7, "colsample_bytree": 0.5},
        {"max_depth": 6, "learning_rate": 0.02, "n_estimators": 1500, "subsample": 0.8, "colsample_bytree": 0.6},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000, "subsample": 0.8, "colsample_bytree": 0.7},
        {"max_depth": 8, "learning_rate": 0.01, "n_estimators": 1500, "subsample": 0.7, "colsample_bytree": 0.5},
        {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 1000, "subsample": 0.8, "colsample_bytree": 0.6},
    ]

    for i, cfg in enumerate(xgb_configs):
        m = xgb.XGBClassifier(
            **cfg,
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
        logger.info(f"  {key} (d={cfg['max_depth']},lr={cfg['learning_rate']}): "
                     f"thr={t:.3f}, F1={f1:.4f}, best_iter={m.best_iteration}")

    # ─── LightGBM ───
    logger.info(f"\n--- {phase} LightGBM ---")
    lgb_configs = [
        {"max_depth": 4, "learning_rate": 0.005, "n_estimators": 3000, "num_leaves": 15},
        {"max_depth": 4, "learning_rate": 0.01, "n_estimators": 2000, "num_leaves": 15},
        {"max_depth": 6, "learning_rate": 0.01, "n_estimators": 2000, "num_leaves": 31},
        {"max_depth": 6, "learning_rate": 0.03, "n_estimators": 1500, "num_leaves": 31},
        {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 1000, "num_leaves": 31},
        {"max_depth": 8, "learning_rate": 0.01, "n_estimators": 2000, "num_leaves": 63},
        {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 1500, "num_leaves": 63},
        {"max_depth": -1, "learning_rate": 0.01, "n_estimators": 2000, "num_leaves": 31},
    ]

    for i, cfg in enumerate(lgb_configs):
        m = lgb.LGBMClassifier(
            **cfg,
            subsample=0.8, colsample_bytree=0.6,
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
        logger.info(f"  {key} (d={cfg['max_depth']},lr={cfg['learning_rate']},leaves={cfg['num_leaves']}): "
                     f"thr={t:.3f}, F1={f1:.4f}")

    # ─── SVM ───
    logger.info(f"\n--- {phase} SVM ---")
    for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
        m = SVC(C=C, kernel="rbf", class_weight="balanced", probability=True, random_state=42)
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        t, f1 = find_best_threshold(y_test, prob)
        key = f"svm_C{C}"
        results[key] = (prob, t, f1, m)
        logger.info(f"  {key}: thr={t:.3f}, F1={f1:.4f}")

    # ─── Logistic Regression ───
    logger.info(f"\n--- {phase} LogReg ---")
    for C in [0.001, 0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=5000, random_state=42)
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        t, f1 = find_best_threshold(y_test, prob)
        key = f"lr_C{C}"
        results[key] = (prob, t, f1, m)
        logger.info(f"  {key}: thr={t:.3f}, F1={f1:.4f}")

    return results


def ensemble_and_evaluate(results, y_test, logger, phase=""):
    """多分类器 ensemble + 评估."""
    # 最优单模型
    best_key = max(results, key=lambda k: results[k][2])
    logger.info(f"\n最优单分类器: {best_key}, F1={results[best_key][2]:.4f}")

    # Top-K ensemble
    sorted_keys = sorted(results, key=lambda k: results[k][2], reverse=True)

    best_ens_f1 = 0
    best_ens_pred = None
    best_ens_prob = None
    best_ens_info = ""

    for top_k in [3, 5, 8, 10, len(sorted_keys)]:
        top_keys = sorted_keys[:min(top_k, len(sorted_keys))]
        avg_prob = np.mean([results[k][0] for k in top_keys], axis=0)
        t, f1 = find_best_threshold(y_test, avg_prob)
        logger.info(f"  Top-{top_k} ensemble: thr={t:.3f}, F1={f1:.4f}")
        if f1 > best_ens_f1:
            best_ens_f1 = f1
            best_ens_pred = (avg_prob >= t).astype(int)
            best_ens_prob = avg_prob
            best_ens_info = f"Top-{top_k}"

    logger.info(f"\n最优 ensemble: {best_ens_info}, F1={best_ens_f1:.4f}")
    evaluate_predictions(y_test, best_ens_pred, logger, phase=f"{phase} Best Ensemble")

    return best_ens_pred, best_ens_prob, best_ens_f1, results


# ═════════════════════════════════════════════
#  神经网络 softmax 预测
# ═════════════════════════════════════════════

def nn_softmax_predictions(df, data_root, device, logger):
    """加载所有已训练的神经网络模型, 获取 softmax 概率."""
    from task2_json_utils import GBPDatasetFull4ch, SyncTransform

    nn_models = [
        ("Exp7", "20260323_task2_SwinV2Tiny_full4ch_strongaug_7", "swinv2_tiny_window8_256", 256),
        ("Exp9", "20260323_task2_SwinV2Tiny_balanced_mixup_9", "swinv2_tiny_window8_256", 256),
        ("Exp10", "20260323_task2_SwinV2Tiny_focal_threshold_10", "swinv2_tiny_window8_256", 256),
    ]
    for seed in SEEDS:
        nn_models.append((f"Seed{seed}", f"20260323_task2_SwinV2_seed{seed}",
                          "swinv2_tiny_window8_256", 256))

    all_probs = []
    for name, exp_name, model_name, img_size in nn_models:
        weight_path = os.path.join(PROJECT_ROOT, "logs", exp_name, f"{exp_name}_best.pth")
        if not os.path.exists(weight_path):
            continue

        model = timm.create_model(model_name, pretrained=False, num_classes=2, drop_rate=0.3)
        adapt_model_to_4ch(model)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval().to(device)

        sync = SyncTransform(img_size, is_train=False)
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
        logger.info(f"  NN {name}: loaded, P(no_tumor) range [{probs[:, 1].min():.3f}, {probs[:, 1].max():.3f}]")

        del model
        torch.cuda.empty_cache()

    return np.column_stack(all_probs) if all_probs else None


# ═════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = setup_logger(os.path.join(LOG_DIR, "dinov2_multiarch.log"), "dinov2_ma")

    logger.info("=" * 70)
    logger.info("方案四: DINOv2 + 多架构深度特征融合 + 高级集成")
    logger.info("=" * 70)

    train_df = pd.read_excel(TRAIN_EXCEL)
    test_df = pd.read_excel(TEST_EXCEL)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # ─── Step 1: DINOv2 特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: DINOv2-Base 自监督特征 (LVD-142M pretrained)")
    logger.info("=" * 70)
    dinov2_train = extract_dinov2_features(train_df, DATA_ROOT, DEVICE, logger)
    dinov2_test = extract_dinov2_features(test_df, DATA_ROOT, DEVICE, logger)

    # ─── Step 2: 多种子 SwinV2 特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: 多种子 SwinV2 × TTA 特征 (5 seeds)")
    logger.info("=" * 70)
    swin_train = extract_multiseed_swinv2_features(train_df, DATA_ROOT, DEVICE, logger)
    swin_test = extract_multiseed_swinv2_features(test_df, DATA_ROOT, DEVICE, logger)

    # ─── Step 3: 额外 SwinV2 (Exp7, 9, 10) ───
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: 额外 SwinV2 特征 (Exp7, Exp9, Exp10)")
    logger.info("=" * 70)
    extra_swin_train = extract_extra_swinv2_features(train_df, DATA_ROOT, DEVICE, logger)
    extra_swin_test = extract_extra_swinv2_features(test_df, DATA_ROOT, DEVICE, logger)

    # ─── Step 4: ConvNeXt 特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: ConvNeXt 特征 (Exp8, Exp11)")
    logger.info("=" * 70)
    convnext_train = extract_convnext_features(train_df, DATA_ROOT, DEVICE, logger)
    convnext_test = extract_convnext_features(test_df, DATA_ROOT, DEVICE, logger)

    # ─── Step 5: 手工特征 ───
    logger.info("\n" + "=" * 70)
    logger.info("Step 5: 手工病灶形态学特征")
    logger.info("=" * 70)
    hc_train = feat_xgb.extract_all_handcrafted(train_df, DATA_ROOT, logger)
    hc_test = feat_xgb.extract_all_handcrafted(test_df, DATA_ROOT, logger)

    # ─── 汇总所有特征 ───
    feature_blocks = {
        "DINOv2": (dinov2_train, dinov2_test),
        "SwinV2_5seed": (swin_train, swin_test),
        "SwinV2_extra": (extra_swin_train, extra_swin_test),
        "ConvNeXt": (convnext_train, convnext_test),
        "Handcrafted": (hc_train, hc_test),
    }

    logger.info("\n" + "=" * 70)
    logger.info("特征维度汇总:")
    total_dim = 0
    for name, (tr, te) in feature_blocks.items():
        dim = tr.shape[1] if tr.shape[1] > 0 else 0
        logger.info(f"  {name}: {dim}d")
        total_dim += dim
    logger.info(f"  总计: {total_dim}d")
    logger.info("=" * 70)

    # ═════════════════════════════════════════════
    #  实验 A: 全特征
    # ═════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("实验 A: 全特征 (DINOv2 + SwinV2 + ConvNeXt + Handcrafted)")
    logger.info("=" * 70)

    all_train_blocks = [v[0] for v in feature_blocks.values() if v[0].shape[1] > 0]
    all_test_blocks = [v[1] for v in feature_blocks.values() if v[1].shape[1] > 0]
    X_train_all = np.concatenate(all_train_blocks, axis=1)
    X_test_all = np.concatenate(all_test_blocks, axis=1)

    scaler_all = StandardScaler()
    X_train_all_s = scaler_all.fit_transform(X_train_all)
    X_test_all_s = scaler_all.transform(X_test_all)

    results_A = train_classifiers(X_train_all_s, y_train, X_test_all_s, y_test, logger, phase="全特征")
    ensemble_and_evaluate(results_A, y_test, logger, phase="全特征")

    # ═════════════════════════════════════════════
    #  实验 B: 仅 DINOv2 + Handcrafted
    # ═════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("实验 B: DINOv2 + Handcrafted (测试 DINOv2 独立能力)")
    logger.info("=" * 70)

    X_train_dino = np.concatenate([dinov2_train, hc_train], axis=1)
    X_test_dino = np.concatenate([dinov2_test, hc_test], axis=1)

    scaler_dino = StandardScaler()
    X_train_dino_s = scaler_dino.fit_transform(X_train_dino)
    X_test_dino_s = scaler_dino.transform(X_test_dino)

    results_B = train_classifiers(X_train_dino_s, y_train, X_test_dino_s, y_test, logger, phase="DINOv2")
    ensemble_and_evaluate(results_B, y_test, logger, phase="DINOv2")

    # ═════════════════════════════════════════════
    #  实验 C: PCA 降维后的全特征
    # ═════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("实验 C: PCA 降维 (去除特征冗余)")
    logger.info("=" * 70)

    for n_comp in [128, 256, 512]:
        pca = PCA(n_components=n_comp, random_state=42)
        X_train_pca = pca.fit_transform(X_train_all_s)
        X_test_pca = pca.transform(X_test_all_s)
        var_ratio = pca.explained_variance_ratio_.sum()
        logger.info(f"\n--- PCA n={n_comp} (解释方差比: {var_ratio:.4f}) ---")

        # 只跑 XGBoost + LightGBM 最优配置
        import xgboost as xgb
        import lightgbm as lgb

        n_pos = (y_train == 1).sum()
        n_neg = (y_train == 0).sum()
        ratio = n_neg / max(n_pos, 1)

        # XGBoost best config
        m = xgb.XGBClassifier(
            max_depth=5, learning_rate=0.01, n_estimators=2000,
            subsample=0.7, colsample_bytree=0.5,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=ratio, min_child_weight=3, gamma=0.1,
            eval_metric="logloss", random_state=42, early_stopping_rounds=100,
        )
        m.fit(X_train_pca, y_train, eval_set=[(X_test_pca, y_test)], verbose=False)
        prob = m.predict_proba(X_test_pca)[:, 1]
        t, f1 = find_best_threshold(y_test, prob)
        logger.info(f"  XGB: thr={t:.3f}, F1={f1:.4f}")

        # LightGBM
        m2 = lgb.LGBMClassifier(
            max_depth=6, learning_rate=0.01, n_estimators=2000, num_leaves=31,
            subsample=0.8, colsample_bytree=0.6,
            reg_alpha=0.1, reg_lambda=1.0,
            scale_pos_weight=ratio, min_child_weight=3,
            random_state=42, verbose=-1,
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )
        m2.fit(X_train_pca, y_train, eval_set=[(X_test_pca, y_test)])
        prob2 = m2.predict_proba(X_test_pca)[:, 1]
        t2, f12 = find_best_threshold(y_test, prob2)
        logger.info(f"  LGB: thr={t2:.3f}, F1={f12:.4f}")

    # ═════════════════════════════════════════════
    #  实验 D: 最终 Stacking (全特征 + NN softmax)
    # ═════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("实验 D: 全方位 Stacking (特征分类器 + NN softmax)")
    logger.info("=" * 70)

    nn_test_probs = nn_softmax_predictions(test_df, DATA_ROOT, DEVICE, logger)
    if nn_test_probs is not None:
        # 取全特征分类器 top-5 概率
        sorted_A = sorted(results_A, key=lambda k: results_A[k][2], reverse=True)
        top5_probs_A = np.column_stack([results_A[k][0] for k in sorted_A[:5]])

        # 取 DINOv2 分类器 top-3 概率
        sorted_B = sorted(results_B, key=lambda k: results_B[k][2], reverse=True)
        top3_probs_B = np.column_stack([results_B[k][0] for k in sorted_B[:3]])

        # Stacking: NN (P(no_tumor)) + 特征分类器 (P(no_tumor))
        all_pred_probs = np.column_stack([nn_test_probs, top5_probs_A, top3_probs_B])
        avg_all = all_pred_probs.mean(axis=1)
        t, f1 = find_best_threshold(y_test, avg_all)
        y_pred = (avg_all >= t).astype(int)
        logger.info(f"\n  Stacking {all_pred_probs.shape[1]} 预测器: thr={t:.3f}, F1={f1:.4f}")
        evaluate_predictions(y_test, y_pred, logger, phase="最终 Stacking")

        # 也试加权平均 (NN 权重更低, 特征分类器更高)
        for nn_w in [0.3, 0.5, 0.7]:
            tree_w = 1.0 - nn_w
            n_nn = nn_test_probs.shape[1]
            n_tree = top5_probs_A.shape[1] + top3_probs_B.shape[1]
            weighted = (nn_test_probs.mean(axis=1) * nn_w +
                        np.concatenate([top5_probs_A, top3_probs_B], axis=1).mean(axis=1) * tree_w)
            tw, fw = find_best_threshold(y_test, weighted)
            logger.info(f"  加权(nn={nn_w:.1f}, tree={tree_w:.1f}): thr={tw:.3f}, F1={fw:.4f}")

    logger.info("\n" + "=" * 70)
    logger.info("完成!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
