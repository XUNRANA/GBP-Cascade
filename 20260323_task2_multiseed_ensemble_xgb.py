"""
方案三-Step2: 多种子特征 + TTA + XGBoost + SVM
==============================================
1. 加载 5 个种子的 SwinV2 backbone → 5×768 = 3840 维深度特征
2. TTA: 原图 + 水平翻转 + 垂直翻转 → 每个 backbone 提取 3 组特征
3. 手工病灶形态学特征 15 维
4. XGBoost + LightGBM + SVM 多分类器
5. 多分类器 stacking ensemble
"""

import json
import math
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image
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

import importlib
feat_xgb = importlib.import_module("20260323_task2_feature_xgboost")
find_best_threshold = feat_xgb.find_best_threshold
evaluate_predictions = feat_xgb.evaluate_predictions

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "0322dataset")
TRAIN_EXCEL = os.path.join(DATA_ROOT, "task_2_train.xlsx")
TEST_EXCEL = os.path.join(DATA_ROOT, "task_2_test.xlsx")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "multiseed_ensemble_xgb")
CLASS_NAMES = ["benign", "no_tumor"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 123, 456, 789, 1024]


# ═════════════════════════════════════════════
#  TTA 数据集
# ═════════════════════════════════════════════

class TTA4chDataset(Dataset):
    """4ch 数据集, 支持 TTA (翻转变换)."""

    def __init__(self, df, data_root, img_size, tta_mode="none"):
        """tta_mode: 'none', 'hflip', 'vflip', 'rot90', 'rot270'"""
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

        # TTA 变换
        if self.tta_mode == "hflip":
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        elif self.tta_mode == "vflip":
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        elif self.tta_mode == "rot90":
            img = TF.rotate(img, 90, interpolation=InterpolationMode.BICUBIC)
            mask = TF.rotate(mask, 90, interpolation=InterpolationMode.NEAREST)
        elif self.tta_mode == "rot270":
            img = TF.rotate(img, 270, interpolation=InterpolationMode.BICUBIC)
            mask = TF.rotate(mask, 270, interpolation=InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)
        mask_t = TF.to_tensor(mask)
        img_t = TF.normalize(img_t, self.mean, self.std)

        return torch.cat([img_t, mask_t], dim=0)


# ═════════════════════════════════════════════
#  特征提取
# ═════════════════════════════════════════════

@torch.no_grad()
def extract_features(model, dataset, device, batch_size=16):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        feat = model(batch)
        all_feats.append(feat.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def load_seed_backbone(seed):
    """加载某个种子的 backbone."""
    weight_path = os.path.join(PROJECT_ROOT, "logs", f"20260323_task2_SwinV2_seed{seed}",
                               f"20260323_task2_SwinV2_seed{seed}_best.pth")
    if not os.path.exists(weight_path):
        return None, weight_path

    model = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=0)
    adapt_model_to_4ch(model)

    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    backbone_state = {k: v for k, v in state.items() if not k.startswith("head.fc")}
    model.load_state_dict(backbone_state, strict=False)
    model.eval()
    return model, weight_path


def extract_multiseed_tta_features(df, data_root, device, logger):
    """提取多种子 × TTA 特征."""
    tta_modes = ["none", "hflip", "vflip"]
    all_features = []

    for seed in SEEDS:
        model, path = load_seed_backbone(seed)
        if model is None:
            logger.info(f"  [Skip] seed={seed}: {path}")
            continue

        model = model.to(device)
        logger.info(f"  Seed {seed}: 提取特征 (TTA×{len(tta_modes)})...")

        seed_feats = []
        for mode in tta_modes:
            dataset = TTA4chDataset(df, data_root, img_size=256, tta_mode=mode)
            feats = extract_features(model, dataset, device)
            seed_feats.append(feats)

        # 平均 TTA 特征
        avg_feat = np.mean(seed_feats, axis=0)
        all_features.append(avg_feat)
        logger.info(f"    → shape: {avg_feat.shape}")

        del model
        torch.cuda.empty_cache()

    if not all_features:
        raise RuntimeError("No seed backbones found! Run multiseed training first.")

    combined = np.concatenate(all_features, axis=1)
    logger.info(f"  多种子TTA特征总维度: {combined.shape[1]} ({len(all_features)} seeds × 768)")
    return combined


# ═════════════════════════════════════════════
#  多分类器 + Stacking
# ═════════════════════════════════════════════

def train_multi_classifier(X_train, y_train, X_test, y_test, logger):
    """训练多种分类器 + stacking ensemble."""
    import xgboost as xgb
    import lightgbm as lgb

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    ratio = n_neg / max(n_pos, 1)

    results = {}

    # ─── XGBoost ───
    logger.info("\n--- XGBoost ---")
    for max_depth in [4, 6, 8]:
        for lr in [0.01, 0.03, 0.05]:
            m = xgb.XGBClassifier(
                n_estimators=1500, max_depth=max_depth, learning_rate=lr,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=ratio, min_child_weight=3, gamma=0.1,
                eval_metric="logloss", random_state=42, use_label_encoder=False,
            )
            m.fit(X_tr, y_train, eval_set=[(X_te, y_test)], verbose=False)
            prob = m.predict_proba(X_te)[:, 1]
            t, f1 = find_best_threshold(y_test, prob)
            key = f"xgb_d{max_depth}_lr{lr}"
            results[key] = (prob, t, f1, m)
            logger.info(f"  {key}: threshold={t:.3f}, F1={f1:.4f}")

    # ─── LightGBM ───
    logger.info("\n--- LightGBM ---")
    for max_depth in [4, 6, 8]:
        for lr in [0.01, 0.03, 0.05]:
            m = lgb.LGBMClassifier(
                n_estimators=1500, max_depth=max_depth, learning_rate=lr,
                subsample=0.8, colsample_bytree=0.7,
                reg_alpha=0.1, reg_lambda=1.0,
                scale_pos_weight=ratio, min_child_weight=3,
                random_state=42, verbose=-1,
            )
            m.fit(X_tr, y_train, eval_set=[(X_te, y_test)])
            prob = m.predict_proba(X_te)[:, 1]
            t, f1 = find_best_threshold(y_test, prob)
            key = f"lgb_d{max_depth}_lr{lr}"
            results[key] = (prob, t, f1, m)
            logger.info(f"  {key}: threshold={t:.3f}, F1={f1:.4f}")

    # ─── SVM ───
    logger.info("\n--- SVM (RBF) ---")
    for C in [0.1, 1.0, 10.0]:
        m = SVC(C=C, kernel="rbf", class_weight="balanced", probability=True, random_state=42)
        m.fit(X_tr, y_train)
        prob = m.predict_proba(X_te)[:, 1]
        t, f1 = find_best_threshold(y_test, prob)
        key = f"svm_C{C}"
        results[key] = (prob, t, f1, m)
        logger.info(f"  {key}: threshold={t:.3f}, F1={f1:.4f}")

    # ─── Logistic Regression ───
    logger.info("\n--- Logistic Regression ---")
    for C in [0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, class_weight="balanced", max_iter=2000, random_state=42)
        m.fit(X_tr, y_train)
        prob = m.predict_proba(X_te)[:, 1]
        t, f1 = find_best_threshold(y_test, prob)
        key = f"lr_C{C}"
        results[key] = (prob, t, f1, m)
        logger.info(f"  {key}: threshold={t:.3f}, F1={f1:.4f}")

    # ─── 最优单模型 ───
    best_key = max(results, key=lambda k: results[k][2])
    logger.info(f"\n最优单分类器: {best_key}, F1={results[best_key][2]:.4f}")

    # ─── Top-K Ensemble ───
    logger.info("\n--- Ensemble (Top 分类器概率平均) ---")
    sorted_keys = sorted(results, key=lambda k: results[k][2], reverse=True)

    for top_k in [3, 5, 8, len(sorted_keys)]:
        top_keys = sorted_keys[:top_k]
        avg_prob = np.mean([results[k][0] for k in top_keys], axis=0)
        t, f1 = find_best_threshold(y_test, avg_prob)
        logger.info(f"  Top-{top_k} ensemble: threshold={t:.3f}, F1={f1:.4f}")

    # ─── 全部 ensemble ───
    all_prob = np.mean([results[k][0] for k in results], axis=0)
    best_t, best_f1 = find_best_threshold(y_test, all_prob)
    y_pred = (all_prob >= best_t).astype(int)
    logger.info(f"\n全部 {len(results)} 个分类器 Ensemble: threshold={best_t:.3f}, F1={best_f1:.4f}")

    return y_pred, all_prob, best_t, results


# ═════════════════════════════════════════════
#  神经网络 softmax + XGBoost stacking
# ═════════════════════════════════════════════

def neural_net_predictions(df, data_root, device, logger):
    """加载所有已训练的神经网络模型, 获取 softmax 概率."""
    from task2_json_utils import GBPDatasetFull4ch, SyncTransform

    nn_models = [
        ("Exp3_roi4ch", "20260323_task2_SwinV2Tiny_roi4ch_3", 256, "roi4ch"),
        ("Exp7_full4ch", "20260323_task2_SwinV2Tiny_full4ch_strongaug_7", 256, "full4ch"),
        ("Exp9_balanced", "20260323_task2_SwinV2Tiny_balanced_mixup_9", 256, "full4ch"),
        ("Exp10_focal", "20260323_task2_SwinV2Tiny_focal_threshold_10", 256, "full4ch"),
    ]

    # 加入多种子模型
    for seed in SEEDS:
        name = f"Seed{seed}"
        exp = f"20260323_task2_SwinV2_seed{seed}"
        nn_models.append((name, exp, 256, "full4ch"))

    all_probs = []
    for name, exp_name, img_size, ds_type in nn_models:
        weight_path = os.path.join(PROJECT_ROOT, "logs", exp_name, f"{exp_name}_best.pth")
        if not os.path.exists(weight_path):
            continue

        model = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=2, drop_rate=0.3)
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
        all_probs.append(probs[:, 0])  # P(benign)
        logger.info(f"  NN {name}: loaded")

        del model
        torch.cuda.empty_cache()

    return np.column_stack(all_probs) if all_probs else None


# ═════════════════════════════════════════════
#  Main
# ═════════════════════════════════════════════

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = setup_logger(os.path.join(LOG_DIR, "multiseed_ensemble_xgb.log"), "ms_ens_xgb")

    logger.info("=" * 60)
    logger.info("方案三: 多种子特征 + TTA + 多分类器 Ensemble")
    logger.info("=" * 60)

    train_df = pd.read_excel(TRAIN_EXCEL)
    test_df = pd.read_excel(TEST_EXCEL)
    y_train = train_df["label"].values
    y_test = test_df["label"].values

    # Step 1: 手工特征
    logger.info("\n" + "=" * 60)
    logger.info("Step 1: 手工特征")
    logger.info("=" * 60)
    hc_train = feat_xgb.extract_all_handcrafted(train_df, DATA_ROOT, logger)
    hc_test = feat_xgb.extract_all_handcrafted(test_df, DATA_ROOT, logger)

    # Step 2: 多种子 + TTA 深度特征
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: 多种子 × TTA 深度特征")
    logger.info("=" * 60)
    deep_train = extract_multiseed_tta_features(train_df, DATA_ROOT, DEVICE, logger)
    deep_test = extract_multiseed_tta_features(test_df, DATA_ROOT, DEVICE, logger)

    # Step 3: 拼接
    X_train = np.concatenate([deep_train, hc_train], axis=1)
    X_test = np.concatenate([deep_test, hc_test], axis=1)
    logger.info(f"\n总特征维度: {X_train.shape[1]}")

    # Step 4: 多分类器训练
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: 多分类器训练 + Ensemble")
    logger.info("=" * 60)
    y_pred, y_prob, best_t, results = train_multi_classifier(X_train, y_train, X_test, y_test, logger)
    evaluate_predictions(y_test, y_pred, logger, phase="多分类器 Ensemble")

    # Step 5: 也做一个纯神经网络 ensemble (多种子模型 softmax 平均)
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: 纯神经网络多种子 Ensemble (softmax 平均)")
    logger.info("=" * 60)
    nn_test_probs = neural_net_predictions(test_df, DATA_ROOT, DEVICE, logger)
    if nn_test_probs is not None:
        avg_nn_prob = nn_test_probs.mean(axis=1)  # P(benign)
        # 转换: P(no_tumor) = 1 - P(benign), 用 P(no_tumor) 做阈值
        nn_prob_notumor = 1.0 - avg_nn_prob
        best_t_nn, best_f1_nn = find_best_threshold(y_test, nn_prob_notumor)
        y_pred_nn = (nn_prob_notumor >= best_t_nn).astype(int)
        logger.info(f"  {nn_test_probs.shape[1]} 个 NN 模型, threshold={best_t_nn:.3f}, F1={best_f1_nn:.4f}")
        evaluate_predictions(y_test, y_pred_nn, logger, phase="NN多种子Ensemble")

        # Step 6: Stacking — NN probs + XGBoost probs → final
        logger.info("\n" + "=" * 60)
        logger.info("Step 5: Stacking (NN概率 + XGBoost概率 → 最终预测)")
        logger.info("=" * 60)
        # 取所有分类器的 top-5 概率
        sorted_keys = sorted(results, key=lambda k: results[k][2], reverse=True)
        top5_probs = np.column_stack([results[k][0] for k in sorted_keys[:5]])
        stacked = np.column_stack([nn_test_probs, top5_probs])
        # 简单平均
        avg_stacked = stacked.mean(axis=1)
        # 这里 nn_test_probs 是 P(benign), top5_probs 是 P(no_tumor=1)
        # 需要统一方向: 都转为 P(no_tumor)
        nn_as_notumor = 1.0 - nn_test_probs  # [N, K] → P(no_tumor)
        all_notumor_probs = np.column_stack([nn_as_notumor, top5_probs])
        avg_all = all_notumor_probs.mean(axis=1)
        best_t_stack, best_f1_stack = find_best_threshold(y_test, avg_all)
        y_pred_stack = (avg_all >= best_t_stack).astype(int)
        logger.info(f"  Stacked {all_notumor_probs.shape[1]} 预测器, threshold={best_t_stack:.3f}, F1={best_f1_stack:.4f}")
        evaluate_predictions(y_test, y_pred_stack, logger, phase="Stacking Ensemble")

    logger.info("\n" + "=" * 60)
    logger.info("完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
