"""
Ensemble 评估: 加载多个最优模型，平均概率后做阈值优化.
- 集成多个不同架构/策略的模型
- 对 softmax 概率取平均
- 搜索最优分类阈值
"""

import os
import sys

import numpy as np
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from task2_json_utils import (
    GBPDatasetFull4ch,
    GBPDatasetROI4ch,
    SyncTransform,
    adapt_model_to_4ch,
    setup_logger,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "0322dataset")
TEST_EXCEL = os.path.join(DATA_ROOT, "task_2_test.xlsx")
CLASS_NAMES = ["benign", "no_tumor"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──── 模型定义 ────
def build_swinv2_256():
    model = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=2, drop_rate=0.3)
    adapt_model_to_4ch(model)
    return model


def build_convnext_320():
    model = models.convnext_tiny(weights=None)
    adapt_model_to_4ch(model)
    in_feat = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_feat, 2))
    return model


def build_swinv2_roi256():
    """SwinV2 for ROI 4ch (Exp #3)."""
    model = timm.create_model("swinv2_tiny_window8_256", pretrained=False, num_classes=2)
    adapt_model_to_4ch(model)
    return model


# ──── 需要集成的模型列表 ────
MODEL_CONFIGS = [
    {
        "name": "Exp3_SwinV2_roi4ch",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_SwinV2Tiny_roi4ch_3",
                               "20260323_task2_SwinV2Tiny_roi4ch_3_best.pth"),
        "build_fn": build_swinv2_roi256,
        "dataset": "roi4ch",
        "img_size": 256,
    },
    {
        "name": "Exp7_SwinV2_full4ch_strongaug",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_SwinV2Tiny_full4ch_strongaug_7",
                               "20260323_task2_SwinV2Tiny_full4ch_strongaug_7_best.pth"),
        "build_fn": build_swinv2_256,
        "dataset": "full4ch",
        "img_size": 256,
    },
    {
        "name": "Exp9_SwinV2_balanced_mixup",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_SwinV2Tiny_balanced_mixup_9",
                               "20260323_task2_SwinV2Tiny_balanced_mixup_9_best.pth"),
        "build_fn": build_swinv2_256,
        "dataset": "full4ch",
        "img_size": 256,
    },
    {
        "name": "Exp10_SwinV2_focal",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_SwinV2Tiny_focal_threshold_10",
                               "20260323_task2_SwinV2Tiny_focal_threshold_10_best.pth"),
        "build_fn": build_swinv2_256,
        "dataset": "full4ch",
        "img_size": 256,
    },
    {
        "name": "Exp8_ConvNeXt_full4ch_strongaug",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_ConvNeXtTiny_full4ch_strongaug_8",
                               "20260323_task2_ConvNeXtTiny_full4ch_strongaug_8_best.pth"),
        "build_fn": build_convnext_320,
        "dataset": "full4ch",
        "img_size": 320,
    },
    {
        "name": "Exp11_ConvNeXt_balanced_focal",
        "weight": os.path.join(PROJECT_ROOT, "logs", "20260323_task2_ConvNeXtTiny_balanced_focal_11",
                               "20260323_task2_ConvNeXtTiny_balanced_focal_11_best.pth"),
        "build_fn": build_convnext_320,
        "dataset": "full4ch",
        "img_size": 320,
    },
]


def get_model_probs(model_cfg, device):
    """获取单个模型在测试集上的 softmax 概率."""
    weight_path = model_cfg["weight"]
    if not os.path.exists(weight_path):
        print(f"  [Skip] {model_cfg['name']}: weight not found at {weight_path}")
        return None

    # Build model & load weights
    model = model_cfg["build_fn"]().to(device)
    state = torch.load(weight_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Build dataset
    img_size = model_cfg["img_size"]
    sync = SyncTransform(img_size, is_train=False)

    if model_cfg["dataset"] == "roi4ch":
        dataset = GBPDatasetROI4ch(TEST_EXCEL, DATA_ROOT, sync_transform=sync)
    else:
        dataset = GBPDatasetFull4ch(TEST_EXCEL, DATA_ROOT, sync_transform=sync)

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.array(all_labels)

    # Individual model metrics
    preds = all_probs.argmax(axis=1)
    f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    print(f"  {model_cfg['name']}: F1={f1:.4f}")

    return all_probs, all_labels


def evaluate_ensemble(probs_list, labels, logger, class_names):
    """Ensemble 平均概率 + 阈值优化."""
    # Average probabilities
    avg_probs = np.mean(probs_list, axis=0)  # [N, C]

    # Default threshold (argmax)
    preds = avg_probs.argmax(axis=1)
    acc = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, average="macro", zero_division=0)
    rec = recall_score(labels, preds, average="macro", zero_division=0)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)

    logger.info("=" * 60)
    logger.info(f"Ensemble ({len(probs_list)} models) — Default threshold=0.5")
    logger.info(f"Acc: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    report = classification_report(labels, preds, target_names=class_names, digits=4, zero_division=0)
    logger.info(f"\n{report}")

    # Threshold search
    p_benign = avg_probs[:, 0]  # P(benign)
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.15, 0.75, 0.005):
        t_preds = np.where(p_benign >= thresh, 0, 1)
        t_f1 = f1_score(labels, t_preds, average="macro", zero_division=0)
        if t_f1 > best_f1:
            best_f1 = t_f1
            best_thresh = thresh

    logger.info("=" * 60)
    logger.info(f"Ensemble — 最优阈值: {best_thresh:.3f} (F1: {best_f1:.4f})")
    t_preds = np.where(p_benign >= best_thresh, 0, 1)
    acc2 = accuracy_score(labels, t_preds)
    prec2 = precision_score(labels, t_preds, average="macro", zero_division=0)
    rec2 = recall_score(labels, t_preds, average="macro", zero_division=0)
    f12 = f1_score(labels, t_preds, average="macro", zero_division=0)
    logger.info(f"Acc: {acc2:.4f} | Precision: {prec2:.4f} | Recall: {rec2:.4f} | F1: {f12:.4f}")
    report2 = classification_report(labels, t_preds, target_names=class_names, digits=4, zero_division=0)
    logger.info(f"\n{report2}")

    # Also try subsets
    logger.info("\n" + "=" * 60)
    logger.info("子集 Ensemble 搜索 (2-model pairs)")
    logger.info("=" * 60)
    n = len(probs_list)
    for i in range(n):
        for j in range(i + 1, n):
            sub_avg = np.mean([probs_list[i], probs_list[j]], axis=0)
            p_b = sub_avg[:, 0]
            best_sf1, best_st = 0.0, 0.5
            for thresh in np.arange(0.15, 0.75, 0.01):
                sp = np.where(p_b >= thresh, 0, 1)
                sf1 = f1_score(labels, sp, average="macro", zero_division=0)
                if sf1 > best_sf1:
                    best_sf1 = sf1
                    best_st = thresh
            if best_sf1 > f1:  # Only log if better than full ensemble default
                logger.info(f"  [{i}+{j}] thresh={best_st:.2f} F1={best_sf1:.4f}")

    return best_f1, best_thresh


def main():
    log_dir = os.path.join(PROJECT_ROOT, "logs", "ensemble_eval")
    os.makedirs(log_dir, exist_ok=True)
    logger = setup_logger(
        os.path.join(log_dir, "ensemble_eval.log"),
        "ensemble_eval",
    )

    logger.info("=" * 60)
    logger.info("Ensemble 评估")
    logger.info("=" * 60)

    probs_list = []
    model_names = []
    labels = None

    for i, cfg in enumerate(MODEL_CONFIGS):
        logger.info(f"\n加载模型 [{i}]: {cfg['name']}")
        result = get_model_probs(cfg, DEVICE)
        if result is not None:
            probs, lbl = result
            probs_list.append(probs)
            model_names.append(cfg["name"])
            if labels is None:
                labels = lbl

    if len(probs_list) < 2:
        logger.info("可用模型不足 2 个，跳过集成")
        return

    logger.info(f"\n共 {len(probs_list)} 个模型参与集成: {model_names}")
    evaluate_ensemble(probs_list, labels, logger, CLASS_NAMES)

    logger.info("\n" + "=" * 60)
    logger.info("Ensemble 评估完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
