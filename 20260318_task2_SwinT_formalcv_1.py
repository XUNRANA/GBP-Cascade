"""
Task 2 正式版: Swin-T + 患者级 5 折 CV + 内层验证集

目标:
1. 外层 5 折 patient-level cross-validation，降低单次划分方差
2. 内层 patient-level 验证集，仅用于选 epoch 和阈值
3. 同时输出 image-level / patient-level 指标
4. 统一记录 AUC、PR-AUC、macro F1、混淆矩阵
5. 提供增强 / 采样 / 正则化的消融开关
"""

import os
import sys
import time
import shutil
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


@dataclass
class Config:
    project_root: str = os.path.dirname(os.path.abspath(__file__))
    data_root: str = os.path.join(project_root, "dataset", "Processed")
    train_excel_orig: str = os.path.join(data_root, "task_2_train.xlsx")
    test_excel_orig: str = os.path.join(data_root, "task_2_test.xlsx")

    exp_name: str = "20260318_task2_SwinT_formalcv_1"
    log_dir: str = os.path.join(project_root, "logs", exp_name)
    log_file: str = os.path.join(log_dir, f"{exp_name}.log")

    img_size: int = 224
    batch_size: int = 32
    num_epochs: int = 40
    lr: float = 2e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 3
    min_lr: float = 1e-6
    num_workers: int = 4
    eval_interval: int = 1
    early_stop_patience: int = 8

    outer_folds: int = 5
    inner_val_ratio: float = 0.15
    seed: int = 42

    dropout: float = 0.3
    threshold_grid_start: float = 0.20
    threshold_grid_end: float = 0.81
    threshold_grid_step: float = 0.01

    use_weighted_sampler: bool = True
    use_random_resized_crop: bool = True
    crop_scale_min: float = 0.85
    crop_scale_max: float = 1.00
    use_horizontal_flip: bool = True
    use_vertical_flip: bool = False
    use_randaugment: bool = False
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 7
    use_random_erasing: bool = False
    random_erasing_p: float = 0.10
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_label_smoothing: bool = False
    label_smoothing: float = 0.1

    patient_agg_mode: str = "mean_prob"  # mean_prob or vote_fraction
    vote_image_threshold: float = 0.5

    save_fold_predictions: bool = True
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names: tuple = ("benign", "no_tumor")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file):
    logger = logging.getLogger("task2_formalcv")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    stream_handler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_all_data(cfg):
    train_df = pd.read_excel(cfg.train_excel_orig)
    test_df = pd.read_excel(cfg.test_excel_orig)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    all_df["pid"] = all_df["image_path"].apply(lambda x: x.split("/")[-1].split("_")[0])

    patient_label_check = all_df.groupby("pid")["label"].nunique()
    bad_patients = patient_label_check[patient_label_check > 1]
    if not bad_patients.empty:
        raise ValueError(f"发现标签不一致患者: {bad_patients.index.tolist()[:10]}")

    return all_df


def build_patient_table(all_df):
    patient_df = (
        all_df.groupby("pid")
        .agg(label=("label", "first"), n_images=("image_path", "size"))
        .reset_index()
    )
    return patient_df


def log_dataset_overview(all_df, patient_df, logger):
    logger.info("=" * 60)
    logger.info("实验名称: 20260318_task2_SwinT_formalcv_1")
    logger.info("任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info("模式: 外层 5 折 patient-level CV + 内层 patient-level val")
    logger.info(f"设备: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    logger.info("=" * 60)
    logger.info(f"总图像: {len(all_df)}")
    logger.info(f"总患者: {len(patient_df)}")
    logger.info(
        "图像级类别分布: "
        f"benign={int((all_df['label'] == 0).sum())}, "
        f"no_tumor={int((all_df['label'] == 1).sum())}"
    )
    logger.info(
        "患者级类别分布: "
        f"benign={int((patient_df['label'] == 0).sum())}, "
        f"no_tumor={int((patient_df['label'] == 1).sum())}"
    )
    logger.info(f"每患者图像数均值: {patient_df['n_images'].mean():.3f}")
    logger.info(f"每患者图像数中位数: {patient_df['n_images'].median():.1f}")


def build_outer_folds(patient_df, cfg):
    splitter = StratifiedGroupKFold(
        n_splits=cfg.outer_folds,
        shuffle=True,
        random_state=cfg.seed,
    )
    indices = []
    x_dummy = np.zeros(len(patient_df))
    y = patient_df["label"].values
    groups = patient_df["pid"].values

    for fold_id, (train_idx, test_idx) in enumerate(
        splitter.split(x_dummy, y, groups=groups),
        start=1,
    ):
        indices.append((fold_id, train_idx, test_idx))
    return indices


def split_inner_train_val(patient_train_df, cfg):
    inner_train_idx, inner_val_idx = train_test_split(
        np.arange(len(patient_train_df)),
        test_size=cfg.inner_val_ratio,
        stratify=patient_train_df["label"].values,
        random_state=cfg.seed,
    )
    inner_train_patients = patient_train_df.iloc[inner_train_idx].reset_index(drop=True)
    inner_val_patients = patient_train_df.iloc[inner_val_idx].reset_index(drop=True)
    return inner_train_patients, inner_val_patients


class GBPDataset(Dataset):
    def __init__(self, df, data_root, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(row["label"]), row["pid"], row["image_path"]


def build_train_transform(cfg):
    transform_list = [transforms.Resize((256, 256))]

    if cfg.use_random_resized_crop:
        transform_list.append(
            transforms.RandomResizedCrop(
                cfg.img_size,
                scale=(cfg.crop_scale_min, cfg.crop_scale_max),
                ratio=(0.95, 1.05),
            )
        )
    else:
        transform_list.append(transforms.CenterCrop(cfg.img_size))

    if cfg.use_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if cfg.use_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    if cfg.use_randaugment:
        transform_list.append(
            transforms.RandAugment(
                num_ops=cfg.randaugment_num_ops,
                magnitude=cfg.randaugment_magnitude,
            )
        )

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if cfg.use_random_erasing:
        transform_list.append(
            transforms.RandomErasing(
                p=cfg.random_erasing_p,
                scale=(0.02, 0.10),
            )
        )

    return transforms.Compose(transform_list)


def build_eval_transform(cfg):
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_model(cfg):
    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

    for i in range(6):
        for param in model.features[i].parameters():
            param.requires_grad = False

    in_features = model.head.in_features
    model.head = nn.Sequential(
        nn.Dropout(p=cfg.dropout),
        nn.Linear(in_features, 2),
    )
    return model


def build_scheduler(optimizer, cfg):
    if cfg.warmup_epochs > 0:
        warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(cfg.num_epochs - cfg.warmup_epochs, 1),
            eta_min=cfg.min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[cfg.warmup_epochs],
        )

    return CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=cfg.min_lr)


def mixup_data(images, labels, alpha):
    if alpha <= 0:
        return images, labels, labels, 1.0

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    return mixed_images, labels, labels[index], lam


def train_one_epoch(model, dataloader, criterion, optimizer, cfg):
    model.train()
    running_loss = 0.0
    correct = 0.0
    total = 0

    for images, labels, _, _ in dataloader:
        images = images.to(cfg.device)
        labels = labels.to(cfg.device)

        optimizer.zero_grad()

        if cfg.use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, cfg.mixup_alpha)
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (
                lam * (preds == labels_a).float().sum().item()
                + (1.0 - lam) * (preds == labels_b).float().sum().item()
            )
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total, correct / total


def infer_probabilities(model, dataloader, cfg):
    model.eval()
    rows = []

    with torch.no_grad():
        for images, labels, pids, image_paths in dataloader:
            images = images.to(cfg.device)
            outputs = model(images)
            probs_benign = torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()

            for pid, image_path, label, prob in zip(pids, image_paths, labels.numpy(), probs_benign):
                rows.append(
                    {
                        "pid": pid,
                        "image_path": image_path,
                        "label": int(label),
                        "p_benign": float(prob),
                    }
                )

    return pd.DataFrame(rows)


def safe_auc(labels, scores):
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.nan
    return roc_auc_score(labels, scores)


def safe_ap(labels, scores):
    unique = np.unique(labels)
    if len(unique) < 2:
        return np.nan
    return average_precision_score(labels, scores)


def compute_binary_metrics(labels, scores_benign, threshold, class_names):
    labels = np.asarray(labels)
    scores_benign = np.asarray(scores_benign)
    preds = np.where(scores_benign >= threshold, 0, 1)
    benign_labels = (labels == 0).astype(int)

    report_text = classification_report(
        labels,
        preds,
        target_names=list(class_names),
        digits=4,
        zero_division=0,
    )
    report_dict = classification_report(
        labels,
        preds,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "acc": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "auc_benign": float(safe_auc(benign_labels, scores_benign)),
        "pr_auc_benign": float(safe_ap(benign_labels, scores_benign)),
        "confusion_matrix": cm.tolist(),
        "report_text": report_text,
        "report_dict": report_dict,
    }


def aggregate_patient_predictions(pred_df, cfg):
    work_df = pred_df.copy()

    if cfg.patient_agg_mode == "mean_prob":
        patient_df = (
            work_df.groupby("pid")
            .agg(
                label=("label", "first"),
                patient_score=("p_benign", "mean"),
                n_images=("image_path", "size"),
            )
            .reset_index()
        )
    elif cfg.patient_agg_mode == "vote_fraction":
        work_df["benign_vote"] = (work_df["p_benign"] >= cfg.vote_image_threshold).astype(float)
        patient_df = (
            work_df.groupby("pid")
            .agg(
                label=("label", "first"),
                patient_score=("benign_vote", "mean"),
                n_images=("image_path", "size"),
            )
            .reset_index()
        )
    else:
        raise ValueError(f"未知 patient_agg_mode: {cfg.patient_agg_mode}")

    return patient_df


def evaluate_prediction_frame(pred_df, cfg, logger, phase, threshold):
    image_metrics = compute_binary_metrics(
        labels=pred_df["label"].values,
        scores_benign=pred_df["p_benign"].values,
        threshold=threshold,
        class_names=cfg.class_names,
    )

    patient_df = aggregate_patient_predictions(pred_df, cfg)
    patient_metrics = compute_binary_metrics(
        labels=patient_df["label"].values,
        scores_benign=patient_df["patient_score"].values,
        threshold=threshold,
        class_names=cfg.class_names,
    )

    log_metric_block(logger, phase, "Image", image_metrics)
    log_metric_block(logger, phase, "Patient", patient_metrics)

    return {
        "image": image_metrics,
        "patient": patient_metrics,
        "patient_df": patient_df,
    }


def log_metric_block(logger, phase, level_name, metrics):
    logger.info(
        f"[{phase}][{level_name}] Thresh: {metrics['threshold']:.2f} | "
        f"Acc: {metrics['acc']:.4f} | "
        f"Precision(macro): {metrics['precision_macro']:.4f} | "
        f"Recall(macro): {metrics['recall_macro']:.4f} | "
        f"F1(macro): {metrics['f1_macro']:.4f}"
    )
    logger.info(
        f"[{phase}][{level_name}] AUC(benign): {metrics['auc_benign']:.4f} | "
        f"PR-AUC(benign): {metrics['pr_auc_benign']:.4f}"
    )
    logger.info(
        f"[{phase}][{level_name}] Confusion Matrix "
        f"(rows=true[benign,no_tumor], cols=pred[benign,no_tumor]): "
        f"{metrics['confusion_matrix']}"
    )
    logger.info(f"[{phase}][{level_name}] Classification Report:\n{metrics['report_text']}")


def find_optimal_threshold(pred_df, cfg):
    patient_df = aggregate_patient_predictions(pred_df, cfg)

    best_threshold = 0.5
    best_f1 = -1.0
    grid = np.arange(
        cfg.threshold_grid_start,
        cfg.threshold_grid_end,
        cfg.threshold_grid_step,
    )

    for threshold in grid:
        metrics = compute_binary_metrics(
            labels=patient_df["label"].values,
            scores_benign=patient_df["patient_score"].values,
            threshold=threshold,
            class_names=cfg.class_names,
        )
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_threshold = float(threshold)

    return best_threshold, best_f1


def make_dataloader(df, transform, cfg, shuffle=False, sampler=None):
    dataset = GBPDataset(df, cfg.data_root, transform=transform)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_train_sampler(train_df, cfg):
    if not cfg.use_weighted_sampler:
        return None

    labels = train_df["label"].values
    class_counts = np.bincount(labels)
    sample_weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True,
    )


def save_predictions(pred_df, patient_df, save_prefix):
    pred_df.to_csv(f"{save_prefix}_image_predictions.csv", index=False)
    patient_df.to_csv(f"{save_prefix}_patient_predictions.csv", index=False)


def fold_patient_summary(df, prefix):
    return {
        f"{prefix}_patients": int(df["pid"].nunique()),
        f"{prefix}_images": int(len(df)),
        f"{prefix}_benign_images": int((df["label"] == 0).sum()),
        f"{prefix}_no_tumor_images": int((df["label"] == 1).sum()),
    }


def run_one_fold(fold_id, all_df, patient_df, train_idx, outer_test_idx, cfg, logger):
    fold_dir = os.path.join(cfg.log_dir, f"fold_{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    best_weight_path = os.path.join(fold_dir, f"fold_{fold_id}_best.pth")

    outer_train_patients = patient_df.iloc[train_idx].reset_index(drop=True)
    outer_test_patients = patient_df.iloc[outer_test_idx].reset_index(drop=True)
    inner_train_patients, inner_val_patients = split_inner_train_val(outer_train_patients, cfg)

    inner_train_df = all_df[all_df["pid"].isin(set(inner_train_patients["pid"]))].reset_index(drop=True)
    inner_val_df = all_df[all_df["pid"].isin(set(inner_val_patients["pid"]))].reset_index(drop=True)
    outer_test_df = all_df[all_df["pid"].isin(set(outer_test_patients["pid"]))].reset_index(drop=True)

    logger.info("\n" + "=" * 60)
    logger.info(f"开始 Fold {fold_id}/{cfg.outer_folds}")
    logger.info("=" * 60)

    fold_stats = {}
    fold_stats.update(fold_patient_summary(inner_train_df, "inner_train"))
    fold_stats.update(fold_patient_summary(inner_val_df, "inner_val"))
    fold_stats.update(fold_patient_summary(outer_test_df, "outer_test"))

    logger.info(
        f"[Fold {fold_id}] inner_train: {fold_stats['inner_train_images']}张 / "
        f"{fold_stats['inner_train_patients']}患者 "
        f"(benign={fold_stats['inner_train_benign_images']}, "
        f"no_tumor={fold_stats['inner_train_no_tumor_images']})"
    )
    logger.info(
        f"[Fold {fold_id}] inner_val: {fold_stats['inner_val_images']}张 / "
        f"{fold_stats['inner_val_patients']}患者 "
        f"(benign={fold_stats['inner_val_benign_images']}, "
        f"no_tumor={fold_stats['inner_val_no_tumor_images']})"
    )
    logger.info(
        f"[Fold {fold_id}] outer_test: {fold_stats['outer_test_images']}张 / "
        f"{fold_stats['outer_test_patients']}患者 "
        f"(benign={fold_stats['outer_test_benign_images']}, "
        f"no_tumor={fold_stats['outer_test_no_tumor_images']})"
    )

    train_transform = build_train_transform(cfg)
    eval_transform = build_eval_transform(cfg)

    train_sampler = build_train_sampler(inner_train_df, cfg)
    train_loader = make_dataloader(inner_train_df, train_transform, cfg, shuffle=train_sampler is None, sampler=train_sampler)
    inner_val_loader = make_dataloader(inner_val_df, eval_transform, cfg, shuffle=False)
    outer_test_loader = make_dataloader(outer_test_df, eval_transform, cfg, shuffle=False)

    model = build_model(cfg).to(cfg.device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"[Fold {fold_id}] 模型参数量: {total_params:,}, "
        f"可训练参数量: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)"
    )

    label_smoothing = cfg.label_smoothing if cfg.use_label_smoothing else 0.0
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg)

    best_epoch = 0
    best_val_f1 = -1.0
    epochs_without_improve = 0

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, cfg)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start

        logger.info(
            f"[Fold {fold_id}] Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {current_lr:.6f} | Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | Time: {epoch_time:.1f}s"
        )

        if epoch % cfg.eval_interval != 0:
            continue

        val_pred_df = infer_probabilities(model, inner_val_loader, cfg)
        val_results = evaluate_prediction_frame(
            val_pred_df,
            cfg,
            logger,
            phase=f"Fold{fold_id}-InnerVal-Epoch{epoch}",
            threshold=0.50,
        )
        current_val_f1 = val_results["patient"]["f1_macro"]

        if current_val_f1 > best_val_f1:
            best_val_f1 = current_val_f1
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_weight_path)
            logger.info(
                f"[Fold {fold_id}] *** 保存最优模型 "
                f"(Inner Val Patient F1: {best_val_f1:.4f}, Epoch: {best_epoch}) ***"
            )
        else:
            epochs_without_improve += 1
            logger.info(
                f"[Fold {fold_id}] 未提升轮数: "
                f"{epochs_without_improve}/{cfg.early_stop_patience}"
            )

        if epochs_without_improve >= cfg.early_stop_patience:
            logger.info(f"[Fold {fold_id}] 触发早停")
            break

    logger.info(
        f"[Fold {fold_id}] 训练完成，最佳 Epoch: {best_epoch}, "
        f"最佳 Inner Val Patient F1: {best_val_f1:.4f}"
    )

    model.load_state_dict(torch.load(best_weight_path, map_location=cfg.device))

    final_inner_val_pred_df = infer_probabilities(model, inner_val_loader, cfg)
    default_threshold = 0.50
    opt_threshold, opt_val_f1 = find_optimal_threshold(final_inner_val_pred_df, cfg)
    logger.info(
        f"[Fold {fold_id}] Inner Val 最优阈值: {opt_threshold:.2f} "
        f"(Patient F1: {opt_val_f1:.4f})"
    )

    outer_test_pred_df = infer_probabilities(model, outer_test_loader, cfg)
    outer_default = evaluate_prediction_frame(
        outer_test_pred_df,
        cfg,
        logger,
        phase=f"Fold{fold_id}-OuterTest-Default",
        threshold=default_threshold,
    )
    outer_opt = evaluate_prediction_frame(
        outer_test_pred_df,
        cfg,
        logger,
        phase=f"Fold{fold_id}-OuterTest-Opt",
        threshold=opt_threshold,
    )

    if cfg.save_fold_predictions:
        save_predictions(
            pred_df=outer_test_pred_df.assign(
                fold=fold_id,
                threshold_default=default_threshold,
                threshold_opt=opt_threshold,
            ),
            patient_df=outer_default["patient_df"].assign(
                fold=fold_id,
                threshold_default=default_threshold,
                threshold_opt=opt_threshold,
            ),
            save_prefix=os.path.join(fold_dir, f"fold_{fold_id}_outer_test"),
        )

    fold_result = {
        "fold": fold_id,
        "best_epoch": best_epoch,
        "inner_val_best_patient_f1_default": best_val_f1,
        "inner_val_best_threshold": opt_threshold,
        "inner_val_best_patient_f1_opt": opt_val_f1,
        "outer_default_image_f1": outer_default["image"]["f1_macro"],
        "outer_default_image_auc": outer_default["image"]["auc_benign"],
        "outer_default_image_pr_auc": outer_default["image"]["pr_auc_benign"],
        "outer_default_patient_f1": outer_default["patient"]["f1_macro"],
        "outer_default_patient_auc": outer_default["patient"]["auc_benign"],
        "outer_default_patient_pr_auc": outer_default["patient"]["pr_auc_benign"],
        "outer_opt_image_f1": outer_opt["image"]["f1_macro"],
        "outer_opt_image_auc": outer_opt["image"]["auc_benign"],
        "outer_opt_image_pr_auc": outer_opt["image"]["pr_auc_benign"],
        "outer_opt_patient_f1": outer_opt["patient"]["f1_macro"],
        "outer_opt_patient_auc": outer_opt["patient"]["auc_benign"],
        "outer_opt_patient_pr_auc": outer_opt["patient"]["pr_auc_benign"],
    }
    fold_result.update(fold_stats)

    outer_default_oof = outer_test_pred_df.copy()
    outer_default_oof["fold"] = fold_id
    outer_default_oof["eval_threshold"] = default_threshold
    outer_default_oof["pred_label"] = np.where(outer_default_oof["p_benign"] >= default_threshold, 0, 1)

    outer_opt_oof = outer_test_pred_df.copy()
    outer_opt_oof["fold"] = fold_id
    outer_opt_oof["eval_threshold"] = opt_threshold
    outer_opt_oof["pred_label"] = np.where(outer_opt_oof["p_benign"] >= opt_threshold, 0, 1)

    return fold_result, outer_default_oof, outer_opt_oof


def summarize_cv_results(results_df):
    metric_cols = [
        "outer_default_image_f1",
        "outer_default_image_auc",
        "outer_default_image_pr_auc",
        "outer_default_patient_f1",
        "outer_default_patient_auc",
        "outer_default_patient_pr_auc",
        "outer_opt_image_f1",
        "outer_opt_image_auc",
        "outer_opt_image_pr_auc",
        "outer_opt_patient_f1",
        "outer_opt_patient_auc",
        "outer_opt_patient_pr_auc",
    ]

    rows = []
    for col in metric_cols:
        rows.append(
            {
                "metric": col,
                "mean": results_df[col].mean(),
                "std": results_df[col].std(ddof=0),
                "min": results_df[col].min(),
                "max": results_df[col].max(),
            }
        )
    return pd.DataFrame(rows)


def log_cv_summary(summary_df, logger):
    logger.info("\n" + "=" * 60)
    logger.info("Cross-Validation Summary")
    logger.info("=" * 60)
    for row in summary_df.itertuples(index=False):
        logger.info(
            f"{row.metric}: mean={row.mean:.4f}, std={row.std:.4f}, "
            f"min={row.min:.4f}, max={row.max:.4f}"
        )


def copy_script_to_log_dir(cfg):
    script_path = os.path.abspath(__file__)
    dst_path = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst_path):
        shutil.copy2(script_path, dst_path)


def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = setup_logger(cfg.log_file)

    all_df = load_all_data(cfg)
    patient_df = build_patient_table(all_df)
    log_dataset_overview(all_df, patient_df, logger)

    logger.info(
        "配置开关: "
        f"weighted_sampler={cfg.use_weighted_sampler}, "
        f"mixup={cfg.use_mixup}, "
        f"label_smoothing={cfg.use_label_smoothing}, "
        f"randaugment={cfg.use_randaugment}, "
        f"random_erasing={cfg.use_random_erasing}, "
        f"vertical_flip={cfg.use_vertical_flip}, "
        f"patient_agg_mode={cfg.patient_agg_mode}"
    )

    outer_folds = build_outer_folds(patient_df, cfg)
    fold_results = []
    oof_default_frames = []
    oof_opt_frames = []

    for fold_id, train_idx, outer_test_idx in outer_folds:
        fold_seed = cfg.seed + fold_id
        set_seed(fold_seed)
        logger.info(f"[Fold {fold_id}] 使用随机种子: {fold_seed}")
        fold_result, outer_default_oof, outer_opt_oof = run_one_fold(
            fold_id=fold_id,
            all_df=all_df,
            patient_df=patient_df,
            train_idx=train_idx,
            outer_test_idx=outer_test_idx,
            cfg=cfg,
            logger=logger,
        )
        fold_results.append(fold_result)
        oof_default_frames.append(outer_default_oof)
        oof_opt_frames.append(outer_opt_oof)

    fold_results_df = pd.DataFrame(fold_results)
    fold_results_path = os.path.join(cfg.log_dir, "cv_fold_metrics.csv")
    fold_results_df.to_csv(fold_results_path, index=False)

    summary_df = summarize_cv_results(fold_results_df)
    summary_path = os.path.join(cfg.log_dir, "cv_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    log_cv_summary(summary_df, logger)

    oof_default_df = pd.concat(oof_default_frames, ignore_index=True)
    oof_opt_df = pd.concat(oof_opt_frames, ignore_index=True)
    oof_default_df.to_csv(os.path.join(cfg.log_dir, "oof_default_predictions.csv"), index=False)
    oof_opt_df.to_csv(os.path.join(cfg.log_dir, "oof_opt_predictions.csv"), index=False)

    logger.info(f"Fold 指标已保存到: {fold_results_path}")
    logger.info(f"CV 汇总已保存到: {summary_path}")
    logger.info(f"OOF 默认阈值预测已保存到: {os.path.join(cfg.log_dir, 'oof_default_predictions.csv')}")
    logger.info(f"OOF 优化阈值预测已保存到: {os.path.join(cfg.log_dir, 'oof_opt_predictions.csv')}")

    copy_script_to_log_dir(cfg)
    logger.info(f"训练脚本已复制到: {os.path.join(cfg.log_dir, os.path.basename(__file__))}")


if __name__ == "__main__":
    main()
