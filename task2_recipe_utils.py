import logging
import math
import os
import random
import shutil
import sys
import time
import atexit

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score


class GBPDataset(Dataset):
    def __init__(self, excel_path, data_root, transform=None):
        self.df = pd.read_excel(excel_path)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(log_file, logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def _pid_is_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_run_lock(lock_path):
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)

    if os.path.exists(lock_path):
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                existing_pid = int(f.read().strip() or "0")
        except (OSError, ValueError):
            existing_pid = 0

        if _pid_is_alive(existing_pid):
            return False, existing_pid

        try:
            os.remove(lock_path)
        except OSError:
            pass

    fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))

    def _cleanup():
        try:
            if os.path.exists(lock_path):
                with open(lock_path, "r", encoding="utf-8") as f:
                    owner_pid = int(f.read().strip() or "0")
                if owner_pid == os.getpid():
                    os.remove(lock_path)
        except (OSError, ValueError):
            pass

    atexit.register(_cleanup)
    return True, os.getpid()


def evaluate(model, dataloader, device, class_names, logger, phase="Test"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(
        f"[{phase}] Acc: {acc:.4f} | Precision(macro): {precision:.4f} | "
        f"Recall(macro): {recall:.4f} | F1(macro): {f1:.4f}"
    )

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, precision, recall, f1


def build_class_weights(train_df, class_names, device):
    label_counts = train_df["label"].value_counts().sort_index()
    total_samples = len(train_df)
    weights = []

    for label_idx in range(len(class_names)):
        class_count = int(label_counts[label_idx])
        class_weight = total_samples / (len(class_names) * class_count)
        weights.append(class_weight)

    return torch.tensor(weights, dtype=torch.float32, device=device)


def split_backbone_and_head(model, head_module):
    head_params = [p for p in head_module.parameters() if p.requires_grad]
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_param_ids]
    return backbone_params, head_params


def build_optimizer_with_diff_lr(optimizer_cls, backbone_params, head_params, cfg):
    optimizer = optimizer_cls(
        [
            {"params": backbone_params, "lr": cfg.backbone_lr, "base_lr": cfg.backbone_lr},
            {"params": head_params, "lr": cfg.head_lr, "base_lr": cfg.head_lr},
        ],
        weight_decay=cfg.weight_decay,
    )
    return optimizer


def cosine_warmup_factor(epoch, num_epochs, warmup_epochs, min_lr_ratio):
    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return epoch / warmup_epochs

    if num_epochs <= warmup_epochs:
        return 1.0

    progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def set_epoch_lrs(optimizer, epoch, cfg):
    factor = cosine_warmup_factor(
        epoch=epoch,
        num_epochs=cfg.num_epochs,
        warmup_epochs=cfg.warmup_epochs,
        min_lr_ratio=cfg.min_lr_ratio,
    )
    for param_group in optimizer.param_groups:
        base_lr = param_group.get("base_lr", param_group["lr"])
        param_group["lr"] = base_lr * factor
    return factor


def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, use_amp, grad_clip=None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        amp_device = "cuda" if device.type == "cuda" else "cpu"
        with torch.amp.autocast(device_type=amp_device, enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        if grad_clip is not None:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def build_dataloaders(cfg, train_transform, test_transform):
    train_dataset = GBPDataset(cfg.train_excel, cfg.data_root, transform=train_transform)
    test_dataset = GBPDataset(cfg.test_excel, cfg.data_root, transform=test_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.type == "cuda"),
    )
    return train_dataset, test_dataset, train_loader, test_loader


def run_experiment(
    cfg,
    build_model_fn,
    build_train_transform_fn,
    build_test_transform_fn,
    build_optimizer_fn,
    script_path,
):
    os.makedirs(cfg.log_dir, exist_ok=True)
    lock_path = getattr(cfg, "lock_path", os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock"))
    lock_acquired, lock_owner = acquire_run_lock(lock_path)
    if not lock_acquired:
        print(
            f"[Skip] {cfg.exp_name} is already running under PID {lock_owner}. "
            f"Current PID {os.getpid()} exits."
        )
        return

    set_seed(cfg.seed)

    logger = setup_logger(cfg.log_file, cfg.exp_name)

    logger.info("=" * 60)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info("任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"图像尺寸: train {cfg.train_transform_desc}, test {cfg.test_transform_desc}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Backbone LR: {cfg.backbone_lr}")
    logger.info(f"Head LR: {cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Warmup Epochs: {cfg.warmup_epochs}")
    logger.info(f"Min LR Ratio: {cfg.min_lr_ratio}")
    logger.info(f"Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"Grad Clip: {cfg.grad_clip}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 60)

    train_transform = build_train_transform_fn(cfg)
    test_transform = build_test_transform_fn(cfg)
    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(
        cfg,
        train_transform,
        test_transform,
    )

    logger.info(
        f"训练集: {len(train_dataset)} 张 "
        f"(benign={sum(train_dataset.df['label'] == 0)}, "
        f"no_tumor={sum(train_dataset.df['label'] == 1)})"
    )
    logger.info(
        f"测试集: {len(test_dataset)} 张 "
        f"(benign={sum(test_dataset.df['label'] == 0)}, "
        f"no_tumor={sum(test_dataset.df['label'] == 1)})"
    )

    model = build_model_fn().to(cfg.device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    class_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    logger.info(
        f"类别权重: benign={class_weights[0].item():.4f}, "
        f"no_tumor={class_weights[1].item():.4f}"
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer_fn(model, cfg)

    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )
    best_f1 = 0.0
    best_epoch = 0

    logger.info("\n" + "=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(1, cfg.num_epochs + 1):
        lr_factor = set_epoch_lrs(optimizer, epoch, cfg)
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=cfg.device,
            scaler=scaler,
            use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip,
        )
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR(backbone/head): {optimizer.param_groups[0]['lr']:.6e}/{optimizer.param_groups[1]['lr']:.6e} "
            f"| WarmupCosineFactor: {lr_factor:.4f} "
            f"| Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.4f} "
            f"| Time: {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 40)
            _, _, _, f1 = evaluate(model, test_loader, cfg.device, cfg.class_names, logger, phase="Test")

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")
            logger.info("-" * 40)

    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成! 最优模型: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 60)

    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(torch.load(cfg.best_weight_path, map_location=cfg.device))
    logger.info("=" * 60)
    logger.info("最终测试结果 (最优权重)")
    logger.info("=" * 60)
    evaluate(model, test_loader, cfg.device, cfg.class_names, logger, phase="Final Test")

    dst_path = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst_path):
        shutil.copy2(script_path, dst_path)
        logger.info(f"训练脚本已复制到: {dst_path}")
