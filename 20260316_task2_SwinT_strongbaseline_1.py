"""
Task 2 强基线: Swin-Transformer-Tiny + 患者级划分 + 强增强
关键修复（相对之前所有实验）：
1. 患者级别 train/val/test 划分（杜绝 45% 数据泄漏）
2. 独立验证集用于选模型 + 阈值优化
3. 强数据增强（RandomResizedCrop + RandAugment + RandomErasing）
4. Mixup 正则化（小数据集最有效的正则化之一）
5. Label Smoothing（防止过度自信）
6. 冻结浅层，仅微调 stage4 + head
7. AdamW + Warmup + CosineAnnealing
8. 测试时阈值优化（提升少数类指标）
"""

import os
import sys
import logging
import time
import shutil
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report)
from sklearn.model_selection import train_test_split


# ======================== 配置 ========================
class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "dataset", "Processed")
    train_excel_orig = os.path.join(data_root, "task_2_train.xlsx")
    test_excel_orig = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260316_task2_SwinT_strongbaseline_1"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 224
    batch_size = 32
    num_epochs = 80
    lr = 2e-4
    weight_decay = 0.05
    warmup_epochs = 5
    num_workers = 4
    eval_interval = 3

    label_smoothing = 0.1
    mixup_alpha = 0.2
    dropout = 0.3

    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]


# ======================== 工具函数 ========================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file):
    logger = logging.getLogger("task2_swin_baseline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ======================== 患者级划分（修复数据泄漏） ========================
def patient_level_split(cfg, logger):
    """
    合并原始两个 excel，按患者 ID 重新划分 train/val/test。
    确保同一患者的所有图像只出现在一个集合中。
    """
    df1 = pd.read_excel(cfg.train_excel_orig)
    df2 = pd.read_excel(cfg.test_excel_orig)
    all_df = pd.concat([df1, df2], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=['image_path']).reset_index(drop=True)

    # 从文件名提取患者 ID：benign/00630757_US_Image10_1.png → 00630757
    all_df['pid'] = all_df['image_path'].apply(lambda x: x.split('/')[-1].split('_')[0])

    # 患者级标签（同一患者所有图片标签一致）
    patient_info = all_df.groupby('pid')['label'].first().reset_index()
    pids = patient_info['pid'].values
    labels = patient_info['label'].values

    # 第一次：train+val vs test (85:15)
    pids_trainval, pids_test, labels_tv, _ = train_test_split(
        pids, labels, test_size=0.15, stratify=labels, random_state=cfg.seed
    )

    # 第二次：train vs val (从 trainval 中取 ~17.6% 使 val 占总体 ~15%)
    pids_train, pids_val = train_test_split(
        pids_trainval, test_size=0.176, stratify=labels_tv, random_state=cfg.seed
    )

    set_train = set(pids_train)
    set_val = set(pids_val)
    set_test = set(pids_test)

    # 验证无泄漏
    assert len(set_train & set_val) == 0, "Train-Val leakage!"
    assert len(set_train & set_test) == 0, "Train-Test leakage!"
    assert len(set_val & set_test) == 0, "Val-Test leakage!"

    train_df = all_df[all_df['pid'].isin(set_train)][['image_path', 'label']].reset_index(drop=True)
    val_df = all_df[all_df['pid'].isin(set_val)][['image_path', 'label']].reset_index(drop=True)
    test_df = all_df[all_df['pid'].isin(set_test)][['image_path', 'label']].reset_index(drop=True)

    logger.info(f"患者级划分 (seed={cfg.seed}):")
    logger.info(f"  总图像: {len(all_df)}, 总患者: {len(pids)}")
    logger.info(f"  Train: {len(train_df)}张, {len(pids_train)}患者 "
                f"(benign={int((train_df['label']==0).sum())}, "
                f"no_tumor={int((train_df['label']==1).sum())})")
    logger.info(f"  Val:   {len(val_df)}张, {len(pids_val)}患者 "
                f"(benign={int((val_df['label']==0).sum())}, "
                f"no_tumor={int((val_df['label']==1).sum())})")
    logger.info(f"  Test:  {len(test_df)}张, {len(pids_test)}患者 "
                f"(benign={int((test_df['label']==0).sum())}, "
                f"no_tumor={int((test_df['label']==1).sum())})")
    logger.info(f"  患者泄漏检查: 通过")

    return train_df, val_df, test_df


# ======================== 数据集 ========================
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
        label = int(row["label"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ======================== Mixup ========================
def mixup_data(x, y, alpha=0.2):
    """Mixup: 混合两个样本的图像和标签"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# ======================== 模型 ========================
def build_model(cfg):
    """
    Swin-Transformer-Tiny (ImageNet预训练)
    features 结构:
      [0] stem (patch embed)
      [1] stage1 (2 blocks, dim=96)
      [2] downsample1
      [3] stage2 (2 blocks, dim=192)
      [4] downsample2
      [5] stage3 (6 blocks, dim=384)
      [6] downsample3
      [7] stage4 (2 blocks, dim=768)
    冻结 [0:6]，仅微调 downsample3 + stage4 + norm + head
    """
    model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)

    # 冻结 stem + stage1 + stage2 + stage3
    for i in range(6):
        for param in model.features[i].parameters():
            param.requires_grad = False

    # 替换分类头
    in_features = model.head.in_features  # 768
    model.head = nn.Sequential(
        nn.Dropout(p=cfg.dropout),
        nn.Linear(in_features, 2)
    )

    return model


# ======================== 评估 ========================
def evaluate(model, dataloader, device, logger, phase="Val", threshold=0.5):
    """评估模型，支持自定义阈值"""
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()  # P(benign)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # P(benign) >= threshold → 预测 0 (benign)，否则 1 (no_tumor)
    all_preds = np.where(all_probs >= threshold, 0, 1)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    logger.info(f"[{phase}] Thresh: {threshold:.2f} | Acc: {acc:.4f} | "
                f"Precision(macro): {precision:.4f} | "
                f"Recall(macro): {recall:.4f} | F1(macro): {f1:.4f}")

    report = classification_report(
        all_labels, all_preds,
        target_names=Config.class_names,
        digits=4,
        zero_division=0
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")

    return acc, precision, recall, f1, all_probs, all_labels


def find_optimal_threshold(all_probs, all_labels):
    """在验证集上搜索最优阈值，最大化 macro F1"""
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.arange(0.10, 0.90, 0.01):
        preds = np.where(all_probs >= thresh, 0, 1)
        f1 = f1_score(all_labels, preds, average='macro', zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


# ======================== 训练 ========================
def train_one_epoch(model, dataloader, criterion, optimizer, device, mixup_alpha=0.0):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 50% 概率使用 Mixup
        use_mixup = mixup_alpha > 0 and np.random.rand() < 0.5

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            optimizer.zero_grad()
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (lam * (preds == labels_a).float().sum().item() +
                        (1 - lam) * (preds == labels_b).float().sum().item())
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return running_loss / total, correct / total


# ======================== 主函数 ========================
def main():
    cfg = Config()
    set_seed(cfg.seed)
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = setup_logger(cfg.log_file)

    logger.info("=" * 60)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info(f"模型: Swin-Transformer-Tiny (冻结stem+stage1-3, 微调stage4+head)")
    logger.info(f"关键改进: 患者级划分 + RandAugment + Mixup + LabelSmoothing + 阈值优化")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"学习率: {cfg.lr}, Warmup: {cfg.warmup_epochs} epochs")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Mixup Alpha: {cfg.mixup_alpha}")
    logger.info(f"Dropout: {cfg.dropout}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 60)

    # -------- 数据划分 --------
    train_df, val_df, test_df = patient_level_split(cfg, logger)

    # -------- 数据增强 --------
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # -------- 数据集 & DataLoader --------
    train_dataset = GBPDataset(train_df, cfg.data_root, transform=train_transform)
    val_dataset = GBPDataset(val_df, cfg.data_root, transform=eval_transform)
    test_dataset = GBPDataset(test_df, cfg.data_root, transform=eval_transform)

    # WeightedRandomSampler：平衡采样（仅训练集）
    train_labels = train_df['label'].values
    class_counts = np.bincount(train_labels)
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              sampler=sampler, num_workers=cfg.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size,
                            shuffle=False, num_workers=cfg.num_workers,
                            pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=True)

    logger.info(f"采样权重: benign={1.0/class_counts[0]:.6f}, "
                f"no_tumor={1.0/class_counts[1]:.6f}")

    # -------- 模型 --------
    model = build_model(cfg).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    # -------- 损失函数（Label Smoothing，不叠加类别权重） --------
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # -------- 优化器 + 学习率调度 --------
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=cfg.num_epochs - cfg.warmup_epochs, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[cfg.warmup_epochs]
    )

    best_val_f1 = 0.0
    best_epoch = 0

    logger.info("\n" + "=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device, cfg.mixup_alpha
        )
        scheduler.step()
        elapsed = time.time() - start_time

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch}/{cfg.num_epochs}] "
                    f"LR: {current_lr:.6f} | Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | Time: {elapsed:.1f}s")

        # 在验证集上评估
        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 40)
            acc, prec, rec, f1, _, _ = evaluate(
                model, val_loader, cfg.device, logger, phase="Val"
            )

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优模型 (Val F1: {best_val_f1:.4f}, Epoch: {best_epoch}) ***")
            logger.info("-" * 40)

    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成! 最优模型: Epoch {best_epoch}, Val F1: {best_val_f1:.4f}")
    logger.info("=" * 60)

    # =============== 最终测试 ===============
    logger.info("\n加载最优权重...")
    model.load_state_dict(torch.load(cfg.best_weight_path, map_location=cfg.device))

    # 1) 在验证集上搜索最优阈值
    logger.info("\n在验证集上搜索最优阈值...")
    _, _, _, _, val_probs, val_labels = evaluate(
        model, val_loader, cfg.device, logger, phase="Val(最优权重)"
    )
    opt_thresh, opt_val_f1 = find_optimal_threshold(val_probs, val_labels)
    logger.info(f"最优阈值: {opt_thresh:.2f} (Val F1: {opt_val_f1:.4f})")

    # 2) 默认阈值 (0.50) 测试
    logger.info("\n" + "=" * 60)
    logger.info("最终测试 (默认阈值 0.50)")
    logger.info("=" * 60)
    evaluate(model, test_loader, cfg.device, logger,
             phase="Final Test", threshold=0.50)

    # 3) 优化阈值测试
    logger.info("\n" + "=" * 60)
    logger.info(f"最终测试 (优化阈值 {opt_thresh:.2f})")
    logger.info("=" * 60)
    evaluate(model, test_loader, cfg.device, logger,
             phase="Final Test(OPT)", threshold=opt_thresh)

    # 复制脚本存档
    script_path = os.path.abspath(__file__)
    dst_path = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst_path):
        shutil.copy2(script_path, dst_path)
        logger.info(f"训练脚本已复制到: {dst_path}")


if __name__ == "__main__":
    main()
