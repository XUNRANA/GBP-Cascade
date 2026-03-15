"""
Task 2 V3: ConvNeXt-Base + Focal Loss + 平衡采样
改进点（相对V2）：
1. ConvNeXt-Base 替代 ResNet18（特征提取能力大幅提升）
2. Focal Loss 替代加权CE（聚焦难分样本，避免loss震荡）
3. WeightedRandomSampler 平衡采样（保证每个batch类别均衡）
4. 图像尺寸320（保持原始ROI分辨率，不下采样）
5. 冻结 stages 0-2，仅微调 stage 3 + classifier
6. AdamW + CosineAnnealingLR
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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ======================== 配置 ========================
class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "dataset", "Processed")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260315_task2_ConvNeXtB_focal_3"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 320  # 保持原始ROI分辨率
    batch_size = 16  # ConvNeXt-Base显存需求更大
    num_epochs = 60
    lr = 1e-4
    weight_decay = 1e-2
    num_workers = 4
    eval_interval = 3

    # Focal Loss参数
    focal_gamma = 2.0
    focal_alpha = [0.6, 0.4]  # benign略高权重

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]


# ======================== Focal Loss ========================
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        if alpha is not None:
            self.alpha = torch.FloatTensor(alpha)
        else:
            self.alpha = None
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            at = alpha[targets]
            focal_weight = focal_weight * at

        return (focal_weight * ce_loss).mean()


# ======================== 日志设置 ========================
def setup_logger(log_file):
    logger = logging.getLogger("task2_v3")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ======================== 数据集 ========================
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


# ======================== 模型 ========================
def build_model(num_classes=2):
    """ConvNeXt-Base，冻结 stages 0-2，微调 stage 3 + classifier"""
    model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)

    # 冻结 stages 0, 1, 2 (features[0] ~ features[5])
    # ConvNeXt features: [0]=stem, [1]=stage1, [2]=downsample, [3]=stage2, [4]=downsample, [5]=stage3, [6]=downsample, [7]=stage4
    for i in range(6):  # 冻结 0~5 (stem + stage1 + stage2 + stage3 的前部分)
        for param in model.features[i].parameters():
            param.requires_grad = False

    # 替换分类头
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    return model


# ======================== 评估函数 ========================
def evaluate(model, dataloader, device, logger, phase="Test"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    logger.info(f"[{phase}] Acc: {acc:.4f} | Precision(macro): {precision:.4f} | "
                f"Recall(macro): {recall:.4f} | F1(macro): {f1:.4f}")

    report = classification_report(
        all_labels, all_preds,
        target_names=Config.class_names,
        digits=4,
        zero_division=0
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")

    return acc, precision, recall, f1


# ======================== 训练函数 ========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# ======================== 主函数 ========================
def main():
    cfg = Config()
    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = setup_logger(cfg.log_file)

    logger.info("=" * 60)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info(f"模型: ConvNeXt-Base (冻结stages 0-2, 微调stage3+classifier)")
    logger.info(f"改进: Focal Loss(gamma={cfg.focal_gamma}) + WeightedRandomSampler + Dropout(0.5)")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"学习率: {cfg.lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 60)

    # 训练集数据增强
    train_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = GBPDataset(cfg.train_excel, cfg.data_root, transform=train_transform)
    test_dataset = GBPDataset(cfg.test_excel, cfg.data_root, transform=test_transform)

    # WeightedRandomSampler: 平衡采样
    train_labels = train_dataset.df['label'].values
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              sampler=sampler, num_workers=cfg.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=True)

    n_benign = int(class_counts[0])
    n_notumor = int(class_counts[1])
    logger.info(f"训练集: {len(train_dataset)} 张 (benign={n_benign}, no_tumor={n_notumor})")
    logger.info(f"测试集: {len(test_dataset)} 张 "
                f"(benign={sum(test_dataset.df['label']==0)}, "
                f"no_tumor={sum(test_dataset.df['label']==1)})")
    logger.info(f"采样权重: benign={class_weights[0]:.6f}, no_tumor={class_weights[1]:.6f}")

    # 模型
    model = build_model(num_classes=2).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    # Focal Loss
    criterion = FocalLoss(alpha=cfg.focal_alpha, gamma=cfg.focal_gamma)
    logger.info(f"Focal Loss: alpha={cfg.focal_alpha}, gamma={cfg.focal_gamma}")

    # AdamW优化器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs, eta_min=1e-6)

    best_f1 = 0.0
    best_epoch = 0

    logger.info("\n" + "=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(1, cfg.num_epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.device
        )
        scheduler.step()
        elapsed = time.time() - start_time

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch [{epoch}/{cfg.num_epochs}] "
                    f"LR: {current_lr:.6f} | "
                    f"Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.4f} | "
                    f"Time: {elapsed:.1f}s")

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 40)
            acc, precision, recall, f1 = evaluate(
                model, test_loader, cfg.device, logger, phase="Test"
            )

            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")
            logger.info("-" * 40)

    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成! 最优模型: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 60)

    # 加载最优权重进行最终测试
    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(torch.load(cfg.best_weight_path, map_location=cfg.device))
    logger.info("=" * 60)
    logger.info("最终测试结果 (最优权重)")
    logger.info("=" * 60)
    evaluate(model, test_loader, cfg.device, logger, phase="Final Test")

    # 复制训练脚本到日志目录存档
    script_path = os.path.abspath(__file__)
    dst_path = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst_path):
        shutil.copy2(script_path, dst_path)
        logger.info(f"训练脚本已复制到: {dst_path}")


if __name__ == "__main__":
    main()
