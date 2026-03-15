"""
Task 2 V2: ResNet18 二分类 (良性肿瘤 vs 非肿瘤性息肉)
改进点（相对baseline）：
1. 加权CrossEntropyLoss（按类别频率反比加权）
2. CosineAnnealingLR 学习率调度
3. 数据增强（翻转、旋转、颜色扰动）
4. 冻结浅层（仅微调layer4 + fc）
5. fc前加Dropout(0.5)
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ======================== 配置 ========================
class Config:
    # 路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "dataset", "Processed")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    # 实验名称和日志目录
    exp_name = "20260315_task2_Resnet18_augweight_2"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    # 训练参数
    img_size = 224
    batch_size = 32
    num_epochs = 80
    lr = 5e-4
    weight_decay = 1e-3
    num_workers = 4
    eval_interval = 3
    dropout = 0.5

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 类别名称
    class_names = ["benign", "no_tumor"]


# ======================== 日志设置 ========================
def setup_logger(log_file):
    logger = logging.getLogger("task2_v2")
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
class ResNet18WithDropout(nn.Module):
    """ResNet18 冻结浅层 + Dropout + 新fc"""
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # 冻结 conv1, bn1, layer1, layer2, layer3
        frozen_layers = [backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
                         backbone.layer1, backbone.layer2, backbone.layer3]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

        # 保留所有特征提取层
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4
        )
        self.avgpool = backbone.avgpool
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ======================== 评估函数 ========================
def evaluate(model, dataloader, device, logger, phase="Test"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


# ======================== 计算类别权重 ========================
def compute_class_weights(excel_path):
    """按类别频率反比计算权重"""
    df = pd.read_excel(excel_path)
    label_counts = df['label'].value_counts().sort_index()
    total = len(df)
    num_classes = len(label_counts)
    weights = []
    for cls in range(num_classes):
        w = total / (num_classes * label_counts[cls])
        weights.append(w)
    return torch.FloatTensor(weights)


# ======================== 主函数 ========================
def main():
    cfg = Config()

    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = setup_logger(cfg.log_file)

    logger.info("=" * 60)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info(f"模型: ResNet18 (冻结layer1-3, 微调layer4+fc)")
    logger.info(f"改进: 加权损失 + CosineAnnealingLR + 数据增强 + Dropout({cfg.dropout})")
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

    # 测试集仅Resize + Normalize
    test_transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = GBPDataset(cfg.train_excel, cfg.data_root, transform=train_transform)
    test_dataset = GBPDataset(cfg.test_excel, cfg.data_root, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=True)

    n_benign_train = sum(train_dataset.df['label'] == 0)
    n_notumor_train = sum(train_dataset.df['label'] == 1)
    logger.info(f"训练集: {len(train_dataset)} 张 "
                f"(benign={n_benign_train}, no_tumor={n_notumor_train})")
    logger.info(f"测试集: {len(test_dataset)} 张 "
                f"(benign={sum(test_dataset.df['label']==0)}, "
                f"no_tumor={sum(test_dataset.df['label']==1)})")

    # 类别权重
    class_weights = compute_class_weights(cfg.train_excel).to(cfg.device)
    logger.info(f"类别权重: benign={class_weights[0]:.4f}, no_tumor={class_weights[1]:.4f}")

    # 模型
    model = ResNet18WithDropout(num_classes=2, dropout=cfg.dropout).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    # 加权交叉熵损失
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=cfg.lr, weight_decay=cfg.weight_decay)
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
