"""
Task 2 Ablation B: ResNet18 + Class Weight + WeightedRandomSampler

目标:
1. 在 weight-only 消融基础上加入 WeightedRandomSampler
2. 检查“损失重加权 + 采样平衡”是否比单独 class weight 更稳定
3. 去掉训练增强，避免 augmentation 噪声掩盖 sampler 的作用
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "dataset", "Processed")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260318_task2_Resnet18_weightsampler_1"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 224
    batch_size = 32
    num_epochs = 80
    lr = 5e-4
    weight_decay = 1e-3
    num_workers = 4
    eval_interval = 3
    dropout = 0.5
    seed = 42

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(log_file):
    logger = logging.getLogger("task2_weightsampler")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


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


class ResNet18WithDropout(nn.Module):
    """ResNet18 冻结浅层，仅训练 layer4 + fc"""

    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        frozen_layers = [
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        ]
        for layer in frozen_layers:
            for param in layer.parameters():
                param.requires_grad = False

        self.features = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
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
        target_names=Config.class_names,
        digits=4,
        zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")

    return acc, precision, recall, f1


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


def compute_class_weights_and_sampler(excel_path):
    df = pd.read_excel(excel_path)
    labels = df["label"].values
    class_counts = np.bincount(labels)

    total = len(df)
    num_classes = len(class_counts)
    class_weights = []
    for cls in range(num_classes):
        class_weights.append(total / (num_classes * class_counts[cls]))
    class_weights = torch.FloatTensor(class_weights)

    sample_weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(df),
        replacement=True,
    )

    return class_weights, class_counts, sampler


def main():
    cfg = Config()
    set_seed(cfg.seed)

    os.makedirs(cfg.log_dir, exist_ok=True)
    logger = setup_logger(cfg.log_file)

    logger.info("=" * 60)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info("任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info("模型: ResNet18 (冻结layer1-3, 微调layer4+fc)")
    logger.info("消融: class weight + WeightedRandomSampler，无训练增强")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"学习率: {cfg.lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"随机种子: {cfg.seed}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 60)

    transform = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = GBPDataset(cfg.train_excel, cfg.data_root, transform=transform)
    test_dataset = GBPDataset(cfg.test_excel, cfg.data_root, transform=transform)

    class_weights, class_counts, sampler = compute_class_weights_and_sampler(cfg.train_excel)
    class_weights = class_weights.to(cfg.device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
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
    logger.info(f"类别权重: benign={class_weights[0]:.4f}, no_tumor={class_weights[1]:.4f}")
    logger.info(f"采样权重: benign={1.0 / class_counts[0]:.6f}, no_tumor={1.0 / class_counts[1]:.6f}")

    model = ResNet18WithDropout(num_classes=2, dropout=cfg.dropout).to(cfg.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {total_params:,}")
    logger.info(f"可训练参数量: {trainable_params:,} ({trainable_params / total_params * 100:.1f}%)")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
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

        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {current_lr:.6f} | "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Time: {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 40)
            _, _, _, f1 = evaluate(model, test_loader, cfg.device, logger, phase="Test")

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
    evaluate(model, test_loader, cfg.device, logger, phase="Final Test")

    script_path = os.path.abspath(__file__)
    dst_path = os.path.join(cfg.log_dir, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst_path):
        shutil.copy2(script_path, dst_path)
        logger.info(f"训练脚本已复制到: {dst_path}")


if __name__ == "__main__":
    main()
