"""
Task 2 五模型 Baseline 对比实验
ResNet18 | VGG16 | Swin-T | U-Net Encoder | ViT-B/16

- 良性肿瘤 label=0, 非肿瘤 label=1
- 无数据增强，仅 Resize + Normalize
- 全参数微调 (U-Net 从零训练)
- 每3个epoch在测试集上评估
- 保存最优权重 (基于 macro F1)
- 最后输出五模型对比表
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, confusion_matrix,
)

# ======================== 配置 ========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(PROJECT_ROOT, "dataset", "Processed")
TRAIN_EXCEL = os.path.join(DATA_ROOT, "task_2_train.xlsx")
TEST_EXCEL = os.path.join(DATA_ROOT, "task_2_test.xlsx")

EXP_NAME = "20260318_task2_baseline_5models"
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", EXP_NAME)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
EVAL_INTERVAL = 3
CLASS_NAMES = ["benign", "no_tumor"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 每个模型的学习率 (Transformer 系需要更低的 lr)
MODEL_CONFIGS = {
    "ResNet18":  {"lr": 1e-3},
    "VGG16":     {"lr": 1e-3},
    "SwinT":     {"lr": 1e-4},
    "UNet_Enc":  {"lr": 1e-3},
    "ViT_B16":   {"lr": 1e-4},
}


# ======================== U-Net Encoder ========================
class DoubleConv(nn.Module):
    """U-Net 标准双卷积块: Conv3x3 + BN + ReLU × 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetEncoder(nn.Module):
    """
    U-Net 编码器路径 (无解码器) 用于分类
    4 层下采样 + bottleneck + GAP + FC
    从零训练，无预训练权重
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.enc1 = DoubleConv(3, 64)       # 224 -> 112
        self.enc2 = DoubleConv(64, 128)      # 112 -> 56
        self.enc3 = DoubleConv(128, 256)     # 56  -> 28
        self.enc4 = DoubleConv(256, 512)     # 28  -> 14
        self.bottleneck = DoubleConv(512, 1024)  # 14
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(self.enc1(x))
        x = self.pool(self.enc2(x))
        x = self.pool(self.enc3(x))
        x = self.pool(self.enc4(x))
        x = self.bottleneck(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# ======================== 模型构建 ========================
def build_model(model_name):
    if model_name == "ResNet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 2)

    elif model_name == "VGG16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(4096, 2)

    elif model_name == "SwinT":
        model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1)
        model.head = nn.Linear(model.head.in_features, 2)

    elif model_name == "UNet_Enc":
        model = UNetEncoder(num_classes=2)

    elif model_name == "ViT_B16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)

    else:
        raise ValueError(f"未知模型: {model_name}")

    return model


# ======================== 日志 ========================
def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    ch = logging.StreamHandler(sys.stdout)
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


# ======================== 评估 ========================
def evaluate(model, dataloader, logger, phase="Test"):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 0].cpu().numpy())  # P(benign)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # AUC (benign vs no_tumor)
    try:
        benign_labels = (all_labels == 0).astype(int)
        auc = roc_auc_score(benign_labels, all_probs)
    except ValueError:
        auc = float('nan')

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])

    logger.info(f"[{phase}] Acc: {acc:.4f} | Precision(macro): {precision:.4f} | "
                f"Recall(macro): {recall:.4f} | F1(macro): {f1:.4f} | AUC: {auc:.4f}")
    logger.info(f"[{phase}] Confusion Matrix (rows=true, cols=pred): {cm.tolist()}")

    report = classification_report(all_labels, all_preds,
                                   target_names=CLASS_NAMES, digits=4, zero_division=0)
    logger.info(f"[{phase}] Classification Report:\n{report}")

    return {"acc": acc, "precision": precision, "recall": recall, "f1": f1, "auc": auc}


# ======================== 训练一个 epoch ========================
def train_one_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

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


# ======================== 单模型完整训练流程 ========================
def run_single_model(model_name, train_loader, test_loader, train_dataset, test_dataset):
    """训练单个模型，返回最佳指标字典"""
    model_dir = os.path.join(LOG_DIR, model_name)
    os.makedirs(model_dir, exist_ok=True)

    log_file = os.path.join(model_dir, f"{model_name}.log")
    logger = setup_logger(f"task2_{model_name}", log_file)

    lr = MODEL_CONFIGS[model_name]["lr"]

    logger.info("=" * 60)
    logger.info(f"模型: {model_name}")
    logger.info(f"任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info(f"设置: Baseline (无数据增强, 全参数微调, CE Loss)")
    logger.info(f"图像尺寸: {IMG_SIZE} | Batch: {BATCH_SIZE} | LR: {lr}")
    logger.info(f"Epochs: {NUM_EPOCHS} | Eval Interval: {EVAL_INTERVAL}")
    logger.info(f"设备: {DEVICE}")
    logger.info(f"训练集: {len(train_dataset)} 张 | 测试集: {len(test_dataset)} 张")
    logger.info("=" * 60)

    # 构建模型
    model = build_model(model_name).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"参数量: {total_params:,} | 可训练: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")

    pretrained_tag = "从零训练" if model_name == "UNet_Enc" else "ImageNet 预训练"
    logger.info(f"权重初始化: {pretrained_tag}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    best_f1 = 0.0
    best_epoch = 0
    best_metrics = {}
    best_weight_path = os.path.join(model_dir, f"{model_name}_best.pth")

    logger.info("\n开始训练")
    logger.info("-" * 60)

    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        elapsed = time.time() - start

        logger.info(f"Epoch [{epoch}/{NUM_EPOCHS}] "
                    f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Time: {elapsed:.1f}s")

        if epoch % EVAL_INTERVAL == 0 or epoch == NUM_EPOCHS:
            metrics = evaluate(model, test_loader, logger, phase="Test")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_epoch = epoch
                best_metrics = metrics.copy()
                best_metrics["best_epoch"] = best_epoch
                torch.save(model.state_dict(), best_weight_path)
                logger.info(f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")

    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成! 最优: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 60)

    # 加载最优权重做最终评估
    if os.path.exists(best_weight_path):
        model.load_state_dict(torch.load(best_weight_path, map_location=DEVICE))
        logger.info("\n加载最优权重 - 最终测试:")
        final_metrics = evaluate(model, test_loader, logger, phase="Final Test")
        best_metrics.update(final_metrics)
        best_metrics["best_epoch"] = best_epoch

    # 清理 GPU 显存
    del model, optimizer, criterion
    torch.cuda.empty_cache()

    return best_metrics


# ======================== 主函数 ========================
def main():
    os.makedirs(LOG_DIR, exist_ok=True)

    # 主日志
    main_logger = setup_logger("task2_main", os.path.join(LOG_DIR, f"{EXP_NAME}.log"))
    main_logger.info("=" * 60)
    main_logger.info("Task 2 五模型 Baseline 对比实验")
    main_logger.info(f"模型: {list(MODEL_CONFIGS.keys())}")
    main_logger.info(f"设备: {DEVICE}")
    main_logger.info("=" * 60)

    # 数据预处理: 仅 Resize + Normalize
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = GBPDataset(TRAIN_EXCEL, DATA_ROOT, transform=transform)
    test_dataset = GBPDataset(TEST_EXCEL, DATA_ROOT, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    n_benign_train = sum(train_dataset.df['label'] == 0)
    n_notumor_train = sum(train_dataset.df['label'] == 1)
    n_benign_test = sum(test_dataset.df['label'] == 0)
    n_notumor_test = sum(test_dataset.df['label'] == 1)

    main_logger.info(f"训练集: {len(train_dataset)} 张 (benign={n_benign_train}, no_tumor={n_notumor_train})")
    main_logger.info(f"测试集: {len(test_dataset)} 张 (benign={n_benign_test}, no_tumor={n_notumor_test})")
    main_logger.info("")

    # 依次训练 5 个模型
    all_results = {}
    for model_name in MODEL_CONFIGS:
        main_logger.info(f"\n{'#' * 60}")
        main_logger.info(f"# 开始训练: {model_name}")
        main_logger.info(f"{'#' * 60}")

        try:
            metrics = run_single_model(model_name, train_loader, test_loader,
                                       train_dataset, test_dataset)
            all_results[model_name] = metrics
            main_logger.info(f"[{model_name}] 完成 - Best F1: {metrics['f1']:.4f} (Epoch {metrics['best_epoch']})")
        except Exception as e:
            main_logger.info(f"[{model_name}] 训练失败: {e}")
            all_results[model_name] = {"f1": 0, "acc": 0, "auc": 0, "precision": 0, "recall": 0, "best_epoch": -1}
            import traceback
            main_logger.info(traceback.format_exc())
            torch.cuda.empty_cache()

    # ======================== 对比汇总表 ========================
    main_logger.info("\n" + "=" * 80)
    main_logger.info("五模型 Baseline 对比汇总")
    main_logger.info("=" * 80)

    header = f"{'模型':<12} {'F1(macro)':>10} {'AUC':>10} {'Acc':>10} {'Precision':>10} {'Recall':>10} {'Best Ep':>8}"
    main_logger.info(header)
    main_logger.info("-" * 80)

    rows = []
    for model_name in MODEL_CONFIGS:
        m = all_results[model_name]
        line = (f"{model_name:<12} {m['f1']:>10.4f} {m['auc']:>10.4f} "
                f"{m['acc']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['best_epoch']:>8}")
        main_logger.info(line)
        rows.append({"model": model_name, **m})

    main_logger.info("-" * 80)

    # 找最佳模型
    best_model = max(all_results, key=lambda k: all_results[k]["f1"])
    main_logger.info(f"\n最佳模型: {best_model} (F1: {all_results[best_model]['f1']:.4f})")

    # 保存为 CSV
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(LOG_DIR, "comparison_results.csv")
    results_df.to_csv(csv_path, index=False)
    main_logger.info(f"对比结果已保存: {csv_path}")

    # 复制脚本存档
    script_path = os.path.abspath(__file__)
    dst_path = os.path.join(LOG_DIR, os.path.basename(script_path))
    if os.path.abspath(script_path) != os.path.abspath(dst_path):
        shutil.copy2(script_path, dst_path)
        main_logger.info(f"训练脚本已复制到: {dst_path}")


if __name__ == "__main__":
    main()
