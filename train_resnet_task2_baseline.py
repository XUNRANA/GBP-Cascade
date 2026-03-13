"""
ResNet Baseline（从零训练）：良性腺瘤 benign(1) vs 非肿瘤性息肉 non_neoplastic(0)
================================================================
- 数据集：dataset0312/{benign, non_neoplastic}
- 患者级划分：默认 7:3（test_size=0.30）
- 训练流程：
    读图 -> ROI裁剪 -> OpenCV随机增强 -> Resize(256) -> RandomCrop(224)
    -> 随机翻转/旋转/颜色扰动/仿射 -> ToTensor -> Normalize -> RandomErasing
- 测试流程：
    读图 -> ROI裁剪 -> Resize(224) -> ToTensor -> Normalize
- 模型：原生 ResNet18，从零训练（weights=None）
- 指标：acc / precision / recall / f1 / auc(可计算时)
- 保存：
    1) 最优 AUC 权重
    2) 最终权重
    3) 指标曲线图 logs/resnet_task2_baseline_metrics_<timestamp>.png
"""

import os
import glob
import json
import random
import logging
import warnings
import argparse
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

warnings.filterwarnings('ignore')


# ============================================================
# OpenCV 增强函数（复用现有脚本风格）
# ============================================================
def deidentify(img: np.ndarray,
               top: float = 0.08,
               bottom: float = 0.05,
               side: float = 0.06) -> np.ndarray:
    """信息脱敏：遮盖边缘文字区域。"""
    h, w = img.shape[:2]
    fill = int(img.mean())

    def jitter(base):
        return max(0.0, base + random.uniform(-0.02, 0.02))

    out = img.copy()
    t = int(h * jitter(top))
    b = int(h * jitter(bottom))
    l = int(w * jitter(side))
    r = int(w * jitter(side))

    if t > 0:
        out[:t, :] = fill
    if b > 0:
        out[h - b:, :] = fill
    if l > 0:
        out[:, :l] = fill
    if r > 0:
        out[:, w - r:] = fill
    return out


def global_histeq(img: np.ndarray) -> np.ndarray:
    """全局直方图均衡化（亮度通道）。"""
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def gamma_correction(img: np.ndarray,
                     lo: float = 0.6,
                     hi: float = 1.8) -> np.ndarray:
    """随机伽马校正。"""
    gamma = random.uniform(lo, hi)
    table = np.clip(
        (np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0,
        0,
        255,
    ).astype(np.uint8)
    return cv2.LUT(img, table)


def gaussian_clahe(img: np.ndarray,
                   ksize: int = 3,
                   clip: float = 2.0,
                   grid: tuple = (8, 8)) -> np.ndarray:
    """高斯去噪 + CLAHE。"""
    denoised = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)
    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


def opencv_augment(pil_img: Image.Image,
                   training: bool,
                   p_a: float,
                   p_b: float,
                   p_c: float,
                   p_d: float) -> Image.Image:
    """训练时随机触发 A/B/C/D 增强。"""
    if not training:
        return pil_img

    img = np.array(pil_img)
    if random.random() < p_a:
        img = deidentify(img)
    if random.random() < p_b:
        img = global_histeq(img)
    if random.random() < p_c:
        img = gamma_correction(img)
    if random.random() < p_d:
        img = gaussian_clahe(img)
    return Image.fromarray(img)


# ============================================================
# ROI 裁剪
# ============================================================
def crop_roi(img_path: str,
             pad_ratio: float = 0.15) -> Image.Image:
    """读取同名 json 标注框并做 ROI 裁剪；若失败则返回原图。"""
    image = Image.open(img_path).convert('RGB')
    json_path = os.path.splitext(img_path)[0] + '.json'
    if not os.path.exists(json_path):
        return image

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        shapes = data.get('shapes', [])
        if not shapes:
            return image

        points = shapes[0].get('points', [])
        if not points:
            return image

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

        width, height = image.size
        pad_x = (x2 - x1) * pad_ratio
        pad_y = (y2 - y1) * pad_ratio

        image = image.crop((
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(width, x2 + pad_x),
            min(height, y2 + pad_y),
        ))
    except Exception:
        return image

    return image


# ============================================================
# Transform
# ============================================================
def build_transforms(image_size: int,
                     resize_size: int,
                     use_imagenet_norm: bool = False,
                     no_augmentation: bool = False):
    if use_imagenet_norm:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    if no_augmentation:
        train_tf = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        train_tf = T.Compose([
            T.Resize((resize_size, resize_size)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=20),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
            T.RandomAffine(
                degrees=10,
                translate=(0.1, 0.1),
                scale=(0.85, 1.15),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
            T.RandomErasing(p=0.2, scale=(0.02, 0.08)),
        ])

    test_tf = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return train_tf, test_tf


# ============================================================
# Dataset
# ============================================================
class Task2BaselineDataset(Dataset):
    """Task2 二分类数据集。

    samples: List[Tuple[img_path, label]]
        - benign -> 1
        - non_neoplastic -> 0
    """
    def __init__(self,
                 samples,
                 transform,
                 training: bool,
                 roi_pad_ratio: float,
                 p_a: float,
                 p_b: float,
                 p_c: float,
                 p_d: float):
        self.samples = samples
        self.transform = transform
        self.training = training
        self.roi_pad_ratio = roi_pad_ratio
        self.p_a = p_a
        self.p_b = p_b
        self.p_c = p_c
        self.p_d = p_d

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        pil_img = crop_roi(img_path, pad_ratio=self.roi_pad_ratio)
        pil_img = opencv_augment(
            pil_img,
            training=self.training,
            p_a=self.p_a,
            p_b=self.p_b,
            p_c=self.p_c,
            p_d=self.p_d,
        )
        x = self.transform(pil_img)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


# ============================================================
# 工具函数
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_patients(dataset_root: str,
                  cls_name: str):
    """返回 [(patient_id, [img_paths]), ...]"""
    cls_dir = os.path.join(dataset_root, cls_name)
    result = []
    for pid in sorted(os.listdir(cls_dir)):
        p_dir = os.path.join(cls_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        imgs = sorted(
            glob.glob(os.path.join(p_dir, '*.png')) +
            glob.glob(os.path.join(p_dir, '*.jpg')) +
            glob.glob(os.path.join(p_dir, '*.JPG')) +
            glob.glob(os.path.join(p_dir, '*.jpeg')) +
            glob.glob(os.path.join(p_dir, '*.JPEG'))
        )
        if imgs:
            result.append((pid, imgs))
    return result


def to_samples(patient_list, label: int):
    return [(img, label) for _, imgs in patient_list for img in imgs]


def log_split(log, tag: str, samples):
    pos = sum(s[1] for s in samples)
    neg = len(samples) - pos
    log.info(
        f'{tag}: 共 {len(samples)} 张 | '
        f'benign(1)={pos} | non_neoplastic(0)={neg} | '
        f'比例 1:{neg / max(1, pos):.2f}'
    )


def build_model(dropout: float = 0.3):
    """原生 ResNet18 from scratch。"""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 1),
    )
    return model


@torch.no_grad()
def evaluate(model,
             loader,
             device,
             threshold: float):
    model.eval()
    all_probs = []
    all_preds = []
    all_targets = []

    for imgs, labels in loader:
        logits = model(imgs.to(device)).squeeze(1)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= threshold).astype(np.int64)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_targets.extend(labels.numpy().astype(np.int64).tolist())

    acc = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = float('nan')

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'targets': all_targets,
        'preds': all_preds,
        'probs': all_probs,
    }


def save_metrics_figure(history,
                        fig_path: str,
                        eval_interval: int):
    if not history['epoch']:
        return

    epochs = history['epoch']
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    metric_names = ['acc', 'precision', 'recall', 'f1', 'auc']
    for i, name in enumerate(metric_names):
        axes[i].plot(epochs, history[name], marker='o', linewidth=1.8)
        axes[i].set_title(name.upper())
        axes[i].set_xlabel(f'Epoch (eval every {eval_interval})')
        axes[i].set_ylabel(name)
        axes[i].grid(alpha=0.3)

    # 最后一个子图显示文字摘要
    axes[-1].axis('off')
    best_f1 = np.nanmax(history['f1']) if history['f1'] else float('nan')
    valid_auc = [x for x in history['auc'] if np.isfinite(x)]
    best_auc = max(valid_auc) if valid_auc else float('nan')
    axes[-1].text(
        0.02,
        0.75,
        f'Best F1  : {best_f1:.4f}\nBest AUC : {best_auc:.4f}\nEvaluations: {len(epochs)}',
        fontsize=12,
    )

    fig.suptitle('ResNet Task2 Baseline Metrics', fontsize=15)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=180)
    plt.close(fig)


def setup_logger(log_path: str):
    logger = logging.getLogger('resnet_task2_baseline')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def parse_args():
    parser = argparse.ArgumentParser(
        description='ResNet Task2 baseline: benign(1) vs non_neoplastic(0)'
    )
    parser.add_argument('--dataset_root', type=str, default='dataset0312')
    parser.add_argument('--test_size', type=float, default=0.30)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_interval', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--dropout', type=float, default=0.3)

    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--resize_size', type=int, default=256)
    parser.add_argument('--roi_pad_ratio', type=float, default=0.15)
    parser.add_argument('--use_imagenet_norm', action='store_true')
    parser.add_argument('--no_augmentation', action='store_true',
                        help='Disable all train-time data augmentations')

    parser.add_argument('--p_a', type=float, default=0.8, help='OpenCV augment A(deidentify) prob')
    parser.add_argument('--p_b', type=float, default=0.4, help='OpenCV augment B(histeq) prob')
    parser.add_argument('--p_c', type=float, default=0.5, help='OpenCV augment C(gamma) prob')
    parser.add_argument('--p_d', type=float, default=0.5, help='OpenCV augment D(gaussian+clahe) prob')

    parser.add_argument('--save_dir_logs', type=str, default='logs')
    parser.add_argument('--save_dir_ckpt', type=str, default='checkpoints')

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir_logs, exist_ok=True)
    os.makedirs(args.save_dir_ckpt, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(args.save_dir_logs, f'resnet_task2_baseline_{timestamp}.log')
    fig_path = os.path.join(args.save_dir_logs, f'resnet_task2_baseline_metrics_{timestamp}.png')
    ckpt_best = os.path.join(args.save_dir_ckpt, f'resnet_task2_baseline_best_auc_{timestamp}.pth')
    ckpt_last = os.path.join(args.save_dir_ckpt, f'resnet_task2_baseline_final_{timestamp}.pth')

    log = setup_logger(log_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    log.info('=' * 72)
    log.info('ResNet Task2 Baseline: benign(1) vs non_neoplastic(0)')
    log.info('=' * 72)
    log.info(f'设备: {device}')
    log.info(f'日志: {log_path}')
    log.info(f'指标曲线: {fig_path}')

    if args.no_augmentation:
        args.p_a = 0.0
        args.p_b = 0.0
        args.p_c = 0.0
        args.p_d = 0.0
        log.info('训练模式: 无数据增强（关闭 OpenCV 随机增强 + 随机变换）')
    else:
        log.info('训练模式: 启用数据增强（OpenCV + 随机几何/颜色增强）')

    set_seed(args.seed)

    # 1) 患者级加载与划分
    benign_pts = load_patients(args.dataset_root, 'benign')
    non_neo_pts = load_patients(args.dataset_root, 'non_neoplastic')
    log.info(f'患者数: benign={len(benign_pts)}, non_neoplastic={len(non_neo_pts)}')

    b_tr, b_te = train_test_split(
        benign_pts,
        test_size=args.test_size,
        random_state=args.seed,
    )
    n_tr, n_te = train_test_split(
        non_neo_pts,
        test_size=args.test_size,
        random_state=args.seed,
    )

    log.info(f'训练患者: benign={len(b_tr)}, non_neoplastic={len(n_tr)}')
    log.info(f'测试患者: benign={len(b_te)}, non_neoplastic={len(n_te)}')

    train_samples = to_samples(b_tr, 1) + to_samples(n_tr, 0)
    test_samples = to_samples(b_te, 1) + to_samples(n_te, 0)
    random.shuffle(train_samples)

    log_split(log, '训练集', train_samples)
    log_split(log, '测试集', test_samples)

    # 2) Transform + DataLoader
    train_tf, test_tf = build_transforms(
        image_size=args.image_size,
        resize_size=args.resize_size,
        use_imagenet_norm=args.use_imagenet_norm,
        no_augmentation=args.no_augmentation,
    )

    train_loader = DataLoader(
        Task2BaselineDataset(
            train_samples,
            transform=train_tf,
            training=True,
            roi_pad_ratio=args.roi_pad_ratio,
            p_a=args.p_a,
            p_b=args.p_b,
            p_c=args.p_c,
            p_d=args.p_d,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        Task2BaselineDataset(
            test_samples,
            transform=test_tf,
            training=False,
            roi_pad_ratio=args.roi_pad_ratio,
            p_a=args.p_a,
            p_b=args.p_b,
            p_c=args.p_c,
            p_d=args.p_d,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # 3) 模型、损失、优化器
    model = build_model(dropout=args.dropout).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    log.info(f'ResNet18(from scratch) 参数量: {total_params:,}')

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )

    # 4) 训练 + 周期评估
    history = {
        'epoch': [],
        'acc': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auc': [],
    }

    best_auc = -1.0
    best_epoch = -1

    log.info(f'开始训练: epochs={args.epochs}, eval_interval={args.eval_interval}')

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_total = 0
        n_correct = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}/{args.epochs}', leave=False, ncols=95)
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs >= args.threshold).long()
            n_correct += (preds == labels.long()).sum().item()
            n_total += imgs.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()

        train_loss = total_loss / max(1, n_total)
        train_acc = n_correct / max(1, n_total)
        lr_now = optimizer.param_groups[0]['lr']
        log.info(
            f'Epoch {epoch:03d}/{args.epochs} | '
            f'loss={train_loss:.4f}  train_acc={train_acc:.4f}  lr={lr_now:.2e}'
        )

        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            metrics = evaluate(model, test_loader, device=device, threshold=args.threshold)
            log.info(
                f'  >> [Eval {epoch:03d}] '
                f'acc={metrics["acc"]:.4f}  '
                f'precision={metrics["precision"]:.4f}  '
                f'recall={metrics["recall"]:.4f}  '
                f'f1={metrics["f1"]:.4f}  '
                f'auc={metrics["auc"]:.4f}' if np.isfinite(metrics['auc'])
                else f'  >> [Eval {epoch:03d}] '
                     f'acc={metrics["acc"]:.4f}  '
                     f'precision={metrics["precision"]:.4f}  '
                     f'recall={metrics["recall"]:.4f}  '
                     f'f1={metrics["f1"]:.4f}  '
                     f'auc=nan'
            )

            history['epoch'].append(epoch)
            history['acc'].append(metrics['acc'])
            history['precision'].append(metrics['precision'])
            history['recall'].append(metrics['recall'])
            history['f1'].append(metrics['f1'])
            history['auc'].append(metrics['auc'])

            auc_for_select = metrics['auc'] if np.isfinite(metrics['auc']) else -1.0
            if auc_for_select > best_auc:
                best_auc = auc_for_select
                best_epoch = epoch
                torch.save(model.state_dict(), ckpt_best)
                log.info(f'  >> [★] 保存最优AUC权重: epoch={epoch}, auc={auc_for_select:.4f}')

    # 5) 保存最终模型
    torch.save(model.state_dict(), ckpt_last)
    log.info(f'保存最终权重: {ckpt_last}')

    # 6) 若存在 best，则加载并做最终报告；否则用最后模型
    if os.path.exists(ckpt_best):
        model.load_state_dict(torch.load(ckpt_best, map_location=device))
        log.info(f'加载 best AUC 权重进行最终评估: epoch={best_epoch}, auc={best_auc:.4f}')
    else:
        log.warning('未保存到有效 best AUC 权重，将使用最终模型做评估。')

    final_metrics = evaluate(model, test_loader, device=device, threshold=args.threshold)

    cm = confusion_matrix(final_metrics['targets'], final_metrics['preds'])
    report = classification_report(
        final_metrics['targets'],
        final_metrics['preds'],
        target_names=['non_neoplastic(0)', 'benign(1)'],
        zero_division=0,
    )

    log.info('=' * 72)
    log.info('最终评估结果')
    log.info(
        f'acc={final_metrics["acc"]:.4f} | '
        f'precision={final_metrics["precision"]:.4f} | '
        f'recall={final_metrics["recall"]:.4f} | '
        f'f1={final_metrics["f1"]:.4f} | '
        f'auc={final_metrics["auc"]:.4f}' if np.isfinite(final_metrics['auc'])
        else f'acc={final_metrics["acc"]:.4f} | '
             f'precision={final_metrics["precision"]:.4f} | '
             f'recall={final_metrics["recall"]:.4f} | '
             f'f1={final_metrics["f1"]:.4f} | '
             f'auc=nan'
    )
    log.info(f'混淆矩阵:\n{cm}')
    log.info(f'分类报告:\n{report}')

    save_metrics_figure(history, fig_path=fig_path, eval_interval=args.eval_interval)
    log.info(f'指标曲线已保存: {fig_path}')

    log.info('=' * 72)
    log.info('训练完成')
    log.info(f'Best(AUC) : {ckpt_best}')
    log.info(f'Final     : {ckpt_last}')
    log.info(f'Log       : {log_path}')


if __name__ == '__main__':
    main()
