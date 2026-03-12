"""
ResNet50 从零训练：良性腺瘤(benign=1) vs 非肿瘤性息肉(non_neoplastic=0)
========================================================================
- 数据集：dataset0312/{benign, non_neoplastic}，患者级 7:3 分割
- benign : non_neoplastic ≈ 1:3（患者级随机采样）
- 全参数从零训练（无预训练权重）
- 数据增强：
    A. 信息脱敏（遮盖超声图像边缘文字区域）
    B. 全局直方图均衡化
    C. 伽马变化算法提升
    D. 高斯去噪 + CLAHE
- 评估指标：AUC / Acc / Precision / Recall / F1
- 每 EVAL_INTERVAL 个 epoch 评估并写入日志，最优 AUC 自动保存权重
"""

import os
import glob
import json
import random
import logging
import warnings
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
)
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# 超参数
# ============================================================
DATASET_ROOT   = 'dataset0312'
RATIO_NEG      = 3          # non_neoplastic : benign = 3 : 1（患者级）
TEST_SIZE      = 0.30       # 患者级测试比例
RANDOM_SEED    = 42
IMAGE_SIZE     = 224
BATCH_SIZE     = 32
NUM_EPOCHS     = 60
EVAL_INTERVAL  = 3          # 每 N epoch 评估一次
LR_INIT        = 1e-3       # 从零训练，SGD 配大 LR
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 4
THRESHOLD      = 0.5        # benign(正类=1) 判定阈值

# ============================================================
# 日志 & 路径
# ============================================================
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
TIMESTAMP  = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH   = f'logs/resnet_task2_{TIMESTAMP}.log'
CKPT_PATH  = f'checkpoints/resnet50_task2_{TIMESTAMP}.pth'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info('=' * 68)
log.info('  ResNet50 Task2: 良性腺瘤(1) vs 非肿瘤性息肉(0) — 从零训练')
log.info('=' * 68)
log.info(f'设备: {device}  |  日志: {LOG_PATH}  |  权重: {CKPT_PATH}')


# ============================================================
# A. 信息脱敏 —— 遮盖超声图像四边文字/机器信息区域
# ============================================================
def deidentify(img: np.ndarray,
               top: float = 0.08, bottom: float = 0.05,
               side: float = 0.06) -> np.ndarray:
    h, w = img.shape[:2]
    fill = int(img.mean())          # 用图像均值填充，避免引入纯黑/白伪影

    def r(base):                    # 加 ±0.02 随机扰动
        return max(0.0, base + random.uniform(-0.02, 0.02))

    out = img.copy()
    t, b, l, ri = int(h * r(top)), int(h * r(bottom)), int(w * r(side)), int(w * r(side))
    if t  > 0: out[:t,    :]  = fill
    if b  > 0: out[h - b:, :] = fill
    if l  > 0: out[:, :l]     = fill
    if ri > 0: out[:, w - ri:] = fill
    return out


# ============================================================
# B. 全局直方图均衡化 —— 对亮度通道操作
# ============================================================
def global_histeq(img: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# ============================================================
# C. 伽马变化算法提升 —— 随机伽马矫正
# ============================================================
def gamma_correction(img: np.ndarray,
                     lo: float = 0.6, hi: float = 1.8) -> np.ndarray:
    gamma = random.uniform(lo, hi)
    table = np.clip((np.arange(256) / 255.0) ** (1.0 / gamma) * 255.0,
                    0, 255).astype(np.uint8)
    return cv2.LUT(img, table)


# ============================================================
# D. 高斯去噪 + CLAHE
# ============================================================
def gaussian_clahe(img: np.ndarray,
                   ksize: int = 3,
                   clip: float = 2.0,
                   grid: tuple = (8, 8)) -> np.ndarray:
    denoised = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=0)
    ycrcb = cv2.cvtColor(denoised, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)


# ============================================================
# ROI 裁剪（读取同名 JSON 边界框，+15% padding）
# ============================================================
def crop_roi(img_path: str) -> Image.Image:
    image = Image.open(img_path).convert('RGB')
    json_path = os.path.splitext(img_path)[0] + '.json'
    if not os.path.exists(json_path):
        return image
    try:
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        shapes = data.get('shapes', [])
        if not shapes:
            return image
        pts = shapes[0]['points']
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        W, H = image.size
        px, py = (x2 - x1) * 0.15, (y2 - y1) * 0.15
        image = image.crop((max(0, x1 - px), max(0, y1 - py),
                            min(W, x2 + px), min(H, y2 + py)))
    except Exception:
        pass
    return image


# ============================================================
# OpenCV 增强管线（仅训练时随机触发）
# ============================================================
P_A, P_B, P_C, P_D = 0.8, 0.4, 0.5, 0.5

def opencv_augment(pil_img: Image.Image, training: bool) -> Image.Image:
    if not training:
        return pil_img
    img = np.array(pil_img)          # RGB uint8
    if random.random() < P_A:
        img = deidentify(img)
    if random.random() < P_B:
        img = global_histeq(img)
    if random.random() < P_C:
        img = gamma_correction(img)
    if random.random() < P_D:
        img = gaussian_clahe(img)
    return Image.fromarray(img)


# ============================================================
# torchvision Transform（从零训练用 [-1,1] 归一化）
# ============================================================
TRAIN_TF = T.Compose([
    T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    T.RandomCrop(IMAGE_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=20),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.15, hue=0.05),
    T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15),
                   interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    T.RandomErasing(p=0.2, scale=(0.02, 0.08)),
])

TEST_TF = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ============================================================
# Dataset
# ============================================================
class Task2Dataset(Dataset):
    def __init__(self, samples: list, training: bool = True):
        # samples: [(img_path, label), ...]  label: 1=benign, 0=non_neoplastic
        self.samples  = samples
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        pil = crop_roi(img_path)
        pil = opencv_augment(pil, self.training)
        tf  = TRAIN_TF if self.training else TEST_TF
        return tf(pil), torch.tensor(label, dtype=torch.float32)


# ============================================================
# 数据加载
# ============================================================
def load_patients(cls_name: str) -> list:
    """返回 [(pid, [img_paths]), ...]"""
    cls_dir = os.path.join(DATASET_ROOT, cls_name)
    result  = []
    for pid in sorted(os.listdir(cls_dir)):
        p_dir = os.path.join(cls_dir, pid)
        if not os.path.isdir(p_dir):
            continue
        imgs = sorted(glob.glob(os.path.join(p_dir, '*.png')) +
                      glob.glob(os.path.join(p_dir, '*.JPG')))
        if imgs:
            result.append((pid, imgs))
    return result


def to_samples(patient_list: list, label: int) -> list:
    return [(img, label) for pid, imgs in patient_list for img in imgs]


def log_split(tag: str, samples: list):
    pos = sum(s[1] for s in samples)
    neg = len(samples) - pos
    log.info(f'{tag}: 共 {len(samples)} 张 | '
             f'benign(1)={pos} | non_neo(0)={neg} | 比例 1:{neg/max(1,pos):.2f}')


# ============================================================
# 模型：ResNet50 从零训练
# ============================================================
def build_model() -> nn.Module:
    model = models.resnet50(weights=None)        # 无预训练
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, 1),
    )
    total = sum(p.numel() for p in model.parameters())
    log.info(f'ResNet50 (从零训练) 全参数: {total:,}')
    return model.to(device)


# ============================================================
# 评估：返回完整指标字典
# ============================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    for imgs, lbls in loader:
        logits = model(imgs.to(device)).squeeze(1)
        probs  = torch.sigmoid(logits).cpu().numpy()
        preds  = (probs > THRESHOLD).astype(int)
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_targets.extend(lbls.numpy().astype(int).tolist())

    try:
        auc = roc_auc_score(all_targets, all_probs)
    except Exception:
        auc = 0.0

    acc  = accuracy_score(all_targets, all_preds)
    prec = precision_score(all_targets, all_preds, zero_division=0)
    rec  = recall_score(all_targets, all_preds, zero_division=0)
    f1   = f1_score(all_targets, all_preds, zero_division=0)

    return dict(auc=auc, acc=acc, precision=prec, recall=rec, f1=f1,
                targets=all_targets, preds=all_preds, probs=all_probs)


def log_metrics(tag: str, m: dict):
    log.info(
        f'{tag} | AUC={m["auc"]:.4f}  acc={m["acc"]:.4f}  '
        f'precision={m["precision"]:.4f}  recall={m["recall"]:.4f}  '
        f'f1={m["f1"]:.4f}'
    )


def full_report(model: nn.Module, loader: DataLoader, tag: str):
    m = evaluate(model, loader)
    log_metrics(tag, m)
    cm = confusion_matrix(m['targets'], m['preds'])
    report = classification_report(
        m['targets'], m['preds'],
        target_names=['non_neoplastic(0)', 'benign(1)'],
        zero_division=0,
    )
    log.info(f'混淆矩阵:\n{cm}')
    log.info(f'\n{report}')
    return m['auc']


# ============================================================
# 主流程
# ============================================================
def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    log.info(f'PyTorch {torch.__version__}  |  数据集: {DATASET_ROOT}')

    # ── 1. 加载患者列表 ──────────────────────────────────────
    benign_pts  = load_patients('benign')
    non_neo_pts = load_patients('non_neoplastic')
    log.info(f'原始: benign={len(benign_pts)}, non_neoplastic={len(non_neo_pts)}')

    # 按 1:RATIO_NEG 采样 non_neoplastic（患者级）
    rng   = random.Random(RANDOM_SEED)
    n_neg = min(len(non_neo_pts), len(benign_pts) * RATIO_NEG)
    non_neo_pts = rng.sample(non_neo_pts, n_neg)
    log.info(f'采样后: benign={len(benign_pts)}, non_neoplastic={len(non_neo_pts)} '
             f'(1:{n_neg/len(benign_pts):.2f})')

    # ── 2. 患者级 7:3 分割 ───────────────────────────────────
    b_tr, b_te = train_test_split(benign_pts,  test_size=TEST_SIZE, random_state=RANDOM_SEED)
    n_tr, n_te = train_test_split(non_neo_pts, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    log.info(f'训练患者: benign={len(b_tr)}, non_neo={len(n_tr)}')
    log.info(f'测试患者: benign={len(b_te)}, non_neo={len(n_te)}')

    train_samples = to_samples(b_tr, 1) + to_samples(n_tr, 0)
    test_samples  = to_samples(b_te, 1) + to_samples(n_te, 0)
    random.shuffle(train_samples)

    log_split('训练集', train_samples)
    log_split('测试集',  test_samples)

    # ── 3. DataLoader ────────────────────────────────────────
    train_loader = DataLoader(
        Task2Dataset(train_samples, training=True),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        Task2Dataset(test_samples, training=False),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── 4. 模型 / 损失 / 优化器 ─────────────────────────────
    model = build_model()

    pos   = sum(s[1] for s in train_samples)
    neg   = len(train_samples) - pos
    pw    = torch.tensor([neg / max(1, pos)], device=device)
    log.info(f'BCEWithLogitsLoss pos_weight(benign): {pw.item():.3f}')
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LR_INIT,
        momentum=0.9, weight_decay=WEIGHT_DECAY, nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=LR_INIT * 0.01,
    )

    # ── 5. 训练循环 ──────────────────────────────────────────
    best_auc = 0.0
    log.info(f'开始训练: {NUM_EPOCHS} epochs，每 {EVAL_INTERVAL} epoch 评估一次')
    log.info(f'权重保存路径: {CKPT_PATH}')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = n_correct = n_total = 0

        pbar = tqdm(train_loader,
                    desc=f'Epoch {epoch:03d}/{NUM_EPOCHS}',
                    leave=False, ncols=90)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds       = (torch.sigmoid(logits) > THRESHOLD).long()
            n_correct  += (preds == lbls.long()).sum().item()
            n_total    += imgs.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        scheduler.step()

        avg_loss  = total_loss / n_total
        train_acc = n_correct  / n_total
        lr_now    = optimizer.param_groups[0]['lr']

        log.info(f'Epoch {epoch:03d}/{NUM_EPOCHS} | '
                 f'loss={avg_loss:.4f}  train_acc={train_acc:.4f}  lr={lr_now:.2e}')

        # 评估
        if epoch % EVAL_INTERVAL == 0 or epoch == NUM_EPOCHS:
            m = evaluate(model, test_loader)
            log_metrics(f'  >> [Eval Epoch {epoch:03d}]', m)

            if m['auc'] > best_auc:
                best_auc = m['auc']
                torch.save(model.state_dict(), CKPT_PATH)
                log.info(f'  >> [★] 保存最优权重  AUC={best_auc:.4f}  '
                         f'acc={m["acc"]:.4f}  precision={m["precision"]:.4f}  '
                         f'recall={m["recall"]:.4f}  f1={m["f1"]:.4f}')

    # ── 6. 最终完整报告 ──────────────────────────────────────
    log.info('\n' + '=' * 68)
    log.info(f'加载最优权重 (best AUC={best_auc:.4f}) 做完整评估')
    log.info('=' * 68)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device, weights_only=True))
    final_auc = full_report(model, test_loader, '最终评估')
    log.info(f'\n训练完成 | 最优AUC={best_auc:.4f} | 最终AUC={final_auc:.4f}')
    log.info(f'日志: {LOG_PATH}')
    log.info(f'权重: {CKPT_PATH}')


if __name__ == '__main__':
    main()
