"""
ResNet18 从零训练 v2：良性腺瘤(benign=1) vs 非肿瘤性息肉(non_neoplastic=0)
==========================================================================
vs v1 主要改动（解决 loss=1.1 卡死 / 预测永远为0 问题）：
  1. ResNet18 (11M) 替换 ResNet50 (23M) — 小数据从零训练更易收敛
  2. WeightedRandomSampler 代替 RATIO_NEG 截断 — 全量数据+batch内强制1:1
  3. AdamW + OneCycleLR (warmup+cosine) 替换 SGD — 小数据更快找到最优
  4. Focal Loss (gamma=2.5) — 聚焦难分样本，不被易分多数类淹没
  5. MixUp 数据增强 — 小数据场景最有效的正则化手段
  6. 额外 SE 注意力模块 — 通道权重自适应
  7. batch_size=16 — 梯度更新更频繁，适合小数据集
  8. 100 epochs + eval_interval=2

增强管线（A-D 保留）：
  A. 信息脱敏    B. 全局直方图均衡化    C. 伽马变化    D. 高斯去噪+CLAHE
"""

import os, glob, json, random, logging, warnings
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
TEST_SIZE      = 0.30
RANDOM_SEED    = 42
IMAGE_SIZE     = 224
BATCH_SIZE     = 16          # 小 batch → 更多梯度更新步数
NUM_EPOCHS     = 100
EVAL_INTERVAL  = 2
MAX_LR         = 3e-3        # OneCycleLR 峰值 LR
MIN_LR         = 1e-5
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 4
THRESHOLD      = 0.5
MIXUP_ALPHA    = 0.4         # MixUp 强度，0 表示关闭
FOCAL_GAMMA    = 2.5

# ============================================================
# 日志 & 路径
# ============================================================
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
TIMESTAMP  = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH   = f'logs/resnet18_task2_v2_{TIMESTAMP}.log'
CKPT_BEST  = f'checkpoints/resnet18_task2_v2_best_{TIMESTAMP}.pth'
CKPT_LAST  = f'checkpoints/resnet18_task2_v2_last_{TIMESTAMP}.pth'

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
log.info('=' * 70)
log.info('  ResNet18 Task2 v2: 良性腺瘤(1) vs 非肿瘤性息肉(0) — 从零训练')
log.info('=' * 70)
log.info(f'设备: {device}')
log.info(f'日志: {LOG_PATH}')
log.info(f'最优权重: {CKPT_BEST}')


# ============================================================
# 数据增强 A-D（OpenCV）
# ============================================================
def deidentify(img: np.ndarray) -> np.ndarray:
    """A. 信息脱敏：用均值填充四边文字/机器信息区域"""
    h, w = img.shape[:2]
    fill = int(img.mean())
    def r(base): return max(0.0, base + random.uniform(-0.02, 0.02))
    t, b, l, ri = int(h*r(0.08)), int(h*r(0.05)), int(w*r(0.06)), int(w*r(0.06))
    out = img.copy()
    if t  > 0: out[:t, :]      = fill
    if b  > 0: out[h-b:, :]    = fill
    if l  > 0: out[:, :l]      = fill
    if ri > 0: out[:, w-ri:]   = fill
    return out

def global_histeq(img: np.ndarray) -> np.ndarray:
    """B. 全局直方图均衡化（亮度通道）"""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y[:, :, 0] = cv2.equalizeHist(y[:, :, 0])
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB)

def gamma_correction(img: np.ndarray, lo=0.5, hi=2.0) -> np.ndarray:
    """C. 随机伽马变化（扩大范围到 0.5-2.0）"""
    gamma = random.uniform(lo, hi)
    table = np.clip((np.arange(256)/255.0)**(1.0/gamma)*255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)

def gaussian_clahe(img: np.ndarray) -> np.ndarray:
    """D. 高斯去噪 + CLAHE（随机参数）"""
    ksize = random.choice([3, 5])
    clip  = random.uniform(1.5, 3.5)
    denoised = cv2.GaussianBlur(img, (ksize, ksize), 0)
    y = cv2.cvtColor(denoised, cv2.COLOR_RGB2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    y[:, :, 0] = clahe.apply(y[:, :, 0])
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB)

def opencv_augment(pil_img: Image.Image, training: bool) -> Image.Image:
    if not training:
        return pil_img
    img = np.array(pil_img)
    if random.random() < 0.80: img = deidentify(img)
    if random.random() < 0.45: img = global_histeq(img)
    if random.random() < 0.55: img = gamma_correction(img)
    if random.random() < 0.55: img = gaussian_clahe(img)
    return Image.fromarray(img)


# ============================================================
# ROI 裁剪
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
        px, py = (x2-x1)*0.15, (y2-y1)*0.15
        image = image.crop((max(0,x1-px), max(0,y1-py),
                            min(W,x2+px), min(H,y2+py)))
    except Exception:
        pass
    return image


# ============================================================
# torchvision transforms（从零训练用 [-1,1]）
# ============================================================
TRAIN_TF = T.Compose([
    T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    T.RandomCrop(IMAGE_SIZE),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.3),
    T.RandomRotation(degrees=25),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.08),
    T.RandomAffine(degrees=15, translate=(0.15, 0.15), scale=(0.8, 1.2),
                   interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
    T.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.0)),
])

TEST_TF = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])


# ============================================================
# Dataset
# ============================================================
class Task2Dataset(Dataset):
    def __init__(self, samples: list, training: bool = True):
        self.samples  = samples
        self.training = training

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        pil = crop_roi(img_path)
        pil = opencv_augment(pil, self.training)
        tf  = TRAIN_TF if self.training else TEST_TF
        return tf(pil), torch.tensor(label, dtype=torch.float32)


# ============================================================
# MixUp
# ============================================================
def mixup_batch(imgs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.4):
    """对一个 batch 做 MixUp，返回混合图像和混合标签"""
    if alpha <= 0:
        return imgs, labels
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(imgs.size(0), device=imgs.device)
    imgs  = lam * imgs + (1 - lam) * imgs[idx]
    labels = lam * labels + (1 - lam) * labels[idx]
    return imgs, labels


# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.5):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ============================================================
# SE 注意力模块（插入 ResNet18 最后一层前）
# ============================================================
class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C)
        w = self.fc(x)
        return x * w


# ============================================================
# 模型：ResNet18 from scratch + SE Head
# ============================================================
class ResNet18Task2(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)     # 无预训练
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # 去掉 fc
        feat_dim = 512   # ResNet18 global avg pool 输出维度
        self.se   = SEBlock(feat_dim, reduction=16)
        self.head = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(feat_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x).flatten(1)   # (B, 512)
        feat = self.se(feat)
        return self.head(feat).squeeze(1)    # (B,)


def build_model() -> nn.Module:
    model = ResNet18Task2()
    total = sum(p.numel() for p in model.parameters())
    log.info(f'ResNet18+SE (从零训练) 全参数: {total:,}')
    return model.to(device)


# ============================================================
# 数据加载
# ============================================================
def load_patients(cls_name: str) -> list:
    cls_dir = os.path.join(DATASET_ROOT, cls_name)
    result  = []
    for pid in sorted(os.listdir(cls_dir)):
        p_dir = os.path.join(cls_dir, pid)
        if not os.path.isdir(p_dir): continue
        imgs = sorted(glob.glob(os.path.join(p_dir, '*.png')) +
                      glob.glob(os.path.join(p_dir, '*.JPG')))
        if imgs:
            result.append((pid, imgs))
    return result

def to_samples(patient_list: list, label: int) -> list:
    return [(img, label) for _, imgs in patient_list for img in imgs]

def log_split(tag: str, samples: list):
    pos = sum(s[1] for s in samples)
    neg = len(samples) - pos
    log.info(f'{tag}: 共 {len(samples)} 张 | '
             f'benign(1)={pos} | non_neo(0)={neg} | 比例 1:{neg/max(1,pos):.2f}')


# ============================================================
# WeightedRandomSampler：强制 batch 内 1:1 均衡
# ============================================================
def make_sampler(samples: list) -> WeightedRandomSampler:
    labels  = [s[1] for s in samples]
    n_pos   = sum(labels)
    n_neg   = len(labels) - n_pos
    w_pos   = 1.0 / n_pos if n_pos > 0 else 0
    w_neg   = 1.0 / n_neg if n_neg > 0 else 0
    weights = [w_pos if l == 1 else w_neg for l in labels]
    # num_samples 翻倍：让少数类被大量过采样，相当于扩充训练集
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(samples) * 2,
        replacement=True,
    )


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> dict:
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    for imgs, lbls in loader:
        logits = model(imgs.to(device))
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

def full_report(model: nn.Module, loader: DataLoader, tag: str) -> float:
    m = evaluate(model, loader)
    log_metrics(tag, m)
    log.info(f'混淆矩阵:\n{confusion_matrix(m["targets"], m["preds"])}')
    log.info('\n' + classification_report(
        m['targets'], m['preds'],
        target_names=['non_neoplastic(0)', 'benign(1)'],
        zero_division=0,
    ))
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
    log.info(f'超参: epochs={NUM_EPOCHS}  batch={BATCH_SIZE}  '
             f'max_lr={MAX_LR}  focal_gamma={FOCAL_GAMMA}  mixup_alpha={MIXUP_ALPHA}')

    # ── 1. 加载全量患者（不截断）──────────────────────────────
    benign_pts  = load_patients('benign')
    non_neo_pts = load_patients('non_neoplastic')
    log.info(f'全量患者: benign={len(benign_pts)}, non_neoplastic={len(non_neo_pts)}')

    # ── 2. 患者级 7:3 分层分割 ───────────────────────────────
    b_tr, b_te = train_test_split(benign_pts,  test_size=TEST_SIZE, random_state=RANDOM_SEED)
    n_tr, n_te = train_test_split(non_neo_pts, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    log.info(f'训练患者: benign={len(b_tr)}, non_neo={len(n_tr)}')
    log.info(f'测试患者: benign={len(b_te)},  non_neo={len(n_te)}')

    train_samples = to_samples(b_tr, 1) + to_samples(n_tr, 0)
    test_samples  = to_samples(b_te, 1) + to_samples(n_te, 0)
    random.shuffle(train_samples)
    log_split('训练集', train_samples)
    log_split('测试集',  test_samples)

    # ── 3. DataLoader（训练集用 WeightedRandomSampler）───────
    sampler = make_sampler(train_samples)
    train_loader = DataLoader(
        Task2Dataset(train_samples, training=True),
        batch_size=BATCH_SIZE,
        sampler=sampler,              # 不能与 shuffle=True 共存
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        Task2Dataset(test_samples, training=False),
        batch_size=32, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    steps_per_epoch = len(train_loader)
    log.info(f'每 epoch 步数: {steps_per_epoch}  (sampler 过采样后: {len(sampler)} 样本)')

    # ── 4. 模型 / 损失 / 优化器 ──────────────────────────────
    model     = build_model()
    criterion = FocalLoss(gamma=FOCAL_GAMMA)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=MAX_LR / 10,   # OneCycleLR 会自动调整
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=MAX_LR,
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.25,         # 前 25% epoch 线性 warmup
        anneal_strategy='cos',
        div_factor=10,          # 初始 LR = max_lr/10
        final_div_factor=1000,  # 最终 LR = max_lr/1000
    )

    # ── 5. 训练循环 ──────────────────────────────────────────
    best_auc = 0.0
    log.info(f'开始训练: {NUM_EPOCHS} epochs | 每 {EVAL_INTERVAL} epoch 评估')

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = n_correct = n_total = 0

        pbar = tqdm(train_loader, desc=f'Ep{epoch:03d}/{NUM_EPOCHS}',
                    leave=False, ncols=88)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)

            # MixUp
            imgs, lbls = mixup_batch(imgs, lbls, MIXUP_ALPHA)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * imgs.size(0)
            # 用硬标签算训练准确率（MixUp 后取最近整数标签）
            hard_lbls = lbls.round().long()
            preds     = (torch.sigmoid(logits) > THRESHOLD).long()
            n_correct += (preds == hard_lbls).sum().item()
            n_total   += imgs.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}',
                             lr=f'{scheduler.get_last_lr()[0]:.1e}')

        avg_loss  = total_loss / n_total
        train_acc = n_correct  / n_total
        lr_now    = scheduler.get_last_lr()[0]
        log.info(f'Epoch {epoch:03d}/{NUM_EPOCHS} | '
                 f'loss={avg_loss:.4f}  train_acc={train_acc:.4f}  lr={lr_now:.2e}')

        if epoch % EVAL_INTERVAL == 0 or epoch == NUM_EPOCHS:
            m = evaluate(model, test_loader)
            log_metrics(f'  >> [Eval {epoch:03d}]', m)

            if m['auc'] > best_auc:
                best_auc = m['auc']
                torch.save(model.state_dict(), CKPT_BEST)
                log.info(
                    f'  >> [★] 保存最优权重 '
                    f'AUC={m["auc"]:.4f}  acc={m["acc"]:.4f}  '
                    f'precision={m["precision"]:.4f}  recall={m["recall"]:.4f}  '
                    f'f1={m["f1"]:.4f}'
                )

    # ── 6. 保存最后权重 & 最终完整报告 ───────────────────────
    torch.save(model.state_dict(), CKPT_LAST)

    log.info('\n' + '=' * 70)
    log.info(f'加载最优权重 (AUC={best_auc:.4f}) 做最终完整评估')
    log.info('=' * 70)
    model.load_state_dict(torch.load(CKPT_BEST, map_location=device, weights_only=True))
    final_auc = full_report(model, test_loader, '最终评估')

    log.info(f'\n训练完成 | 最优AUC={best_auc:.4f} | 最终AUC={final_auc:.4f}')
    log.info(f'日志:      {LOG_PATH}')
    log.info(f'最优权重:  {CKPT_BEST}')
    log.info(f'最终权重:  {CKPT_LAST}')


if __name__ == '__main__':
    main()
