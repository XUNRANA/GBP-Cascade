"""
ResNet18 ImageNet预训练微调 v3：良性腺瘤(1) vs 非肿瘤性息肉(0)
================================================================
vs v2 改动：
  1. ImageNet 预训练权重 + 分层微调（冻结 layer1-2，解冻 layer3-4 + head）
  2. 差分学习率：backbone 1e-4 / head 1e-3
  3. ImageNet 归一化（匹配预训练分布）
  4. 60 epochs（预训练收敛更快）
  其余保留 v2：WeightedRandomSampler / Focal Loss / MixUp / 增强 A-D
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
BATCH_SIZE     = 16
NUM_EPOCHS     = 60
EVAL_INTERVAL  = 2
LR_BACKBONE    = 1e-4        # 预训练层用小 LR 防遗忘
LR_HEAD        = 1e-3        # 新分类头用大 LR 快速收敛
WEIGHT_DECAY   = 1e-4
NUM_WORKERS    = 4
THRESHOLD      = 0.5
MIXUP_ALPHA    = 0.3
FOCAL_GAMMA    = 2.0

# ============================================================
# 日志 & 路径
# ============================================================
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
TIMESTAMP  = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH   = f'logs/resnet18_task2_v3_{TIMESTAMP}.log'
CKPT_BEST  = f'checkpoints/resnet18_task2_v3_best_{TIMESTAMP}.pth'
CKPT_LAST  = f'checkpoints/resnet18_task2_v3_last_{TIMESTAMP}.pth'

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
log.info('  ResNet18 Task2 v3: ImageNet预训练 + 分层微调')
log.info('  良性腺瘤(1) vs 非肿瘤性息肉(0)')
log.info('=' * 70)
log.info(f'设备: {device}  |  日志: {LOG_PATH}')


# ============================================================
# 数据增强 A-D（OpenCV）
# ============================================================
def deidentify(img: np.ndarray) -> np.ndarray:
    """A. 信息脱敏"""
    h, w = img.shape[:2]
    fill = int(img.mean())
    def r(base): return max(0.0, base + random.uniform(-0.02, 0.02))
    out = img.copy()
    t, b, l, ri = int(h*r(0.08)), int(h*r(0.05)), int(w*r(0.06)), int(w*r(0.06))
    if t  > 0: out[:t, :]      = fill
    if b  > 0: out[h-b:, :]    = fill
    if l  > 0: out[:, :l]      = fill
    if ri > 0: out[:, w-ri:]   = fill
    return out

def global_histeq(img: np.ndarray) -> np.ndarray:
    """B. 全局直方图均衡化"""
    y = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    y[:, :, 0] = cv2.equalizeHist(y[:, :, 0])
    return cv2.cvtColor(y, cv2.COLOR_YCrCb2RGB)

def gamma_correction(img: np.ndarray) -> np.ndarray:
    """C. 随机伽马变化"""
    gamma = random.uniform(0.5, 2.0)
    table = np.clip((np.arange(256)/255.0)**(1.0/gamma)*255, 0, 255).astype(np.uint8)
    return cv2.LUT(img, table)

def gaussian_clahe(img: np.ndarray) -> np.ndarray:
    """D. 高斯去噪 + CLAHE"""
    ksize = random.choice([3, 5])
    clip  = random.uniform(1.5, 3.5)
    d = cv2.GaussianBlur(img, (ksize, ksize), 0)
    y = cv2.cvtColor(d, cv2.COLOR_RGB2YCrCb)
    y[:, :, 0] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8)).apply(y[:, :, 0])
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
# transforms（ImageNet 归一化）
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

TRAIN_TF = T.Compose([
    T.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
    T.RandomCrop(IMAGE_SIZE),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.3),
    T.RandomRotation(25),
    T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.08),
    T.RandomAffine(15, translate=(0.15, 0.15), scale=(0.8, 1.2),
                   interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
])

TEST_TF = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# ============================================================
# Dataset
# ============================================================
class Task2Dataset(Dataset):
    def __init__(self, samples, training=True):
        self.samples  = samples
        self.training = training
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pil = crop_roi(path)
        pil = opencv_augment(pil, self.training)
        tf  = TRAIN_TF if self.training else TEST_TF
        return tf(pil), torch.tensor(label, dtype=torch.float32)


# ============================================================
# MixUp
# ============================================================
def mixup_batch(imgs, labels, alpha=0.3):
    if alpha <= 0:
        return imgs, labels
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(imgs.size(0), device=imgs.device)
    return lam * imgs + (1-lam) * imgs[idx], lam * labels + (1-lam) * labels[idx]


# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt  = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ============================================================
# 模型：ResNet18 ImageNet预训练 + 分层微调
# ============================================================
class ResNet18Pretrained(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # 冻结 conv1 + bn1 + layer1 + layer2（低级特征，保留 ImageNet 知识）
        frozen = [base.conv1, base.bn1, base.relu, base.maxpool,
                  base.layer1, base.layer2]
        for module in frozen:
            for p in module.parameters():
                p.requires_grad = False

        # 解冻 layer3 + layer4（高级语义，需要适配超声域）
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool,
            base.layer1, base.layer2,
            base.layer3, base.layer4,
            base.avgpool,
        )

        feat_dim = 512
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feat_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.head(feat).squeeze(1)


def build_model():
    model = ResNet18Pretrained().to(device)
    frozen   = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total    = frozen + trainable
    log.info(f'ResNet18 (ImageNet预训练): '
             f'冻结 {frozen:,} | 可训练 {trainable:,} | 总 {total:,} '
             f'({100*trainable/total:.1f}%)')
    return model


# ============================================================
# 数据加载
# ============================================================
def load_patients(cls_name):
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

def to_samples(patient_list, label):
    return [(img, label) for _, imgs in patient_list for img in imgs]

def log_split(tag, samples):
    pos = sum(s[1] for s in samples)
    neg = len(samples) - pos
    log.info(f'{tag}: 共 {len(samples)} 张 | '
             f'benign(1)={pos} | non_neo(0)={neg} | 比例 1:{neg/max(1,pos):.2f}')


# ============================================================
# WeightedRandomSampler
# ============================================================
def make_sampler(samples):
    labels = [s[1] for s in samples]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    w_pos, w_neg = 1.0/n_pos, 1.0/n_neg
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(samples)*2, replacement=True)


# ============================================================
# 评估
# ============================================================
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    for imgs, lbls in loader:
        logits = model(imgs.to(device))
        probs  = torch.sigmoid(logits).cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend((probs > THRESHOLD).astype(int).tolist())
        all_targets.extend(lbls.numpy().astype(int).tolist())
    try:    auc = roc_auc_score(all_targets, all_probs)
    except: auc = 0.0
    return dict(
        auc=auc,
        acc=accuracy_score(all_targets, all_preds),
        precision=precision_score(all_targets, all_preds, zero_division=0),
        recall=recall_score(all_targets, all_preds, zero_division=0),
        f1=f1_score(all_targets, all_preds, zero_division=0),
        targets=all_targets, preds=all_preds, probs=all_probs,
    )

def log_metrics(tag, m):
    log.info(f'{tag} | AUC={m["auc"]:.4f}  acc={m["acc"]:.4f}  '
             f'precision={m["precision"]:.4f}  recall={m["recall"]:.4f}  '
             f'f1={m["f1"]:.4f}')

def full_report(model, loader, tag):
    m = evaluate(model, loader)
    log_metrics(tag, m)
    log.info(f'混淆矩阵:\n{confusion_matrix(m["targets"], m["preds"])}')
    log.info('\n' + classification_report(
        m['targets'], m['preds'],
        target_names=['non_neoplastic(0)', 'benign(1)'], zero_division=0))
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

    # ── 1. 全量患者 ──────────────────────────────────────────
    benign_pts  = load_patients('benign')
    non_neo_pts = load_patients('non_neoplastic')
    log.info(f'全量: benign={len(benign_pts)}, non_neo={len(non_neo_pts)}')

    # ── 2. 患者级 7:3 分割 ───────────────────────────────────
    b_tr, b_te = train_test_split(benign_pts,  test_size=TEST_SIZE, random_state=RANDOM_SEED)
    n_tr, n_te = train_test_split(non_neo_pts, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    log.info(f'训练: benign={len(b_tr)}, non_neo={len(n_tr)}')
    log.info(f'测试: benign={len(b_te)}, non_neo={len(n_te)}')

    train_samples = to_samples(b_tr, 1) + to_samples(n_tr, 0)
    test_samples  = to_samples(b_te, 1) + to_samples(n_te, 0)
    random.shuffle(train_samples)
    log_split('训练集', train_samples)
    log_split('测试集', test_samples)

    # ── 3. DataLoader ────────────────────────────────────────
    sampler = make_sampler(train_samples)
    train_loader = DataLoader(
        Task2Dataset(train_samples, training=True),
        batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    test_loader = DataLoader(
        Task2Dataset(test_samples, training=False),
        batch_size=32, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True)
    steps_per_epoch = len(train_loader)

    # ── 4. 模型 / 损失 / 优化器 ─────────────────────────────
    model     = build_model()
    criterion = FocalLoss(gamma=FOCAL_GAMMA)

    # 差分学习率：backbone 小 LR / head 大 LR
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'head' not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'head' in n]
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE},
        {'params': head_params,     'lr': LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[LR_BACKBONE * 3, LR_HEAD * 3],   # 峰值
        epochs=NUM_EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=3,
        final_div_factor=100,
    )

    log.info(f'差分LR: backbone={LR_BACKBONE}, head={LR_HEAD}')
    log.info(f'开始训练: {NUM_EPOCHS} epochs | 每 {EVAL_INTERVAL} epoch 评估')

    # ── 5. 训练 ──────────────────────────────────────────────
    best_auc = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total_loss = n_correct = n_total = 0

        pbar = tqdm(train_loader, desc=f'Ep{epoch:03d}/{NUM_EPOCHS}',
                    leave=False, ncols=88)
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            imgs, lbls = mixup_batch(imgs, lbls, MIXUP_ALPHA)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * imgs.size(0)
            hard = lbls.round().long()
            preds = (torch.sigmoid(logits) > THRESHOLD).long()
            n_correct += (preds == hard).sum().item()
            n_total   += imgs.size(0)
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss  = total_loss / n_total
        train_acc = n_correct  / n_total
        lr_bb = optimizer.param_groups[0]['lr']
        lr_hd = optimizer.param_groups[1]['lr']
        log.info(f'Epoch {epoch:03d}/{NUM_EPOCHS} | loss={avg_loss:.4f}  '
                 f'train_acc={train_acc:.4f}  lr_bb={lr_bb:.2e}  lr_hd={lr_hd:.2e}')

        if epoch % EVAL_INTERVAL == 0 or epoch == NUM_EPOCHS:
            m = evaluate(model, test_loader)
            log_metrics(f'  >> [Eval {epoch:03d}]', m)
            if m['auc'] > best_auc:
                best_auc = m['auc']
                torch.save(model.state_dict(), CKPT_BEST)
                log.info(
                    f'  >> [★] 保存最优  AUC={m["auc"]:.4f}  acc={m["acc"]:.4f}  '
                    f'prec={m["precision"]:.4f}  rec={m["recall"]:.4f}  f1={m["f1"]:.4f}')

    # ── 6. 最终报告 ──────────────────────────────────────────
    torch.save(model.state_dict(), CKPT_LAST)
    log.info('\n' + '=' * 70)
    log.info(f'加载最优权重 (AUC={best_auc:.4f})')
    log.info('=' * 70)
    model.load_state_dict(torch.load(CKPT_BEST, map_location=device, weights_only=True))
    final_auc = full_report(model, test_loader, '最终评估')
    log.info(f'\n完成 | best_AUC={best_auc:.4f} | final_AUC={final_auc:.4f}')
    log.info(f'日志: {LOG_PATH}')
    log.info(f'最优: {CKPT_BEST}')
    log.info(f'最终: {CKPT_LAST}')


if __name__ == '__main__':
    main()
