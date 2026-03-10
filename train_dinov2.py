"""
DINOv2 ViT-B/14 胆囊息肉分类训练脚本
- 使用 facebookresearch/dinov2 官方预训练权重（自监督，泛化能力强）
- 两级级联任务（恶性拦截 + 腺瘤/非肿瘤区分）
- 患者级数据集分割（防止数据泄漏）
- 全量图片利用（每位患者所有超声图）
- 激进数据增强
- Focal Loss 处理极度类别不平衡
- AdamW + CosineAnnealingLR

依赖说明：
  DINOv2 通过 torch.hub 加载，首次运行需联网下载权重（约330MB）
  缓存路径：~/.cache/torch/hub/facebookresearch_dinov2_main/
"""

import os
import glob
import json
import logging
import warnings
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ============================================================
# 日志配置
# ============================================================
os.makedirs('logs', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH = f'logs/dinov2_{TIMESTAMP}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_PATH, encoding='utf-8'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f'======== DINOv2 ViT-B/14 训练脚本 ========')
log.info(f'使用设备: {device}')
log.info(f'日志路径: {LOG_PATH}')

# ============================================================
# ROI 裁剪
# ============================================================
def crop_roi(img_path: str) -> Image.Image:
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
        points = shapes[0]['points']
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        w, h = image.size
        px = (x2 - x1) * 0.15
        py = (y2 - y1) * 0.15
        image = image.crop((
            max(0, x1 - px), max(0, y1 - py),
            min(w, x2 + px), min(h, y2 + py)
        ))
    except Exception:
        pass
    return image


# ============================================================
# Dataset
# ============================================================
class ROIDataset(Dataset):
    def __init__(self, data_list: list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        image = crop_roi(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label


# ============================================================
# 数据加载
# ============================================================
def load_all_patients(label_file: str = 'dataset_labels.json') -> list:
    with open(label_file, 'r') as f:
        labels = json.load(f)

    patients = []
    skipped = 0
    for cls_name, p_list in labels.items():
        if cls_name == 'unknown':
            continue
        for p in p_list:
            imgs = sorted(
                glob.glob(os.path.join('dataset', p, '*.png')) +
                glob.glob(os.path.join('dataset', p, '*.JPG'))
            )
            if not imgs:
                skipped += 1
                continue
            patients.append((p, cls_name, imgs))

    log.info(f'加载患者: {len(patients)} 人（跳过无图像: {skipped} 人）')
    cls_dist = {}
    for _, cls_name, imgs in patients:
        cls_dist[cls_name] = cls_dist.get(cls_name, 0) + 1
    log.info(f'患者类别分布: {cls_dist}')
    return patients


def patients_to_images(patients: list, task: str = 'task1') -> list:
    data = []
    for pid, cls_name, imgs in patients:
        if task == 'task1':
            label = 1 if cls_name == 'malignant' else 0
        else:
            if cls_name == 'malignant':
                continue
            label = 1 if cls_name == 'benign' else 0
        for img_path in imgs:
            data.append((img_path, label))
    return data


def log_data_stats(tag: str, data: list):
    pos = sum(d[1] for d in data)
    neg = len(data) - pos
    log.info(f'{tag}: 共{len(data)}张 | 正样本(1)={pos} | 负样本(0)={neg} | '
             f'正负比=1:{neg/max(1,pos):.1f}')


# ============================================================
# 数据增强
# DINOv2 输入尺寸须为 patch_size(14) 的整数倍，使用 224x224
# ============================================================
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.85, 1.15),
                            interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ============================================================
# Focal Loss
# ============================================================
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none', pos_weight=self.pos_weight
        )
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()


# ============================================================
# 模型：DINOv2 ViT-B/14 + 分类头
#   冻结所有参数，仅解冻最后3个 Transformer Block（9, 10, 11）
#   + 最终 LayerNorm + 自定义分类头
# ============================================================
class DINOv2Classifier(nn.Module):
    """
    包装 DINOv2 backbone，添加分类头。
    backbone(x) 返回 CLS token 特征向量 [B, 768]
    """
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim  # ViT-B: 768
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(p=0.3),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone forward: returns CLS token or dict
        out = self.backbone(x)
        if isinstance(out, dict):
            features = out['x_norm_clstoken']  # [B, 768]
        else:
            features = out                     # [B, 768]
        return self.classifier(features)


def build_dinov2() -> nn.Module:
    log.info('正在加载 DINOv2 ViT-B/14 预训练权重（首次运行需联网下载）...')
    try:
        backbone = torch.hub.load(
            'facebookresearch/dinov2', 'dinov2_vitb14',
            verbose=False
        )
    except Exception as e:
        log.error(f'DINOv2 加载失败: {e}')
        log.error('请检查网络连接，或手动下载权重后放置于 torch hub 缓存目录')
        raise

    # 冻结所有参数
    for param in backbone.parameters():
        param.requires_grad = False

    # 解冻最后 3 个 Transformer Block + LayerNorm
    UNFREEZE_BLOCKS = {9, 10, 11}
    for name, param in backbone.named_parameters():
        block_match = any(f'blocks.{i}.' in name for i in UNFREEZE_BLOCKS)
        norm_match = name.startswith('norm.')
        if block_match or norm_match:
            param.requires_grad = True

    model = DINOv2Classifier(backbone).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    log.info(f'DINOv2 ViT-B/14: 可训练参数 {trainable:,} / 全部 {total:,} '
             f'({100*trainable/total:.1f}%)')
    return model


# ============================================================
# 评估
# ============================================================
def evaluate(model: nn.Module, loader: DataLoader, threshold: float = 0.5):
    model.eval()
    all_probs, all_preds, all_targets = [], [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            logits = model(imgs.to(device)).squeeze()
            if logits.ndim == 0:
                logits = logits.unsqueeze(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend((probs > threshold).astype(int))
            all_targets.extend(lbls.numpy())
    return all_targets, all_preds, all_probs


def quick_auc(model: nn.Module, loader: DataLoader) -> float:
    targets, _, probs = evaluate(model, loader)
    try:
        return roc_auc_score(targets, probs)
    except Exception:
        return 0.0


def full_report(model: nn.Module, loader: DataLoader, task_name: str,
                threshold: float = 0.5) -> float:
    targets, preds, probs = evaluate(model, loader, threshold)
    try:
        auc = roc_auc_score(targets, probs)
        log.info(f'ROC-AUC: {auc:.4f}')
    except Exception:
        auc = 0.0
        log.info('AUC 计算失败（测试集可能只含单一类别）')
    report = classification_report(
        targets, preds,
        target_names=['0类（负样本）', '1类（正样本/重点）'],
        zero_division=0
    )
    log.info(f'\n{report}')
    return auc


# ============================================================
# 单任务训练流程
# ============================================================
def train_task(task_name: str, task_key: str,
               train_patients: list, test_patients: list,
               num_epochs: int = 20):
    log.info(f'\n{"="*60}')
    log.info(f'开始训练: {task_name}')
    log.info(f'{"="*60}')

    train_data = patients_to_images(train_patients, task=task_key)
    test_data  = patients_to_images(test_patients,  task=task_key)
    log_data_stats(f'[训练集] {task_key}', train_data)
    log_data_stats(f'[测试集] {task_key}', test_data)

    if not train_data or not test_data:
        log.warning('数据为空，跳过该任务。')
        return

    train_loader = DataLoader(
        ROIDataset(train_data, TRAIN_TRANSFORM),
        batch_size=32, shuffle=True, num_workers=4,
        pin_memory=True, drop_last=True
    )
    test_loader = DataLoader(
        ROIDataset(test_data, TEST_TRANSFORM),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    pos_count = sum(d[1] for d in train_data)
    neg_count = len(train_data) - pos_count
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], device=device)
    log.info(f'Focal Loss pos_weight: {pos_weight.item():.2f}')
    criterion = FocalLoss(gamma=2.0, pos_weight=pos_weight)

    model = build_dinov2()

    # DINOv2 Transformer 使用更小的学习率（精细微调）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-7
    )

    best_auc = 0.0
    ckpt_path = f'checkpoints/dinov2_{task_key}_{TIMESTAMP}.pth'
    log.info(f'最优模型将保存至: {ckpt_path}')

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for imgs, lbls in tqdm(train_loader, desc=f'Epoch {epoch:02d}/{num_epochs}',
                               leave=False, ncols=80):
            imgs = imgs.to(device)
            lbls = lbls.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs).squeeze(), lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_data)
        lr_now = optimizer.param_groups[0]['lr']
        log.info(f'Epoch {epoch:02d}/{num_epochs} | Loss={avg_loss:.4f} | LR={lr_now:.2e}')
        scheduler.step()

        if epoch % 5 == 0 or epoch == num_epochs:
            auc = quick_auc(model, test_loader)
            log.info(f'           >> Val AUC: {auc:.4f}')
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), ckpt_path)
                log.info(f'           >> 保存最优模型 (AUC={best_auc:.4f})')

    log.info(f'\n--- 最终完整评估: {task_name} ---')
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    final_auc = full_report(model, test_loader, task_name)
    log.info(f'最优 AUC: {best_auc:.4f} | 最终评估 AUC: {final_auc:.4f}')


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    log.info(f'PyTorch: {torch.__version__}')

    all_patients = load_all_patients()

    strat = [1 if p[1] == 'malignant' else 0 for p in all_patients]
    train_patients, test_patients = train_test_split(
        all_patients, test_size=0.2, random_state=42, stratify=strat
    )
    log.info(f'患者分割: 训练={len(train_patients)} | 测试={len(test_patients)}')

    # 任务一：恶性 vs 非恶性
    train_task(
        task_name='任务一: 恶性(1) vs 非恶性(0)',
        task_key='task1',
        train_patients=train_patients,
        test_patients=test_patients,
        num_epochs=20,
    )

    # 任务二：良性腺瘤 vs 非肿瘤性息肉
    train_task(
        task_name='任务二: 良性腺瘤(1) vs 非肿瘤性息肉(0)',
        task_key='task2',
        train_patients=train_patients,
        test_patients=test_patients,
        num_epochs=20,
    )

    log.info('\n======== DINOv2 所有任务完成 ========')
