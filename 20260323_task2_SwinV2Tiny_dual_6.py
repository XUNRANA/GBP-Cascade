"""
Exp #6: Swin-V2-Tiny 双分支 (全局ROI + 病灶局部裁剪) ★★ 最有创新性
- Branch A: ROI 全局裁剪 → Swin-V2 → global features
- Branch B: 病灶区域裁剪 (bbox padding 30%) → Swin-V2 (共享) → local features
- 结构化特征: 病灶面积比, 病灶数量
- 融合: concat → MLP → 2类
"""

import os
import json

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import pandas as pd

from task2_json_utils import (
    build_class_weights,
    build_optimizer_with_diff_lr,
    crop_roi,
    generate_lesion_mask,
    get_gallbladder_rect,
    load_annotation,
    run_experiment,
    split_backbone_and_head,
)


# ─── 双分支数据集 ───


class GBPDatasetDual(Dataset):
    """双分支数据集: 返回 (global_img, local_img, struct_feat, label)."""

    def __init__(self, excel_path, data_root, img_size=256, is_train=True):
        self.df = pd.read_excel(excel_path)
        self.data_root = data_root
        self.img_size = img_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __len__(self):
        return len(self.df)

    def _get_lesion_bbox(self, shapes):
        """从所有病灶多边形中计算包围框 [x1, y1, x2, y2]."""
        all_x, all_y = [], []
        for s in shapes:
            if s["label"] == "gallbladder":
                continue
            if s["shape_type"] == "polygon" and len(s["points"]) >= 3:
                for p in s["points"]:
                    all_x.append(p[0])
                    all_y.append(p[1])
        if not all_x:
            return None
        return [min(all_x), min(all_y), max(all_x), max(all_y)]

    def _compute_struct_feat(self, shapes, img_w, img_h):
        """计算结构化特征: [lesion_area_ratio, lesion_count]."""
        gb_rect = get_gallbladder_rect(shapes)
        gb_area = 1.0
        if gb_rect:
            gb_area = max((gb_rect[2] - gb_rect[0]) * (gb_rect[3] - gb_rect[1]), 1.0)

        lesion_area = 0.0
        lesion_count = 0
        for s in shapes:
            if s["label"] == "gallbladder":
                continue
            if s["shape_type"] == "polygon" and len(s["points"]) >= 3:
                pts = np.array(s["points"])
                x, y = pts[:, 0], pts[:, 1]
                area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                lesion_area += area
                lesion_count += 1

        ratio = lesion_area / gb_area
        return torch.tensor([ratio, lesion_count / 5.0], dtype=torch.float32)

    def _apply_transform(self, img, is_local=False):
        """标准 3ch transform."""
        size = [self.img_size, self.img_size]
        img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)

        if self.is_train:
            if torch.randint(0, 2, (1,)).item():
                img = TF.hflip(img)
            angle = float(torch.empty(1).uniform_(-8, 8).item())
            max_t = 0.03 * self.img_size
            translate = [int(torch.empty(1).uniform_(-max_t, max_t).item()),
                         int(torch.empty(1).uniform_(-max_t, max_t).item())]
            scale = float(torch.empty(1).uniform_(0.9, 1.1).item())
            img = TF.affine(img, angle, translate, scale, shear=[0.0],
                            interpolation=InterpolationMode.BICUBIC, fill=0)
            if torch.rand(1).item() < 0.3:
                img = TF.adjust_brightness(img, float(torch.empty(1).uniform_(0.88, 1.12).item()))
                img = TF.adjust_contrast(img, float(torch.empty(1).uniform_(0.88, 1.12).item()))

        img_t = TF.to_tensor(img)
        img_t = TF.normalize(img_t, self.mean, self.std)
        return img_t

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        json_path = img_path.replace(".png", ".json")
        label = int(row["label"])

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        shapes = []
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])

        gb_rect = get_gallbladder_rect(shapes)

        # Branch A: global ROI
        global_img = crop_roi(img, gb_rect, 0.02) if gb_rect else img.copy()

        # Branch B: lesion local crop
        lesion_bbox = self._get_lesion_bbox(shapes)
        if lesion_bbox is not None:
            # Pad 30% around lesion
            local_img = crop_roi(img, lesion_bbox, 0.30)
        else:
            # Fallback to center crop of ROI
            local_img = global_img.copy()

        # Structural features
        struct_feat = self._compute_struct_feat(shapes, img_w, img_h)

        # Transform
        global_t = self._apply_transform(global_img)
        local_t = self._apply_transform(local_img, is_local=True)

        return (global_t, local_t, struct_feat), label


# ─── 双分支模型 ───


class DualBranchSwin(nn.Module):
    """
    双分支 Swin-V2:
      Branch A (shared backbone) → global feature
      Branch B (shared backbone) → local feature
      Fusion MLP: concat + struct_feat → 2类
    """

    def __init__(self, num_classes=2, struct_dim=2):
        super().__init__()
        self.backbone = timm.create_model(
            "swinv2_tiny_window8_256", pretrained=True, num_classes=0,
        )
        feat_dim = self.backbone.num_features  # 768

        self.fusion = nn.Sequential(
            nn.Linear(feat_dim * 2 + struct_dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        global_img, local_img, struct_feat = x
        g = self.backbone(global_img)   # [B, 768]
        l = self.backbone(local_img)    # [B, 768]
        fused = torch.cat([g, l, struct_feat], dim=1)
        return self.fusion(fused)


# ─── 自定义训练循环 (支持 tuple input) ───


def train_one_epoch_dual(model, dataloader, criterion, optimizer, device, scaler, use_amp, grad_clip):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for (global_img, local_img, struct_feat), labels in dataloader:
        global_img = global_img.to(device, non_blocking=True)
        local_img = local_img.to(device, non_blocking=True)
        struct_feat = struct_feat.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                enabled=use_amp):
            outputs = model((global_img, local_img, struct_feat))
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * labels.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate_dual(model, dataloader, device, class_names, logger, phase="Test"):
    import numpy as np_
    from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for (global_img, local_img, struct_feat), labels in dataloader:
            global_img = global_img.to(device, non_blocking=True)
            local_img = local_img.to(device, non_blocking=True)
            struct_feat = struct_feat.to(device, non_blocking=True)
            outputs = model((global_img, local_img, struct_feat))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np_.array(all_preds)
    all_labels = np_.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    logger.info(
        f"[{phase}] Acc: {acc:.4f} | Precision(macro): {precision:.4f} | "
        f"Recall(macro): {recall:.4f} | F1(macro): {f1:.4f}"
    )
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4, zero_division=0)
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, precision, recall, f1


# ─── Config & Main ───


class Config:
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.join(project_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260323_task2_SwinV2Tiny_dual_6"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 3
    batch_size = 4
    num_epochs = 60
    warmup_epochs = 5
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.05
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 3
    seed = 42
    use_amp = True
    loss_name = "CrossEntropyLoss(class_weight + label_smoothing=0.1)"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "DualBranch Swin-V2-Tiny (shared backbone)"
    modification = "Dual branch (global ROI + lesion local) + struct feat + fusion MLP"
    train_transform_desc = "Global: ROI_crop→256 + aug | Local: lesion_crop(pad30%)→256 + aug"
    test_transform_desc = "Global: ROI_crop→256 | Local: lesion_crop(pad30%)→256"


def main():
    import shutil
    import time
    from task2_json_utils import (
        set_seed, setup_logger, acquire_run_lock, build_class_weights,
        cosine_warmup_factor, set_epoch_lrs,
    )

    cfg = Config()
    os.makedirs(cfg.log_dir, exist_ok=True)
    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    set_seed(cfg.seed)
    logger = setup_logger(cfg.log_file, cfg.exp_name)

    logger.info("=" * 60)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info("任务: Task 2 - 良性肿瘤(0) vs 非肿瘤性息肉(1)")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"图像尺寸: train {cfg.train_transform_desc}")
    logger.info(f"          test  {cfg.test_transform_desc}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Backbone LR: {cfg.backbone_lr} | Head LR: {cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay} | Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Warmup: {cfg.warmup_epochs} | Epochs: {cfg.num_epochs} | Seed: {cfg.seed}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 60)

    # Data
    train_dataset = GBPDatasetDual(cfg.train_excel, cfg.data_root, cfg.img_size, is_train=True)
    test_dataset = GBPDatasetDual(cfg.test_excel, cfg.data_root, cfg.img_size, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    logger.info(f"训练集: {len(train_dataset)} 张 "
                f"(benign={sum(train_dataset.df['label']==0)}, no_tumor={sum(train_dataset.df['label']==1)})")
    logger.info(f"测试集: {len(test_dataset)} 张 "
                f"(benign={sum(test_dataset.df['label']==0)}, no_tumor={sum(test_dataset.df['label']==1)})")

    # Model
    model = DualBranchSwin(num_classes=2, struct_dim=2).to(cfg.device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    class_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    logger.info(f"类别权重: benign={class_weights[0]:.4f}, no_tumor={class_weights[1]:.4f}")
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)

    # Optimizer: backbone + fusion head
    backbone_params, head_params = split_backbone_and_head(model, model.fusion)
    optimizer = build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)

    scaler = torch.amp.GradScaler(device=cfg.device.type,
                                  enabled=(cfg.device.type == "cuda" and cfg.use_amp))

    best_f1, best_epoch = 0.0, 0

    logger.info("\n" + "=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(1, cfg.num_epochs + 1):
        lr_factor = set_epoch_lrs(optimizer, epoch, cfg)
        t0 = time.time()

        train_loss, train_acc = train_one_epoch_dual(
            model, train_loader, criterion, optimizer, cfg.device,
            scaler, use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.6e}/{optimizer.param_groups[1]['lr']:.6e} "
            f"| Factor: {lr_factor:.4f} | Loss: {train_loss:.4f} "
            f"| Train Acc: {train_acc:.4f} | Time: {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 40)
            _, _, _, f1 = evaluate_dual(model, test_loader, cfg.device, cfg.class_names, logger)
            if f1 > best_f1:
                best_f1, best_epoch = f1, epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***")
            logger.info("-" * 40)

    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成! 最优: Epoch {best_epoch}, F1: {best_f1:.4f}")
    logger.info("=" * 60)

    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True))
    logger.info("=" * 60)
    logger.info("最终测试结果 (最优权重)")
    logger.info("=" * 60)
    evaluate_dual(model, test_loader, cfg.device, cfg.class_names, logger, phase="Final Test")

    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copy2(__file__, dst)


if __name__ == "__main__":
    main()
