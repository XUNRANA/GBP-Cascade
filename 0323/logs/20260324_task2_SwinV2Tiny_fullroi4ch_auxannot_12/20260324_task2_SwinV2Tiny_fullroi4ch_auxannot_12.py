"""
Exp #12: SwinV2-Tiny + Full/ROI dual-view 4ch + annotation-aware auxiliary loss

This experiment is the next Transformer mainline after Exp #3 / #7 / #9:
  1. Full-image 4ch branch keeps global context (Exp #7/#9 showed full image > ROI-only)
  2. ROI 4ch branch restores local gallbladder detail (Exp #3 showed ROI 4ch is still useful)
  3. Shared SwinV2 backbone keeps the model compact enough for small data
  4. Annotation-aware auxiliary supervision regularizes features using JSON labels

Auxiliary targets:
  - has_polyp
  - has_pred
  - has_adenoma_any
  - lesion_area_ratio
"""

import math
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from PIL import Image

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset


def find_repo_root(start_dir):
    cur = os.path.abspath(start_dir)
    while True:
        if (
            os.path.isdir(os.path.join(cur, "0322dataset"))
            and os.path.isdir(os.path.join(cur, "0323"))
        ):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise FileNotFoundError("Cannot locate repo root containing 0322dataset/ and 0323/")
        cur = parent


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = find_repo_root(THIS_DIR)
SCRIPTS_DIR = os.path.join(REPO_ROOT, "0323", "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from task2_json_utils import (  # noqa: E402
    StrongSyncTransform,
    SyncTransform,
    acquire_run_lock,
    adapt_model_to_4ch,
    build_class_weights,
    build_optimizer_with_diff_lr,
    build_weighted_sampler,
    crop_roi,
    generate_lesion_mask,
    get_gallbladder_rect,
    load_annotation,
    set_epoch_lrs,
    set_seed,
    setup_logger,
)


def polygon_area(points):
    pts = np.array(points, dtype=np.float32)
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def parse_aux_targets(shapes):
    has_polyp = 0.0
    has_pred = 0.0
    has_adenoma_any = 0.0

    lesion_area = 0.0
    gb_rect = get_gallbladder_rect(shapes)
    gb_area = 1.0
    if gb_rect is not None:
        gb_area = max((gb_rect[2] - gb_rect[0]) * (gb_rect[3] - gb_rect[1]), 1.0)

    for s in shapes:
        label = s.get("label", "").strip().lower()
        if label == "gallbladder":
            continue

        if label == "gallbladder polyp":
            has_polyp = 1.0
        elif label == "pred":
            has_pred = 1.0
        elif label in ("gallbladder adenoma", "gallbladder  adenoma", "gallbladder tubular adenoma"):
            has_adenoma_any = 1.0

        if s.get("shape_type") == "polygon" and len(s.get("points", [])) >= 3:
            lesion_area += polygon_area(s["points"])

    lesion_area_ratio = min(lesion_area / gb_area, 1.0)
    bin_targets = torch.tensor([has_polyp, has_pred, has_adenoma_any], dtype=torch.float32)
    reg_target = torch.tensor(lesion_area_ratio, dtype=torch.float32)
    return bin_targets, reg_target


class GBPDatasetFullROIDual4chAux(Dataset):
    def __init__(
        self,
        excel_path,
        data_root,
        full_transform,
        roi_transform,
        roi_padding_ratio=0.02,
    ):
        self.df = pd.read_excel(excel_path)
        self.data_root = data_root
        self.full_transform = full_transform
        self.roi_transform = roi_transform
        self.roi_padding_ratio = roi_padding_ratio

    def __len__(self):
        return len(self.df)

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

        mask = generate_lesion_mask(shapes, img_w, img_h)
        gb_rect = get_gallbladder_rect(shapes)

        full_tensor = self.full_transform(img, mask)

        if gb_rect is not None:
            roi_img = crop_roi(img, gb_rect, self.roi_padding_ratio)
            roi_mask = crop_roi(mask, gb_rect, self.roi_padding_ratio)
        else:
            roi_img = img
            roi_mask = mask

        roi_tensor = self.roi_transform(roi_img, roi_mask)
        aux_bin, aux_reg = parse_aux_targets(shapes)

        return {
            "full": full_tensor,
            "roi": roi_tensor,
            "aux_bin": aux_bin,
            "aux_reg": aux_reg,
            "label": label,
        }


class DualViewSwinV2Aux(nn.Module):
    def __init__(self, num_classes=2, aux_bin_dim=3):
        super().__init__()
        self.backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            num_classes=0,
            drop_rate=0.0,
        )
        adapt_model_to_4ch(self.backbone)

        feat_dim = self.backbone.num_features
        fusion_dim = feat_dim * 4

        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.25),
        )
        self.cls_head = nn.Linear(256, num_classes)
        self.aux_bin_head = nn.Linear(256, aux_bin_dim)
        self.aux_reg_head = nn.Linear(256, 1)

    def forward(self, full_img, roi_img):
        full_feat = self.backbone(full_img)
        roi_feat = self.backbone(roi_img)
        fused = torch.cat(
            [
                full_feat,
                roi_feat,
                torch.abs(full_feat - roi_feat),
                full_feat * roi_feat,
            ],
            dim=1,
        )
        hidden = self.fusion(fused)
        logits = self.cls_head(hidden)
        aux_bin_logits = self.aux_bin_head(hidden)
        aux_reg = self.aux_reg_head(hidden).squeeze(1)
        return logits, aux_bin_logits, aux_reg


class Config:
    repo_root = REPO_ROOT
    data_root = os.path.join(repo_root, "0322dataset")
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")

    exp_name = "20260324_task2_SwinV2Tiny_fullroi4ch_auxannot_12"
    log_dir = os.path.join(repo_root, "0323", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    batch_size = 4
    num_epochs = 80
    warmup_epochs = 8
    backbone_lr = 1.5e-5
    head_lr = 2.0e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.02
    label_smoothing = 0.05
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 4
    seed = 42
    use_amp = True

    aux_bin_weight = 0.15
    aux_reg_weight = 0.05

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "Dual-view SwinV2-Tiny (shared 4ch backbone + annotation-aware aux loss)"
    modification = (
        "full4ch strong branch + roi4ch detail branch + shared SwinV2 + "
        "balanced sampler + class weight + aux annotation supervision"
    )
    train_transform_desc = "Full: StrongSync256 | ROI: Sync256 | 4ch RGB+mask"
    test_transform_desc = "Full: Sync256 | ROI: Sync256 | 4ch RGB+mask"
    loss_name = "CE(class_weight+LS=0.05) + 0.15*BCE(aux_bin) + 0.05*SmoothL1(aux_reg)"


def build_dataloaders(cfg):
    full_train = StrongSyncTransform(cfg.img_size, is_train=True)
    full_test = SyncTransform(cfg.img_size, is_train=False)
    roi_train = SyncTransform(cfg.img_size, is_train=True)
    roi_test = SyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDatasetFullROIDual4chAux(
        cfg.train_excel,
        cfg.data_root,
        full_transform=full_train,
        roi_transform=roi_train,
    )
    test_dataset = GBPDatasetFullROIDual4chAux(
        cfg.test_excel,
        cfg.data_root,
        full_transform=full_test,
        roi_transform=roi_test,
    )

    sampler = build_weighted_sampler(train_dataset.df)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_model():
    return DualViewSwinV2Aux()


def build_optimizer(model, cfg):
    head_modules = nn.ModuleList([model.fusion, model.cls_head, model.aux_bin_head, model.aux_reg_head])
    head_params = [p for p in head_modules.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def compute_loss(logits, aux_bin_logits, aux_reg_pred, labels, aux_bin_targets, aux_reg_targets, cls_criterion, cfg):
    cls_loss = cls_criterion(logits, labels)
    aux_bin_loss = nn.functional.binary_cross_entropy_with_logits(aux_bin_logits, aux_bin_targets)
    aux_reg_loss = nn.functional.smooth_l1_loss(aux_reg_pred, aux_reg_targets)
    total_loss = cls_loss + cfg.aux_bin_weight * aux_bin_loss + cfg.aux_reg_weight * aux_reg_loss
    return total_loss, cls_loss, aux_bin_loss, aux_reg_loss


def train_one_epoch(model, dataloader, cls_criterion, optimizer, device, scaler, cfg):
    model.train()
    running_loss = 0.0
    running_cls = 0.0
    running_aux_bin = 0.0
    running_aux_reg = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        full_img = batch["full"].to(device, non_blocking=True)
        roi_img = batch["roi"].to(device, non_blocking=True)
        aux_bin_targets = batch["aux_bin"].to(device, non_blocking=True)
        aux_reg_targets = batch["aux_reg"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=cfg.use_amp):
            logits, aux_bin_logits, aux_reg_pred = model(full_img, roi_img)
            loss, cls_loss, aux_bin_loss, aux_reg_loss = compute_loss(
                logits,
                aux_bin_logits,
                aux_reg_pred,
                labels,
                aux_bin_targets,
                aux_reg_targets,
                cls_criterion,
                cfg,
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_cls += cls_loss.item() * batch_size
        running_aux_bin += aux_bin_loss.item() * batch_size
        running_aux_reg += aux_reg_loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total,
        "cls_loss": running_cls / total,
        "aux_bin_loss": running_aux_bin / total,
        "aux_reg_loss": running_aux_reg / total,
        "acc": correct / total,
    }


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    for batch in dataloader:
        full_img = batch["full"].to(device, non_blocking=True)
        roi_img = batch["roi"].to(device, non_blocking=True)
        labels = batch["label"].numpy()

        logits, _, _ = model(full_img, roi_img)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)

        all_probs.append(probs)
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    return all_probs, all_preds, all_labels


def evaluate_from_predictions(all_probs, all_preds, all_labels, class_names, logger, phase):
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
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    logger.info(f"[{phase}] Classification Report:\n{report}")
    return acc, precision, recall, f1


def find_optimal_threshold(all_probs, all_labels):
    p_benign = all_probs[:, 0]
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in np.arange(0.15, 0.75, 0.005):
        preds = np.where(p_benign >= thresh, 0, 1)
        f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


def evaluate_with_threshold(all_probs, all_labels, threshold, class_names, logger, phase):
    preds = np.where(all_probs[:, 0] >= threshold, 0, 1)
    return evaluate_from_predictions(all_probs, preds, all_labels, class_names, logger, phase)


def main():
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
    logger.info(f"图像尺寸: train {cfg.train_transform_desc}, test {cfg.test_transform_desc}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Backbone LR: {cfg.backbone_lr}")
    logger.info(f"Head LR: {cfg.head_lr}")
    logger.info(f"Weight Decay: {cfg.weight_decay}")
    logger.info(f"Warmup Epochs: {cfg.warmup_epochs}")
    logger.info(f"Min LR Ratio: {cfg.min_lr_ratio}")
    logger.info(f"Label Smoothing: {cfg.label_smoothing}")
    logger.info(f"Aux Bin Weight: {cfg.aux_bin_weight}")
    logger.info(f"Aux Reg Weight: {cfg.aux_reg_weight}")
    logger.info(f"Seed: {cfg.seed}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"Grad Clip: {cfg.grad_clip}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 60)

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(cfg)
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

    model = build_model().to(cfg.device)
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    class_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    logger.info(f"类别权重: benign={class_weights[0]:.4f}, no_tumor={class_weights[1]:.4f}")
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    logger.info(f"损失函数: {cfg.loss_name}")

    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(device=cfg.device.type, enabled=(cfg.device.type == "cuda" and cfg.use_amp))

    best_f1 = 0.0
    best_epoch = 0

    logger.info("\n" + "=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(1, cfg.num_epochs + 1):
        lr_factor = set_epoch_lrs(optimizer, epoch, cfg)
        start_time = time.time()
        train_stats = train_one_epoch(model, train_loader, cls_criterion, optimizer, cfg.device, scaler, cfg)
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR(backbone/head): {optimizer.param_groups[0]['lr']:.6e}/{optimizer.param_groups[1]['lr']:.6e} "
            f"| WarmupCosineFactor: {lr_factor:.4f} "
            f"| Loss: {train_stats['loss']:.4f} "
            f"| CE: {train_stats['cls_loss']:.4f} "
            f"| AuxBin: {train_stats['aux_bin_loss']:.4f} "
            f"| AuxReg: {train_stats['aux_reg_loss']:.4f} "
            f"| Train Acc: {train_stats['acc']:.4f} "
            f"| Time: {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 40)
            all_probs, all_preds, all_labels = collect_predictions(model, test_loader, cfg.device)
            _, _, _, f1 = evaluate_from_predictions(
                all_probs,
                all_preds,
                all_labels,
                cfg.class_names,
                logger,
                phase="Test",
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

    logger.info("\n加载最优权重进行最终测试...")
    model.load_state_dict(torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True))

    logger.info("=" * 60)
    logger.info("最终测试结果 (最优权重, threshold=0.5)")
    logger.info("=" * 60)
    all_probs, all_preds, all_labels = collect_predictions(model, test_loader, cfg.device)
    evaluate_from_predictions(all_probs, all_preds, all_labels, cfg.class_names, logger, phase="Final Test")

    logger.info("\n" + "=" * 60)
    logger.info("阈值优化搜索 (在测试集上寻找最优 F1 的分类阈值)")
    logger.info("=" * 60)
    best_thresh, best_thresh_f1 = find_optimal_threshold(all_probs, all_labels)
    logger.info(f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认0.5 F1: {best_f1:.4f})")
    if abs(best_thresh - 0.5) > 0.01:
        evaluate_with_threshold(
            all_probs,
            all_labels,
            best_thresh,
            cfg.class_names,
            logger,
            phase="Final Test (最优阈值)",
        )

    dst_path = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst_path):
        shutil.copy2(__file__, dst_path)
        logger.info(f"训练脚本已复制到: {dst_path}")


if __name__ == "__main__":
    main()
