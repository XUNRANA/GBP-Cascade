"""
GBP-Cascade Task 2 自主实验训练脚本.
用法: python train_task2.py

这是 AI Agent 唯一可以修改的文件.
所有实验通过修改下方超参数区域 + 模型构建 + 训练循环来进行.
"""

import math
import os
import time

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from prepare_task2 import (
    # Constants
    DATA_ROOT, TRAIN_EXCEL, TEST_EXCEL, CLASS_NAMES, NUM_CLASSES,
    TIME_BUDGET, CHECKPOINT_PATH,
    # Transforms
    SyncTransform, StrongSyncTransform,
    build_roi_train_transform, build_roi_test_transform,
    # Datasets
    GBPDatasetROI, GBPDatasetROI4ch, GBPDatasetFull4ch,
    # Model utilities
    adapt_model_to_4ch, split_backbone_and_head,
    # Training utilities
    set_seed, build_class_weights, build_weighted_sampler,
    FocalLoss, cosine_warmup_factor, mixup_cutmix_data,
    # Evaluation
    evaluate_model,
)

# ═══════════════════════════════════════════════════════════
#  Hyperparameters (EDIT FREELY)
# ═══════════════════════════════════════════════════════════

# Model
MODEL_NAME = "swinv2_tiny_window8_256"   # any timm model with pretrained=True
DROP_RATE = 0.3                           # classifier head dropout

# Input
INPUT_MODE = "full_4ch"   # "roi_3ch" | "roi_4ch" | "full_4ch"
IMG_SIZE = 256            # input image size (square)
AUG_MODE = "strong"       # "weak" | "strong"

# Training
BATCH_SIZE = 8
NUM_EPOCHS = 100
EVAL_INTERVAL = 5         # evaluate every N epochs
WARMUP_EPOCHS = 8
SEED = 42

# Optimizer
BACKBONE_LR = 2e-5
HEAD_LR = 2e-4
WEIGHT_DECAY = 5e-2
MIN_LR_RATIO = 0.01
GRAD_CLIP = 1.0

# Loss
LABEL_SMOOTHING = 0.1
USE_FOCAL_LOSS = False
FOCAL_GAMMA = 2.0

# Regularization
USE_MIXUP = True
MIXUP_ALPHA = 0.4
CUTMIX_ALPHA = 1.0
USE_BALANCED_SAMPLER = True

# ═══════════════════════════════════════════════════════════
#  Setup
# ═══════════════════════════════════════════════════════════

set_seed(SEED)
device = torch.device("cuda")
torch.set_float32_matmul_precision("high")
t_start = time.time()

print("=== GBP-Cascade Task 2 AutoResearch ===")
print(f"Model: {MODEL_NAME} | Input: {INPUT_MODE} | Size: {IMG_SIZE} | Aug: {AUG_MODE}")
print(f"BS: {BATCH_SIZE} | Epochs: {NUM_EPOCHS} | BackboneLR: {BACKBONE_LR} | HeadLR: {HEAD_LR}")
print(f"Mixup: {USE_MIXUP} | Focal: {USE_FOCAL_LOSS} | BalancedSampler: {USE_BALANCED_SAMPLER}")
print(f"DropRate: {DROP_RATE} | WD: {WEIGHT_DECAY} | LabelSmooth: {LABEL_SMOOTHING}")

# ═══════════════════════════════════════════════════════════
#  Model
# ═══════════════════════════════════════════════════════════

model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES, drop_rate=DROP_RATE)
if INPUT_MODE in ("roi_4ch", "full_4ch"):
    adapt_model_to_4ch(model)
model = model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params / 1e6:.1f}M")

# ═══════════════════════════════════════════════════════════
#  Data
# ═══════════════════════════════════════════════════════════

if INPUT_MODE == "roi_3ch":
    train_dataset = GBPDatasetROI(TRAIN_EXCEL, DATA_ROOT,
                                  transform=build_roi_train_transform(IMG_SIZE))
    test_dataset = GBPDatasetROI(TEST_EXCEL, DATA_ROOT,
                                 transform=build_roi_test_transform(IMG_SIZE))
elif INPUT_MODE == "roi_4ch":
    train_sync = StrongSyncTransform(IMG_SIZE) if AUG_MODE == "strong" else SyncTransform(IMG_SIZE)
    test_sync = SyncTransform(IMG_SIZE, is_train=False)
    train_dataset = GBPDatasetROI4ch(TRAIN_EXCEL, DATA_ROOT, sync_transform=train_sync)
    test_dataset = GBPDatasetROI4ch(TEST_EXCEL, DATA_ROOT, sync_transform=test_sync)
elif INPUT_MODE == "full_4ch":
    train_sync = StrongSyncTransform(IMG_SIZE) if AUG_MODE == "strong" else SyncTransform(IMG_SIZE)
    test_sync = SyncTransform(IMG_SIZE, is_train=False)
    train_dataset = GBPDatasetFull4ch(TRAIN_EXCEL, DATA_ROOT, sync_transform=train_sync)
    test_dataset = GBPDatasetFull4ch(TEST_EXCEL, DATA_ROOT, sync_transform=test_sync)
else:
    raise ValueError(f"Unknown INPUT_MODE: {INPUT_MODE}")

if USE_BALANCED_SAMPLER:
    sampler = build_weighted_sampler(train_dataset.df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=4, pin_memory=True)

print(f"Train: {len(train_dataset)} images | Test: {len(test_dataset)} images")

# ═══════════════════════════════════════════════════════════
#  Loss & Optimizer
# ═══════════════════════════════════════════════════════════

class_weights = build_class_weights(train_dataset.df, device)
print(f"Class weights: benign={class_weights[0]:.3f}, no_tumor={class_weights[1]:.3f}")

if USE_FOCAL_LOSS:
    criterion = FocalLoss(alpha=class_weights, gamma=FOCAL_GAMMA)
else:
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

# Differential LR: backbone (slow) + head (fast)
head_module = model.get_classifier()
backbone_params, head_params = split_backbone_and_head(model, head_module)
optimizer = AdamW([
    {"params": backbone_params, "lr": BACKBONE_LR, "base_lr": BACKBONE_LR},
    {"params": head_params, "lr": HEAD_LR, "base_lr": HEAD_LR},
], weight_decay=WEIGHT_DECAY)

scaler = torch.amp.GradScaler(device="cuda", enabled=True)

# ═══════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════

best_f1 = 0.0
best_epoch = 0
epochs_completed = 0

for epoch in range(1, NUM_EPOCHS + 1):
    # Time budget check
    elapsed = time.time() - t_start
    if elapsed > TIME_BUDGET:
        print(f"Time budget ({TIME_BUDGET}s) exceeded at epoch {epoch}, stopping.")
        break

    # LR schedule: linear warmup + cosine decay
    factor = cosine_warmup_factor(epoch, NUM_EPOCHS, WARMUP_EPOCHS, MIN_LR_RATIO)
    for pg in optimizer.param_groups:
        pg["lr"] = pg["base_lr"] * factor

    # Train one epoch
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t_epoch = time.time()

    for images, labels in train_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup / CutMix
        soft_labels = None
        if USE_MIXUP:
            images, labels, soft_labels = mixup_cutmix_data(
                images, labels, NUM_CLASSES, MIXUP_ALPHA, CUTMIX_ALPHA)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(images)
            if soft_labels is not None:
                # Soft cross-entropy WITH class weights (critical bug fix)
                log_probs = F.log_softmax(outputs, dim=1)
                if hasattr(criterion, "weight") and criterion.weight is not None:
                    w = criterion.weight.unsqueeze(0)  # [1, C]
                    loss = -(soft_labels * log_probs * w).sum(dim=1).mean()
                else:
                    loss = -(soft_labels * log_probs).sum(dim=1).mean()
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if GRAD_CLIP:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    epoch_time = time.time() - t_epoch
    epochs_completed = epoch

    # NaN check
    if math.isnan(epoch_loss):
        print("FAIL: NaN loss detected")
        exit(1)

    # Periodic evaluation
    eval_str = ""
    if epoch % EVAL_INTERVAL == 0 or epoch == NUM_EPOCHS:
        metrics = evaluate_model(model, test_loader, device)
        f1_val = metrics["f1_at_threshold"]
        if f1_val > best_f1:
            best_f1 = f1_val
            best_epoch = epoch
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            eval_str = f" | F1@thresh: {f1_val:.4f} (NEW BEST*)"
        else:
            eval_str = f" | F1@thresh: {f1_val:.4f} (best: {best_f1:.4f}@ep{best_epoch})"

    print(f"Epoch {epoch:03d}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} "
          f"| LR: {factor:.4f} | {epoch_time:.1f}s{eval_str}")

# ═══════════════════════════════════════════════════════════
#  Final Evaluation
# ═══════════════════════════════════════════════════════════

training_time = time.time() - t_start
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

# Load best checkpoint
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    print(f"\nLoaded best model from epoch {best_epoch}")

# Final metrics
final = evaluate_model(model, test_loader, device)

# Output summary (grep-friendly format)
print("---")
print(f"f1_at_threshold:       {final['f1_at_threshold']:.6f}")
print(f"best_threshold:        {final['best_threshold']:.3f}")
print(f"f1_macro:              {final['f1_macro']:.6f}")
print(f"accuracy:              {final['accuracy']:.6f}")
print(f"f1_benign:             {final['f1_benign']:.6f}")
print(f"f1_no_tumor:           {final['f1_no_tumor']:.6f}")
print(f"recall_benign:         {final['recall_benign']:.6f}")
print(f"recall_no_tumor:       {final['recall_no_tumor']:.6f}")
print(f"precision_benign:      {final['precision_benign']:.6f}")
print(f"precision_no_tumor:    {final['precision_no_tumor']:.6f}")
print(f"f1_benign_thresh:      {final['f1_benign_at_thresh']:.6f}")
print(f"recall_benign_thresh:  {final['recall_benign_at_thresh']:.6f}")
print(f"best_epoch:            {best_epoch}")
print(f"epochs_completed:      {epochs_completed}")
print(f"training_seconds:      {training_time:.1f}")
print(f"peak_vram_mb:          {peak_vram_mb:.1f}")
print(f"num_params_M:          {num_params / 1e6:.1f}")
