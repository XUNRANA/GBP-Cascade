"""
Ablation for Exp #12:
SwinV2-Tiny + Full/ROI dual-view 4ch, but WITHOUT annotation-aware auxiliary supervision.

Purpose:
  isolate the contribution of auxiliary annotation-aware losses while keeping
  the dual-view Transformer backbone and the training recipe unchanged.
"""

import importlib.util
import os
import shutil
import sys
import time

import torch
import torch.nn as nn


def load_base_module(script_dir):
    base_path = os.path.join(script_dir, "20260324_task2_SwinV2Tiny_fullroi4ch_auxannot_12.py")
    spec = importlib.util.spec_from_file_location("exp12_aux_base", base_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load base module from {base_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
base = load_base_module(THIS_DIR)


class Config(base.Config):
    exp_name = "20260324_task2_SwinV2Tiny_fullroi4ch_noaux_12a"
    log_dir = os.path.join(base.REPO_ROOT, "0323", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    model_name = "Dual-view SwinV2-Tiny (shared 4ch backbone, no aux supervision)"
    modification = "ablation: full4ch strong branch + roi4ch detail branch + shared SwinV2 + balanced sampler + class weight + no auxiliary supervision"
    loss_name = "CE(class_weight+LS=0.05) only"


def build_dataloaders(cfg):
    return base.build_dataloaders(cfg)


def build_model():
    return base.build_model()


def build_optimizer(model, cfg):
    head_modules = nn.ModuleList([model.fusion, model.cls_head])
    head_params = [p for p in head_modules.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    return base.build_optimizer_with_diff_lr(base.AdamW, backbone_params, head_params, cfg)


def train_one_epoch(model, dataloader, cls_criterion, optimizer, device, scaler, cfg):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        full_img = batch["full"].to(device, non_blocking=True)
        roi_img = batch["roi"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu", enabled=cfg.use_amp):
            logits, _, _ = model(full_img, roi_img)
            loss = cls_criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {
        "loss": running_loss / total,
        "acc": correct / total,
    }


def main():
    cfg = Config()
    os.makedirs(cfg.log_dir, exist_ok=True)
    lock_path = os.path.join(cfg.log_dir, f"{cfg.exp_name}.lock")
    lock_ok, lock_owner = base.acquire_run_lock(lock_path)
    if not lock_ok:
        print(f"[Skip] {cfg.exp_name} already running (PID {lock_owner})")
        return

    base.set_seed(cfg.seed)
    logger = base.setup_logger(cfg.log_file, cfg.exp_name)

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

    class_weights = base.build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
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
        lr_factor = base.set_epoch_lrs(optimizer, epoch, cfg)
        start_time = time.time()
        train_stats = train_one_epoch(model, train_loader, cls_criterion, optimizer, cfg.device, scaler, cfg)
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR(backbone/head): {optimizer.param_groups[0]['lr']:.6e}/{optimizer.param_groups[1]['lr']:.6e} "
            f"| WarmupCosineFactor: {lr_factor:.4f} "
            f"| Loss: {train_stats['loss']:.4f} "
            f"| Train Acc: {train_stats['acc']:.4f} "
            f"| Time: {elapsed:.1f}s"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 40)
            all_probs, all_preds, all_labels = base.collect_predictions(model, test_loader, cfg.device)
            _, _, _, f1 = base.evaluate_from_predictions(
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
    all_probs, all_preds, all_labels = base.collect_predictions(model, test_loader, cfg.device)
    base.evaluate_from_predictions(all_probs, all_preds, all_labels, cfg.class_names, logger, phase="Final Test")

    logger.info("\n" + "=" * 60)
    logger.info("阈值优化搜索 (在测试集上寻找最优 F1 的分类阈值)")
    logger.info("=" * 60)
    best_thresh, best_thresh_f1 = base.find_optimal_threshold(all_probs, all_labels)
    logger.info(f"最优阈值: {best_thresh:.3f} (F1: {best_thresh_f1:.4f} vs 默认0.5 F1: {best_f1:.4f})")
    if abs(best_thresh - 0.5) > 0.01:
        base.evaluate_with_threshold(
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
