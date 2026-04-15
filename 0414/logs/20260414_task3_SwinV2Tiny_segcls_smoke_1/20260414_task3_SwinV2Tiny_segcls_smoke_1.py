"""
0414 三分类联合分类+分割 smoke baseline.

目标:
  - 先基于 0414dataset 跑通一个最小可运行的三分类联合脚本
  - 同时做 3 类分类 + lesion segmentation
  - 使用 4ch 输入: RGB + gallbladder ROI mask
  - 恶性样本没有 lesion polygon, 因此 segmentation loss 自动置零

说明:
  - 这是第一版粗测脚本, 优先保证能跑和能看趋势
  - 暂时不引入 text / metadata / ordinal score / cost matrix
  - 使用官方 train/test Excel, 并在 test 上做粗评估
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
SCRIPTS_0402 = os.path.join(ROOT_DIR, "0402", "scripts")
if SCRIPTS_0402 not in sys.path:
    sys.path.insert(0, SCRIPTS_0402)

from seg_cls_utils_v2 import (  # noqa: E402
    load_annotation,
    generate_lesion_mask,
    SegClsLoss,
    set_seed,
    setup_logger,
    acquire_run_lock,
    build_class_weights,
    set_epoch_lrs,
    build_optimizer_with_diff_lr,
    train_one_epoch_v2,
    evaluate_v2,
    seg_cls_4ch_collate_fn,
    SwinV2SegCls4chModel,
)


def generate_gallbladder_mask(shapes, width, height):
    """Generate gallbladder ROI mask from LabelMe shapes."""
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        if shape.get("label") != "gallbladder":
            continue
        points = shape.get("points", [])
        shape_type = shape.get("shape_type")
        if shape_type == "rectangle" and len(points) >= 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            draw.rectangle([x1, y1, x2, y2], fill=255)
        elif shape_type == "polygon" and len(points) >= 3:
            draw.polygon([(p[0], p[1]) for p in points], fill=255)
    return mask


class SegCls0414SyncTransform:
    """
    Synchronized transform for:
      - RGB image
      - gallbladder ROI mask (input channel)
      - lesion mask (seg target)
    """

    def __init__(self, img_size, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __call__(self, img, gb_mask, lesion_mask):
        size = [self.img_size, self.img_size]

        if self.is_train:
            i, j, h, w = T.RandomResizedCrop.get_params(
                img, scale=(0.7, 1.0), ratio=(0.85, 1.15)
            )
            img = TF.resized_crop(img, i, j, h, w, size, InterpolationMode.BICUBIC)
            gb_mask = TF.resized_crop(gb_mask, i, j, h, w, size, InterpolationMode.NEAREST)
            lesion_mask = TF.resized_crop(
                lesion_mask, i, j, h, w, size, InterpolationMode.NEAREST
            )

            if np.random.rand() < 0.5:
                img = TF.hflip(img)
                gb_mask = TF.hflip(gb_mask)
                lesion_mask = TF.hflip(lesion_mask)
            if np.random.rand() < 0.3:
                img = TF.vflip(img)
                gb_mask = TF.vflip(gb_mask)
                lesion_mask = TF.vflip(lesion_mask)
            if np.random.rand() < 0.5:
                angle = float(np.random.uniform(-20, 20))
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BICUBIC, fill=0)
                gb_mask = TF.rotate(
                    gb_mask, angle, interpolation=InterpolationMode.NEAREST, fill=0
                )
                lesion_mask = TF.rotate(
                    lesion_mask, angle, interpolation=InterpolationMode.NEAREST, fill=0
                )
            if np.random.rand() < 0.5:
                angle = float(np.random.uniform(-5, 5))
                max_t = 0.06 * self.img_size
                translate = [
                    int(np.random.uniform(-max_t, max_t)),
                    int(np.random.uniform(-max_t, max_t)),
                ]
                scale = float(np.random.uniform(0.9, 1.1))
                shear = [float(np.random.uniform(-5, 5))]
                img = TF.affine(
                    img,
                    angle,
                    translate,
                    scale,
                    shear,
                    interpolation=InterpolationMode.BICUBIC,
                    fill=0,
                )
                gb_mask = TF.affine(
                    gb_mask,
                    angle,
                    translate,
                    scale,
                    shear,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )
                lesion_mask = TF.affine(
                    lesion_mask,
                    angle,
                    translate,
                    scale,
                    shear,
                    interpolation=InterpolationMode.NEAREST,
                    fill=0,
                )
            if np.random.rand() < 0.6:
                img = TF.adjust_brightness(img, float(np.random.uniform(0.7, 1.3)))
                img = TF.adjust_contrast(img, float(np.random.uniform(0.7, 1.3)))
                img = TF.adjust_saturation(img, float(np.random.uniform(0.8, 1.2)))
            if np.random.rand() < 0.2:
                img = TF.gaussian_blur(img, kernel_size=3)
        else:
            img = TF.resize(img, size, interpolation=InterpolationMode.BICUBIC)
            gb_mask = TF.resize(gb_mask, size, interpolation=InterpolationMode.NEAREST)
            lesion_mask = TF.resize(lesion_mask, size, interpolation=InterpolationMode.NEAREST)

        img_t = TF.to_tensor(img)
        gb_t = TF.to_tensor(gb_mask)
        lesion_t = TF.to_tensor(lesion_mask)

        img_t = TF.normalize(img_t, self.mean, self.std)

        if self.is_train and np.random.rand() < 0.2:
            img_t = T.RandomErasing(p=1.0, scale=(0.02, 0.15), ratio=(0.3, 3.3))(img_t)
        if self.is_train and np.random.rand() < 0.3:
            img_t = img_t + torch.randn_like(img_t) * 0.03

        input_4ch = torch.cat([img_t, gb_t], dim=0)
        seg_target = (lesion_t.squeeze(0) > 0.5).long()
        return input_4ch, seg_target


class GBPDataset0414SegCls4ch(Dataset):
    """0414 dataset: 4ch input (RGB + gallbladder ROI) + lesion seg target + 3-class label."""

    def __init__(self, excel_path, data_root, sync_transform=None):
        self.df = pd.read_excel(excel_path).copy()
        self.data_root = data_root
        self.sync_transform = sync_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, row["image_path"])
        json_path = img_path.replace(".png", ".json")
        label = int(row["label"])  # 0=malignant, 1=benign, 2=no_tumor

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        shapes = []
        has_lesion_mask = False
        if os.path.exists(json_path):
            ann = load_annotation(json_path)
            shapes = ann.get("shapes", [])
            has_lesion_mask = any(
                s.get("label") != "gallbladder"
                and s.get("shape_type") == "polygon"
                and len(s.get("points", [])) >= 3
                for s in shapes
            )

        gb_mask = generate_gallbladder_mask(shapes, img_w, img_h)
        lesion_mask = generate_lesion_mask(shapes, img_w, img_h)

        if self.sync_transform:
            input_4ch, seg_target = self.sync_transform(img, gb_mask, lesion_mask)
        else:
            img_t = TF.to_tensor(img)
            gb_t = TF.to_tensor(gb_mask)
            lesion_t = TF.to_tensor(lesion_mask)
            input_4ch = torch.cat([img_t, gb_t], dim=0)
            seg_target = (lesion_t.squeeze(0) > 0.5).long()

        return input_4ch, seg_target, label, has_lesion_mask


class Config:
    project_root = ROOT_DIR
    data_root = os.path.join(project_root, "0414dataset")
    train_excel = os.path.join(data_root, "task_3class_train.xlsx")
    test_excel = os.path.join(data_root, "task_3class_test.xlsx")

    exp_name = "20260414_task3_SwinV2Tiny_segcls_smoke_1"
    log_dir = os.path.join(project_root, "0414", "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    num_seg_classes = 2
    num_cls_classes = 3
    cls_dropout = 0.4
    pretrained = True

    batch_size = 8
    num_epochs = 30
    warmup_epochs = 5
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 2
    seed = 42
    use_amp = True

    lambda_cls = 2.0
    seg_bg_weight = 1.0
    seg_lesion_weight = 5.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["malignant", "benign", "no_tumor"]
    model_name = "SwinV2-Tiny@256 + 4ch(RGB+GallbladderROI) + LesionSeg + 3ClassCls"
    modification = (
        "0414 smoke baseline: 4ch输入改为RGB+gallbladder ROI, "
        "seg监督只用lesion polygon, malignant自动mask-off, "
        "分类为malignant/benign/no_tumor三分类"
    )


def build_model(cfg):
    try:
        return SwinV2SegCls4chModel(
            num_seg_classes=cfg.num_seg_classes,
            num_cls_classes=cfg.num_cls_classes,
            cls_dropout=cfg.cls_dropout,
            pretrained=cfg.pretrained,
        )
    except Exception as exc:
        print(f"[Warn] pretrained backbone load failed, fallback to pretrained=False: {exc}")
        return SwinV2SegCls4chModel(
            num_seg_classes=cfg.num_seg_classes,
            num_cls_classes=cfg.num_cls_classes,
            cls_dropout=cfg.cls_dropout,
            pretrained=False,
        )


def build_dataloaders(cfg):
    train_tf = SegCls0414SyncTransform(cfg.img_size, is_train=True)
    test_tf = SegCls0414SyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDataset0414SegCls4ch(cfg.train_excel, cfg.data_root, train_tf)
    test_dataset = GBPDataset0414SegCls4ch(cfg.test_excel, cfg.data_root, test_tf)

    labels = train_dataset.df["label"].to_numpy()
    class_counts = np.bincount(labels, minlength=cfg.num_cls_classes)
    sample_weights = 1.0 / class_counts[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=seg_cls_4ch_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=seg_cls_4ch_collate_fn,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    backbone_params = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = [
        p for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.")
    ]
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def log_split_counts(logger, prefix, dataset, class_names):
    counts = dataset.df["label"].value_counts().sort_index().to_dict()
    msg = ", ".join(f"{class_names[i]}={counts.get(i, 0)}" for i in range(len(class_names)))
    logger.info(f"{prefix}: {len(dataset)} 张 ({msg})")


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

    logger.info("=" * 70)
    logger.info(f"实验名称: {cfg.exp_name}")
    logger.info(f"模型: {cfg.model_name}")
    logger.info(f"修改: {cfg.modification}")
    logger.info(f"数据集: {cfg.data_root}")
    logger.info(f"图像尺寸: {cfg.img_size}")
    logger.info(f"分类类别: {cfg.class_names}")
    logger.info(f"Batch Size: {cfg.batch_size}")
    logger.info(f"Backbone LR: {cfg.backbone_lr}")
    logger.info(f"Head LR: {cfg.head_lr}")
    logger.info(f"Epochs: {cfg.num_epochs}")
    logger.info(f"设备: {cfg.device}")
    logger.info("=" * 70)

    train_dataset, test_dataset, train_loader, test_loader = build_dataloaders(cfg)
    log_split_counts(logger, "训练集", train_dataset, cfg.class_names)
    log_split_counts(logger, "测试集", test_dataset, cfg.class_names)

    model = build_model(cfg).to(cfg.device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数量: {n_params:,}")
    logger.info(f"可训练参数量: {n_trainable:,}")

    cls_weights = build_class_weights(train_dataset.df, cfg.class_names, cfg.device)
    logger.info(
        "分类类别权重: "
        + ", ".join(f"{name}={cls_weights[i]:.4f}" for i, name in enumerate(cfg.class_names))
    )

    seg_ce_weight = torch.tensor(
        [cfg.seg_bg_weight, cfg.seg_lesion_weight],
        dtype=torch.float32,
        device=cfg.device,
    )
    criterion = SegClsLoss(
        cls_weights=cls_weights,
        lambda_cls=cfg.lambda_cls,
        label_smoothing=cfg.label_smoothing,
        seg_ce_weight=seg_ce_weight,
    )
    optimizer = build_optimizer(model, cfg)
    scaler = torch.amp.GradScaler(
        device=cfg.device.type,
        enabled=(cfg.device.type == "cuda" and cfg.use_amp),
    )

    best_f1 = 0.0
    best_epoch = 0

    logger.info("=" * 70)
    logger.info("开始训练")
    logger.info("=" * 70)

    for epoch in range(1, cfg.num_epochs + 1):
        set_epoch_lrs(optimizer, epoch, cfg)
        train_metrics = train_one_epoch_v2(
            model,
            train_loader,
            criterion,
            optimizer,
            cfg.device,
            scaler,
            use_amp=(cfg.device.type == "cuda" and cfg.use_amp),
            grad_clip=cfg.grad_clip,
            num_seg_classes=cfg.num_seg_classes,
        )

        logger.info(
            f"Epoch [{epoch}/{cfg.num_epochs}] "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}/{optimizer.param_groups[1]['lr']:.2e} "
            f"| Loss: {train_metrics['loss']:.4f} "
            f"(seg={train_metrics['seg_loss']:.4f}, cls={train_metrics['cls_loss']:.4f}) "
            f"| Cls Acc: {train_metrics['cls_acc']:.4f} "
            f"| Seg Dice: {train_metrics['seg_dice']:.4f}"
        )

        if epoch % cfg.eval_interval == 0 or epoch == cfg.num_epochs:
            logger.info("-" * 50)
            acc, prec, rec, f1, seg_iou, seg_dice = evaluate_v2(
                model,
                test_loader,
                cfg.device,
                cfg.class_names,
                logger,
                phase="Test",
                num_seg_classes=cfg.num_seg_classes,
            )
            if f1 > best_f1:
                best_f1 = f1
                best_epoch = epoch
                torch.save(model.state_dict(), cfg.best_weight_path)
                logger.info(
                    f"*** 保存最优模型 (F1: {best_f1:.4f}, Epoch: {best_epoch}) ***"
                )
            logger.info("-" * 50)

    logger.info("=" * 70)
    logger.info(f"训练完成! 最优 Epoch: {best_epoch}, Best F1: {best_f1:.4f}")
    logger.info("=" * 70)

    if os.path.exists(cfg.best_weight_path):
        try:
            state_dict = torch.load(cfg.best_weight_path, map_location=cfg.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(cfg.best_weight_path, map_location=cfg.device)
        model.load_state_dict(state_dict)

        logger.info("最终测试结果 (最优权重)")
        evaluate_v2(
            model,
            test_loader,
            cfg.device,
            cfg.class_names,
            logger,
            phase="Final Test",
            num_seg_classes=cfg.num_seg_classes,
        )

    dst = os.path.join(cfg.log_dir, os.path.basename(__file__))
    if os.path.abspath(__file__) != os.path.abspath(dst):
        Path(dst).write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")
        logger.info(f"训练脚本已复制到: {dst}")


if __name__ == "__main__":
    main()
