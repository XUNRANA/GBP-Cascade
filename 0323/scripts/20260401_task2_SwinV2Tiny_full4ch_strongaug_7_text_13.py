"""
Exp #7: Swin-V2-Tiny + 全图4ch(不做ROI裁剪) + 强增强 + Mixup/CutMix ★★★
改进点:
  1. 去掉 ROI 裁剪 → 保留全局上下文（之前ROI裁剪反而掉点）
  2. 强增强: RandomResizedCrop + 旋转±20° + shear + 更强ColorJitter + GaussianBlur + RandomErasing + GaussianNoise
  3. Mixup + CutMix: batch 级别数据混合，最有效的小数据集正则化
  4. 训练更长: 100 epochs (之前60 epoch训练acc已93%严重过拟合)
  5. 更大 Dropout: head 增加 dropout
"""

import os

import timm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from test_yqh import (
    GBPDatasetFull4chWithMeta,
    StrongSyncTransform,
    SyncTransform,
    adapt_model_to_4ch,
    build_optimizer_with_diff_lr,
    run_experiment,
    split_backbone_and_head,
)


class Config:
    project_root = "/data1/ouyangxinglong/GBP-Cascade"
    data_root = os.path.join(project_root, "0322dataset")
    print('project_root', project_root, 'data_root', data_root)
    train_excel = os.path.join(data_root, "task_2_train.xlsx")
    test_excel = os.path.join(data_root, "task_2_test.xlsx")
    clinical_excel = os.path.join(project_root, "胆囊超声组学_分析.xlsx")
    json_feature_root = os.path.join(project_root, "json_text")

    exp_name = "20260323_task2_SwinV2Tiny_full4ch_strongaug_7"
    log_dir = os.path.join(project_root, "logs", exp_name)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    best_weight_path = os.path.join(log_dir, f"{exp_name}_best.pth")

    img_size = 256
    in_channels = 4
    batch_size = 8
    num_epochs = 100
    warmup_epochs = 8
    backbone_lr = 2e-5
    head_lr = 2e-4
    weight_decay = 5e-2
    min_lr_ratio = 0.01
    label_smoothing = 0.1
    grad_clip = 1.0
    num_workers = 4
    eval_interval = 5
    seed = 42
    use_amp = True
    use_mixup = True
    loss_name = "CrossEntropyLoss(class_weight + LS=0.1) + Mixup/CutMix"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = ["benign", "no_tumor"]
    model_name = "Swin-V2-Tiny (ImageNet, 4ch) + metadata fusion"
    modification = (
        "全图(不裁ROI)+病灶mask 4ch + metadata(age/sex/size_mm/size_bin/flow_bin/morph_bin) "
        "+ flatten后融合 + 强增强 + Mixup/CutMix + 100ep"
    )
    train_transform_desc = "Full_img → StrongSync(RRC+Rot20+Shear+ColorJitter+Blur+Erase+Noise) + mask"
    test_transform_desc = "Full_img → Resize256 + mask"
    meta_hidden_dim = 64
    meta_branch_dropout = 0.2
    fusion_dropout = 0.3
    meta_dim = 6


def _pick_existing_file(*candidates):
    for p in candidates:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Cannot find file from candidates: {candidates}")


def _pick_existing_dir(*candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    raise FileNotFoundError(f"Cannot find directory from candidates: {candidates}")


class SwinV2TinyMetaFusion(nn.Module):
    def __init__(
        self,
        meta_dim,
        num_classes=2,
        meta_hidden_dim=64,
        meta_dropout=0.2,
        fusion_dropout=0.3,
    ):
        super().__init__()
        self.meta_dim = int(meta_dim)

        self.backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            num_classes=0,   # return pre-logits feature
            drop_rate=0.0,
        )
        adapt_model_to_4ch(self.backbone)

        self.meta_encoder = nn.Sequential(
            nn.Linear(self.meta_dim, meta_hidden_dim),
            nn.LayerNorm(meta_hidden_dim),
            nn.GELU(),
            nn.Dropout(meta_dropout),
            nn.Linear(meta_hidden_dim, meta_hidden_dim),
            nn.GELU(),
            nn.Dropout(meta_dropout),
        )

        fusion_in_dim = int(self.backbone.num_features) + meta_hidden_dim
        fusion_hidden_dim = max(128, int(self.backbone.num_features) // 2)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_in_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(fusion_dropout),
            nn.Linear(fusion_hidden_dim, num_classes),
        )

    def forward(self, x, metadata=None):
        img_feat = self.backbone.forward_features(x)
        if img_feat.ndim != 2:
            if hasattr(self.backbone, "forward_head"):
                img_feat = self.backbone.forward_head(img_feat, pre_logits=True)
            else:
                img_feat = torch.flatten(img_feat, 1)

        if metadata is None:
            metadata = torch.zeros(
                x.size(0), self.meta_dim, device=x.device, dtype=img_feat.dtype
            )
        meta_feat = self.meta_encoder(metadata.float())

        fused = torch.cat([img_feat, meta_feat], dim=1)
        return self.fusion_head(fused)


def build_model(cfg):
    return SwinV2TinyMetaFusion(
        meta_dim=getattr(cfg, "meta_dim", 6),
        num_classes=len(cfg.class_names),
        meta_hidden_dim=cfg.meta_hidden_dim,
        meta_dropout=cfg.meta_branch_dropout,
        fusion_dropout=cfg.fusion_dropout,
    )


def build_dataloaders(cfg):
    train_excel = _pick_existing_file(
        cfg.train_excel,
        os.path.join(cfg.project_root, "task_2_train.xlsx"),
    )
    test_excel = _pick_existing_file(
        cfg.test_excel,
        os.path.join(cfg.project_root, "task_2_test.xlsx"),
    )
    clinical_excel = _pick_existing_file(
        cfg.clinical_excel,
        os.path.join(cfg.data_root, "胆囊超声组学_分析.xlsx"),
    )
    json_feature_root = _pick_existing_dir(
        cfg.json_feature_root,
        os.path.join(cfg.data_root, "json_text"),
        os.path.join(cfg.project_root, "json_text"),
    )
    data_root = _pick_existing_dir(
        cfg.data_root,
        cfg.project_root,
    )

    cfg.train_excel = train_excel
    cfg.test_excel = test_excel
    cfg.clinical_excel = clinical_excel
    cfg.json_feature_root = json_feature_root
    cfg.data_root = data_root

    train_sync = StrongSyncTransform(cfg.img_size, is_train=True)
    test_sync = SyncTransform(cfg.img_size, is_train=False)

    train_dataset = GBPDatasetFull4chWithMeta(
        cfg.train_excel,
        cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=train_sync,
        meta_stats=None,
    )
    test_dataset = GBPDatasetFull4chWithMeta(
        cfg.test_excel,
        cfg.data_root,
        clinical_excel_path=cfg.clinical_excel,
        json_feature_root=cfg.json_feature_root,
        sync_transform=test_sync,
        meta_stats=train_dataset.meta_stats,
    )
    cfg.meta_dim = train_dataset.meta_dim

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_dataset, test_dataset, train_loader, test_loader


def build_optimizer(model, cfg):
    if hasattr(model, "backbone") and hasattr(model, "fusion_head"):
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = [
            p for p in list(model.meta_encoder.parameters()) + list(model.fusion_head.parameters())
            if p.requires_grad
        ]
        return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)

    head = model.head.fc if hasattr(model.head, "fc") else model.head
    backbone_params, head_params = split_backbone_and_head(model, head)
    return build_optimizer_with_diff_lr(AdamW, backbone_params, head_params, cfg)


def main():
    run_experiment(
        cfg=Config(),
        build_model_fn=build_model,
        build_dataloaders_fn=build_dataloaders,
        build_optimizer_fn=build_optimizer,
        script_path=__file__,
    )


if __name__ == "__main__":
    main()
