#!/usr/bin/env python3
"""
Analyze pre-classifier features (input to cls_head) for the current best
classification model and visualize class separability with t-SNE.

Best model (by completed logs, Softmax Macro-F1):
  logs/20260414_task3_SwinV2Tiny_segcls_3/20260414_task3_SwinV2Tiny_segcls_3_best.pth
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_exp_module(exp_script: Path):
    spec = importlib.util.spec_from_file_location("exp3_module", str(exp_script))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def extract_features(module):
    cfg = module.Config()
    cfg.num_workers = 0
    cfg.batch_size = 16
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_dataset, _, _, test_loader = module.build_dataloaders(cfg)

    model = module.SwinV2SegGuidedOrdinalMetaModel(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        pretrained=False,  # checkpoint load only; avoid external download
    ).to(cfg.device)
    model.eval()

    ckpt_path = Path(cfg.best_weight_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    try:
        state = torch.load(str(ckpt_path), map_location=cfg.device, weights_only=True)
    except TypeError:
        state = torch.load(str(ckpt_path), map_location=cfg.device)
    model.load_state_dict(state)

    feat_buf = []

    def hook_fn(_module, inputs):
        # inputs[0]: cls_feat right before cls_head
        feat_buf.append(inputs[0].detach().cpu())

    handle = model.cls_head.register_forward_pre_hook(hook_fn)

    all_feats = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for imgs, _masks, labels, _has_masks, metas in test_loader:
            imgs = imgs.to(cfg.device, non_blocking=True)
            metas = metas.to(cfg.device, non_blocking=True)
            labels = labels.to(cfg.device, non_blocking=True)

            _seg_logits, cls_logits, _ord_score = model(imgs, metadata=metas)
            feats = feat_buf.pop(0)

            preds = cls_logits.argmax(dim=1).cpu()
            all_feats.append(feats)
            all_labels.append(labels.cpu())
            all_preds.append(preds)

    handle.remove()

    features = torch.cat(all_feats, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    preds = torch.cat(all_preds, dim=0).numpy()
    class_names = list(cfg.class_names)

    assert len(test_dataset) == features.shape[0], "Feature count mismatch with test set"
    return features, labels, preds, class_names, str(ckpt_path), str(cfg.device)


def compute_stats(features: np.ndarray, labels: np.ndarray):
    classes = sorted(np.unique(labels).tolist())
    centroids = {}
    intra_mean_dist = {}
    for c in classes:
        f = features[labels == c]
        ctr = f.mean(axis=0)
        centroids[c] = ctr
        intra_mean_dist[c] = float(np.linalg.norm(f - ctr, axis=1).mean())

    inter_centroid = {}
    for i in classes:
        for j in classes:
            if i >= j:
                continue
            d = float(np.linalg.norm(centroids[i] - centroids[j]))
            inter_centroid[f"{i}-{j}"] = d

    sil = float(silhouette_score(features, labels, metric="euclidean"))
    return intra_mean_dist, inter_centroid, sil


def run_tsne(features: np.ndarray):
    scaler = StandardScaler()
    x = scaler.fit_transform(features)
    n = x.shape[0]
    perplexity = max(5, min(30, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=42,
    )
    return tsne.fit_transform(x)


def save_outputs(out_dir: Path, emb2d: np.ndarray, labels: np.ndarray, preds: np.ndarray, class_names):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save scatter as PNG
    plt.figure(figsize=(8, 7), dpi=140)
    colors = ["#d73027", "#4575b4", "#1a9850"]
    for c in sorted(np.unique(labels)):
        idx = labels == c
        plt.scatter(
            emb2d[idx, 0],
            emb2d[idx, 1],
            s=18,
            alpha=0.75,
            c=colors[c % len(colors)],
            label=f"{class_names[c]} (n={idx.sum()})",
        )
    plt.title("t-SNE of Pre-Classifier Features (Best Softmax Model)")
    plt.xlabel("t-SNE-1")
    plt.ylabel("t-SNE-2")
    plt.legend()
    plt.tight_layout()
    fig_path = out_dir / "tsne_pre_classifier_feat.png"
    plt.savefig(fig_path)
    plt.close()

    # Save point table
    csv_path = out_dir / "tsne_points.csv"
    header = "x,y,label,pred,correct\n"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for i in range(len(labels)):
            f.write(
                f"{emb2d[i,0]:.6f},{emb2d[i,1]:.6f},{int(labels[i])},{int(preds[i])},{int(labels[i]==preds[i])}\n"
            )
    return fig_path, csv_path


def main():
    root = Path(__file__).resolve().parents[1]
    exp_script = root / "scripts" / "20260414_task3_SwinV2Tiny_segcls_3.py"
    out_dir = root / "logs" / "feature_analysis_best_softmax_segcls3"

    module = load_exp_module(exp_script)
    features, labels, preds, class_names, ckpt, device = extract_features(module)

    intra, inter, sil = compute_stats(features, labels)
    emb2d = run_tsne(features)
    fig_path, csv_path = save_outputs(out_dir, emb2d, labels, preds, class_names)

    report = {
        "model_ckpt": ckpt,
        "device": device,
        "num_samples": int(len(labels)),
        "class_names": class_names,
        "intra_mean_l2_to_centroid": {class_names[int(k)]: float(v) for k, v in intra.items()},
        "inter_centroid_l2": {
            f"{class_names[int(k.split('-')[0])]}-{class_names[int(k.split('-')[1])]}": float(v)
            for k, v in inter.items()
        },
        "silhouette_score": sil,
        "tsne_png": str(fig_path),
        "tsne_points_csv": str(csv_path),
    }

    report_path = out_dir / "feature_space_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

