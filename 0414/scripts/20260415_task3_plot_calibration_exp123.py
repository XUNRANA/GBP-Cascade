"""
Plot calibration curves for 0414 task-3 experiments 1/2/3.

Focus:
  - Compare softmax P(no_tumor) calibration on the same 0414 test set
  - Reuse existing best checkpoints from Exp-1 / Exp-2 / Exp-3
  - Save reliability diagram + per-experiment CSV summaries

Notes:
  - We use softmax P(no_tumor) for all three experiments so the comparison is
    directly apples-to-apples.
  - Exp-2 / Exp-3 also have ordinal heads, but ordinal score is not a calibrated
    class probability, so it is not used for the reliability diagram here.
"""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-gbp-cascade")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import brier_score_loss


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent.parent
OUTPUT_DIR = ROOT_DIR / "0414" / "calibration" / "exp123_no_tumor"
POSITIVE_LABEL = 2
POSITIVE_NAME = "no_tumor"
N_BINS = 10


EXPERIMENTS = [
    {
        "key": "exp1",
        "display_name": "Exp-1 Smoke",
        "script_path": ROOT_DIR / "0414" / "scripts" / "20260414_task3_SwinV2Tiny_segcls_smoke_1.py",
    },
    {
        "key": "exp2",
        "display_name": "Exp-2 CostSensitive+Ordinal",
        "script_path": ROOT_DIR / "0414" / "scripts" / "20260414_task3_SwinV2Tiny_segcls_2.py",
    },
    {
        "key": "exp3",
        "display_name": "Exp-3 SegGuided+Meta",
        "script_path": ROOT_DIR / "0414" / "scripts" / "20260414_task3_SwinV2Tiny_segcls_3.py",
    },
]


def load_module_from_path(path: Path):
    module_name = f"calibration_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_state_dict(weight_path: Path, device: torch.device):
    try:
        return torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def build_experiment(exp_def, device: torch.device):
    module = load_module_from_path(exp_def["script_path"])
    cfg = module.Config()
    cfg.pretrained = False
    cfg.num_workers = 0
    cfg.device = device

    if exp_def["key"] == "exp1":
        model = module.build_model(cfg).to(device)
    elif exp_def["key"] == "exp2":
        model = module.SwinV2SegCls4chOrdinalModel(
            num_seg_classes=cfg.num_seg_classes,
            num_cls_classes=cfg.num_cls_classes,
            cls_dropout=cfg.cls_dropout,
            pretrained=False,
        ).to(device)
    elif exp_def["key"] == "exp3":
        model = module.SwinV2SegGuidedOrdinalMetaModel(
            num_seg_classes=cfg.num_seg_classes,
            num_cls_classes=cfg.num_cls_classes,
            meta_dim=cfg.meta_dim,
            meta_hidden=cfg.meta_hidden,
            meta_dropout=cfg.meta_dropout,
            cls_dropout=cfg.cls_dropout,
            pretrained=False,
        ).to(device)
    else:
        raise ValueError(f"Unknown experiment key: {exp_def['key']}")

    weight_path = Path(cfg.best_weight_path)
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {weight_path}")
    model.load_state_dict(load_state_dict(weight_path, device))
    model.eval()

    loaders = module.build_dataloaders(cfg)
    test_loader = loaders[-1]
    return module, cfg, model, test_loader


@torch.no_grad()
def collect_softmax_predictions(exp_key: str, model, dataloader, device: torch.device):
    all_probs = []
    all_labels = []

    for batch in dataloader:
        if exp_key in {"exp1", "exp2"}:
            imgs, _masks, labels, _has_masks = batch
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
        elif exp_key == "exp3":
            imgs, _masks, labels, _has_masks, metas = batch
            imgs = imgs.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)
            outputs = model(imgs, metadata=metas)
        else:
            raise ValueError(f"Unknown experiment key: {exp_key}")

        cls_logits = outputs[1]
        probs = F.softmax(cls_logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def compute_calibration_bins(prob_pos: np.ndarray, y_true: np.ndarray, n_bins: int = 10):
    y_bin = (y_true == POSITIVE_LABEL).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    ece = 0.0

    for idx in range(n_bins):
        left = float(bin_edges[idx])
        right = float(bin_edges[idx + 1])
        if idx == n_bins - 1:
            mask = (prob_pos >= left) & (prob_pos <= right)
        else:
            mask = (prob_pos >= left) & (prob_pos < right)

        count = int(mask.sum())
        if count == 0:
            mean_conf = np.nan
            frac_pos = np.nan
            gap = np.nan
        else:
            mean_conf = float(prob_pos[mask].mean())
            frac_pos = float(y_bin[mask].mean())
            gap = abs(mean_conf - frac_pos)
            ece += (count / len(prob_pos)) * gap

        rows.append(
            {
                "bin_idx": idx,
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "mean_confidence": mean_conf,
                "fraction_positive": frac_pos,
                "abs_gap": gap,
            }
        )

    brier = float(brier_score_loss(y_bin, prob_pos))
    return pd.DataFrame(rows), float(ece), brier


def compute_summary(display_name: str, probs: np.ndarray, labels: np.ndarray):
    prob_no_tumor = probs[:, POSITIVE_LABEL]
    pred_labels = probs.argmax(axis=1)
    pred_no_tumor_mask = pred_labels == POSITIVE_LABEL
    high90_mask = prob_no_tumor >= 0.90
    high80_mask = prob_no_tumor >= 0.80

    def safe_mean(mask):
        if int(mask.sum()) == 0:
            return np.nan
        return float((labels[mask] == POSITIVE_LABEL).mean())

    return {
        "experiment": display_name,
        "n_samples": int(len(labels)),
        "true_no_tumor_rate": float((labels == POSITIVE_LABEL).mean()),
        "pred_no_tumor_count": int(pred_no_tumor_mask.sum()),
        "pred_no_tumor_precision": safe_mean(pred_no_tumor_mask),
        "p_no_tumor_ge_0.80_count": int(high80_mask.sum()),
        "p_no_tumor_ge_0.80_true_rate": safe_mean(high80_mask),
        "p_no_tumor_ge_0.90_count": int(high90_mask.sum()),
        "p_no_tumor_ge_0.90_true_rate": safe_mean(high90_mask),
    }


def plot_reliability(ax, bin_df: pd.DataFrame, display_name: str, summary: dict, ece: float, brier: float):
    valid = bin_df["count"] > 0
    x = bin_df.loc[valid, "mean_confidence"].to_numpy()
    y = bin_df.loc[valid, "fraction_positive"].to_numpy()
    counts = bin_df.loc[valid, "count"].to_numpy()

    ax.plot([0, 1], [0, 1], "--", color="0.75", linewidth=1.5)
    ax.plot(x, y, color="#c34a36", linewidth=2.0, marker="o", markersize=5)
    ax.scatter(x, y, s=30 + 4 * counts, color="#c34a36", alpha=0.35)

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title(display_name, fontsize=11)
    ax.set_xlabel(f"Predicted P({POSITIVE_NAME})")
    ax.set_ylabel(f"Observed freq({POSITIVE_NAME})")
    ax.grid(True, alpha=0.25)

    high90_rate = summary["p_no_tumor_ge_0.90_true_rate"]
    high90_text = "NA" if np.isnan(high90_rate) else f"{high90_rate:.3f}"
    info = (
        f"ECE={ece:.3f}\n"
        f"Brier={brier:.3f}\n"
        f">=0.90: n={summary['p_no_tumor_ge_0.90_count']}, true={high90_text}"
    )
    ax.text(
        0.03,
        0.97,
        info,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.95},
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160, constrained_layout=True)
    all_summary_rows = []

    for ax, exp_def in zip(axes, EXPERIMENTS):
        print(f"[Run] {exp_def['display_name']}")
        _module, cfg, model, test_loader = build_experiment(exp_def, device)
        probs, labels = collect_softmax_predictions(exp_def["key"], model, test_loader, device)

        prob_no_tumor = probs[:, POSITIVE_LABEL]
        bin_df, ece, brier = compute_calibration_bins(prob_no_tumor, labels, n_bins=N_BINS)
        summary = compute_summary(exp_def["display_name"], probs, labels)
        summary["ece"] = ece
        summary["brier_no_tumor"] = brier
        all_summary_rows.append(summary)

        pred_df = pd.DataFrame(
            {
                "true_label": labels,
                "pred_label": probs.argmax(axis=1),
                "prob_malignant": probs[:, 0],
                "prob_benign": probs[:, 1],
                "prob_no_tumor": probs[:, 2],
            }
        )
        pred_df.to_csv(OUTPUT_DIR / f"{cfg.exp_name}_predictions.csv", index=False)
        bin_df.to_csv(OUTPUT_DIR / f"{cfg.exp_name}_calibration_bins.csv", index=False)

        plot_reliability(ax, bin_df, exp_def["display_name"], summary, ece, brier)

    figure.suptitle("0414 Task-3 Reliability Diagram for P(no_tumor)", fontsize=14)
    figure_path = OUTPUT_DIR / "exp123_no_tumor_reliability.png"
    figure.savefig(figure_path, bbox_inches="tight")
    plt.close(figure)

    summary_df = pd.DataFrame(all_summary_rows)
    summary_df.to_csv(OUTPUT_DIR / "exp123_no_tumor_summary.csv", index=False)

    print(f"[Saved] figure: {figure_path}")
    print(f"[Saved] summary: {OUTPUT_DIR / 'exp123_no_tumor_summary.csv'}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
