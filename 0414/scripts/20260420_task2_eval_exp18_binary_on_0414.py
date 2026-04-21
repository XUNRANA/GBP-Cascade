"""
Evaluate Exp#18 binary checkpoint on 0414 dataset (benign vs no_tumor only).

This script:
1) Loads 0414 test excel (prefers `test.xlsx`, falls back to `task_3class_test.xlsx`)
2) Filters to benign/no_tumor and remaps labels to binary:
      benign -> 0, no_tumor -> 1
3) Loads Exp#18 checkpoint and runs inference only (no training)
4) Exports detailed classification reports and analysis artifacts.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader


ROOT = Path("/data1/ouyangxinglong/GBP-Cascade")
sys.path.insert(0, str(ROOT / "0408" / "scripts"))

from seg_cls_utils_v5 import (  # noqa: E402
    EXT_CLINICAL_FEATURE_NAMES,
    GBPDatasetSegCls4chWithTextMeta,
    SegCls4chSyncTransform,
    SwinV2SegGuidedCls4chTrimodal,
    analyze_high_confidence_positive,
    compute_binary_reliability_stats,
    find_constrained_threshold_text,
    load_text_bert_dict,
    predict_probs_text,
    save_reliability_diagram,
    save_reliability_stats_csv,
    seg_cls_text_collate_fn,
    set_seed,
    setup_logger,
)


class Config:
    project_root = ROOT

    # Model checkpoint from user
    weight_path = project_root / "0408" / "logs" / "20260408_task2_SwinV2Tiny_segcls_18" / "20260408_task2_SwinV2Tiny_segcls_18_best.pth"

    # 0414 target dataset
    target_data_root = project_root / "0414dataset"
    target_test_excel_candidates = [
        target_data_root / "test.xlsx",
        target_data_root / "task_3class_test.xlsx",
    ]

    # Reference training distribution (for meta normalization + tokenizer reuse)
    ref_data_root = project_root / "0322dataset"
    ref_train_excel = ref_data_root / "task_2_train.xlsx"

    clinical_excel = project_root / "胆囊超声组学_分析.xlsx"
    json_feature_root = project_root / "json_text"

    exp_name = "20260420_eval_exp18_binary_on_0414_bn_vs_no_tumor"
    log_dir = project_root / "0414" / "logs" / exp_name
    log_file = log_dir / f"{exp_name}.log"

    # Data/model settings (match Exp#18)
    img_size = 256
    batch_size = 8
    num_workers = 4
    max_text_len = 128
    class_names = ["benign", "no_tumor"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Arch params for Exp#18
    num_seg_classes = 2
    num_cls_classes = 2
    cls_dropout = 0.4
    meta_dim = len(EXT_CLINICAL_FEATURE_NAMES)
    meta_hidden = 96
    meta_dropout = 0.2
    text_proj_dim = 128
    text_dropout = 0.3
    ca_hidden = 128
    ca_heads = 4
    ca_dropout = 0.1
    fusion_dim = 256

    # Analysis settings
    calibration_bins = 10
    high_confidence_no_tumor_prob = 0.90
    benign_miss_rate_targets = [0.10, 0.05]
    seed = 42


def load_state_dict_compat(weight_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    try:
        raw = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        raw = torch.load(weight_path, map_location=device)

    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        state = raw["state_dict"]
    else:
        state = raw

    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {type(state)}")

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    return state


def resolve_target_test_excel(cfg: Config, logger):
    for path in cfg.target_test_excel_candidates:
        if path.exists():
            logger.info(f"Using target test excel: {path}")
            return path
    candidates = ", ".join(str(p) for p in cfg.target_test_excel_candidates)
    raise FileNotFoundError(f"No target test excel found. Tried: {candidates}")


def build_binary_eval_excel(src_excel: Path, out_excel: Path, logger) -> Dict[str, int]:
    df = pd.read_excel(src_excel).copy()
    if "label" not in df.columns:
        raise KeyError(f"`label` column not found in {src_excel}")

    original_counts = df["label"].value_counts(dropna=False).sort_index().to_dict()
    logger.info(f"Original label counts: {original_counts}")

    df = df[df["label"].isin([1, 2])].copy()
    if df.empty:
        raise ValueError("No benign/no_tumor samples after filtering labels in [1, 2].")

    df["label_3class_original"] = df["label"].astype(int)
    df["label"] = df["label"].map({1: 0, 2: 1}).astype(int)
    df["label_name"] = df["label"].map({0: "benign", 1: "no_tumor"})

    out_excel.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_excel, index=False)
    logger.info(f"Saved binary eval excel: {out_excel}")

    binary_counts = df["label"].value_counts(dropna=False).sort_index().to_dict()
    logger.info(f"Binary label counts: {binary_counts}")
    return {
        "n_total_original": int(sum(int(v) for v in original_counts.values())),
        "n_binary_eval": int(len(df)),
        "n_benign": int(binary_counts.get(0, 0)),
        "n_no_tumor": int(binary_counts.get(1, 0)),
    }


def build_model(cfg: Config):
    # Use pretrained=False for Swin since checkpoint weights are loaded immediately.
    # BERT weights still come from the checkpoint state_dict.
    return SwinV2SegGuidedCls4chTrimodal(
        num_seg_classes=cfg.num_seg_classes,
        num_cls_classes=cfg.num_cls_classes,
        meta_dim=cfg.meta_dim,
        meta_hidden=cfg.meta_hidden,
        meta_dropout=cfg.meta_dropout,
        cls_dropout=cfg.cls_dropout,
        text_proj_dim=cfg.text_proj_dim,
        text_dropout=cfg.text_dropout,
        ca_hidden=cfg.ca_hidden,
        ca_heads=cfg.ca_heads,
        ca_dropout=cfg.ca_dropout,
        fusion_dim=cfg.fusion_dim,
        bert_name="bert-base-chinese",
        pretrained=False,
    )


def build_eval_loader(cfg: Config, eval_excel: Path):
    test_sync = SegCls4chSyncTransform(cfg.img_size, is_train=False)
    text_dict = load_text_bert_dict(str(cfg.json_feature_root))

    # Reference train set is only used to get normalization stats and tokenizer.
    ref_train_dataset = GBPDatasetSegCls4chWithTextMeta(
        str(cfg.ref_train_excel),
        str(cfg.ref_data_root),
        clinical_excel_path=str(cfg.clinical_excel),
        json_feature_root=str(cfg.json_feature_root),
        sync_transform=test_sync,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        max_text_len=cfg.max_text_len,
    )

    eval_dataset = GBPDatasetSegCls4chWithTextMeta(
        str(eval_excel),
        str(cfg.target_data_root),
        clinical_excel_path=str(cfg.clinical_excel),
        json_feature_root=str(cfg.json_feature_root),
        sync_transform=test_sync,
        meta_stats=ref_train_dataset.meta_stats,
        meta_feature_names=list(EXT_CLINICAL_FEATURE_NAMES),
        text_dict=text_dict,
        tokenizer=ref_train_dataset.tokenizer,
        max_text_len=cfg.max_text_len,
    )

    loader = DataLoader(
        eval_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=seg_cls_text_collate_fn,
    )
    return eval_dataset, loader


def metrics_from_threshold(labels: np.ndarray, probs_benign: np.ndarray, threshold: float) -> Dict[str, object]:
    preds = np.where(probs_benign >= threshold, 0, 1)
    report_dict = classification_report(
        labels,
        preds,
        target_names=["benign", "no_tumor"],
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        labels,
        preds,
        target_names=["benign", "no_tumor"],
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    return {
        "threshold": float(threshold),
        "preds": preds,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision_macro": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(labels, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "classification_report_dict": report_dict,
        "classification_report_text": report_text,
        "confusion_matrix": cm.tolist(),
    }


def find_best_f1_threshold(labels: np.ndarray, probs_benign: np.ndarray) -> Tuple[float, float]:
    best_t = 0.5
    best_f1 = -1.0
    for t in np.arange(0.15, 0.75, 0.005):
        preds = np.where(probs_benign >= t, 0, 1)
        cur = f1_score(labels, preds, average="macro", zero_division=0)
        if cur > best_f1:
            best_f1 = float(cur)
            best_t = float(t)
    return best_t, best_f1


def save_markdown_report(
    out_md: Path,
    dataset_info: Dict[str, int],
    auc_info: Dict[str, float],
    default_res: Dict[str, object],
    best_f1_res: Dict[str, object],
    policies: Dict[str, Dict[str, object]],
    rel_info: Dict[str, float],
    files: Dict[str, Path],
):
    def line_for_result(name: str, res: Dict[str, object]) -> str:
        return (
            f"| {name} | {res['threshold']:.3f} | {res['accuracy']:.4f} | "
            f"{res['precision_macro']:.4f} | {res['recall_macro']:.4f} | {res['f1_macro']:.4f} |"
        )

    lines = []
    lines.append(f"# {Config.exp_name}")
    lines.append("")
    lines.append("## Dataset")
    lines.append(f"- Binary eval samples: {dataset_info['n_binary_eval']}")
    lines.append(f"- Benign: {dataset_info['n_benign']}")
    lines.append(f"- No_tumor: {dataset_info['n_no_tumor']}")
    lines.append("")
    lines.append("## Probability Quality")
    lines.append(f"- ROC-AUC (no_tumor positive): {auc_info['roc_auc_no_tumor']:.4f}")
    lines.append(f"- PR-AUC (no_tumor positive): {auc_info['pr_auc_no_tumor']:.4f}")
    lines.append(f"- ECE (P(no_tumor)): {rel_info['ece']:.4f}")
    lines.append(f"- Brier (P(no_tumor)): {rel_info['brier']:.4f}")
    lines.append("")
    lines.append("## Threshold Comparison")
    lines.append("| Setting | Threshold | Acc | Precision(macro) | Recall(macro) | F1(macro) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(line_for_result("Default", default_res))
    lines.append(line_for_result("Best-F1", best_f1_res))
    for key, item in policies.items():
        lines.append(line_for_result(key, item["metrics"]))
    lines.append("")
    lines.append("## Default Threshold Classification Report")
    lines.append("```text")
    lines.append(default_res["classification_report_text"].rstrip())
    lines.append("```")
    lines.append("")
    lines.append("## Best-F1 Threshold Classification Report")
    lines.append("```text")
    lines.append(best_f1_res["classification_report_text"].rstrip())
    lines.append("```")
    lines.append("")
    for key, item in policies.items():
        lines.append(f"## {key} Classification Report")
        lines.append("```text")
        lines.append(item["metrics"]["classification_report_text"].rstrip())
        lines.append("```")
        lines.append("")
        p = item["policy"]
        lines.append(
            f"- Policy details: benign_miss_rate={p['benign_miss_rate']:.2%}, "
            f"benign_recall={p['benign_recall']:.2%}, "
            f"no_tumor_recall={p['no_tumor_recall']:.2%}, "
            f"no_tumor_precision={p['no_tumor_precision']:.2%}, "
            f"constraint_satisfied={p['constraint_satisfied']}"
        )
        lines.append("")

    lines.append("## Artifacts")
    lines.append(f"- Binary eval excel: `{files['binary_excel']}`")
    lines.append(f"- Per-case probabilities: `{files['probs_csv']}`")
    lines.append(f"- Confusion matrix (default): `{files['cm_csv']}`")
    lines.append(f"- Reliability bins csv: `{files['reliability_csv']}`")
    lines.append(f"- Reliability diagram png: `{files['reliability_png']}`")
    lines.append(f"- Metrics json: `{files['metrics_json']}`")
    lines.append("")

    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    cfg = Config()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(cfg.log_file), cfg.exp_name)
    set_seed(cfg.seed)

    logger.info("=" * 80)
    logger.info("Binary inference-only evaluation on 0414 dataset")
    logger.info(f"Checkpoint: {cfg.weight_path}")
    logger.info(f"Device: {cfg.device}")
    logger.info("=" * 80)

    if not cfg.weight_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {cfg.weight_path}")

    target_excel = resolve_target_test_excel(cfg, logger)
    binary_excel = cfg.log_dir / "0414_binary_eval_test.xlsx"
    dataset_info = build_binary_eval_excel(target_excel, binary_excel, logger)

    eval_dataset, eval_loader = build_eval_loader(cfg, binary_excel)
    logger.info(
        f"Eval dataset ready: n={len(eval_dataset)} "
        f"(benign={(eval_dataset.df['label'] == 0).sum()}, "
        f"no_tumor={(eval_dataset.df['label'] == 1).sum()})"
    )

    model = build_model(cfg).to(cfg.device)
    state = load_state_dict_compat(cfg.weight_path, cfg.device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning(f"Missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")
    model.eval()

    probs_benign, probs_no_tumor, labels = predict_probs_text(model, eval_loader, cfg.device)
    if len(labels) != len(eval_dataset.df):
        raise RuntimeError(
            f"Prediction length mismatch: preds={len(labels)} vs df={len(eval_dataset.df)}"
        )

    logger.info(f"Inference done for {len(labels)} samples.")

    # Core metrics at default threshold
    default_res = metrics_from_threshold(labels, probs_benign, threshold=0.5)
    logger.info(
        "[Default@0.5] Acc=%.4f P=%.4f R=%.4f F1=%.4f",
        default_res["accuracy"],
        default_res["precision_macro"],
        default_res["recall_macro"],
        default_res["f1_macro"],
    )
    logger.info("Default report:\n%s", default_res["classification_report_text"])

    # Best-F1 threshold
    best_f1_t, best_f1 = find_best_f1_threshold(labels, probs_benign)
    best_f1_res = metrics_from_threshold(labels, probs_benign, threshold=best_f1_t)
    logger.info(
        "[Best-F1] threshold=%.3f F1=%.4f | Acc=%.4f P=%.4f R=%.4f",
        best_f1_t,
        best_f1,
        best_f1_res["accuracy"],
        best_f1_res["precision_macro"],
        best_f1_res["recall_macro"],
    )
    logger.info("Best-F1 report:\n%s", best_f1_res["classification_report_text"])

    # Policy thresholds under benign miss-rate constraints
    policies = {}
    for miss_rate in cfg.benign_miss_rate_targets:
        policy = find_constrained_threshold_text(
            probs_benign,
            labels,
            max_benign_miss_rate=float(miss_rate),
        )
        key = f"Policy(miss<={int(round(miss_rate * 100)):02d}%)"
        metrics = metrics_from_threshold(labels, probs_benign, policy["threshold"])
        policies[key] = {"policy": policy, "metrics": metrics}
        logger.info(
            "[%s] threshold=%.3f benign_miss=%.2f%% benign_recall=%.2f%% "
            "no_tumor_recall=%.2f%% no_tumor_precision=%.2f%% F1=%.4f",
            key,
            policy["threshold"],
            100.0 * policy["benign_miss_rate"],
            100.0 * policy["benign_recall"],
            100.0 * policy["no_tumor_recall"],
            100.0 * policy["no_tumor_precision"],
            metrics["f1_macro"],
        )
        logger.info("%s report:\n%s", key, metrics["classification_report_text"])

    # AUC / PR-AUC
    y_no_tumor = (labels == 1).astype(np.int64)
    auc_info = {
        "roc_auc_no_tumor": float(roc_auc_score(y_no_tumor, probs_no_tumor)),
        "pr_auc_no_tumor": float(average_precision_score(y_no_tumor, probs_no_tumor)),
    }
    logger.info(
        "AUC(no_tumor): ROC=%.4f PR=%.4f",
        auc_info["roc_auc_no_tumor"],
        auc_info["pr_auc_no_tumor"],
    )

    # High-confidence no_tumor analysis
    high_conf = analyze_high_confidence_positive(
        labels,
        probs_no_tumor,
        positive_label=1,
        min_confidence=cfg.high_confidence_no_tumor_prob,
    )
    logger.info(
        "HighConf P(no_tumor)>=%.2f: selected=%d/%d coverage=%.2f%% precision=%s",
        cfg.high_confidence_no_tumor_prob,
        high_conf["selected"],
        high_conf["total"],
        100.0 * high_conf["coverage"],
        "nan" if np.isnan(high_conf["precision"]) else f"{100.0 * high_conf['precision']:.2f}%",
    )

    # Reliability
    rel_rows, ece, brier = compute_binary_reliability_stats(
        y_no_tumor, probs_no_tumor, n_bins=cfg.calibration_bins
    )
    rel_info = {"ece": float(ece), "brier": float(brier)}
    logger.info(
        "Reliability: ECE=%.4f Brier=%.4f bins=%d/%d",
        rel_info["ece"],
        rel_info["brier"],
        len(rel_rows),
        cfg.calibration_bins,
    )

    # Save artifacts
    probs_csv = cfg.log_dir / f"{cfg.exp_name}_probs.csv"
    cm_csv = cfg.log_dir / f"{cfg.exp_name}_confusion_matrix_default.csv"
    reliability_csv = cfg.log_dir / f"{cfg.exp_name}_reliability_no_tumor.csv"
    reliability_png = cfg.log_dir / f"{cfg.exp_name}_reliability_no_tumor.png"
    metrics_json = cfg.log_dir / f"{cfg.exp_name}_metrics.json"
    report_md = cfg.log_dir / f"{cfg.exp_name}_detailed_report.md"

    out_df = eval_dataset.df.reset_index(drop=True).copy()
    out_df["prob_benign"] = probs_benign.astype(np.float32)
    out_df["prob_no_tumor"] = probs_no_tumor.astype(np.float32)
    out_df["pred_default"] = default_res["preds"].astype(int)
    out_df["pred_best_f1"] = best_f1_res["preds"].astype(int)
    for key, item in policies.items():
        col = "pred_" + key.replace("(", "_").replace(")", "").replace("%", "pct").replace("<=", "le").replace(" ", "")
        out_df[col] = item["metrics"]["preds"].astype(int)
    out_df.to_csv(probs_csv, index=False, encoding="utf-8")

    cm = np.array(default_res["confusion_matrix"], dtype=np.int64)
    cm_df = pd.DataFrame(
        cm,
        index=["true_benign", "true_no_tumor"],
        columns=["pred_benign", "pred_no_tumor"],
    )
    cm_df.to_csv(cm_csv, encoding="utf-8")

    save_reliability_stats_csv(rel_rows, str(reliability_csv))
    save_reliability_diagram(
        rel_rows,
        str(reliability_png),
        title=f"{cfg.exp_name} reliability for P(no_tumor)",
    )

    metrics_payload = {
        "dataset_info": dataset_info,
        "default_threshold": {k: v for k, v in default_res.items() if k != "preds"},
        "best_f1_threshold": {k: v for k, v in best_f1_res.items() if k != "preds"},
        "best_f1_threshold_value": float(best_f1_t),
        "best_f1_score": float(best_f1),
        "policy_thresholds": {
            k: {
                "policy": v["policy"],
                "metrics": {mk: mv for mk, mv in v["metrics"].items() if mk != "preds"},
            }
            for k, v in policies.items()
        },
        "auc_info": auc_info,
        "high_confidence_no_tumor": high_conf,
        "reliability": rel_info,
    }
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    save_markdown_report(
        out_md=report_md,
        dataset_info=dataset_info,
        auc_info=auc_info,
        default_res=default_res,
        best_f1_res=best_f1_res,
        policies=policies,
        rel_info=rel_info,
        files={
            "binary_excel": binary_excel,
            "probs_csv": probs_csv,
            "cm_csv": cm_csv,
            "reliability_csv": reliability_csv,
            "reliability_png": reliability_png,
            "metrics_json": metrics_json,
        },
    )

    logger.info("=" * 80)
    logger.info("Evaluation finished.")
    logger.info(f"Detailed report: {report_md}")
    logger.info(f"Metrics json:    {metrics_json}")
    logger.info(f"Probs csv:       {probs_csv}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

