"""H 实验 eval-only 补救脚本: 用现有 best.pth + thresholds.json 跑 test eval。

H 主脚本训练阶段已完成 (best epoch=6, val 5 底线全过),
但 test eval 段一开始还是 G 风格的代码导致 KeyError 崩了。
本脚本只做最后的 test 评估 + 混淆矩阵图,跳过训练。

跑法:
  CUDA_VISIBLE_DEVICES=3 /home/ubuntu/anaconda3/envs/gbp/bin/python \
      /data1/ouyangxinglong/GBP-Cascade/0510/scripts/_eval_only_H.py
"""
from __future__ import annotations

import os
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
for _p in [
    os.path.join(ROOT_DIR, "0408", "scripts"),
    os.path.join(ROOT_DIR, "0402", "scripts"),
    os.path.join(ROOT_DIR, "0323", "scripts"),
    os.path.join(ROOT_DIR, "0502", "scripts"),
    SCRIPT_DIR,
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


# 用 importlib 加载主脚本 (文件名以数字开头不能直接 import)
H_MAIN_PATH = os.path.join(SCRIPT_DIR, "20260510_task_risk_H_cascade.py")
spec = importlib.util.spec_from_file_location("h_main", H_MAIN_PATH)
h_main = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h_main)


from risk_utils import dump_json, plot_risk_confusion_matrices  # noqa: E402
from risk_utils_B_calib import apply_temperature, quick_brier  # noqa: E402
from risk_utils_E_aux_cls import evaluate_bands_external  # noqa: E402
from risk_utils_H_cascade import (  # noqa: E402
    decide_bands_cascade, search_thresholds_cascade,
)


def main():
    cfg = h_main.Config()

    # 加载 thresholds.json (训练阶段已写好)
    thr_path = os.path.join(cfg.log_dir, "thresholds.json")
    with open(thr_path) as f:
        thr = json.load(f)
    T_tumor = thr["T_tumor"]
    T_mal = thr["T_mal"]
    tau1 = thr["tau1"]
    tau2 = thr["tau2"]
    print(f"[Loaded thresholds] T_tumor={T_tumor:.4f} T_mal={T_mal:.4f} "
          f"tau1={tau1:.3f} tau2={tau2:.3f}")
    print(f"  val obj: {thr.get('h_search_result', {}).get('objective', 'N/A')}")
    print(f"  val ben_to_low: {thr.get('h_search_result', {}).get('ben_to_low_share', 'N/A')}")
    print(f"  val nt_to_high: {thr.get('h_search_result', {}).get('nt_to_high_share', 'N/A')}")

    # 简单 stub logger
    class _L:
        def info(self, *a):    print("[info]", *a)
        def warning(self, *a): print("[warn]", *a)
        def error(self, *a):   print("[err]", *a)
    logger = _L()

    # 重新构数据加载器(只用 test111/test112,但 build_dataloaders 会一起建,无所谓)
    print("[1] Building dataloaders ...")
    _, _, test111_loader, test112_loader = h_main.build_dataloaders(cfg, logger)

    # 重新构模型 + 加载 best.pth
    print("[2] Building model + loading best.pth ...")
    model = h_main.build_model(cfg, logger)
    if not os.path.exists(cfg.best_weight_path):
        print(f"[ERR] best.pth 不存在: {cfg.best_weight_path}")
        return
    try:
        state = torch.load(cfg.best_weight_path, map_location=cfg.device,
                           weights_only=True)
    except TypeError:
        state = torch.load(cfg.best_weight_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()
    print(f"[2] Loaded {cfg.best_weight_path}")

    # 跑 test111/test112
    eval_results = {}
    for phase, loader, tag in [
        ("Test-111", test111_loader, "111"),
        ("Test-112", test112_loader, "112"),
    ]:
        print(f"\n[3-{tag}] Predict {phase} ...")
        pred = h_main.collect_predictions_with_logits(model, loader, cfg.device)
        p_tumor_calib = apply_temperature(pred["tumor_logit"], T_tumor)
        p_mal_calib = apply_temperature(pred["mal_logit"], T_mal)
        bands = decide_bands_cascade(p_tumor_calib, p_mal_calib, tau1, tau2)

        ev = evaluate_bands_external(
            p_mal_calib, bands, pred["labels"],
            phase=f"{phase} (H cascade val-thr)",
            extra_meta=dict(
                thresholds=dict(tau1=tau1, tau2=tau2,
                                T_tumor=T_tumor, T_mal=T_mal),
                seg_dice=pred["seg_dice"],
                p_tumor_brier=quick_brier(pred["p_tumor"], pred["labels"]),
                p_mal_brier=quick_brier(pred["p_mal"], pred["labels"]),
                p_tumor_calib_brier=quick_brier(p_tumor_calib, pred["labels"]),
                p_mal_calib_brier=quick_brier(p_mal_calib, pred["labels"]),
            ))
        print(f"[{phase}] safety: {ev['safety']}")

        diag = search_thresholds_cascade(p_tumor_calib, p_mal_calib, pred["labels"])
        ev["diagnostic_search_on_self"] = diag

        json_path = os.path.join(cfg.log_dir, f"eval_{tag}.json")
        dump_json(ev, json_path)
        print(f"[{phase}] 评估已保存: {json_path}")
        eval_results[phase] = ev

    fig_path = os.path.join(cfg.log_dir, "confusion_matrices.png")
    plot_risk_confusion_matrices(eval_results, fig_path)
    print(f"\n[Done] 混淆矩阵图: {fig_path}")


if __name__ == "__main__":
    main()
