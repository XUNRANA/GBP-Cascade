"""
prepare_0514_dataset.py

将 0514dataset（按患者分层、含 JPG + TXT 报告）整理为训练可用的扁平格式：

输出 1: 0514dataset_flat/
  malignant / benign / no_tumor 下每个样本命名为
    {patient_id}_{image_stem}.png  +  同名 .json (更新 imagePath 字段)

输出 2: json_text_0514/
  {patient_id}.json  格式与项目已有 json_text/ 一致:
    {"case_id": "...", "text_bert": "...", "feat": {}}
  - 已在 json_text/ 中的患者 直接复制原文件
  - 仅 0514dataset 独有的患者 从 {patient_id}.txt 提取超声报告文本

输出 3: 0514dataset_flat/task_3class_train.xlsx
         0514dataset_flat/task_3class_test.xlsx
         0514dataset_flat/划分报告_3class.txt
  patient-level 7:3 split, 保留全部图像（不预平衡，BalancedBatchSampler 在线平衡）
"""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

# ─── 路径 ────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent.resolve()
ROOT_DIR    = (SCRIPT_DIR / ".." / "..").resolve()

SRC_ROOT    = ROOT_DIR / "0514dataset"
DST_ROOT    = ROOT_DIR / "0514dataset_flat"
JSON_TEXT_SRC = ROOT_DIR / "json_text"
JSON_TEXT_DST = ROOT_DIR / "json_text_0514"

CLASS_DIRS  = ["malignant", "benign", "no_tumor"]
LABEL_MAP   = {"malignant": 0, "benign": 1, "no_tumor": 2}
TEST_RATIO  = 0.30
SEED        = 42


# ════════════════════════════════════════════════════════════
#  Step 1: 整理图片 + JSON → 0514dataset_flat/
# ════════════════════════════════════════════════════════════

def _load_image_as_png(src_path: Path) -> Image.Image:
    return Image.open(src_path).convert("RGB")


def _update_json_imagepath(json_data: dict, new_image_filename: str) -> dict:
    """labelme JSON 里的 imagePath 指向新文件名（不含目录）。"""
    out = dict(json_data)
    out["imagePath"] = new_image_filename
    out.pop("imageData", None)  # 不存原始 base64，节省空间
    return out


def process_images(verbose: bool = True) -> pd.DataFrame:
    """
    遍历 0514dataset 所有患者，把图片/JSON 复制到 0514dataset_flat，
    返回包含 (image_path, label, patient_id) 的 DataFrame。
    """
    rows = []
    total_copied = 0
    total_converted = 0

    for cls in CLASS_DIRS:
        cls_src = SRC_ROOT / cls
        cls_dst = DST_ROOT / cls
        cls_dst.mkdir(parents=True, exist_ok=True)
        label = LABEL_MAP[cls]

        patients = sorted(cls_src.iterdir()) if cls_src.exists() else []
        for pat_dir in patients:
            if not pat_dir.is_dir():
                continue
            pid = pat_dir.name

            # 找该患者目录下所有图片
            img_files = sorted(
                f for f in pat_dir.iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg")
            )

            for img_src in img_files:
                stem = img_src.stem  # e.g. "US_Image4"
                new_stem = f"{pid}_{stem}"  # e.g. "00320921_US_Image4"
                new_png_name = f"{new_stem}.png"
                new_json_name = f"{new_stem}.json"
                dst_png = cls_dst / new_png_name
                dst_json = cls_dst / new_json_name

                # 图片
                if not dst_png.exists():
                    img = _load_image_as_png(img_src)
                    img.save(dst_png, format="PNG")
                    if img_src.suffix.lower() != ".png":
                        total_converted += 1
                    else:
                        total_copied += 1

                # JSON (同目录同名)
                json_src = img_src.with_suffix(".json")
                if json_src.exists() and not dst_json.exists():
                    with open(json_src, "r", encoding="utf-8") as f:
                        jdata = json.load(f)
                    jdata = _update_json_imagepath(jdata, new_png_name)
                    with open(dst_json, "w", encoding="utf-8") as f:
                        json.dump(jdata, f, ensure_ascii=False, indent=2)

                # 记录
                relative_path = f"{cls}/{new_png_name}"
                rows.append({
                    "image_path": relative_path,
                    "label": label,
                    "patient_id": pid,
                })

        if verbose:
            n_patients = len([p for p in cls_src.iterdir() if p.is_dir()])
            n_images = len([r for r in rows if r["label"] == label])
            print(f"  [{cls}] {n_patients} 患者 → {n_images} 图")

    print(f"图片复制完成: {total_copied} PNG 复制, {total_converted} JPG→PNG 转换")
    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
#  Step 2: 构建 json_text_0514/
# ════════════════════════════════════════════════════════════

def _extract_gallbladder_text(txt_content: str) -> str:
    """
    从超声报告里提取胆囊相关段落。
    策略: 找以 "胆囊" 开头的行或段落，优先取该段；
          若找不到，返回全文（BERT 自行处理长截断）。
    """
    lines = [l.strip() for l in txt_content.split("\n") if l.strip()]
    gb_lines = []
    in_gb = False
    for line in lines:
        if re.search(r"胆囊", line):
            in_gb = True
        if in_gb:
            gb_lines.append(line)
            # 遇到下一器官关键词就停（胰腺、脾、肾 etc.）
            if len(gb_lines) > 1 and re.search(r"胰腺|脾[：:]|双肾|膀胱|前列腺", line):
                gb_lines.pop()  # 去掉触发终止的那行
                break
    if gb_lines:
        return "".join(gb_lines)
    return txt_content.replace("\n", "").replace(" ", "")


def build_json_text_0514(df: pd.DataFrame, verbose: bool = True):
    """
    构建 json_text_0514/:
    - 已有 json_text/ 的患者: 直接复制
    - 仅在 0514dataset 的新患者: 从 .txt 提取超声报告
    """
    JSON_TEXT_DST.mkdir(exist_ok=True)

    # 已有患者
    existing_patients = {f.stem for f in JSON_TEXT_SRC.glob("*.json")}
    all_0514_patients = set(df["patient_id"].unique())

    copied = 0
    for pid in all_0514_patients & existing_patients:
        src = JSON_TEXT_SRC / f"{pid}.json"
        dst = JSON_TEXT_DST / f"{pid}.json"
        if not dst.exists():
            shutil.copy2(src, dst)
        copied += 1

    # 0514 独有的患者 → 从 txt 提取
    new_patients = all_0514_patients - existing_patients
    extracted = 0
    missing_txt = 0

    for cls in CLASS_DIRS:
        for pid in new_patients:
            txt_path = SRC_ROOT / cls / pid / f"{pid}.txt"
            if not txt_path.exists():
                continue
            dst = JSON_TEXT_DST / f"{pid}.json"
            if dst.exists():
                continue

            # 读取（GBK 编码）
            text = ""
            for enc in ("gbk", "gb2312", "utf-8", "latin-1"):
                try:
                    text = txt_path.read_text(encoding=enc)
                    break
                except Exception:
                    continue

            if text:
                gb_text = _extract_gallbladder_text(text)
                out = {"case_id": pid, "text_bert": gb_text, "feat": {}}
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                extracted += 1
            else:
                missing_txt += 1

    if verbose:
        print(f"json_text_0514: 复制已有={copied}, 新提取={extracted}, "
              f"缺少txt={missing_txt}, 合计={len(list(JSON_TEXT_DST.glob('*.json')))}")


# ════════════════════════════════════════════════════════════
#  Step 3: Patient-level Train/Test Split
# ════════════════════════════════════════════════════════════

def make_splits(df: pd.DataFrame, verbose: bool = True):
    """Patient-level 7:3 分层划分，保留全部图像。"""
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_RATIO, random_state=SEED)
    train_idx, test_idx = next(
        gss.split(df, df["label"], groups=df["patient_id"])
    )
    train_df = df.iloc[train_idx][["image_path", "label"]].reset_index(drop=True)
    test_df  = df.iloc[test_idx][["image_path", "label"]].reset_index(drop=True)

    train_path = DST_ROOT / "task_3class_train.xlsx"
    test_path  = DST_ROOT / "task_3class_test.xlsx"
    train_df.to_excel(train_path, index=False)
    test_df.to_excel(test_path, index=False)

    # 划分报告
    class_names = {0: "Malignant", 1: "Benign", 2: "No_Tumor"}
    report_lines = [
        "=== 三分类数据集划分报告 (0514dataset, 全量 - 不预平衡) ===",
        f"划分比例: Train:Test ≈ {1-TEST_RATIO:.0%}:{TEST_RATIO:.0%}",
        "标签映射: {'malignant': 0, 'benign': 1, 'no_tumor': 2}",
        "",
        f"训练集总数: {len(train_df)}",
    ]
    for l, n in sorted(class_names.items()):
        cnt = (train_df["label"] == l).sum()
        report_lines.append(f"  - {n}: {cnt}")
    report_lines += ["", f"测试集总数: {len(test_df)}"]
    for l, n in sorted(class_names.items()):
        cnt = (test_df["label"] == l).sum()
        report_lines.append(f"  - {n}: {cnt}")
    report_lines += ["", "患者泄露校验:"]

    # 泄露校验
    train_pids = set(
        df.iloc[train_idx]["patient_id"].unique()
    )
    test_pids = set(
        df.iloc[test_idx]["patient_id"].unique()
    )
    overlap = train_pids & test_pids
    if overlap:
        report_lines.append(f"  ⚠️ 发现泄露 {len(overlap)} 位患者!")
    else:
        for cls in CLASS_DIRS:
            report_lines.append(f"  - {cls}: ✅ 无泄露")

    report_text = "\n".join(report_lines)
    (DST_ROOT / "划分报告_3class.txt").write_text(report_text, encoding="utf-8")

    if verbose:
        print(f"Train: {len(train_df)} 图 | Test: {len(test_df)} 图")
        for l, n in sorted(class_names.items()):
            tr = (train_df["label"] == l).sum()
            te = (test_df["label"] == l).sum()
            print(f"  {n}: train={tr}, test={te}")
        print(f"患者泄露: {len(overlap)} 位" if overlap else "患者泄露: 无")


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("0514dataset 数据集准备")
    print(f"  源目录:    {SRC_ROOT}")
    print(f"  目标目录:  {DST_ROOT}")
    print(f"  json_text: {JSON_TEXT_DST}")
    print("=" * 60)

    print("\n[Step 1] 处理图片 + JSON → 扁平结构 ...")
    df = process_images(verbose=True)
    print(f"  总计: {len(df)} 个样本, {df['patient_id'].nunique()} 位患者")

    print("\n[Step 2] 构建 json_text_0514/ ...")
    build_json_text_0514(df, verbose=True)

    print("\n[Step 3] 生成 Train/Test Excel 划分 ...")
    make_splits(df, verbose=True)

    print("\n✅ 完成！")
    print(f"  0514dataset_flat: {DST_ROOT}")
    print(f"  json_text_0514:   {JSON_TEXT_DST}")
    print(f"  train.xlsx: {DST_ROOT / 'task_3class_train.xlsx'}")
    print(f"  test.xlsx:  {DST_ROOT / 'task_3class_test.xlsx'}")


if __name__ == "__main__":
    main()
