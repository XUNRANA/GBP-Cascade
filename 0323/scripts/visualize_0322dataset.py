"""
可视化 0322dataset 中所有 PNG + JSON 标注
- gallbladder: 绿色矩形框
- gallbladder polyp: 红色多边形轮廓
输出:
  1. 每个类别的样本概览图 (overview_{class}.png)
  2. 全部标注图保存到 0322dataset_vis/ 目录
"""

import os
import json
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

# ──────────────── 配置 ────────────────
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "0322dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "0322dataset_vis")
CLASSES = ["benign", "no_tumor", "malignant"]

# 颜色: (outline, fill with alpha)
COLORS = {
    "gallbladder": (0, 255, 0),        # 绿色
    "gallbladder polyp": (255, 0, 0),   # 红色
}
DEFAULT_COLOR = (255, 255, 0)  # 黄色 fallback

OVERVIEW_COLS = 8
OVERVIEW_ROWS = 4  # 每类展示 32 张样本


def draw_annotations(img, shapes):
    """在图片上绘制 JSON 标注"""
    overlay = img.copy().convert("RGBA")
    draw_overlay = ImageDraw.Draw(overlay)
    draw = ImageDraw.Draw(img)

    for shape in shapes:
        label = shape.get("label", "unknown")
        points = shape.get("points", [])
        shape_type = shape.get("shape_type", "")
        color = COLORS.get(label, DEFAULT_COLOR)

        if shape_type == "rectangle" and len(points) == 2:
            x1, y1 = points[0]
            x2, y2 = points[1]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            # 标签文字
            draw.text((x1 + 2, max(y1 - 12, 0)), label, fill=color)

        elif shape_type == "polygon" and len(points) >= 3:
            flat_points = [(p[0], p[1]) for p in points]
            # 半透明填充
            fill_color = color + (60,)
            draw_overlay.polygon(flat_points, outline=color, fill=fill_color)
            # 轮廓
            draw.polygon(flat_points, outline=color)
            # 标签文字
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            draw.text((cx - 20, cy - 6), label, fill=color)

    # 合成半透明层
    img_rgba = img.convert("RGBA")
    result = Image.alpha_composite(img_rgba, overlay)
    return result.convert("RGB")


def process_one(class_name, png_name):
    """处理单张图: 读取 PNG + JSON, 绘制标注, 返回标注图"""
    png_path = os.path.join(DATASET_DIR, class_name, png_name)
    json_path = os.path.join(DATASET_DIR, class_name, png_name.replace(".png", ".json"))

    img = Image.open(png_path).convert("RGB")

    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            data = json.load(f)
        shapes = data.get("shapes", [])
        img = draw_annotations(img, shapes)

    return img


def save_all_annotated():
    """保存所有标注图到 OUTPUT_DIR"""
    total = 0
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        out_dir = os.path.join(OUTPUT_DIR, cls)
        os.makedirs(out_dir, exist_ok=True)

        pngs = sorted([f for f in os.listdir(cls_dir) if f.endswith(".png")])
        for png_name in pngs:
            img = process_one(cls, png_name)
            img.save(os.path.join(out_dir, png_name))
            total += 1

        print(f"  {cls}: {len(pngs)} 张已保存")
    print(f"  共 {total} 张标注图 -> {OUTPUT_DIR}")


def make_overview():
    """每个类别生成一张概览大图"""
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        pngs = sorted([f for f in os.listdir(cls_dir) if f.endswith(".png")])

        n_show = min(OVERVIEW_COLS * OVERVIEW_ROWS, len(pngs))
        # 均匀采样
        indices = np.linspace(0, len(pngs) - 1, n_show, dtype=int)
        selected = [pngs[i] for i in indices]

        cols = OVERVIEW_COLS
        rows = math.ceil(n_show / cols)
        thumb_size = 256
        title_h = 40
        canvas_w = cols * thumb_size
        canvas_h = title_h + rows * thumb_size
        canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
        draw_canvas = ImageDraw.Draw(canvas)

        # 标题
        title = f"{cls}  ({len(pngs)} images)"
        draw_canvas.text((10, 10), title, fill=(255, 255, 255))

        for idx, png_name in enumerate(selected):
            img = process_one(cls, png_name)
            img = img.resize((thumb_size, thumb_size), Image.LANCZOS)

            r, c = divmod(idx, cols)
            x = c * thumb_size
            y = title_h + r * thumb_size
            canvas.paste(img, (x, y))

        out_path = os.path.join(OUTPUT_DIR, f"overview_{cls}.png")
        canvas.save(out_path)
        print(f"  概览图: {out_path}  ({rows}x{cols}, 采样 {n_show}/{len(pngs)})")


def print_stats():
    """打印标注统计"""
    print("\n===== 0322dataset 标注统计 =====")
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        pngs = sorted([f for f in os.listdir(cls_dir) if f.endswith(".png")])
        label_counts = defaultdict(int)
        shape_type_counts = defaultdict(int)
        no_annotation = 0

        for png_name in pngs:
            json_path = os.path.join(cls_dir, png_name.replace(".png", ".json"))
            if os.path.exists(json_path):
                with open(json_path) as f:
                    data = json.load(f)
                shapes = data.get("shapes", [])
                if not shapes:
                    no_annotation += 1
                for s in shapes:
                    label_counts[s.get("label", "?")] += 1
                    shape_type_counts[s.get("shape_type", "?")] += 1
            else:
                no_annotation += 1

        print(f"\n[{cls}] {len(pngs)} 张图")
        print(f"  标签分布: {dict(label_counts)}")
        print(f"  形状类型: {dict(shape_type_counts)}")
        if no_annotation:
            print(f"  无标注: {no_annotation}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_stats()

    print("\n===== 生成概览图 =====")
    make_overview()

    print("\n===== 保存全部标注图 =====")
    save_all_annotated()

    print("\n完成!")
