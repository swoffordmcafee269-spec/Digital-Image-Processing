# -*- coding: utf-8 -*-
"""
评估1: 三联可视化对比图 (原图 | baseline | UVDoc)
输出: results/compare/<stem>.jpg (全部200张) + results/compare/_grid_sample.jpg 抽样大图
"""
import os
import sys
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from imgio import imread_u, imwrite_u  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "GeoCorrection", "_extract", "GeoCorrection", "images")
BASE_DIR = os.path.join(ROOT, "results", "baseline")
UVDOC_DIR = os.path.join(ROOT, "results", "uvdoc")
OUT_DIR = os.path.join(ROOT, "results", "compare")
os.makedirs(OUT_DIR, exist_ok=True)

TH = 360  # 缩略高度


def thumb(img, th=TH):
    h, w = img.shape[:2]
    s = th / h
    return cv2.resize(img, (int(w * s), th))


def add_label(img, text):
    bar = np.full((34, img.shape[1], 3), 255, np.uint8)
    cv2.putText(bar, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return np.vstack([bar, img])


def pad_to_width(img, w):
    if img.shape[1] >= w:
        return img
    pad = np.full((img.shape[0], w - img.shape[1], 3), 255, np.uint8)
    return np.hstack([img, pad])


def pad_to_width_h(img, w):
    """同 pad_to_width, 但按图像高度补白对齐"""
    return pad_to_width(img, w)


def main():
    names = sorted(os.listdir(IMG_DIR))
    samples = []
    for name in tqdm(names, desc="compare"):
        stem = os.path.splitext(name)[0]
        img = imread_u(os.path.join(IMG_DIR, name))
        base = imread_u(os.path.join(BASE_DIR, stem + ".png"))
        uv = imread_u(os.path.join(UVDOC_DIR, stem + ".png"))
        if img is None or base is None or uv is None:
            continue
        t1, t2, t3 = thumb(img), thumb(base), thumb(uv)
        W = max(t1.shape[1], t2.shape[1], t3.shape[1])
        panel = np.hstack([add_label(pad_to_width(t1, W), "Original"),
                           add_label(pad_to_width(t2, W), "Baseline"),
                           add_label(pad_to_width(t3, W), "UVDoc")])
        imwrite_u(os.path.join(OUT_DIR, stem + ".jpg"), panel)
        samples.append(panel)

    # 抽样 12 张拼成总览
    idx = np.linspace(0, len(samples) - 1, min(12, len(samples))).astype(int)
    GW = max(p.shape[1] for p in samples)
    grid = np.vstack([pad_to_width_h(samples[i], GW) for i in idx])
    # 限宽保存
    if grid.shape[1] > 3000:
        s = 3000 / grid.shape[1]
        grid = cv2.resize(grid, (3000, int(grid.shape[0] * s)))
    imwrite_u(os.path.join(OUT_DIR, "_grid_sample.jpg"), grid)
    print("对比图完成:", OUT_DIR)


if __name__ == "__main__":
    main()
