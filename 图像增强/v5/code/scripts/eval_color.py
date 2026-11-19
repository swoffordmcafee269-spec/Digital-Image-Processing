# -*- coding: utf-8 -*-
"""颜色恢复度评估：回答「到底还灰不灰」。

用三个与主观观感强相关的量，而不是 PSNR：

  1. 色度相关系数 corr(Cb, Cr)
     在 YCbCr 里把每个像素当成一个二维色度点。pred 与 GT 的色度点云
     相关性越高，说明「哪里偏红、哪里偏蓝」判断得越对。
     全局色偏修正在这里几乎拿不到分（整体平移不改相关），
     所以它真正衡量的是**空间上的色彩结构是否复原**。

  2. 色度标准差比 std_ratio = std(pred_c) / std(gt_c)
     衡量颜色「鲜艳度」。v1 若是灰白，这个值会明显 < 1。

  3. 平均色度误差 |mean(pred_c) - mean(gt_c)|
     衡量整体偏色（白平衡是否准）。WB 生效后这项应显著下降。

用法：
    D:/Python文件夹/python.exe v5/code/scripts/eval_color.py sony
"""
import csv
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_HERE)
_V5 = os.path.dirname(_CODE)
_PROJ = os.path.dirname(_V5)

LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def read_img(path):
    """中文路径安全读图，返回 RGB float32 [0,1]。"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"读图失败: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def chroma_components(rgb):
    """返回 (Y, Cb, Cr)，Cb/Cr 已去均值（只保留色彩结构）。"""
    y = rgb @ LUMA
    cb = rgb[..., 2] - y          # B - Y
    cr = rgb[..., 0] - y          # R - Y
    return y, cb, cr


def color_metrics(pred, gt):
    if pred.shape != gt.shape:
        pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                          interpolation=cv2.INTER_AREA)
    _, pcb, pcr = chroma_components(pred)
    _, gcb, gcr = chroma_components(gt)

    # 整体偏色（在去均值之前算）
    bias = (abs(pcb.mean() - gcb.mean()) + abs(pcr.mean() - gcr.mean())) / 2.0

    pcb = pcb - pcb.mean(); pcr = pcr - pcr.mean()
    gcb = gcb - gcb.mean(); gcr = gcr - gcr.mean()

    a = np.stack([pcb.ravel(), pcr.ravel()])
    b = np.stack([gcb.ravel(), gcr.ravel()])
    # 2D 色度点云的相关系数：把 Cb、Cr 拉平成 2N 维后求 Pearson
    a1 = a.ravel(); b1 = b.ravel()
    sa, sb = a1.std(), b1.std()
    corr = float(np.mean((a1 - a1.mean()) * (b1 - b1.mean())) / (sa * sb + 1e-12))

    std_p = float(np.sqrt((pcb.var() + pcr.var()) / 2))
    std_g = float(np.sqrt((gcb.var() + gcr.var()) / 2))
    return corr, std_p / (std_g + 1e-12), float(bias), std_p, std_g


def main():
    tok = (sys.argv[1] if len(sys.argv) > 1 else "sony").strip().lower()
    if tok == "sony":
        gt_root = os.path.join(_PROJ, "data", "Sony", "Sony_0.001", "Sony_gt", "gt")
        v1_dir = os.path.join(_PROJ, "outputs", "results", "Sony", "images")
        v5_dir = os.path.join(_PROJ, "v5", "results", "sony", "Sony", "images")
        ext = ".png"
    else:
        gt_root = os.path.join(_PROJ, "data", "Canon", "Canon_0.0001", "Canon_gt", "gt")
        v1_dir = os.path.join(_PROJ, "outputs", "results", "Canon", "images")
        v5_dir = os.path.join(_PROJ, "v5", "results", "canon", "Canon", "images")
        ext = ".jpg"

    names = sorted([os.path.splitext(f)[0] for f in os.listdir(gt_root)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    # 只评 v5 已经出完图的那部分，支持中途查看
    names = [n for n in names
             if os.path.exists(os.path.join(v1_dir, n + ext))
             and os.path.exists(os.path.join(v5_dir, n + ext))]
    print(f"[{tok}] 参与评估 {len(names)} 张")

    rows = []
    for i, n in enumerate(names):
        gt = read_img(os.path.join(gt_root, n + ".jpg"))
        p1 = read_img(os.path.join(v1_dir, n + ext))
        p5 = read_img(os.path.join(v5_dir, n + ext))
        c1, s1, b1, _, _ = color_metrics(p1, gt)
        c5, s5, b5, sp5, sg = color_metrics(p5, gt)
        rows.append((n, c1, s1, b1, c5, s5, b5, sp5, sg))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(names)}")

    arr = np.array([r[1:] for r in rows], dtype=np.float64)
    m = arr.mean(axis=0)

    print("\n" + "=" * 68)
    print(f"{'指标':<26}{'v1':>12}{'v5':>12}{'Δ':>14}")
    print("-" * 68)
    labels = ["色度相关 corr", "色度标准差比 std_ratio", "整体色偏 bias"]
    for j, lb in enumerate(labels):
        a, b = m[j], m[j + 3]
        d = b - a
        better = "↑好" if j < 2 else "↓好"
        print(f"{lb:<24}{a:>12.5f}{b:>12.5f}{d:>+13.5f}  {better}")
    print("=" * 68)
    print(f"GT 色度标准差 = {m[7]:.5f}")

    out_csv = os.path.join(_V5, "reports", f"color_{tok}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "v1_corr", "v1_std_ratio", "v1_bias",
                    "v5_corr", "v5_std_ratio", "v5_bias", "v5_std", "gt_std"])
        for r in rows:
            w.writerow([r[0]] + [round(v, 6) for v in r[1:]])
    print(f"明细已存: {out_csv}")


if __name__ == "__main__":
    main()
