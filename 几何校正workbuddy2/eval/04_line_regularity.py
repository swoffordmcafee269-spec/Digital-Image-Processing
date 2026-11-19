# -*- coding: utf-8 -*-
"""
评估2: 文字行规整度量化
思路: 对文档区域灰度图做水平投影/行方向分析:
  - line_var: 水平投影峰的周期稳定性(行距一致程度), 越大越规整
  - curvature: 用每行文字的质心横向偏移标准差近似行弯曲程度, 越小越平直
输出: eval/metric_line_regularity.csv (original/baseline/uvdoc 三列对比)
"""
import os
import sys
import csv
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from imgio import imread_u  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "GeoCorrection", "_extract", "GeoCorrection", "images")
BASE_DIR = os.path.join(ROOT, "results", "baseline")
UVDOC_DIR = os.path.join(ROOT, "results", "uvdoc")
OUT_CSV = os.path.join(ROOT, "eval", "metric_line_regularity.csv")


def binarize_text(img):
    """文档区域内提取文字二值图 (文字=1)"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自适应阈值突出文字
    bw = cv2.adaptiveThreshold(255 - gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 25, 15)
    # 限制在较亮区域内(排除深色背景)
    bright = gray > 60
    return (bw > 0) & bright


def line_metrics(img, work_h=1000):
    """计算行规整度指标"""
    h, w = img.shape[:2]
    s = work_h / h
    im = cv2.resize(img, (int(w * s), work_h))
    txt = binarize_text(im).astype(np.uint8)
    if txt.sum() < 500:
        return None
    # 裁到文字包围盒
    ys, xs = np.where(txt)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    txt = txt[y0:y1 + 1, x0:x1 + 1]
    H, W = txt.shape
    # 行质心横向偏移 (把每行文字的 x 质心相对行块中心的偏移做标准差 -> 弯曲程度)
    row_sum = txt.sum(axis=1)
    row_cx = np.full(H, np.nan)
    for r in range(H):
        if row_sum[r] > 3:
            row_cx[r] = np.where(txt[r] > 0)[0].mean()
    valid = ~np.isnan(row_cx)
    if valid.sum() < 50:
        return None
    # 去掉整体倾斜: 对质心序列拟合直线, 残差即弯曲
    r_idx = np.where(valid)[0].astype(float)
    coef = np.polyfit(r_idx, row_cx[valid], 1)
    resid = row_cx[valid] - np.polyval(coef, r_idx)
    curvature = float(np.std(resid))          # 越小越平直
    # 行投影周期性: 对行和序列做FFT, 主频能量占比 (越大行越规整)
    prof = row_sum.astype(float)
    prof = prof - prof.mean()
    fft = np.abs(np.fft.rfft(prof))
    if len(fft) > 10 and fft.sum() > 0:
        fft[:3] = 0  # 去低频趋势
        peak = fft.max()
        line_var = float(peak / (fft.sum() + 1e-6))
    else:
        line_var = 0.0
    return curvature, line_var


def main():
    names = sorted(os.listdir(IMG_DIR))
    rows = []
    for name in tqdm(names, desc="line-reg"):
        stem = os.path.splitext(name)[0]
        rec = {"name": stem}
        for tag, path in [
            ("original", os.path.join(IMG_DIR, name)),
            ("baseline", os.path.join(BASE_DIR, stem + ".png")),
            ("uvdoc", os.path.join(UVDOC_DIR, stem + ".png")),
        ]:
            img = imread_u(path)
            if img is None:
                rec[tag + "_curv"] = ""
                rec[tag + "_lv"] = ""
                continue
            m = line_metrics(img)
            if m is None:
                rec[tag + "_curv"], rec[tag + "_lv"] = "", ""
            else:
                rec[tag + "_curv"], rec[tag + "_lv"] = [f"{v:.4f}" for v in m]
        rows.append(rec)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["name",
                                          "original_curv", "original_lv",
                                          "baseline_curv", "baseline_lv",
                                          "uvdoc_curv", "uvdoc_lv"])
        w.writeheader()
        w.writerows(rows)
    print("完成:", OUT_CSV, f"({len(rows)} 行)")


if __name__ == "__main__":
    main()
