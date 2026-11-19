# -*- coding: utf-8 -*-
"""
方案 A（baseline）：基于 mask 的传统透视/旋转校正
流程: mask 轮廓 -> 四角点检测 -> 排序 -> 透视变换拉平
输出: results/baseline/  +  eval/baseline_log.csv
"""
import os
import sys
import csv
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from imgio import imread_u, imwrite_u  # noqa: E402
import cv2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "GeoCorrection", "_extract", "GeoCorrection", "images")
MSK_DIR = os.path.join(ROOT, "GeoCorrection", "_extract", "GeoCorrection", "mask")
OUT_DIR = os.path.join(ROOT, "results", "baseline")
LOG_CSV = os.path.join(ROOT, "eval", "baseline_log.csv")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)


def order_corners(pts):
    """将4个点按 左上/右上/右下/左下 排序"""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def get_quad_corners(mask):
    """从二值 mask 提取文档四角点; 返回 (4x2 float32) 或 None"""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 5000:
        return None
    # 先用多边形逼近
    peri = cv2.arcLength(c, True)
    for eps in np.arange(0.005, 0.10, 0.005):
        approx = cv2.approxPolyDP(c, eps * peri, True)
        if len(approx) == 4:
            return order_corners(approx.reshape(4, 2))
    # fallback: 最小外接矩形
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return order_corners(box)


def warp_doc(img, quad):
    """按四角点做透视校正, 目标尺寸取对应边长均值"""
    tl, tr, br, bl = quad
    w_top = np.linalg.norm(tr - tl)
    w_bot = np.linalg.norm(br - bl)
    h_left = np.linalg.norm(bl - tl)
    h_right = np.linalg.norm(br - tr)
    W = int(max(w_top, w_bot))
    H = int(max(h_left, h_right))
    if W < 10 or H < 10:
        return None
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, M, (W, H))


def main():
    names = sorted(os.listdir(IMG_DIR))
    log = []
    for name in tqdm(names, desc="baseline"):
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(IMG_DIR, name)
        msk_path = os.path.join(MSK_DIR, stem + ".png")
        try:
            img = imread_u(img_path)
            mask = imread_u(msk_path, cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                log.append([name, "FAIL", "missing file"]); continue
            _, binm = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            quad = get_quad_corners(binm)
            if quad is None:
                log.append([name, "FAIL", "no contour"]); continue
            out = warp_doc(img, quad)
            if out is None:
                log.append([name, "FAIL", "bad size"]); continue
            imwrite_u(os.path.join(OUT_DIR, stem + ".png"), out)
            log.append([name, "OK", ""])
        except Exception as e:
            log.append([name, "FAIL", str(e)[:100]])

    with open(LOG_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["name", "status", "error"])
        w.writerows(log)
    ok = sum(1 for r in log if r[1] == "OK")
    print(f"完成: {ok}/{len(log)} 成功, 日志: {LOG_CSV}")


if __name__ == "__main__":
    main()
