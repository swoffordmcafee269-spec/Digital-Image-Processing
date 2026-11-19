# -*- coding: utf-8 -*-
"""
方案 C（主方法）：UVDoc (SIGGRAPH Asia 2023) 预训练模型 dewarping
流程: mask 裁剪文档 ROI -> 缩放送入 UVDoc -> 预测形变网格 -> 全分辨率重映射
输出: results/uvdoc/  +  eval/uvdoc_log.csv
"""
import os
import sys
import csv
import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "GeoCorrection", "_extract", "GeoCorrection", "images")
MSK_DIR = os.path.join(ROOT, "GeoCorrection", "_extract", "GeoCorrection", "mask")
OUT_DIR = os.path.join(ROOT, "results", "uvdoc")
LOG_CSV = os.path.join(ROOT, "eval", "uvdoc_log.csv")
CKPT = os.path.join(ROOT, "uvdoc", "UVDoc", "model", "best_model.pkl")
sys.path.insert(0, os.path.join(ROOT, "uvdoc", "UVDoc"))
sys.path.insert(0, os.path.join(ROOT, "common"))
from utils import IMG_SIZE, load_model, bilinear_unwarping  # noqa: E402
from imgio import imread_u, imwrite_u  # noqa: E402

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LOG_CSV), exist_ok=True)


def crop_by_mask(img, mask, margin=0.03):
    """按 mask bbox 裁剪文档区域(带边距), 返回裁剪图"""
    ys, xs = np.where(mask > 127)
    if len(xs) == 0:
        return img, (0, 0, img.shape[1], img.shape[0])
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    mx, my = int((x1 - x0) * margin), int((y1 - y0) * margin)
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(img.shape[1], x1 + mx), min(img.shape[0], y1 + my)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(CKPT).to(device).eval()
    print("device:", device)

    names = sorted(os.listdir(IMG_DIR))
    log = []
    with torch.no_grad():
        for name in tqdm(names, desc="uvdoc"):
            stem = os.path.splitext(name)[0]
            img = imread_u(os.path.join(IMG_DIR, name))
            mask = imread_u(os.path.join(MSK_DIR, stem + ".png"), cv2.IMREAD_GRAYSCALE)
            if img is None or mask is None:
                log.append([name, "FAIL", "missing file"]); continue
            try:
                crop, _ = crop_by_mask(img, mask)
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                h, w = rgb.shape[:2]
                inp = torch.from_numpy(
                    cv2.resize(rgb, (IMG_SIZE[0], IMG_SIZE[1])).transpose(2, 0, 1)
                ).unsqueeze(0).to(device)
                point_positions2D, _ = model(inp)
                unwarped = bilinear_unwarping(
                    warped_img=torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device),
                    point_positions=torch.unsqueeze(point_positions2D[0], dim=0),
                    img_size=(w, h),
                )
                out = (unwarped[0].detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                imwrite_u(os.path.join(OUT_DIR, stem + ".png"), out_bgr)
                log.append([name, "OK", ""])
            except Exception as e:
                log.append([name, "FAIL", str(e)[:120]])

    with open(LOG_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["name", "status", "error"])
        w.writerows(log)
    ok = sum(1 for r in log if r[1] == "OK")
    print(f"完成: {ok}/{len(log)} 成功, 日志: {LOG_CSV}")


if __name__ == "__main__":
    main()
