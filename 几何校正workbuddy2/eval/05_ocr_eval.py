# -*- coding: utf-8 -*-
"""
评估3: OCR 对比 (rapidocr, 本地推理)
对 original/baseline/uvdoc 分别识别, 统计字符数/行数/平均置信度
输出: eval/metric_ocr.csv
"""
import os
import sys
import csv
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"))
from imgio import imread_u  # noqa: E402
import cv2  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(ROOT, "GeoCorrection", "_extract", "GeoCorrection", "images")
BASE_DIR = os.path.join(ROOT, "results", "baseline")
UVDOC_DIR = os.path.join(ROOT, "results", "uvdoc")
OUT_CSV = os.path.join(ROOT, "eval", "metric_ocr.csv")

from rapidocr_onnxruntime import RapidOCR  # noqa: E402
ocr = RapidOCR()

MAX_SIDE = 2000  # OCR 前限边, 加速且稳定


def ocr_metrics(img):
    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        s = MAX_SIDE / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
    result, _ = ocr(img)
    if not result:
        return 0, 0, 0.0
    n_chars = sum(len(r[1]) for r in result)
    n_lines = len(result)
    conf = float(np.mean([float(r[2]) for r in result]))
    return n_chars, n_lines, conf


def main():
    names = sorted(os.listdir(IMG_DIR))
    rows = []
    for name in tqdm(names, desc="ocr"):
        stem = os.path.splitext(name)[0]
        rec = {"name": stem}
        for tag, path in [
            ("original", os.path.join(IMG_DIR, name)),
            ("baseline", os.path.join(BASE_DIR, stem + ".png")),
            ("uvdoc", os.path.join(UVDOC_DIR, stem + ".png")),
        ]:
            img = imread_u(path)
            if img is None:
                rec[f"{tag}_chars"], rec[f"{tag}_lines"], rec[f"{tag}_conf"] = "", "", ""
                continue
            c, l, cf = ocr_metrics(img)
            rec[f"{tag}_chars"], rec[f"{tag}_lines"], rec[f"{tag}_conf"] = c, l, f"{cf:.4f}"
        rows.append(rec)
        if len(rows) % 20 == 0:
            _flush(rows)

    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["name",
                                          "original_chars", "original_lines", "original_conf",
                                          "baseline_chars", "baseline_lines", "baseline_conf",
                                          "uvdoc_chars", "uvdoc_lines", "uvdoc_conf"])
        w.writeheader()
        w.writerows(rows)
    print("完成:", OUT_CSV)


def _flush(rows):
    pass  # 预留


if __name__ == "__main__":
    main()
