# -*- coding: utf-8 -*-
"""
评估4: 汇总统计 -> eval/summary_stats.csv + 控制台摘要表
"""
import csv
import os
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, "eval")


def read_csv(name):
    p = os.path.join(EVAL, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def col(rows, key):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
    return np.array(vals)


def main():
    line_rows = read_csv("metric_line_regularity.csv")
    ocr_rows = read_csv("metric_ocr.csv")
    out_rows = []

    def stat(tag, arr, note):
        if arr is None or len(arr) == 0:
            return
        out_rows.append([tag, f"{np.mean(arr):.4f}", f"{np.median(arr):.4f}", f"{np.std(arr):.4f}", len(arr), note])

    print("=" * 70)
    print("文字行规整度 (curvature=行弯曲残差, 越小越平直; lv=行周期性, 越大越规整)")
    print("=" * 70)
    if line_rows:
        for tag in ["original", "baseline", "uvdoc"]:
            c = col(line_rows, f"{tag}_curv")
            lv = col(line_rows, f"{tag}_lv")
            print(f"{tag:10s} curvature mean={np.mean(c):.4f}  line_var mean={np.mean(lv):.4f}  (n={len(c)})")
            out_rows.append([f"curvature_{tag}", f"{np.mean(c):.4f}", f"{np.median(c):.4f}", f"{np.std(c):.4f}", len(c), "越小越好"])
            out_rows.append([f"line_var_{tag}", f"{np.mean(lv):.4f}", f"{np.median(lv):.4f}", f"{np.std(lv):.4f}", len(lv), "越大越好"])

    print("=" * 70)
    print("OCR (chars=识别字符数, lines=文本行数, conf=平均置信度)")
    print("=" * 70)
    if ocr_rows:
        for tag in ["original", "baseline", "uvdoc"]:
            ch = col(ocr_rows, f"{tag}_chars")
            ln = col(ocr_rows, f"{tag}_lines")
            cf = col(ocr_rows, f"{tag}_conf")
            print(f"{tag:10s} chars={np.mean(ch):.1f}  lines={np.mean(ln):.1f}  conf={np.mean(cf):.4f}  (n={len(ch)})")
            out_rows.append([f"ocr_chars_{tag}", f"{np.mean(ch):.1f}", f"{np.median(ch):.1f}", f"{np.std(ch):.1f}", len(ch), "越多越好"])
            out_rows.append([f"ocr_lines_{tag}", f"{np.mean(ln):.1f}", f"{np.median(ln):.1f}", f"{np.std(ln):.1f}", len(ln), "越多越好"])
            out_rows.append([f"ocr_conf_{tag}", f"{np.mean(cf):.4f}", f"{np.median(cf):.4f}", f"{np.std(cf):.4f}", len(cf), "越高越好"])

    out_path = os.path.join(EVAL, "summary_stats.csv")
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["metric", "mean", "median", "std", "n", "note"])
        w.writerows(out_rows)
    print("\n已保存:", out_path)


if __name__ == "__main__":
    main()
