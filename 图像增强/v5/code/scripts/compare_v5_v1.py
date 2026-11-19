# -*- coding: utf-8 -*-
"""v5 vs v1 结果汇总。

读三处数据：
  1. v1 基线  outputs/results/{Sony,Canon}/results.csv
  2. v5 结果  v5/results/{sony,canon}/results.csv
  3. 训练日志 v5/logs/{sony,canon}/train_log.csv

产出：
  - 控制台逐项对比（含配对显著性检验）
  - v5/reports/v5_vs_v1_对比.md

统计口径：
  两批结果在**同一批 1000 张 test 图**上评估，属于配对样本。
  报告 均值差 / 配对 t 检验 p 值 / 改善/退化的图片张数，
  而不是只报两个均值 —— 均值差 0.1 dB 可能是噪声，也可能是稳定改善。
"""
import csv
import math
import os
import sys

_V5 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJ = os.path.dirname(_V5)

V1 = {
    "Sony":  {"psnr": 20.4460, "ssim": 0.7147, "lpips": 0.6526},
    "Canon": {"psnr": 18.8427, "ssim": 0.6948, "lpips": 0.5544},
}
# LPIPS 越低越好，其余越高越好
LOWER_BETTER = {"lpips"}
METRICS = ["psnr", "ssim", "lpips"]
CN = {"psnr": "PSNR(dB)", "ssim": "SSIM", "lpips": "LPIPS"}


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def paired_t(diffs):
    """配对 t 统计量（不是 p 值，够用来判断显著性）。"""
    n = len(diffs)
    if n < 2:
        return 0.0
    m, s = mean(diffs), std(diffs)
    if s < 1e-12:
        return 0.0
    return m / (s / math.sqrt(n))


def approx_p(t, n):
    """大样本下用正态近似估双侧 p（n=1000 时足够准）。"""
    z = abs(t)
    # 正态尾概率的初等近似（Abramowitz-Stegun 26.2.17 的简化版）
    return max(0.0, 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))


def summarize(ds_v1, ds_v5, key, ds):
    v1 = [float(r[key]) for r in ds_v1]
    v5 = [float(r[key]) for r in ds_v5]
    by_name_v1 = {r["name"]: float(r[key]) for r in ds_v1}
    by_name_v5 = {r["name"]: float(r[key]) for r in ds_v5}
    common = sorted(set(by_name_v1) & set(by_name_v5))
    diffs = [by_name_v5[n] - by_name_v1[n] for n in common]
    lower_better = key in LOWER_BETTER

    t = paired_t(diffs)
    p = approx_p(t, len(diffs))
    better = sum(1 for d in diffs if (d < 0 if lower_better else d > 0))
    worse = sum(1 for d in diffs if (d > 0 if lower_better else d < 0))
    return {
        "dataset": ds,
        "metric": CN[key],
        "v1": mean(v1),
        "v5": mean(v5),
        "delta": mean(v5) - mean(v1),
        "delta_pct": (mean(v5) - mean(v1)) / abs(mean(v1)) * 100.0 if mean(v1) else 0.0,
        "n": len(common),
        "better": better,
        "worse": worse,
        "t": t,
        "p": p,
        "lower_better": lower_better,
    }


def read_train_log(ds):
    p = os.path.join(_V5, "logs", ds.lower(), "train_log.csv")
    return read_csv(p)


def main():
    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# v5 vs v1 结果对比\n")
    emit(f"评估集：Sony / Canon 各 1000 张 test（与 v1 同一批图，配对比较）\n")

    rows = []
    for ds in ("Sony", "Canon"):
        p1 = os.path.join(_PROJ, "outputs", "results", ds, "results.csv")
        p5 = os.path.join(_V5, "results", ds.lower(), "results.csv")
        a, b = read_csv(p1), read_csv(p5)
        if a is None or b is None:
            emit(f"## {ds}\n\n**结果缺失**：v1={os.path.exists(p1)} v5={os.path.exists(p5)}\n")
            continue
        emit(f"## {ds}\n")
        emit("| 指标 | v1 | v5 | 差值 | 相对 | 改善张数 | 退化张数 | 配对 t | p |")
        emit("|---|---|---|---|---|---|---|---|---|")
        for k in METRICS:
            s = summarize(a, b, k, ds)
            rows.append(s)
            emit(f"| {s['metric']} | {s['v1']:.4f} | {s['v5']:.4f} | "
                 f"{s['delta']:+.4f} | {s['delta_pct']:+.2f}% | "
                 f"{s['better']}/{s['n']} | {s['worse']}/{s['n']} | "
                 f"{s['t']:+.2f} | {s['p']:.2e} |")
        emit("")

        lg = read_train_log(ds)
        if lg:
            best = max(lg, key=lambda r: float(r["val_psnr"]))
            emit(f"训练：{len(lg)} epoch，验证集最优 PSNR {float(best['val_psnr']):.2f} "
                 f"@ ep{int(best['epoch'])}，对应 SSIM {float(best['val_ssim']):.4f}\n")

    # 判定表
    emit("## 判定\n")
    emit("| 数据集 | 指标 | 结论 |")
    emit("|---|---|---|")
    for s in rows:
        sig = "显著" if s["p"] < 0.01 else ("边缘" if s["p"] < 0.05 else "不显著")
        good = s["delta"] < 0 if s["lower_better"] else s["delta"] > 0
        verdict = "✔ 改善" if good else "✘ 退化"
        emit(f"| {s['dataset']} | {s['metric']} | {verdict}（{sig}, p={s['p']:.2e}）|")
    emit("")

    out = os.path.join(_V5, "reports", "v5_vs_v1_对比.md")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[OK] {out}")


if __name__ == "__main__":
    main()
