"""由评测结果生成 Markdown 总结文档 summary.md。

用法（在 venv_cai 中运行）：
    venv_cai/Scripts/python.exe generate_summary.py
"""

import csv
import os
from collections import defaultdict

import numpy as np

OUT_DIR = "results"


def _family(codec):
    if codec.startswith("jpeg-dct"):
        return "自研 JPEG (DCT)"
    if codec.startswith("avif"):
        return "AVIF (AV1)"
    return codec


def load_metrics():
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "metrics.csv"),
                                    newline="")))
    return rows


def aggregate(rows):
    agg = defaultdict(lambda: {"bpp": [], "psnr": [], "ssim": [],
                               "ratio": [], "t_enc": [], "t_dec": []})
    order = []
    for r in rows:
        c = r["codec"]
        if c not in agg:
            order.append(c)
        agg[c]["bpp"].append(float(r["bpp"]))
        agg[c]["psnr"].append(float(r["psnr"]))
        agg[c]["ssim"].append(float(r["ssim"]))
        agg[c]["ratio"].append(float(r["compression_ratio"]))
        agg[c]["t_enc"].append(float(r["t_enc_ms"]))
        agg[c]["t_dec"].append(float(r["t_dec_ms"]))
    return agg, order


def _q(codec):
    return int(codec.split("-q")[-1])


def build_summary(rows):
    agg, order = aggregate(rows)
    lines = []

    def w(s=""):
        lines.append(s)

    w("# 图像压缩实习报告 · 结果总结")
    w()
    w("> 本文件为评测结果汇总，供撰写正式实习报告时直接引用。")
    w()
    w("## 1. 任务与数据")
    w()
    w("- 数据集：Kodak Lossless True Color Image Suite，24 张 24bit 真彩色 RGB 图（768×512 / 512×768）。")
    w("- 任务：图像压缩算法设计与实现，对比压缩前后质量与压缩效率。")
    w("- 评价指标：压缩率（bpp）、PSNR、SSIM、压缩/解压缩时间。")
    w("- 基准口径：压缩率以原始像素量 `H×W×3` 字节为分母；bpp = 压缩总比特 / 像素数。")
    w()
    w("## 2. 实现方案")
    w()
    w("### 2.1 自研 JPEG 风格 DCT 编码器（对比基线）")
    w()
    w("- 颜色空间：RGB → YCbCr（4:4:4，无色度下采样）。")
    w("- 变换：8×8 分块 DCT（正交归一）。")
    w("- 量化：JPEG 标准量化表按质量因子 Q 缩放（Q∈[1,100]）。")
    w("- 熵编码：Zigzag 扫描 + DC 差分 + AC 游程(RLE) + 规范哈夫曼编码。")
    w()
    w("### 2.2 AVIF（AV1 帧内编码）")
    w()
    w("- AVIF 基于 AV1 视频编码器的帧内编码模式，采用更先进的块划分、")
    w("  多种变换核与更精细的环路滤波，率失真性能显著优于 JPEG。")
    w("- 通过 libavif（pillow-avif-plugin）调用，质量因子 Q 控制量化。")
    w()
    w("## 3. 结果汇总（24 张图均值）")
    w()
    w("| 方案 | bpp | PSNR(dB) | SSIM | 压缩率 | 编码(ms) | 解码(ms) |")
    w("|---|---|---|---|---|---|---|")
    for c in order:
        a = agg[c]
        w(f"| {c} | {np.mean(a['bpp']):.3f} | {np.mean(a['psnr']):.2f} | "
          f"{np.mean(a['ssim']):.5f} | {np.mean(a['ratio']):.1f} | "
          f"{np.mean(a['t_enc']):.1f} | {np.mean(a['t_dec']):.1f} |")
    w()
    w("## 4. 率失真对比（关键结论）")
    w()
    # AVIF 相对自研 JPEG 在相同 PSNR 下的码率节省
    jpeg = sorted([(np.mean(agg[c]['psnr']), np.mean(agg[c]['bpp']), c)
                   for c in order if c.startswith("jpeg-dct")])
    avif_pts = sorted([(np.mean(agg[c]['psnr']), np.mean(agg[c]['bpp']), c)
                       for c in order if c.startswith("avif")])
    for psnr_a, bpp_a, ca in avif_pts:
        psnr_j, bpp_j, cj = min(jpeg, key=lambda p: abs(p[0] - psnr_a))
        if abs(psnr_j - psnr_a) > 1.5:
            continue
        saving = (1 - bpp_a / bpp_j) * 100
        w(f"- PSNR 相当（AVIF {psnr_a:.1f}dB vs 自研JPEG {psnr_j:.1f}dB）时，"
          f"AVIF bpp={bpp_a:.3f} vs JPEG bpp={bpp_j:.3f}，节省约 {saving:.0f}% 码率。")
    w()
    w("- RD 曲线见 `results/rd_psnr.png` 与 `results/rd_ssim.png`；")
    w("  压缩前后对比图见 `results/compare_*.png`。")
    w()
    w("## 5. 结论（可直接引用的话术）")
    w()
    w("1. 自研 DCT 编码器完整实现了「变换 + 量化 + 熵编码」的有损压缩链路，")
    w("   随质量因子 Q 增大，bpp 与 PSNR/SSIM 单调上升，符合率失真基本规律。")
    w("2. AVIF（AV1 帧内编码）在相同 PSNR 下码率显著低于传统 DCT 编码，")
    w("   体现了先进编码标准（更精细的块划分、多种变换与环路滤波）在率失真性能上的优势。")
    w("3. AVIF 的编解码由底层 C 库（libavif）完成，速度快于纯 Python 实现的自研 DCT；")
    w("   这一速度差异主要来自实现语言，而非压缩算法本身的复杂度，报告撰写时应如实说明。")
    w()
    w("## 6. 复现方式")
    w()
    w("```bash")
    w("# 环境：Python 3.14 + numpy/scipy/Pillow + pillow-avif-plugin + scikit-image")
    w("python run_benchmark.py   # 批量评测 -> results/*.csv")
    w("python plot.py            # RD 曲线 + 对比图")
    w("python generate_summary.py # 生成本文档")
    w("```")
    w()

    return "\n".join(lines)


def main():
    rows = load_metrics()
    text = build_summary(rows)
    with open("summary.md", "w", encoding="utf-8") as f:
        f.write(text)
    print(f"summary.md 已生成（{len(text)} 字符）")


if __name__ == "__main__":
    main()
