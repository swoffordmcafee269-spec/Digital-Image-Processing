"""画 RD 曲线（PSNR-bpp / SSIM-bpp）与压缩前后对比图。

用法（在隔离环境 venv_cai 中运行，依赖 matplotlib）：
    venv_cai/Scripts/python.exe plot.py
"""

import csv
import glob
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# 中文字体（Windows）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

import eval as E
from compressors import jpeg_dct

OUT_DIR = "results"
DATA_DIR = "Kodak-Lossless-True-Color-Image-Suite-master/Kodak-Lossless-True-Color-Image-Suite-master/PhotoCD_PCD0992"


def _family(codec):
    """把 codec 名归并到算法家族，用于画同族曲线。"""
    if codec.startswith("jpeg-dct"):
        return "自研 JPEG (DCT)"
    if codec.startswith("avif"):
        return "AVIF (AV1 帧内编码)"
    return codec


def plot_rd_curves():
    rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "metrics.csv"),
                                    newline="")))
    # 按 codec 聚合平均 bpp/psnr/ssim
    agg = defaultdict(lambda: {"bpp": [], "psnr": [], "ssim": []})
    for r in rows:
        agg[r["codec"]]["bpp"].append(float(r["bpp"]))
        agg[r["codec"]]["psnr"].append(float(r["psnr"]))
        agg[r["codec"]]["ssim"].append(float(r["ssim"]))

    # 按家族画曲线：家族内按 quality 升序连线
    families = defaultdict(list)
    for c, v in agg.items():
        fam = _family(c)
        # quality 从名字里提取
        q = int(c.split("-q")[-1])
        families[fam].append((q, np.mean(v["bpp"]), np.mean(v["psnr"]),
                              np.mean(v["ssim"])))
    for fam in families:
        families[fam].sort(key=lambda x: x[0])

    colors = {
        "自研 JPEG (DCT)": "#185FA5",
        "AVIF (AV1 帧内编码)": "#D85A30",
    }
    markers = {
        "自研 JPEG (DCT)": "o",
        "AVIF (AV1 帧内编码)": "^",
    }

    # PSNR-bpp
    fig, ax = plt.subplots(figsize=(7, 5))
    for fam, pts in families.items():
        bpp = [p[1] for p in pts]
        psnr = [p[2] for p in pts]
        ax.plot(bpp, psnr, marker=markers.get(fam, "o"),
                color=colors.get(fam, "#333"), label=fam, linewidth=2)
    ax.set_xlabel("bpp (bit/pixel)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Kodak 数据集率失真曲线（PSNR-bpp，24 张均值）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rd_psnr.png"), dpi=150)
    plt.close(fig)

    # SSIM-bpp
    fig, ax = plt.subplots(figsize=(7, 5))
    for fam, pts in families.items():
        bpp = [p[1] for p in pts]
        ssim = [p[3] for p in pts]
        ax.plot(bpp, ssim, marker=markers.get(fam, "o"),
                color=colors.get(fam, "#333"), label=fam, linewidth=2)
    ax.set_xlabel("bpp (bit/pixel)")
    ax.set_ylabel("SSIM")
    ax.set_title("Kodak 数据集率失真曲线（SSIM-bpp，24 张均值）")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "rd_ssim.png"), dpi=150)
    plt.close(fig)
    print("RD 曲线已保存：rd_psnr.png / rd_ssim.png")


def plot_comparison(image_names=("01", "05", "23")):
    """对指定图画出 原图/自研JPEG/神经压缩 的对比，放大局部细节。"""
    import matplotlib.gridspec as gridspec
    paths = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(os.path.join(DATA_DIR, "*.png"))}

    for name in image_names:
        if name not in paths:
            continue
        img = np.array(Image.open(paths[name]).convert("RGB"))
        h, w = img.shape[:2]

        # 自研 JPEG（Q=50）
        jpeg_rec = jpeg_dct.decode(jpeg_dct.encode(img, 50))

        # AVIF（Q=50）
        avif_rec = None
        try:
            from compressors import avif
            avif_rec = avif.decode(avif.encode(img, 50))
        except Exception as e:
            print(f"AVIF 不可用：{e}")

        n = 3 if avif_rec is not None else 2
        fig = plt.figure(figsize=(4 * n, 4.6))
        gs = gridspec.GridSpec(2, n, height_ratios=[3, 1])
        images = [("原始", img, None), ("自研JPEG Q50", jpeg_rec, None)]
        if avif_rec is not None:
            images.append(("AVIF Q50", avif_rec, None))

        for i, (title, im, _) in enumerate(images):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(im)
            ax.set_title(title, fontsize=11)
            ax.axis("off")
            # 局部放大框（中心 160x160）
            cy, cx = h // 2, w // 2
            rect = plt.Rectangle((cx - 80, cy - 80), 160, 160,
                                 fill=False, edgecolor="red", lw=1.5)
            ax.add_patch(rect)
            ax2 = fig.add_subplot(gs[1, i])
            crop = im[cy - 80:cy + 80, cx - 80:cx + 80]
            ax2.imshow(crop)
            psnr = "—" if title == "原始" else f"{E.psnr_rgb(img, im):.2f} dB"
            ax2.set_title(f"局部放大 · PSNR {psnr}", fontsize=10)
            ax2.axis("off")

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"compare_{name}.png"), dpi=150)
        plt.close(fig)
    print(f"对比图已保存到 {OUT_DIR}/compare_*.png")


if __name__ == "__main__":
    plot_rd_curves()
    plot_comparison()
