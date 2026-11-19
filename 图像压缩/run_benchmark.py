"""批量评测：对 Kodak 24 张图跑「自研 JPEG + AVIF」，输出指标 CSV。

用法（主环境，Python 3.14）：
    python run_benchmark.py
    python run_benchmark.py --jpeg-only   # 仅自研 JPEG
"""

import csv
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

import eval as E
from compressors import jpeg_dct
from compressors import avif

DATA_DIR = "Kodak-Lossless-True-Color-Image-Suite-master/Kodak-Lossless-True-Color-Image-Suite-master/PhotoCD_PCD0992"
OUT_DIR = "results"

JPEG_QUALITIES = [10, 30, 50, 75, 90]
AVIF_QUALITIES = [10, 30, 50, 70, 90]


def build_codecs(jpeg_only=False):
    """返回 [(name, encoder, decoder)]。encoder(img)->bytes, decoder(bytes)->img。"""
    codecs = []
    for q in JPEG_QUALITIES:
        codecs.append((
            f"jpeg-dct-q{q}",
            lambda img, q=q: jpeg_dct.encode(img, q),
            jpeg_dct.decode,
        ))
    if jpeg_only:
        return codecs
    for q in AVIF_QUALITIES:
        codecs.append((
            f"avif-q{q}",
            lambda img, q=q: avif.encode(img, q),
            avif.decode,
        ))
    return codecs


def main():
    jpeg_only = "--jpeg-only" in sys.argv
    os.makedirs(OUT_DIR, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
    print(f"共 {len(paths)} 张图")

    codecs = build_codecs(jpeg_only=jpeg_only)
    print(f"共 {len(codecs)} 个编解码器配置" + ("（仅 JPEG）" if jpeg_only else ""))

    fieldnames = ["image", "codec", "bpp", "psnr", "ssim",
                  "compression_ratio", "t_enc_ms", "t_dec_ms"]
    csv_path = os.path.join(OUT_DIR, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for path in paths:
            name = os.path.splitext(os.path.basename(path))[0]
            img = np.array(Image.open(path).convert("RGB"))
            h, w = img.shape[:2]
            print(f"[{name}] {w}x{h}", flush=True)
            for cname, enc, dec in codecs:
                t0 = time.perf_counter()
                data = enc(img)
                t_enc = (time.perf_counter() - t0) * 1000.0
                t0 = time.perf_counter()
                rec = dec(data)
                t_dec = (time.perf_counter() - t0) * 1000.0

                psnr = E.psnr_rgb(img, rec)
                ssim = E.ssim_rgb(img, rec)
                bpp = E.bpp(data, h, w)
                ratio = E.compression_ratio(img, data)

                writer.writerow({
                    "image": name,
                    "codec": cname,
                    "bpp": f"{bpp:.4f}",
                    "psnr": f"{psnr:.4f}",
                    "ssim": f"{ssim:.6f}",
                    "compression_ratio": f"{ratio:.2f}",
                    "t_enc_ms": f"{t_enc:.2f}",
                    "t_dec_ms": f"{t_dec:.2f}",
                })
        print(f"\n指标已写入 {csv_path}")

    # 汇总每个 codec 的平均指标
    summary_path = os.path.join(OUT_DIR, "summary.csv")
    rows = list(csv.DictReader(open(csv_path, newline="")))
    from collections import defaultdict
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

    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["codec", "avg_bpp", "avg_psnr", "avg_ssim",
                    "avg_ratio", "avg_t_enc_ms", "avg_t_dec_ms"])
        for c in order:
            a = agg[c]
            w.writerow([
                c,
                f"{np.mean(a['bpp']):.4f}",
                f"{np.mean(a['psnr']):.4f}",
                f"{np.mean(a['ssim']):.6f}",
                f"{np.mean(a['ratio']):.2f}",
                f"{np.mean(a['t_enc']):.2f}",
                f"{np.mean(a['t_dec']):.2f}",
            ])
    print(f"汇总已写入 {summary_path}")


if __name__ == "__main__":
    main()
