"""统一评测指标：bpp、压缩率、PSNR、SSIM、编解码时间。

口径（与报告一致）：
- bpp = 压缩总比特数 / 像素总数
- 压缩率 = 原始 RGB 像素字节数 / 压缩字节数（基准为 H*W*3 字节，非 PNG 文件大小）
- PSNR = 10*log10(255^2 / MSE)，MSE 按 RGB 三通道平均
- SSIM = skimage 单尺度 SSIM（data_range=255, channel_axis=-1）
- 时间 = time.perf_counter，encode / decode 分开测
"""

import time

import numpy as np


def psnr_rgb(a, b):
    """RGB 图像 PSNR（dB），a/b 为 uint8[H,W,3]。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_rgb(a, b):
    """RGB 图像单尺度 SSIM，a/b 为 uint8[H,W,3]。"""
    from skimage.metrics import structural_similarity
    return float(structural_similarity(
        np.asarray(a), np.asarray(b), channel_axis=-1, data_range=255
    ))


def bpp(data_bytes, h, w):
    return len(data_bytes) * 8.0 / (h * w)


def compression_ratio(img, data_bytes):
    """原始像素字节数 / 压缩字节数。"""
    return np.asarray(img).nbytes / len(data_bytes)


def time_encode(encoder, img, repeat=3):
    """返回 (平均耗时秒, 编码结果 bytes)。"""
    encoder(img)  # 预热
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = encoder(img)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


def time_decode(decoder, data, repeat=3):
    decoder(data)  # 预热
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = decoder(data)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out
