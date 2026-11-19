# -*- coding: utf-8 -*-
"""传统方法基线（路线 A）：Gamma / CLAHE / Retinex / 直方图拉伸。

输入：Bayer RAW（uint16）或 packed RAW；输出：uint8 sRGB (H, W, 3)。
用于与深度学习结果对比，写入报告基线表。
"""
import cv2
import numpy as np

from ..data.raw_utils import pack_raw, simple_demosaic


def _raw_to_linear_rgb(raw_or_packed, black, white):
    """统一把 Bayer RAW 或 packed RAW 转为线性 RGB (H, W, 3)。"""
    arr = np.asarray(raw_or_packed)
    if arr.ndim == 2:  # Bayer
        packed = pack_raw(arr, black, white)
    else:              # 已经是 packed (H/2, W/2, 4)
        packed = arr.astype(np.float32)
    rgb = simple_demosaic(packed)
    return np.clip(rgb, 0.0, 1.0)


def _to_uint8(rgb):
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def enhance_gamma(rgb, gamma=2.2, gain=4.0):
    """伽马校正 + 曝光增益。"""
    out = np.clip(rgb * gain, 0.0, 1.0) ** (1.0 / gamma)
    return out


def enhance_clahe(rgb):
    """在亮度通道上做对比度受限自适应直方图均衡。"""
    lab = cv2.cvtColor(_to_uint8(rgb), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB).astype(np.float32) / 255.0


def _single_scale_retinex(img, sigma):
    log_img = np.log1p(np.clip(img, 0.0, None))
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return log_img - np.log1p(np.clip(blur, 0.0, None))


def enhance_retinex(rgb, sigmas=(15, 80, 250)):
    """多尺度 Retinex（MSR）。"""
    msr = np.zeros_like(rgb, dtype=np.float32)
    for s in sigmas:
        msr += _single_scale_retinex(rgb, s)
    msr /= len(sigmas)
    # 归一化到 [0, 1]
    lo, hi = np.percentile(msr, 1), np.percentile(msr, 99)
    return np.clip((msr - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def enhance_hist(rgb, low=0.01, high=0.99):
    """对比度拉伸（截断百分位）。"""
    lo = np.percentile(rgb, low * 100)
    hi = np.percentile(rgb, high * 100)
    return np.clip((rgb - lo) / (hi - lo + 1e-8), 0.0, 1.0)


_METHODS = {
    "gamma": lambda rgb: enhance_gamma(rgb),
    "clahe": lambda rgb: enhance_clahe(rgb),
    "retinex": lambda rgb: enhance_retinex(rgb),
    "hist": lambda rgb: enhance_hist(rgb),
}


def enhance_baseline(raw_or_packed, black=512, white=16383, method="clahe"):
    """端到端基线增强，返回 uint8 sRGB。"""
    rgb = _raw_to_linear_rgb(raw_or_packed, black, white)
    fn = _METHODS.get(method, _METHODS["clahe"])
    out = fn(rgb)
    # 最后统一做一次 gamma 提升观感
    out = np.clip(out, 0.0, 1.0) ** (1.0 / 2.2)
    return _to_uint8(out)
