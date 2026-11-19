# -*- coding: utf-8 -*-
"""RAW 图像处理工具：黑电平扣除、RGGB 打包、简单 demosaic、ISP。"""
import numpy as np


def remove_black_level(raw, black_level=512, white_level=16383):
    """黑电平扣除 + 归一化到 [0, 1]。"""
    raw = raw.astype(np.float32)
    return np.maximum(raw - black_level, 0.0) / (white_level - black_level)


def pack_raw(raw, black_level=512, white_level=16383):
    """把 (H, W) Bayer RAW 打包为 (H/2, W/2, 4)，通道顺序 R, Gr, Gb, B。"""
    im = remove_black_level(raw, black_level, white_level)
    im = np.expand_dims(im, axis=2)
    H, W = im.shape[0], im.shape[1]
    out = np.concatenate((
        im[0:H:2, 0:W:2, :],  # R
        im[0:H:2, 1:W:2, :],  # Gr
        im[1:H:2, 1:W:2, :],  # Gb
        im[1:H:2, 0:W:2, :],  # B
    ), axis=2)
    return out


def unpack_raw(packed):
    """(H/2, W/2, 4) -> (H, W) Bayer，pack_raw 的逆操作。"""
    H, W, _ = packed.shape
    raw = np.zeros((H * 2, W * 2), dtype=packed.dtype)
    raw[0::2, 0::2] = packed[:, :, 0]
    raw[0::2, 1::2] = packed[:, :, 1]
    raw[1::2, 1::2] = packed[:, :, 2]
    raw[1::2, 0::2] = packed[:, :, 3]
    return raw


def simple_demosaic(packed):
    """packed (H/2, W/2, 4) RGGB -> (H/2, W/2, 3) RGB（两个绿色通道取均值）。"""
    r = packed[:, :, 0]
    g = (packed[:, :, 1] + packed[:, :, 2]) * 0.5
    b = packed[:, :, 3]
    return np.stack([r, g, b], axis=-1)


def white_balance(rgb, gains=(1.0, 1.0, 1.0)):
    """RGB 白平衡，gains 为 (r, g, b) 系数。"""
    return rgb * np.asarray(gains, dtype=np.float32)


def gamma_correct(rgb, gamma=2.2):
    """gamma 校正（rgb 需在 [0, 1]）。"""
    return np.clip(rgb, 0.0, 1.0) ** (1.0 / gamma)


def raw_to_visual(raw, black_level=512, white_level=16383, gamma=2.2, gain=1.0):
    """把极暗 RAW 转成人眼可看的 sRGB（仅用于可视化输入，非训练流程）。"""
    packed = pack_raw(raw, black_level, white_level)
    rgb = simple_demosaic(packed)
    rgb = np.clip(rgb * gain, 0.0, 1.0)
    return np.clip(rgb ** (1.0 / gamma), 0.0, 1.0)


def read_raw_bayer(path, cfg):
    """读取 RAW，返回 (bayer_array, black_level, white_level)。

    支持 .raw 二进制（用配置中的分辨率）与 rawpy 可读格式（.arw/.cr2/.dng/.nef）。
    """
    import os
    ext = os.path.splitext(path)[1].lower()
    if ext == ".raw":
        arr = np.fromfile(path, dtype=np.uint16)
        expected = cfg.raw_height * cfg.raw_width
        if arr.size != expected:
            raise ValueError(
                f"RAW 文件 {path} 元素数 {arr.size} 与配置分辨率 "
                f"{cfg.raw_height}x{cfg.raw_width}={expected} 不符，请检查 config.yaml 的 raw_width/raw_height")
        arr = arr.reshape(cfg.raw_height, cfg.raw_width)
        return arr, int(cfg.black_level), int(cfg.white_level)
    else:
        import rawpy
        with rawpy.imread(path) as raw:
            arr = raw.raw_image_visible.copy()
            black = int(np.asarray(raw.black_level_per_channel).min())
            white = int(raw.white_level)
        return arr, black, white
