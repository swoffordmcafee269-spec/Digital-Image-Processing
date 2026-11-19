# -*- coding: utf-8 -*-
"""数据增强：随机裁剪、翻转、旋转（numpy 实现，GT 按 scale 倍对齐）。"""
import numpy as np


def random_crop_pair(low, gt, patch_size, scale=2):
    """从低光 packed RAW 随机裁剪 patch，并裁剪对应 GT 区域。

    low: (H, W, 4) packed RAW；gt: (scale*H, scale*W, 3)
    """
    h, w = low.shape[:2]
    ph = pw = patch_size
    x = np.random.randint(0, w - pw + 1)
    y = np.random.randint(0, h - ph + 1)
    low_p = low[y:y + ph, x:x + pw]
    gt_p = gt[y * scale:(y + ph) * scale, x * scale:(x + pw) * scale]
    return low_p, gt_p


def random_flip(low, gt):
    if np.random.rand() > 0.5:
        low = np.flip(low, axis=1)
        gt = np.flip(gt, axis=1)
    if np.random.rand() > 0.5:
        low = np.flip(low, axis=0)
        gt = np.flip(gt, axis=0)
    return low, gt


def random_rotate90(low, gt):
    k = np.random.randint(0, 4)
    if k:
        low = np.rot90(low, k)
        gt = np.rot90(gt, k)
    return low, gt
