# -*- coding: utf-8 -*-
"""数据集：低光 RAW ↔ 正常曝光 GT 配对读取，训练带增强、评估整图。

filelist 每行格式：`<raw相对路径> <gt相对路径>`，路径相对 data.root。
由 scripts/make_filelist.py 自动生成，也可手工编写。
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .raw_utils import pack_raw, read_raw_bayer
from .transforms import random_crop_pair, random_flip, random_rotate90


def _read_gt_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"无法读取 GT 图像: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_gt(gt, target_h, target_w):
    """把 GT 对齐到模型输出分辨率（packed 的 2 倍 = RAW 原尺寸）。"""
    if gt.shape[0] != target_h or gt.shape[1] != target_w:
        gt = cv2.resize(gt, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return gt


def _read_pairs(filelist_path):
    pairs = []
    with open(filelist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"filelist 行格式错误（应为 'raw gt'）: {line}")
            pairs.append((parts[0], parts[1]))
    return pairs


class PairedDataset(Dataset):
    """训练/验证用配对数据集（训练时随机裁剪增强）。"""

    def __init__(self, cfg, filelist_path, patch_size=None):
        self.cfg = cfg
        self.root = cfg.data.root
        self.patch_size = patch_size
        self.pairs = _read_pairs(filelist_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        raw_rel, gt_rel = self.pairs[index]
        raw_path = os.path.join(self.root, raw_rel)
        gt_path = os.path.join(self.root, gt_rel)

        bayer, black, white = read_raw_bayer(raw_path, self.cfg.data)
        packed = pack_raw(bayer, black, white)          # (H/2, W/2, 4)

        gt = _read_gt_rgb(gt_path).astype(np.float32) / 255.0
        # 对齐到 RAW 原尺寸（= packed 的 2 倍）
        gt = _resize_gt(gt, bayer.shape[0], bayer.shape[1])

        if self.patch_size is not None:
            packed, gt = random_crop_pair(packed, gt, self.patch_size, scale=2)
            packed, gt = random_flip(packed, gt)
            packed, gt = random_rotate90(packed, gt)

        low_t = torch.from_numpy(np.ascontiguousarray(packed)).permute(2, 0, 1).float()
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).float()
        return low_t, gt_t


class EvalDataset(Dataset):
    """评估/推理用数据集：整图输入，不做增强，返回文件名。"""

    def __init__(self, cfg, filelist_path):
        self.cfg = cfg
        self.root = cfg.data.root
        self.pairs = _read_pairs(filelist_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        raw_rel, gt_rel = self.pairs[index]
        raw_path = os.path.join(self.root, raw_rel)
        gt_path = os.path.join(self.root, gt_rel)

        bayer, black, white = read_raw_bayer(raw_path, self.cfg.data)
        packed = pack_raw(bayer, black, white)

        gt = _read_gt_rgb(gt_path).astype(np.float32) / 255.0
        gt = _resize_gt(gt, bayer.shape[0], bayer.shape[1])

        low_t = torch.from_numpy(np.ascontiguousarray(packed)).permute(2, 0, 1).float()
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).float()

        name = os.path.splitext(os.path.basename(gt_rel))[0]
        return low_t, gt_t, name
