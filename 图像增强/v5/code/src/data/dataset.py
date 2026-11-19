# -*- coding: utf-8 -*-
"""数据集：低光 RAW ↔ 正常曝光 GT 配对读取，训练带增强、评估整图。

filelist 每行格式：`<raw相对路径> <gt相对路径>`，路径相对 data.root。
由 scripts/make_filelist.py 自动生成，也可手工编写。

v5 变更
-------
- 新增白平衡前处理（`wb.use_wb`，默认关闭以保持向后兼容）
- 增益优先从 `.npy` 缓存按索引取；缓存缺失或长度不符时回退到实时计算
- 缓存与 filelist 行号严格一一对应，命中缓存时用 `index`，否则实时算
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .raw_utils import pack_raw, read_raw_bayer
from .transforms import random_crop_pair, random_flip, random_rotate90
from .wb import apply_wb, estimate_wb_gains, load_wb_cache

# 项目根：本文件在 v5/code/src/data/ → 上溯 3 级到 v5/ → 再 1 级到项目根
_HERE_DIR = os.path.dirname(os.path.abspath(__file__))     # …/v5/code/src/data
_V5_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE_DIR)))  # …/v5
_PROJ_ROOT = os.path.dirname(_V5_ROOT)                     # 项目根


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


def _wb_gains_for(cfg, packed, wb_cache, index, name_for_log=""):
    """取这张图的 WB 增益：优先缓存，缺失则实时估。

    返回 (gains or None)。use_wb 关闭时返回 None，调用方跳过前处理。
    """
    if not bool(cfg.data.get("use_wb", False)):
        return None
    if wb_cache is not None and 0 <= index < len(wb_cache):
        return wb_cache[index]
    gains, _, _, _ = estimate_wb_gains(packed)
    return gains


class _WBBase:
    """PairedDataset / EvalDataset 共用的 WB 缓存加载逻辑。

    缓存路径解析规则（`resolve_wb_cache`）
    ------------------------------------
    - `cfg.data.wb_cache` 是**目录** → 按 `<目录>/wb_gains_v5_<dataset小写>_<split>.npy`
      自动拼，split 取自 filelist 的文件名（train/val/test）。
      这样 train_ds 和 val_ds 各用各的缓存，索引不会错位 —— 这是必需的，
      因为 Dataset 的 index 是相对**自己的** filelist 从 0 开始的。
    - `cfg.data.wb_cache` 是**文件** → 直接用它（单清单场景，需自行保证长度一致）。
    - 为空 → 不回退缓存，全部实时计算。
    """

    def _init_wb(self, filelist_path):
        self.use_wb = bool(self.cfg.data.get("use_wb", False))
        self._wb_cache = None
        if not self.use_wb:
            return
        path = self.cfg.data.get("wb_cache", "") or ""
        if not path:
            return
        split = os.path.splitext(os.path.basename(filelist_path))[0]
        ds = str(self.cfg.data.get("dataset", "")).lower()
        # 相对路径可能相对「项目根」也可能相对 cwd，两处都试
        roots = ["", os.getcwd(), _PROJ_ROOT]
        cands = []
        for r in roots:
            base = path if os.path.isabs(path) else os.path.join(r, path) if r else path
            if not path.lower().endswith(".npy"):
                cands.append(os.path.join(base, f"wb_gains_v5_{ds}_{split}.npy"))
            else:
                cands.append(base)
        # 去重保序
        seen, uniq = set(), []
        for c in cands:
            a = os.path.abspath(c)
            if a not in seen:
                seen.add(a)
                uniq.append(c)

        self._wb_cache = None
        hit = None
        for c in uniq:
            arr = load_wb_cache(c)
            if arr is None:
                continue
            if len(arr) != len(self.pairs):
                print(f"[WB] 长度不符 {len(arr)}≠{len(self.pairs)}: {c}")
                continue
            self._wb_cache, hit = arr, c
            break
        if self._wb_cache is None:
            print(f"[WB] 缓存未命中，回退实时估计（首张图会慢）")
        else:
            print(f"[WB] 命中 {os.path.basename(hit)}  ({len(self.pairs)} 条)")


class PairedDataset(_WBBase, Dataset):
    """训练/验证用配对数据集（训练时随机裁剪增强）。"""

    def __init__(self, cfg, filelist_path, patch_size=None):
        self.cfg = cfg
        self.root = cfg.data.root
        self.patch_size = patch_size
        self.pairs = _read_pairs(filelist_path)
        self._init_wb(filelist_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        raw_rel, gt_rel = self.pairs[index]
        raw_path = os.path.join(self.root, raw_rel)
        gt_path = os.path.join(self.root, gt_rel)

        bayer, black, white = read_raw_bayer(raw_path, self.cfg.data)
        packed = pack_raw(bayer, black, white)          # (H/2, W/2, 4)

        gains = _wb_gains_for(self.cfg, packed, self._wb_cache, index)
        if gains is not None:
            packed = apply_wb(packed, gains)            # v5：白平衡前处理

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


class EvalDataset(_WBBase, Dataset):
    """评估/推理用数据集：整图输入，不做增强，返回文件名。"""

    def __init__(self, cfg, filelist_path):
        self.cfg = cfg
        self.root = cfg.data.root
        self.pairs = _read_pairs(filelist_path)
        self._init_wb(filelist_path)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        raw_rel, gt_rel = self.pairs[index]
        raw_path = os.path.join(self.root, raw_rel)
        gt_path = os.path.join(self.root, gt_rel)

        bayer, black, white = read_raw_bayer(raw_path, self.cfg.data)
        packed = pack_raw(bayer, black, white)

        gains = _wb_gains_for(self.cfg, packed, self._wb_cache, index)
        if gains is not None:
            packed = apply_wb(packed, gains)            # v5：白平衡前处理

        gt = _read_gt_rgb(gt_path).astype(np.float32) / 255.0
        gt = _resize_gt(gt, bayer.shape[0], bayer.shape[1])

        low_t = torch.from_numpy(np.ascontiguousarray(packed)).permute(2, 0, 1).float()
        gt_t = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1).float()

        name = os.path.splitext(os.path.basename(gt_rel))[0]
        return low_t, gt_t, name
