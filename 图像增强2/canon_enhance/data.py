from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .common import source_paths


def read_raw(path, height=2160, width=3840, mmap=True):
    expected = height * width * 2
    if Path(path).stat().st_size != expected:
        raise ValueError(f"Unexpected RAW size: {path}; expected {expected}")
    if mmap:
        return np.memmap(path, dtype="<u2", mode="r", shape=(height, width))
    return np.fromfile(path, dtype="<u2").reshape(height, width)


def pack_raw(raw, black, scale):
    if raw.ndim != 2 or raw.shape[0] % 2 or raw.shape[1] % 2:
        raise ValueError("Bayer input must have even height and width")
    # Physical parity order: top-left, top-right, bottom-left, bottom-right.
    packed = np.stack((raw[0::2, 0::2], raw[0::2, 1::2],
                       raw[1::2, 0::2], raw[1::2, 1::2]), axis=0).astype(np.float32)
    return np.maximum(packed - float(black), 0) / float(scale)


def unpack_raw(packed):
    c, h, w = packed.shape
    if c != 4:
        raise ValueError("Expected four parity channels")
    raw = np.empty((h * 2, w * 2), dtype=packed.dtype)
    raw[0::2, 0::2], raw[0::2, 1::2] = packed[0], packed[1]
    raw[1::2, 0::2], raw[1::2, 1::2] = packed[2], packed[3]
    return raw


def read_rgb(path):
    with Image.open(path) as im:
        if im.mode != "RGB":
            raise ValueError(f"Expected RGB image: {path}, got {im.mode}")
        return np.asarray(im).copy()


def make_context(packed):
    return cv2.resize(packed.transpose(1, 2, 0), (32, 32),
                      interpolation=cv2.INTER_AREA).transpose(2, 0, 1).copy()


def srgb_to_linear_np(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def linear_to_srgb_np(rgb):
    rgb = np.maximum(rgb, 0)
    return np.where(rgb <= 0.0031308, rgb * 12.92,
                    1.055 * np.maximum(rgb, 1e-8) ** (1 / 2.4) - 0.055)


class TrainDataset(Dataset):
    """Random aligned crops; input data always opened in read-only mode.

    Spatial flips are deliberately absent: flipping packed CFA channels as
    ordinary RGB loses subpixel phase alignment. Random crops remain exact.
    """

    def __init__(self, cfg, manifest, patch_size):
        self.cfg = cfg
        self.items = [r for r in manifest["items"] if r["split"] == "train"]
        self.scale = manifest["normalization"]["scale"]
        self.patch = patch_size
        self.contexts = np.load(Path(cfg["artifact_root"]) / "data" / "contexts.npy", mmap_mode="r")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        raw_path, gt_path = source_paths(self.cfg, item)
        raw = read_raw(raw_path, self.cfg["height"], self.cfg["width"])
        p = self.patch
        y = np.random.randint(0, raw.shape[0] // 2 - p + 1)
        x = np.random.randint(0, raw.shape[1] // 2 - p + 1)
        low = pack_raw(raw[2*y:2*(y+p), 2*x:2*(x+p)], self.cfg["black_level"], self.scale)
        cache = Path(self.cfg["artifact_root"]) / "cache" / "gt" / f"{item['id']}.npy"
        gt = np.load(cache, mmap_mode="r") if cache.exists() else read_rgb(gt_path)
        target = np.asarray(gt[2*y:2*(y+p), 2*x:2*(x+p)]).transpose(2, 0, 1).astype(np.float32) / 255
        context = np.asarray(self.contexts[item["index"]], dtype=np.float32).copy()
        return torch.from_numpy(low), torch.from_numpy(context), torch.from_numpy(target)


def load_full_pair(cfg, manifest, item):
    raw_path, gt_path = source_paths(cfg, item)
    raw = read_raw(raw_path, cfg["height"], cfg["width"])
    packed = pack_raw(raw, cfg["black_level"], manifest["normalization"]["scale"])
    gt = read_rgb(gt_path)
    return packed, make_context(packed), gt
