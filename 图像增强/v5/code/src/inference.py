# -*- coding: utf-8 -*-
"""推理工具：模型前向 + 命令行单张/批量增强。"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.data.raw_utils import pack_raw, read_raw_bayer, raw_to_visual  # noqa: E402
from src.models.unet import build_model  # noqa: E402
from src.utils.logger import get_device, load_checkpoint  # noqa: E402


def run_model(model, low_tensor, device):
    """输入 packed tensor (4,H,W)，返回 numpy (H,W,3) float [0,1]。"""
    model.eval()
    with torch.no_grad():
        out = model(low_tensor.unsqueeze(0).to(device))
    out = out[0].cpu().permute(1, 2, 0).numpy()
    return np.clip(out, 0.0, 1.0)


def enhance_raw(model, raw_path, cfg, device, save_visual=False):
    """对单个 RAW 文件做增强，返回 (sRGB_uint8, 输入可视化图)。"""
    bayer, black, white = read_raw_bayer(raw_path, cfg.data)
    packed = pack_raw(bayer, black, white)
    low_t = torch.from_numpy(np.ascontiguousarray(packed)).permute(2, 0, 1).float()
    out = run_model(model, low_t, device)
    out_uint8 = (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
    vis = raw_to_visual(bayer, black, white) if save_visual else None
    return out_uint8, vis


def main():
    parser = argparse.ArgumentParser(description="路线B 推理")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--set", action="append", default=None, metavar="k=v")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("input", nargs="?", help="RAW 文件或目录（缺省则对数据目录批量增强）")
    parser.add_argument("--out", default="outputs/enhanced", help="输出目录")
    a = parser.parse_args()

    cfg = load_config(a.config, a.set)
    device = get_device(a.device)

    ckpt = cfg.train.resume or os.path.join(cfg.train.ckpt_dir, "best.pt")
    if not os.path.exists(ckpt):
        print(f"[错误] 未找到 checkpoint: {ckpt}")
        print("请先训练，或通过 --set train.resume=路径 指定模型。")
        sys.exit(1)

    model = build_model(cfg).to(device)
    load_checkpoint(ckpt, model)
    print(f"已加载模型: {ckpt}")

    out_dir = a.out
    os.makedirs(out_dir, exist_ok=True)

    target = a.input or os.path.join(cfg.data.root, cfg.data.dataset, cfg.data.raw_subdir or "")
    raw_ext = "." + cfg.data.raw_ext.lstrip(".")
    if os.path.isdir(target):
        files = []
        for root, _, fs in os.walk(target):
            for f in fs:
                if f.lower().endswith(raw_ext):
                    files.append(os.path.join(root, f))
        print(f"共 {len(files)} 个 RAW 文件待增强")
        for p in files:
            out, _ = enhance_raw(model, p, cfg, device)
            name = os.path.splitext(os.path.basename(p))[0]
            cv2.imwrite(os.path.join(out_dir, name + ".png"),
                        cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        print(f"完成，结果保存在 {out_dir}")
    else:
        out, vis = enhance_raw(model, target, cfg, device, save_visual=True)
        name = os.path.splitext(os.path.basename(target))[0]
        cv2.imwrite(os.path.join(out_dir, name + ".png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if vis is not None:
            cv2.imwrite(os.path.join(out_dir, name + "_input.png"),
                        cv2.cvtColor((vis * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        print(f"已保存增强结果: {os.path.join(out_dir, name + '.png')}")


if __name__ == "__main__":
    main()
