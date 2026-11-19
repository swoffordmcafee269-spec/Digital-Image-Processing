# -*- coding: utf-8 -*-
"""仅导出增强结果图（不评估指标）。

用途：为测试集逐张生成模型增强输出图，跳过 PSNR/SSIM/LPIPS 计算，速度远快于 evaluate.py。
默认输出到 outputs/results/{dataset}/images/，与 evaluate.py 的 save_images 路径一致。
支持 JPG 有损压缩以节省磁盘（--quality 控制，默认 95）。

用法（项目根目录）：
    python scripts/export_images.py                                        # Sony, PNG
    python scripts/export_images.py --config configs/config_canon.yaml     # Canon
    python scripts/export_images.py --format jpg --quality 92             # 压缩存 JPG
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402
from src.data.dataset import EvalDataset  # noqa: E402
from src.inference import run_model  # noqa: E402
from src.models.unet import build_model  # noqa: E402
from src.utils.logger import get_device, load_checkpoint  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="仅导出增强结果图")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--set", action="append", default=None, metavar="k=v")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--format", choices=["png", "jpg"], default="png",
                        help="输出格式（jpg 为有损压缩，省磁盘）")
    parser.add_argument("--quality", type=int, default=95,
                        help="jpg 质量 1-100（仅 --format jpg 时生效）")
    parser.add_argument("--out", type=str, default=None,
                        help="输出目录（默认 outputs/results/{dataset}/images/）")
    a = parser.parse_args()

    cfg = load_config(a.config, a.set)
    device = get_device(a.device)

    test_path = os.path.join(cfg.data.root, cfg.data.dataset, "test.txt")
    if not os.path.exists(test_path):
        print(f"[错误] 未找到测试 filelist: {test_path}")
        sys.exit(1)

    ckpt = cfg.train.resume or os.path.join(cfg.train.ckpt_dir, "best.pt")
    if not os.path.exists(ckpt):
        print(f"[错误] 未找到 checkpoint: {ckpt}")
        sys.exit(1)
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt, model)
    print(f"已加载模型: {ckpt}")

    ds = EvalDataset(cfg, test_path)
    out_dir = a.out or os.path.join(cfg.eval.output_dir, cfg.data.dataset, "images")
    os.makedirs(out_dir, exist_ok=True)

    ext = a.format
    n = len(ds)
    print(f"数据集: {cfg.data.dataset} | 样本数: {n} | 输出: {out_dir} | 格式: {ext}")

    for i, (low, _, name) in enumerate(tqdm(ds, desc="出图中")):
        out = run_model(model, low, device)  # (H, W, 3) float [0,1]
        out_u8 = (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
        save_path = os.path.join(out_dir, f"{name}.{ext}")
        if ext == "jpg":
            cv2.imwrite(save_path, out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), a.quality])
        else:
            cv2.imwrite(save_path, out_bgr)

    print(f"\n完成，共 {n} 张，保存在 {out_dir}")


if __name__ == "__main__":
    main()
