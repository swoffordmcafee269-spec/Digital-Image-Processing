# -*- coding: utf-8 -*-
"""可视化：生成「输入(暗) | 路线B输出 | 传统基线 | GT」对比图 + 指标条形图。

用法（项目根目录）：
    python src/visualize.py
    python src/visualize.py --set eval.num_samples=6
"""
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, parse_args  # noqa: E402
from src.data.dataset import EvalDataset  # noqa: E402
from src.data.raw_utils import raw_to_visual, read_raw_bayer  # noqa: E402
from src.baselines.traditional import enhance_baseline  # noqa: E402
from src.inference import run_model  # noqa: E402
from src.models.unet import build_model  # noqa: E402
from src.utils.logger import get_device, load_checkpoint  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402


def main():
    args = parse_args("可视化对比")
    cfg = load_config(args.config, args.set)
    device = get_device(args.device)

    test_path = os.path.join(cfg.data.root, cfg.data.dataset, "test.txt")
    ds = EvalDataset(cfg, test_path)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ckpt = cfg.train.resume or os.path.join(cfg.train.ckpt_dir, "best.pt")
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt, model)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "SimSun", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    n = cfg.eval.num_samples
    titles = ["输入(暗)", "路线B输出", "传统基线", "GT"]
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))

    for i, (low, gt, name) in enumerate(loader):
        if i >= n:
            break
        name = name[0]
        raw_rel = ds.pairs[i][0]
        raw_path = os.path.join(ds.root, raw_rel)
        bayer, black, white = read_raw_bayer(raw_path, cfg.data)

        input_vis = raw_to_visual(bayer, black, white)
        out = run_model(model, low[0], device)
        gt_rgb = gt[0].permute(1, 2, 0).numpy()

        packed = low[0].permute(1, 2, 0).numpy()
        base = enhance_baseline(packed, black=black, white=white,
                                method=cfg.eval.baseline_method).astype(np.float32) / 255.0
        base = cv2.resize(base, (gt_rgb.shape[1], gt_rgb.shape[0]))

        imgs = [input_vis, out, base, gt_rgb]
        for j, img in enumerate(imgs):
            ax = axes[i, j]
            ax.imshow(np.clip(img, 0, 1))
            ax.axis("off")
            if i == 0:
                ax.set_title(titles[j])

    fig.tight_layout()
    out_path = os.path.join(cfg.eval.output_dir, cfg.data.dataset, "comparison.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"对比图已保存: {out_path}")


if __name__ == "__main__":
    main()
