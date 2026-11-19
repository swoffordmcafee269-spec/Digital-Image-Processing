# -*- coding: utf-8 -*-
"""全量评估入口：对测试集计算 PSNR/SSIM/LPIPS，输出 CSV 与均值。

用法（项目根目录）：
    python src/evaluate.py
    python src/evaluate.py --set eval.run_baseline=false
"""
import csv
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, parse_args  # noqa: E402
from src.data.dataset import EvalDataset  # noqa: E402
from src.baselines.traditional import enhance_baseline  # noqa: E402
from src.inference import run_model  # noqa: E402
from src.models.unet import build_model  # noqa: E402
from src.utils.logger import get_device, load_checkpoint  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402


def tensor_to_rgb(t):
    return t.permute(1, 2, 0).numpy()


def write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow(r)


def mean_of(rows, col_idx):
    vals = [r[col_idx] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    args = parse_args("路线B 全量评估")
    cfg = load_config(args.config, args.set)
    device = get_device(args.device)

    test_path = os.path.join(cfg.data.root, cfg.data.dataset, "test.txt")
    if not os.path.exists(test_path):
        print(f"[错误] 未找到测试 filelist: {test_path}")
        print("请先执行: python scripts/make_filelist.py")
        sys.exit(1)

    ds = EvalDataset(cfg, test_path)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    metrics_names = cfg.eval.metrics

    ckpt = cfg.train.resume or os.path.join(cfg.train.ckpt_dir, "best.pt")
    if not os.path.exists(ckpt):
        print(f"[错误] 未找到 checkpoint: {ckpt}")
        print("请先训练，或用 --set train.resume=路径 指定。")
        sys.exit(1)
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt, model)
    print(f"已加载模型: {ckpt}")

    model_rows = []
    baseline_rows = []
    save_dir = os.path.join(cfg.eval.output_dir, cfg.data.dataset, "images")
    os.makedirs(save_dir, exist_ok=True)

    for i, (low, gt, name) in enumerate(tqdm(loader, desc="评估中")):
        name = name[0]
        gt_rgb = tensor_to_rgb(gt[0])

        # 模型
        out = run_model(model, low[0], device)
        m = compute_metrics(out, gt_rgb, metrics_names, cfg.eval.lpips_net)
        model_rows.append([name] + [round(m.get(k, 0), 6) for k in metrics_names])

        if cfg.eval.save_images:
            cv2.imwrite(os.path.join(save_dir, name + ".png"),
                        cv2.cvtColor((out * 255 + 0.5).clip(0, 255).astype(np.uint8),
                                     cv2.COLOR_RGB2BGR))

        # 传统基线
        if cfg.eval.run_baseline:
            packed = low[0].permute(1, 2, 0).numpy()
            base = enhance_baseline(packed, black=cfg.data.black_level,
                                    white=cfg.data.white_level,
                                    method=cfg.eval.baseline_method).astype(np.float32) / 255.0
            base = cv2.resize(base, (gt_rgb.shape[1], gt_rgb.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            b = compute_metrics(base, gt_rgb, metrics_names, cfg.eval.lpips_net)
            baseline_rows.append([name] + [round(b.get(k, 0), 6) for k in metrics_names])

    # 写出结果
    columns = ["name"] + metrics_names
    out_csv = os.path.join(cfg.eval.output_dir, cfg.data.dataset, "results.csv")
    write_csv(out_csv, model_rows, columns)
    print(f"\n[模型] 结果已保存: {out_csv}")
    print(f"  均值: " + ", ".join(
        f"{k}={mean_of(model_rows, i + 1):.4f}" for i, k in enumerate(metrics_names)))

    if cfg.eval.run_baseline:
        base_csv = os.path.join(cfg.eval.output_dir, cfg.data.dataset, "baseline.csv")
        write_csv(base_csv, baseline_rows, columns)
        print(f"[基线({cfg.eval.baseline_method})] 结果已保存: {base_csv}")
        print(f"  均值: " + ", ".join(
            f"{k}={mean_of(baseline_rows, i + 1):.4f}" for i, k in enumerate(metrics_names)))


if __name__ == "__main__":
    main()
