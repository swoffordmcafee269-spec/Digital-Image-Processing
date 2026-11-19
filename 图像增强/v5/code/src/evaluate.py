# -*- coding: utf-8 -*-
"""全量评估入口：对测试集计算 PSNR/SSIM/LPIPS，输出 CSV 与均值。

用法（项目根目录）：
    python src/evaluate.py
    python src/evaluate.py --set eval.run_baseline=false

v5 变更
-------
`eval.output_dir` 若为**相对路径**，会相对「项目根（= v5/ 的上一级）」解析，
而不是当前工作目录。这样无论从哪里启动，产物都稳定落在 v5/results/ 下，
不会因为 cwd 不同而误写进项目根的 outputs/。
"""
import csv
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))              # …/v5/code/src
_V5_ROOT = os.path.dirname(os.path.dirname(_SRC_DIR))              # …/v5
_PROJ_ROOT = os.path.dirname(_V5_ROOT)                             # 项目根
sys.path.insert(0, os.path.dirname(_SRC_DIR))

from src.config import load_config, parse_args  # noqa: E402
from src.data.dataset import EvalDataset  # noqa: E402
from src.baselines.traditional import enhance_baseline  # noqa: E402
from src.inference import run_model  # noqa: E402
from src.models.unet import build_model  # noqa: E402
from src.utils.logger import get_device, load_checkpoint  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402


def resolve_dir(p):
    """相对路径按项目根解析，绝对路径原样返回。"""
    return p if os.path.isabs(p) else os.path.join(_PROJ_ROOT, p)


def tensor_to_rgb(t):
    return t.permute(1, 2, 0).numpy()


def write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(columns)
        for r in rows:
            w.writerow(r)


def imwrite_unicode(path, rgb_uint8):
    """写图到可能含中文的路径。

    cv2.imwrite 用的是 ANSI 编码，在非 ASCII **绝对路径**下会直接返回 False
    且不抛异常（相对路径反而能成功，因为 cwd 已经解析过编码），这是本仓库
    踩过的坑：评估跑完 1000 张，images/ 目录空空如也却没有任何报错。
    改用 imencode 拿内存缓冲，再用 Python 的 open 写盘，彻底绕开该问题。
    """
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"PNG 编码失败: {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    # 复核：确保文件真的落盘，避免再次静默失败
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"图片写出失败（路径可能不可写）: {path}")


def mean_of(rows, col_idx):
    vals = [r[col_idx] for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    args = parse_args("路线B 全量评估")
    cfg = load_config(args.config, args.set)
    device = get_device(args.device)

    out_root = resolve_dir(cfg.eval.output_dir)
    print(f"[v5] 结果输出根目录: {out_root}")
    print(f"[v5] WB={'on' if cfg.data.get('use_wb', False) else 'off'}  "
          f"cache={cfg.data.get('wb_cache', '')}")

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
    save_dir = os.path.join(out_root, cfg.data.dataset, "images")
    os.makedirs(save_dir, exist_ok=True)
    print(f"[v5] 出图目录: {save_dir}  (save_images={cfg.eval.save_images})")
    n_saved = 0

    for i, (low, gt, name) in enumerate(tqdm(loader, desc="评估中")):
        name = name[0]
        gt_rgb = tensor_to_rgb(gt[0])

        # 模型
        out = run_model(model, low[0], device)
        m = compute_metrics(out, gt_rgb, metrics_names, cfg.eval.lpips_net)
        model_rows.append([name] + [round(m.get(k, 0), 6) for k in metrics_names])

        if cfg.eval.save_images:
            imwrite_unicode(
                os.path.join(save_dir, name + ".png"),
                (out * 255 + 0.5).clip(0, 255).astype(np.uint8))
            n_saved += 1

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
    out_csv = os.path.join(out_root, cfg.data.dataset, "results.csv")
    write_csv(out_csv, model_rows, columns)
    print(f"\n[模型] 结果已保存: {out_csv}")
    print(f"  均值: " + ", ".join(
        f"{k}={mean_of(model_rows, i + 1):.4f}" for i, k in enumerate(metrics_names)))

    if cfg.eval.save_images:
        n_disk = len([f for f in os.listdir(save_dir) if f.endswith(".png")])
        print(f"[出图] 已写 {n_saved} 张，磁盘实存 {n_disk} 张 -> {save_dir}")
        if n_disk < n_saved:
            print(f"[警告] 磁盘图片数少于应写数，可能存在写出失败")

    if cfg.eval.run_baseline:
        base_csv = os.path.join(out_root, cfg.data.dataset, "baseline.csv")
        write_csv(base_csv, baseline_rows, columns)
        print(f"[基线({cfg.eval.baseline_method})] 结果已保存: {base_csv}")
        print(f"  均值: " + ", ".join(
            f"{k}={mean_of(baseline_rows, i + 1):.4f}" for i, k in enumerate(metrics_names)))


if __name__ == "__main__":
    main()
