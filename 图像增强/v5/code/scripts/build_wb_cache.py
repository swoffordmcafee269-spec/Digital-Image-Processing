# -*- coding: utf-8 -*-
"""构建 v5 的 WB 增益缓存。

对 train.txt / val.txt / test.txt 三个清单**依次拼接**生成一个 (N, 3) 的 .npy，
缓存行的顺序与各 Dataset 内部的 `pairs` 索引**必须严格一致** —— 这是唯一的正确性前提。

因为 Dataset 每次都从某个 filelist 的**第 0 行**开始编号，所以单个缓存文件
没有办法同时服务三个清单。这里的做法：

  为每个 filelist 生成**独立的**缓存文件
    wb_gains_v5_sony_train.npy   (900, 3)
    wb_gains_v5_sony_val.npy     ( 100, 3)
    wb_gains_v5_sony_test.npy    (1000, 3)

  配置里按用途分别指定：
    data.wb_cache        → 训练/验证时用 train 那份（Dataset 构造时传的都是同一个）
    eval.wb_cache        → 评估时用 test 那份

但 train.py 里 train_ds 和 val_ds 用同一个 cfg.data.wb_cache，两者的 pairs 索引
意义不同。为此 dataset.py 做了长度校验：缓存长度 ≠ 样本数就回退实时计算。
想要两个都命中缓存，需要按 `--filelist` 分别构建并在训练时切换——v5 采用更稳的做法：

**统一按 filelist 文件名自动选择缓存**（见 `resolve_wb_cache`），
调用方只需给出缓存目录，dataset 依据自身 filelist 的 stem 自动拼路径。
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))          # …/v5/code/scripts
_CODE = os.path.dirname(_HERE)                              # …/v5/code
_V5 = os.path.dirname(_CODE)                                # …/v5
_PROJ = os.path.dirname(_V5)                                # 项目根
sys.path.insert(0, _CODE)

from src.config import load_config                      # noqa: E402
from src.data.raw_utils import pack_raw, read_raw_bayer  # noqa: E402
from src.data.wb import estimate_wb_gains               # noqa: E402


def read_filelist(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                out.append(p[0])
    return out


def build_for(ds_name, cfg_path, split, out_dir):
    cfg = load_config(cfg_path)
    fl = os.path.join(cfg.data.root, cfg.data.dataset, f"{split}.txt")
    if not os.path.exists(fl):
        print(f"[跳过] {fl} 不存在")
        return None
    rels = read_filelist(fl)
    print(f"\n=== {ds_name} / {split} === {len(rels)} 张")

    gains = np.zeros((len(rels), 3), dtype=np.float32)
    rows = []
    t0 = time.time()
    for i, rel in enumerate(rels):
        raw_path = os.path.join(cfg.data.root, rel)
        bayer, black, white = read_raw_bayer(raw_path, cfg.data)
        packed = pack_raw(bayer, black, white)
        gR, gG, gB, st = estimate_wb_gains(packed)
        gains[i] = (gR, gG, gB)
        rows.append({
            "dataset": ds_name, "split": split,
            "image": os.path.splitext(os.path.basename(rel))[0],
            "gR": round(gR, 4), "gG": round(gG, 4), "gB": round(gB, 4),
            "raw_gR": round(st.get("raw_gR", gR), 4),
            "raw_gB": round(st.get("raw_gB", gB), 4),
            "live_ratio": round(st["live_ratio"], 4),
            "clipped": int(st["clipped"]),
            "reason": st.get("reason", ""),
        })
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {i+1}/{len(rels)}  {el:.0f}s  剩余≈{el/(i+1)*(len(rels)-i-1):.0f}s")

    out = os.path.join(out_dir, f"wb_gains_v5_{ds_name.lower()}_{split}.npy")
    np.save(out, gains)
    print(f"[OK] {out}  shape={gains.shape}  用时 {time.time()-t0:.0f}s")

    n_clip = int(np.sum([r["clipped"] for r in rows]))
    gR = gains[:, 0]
    gB = gains[:, 2]
    print(f"  gR p5/p50/p95 = {np.percentile(gR,5):.3f} / {np.median(gR):.3f} / "
          f"{np.percentile(gR,95):.3f}")
    print(f"  gB p5/p50/p95 = {np.percentile(gB,5):.3f} / {np.median(gB):.3f} / "
          f"{np.percentile(gB,95):.3f}")
    print(f"  裁剪 {n_clip}/{len(rows)} ({n_clip/max(1,len(rows)):.1%})")
    return rows


def main():
    ap = argparse.ArgumentParser("构建 v5 WB 增益缓存")
    ap.add_argument("--dataset", choices=["sony", "canon", "both"], default="both")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--out", default=os.path.join(_V5, "cache"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg_map = {
        "sony": ("Sony", os.path.join(_V5, "configs", "config_v5.yaml")),
        "canon": ("Canon", os.path.join(_V5, "configs", "config_v5_canon.yaml")),
    }
    targets = ["sony", "canon"] if args.dataset == "both" else [args.dataset]
    all_rows = []
    for key in targets:
        ds_name, cfg_path = cfg_map[key]
        for split in args.splits.split(","):
            split = split.strip()
            if not split:
                continue
            rows = build_for(ds_name, cfg_path, split, args.out)
            if rows:
                all_rows += rows

    if all_rows:
        p = os.path.join(_V5, "probe", "wb_cache_stats.csv")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[OK] 统计明细 {p}")


if __name__ == "__main__":
    main()
