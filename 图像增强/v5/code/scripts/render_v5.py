# -*- coding: utf-8 -*-
"""只做出图，不算指标。

背景：`evaluate.py` 原本用 cv2.imwrite 写增强结果，在含中文的绝对路径下
会静默返回 False（不抛异常），导致评估跑完 1000 张、images/ 目录却是空的。
修好 evaluate.py 之后，指标已经在 results.csv 里了，没必要为了补图再把
LPIPS 重算一遍（LPIPS 占评估链路 92% 的时间）。本脚本只跑前向 + 写盘。

用法（项目根目录）：
    D:/Python文件夹/python.exe v5/code/scripts/render_v5.py sony
    D:/Python文件夹/python.exe v5/code/scripts/render_v5.py canon
"""
import os
import sys

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))          # …/v5/code/scripts
_CODE = os.path.dirname(_HERE)                              # …/v5/code
_V5 = os.path.dirname(_CODE)                                # …/v5
_PROJ = os.path.dirname(_V5)                                # 项目根
sys.path.insert(0, _CODE)

from src.config import load_config                          # noqa: E402
from src.data.dataset import EvalDataset                    # noqa: E402
from src.inference import run_model                         # noqa: E402
from src.models.unet import build_model                     # noqa: E402
from src.utils.logger import get_device, load_checkpoint    # noqa: E402

CFG_OF = {"sony": "config_v5.yaml", "canon": "config_v5_canon.yaml"}


def resolve_dir(p):
    return p if os.path.isabs(p) else os.path.join(_PROJ, p)


def imwrite_unicode(path, rgb_uint8, ext=".png"):
    """中文路径安全的写图（cv2.imwrite 会静默失败）。"""
    ok, buf = cv2.imencode(ext, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"编码失败: {path}")
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise RuntimeError(f"写出失败: {path}")


def main():
    tok = (sys.argv[1] if len(sys.argv) > 1 else "sony").strip().lower()
    if tok not in CFG_OF:
        print(f"未知数据集 {tok!r}，可选 sony / canon")
        sys.exit(1)
    ckpt_name = sys.argv[2] if len(sys.argv) > 2 else "best.pt"

    cfg_path = os.path.join(_V5, "configs", CFG_OF[tok])
    cfg = load_config(cfg_path)
    device = get_device(None)

    print(f"[cfg] dataset={cfg.data.dataset}  config={CFG_OF[tok]}")
    print(f"[cfg] ckpt={ckpt_name}  wb={cfg.data.get('use_wb')}")

    test_path = os.path.join(_PROJ, cfg.data.root, cfg.data.dataset, "test.txt")
    if not os.path.exists(test_path):
        print(f"[错误] 找不到 {test_path}")
        sys.exit(1)

    ds = EvalDataset(cfg, test_path)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    ckpt = os.path.join(_PROJ, cfg.train.ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt):
        print(f"[错误] 找不到 checkpoint: {ckpt}")
        sys.exit(1)
    model = build_model(cfg).to(device)
    load_checkpoint(ckpt, model)
    print(f"已加载: {ckpt}")

    out_root = resolve_dir(cfg.eval.output_dir)
    save_dir = os.path.join(out_root, cfg.data.dataset, "images")
    os.makedirs(save_dir, exist_ok=True)
    ext = ".png" if tok == "sony" else ".jpg"
    print(f"[出图] -> {save_dir}  (每张 {ext})")

    n = 0
    for low, _gt, name in tqdm(loader, desc="出图中"):
        name = name[0]
        out = run_model(model, low[0], device)
        img = (out * 255.0 + 0.5).clip(0, 255).astype(np.uint8)
        kwargs = {"ext": ".png"} if ext == ".png" else {}
        if ext == ".jpg":
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                   [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                raise RuntimeError(f"编码失败: {name}")
            with open(os.path.join(save_dir, name + ext), "wb") as f:
                f.write(buf.tobytes())
        else:
            imwrite_unicode(os.path.join(save_dir, name + ".png"), img)
        n += 1

    n_disk = len([f for f in os.listdir(save_dir) if f.endswith(ext)])
    print(f"\n[完成] 写出 {n} 张，磁盘 {n_disk} 张")
    # 体积统计
    total = sum(os.path.getsize(os.path.join(save_dir, f))
                for f in os.listdir(save_dir) if f.endswith(ext))
    print(f"[体积] 合计 {total / 1024 / 1024:.1f} MB，"
          f"平均 {total / max(n_disk, 1) / 1024 / 1024:.2f} MB/张")


if __name__ == "__main__":
    main()
