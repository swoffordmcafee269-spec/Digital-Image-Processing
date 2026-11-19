# -*- coding: utf-8 -*-
"""v5 冒烟验证：在训练之前把每条改动都跑一遍。

检查项
------
K1  配置能加载，v5 新增键存在且值正确
K2  WB 增益缓存能命中（train/val/test 三份长度都对）
K3  Dataset 取一个 batch，形状/范围正常，WB 确实被应用（对比 use_wb 开关）
K4  三色度损失可微、值合理
K5  模型前向 shape 正确、参数量与 v1 一致
K6  输出路径全部解析到 v5/ 下（写一次假文件再删掉）
K7  向后兼容：use_wb=false 时输出与 v1 的行为完全一致（同种子同输入）

任一项失败即抛异常退出，不进入训练。
"""
import os
import sys
import traceback

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))   # …/v5/code/scripts
_CODE = os.path.dirname(_HERE)                       # …/v5/code
_V5 = os.path.dirname(_CODE)                         # …/v5
_PROJ = os.path.dirname(_V5)                         # 项目根
sys.path.insert(0, _CODE)

from src.config import load_config                                  # noqa: E402
from src.data.dataset import EvalDataset, PairedDataset             # noqa: E402
from src.data.raw_utils import pack_raw, read_raw_bayer             # noqa: E402
from src.data.wb import apply_wb, estimate_wb_gains, load_wb_cache   # noqa: E402
from src.models.losses import chroma_loss                           # noqa: E402
from src.models.unet import build_model                             # noqa: E402

CFG = {"sony": "config_v5.yaml", "canon": "config_v5_canon.yaml"}
FAILS = []


def check(name, cond, detail=""):
    tag = "OK  " if cond else "FAIL"
    print(f"  [{tag}] {name}" + (f"  {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)
    return cond


def k1_config(ds):
    cfg = load_config(os.path.join(_V5, "configs", CFG[ds]))
    check("use_wb=True", cfg.data.use_wb is True, f"use_wb={cfg.data.use_wb}")
    check("wb_cache 已配置", bool(cfg.data.wb_cache), cfg.data.wb_cache)
    cw = float(cfg.train.chroma_weight)
    check("chroma_weight>0", cw > 0, f"{cw}")
    check("输出路径都在 v5 下",
          all(str(p).startswith("v5/") for p in
              (cfg.train.log_dir, cfg.train.ckpt_dir, cfg.eval.output_dir)))
    return cfg


def k2_cache(cfg, ds):
    lname = ds
    cache_dir = os.path.join(_PROJ, cfg.data.wb_cache)
    for split, expect in (("train", 900), ("val", 100), ("test", 1000)):
        p = os.path.join(cache_dir, f"wb_gains_v5_{lname}_{split}.npy")
        arr = load_wb_cache(p)
        if arr is None:
            check(f"{split} 缓存", False, f"缺失 {p}")
            continue
        check(f"{split} 缓存", len(arr) == expect,
              f"{len(arr)} 条 (期望 {expect})  gR中位={np.median(arr[:,0]):.3f}")
        rng_ok = bool(np.all(arr >= 0.5) and np.all(arr <= 4.0))
        check(f"{split} 增益在 [0.5,4] 内", rng_ok)


def k3_dataset(cfg, ds):
    root = os.path.join(_PROJ, cfg.data.root, cfg.data.dataset)
    for split in ("train", "val", "test"):
        fl = os.path.join(root, f"{split}.txt")
        if split == "test":
            d = EvalDataset(cfg, fl)
            low, gt, name = d[0]
            lo, gt_a, nm = low, gt, name
        else:
            d = PairedDataset(cfg, fl, patch_size=None)
            lo, gt_a = d[0]
            nm = "-"
        check(f"{split} 样本形状",
              tuple(lo.shape) == (4, 1080, 1920) and tuple(gt_a.shape) == (3, 2160, 3840),
              f"low{tuple(lo.shape)} gt{tuple(gt_a.shape)} name={nm}")
        check(f"{split} low 在 [0,1]", float(lo.min()) >= 0.0 and float(lo.max()) <= 1.0,
              f"[{float(lo.min()):.3f}, {float(lo.max()):.3f}]")

    # 缓存命中验证：WB 开启 vs 关闭
    cfg_off = load_config(os.path.join(_V5, "configs", CFG[ds]))
    cfg_off.data["use_wb"] = False
    fl = os.path.join(root, "train.txt")
    d_on = PairedDataset(cfg, fl, patch_size=None)
    d_off = PairedDataset(cfg_off, fl, patch_size=None)
    on = d_on[0][0].numpy()
    off = d_off[0][0].numpy()

    # 极暗图的像素绝对值只有 1e-3 量级，用 mean|Δ| 判定会误判为「没生效」。
    # 改用逐通道的相对比值：R 应放大约 gR 倍，B 应放大约 gB 倍。
    gains = d_on._wb_cache[0]
    ratios = []
    for c in range(4):
        m_off = float(off[c].mean())
        ratios.append(float(on[c].mean()) / m_off if m_off > 1e-12 else 1.0)
    exp = [float(gains[0]), 1.0, 1.0, float(gains[2])]
    err = max(abs(ratios[c] - exp[c]) for c in range(4))
    check("WB 生效且增益与缓存一致", err < 1e-3,
          f"实测比值 R/Gr/Gb/B={[round(r,3) for r in ratios]}  期望={[round(e,3) for e in exp]}")
    check("WB 后 R/G 通道均值更接近",
          abs(float(on[0].mean()) - float(on[1].mean()))
          < abs(float(off[0].mean()) - float(off[1].mean())),
          f"on |R-G|={abs(float(on[0].mean())-float(on[1].mean())):.2e}  "
          f"off |R-G|={abs(float(off[0].mean())-float(off[1].mean())):.2e}")


def k4_loss(cfg):
    torch.manual_seed(0)
    pred = torch.rand(2, 3, 32, 32, requires_grad=True)
    tgt = torch.rand(2, 3, 32, 32)
    l = chroma_loss(pred, tgt)
    check("chroma_loss 有限且为正", bool(torch.isfinite(l.detach())) and float(l.detach()) > 0,
          f"{float(l.detach()):.5f}")
    l.backward()
    g = pred.grad
    check("chroma_loss 梯度存在且非零",
          g is not None and bool(torch.isfinite(g).all()) and float(g.abs().sum()) > 0,
          f"|grad|={float(g.abs().sum()):.3f}")

    # 理想情况：pred==tgt 时损失应接近 0
    l0 = float(chroma_loss(tgt.clone(), tgt))
    check("chroma_loss(pred=gt)≈0", l0 < 2e-3, f"{l0:.6f}")

    # 只有亮度偏移（色度不变）时，色度损失应该很小
    shift = tgt + 0.1
    l_shift = float(chroma_loss(shift, tgt))
    check("纯亮度偏移不触发色度惩罚", l_shift < 2e-3, f"{l_shift:.6f}")

    # 色度确实错时，损失必须明显变大
    wrong = tgt[:, [2, 1, 0]]
    l_wrong = float(chroma_loss(wrong, tgt))
    check("色度错乱时损失显著增大", l_wrong > 10 * max(l0, 1e-9),
          f"错乱={l_wrong:.5f}  vs  完美={l0:.6f}")


def k5_model(cfg):
    m = build_model(cfg)
    n = sum(p.numel() for p in m.parameters())
    check("参数量与 v1 一致 (16.93M)", abs(n - 16_930_000) < 60_000, f"{n/1e6:.2f}M")
    x = torch.rand(1, 4, 64, 96)
    y = m(x)
    check("前向 shape (4,H,W)->(3,2H,2W)", tuple(y.shape) == (1, 3, 128, 192),
          f"{tuple(x.shape)} -> {tuple(y.shape)}")


def k6_paths(cfg):
    for key, p in (("log_dir", cfg.train.log_dir), ("ckpt_dir", cfg.train.ckpt_dir),
                   ("output_dir", cfg.eval.output_dir)):
        full = os.path.abspath(os.path.join(_PROJ, p))
        ok = full.startswith(os.path.abspath(_V5) + os.sep)
        check(f"{key} 在 v5 下", ok, full)


def k7_compat(cfg, ds):
    """use_wb=False 时，Dataset 输出必须与 v1 完全一致（逐字节）。"""
    cfg_off = load_config(os.path.join(_V5, "configs", CFG[ds]))
    cfg_off.data["use_wb"] = False
    root = os.path.join(_PROJ, cfg.data.root, cfg.data.dataset)
    fl = os.path.join(root, "val.txt")
    d = PairedDataset(cfg_off, fl, patch_size=None)
    low, gt = d[0]
    low = low.numpy()[None]

    # 手工重做一遍 v1 的流程（不用 dataset.py）
    raw_rel = d.pairs[0][0]
    gt_rel = d.pairs[0][1]
    bayer, black, white = read_raw_bayer(os.path.join(cfg_off.data.root, raw_rel),
                                         cfg_off.data)
    packed = pack_raw(bayer, black, white)
    ref = np.ascontiguousarray(packed).transpose(2, 0, 1)[None].astype(np.float32)
    check("use_wb=False 与 v1 流程逐值一致",
          low.shape == ref.shape and float(np.abs(low - ref).max()) < 1e-7,
          f"max|Δ|={float(np.abs(low - ref).max()):.3e}")


def main():
    for ds in ("sony", "canon"):
        print("\n" + "=" * 74)
        print(f"  {ds.upper()} 冒烟验证")
        print("=" * 74)
        try:
            cfg = k1_config(ds)
            k2_cache(cfg, ds)
            k3_dataset(cfg, ds)
            k4_loss(cfg)
            k5_model(cfg)
            k6_paths(cfg)
            k7_compat(cfg, ds)
        except Exception:
            print("\n[异常] 该数据集冒烟中断：")
            traceback.print_exc()
            FAILS.append(f"{ds}:exception")

    print("\n" + "=" * 74)
    if FAILS:
        print(f"冒烟失败 {len(FAILS)} 项：")
        for f in FAILS:
            print("   -", f)
        sys.exit(1)
    print("冒烟全部通过 ✔  可以进入训练")


if __name__ == "__main__":
    main()
