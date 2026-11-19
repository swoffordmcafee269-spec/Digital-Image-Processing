# -*- coding: utf-8 -*-
"""v5 新增：RAW 域白平衡（Gray-World）预校正。

背景
----
v1 直接把 packed RAW 送进网络，RAW 域里 R/G/B 的通道灵敏度差异（典型 R≈0.5G）
由网络隐式学习。但极暗场景下网络拿到的是「整体灰白」的输入，学不到稳定色度。

v5 在**数据层**显式做一次 Gray-World，把通道响应拉平，让网络专注于增强而不是色彩估计。

与 L0 探测的对应
----------------
`v5/reports/L0_探测报告.md` 实验 B 实测 900×2 张：
  - Sony  g_R 中位 1.933，p5~p95 = 0.724~2.564，裁剪率 2.2%
  - Canon g_R 中位 1.968，p5~p95 = 1.450~3.250，裁剪率 1.9%
增益全部落在 [0.5, 4.0] 内（100% 覆盖），裁剪只是安全兜底。

设计要点
--------
1. **只在 packed (H/2, W/2, 4) 上操作**，不碰 RAW 读取层 —— Bayer 顺序已由 L0 实验 A
   证伪「读错」，保持原样。
2. **Gr/Gb 共用同一个增益**，取两者中位数的均值。分开估会让 G 通道产生棋盘纹。
3. **去死黑 mask**：`sum(4ch) > black_thresh`。极暗图死黑像素占 8%~34%，不剔除会把
   中位数拉到 0，导致增益爆炸。
4. **剔高光**：按 99.5 分位截断。饱和像素会压低 R 的估计值，进而高估 g_R。
5. **分块中位数**：16×16 块，块间取中位数。比全局均值抗「大面积单色区域」干扰。
6. **增益裁剪 [0.5, 4.0]** 且记录 clipped 标志，便于事后筛异常图。

离线缓存
--------
900 张全量算一次约 2 分钟，缓存为 `.npy` (N, 3)。训练/评估时 `np.load` + 按索引取，
微秒级。缓存由 `scripts/build_wb_cache.py` 生成，训练/评估会在缺失时自动回退到实时计算。
"""
import numpy as np

# L0 实测确定的安全区间
WB_CLIP = (0.5, 4.0)
BLACK_THRESH = 1e-4
LO_PCT = 60.0
HI_PCT = 99.5
TILE = 16
STRIDE = 4          # 统计前抽稀（1080×1920 → 270×480 仍有 13 万活像素）


def _subsample(packed, live, stride=STRIDE):
    return packed[::stride, ::stride], live[::stride, ::stride]


def estimate_wb_gains(packed, black_thresh=BLACK_THRESH, lo_pct=LO_PCT,
                      hi_pct=HI_PCT, tile=TILE, clip=WB_CLIP, stride=STRIDE):
    """估计单张图的 Gray-World 增益。

    参数
    ----
    packed : (H, W, 4) float32，已去黑电平并归一化到 [0, 1]

    返回
    ----
    (gR, gG, gB, stats)
      gR/gG/gB : float，gG 恒为 1.0（以绿通道为基准）
      stats    : dict，含 live_ratio / clipped / raw_gR / raw_gB / n_blocks / reason
    """
    H, W, _ = packed.shape
    s = packed.sum(axis=2)
    live = s > black_thresh
    stats = {"live_ratio": float(live.mean()), "clipped": False}

    if live.sum() < 256:
        stats["reason"] = "all_black"
        return 1.0, 1.0, 1.0, stats

    # --- 第 1 层：全图逐通道中位数（抽稀后算），作为兜底 ---
    ps, ls = _subsample(packed, live, stride)
    ss = ps.sum(axis=2)
    hi_cut = float(np.percentile(ss[ls], hi_pct)) if ls.sum() > 64 else float(hi_pct)
    base = ls & (ss <= hi_cut)

    med = []
    for c in range(4):
        if base.sum() < 64:
            med.append(np.nan)
            continue
        vals = ps[:, :, c][base]
        lo = np.percentile(vals, lo_pct)
        hi = np.percentile(vals, 99.5)
        sel = vals[(vals >= lo) & (vals <= hi)]
        if sel.size < 32:
            sel = vals
        # 在 gamma 域取中位数：极暗图的线性域动态范围过窄，gamma 域更稳定
        med.append(float(np.median(np.power(np.clip(sel, 1e-6, 1.0), 1.0 / 2.2))))

    g_r, g_gr, g_gb, g_b = med
    if any(not np.isfinite(x) for x in med) or min(g_r, g_gr, g_gb, g_b) <= 0:
        stats["reason"] = "degenerate"
        return 1.0, 1.0, 1.0, stats

    g_g = (g_gr + g_gb) * 0.5           # Gr/Gb 显式取均值，避免棋盘纹
    r_r, r_b = g_g / g_r, g_g / g_b

    # --- 第 2 层：16×16 分块中位数（块内按 2 步长抽稀到 8×8=64 点）---
    nby, nbx = H // tile, W // tile
    if nby >= 4 and nbx >= 4:
        Hc, Wc = nby * tile, nbx * tile
        blk = packed[:Hc, :Wc].reshape(nby, tile, nbx, tile, 4)
        blk_l = live[:Hc, :Wc].reshape(nby, tile, nbx, tile)
        s2 = max(1, tile // 8)
        blk_s = blk[:, ::s2, :, ::s2, :]
        blk_ls = blk_l[:, ::s2, :, ::s2]
        keep = blk_ls.mean(axis=(1, 3)) >= 0.5      # 死黑过半的块丢弃

        if int(keep.sum()) >= 16:
            def _blk_median(ch):
                a = np.where(blk_ls, ch, np.nan)
                return np.nanmedian(a, axis=(1, 3))

            m_g = (blk_s[:, :, :, :, 1] + blk_s[:, :, :, :, 2]) * 0.5
            bmr = _blk_median(blk_s[:, :, :, :, 0])
            bmg = _blk_median(m_g)
            bmb = _blk_median(blk_s[:, :, :, :, 3])
            valid = keep & (bmr > 1e-6) & (bmg > 1e-6) & (bmb > 1e-6)
            valid &= np.isfinite(bmr) & np.isfinite(bmg) & np.isfinite(bmb)
            if int(valid.sum()) >= 16:
                r_r = float(np.median(bmg[valid] / bmr[valid]))
                r_b = float(np.median(bmg[valid] / bmb[valid]))
                stats["n_blocks"] = int(valid.sum())

    stats["raw_gR"] = float(r_r)
    stats["raw_gB"] = float(r_b)
    stats["clipped"] = not (clip[0] <= r_r <= clip[1] and clip[0] <= r_b <= clip[1])
    return float(np.clip(r_r, *clip)), 1.0, float(np.clip(r_b, *clip)), stats


def apply_wb(packed, gains):
    """把增益作用到 packed (H, W, 4) 上，返回同形状的新数组。

    顺序：R * gR, Gr * gG, Gb * gG, B * gB，最后 clip 到 [0, 1]。
    注意是「原地前处理」，之后网络看到的 low 已经白平衡过。
    """
    out = np.empty_like(packed)
    g_r, g_g, g_b = (float(x) for x in gains)
    out[:, :, 0] = packed[:, :, 0] * g_r
    out[:, :, 1] = packed[:, :, 1] * g_g
    out[:, :, 2] = packed[:, :, 2] * g_g
    out[:, :, 3] = packed[:, :, 3] * g_b
    np.clip(out, 0.0, 1.0, out=out)
    return out


def load_wb_cache(path):
    """读缓存 (N, 3) float32；不存在返回 None（由调用方回退到实时计算）。"""
    import os
    if not path or not os.path.exists(path):
        return None
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return None
    return arr.astype(np.float32)
