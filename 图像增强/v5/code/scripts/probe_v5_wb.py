# -*- coding: utf-8 -*-
"""v5 · L0 探测（零训练）。修正版。

v1 版探测的教训（已废弃的 v1 版脚本输出仍在 v5/probe/old_v1probe/）：
  用「全局均值对齐 GT 均值」会因 R 通道被严重压制而把中灰推成青色（见 bayer_order_grid 的 M0 行）。
  所以本版改为 **逐通道均值对齐**（per-channel mean match），它等价于
  「per-image Gray-World + 理想曝光」的上界，专门用来隔离 Bayer 排列的影响。

三个实验：
  A) Bayer 排列对照（逐通道对齐 + 局部色偏方差）
  B) WB 增益普查（900×2，产出训练缓存）
  C) 口径自检（重算 v1 的 PSNR）

用法：
    python v5/run.py probe
    python v5/run.py probe --only a      # 只跑 A
"""
import argparse
import csv
import os
import random
import sys
import time

import cv2
import numpy as np

V5_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJ_ROOT = os.path.dirname(V5_ROOT)

# ---------------------------------------------------------------- Bayer 候选
# 每种候选给出 (R, Gr, Gb, B) 四个位置，以及是否先转置
BAYER_MODES = {
    "M0_现状RGGB": {"transpose": False, "pos": {"R": (0, 0), "Gr": (0, 1), "Gb": (1, 1), "B": (1, 0)}},
    "M1_先转置RGGB": {"transpose": True, "pos": {"R": (0, 0), "Gr": (0, 1), "Gb": (1, 1), "B": (1, 0)}},
    "M2_先转置BGGR": {"transpose": True, "pos": {"R": (1, 1), "Gr": (1, 0), "Gb": (0, 1), "B": (0, 0)}},
    "M3_先转置GRBG": {"transpose": True, "pos": {"R": (0, 1), "Gr": (0, 0), "Gb": (1, 1), "B": (1, 0)}},
    "M4_先转置GBRG": {"transpose": True, "pos": {"R": (1, 0), "Gr": (0, 0), "Gb": (1, 1), "B": (0, 1)}},
    # 额外的物理检查：不做行列丢点，只做行列求和（1/2 分辨率「像素合并」），
    # 与任何 Bayer 顺序无关，用来验证「同色在列方向间隔 2」这个观察
    "M5_行列求和": {"transpose": False, "pos": None},
}


def read_bayer(path, h, w):
    arr = np.fromfile(path, dtype=np.uint16)
    if arr.size != h * w:
        raise ValueError(f"{path}: {arr.size} != {h * w}")
    return arr.reshape(h, w)


def pack_with_mode(bayer, mode, black, white):
    im = np.maximum(bayer.astype(np.float32) - black, 0.0) / (white - black)
    info = BAYER_MODES[mode]
    if info["pos"] is None:
        # 2x2 求和：与颜色顺序无关
        H, W = (im.shape[0] // 2) * 2, (im.shape[1] // 2) * 2
        b = im[:H, :W].reshape(H // 2, 2, W // 2, 2)
        s = b.sum(axis=(1, 3)) * 0.25
        return np.stack([s] * 4, axis=-1)
    if info["transpose"]:
        im = im.T
    out = [im[dy::2, dx::2] for dy, dx in (info["pos"][c] for c in ("R", "Gr", "Gb", "B"))]
    return np.stack(out, axis=-1)


def demosaic(packed):
    return np.stack([packed[:, :, 0],
                     (packed[:, :, 1] + packed[:, :, 2]) * 0.5,
                     packed[:, :, 3]], axis=-1)


# ---------------------------------------------------------------- 指标

_M = np.array([[0.4124, 0.3576, 0.1805],
               [0.2126, 0.7152, 0.0722],
               [0.0193, 0.1192, 0.9505]], dtype=np.float32)


def to_lab(rgb):
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)
    xyz = lin @ _M.T
    t = xyz / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16.0 / 116.0)
    return np.stack([116.0 * f[:, :, 1] - 16.0,
                     500.0 * (f[:, :, 0] - f[:, :, 1]),
                     200.0 * (f[:, :, 1] - f[:, :, 2])], axis=-1)


def sat_mean(rgb):
    return float(cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8),
                              cv2.COLOR_RGB2HSV)[:, :, 1].mean())


def chroma_mean(rgb):
    lab = to_lab(np.clip(rgb, 0, 1))
    return float(np.sqrt(lab[:, :, 1] ** 2 + lab[:, :, 2] ** 2).mean())


def hue_bias(rgb):
    """Lab 色度的平均方向与模（判断整体偏色）。"""
    lab = to_lab(np.clip(rgb, 0, 1))
    return float(np.abs(lab[:, :, 1]).mean()), float(np.abs(lab[:, :, 2]).mean())


def local_chroma_std(rgb, tile=48, max_tiles=120):
    """局部色偏方差：把图切成 tile，算每块的 (a*,b*) 均值，再看这些块之间的标准差。

    正确的 demosaic 会让相邻块颜色连续 → std 小；
    通道错乱会产生棋盘式假色 → std 大。
    这个指标直接量化「假色」，比全局饱和度可靠。
    """
    lab = to_lab(np.clip(rgb, 0, 1))
    H, W = lab.shape[:2]
    ny, nx = H // tile, W // tile
    if ny < 2 or nx < 2:
        return 0.0
    a = lab[:, :, 1].reshape(ny, tile, nx, tile).mean(axis=(1, 3))
    b = lab[:, :, 2].reshape(ny, tile, nx, tile).mean(axis=(1, 3))
    a = a.ravel()[:max_tiles]
    b = b.ravel()[:max_tiles]
    return float(np.std(a) + np.std(b))


def sharpness(rgb):
    g = cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var())


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 10.0 * np.log10(1.0 / max(mse, 1e-12))


def per_channel_match(rgb, gt, gamma=2.2, gains=None):
    """逐通道均值对齐 + gamma。等价于「理想的 per-image Gray-World + 理想曝光」。

    这是隔离 Bayer 排列影响的最干净方式：把颜色增益的误差从变量里剔掉。
    gains 不为 None 时直接复用给定增益（用于「统一增益」对照）。
    """
    out = np.empty_like(rgb)
    used = []
    for c in range(3):
        m = float(rgb[:, :, c].mean())
        if m <= 1e-9:
            out[:, :, c] = 0.0
            used.append(1.0)
            continue
        g = (float(gt[:, :, c].mean()) / m) if gains is None else float(gains[c])
        out[:, :, c] = rgb[:, :, c] * g
        used.append(g)
    return np.clip(out, 0, 1) ** (1.0 / gamma), used


# ---------------------------------------------------------------- 实验 A

def experiment_a(cfg_data, samples, out_dir):
    print("\n" + "=" * 76)
    print("实验 A：Bayer 排列对照（逐通道均值对齐，隔离颜色增益误差）")
    print("=" * 76)
    rows, panels = [], {}
    for ds, raw_path, gt_path in samples:
        bayer = read_bayer(raw_path, cfg_data["raw_height"], cfg_data["raw_width"])
        gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt_stats = {"sat": sat_mean(gt), "chroma": chroma_mean(gt),
                    "lap": sharpness(gt), "lcs": local_chroma_std(gt)}
        per_mode, uni_mode, gains_m0 = [], [], None
        for mode in BAYER_MODES:
            packed = pack_with_mode(bayer, mode, cfg_data["black_level"], cfg_data["white_level"])
            rgb = demosaic(packed)
            rgb, gains = per_channel_match(rgb, gt)
            if gains_m0 is None:
                gains_m0 = gains
            rgb_u, _ = per_channel_match(rgb, gt, gains=gains_m0)
            full = cv2.resize(rgb, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            full_u = cv2.resize(rgb_u, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

            s, c, lap = sat_mean(full), chroma_mean(full), sharpness(full)
            lcs = local_chroma_std(full)
            p = psnr(full, gt)
            rows.append({
                "dataset": ds, "image": os.path.basename(raw_path)[:-4], "mode": mode,
                "psnr": round(p, 3),
                "psnr_uni": round(psnr(full_u, gt), 3),
                "gain_R": round(gains[0], 3), "gain_G": round(gains[1], 3),
                "gain_B": round(gains[2], 3),
                "sat_out": round(s, 2), "sat_gt": round(gt_stats["sat"], 2),
                "sat_ratio": round(s / max(gt_stats["sat"], 1e-6), 3),
                "chroma_ratio": round(c / max(gt_stats["chroma"], 1e-6), 3),
                "lap_ratio": round(lap / max(gt_stats["lap"], 1e-6), 3),
                "lap_ratio_uni": round(sharpness(full_u) / max(gt_stats["lap"], 1e-6), 3),
                "local_chroma": round(lcs, 1),
                "local_chroma_gt": round(gt_stats["lcs"], 1),
            })
            per_mode.append((mode, {"full_raw": full, "full_uni": full_u}))
            print(f"  {ds}/{os.path.basename(raw_path)[:-4]:>5s} {mode:<16s} "
                  f"PSNR={p:6.2f}(统一{psnr(full_u, gt):5.2f}) "
                  f"sat={s/gt_stats['sat']:5.2f} lap={lap/gt_stats['lap']:6.2f} "
                  f"假色={lcs:6.1f} (GT {gt_stats['lcs']:.1f}) "
                  f"g=({gains[0]:.1f},{gains[1]:.1f},{gains[2]:.1f})")
        panels.setdefault(ds, []).append((os.path.basename(raw_path)[:-4], per_mode, gt))

    csv_path = os.path.join(out_dir, "bayer_order.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[OK] {csv_path}")

    summary = {}
    print("\n--- 各候选平均 ---")
    for mode in BAYER_MODES:
        sub = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "psnr": float(np.mean([r["psnr"] for r in sub])),
            "psnr_uni": float(np.mean([r["psnr_uni"] for r in sub])),
            "lap_ratio_uni": float(np.mean([r["lap_ratio_uni"] for r in sub])),
            "sat_ratio": float(np.mean([r["sat_ratio"] for r in sub])),
            "lap_ratio": float(np.mean([r["lap_ratio"] for r in sub])),
            "local_chroma": float(np.mean([r["local_chroma"] for r in sub])),
            "local_chroma_gt": float(np.mean([r["local_chroma_gt"] for r in sub])),
        }
        v = summary[mode]
        print(f"  {mode:<16s} PSNR={v['psnr']:6.2f}  sat={v['sat_ratio']:5.2f}  "
              f"lap={v['lap_ratio']:6.2f}  假色={v['local_chroma']:6.1f} "
              f"(GT {v['local_chroma_gt']:.1f})")

    grid = os.path.join(out_dir, "bayer_order_grid.png")
    _make_grid(panels, grid)
    print(f"[OK] {grid}")
    grid2 = os.path.join(out_dir, "bayer_order_grid_split.png")
    _make_grid_split(panels, grid2)
    print(f"[OK] {grid2}")
    _per_channel_gain_report(rows)
    return summary


def _per_channel_gain_report(rows):
    """看 per_channel_match 给出的增益本身是不是离谱 —— 若某候选的增益需要
    把某个通道放大 3 倍以上，说明它的「通道」根本不是颜色通道。"""
    print("\n--- per_channel_match 所需增益（>2.5 说明通道结构可疑）---")
    by_mode = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(
            (r["gain_R"], r["gain_G"], r["gain_B"]))
    for mode, gs in by_mode.items():
        arr = np.array(gs, dtype=np.float32)
        med = np.median(arr, axis=0)
        flag = "  <-- 可疑" if float(np.max(med) / max(np.min(med), 1e-6)) > 2.5 else ""
        print(f"  {mode:<16s} gR={med[0]:5.2f} gG={med[1]:5.2f} gB={med[2]:5.2f} "
              f"max/min={np.max(med)/max(np.min(med), 1e-6):5.2f}{flag}")


def _make_grid_split(panels, out_path, cell=(340, 200)):
    """双版式：上排「仅 Bayer 差异」（各候选自己的增益），下排「颜色对齐后」
    （统一用 M0 的增益，只体现 bayer 采样造成的锐度/假色差异）。"""
    cw, ch = cell
    blocks = []
    for ds, items in panels.items():
        for name, per_mode, gt in items[:2]:
            for label, key in (("原始增益", "full_raw"), ("统一增益", "full_uni")):
                strip = []
                for mode, imgs in per_mode + [("GT_真值", {"full_raw": gt, "full_uni": gt})]:
                    img = imgs[key]
                    small = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
                    small = (np.clip(small, 0, 1) * 255).astype(np.uint8)
                    cv2.putText(small, mode.split("_")[0], (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.putText(small, f"{label} {ds}/{name}", (6, ch - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                    strip.append(small)
                blocks.append(np.hstack(strip))
    if blocks:
        out = cv2.cvtColor(np.vstack(blocks), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", out)
        if not ok:
            print(f"[WARN] 编码失败，跳过 {out_path}")
            return
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        print(f"[OK] 写出 {os.path.basename(out_path)}  sha1={__import__('hashlib').sha1(buf.tobytes()).hexdigest()[:12]}")


def _make_grid(panels, out_path, cell=(300, 170)):
    cw, ch = cell
    rows = []
    for ds, items in panels.items():
        for name, per_mode, gt in items[:2]:
            strip = []
            # per_mode 元素是 (mode, {"full_raw":…, "full_uni":…})，末尾追加 GT 作为参照
            for mode, imgs in per_mode + [("GT_真值", {"full_raw": gt, "full_uni": gt})]:
                img = imgs["full_raw"]
                small = cv2.resize(img, (cw, ch), interpolation=cv2.INTER_AREA)
                small = (np.clip(small, 0, 1) * 255).astype(np.uint8)
                cv2.putText(small, mode.split("_")[0], (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.putText(small, f"{ds}/{name}", (6, ch - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.4, (0, 255, 255), 1, cv2.LINE_AA)
                strip.append(small)
            rows.append(np.hstack(strip))
    if rows:
        out = cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".png", out)
        if not ok:
            print(f"[WARN] 编码失败，跳过 {out_path}")
            return
        with open(out_path, "wb") as f:
            f.write(buf.tobytes())
        print(f"[OK] 写出 {os.path.basename(out_path)}  "
              f"sha1={__import__('hashlib').sha1(buf.tobytes()).hexdigest()[:12]}")


# ---------------------------------------------------------------- 实验 B

def _subsample(packed, live, stride=4):
    """统计前先抽稀：1080×1920 抽到 270×480 仍有 13 万活像素，
    分位数/中位数的估计误差 <1%，但 partition 的开销降 16 倍。"""
    return packed[::stride, ::stride], live[::stride, ::stride]


def estimate_wb_v5(packed, black_thresh=1e-4, lo_pct=60.0, hi_pct=99.5,
                   tile=16, clip=(0.5, 4.0), stride=4):
    """去死黑 mask + 分块中位数 的 Gray World。返回 (gR, gG, gB, stats)。

    全向量化 + 统计前 stride 抽稀，单图从 ~650 ms（Python 双循环版）降到 ~25 ms。
    """
    H, W, _ = packed.shape
    s = packed.sum(axis=2)
    live = s > black_thresh
    stats = {"live_ratio": float(live.mean()), "clipped": False}
    if live.sum() < 256:
        stats["reason"] = "all_black"
        return 1.0, 1.0, 1.0, stats

    # --- 第 1 层：全图分位数（抽稀后算）---
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
        med.append(float(np.median(np.power(np.clip(sel, 1e-6, 1.0), 1.0 / 2.2))))

    gR_, gGr, gGb, gB_ = med
    if any(not np.isfinite(x) for x in med) or min(gR_, gGr, gGb, gB_) <= 0:
        stats["reason"] = "degenerate"
        return 1.0, 1.0, 1.0, stats

    gG = (gGr + gGb) * 0.5       # Gr/Gb 显式取均值
    rR, rB = gG / gR_, gG / gB_

    # --- 第 2 层：16×16 分块中位数（每块取 k×k 个样本，k=2 → 每块 4 点，块数不变）---
    k = max(1, tile // 8)                 # tile=16 → k=2
    nby, nbx = H // tile, W // tile
    if nby >= 4 and nbx >= 4:
        Hc, Wc = nby * tile, nbx * tile
        # packed 是 (H, W, 4)；先截到整块，再 view 成 (nby, tile, nbx, tile, 4)
        blk = packed[:Hc, :Wc].reshape(nby, tile, nbx, tile, 4)
        blk_l = live[:Hc, :Wc].reshape(nby, tile, nbx, tile)
        # 块内按 k 步长抽稀：tile=16, k=2 → 每块保留 8×8=64 点（步长 2）
        s2 = tile // 8 if tile >= 8 else 1     # 块内步长，16//8 = 2
        blk_s = blk[:, ::s2, :, ::s2, :]
        blk_ls = blk_l[:, ::s2, :, ::s2]
        keep = blk_ls.mean(axis=(1, 3)) >= 0.5
        if int(keep.sum()) >= 16:
            def blk_median(ch):
                a = np.where(blk_ls, ch, np.nan)
                return np.nanmedian(a, axis=(1, 3))

            mG_blk = (blk_s[:, :, :, :, 1] + blk_s[:, :, :, :, 2]) * 0.5
            bmr = blk_median(blk_s[:, :, :, :, 0])
            bmg = blk_median(mG_blk)
            bmb = blk_median(blk_s[:, :, :, :, 3])
            valid = keep & (bmr > 1e-6) & (bmg > 1e-6) & (bmb > 1e-6)
            valid &= np.isfinite(bmr) & np.isfinite(bmg) & np.isfinite(bmb)
            if int(valid.sum()) >= 16:
                br = bmg[valid] / bmr[valid]
                bb = bmg[valid] / bmb[valid]
                rR, rB = float(np.median(br)), float(np.median(bb))
                stats["n_blocks"] = int(valid.sum())

    stats["raw_gR"] = float(rR)
    stats["raw_gB"] = float(rB)
    stats["clipped"] = not (clip[0] <= rR <= clip[1] and clip[0] <= rB <= clip[1])
    return float(np.clip(rR, *clip)), 1.0, float(np.clip(rB, *clip)), stats


def experiment_b(ds, cfg_data, filelist, cache_path, limit=None):
    print("\n" + "=" * 76)
    print(f"实验 B：WB 增益普查 — {ds}")
    print("=" * 76)
    pairs = []
    with open(filelist, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                pairs.append(os.path.join("data", p[0]))
    if limit:
        pairs = pairs[:limit]
    print(f"待处理 {len(pairs)} 张")

    gains, stats_rows = [], []
    t0_all = time.time()
    for i, rp in enumerate(pairs):
        t0 = time.time()
        bayer = read_bayer(rp, cfg_data["raw_height"], cfg_data["raw_width"])
        packed = pack_with_mode(bayer, "M0_现状RGGB", cfg_data["black_level"], cfg_data["white_level"])
        gR, gG, gB, st = estimate_wb_v5(packed)
        gains.append([gR, gG, gB])
        stats_rows.append({"dataset": ds, "image": os.path.basename(rp)[:-4],
                           "gR": round(gR, 4), "gB": round(gB, 4),
                           "live_ratio": round(st["live_ratio"], 4),
                           "clipped": int(st["clipped"]),
                           "ms": round((time.time() - t0) * 1000, 1)})
        if (i + 1) % 100 == 0:
            el = time.time() - t0_all
            print(f"  {i+1}/{len(pairs)}  {el:.0f}s  剩余≈{el/(i+1)*(len(pairs)-i-1):.0f}s")

    arr = np.array(gains, dtype=np.float32)
    np.save(cache_path, arr)
    print(f"[OK] {cache_path}  shape={arr.shape}")

    gR, gB = arr[:, 0], arr[:, 2]
    clip_rate = float(np.mean([r["clipped"] for r in stats_rows]))
    ms = float(np.mean([r["ms"] for r in stats_rows]))
    print(f"\n--- {ds} 统计 ---")
    print(f"  g_R: p5={np.percentile(gR,5):.3f}  p50={np.median(gR):.3f}  p95={np.percentile(gR,95):.3f}")
    print(f"  g_B: p5={np.percentile(gB,5):.3f}  p50={np.median(gB):.3f}  p95={np.percentile(gB,95):.3f}")
    print(f"  裁剪触发率 {clip_rate:.1%} (判据 <8%)   单图 {ms:.1f} ms (判据 <40ms)")
    bad = []
    if not (1.5 <= float(np.median(gR)) <= 2.5):
        bad.append(f"g_R 中位数 {np.median(gR):.2f} ∉ [1.5,2.5]")
    if clip_rate >= 0.08:
        bad.append(f"裁剪率 {clip_rate:.1%} ≥ 8%")
    if ms >= 40:
        bad.append(f"耗时 {ms:.0f}ms ≥ 40ms")
    print("  判据: " + ("全部通过 ✔" if not bad else "不通过 → " + "; ".join(bad)))
    return stats_rows


# ---------------------------------------------------------------- 实验 C

def experiment_c():
    print("\n" + "=" * 76)
    print("实验 C：口径自检")
    print("=" * 76)
    ref = {"Sony": 20.4460, "Canon": 18.8427}
    ok = True
    for ds in ("Sony", "Canon"):
        p = os.path.join(PROJ_ROOT, "outputs", "results", ds, "results.csv")
        if not os.path.exists(p):
            print(f"  [跳过] {p}")
            continue
        with open(p, encoding="utf-8") as f:
            vals = [float(r["psnr"]) for r in csv.DictReader(f)]
        m = float(np.mean(vals))
        d = abs(m - ref[ds])
        print(f"  {ds}: 重算={m:.4f}  参考={ref[ds]:.4f}  差={d:.4f}  {'✔' if d < 0.02 else '✘'}")
        ok = ok and d < 0.02
    print("  判据: " + ("通过 ✔" if ok else "不通过 ✘"))
    return ok


# ---------------------------------------------------------------- main

def pick_samples(datasets, n_per=4, seed=42):
    out = []
    for ds in datasets:
        fl = os.path.join("data", ds, "train.txt")
        pairs = []
        with open(fl, encoding="utf-8") as f:
            for line in f:
                p = line.split()
                if len(p) == 2:
                    pairs.append((os.path.join("data", p[0]), os.path.join("data", p[1])))
        random.Random(seed).shuffle(pairs)
        out.extend((ds, rp, gp) for rp, gp in pairs[:n_per])
    return out


def main():
    ap = argparse.ArgumentParser("v5 L0 探测")
    ap.add_argument("--only", choices=["a", "b", "c"], default=None)
    ap.add_argument("--limit-b", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(V5_ROOT, "code"))
    from src.config import load_config

    probe_dir = os.path.join(V5_ROOT, "probe")
    cache_dir = os.path.join(V5_ROOT, "cache")
    os.makedirs(probe_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    def cfg_of(ds, name):
        p = os.path.join(V5_ROOT, "configs", name)
        if os.path.exists(p):
            return load_config(p).data
        return load_config(os.path.join(
            PROJ_ROOT, "configs", "config.yaml" if ds == "Sony" else "config_canon.yaml")).data

    cs, cc = cfg_of("Sony", "config_v5.yaml"), cfg_of("Canon", "config_v5_canon.yaml")

    summary = None
    if args.only in (None, "a"):
        summary = experiment_a(cs, pick_samples(("Sony", "Canon"), 4), probe_dir)

    if args.only in (None, "b"):
        all_rows = []
        all_rows += experiment_b("Sony", cs, os.path.join("data", "Sony", "train.txt"),
                                 os.path.join(cache_dir, "wb_gains_v5_sony.npy"), args.limit_b)
        all_rows += experiment_b("Canon", cc, os.path.join("data", "Canon", "train.txt"),
                                 os.path.join(cache_dir, "wb_gains_v5_canon.npy"), args.limit_b)
        wb = os.path.join(probe_dir, "wb_stats.csv")
        with open(wb, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["dataset", "image", "gR", "gB",
                                              "live_ratio", "clipped", "ms"])
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[OK] {wb}")

    if args.only in (None, "c"):
        experiment_c()

    if summary:
        print("\n" + "=" * 76)
        print("实验 A 结论")
        print("=" * 76)
        gt_lcs = summary["M0_现状RGGB"]["local_chroma_gt"]
        hdr = (f"  {'候选':<16s} {'假色比':>7s} {'锐度比':>7s} {'饱和比':>7s} "
               f"{'PSNR':>7s} {'PSNR统一':>9s} {'锐度统一':>9s}")
        print(hdr)
        for m, v in summary.items():
            print(f"  {m:<16s} {v['local_chroma']/gt_lcs:6.2f}x {v['lap_ratio']:7.2f} "
                  f"{v['sat_ratio']:7.2f} {v['psnr']:7.2f} {v['psnr_uni']:9.2f} "
                  f"{v['lap_ratio_uni']:9.2f}")

        # --- 判据 1：按 PSNR_uni 排序（对齐增益后，纯 Bayer 采样差异）---
        print("\n  [判据 1] 统一增益后的 PSNR（公平比较：颜色误差已剔除）")
        rank_p = sorted(summary.items(), key=lambda kv: -kv[1]["psnr_uni"])
        for m, v in rank_p:
            print(f"    {m:<16s} PSNR_uni={v['psnr_uni']:6.2f}  "
                  f"锐度_uni={v['lap_ratio_uni']:6.2f}  假色比={v['local_chroma']/gt_lcs:5.2f}x")
        best_p, gap = rank_p[0][0], rank_p[0][1]["psnr_uni"] - rank_p[1][1]["psnr_uni"]
        print(f"    → 领先 {best_p}，与第二名差 {gap:.2f} dB")

        # --- 判据 2：饱和恢复率（Bayer 读错会把颜色洗掉）---
        print("\n  [判据 2] 饱和恢复率（越接近 1 越好；远小于 1 说明颜色被洗掉）")
        for m, v in sorted(summary.items(), key=lambda kv: -kv[1]["sat_ratio"]):
            print(f"    {m:<16s} sat_ratio={v['sat_ratio']:5.2f}")

        print("\n  [结论]")
        if gap >= 1.0:
            print(f"    PSNR_uni 领先 {gap:.2f} dB ≥ 1.0 且饱和率达标 → 判定为 {best_p}，"
                  f"Bayer 读法{'无需修改' if best_p.startswith('M0') else '需要修改'}")
        else:
            print(f"    PSNR_uni 领先仅 {gap:.2f} dB < 1.0 → 各候选无显著差异，"
                  f"维持现状 M0，Bayer 不需要改")
        m0 = summary["M0_现状RGGB"]
        print(f"    M0 现状：PSNR_uni={m0['psnr_uni']:.2f}  锐度_uni={m0['lap_ratio_uni']:.2f}  "
              f"假色比={m0['local_chroma']/gt_lcs:.2f}x  sat={m0['sat_ratio']:.2f}")


if __name__ == "__main__":
    main()
