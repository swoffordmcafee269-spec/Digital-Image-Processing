# -*- coding: utf-8 -*-
"""扫描数据目录，自动配对低光 RAW 与 GT，生成 train/val/test filelist。

用法（下载好数据后，在项目根目录执行）：
    python scripts/make_filelist.py
    python scripts/make_filelist.py --config configs/config.yaml
    python scripts/make_filelist.py --dataset Canon

配对规则：RAW 与 GT 按「文件名主干」匹配（忽略扩展名），并自动剥离
"_gt" / "_GT" / "_ref" / "_REF" 等常见后缀。
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config  # noqa: E402

RAW_EXTS = {".raw", ".arw", ".cr2", ".cr3", ".dng", ".nef"}
GT_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
GT_SUFFIXES = ["_gt", "_GT", "_ref", "_REF"]


def _skip_raw(path):
    """跳过高质量参考 RAW（gt_raw / long 目录），避免与低光 RAW 同名冲突。"""
    parts = path.replace("\\", "/").lower().split("/")
    return any(p in ("gt_raw", "long") for p in parts)


def _stem_match(raw_stem, gt_stem):
    if raw_stem == gt_stem:
        return True
    for s in GT_SUFFIXES:
        if gt_stem.endswith(s) and gt_stem[: -len(s)] == raw_stem:
            return True
        if raw_stem.endswith(s) and raw_stem[: -len(s)] == gt_stem:
            return True
    return False


def scan_pairs(subset_dir, raw_subdir, gt_subdir, raw_ext, gt_ext):
    raw_dir = os.path.join(subset_dir, raw_subdir) if raw_subdir else subset_dir
    gt_dir = os.path.join(subset_dir, gt_subdir) if gt_subdir else subset_dir

    raw_files = {}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if (raw_ext and ext == "." + raw_ext.lstrip(".")) or (not raw_ext and ext in RAW_EXTS):
                full = os.path.join(root, f)
                if _skip_raw(full):
                    continue
                raw_files[os.path.splitext(f)[0]] = full

    gt_files = {}
    for root, _, files in os.walk(gt_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if (gt_ext and ext == "." + gt_ext.lstrip(".")) or (not gt_ext and ext in GT_EXTS):
                full = os.path.join(root, f)
                gt_files[os.path.splitext(f)[0]] = full

    pairs = []
    unmatched = []
    for rstem, rpath in raw_files.items():
        matched = None
        for gstem, gpath in gt_files.items():
            if _stem_match(rstem, gstem):
                matched = gpath
                break
        if matched:
            pairs.append((rpath, matched))
        else:
            unmatched.append(rpath)

    return pairs, unmatched


def main():
    parser = argparse.ArgumentParser(description="生成 train/val/test filelist")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default=None, help="覆盖配置中的 dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    d = cfg.data
    dataset = args.dataset or d.dataset
    root = d.root

    subset_dir = os.path.join(root, dataset)
    if not os.path.isdir(subset_dir):
        print(f"[错误] 未找到子集目录: {subset_dir}")
        print("请确认数据已下载并放入 data/Sony 或 data/Canon，且 config.yaml 中 data.root/dataset 正确。")
        sys.exit(1)

    pairs, unmatched = scan_pairs(subset_dir, d.raw_subdir, d.gt_subdir,
                                  d.raw_ext, d.gt_ext)
    if not pairs:
        print("[错误] 没有匹配到任何 RAW-GT 配对。")
        print("请检查 raw_subdir/gt_subdir/raw_ext/gt_ext 配置，或文件名主干是否一致。")
        sys.exit(1)

    random.seed(args.seed)
    random.shuffle(pairs)
    n_val = max(1, int(len(pairs) * d.val_ratio))
    splits = {
        "train.txt": pairs[:-n_val],
        "val.txt": pairs[-n_val:],
        "test.txt": pairs[:],   # 全量评估
    }

    for fname, part in splits.items():
        out_path = os.path.join(subset_dir, fname)
        with open(out_path, "w", encoding="utf-8") as f:
            for raw_full, gt_full in part:
                # 相对 data.root 的路径
                raw_rel = os.path.relpath(raw_full, root)
                gt_rel = os.path.relpath(gt_full, root)
                f.write(f"{raw_rel} {gt_rel}\n")
        print(f"[OK] {out_path}  共 {len(part)} 对")

    print(f"\n共匹配 {len(pairs)} 对（训练 {len(splits['train.txt'])} / 验证 {len(splits['val.txt'])}）")
    if unmatched:
        print(f"警告：{len(unmatched)} 个 RAW 未匹配到 GT（已跳过）:")
        for p in unmatched[:10]:
            print("  -", p)


if __name__ == "__main__":
    main()
