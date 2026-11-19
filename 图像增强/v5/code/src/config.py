# -*- coding: utf-8 -*-
"""配置加载：读取 YAML，支持默认值合并与命令行覆盖。"""
import argparse
import os
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CONFIG = {
    "data": {
        "root": "data",
        "dataset": "Sony",
        "raw_subdir": "short",
        "gt_subdir": "long",
        "raw_ext": "raw",
        "gt_ext": "jpg",
        "black_level": 512,
        "white_level": 16383,
        "raw_width": 3840,
        "raw_height": 2160,
        "val_ratio": 0.1,
        # --- v5 新增：白平衡前处理。默认关闭，保证 v1/v2 配置行为完全不变 ---
        "use_wb": False,
        "wb_cache": "",
    },
    "model": {
        "in_channels": 4,
        "out_channels": 12,
        "base_channels": 32,
        "num_levels": 4,
        "norm": "instance",
    },
    "train": {
        "epochs": 50,
        "batch_size": 8,
        "patch_size": 512,
        "lr": 1e-4,
        "weight_decay": 0.0,
        "num_workers": 4,
        "amp": True,
        "grad_clip": 1.0,
        "warmup_epochs": 2,
        "seed": 42,
        "l1_weight": 1.0,
        "ssim_weight": 0.15,
        "perceptual": False,
        "perceptual_weight": 0.05,
        "log_dir": "outputs/logs",
        "ckpt_dir": "outputs/checkpoints",
        "resume": "",
        # --- v5 新增 ---
        "chroma_weight": 0.0,      # 单色度损失权重（默认 0 = 关闭）
        "select_metric": "psnr",   # 选模主指标：psnr / ssim
    },
    "eval": {
        "metrics": ["psnr", "ssim", "lpips"],
        "lpips_net": "alex",
        "output_dir": "outputs/results",
        "save_images": True,
        "num_samples": 4,
        "run_baseline": True,
        "baseline_method": "clahe",
    },
}


class Config(dict):
    """支持属性访问的 dict（嵌套结构也会被递归转换）。"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def get(self, key, default=None):
        return dict.get(self, key, default)


def _convert(obj):
    if isinstance(obj, dict):
        return Config({k: _convert(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_convert(v) for v in obj]
    return obj


def _deep_merge(base, override):
    """用 override 递归覆盖 base，返回新 dict。"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(path=None, cli_overrides=None):
    """加载配置：默认值 <- YAML <- 命令行覆盖。"""
    cfg = _convert(DEFAULT_CONFIG)
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        cfg = _convert(_deep_merge(cfg, user_cfg))
    if cli_overrides:
        for kv in cli_overrides:
            if "=" not in kv:
                raise ValueError(f"覆盖参数格式错误: {kv}（应为 key=value）")
            k, v = kv.split("=", 1)
            _set_nested(cfg, k, _auto_type(v))
    return cfg


def _set_nested(cfg, dotted_key, value):
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur:
            cur[k] = Config()
        cur = cur[k]
    cur[keys[-1]] = value


def _auto_type(v):
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def parse_args(description=""):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="配置文件路径")
    parser.add_argument("--set", action="append", default=None, metavar="k=v",
                        help="覆盖配置项，如 --set data.dataset=Canon（可多次）")
    parser.add_argument("--device", type=str, default=None,
                        help="训练设备，如 cuda / cpu（默认自动检测）")
    return parser.parse_args()
