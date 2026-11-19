# -*- coding: utf-8 -*-
"""评价指标封装：PSNR / SSIM / LPIPS。

约定输入均为 numpy 数组 (H, W, 3)，float32 且值域 [0, 1]。
"""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips = None
_lpips_net = None


def _get_lpips(net="alex"):
    global _lpips, _lpips_net
    if _lpips is None or _lpips_net != net:
        import lpips
        _lpips = lpips.LPIPS(net=net)
        _lpips_net = net
    return _lpips


def psnr(img1, img2, data_range=1.0):
    return float(peak_signal_noise_ratio(img1, img2, data_range=data_range))


def ssim(img1, img2):
    return float(structural_similarity(img1, img2, channel_axis=2, data_range=1.0))


def lpips(img1, img2, net="alex"):
    import torch
    model = _get_lpips(net)
    t1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
    t2 = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
    with torch.no_grad():
        score = model(t1, t2).item()
    return float(score)


def compute_metrics(img1, img2, metric_names, lpips_net="alex"):
    """返回 {metric_name: value}。img1 为预测，img2 为 GT。"""
    results = {}
    for m in metric_names:
        if m == "psnr":
            results["psnr"] = psnr(img1, img2)
        elif m == "ssim":
            results["ssim"] = ssim(img1, img2)
        elif m == "lpips":
            results["lpips"] = lpips(img1, img2, lpips_net)
    return results
