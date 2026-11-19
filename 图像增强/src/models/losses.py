# -*- coding: utf-8 -*-
"""损失函数：Charbonnier L1 + SSIM，可选 VGG 感知损失。"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def charbonnier_loss(pred, target, eps=1e-6):
    diff = pred - target
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


def _gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
                          for x in range(window_size)])
    return gauss / gauss.sum()


def _create_window(window_size, channel):
    _1d = _gaussian(window_size, 1.5).unsqueeze(1)
    _2d = _1d.mm(_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return _2d.expand(channel, 1, window_size, window_size).contiguous()


def ssim(img1, img2, window_size=11, size_average=True):
    """结构相似度（返回 0~1，越大越相似）。"""
    (_, channel, _, _) = img1.size()
    window = _create_window(window_size, channel).to(img1.device).type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    # 方差理论上非负，但浮点舍入可能产生微小负数，夹紧以防数值不稳定
    sigma1_sq = torch.clamp(sigma1_sq, min=0.0)
    sigma2_sq = torch.clamp(sigma2_sq, min=0.0)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        return 1.0 - ssim(pred, target, self.window_size)


class PerceptualLoss(nn.Module):
    """VGG16 感知损失（可选，需 torchvision，默认关闭）。"""

    def __init__(self):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.layers = nn.Sequential(
            vgg[0:4], vgg[4:9], vgg[9:16],   # relu1_2, relu2_2, relu3_3
        )
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def forward(self, pred, target):
        pred = (pred - self.mean.to(pred.device)) / self.std.to(pred.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)
        loss = 0.0
        for l in self.layers:
            pred = l(pred)
            target = l(target)
            loss += F.l1_loss(pred, target)
        return loss
