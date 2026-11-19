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


# ---------------------------------------------------------------- v5 新增

# BT.601 亮度权重（与 cv2.COLOR_RGB2GRAY 一致）
_LUMA = (0.299, 0.587, 0.114)


def _chroma(pred, target):
    """把 RGB 拆成「亮度图 + 两条色差通道」，返回 (pred_c, target_c)。

    色差定义：Cb = B - Y，Cr = R - Y。这是一个线性可逆变换，梯度稳定，
    且对整体亮度偏移不敏感 —— 正是「只惩罚色彩偏差、不管明暗」需要的性质。
    """
    yp = pred[:, 0:1] * _LUMA[0] + pred[:, 1:2] * _LUMA[1] + pred[:, 2:3] * _LUMA[2]
    yt = target[:, 0:1] * _LUMA[0] + target[:, 1:2] * _LUMA[1] + target[:, 2:3] * _LUMA[2]
    pred_c = torch.cat([pred[:, 2:3] - yp, pred[:, 0:1] - yp], dim=1)
    target_c = torch.cat([target[:, 2:3] - yt, target[:, 0:1] - yt], dim=1)
    return pred_c, target_c


def chroma_loss(pred, target, mode="l1", eps=1e-6):
    """单色度损失：只在色差通道上算距离，不碰亮度。

    动机：v1 的 L1 + SSIM 里亮度项占绝对主导（极暗图亮度误差远大于色度误差），
    梯度被亮度拉走，色彩恢复得不到优化压力。单独加一条纯色度项可以补这个缺口。

    mode='l1'  : Charbonnier，稳定、不易过冲
    mode='cos' : 1 - 余弦相似度，直接对齐色度的「方向」，对整体色度幅度不敏感

    注意 eps 的选取：Charbonnier 形式下 `sqrt(0 + eps²) = eps`，会在完美匹配时
    留下 eps 的常数偏置（eps=1e-6 时比 fp16 的 1e-3 分辨率还小，可忽略）。
    """
    pc, tc = _chroma(pred, target)
    if mode == "cos":
        pc = pc.flatten(2)
        tc = tc.flatten(2)
        num = (pc * tc).sum(dim=1)
        den = pc.norm(dim=1) * tc.norm(dim=1) + eps
        return (1.0 - num / den).mean()
    return torch.mean(torch.sqrt((pc - tc) ** 2 + eps * eps))


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
