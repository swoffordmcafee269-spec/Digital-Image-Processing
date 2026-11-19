# -*- coding: utf-8 -*-
"""残差 UNet：输入 4 通道 packed RAW，输出 12 通道 -> pixel_shuffle -> 3 通道 RGB。

输出分辨率为输入的 2 倍（对齐 GT 全分辨率）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(ch, norm_type):
    if norm_type == "instance":
        return nn.InstanceNorm2d(ch)
    if norm_type == "group":
        return nn.GroupNorm(8, ch)
    if norm_type == "batch":
        return nn.BatchNorm2d(ch)
    return nn.Identity()


class ResidualBlock(nn.Module):
    def __init__(self, ch, norm="instance"):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.norm1 = _norm(ch, norm)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.norm2 = _norm(ch, norm)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + identity)


class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=12, base_channels=32,
                 num_levels=4, norm="instance"):
        super().__init__()
        self.num_levels = num_levels
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            ResidualBlock(base_channels, norm),
        )
        self.downs = nn.ModuleList()
        ch = base_channels
        for _ in range(num_levels):
            self.downs.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, 3, 2, 1),
                ResidualBlock(ch * 2, norm),
            ))
            ch *= 2
        self.bottleneck = ResidualBlock(ch, norm)

        self.ups = nn.ModuleList()
        for _ in range(num_levels):
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1),
                ResidualBlock(ch // 2, norm),
            ))
            ch //= 2

        self.outc = nn.Conv2d(base_channels, out_channels, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        _, _, H, W = x.shape
        # 反射填充到 16 的倍数，保证任意尺寸可推理（训练时 patch 已是 16 的倍数）
        mod = 16
        pad_h = (mod - H % mod) % mod
        pad_w = (mod - W % mod) % mod
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        x = self.inc(x)
        skips = [x]
        for d in self.downs:
            x = d(x)
            skips.append(x)
        x = self.bottleneck(x)
        for i, u in enumerate(self.ups):
            x = u(x)
            x = self.act(x + skips[-2 - i])
        x = self.outc(x)
        x = self.shuffle(x)

        out_h, out_w = 2 * H, 2 * W
        return x[:, :, :out_h, :out_w]


def build_model(cfg):
    m = cfg.model
    return UNet(
        in_channels=m.in_channels,
        out_channels=m.out_channels,
        base_channels=m.base_channels,
        num_levels=m.num_levels,
        norm=m.norm,
    )
