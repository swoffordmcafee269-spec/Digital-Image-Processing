from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def linear_to_srgb(x):
    x = x.clamp_min(0)
    return torch.where(x <= .0031308, 12.92 * x,
                       1.055 * x.clamp_min(1e-8).pow(1 / 2.4) - .055)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels, 3, padding=1),
                                  nn.LeakyReLU(.1, inplace=True),
                                  nn.Conv2d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + .2 * self.body(x)


class RawToRGB(nn.Module):
    """Normalization-free residual U-Net, initialized around a train-only ISP.

    The camera matrix is fitted once using training pairs. A small input-only
    context branch adjusts global exposure/color consistently across all tiles.
    The learned RGB residual performs denoising and detail reconstruction.
    Output is already encoded sRGB: never apply another gamma/WB afterwards.
    """

    def __init__(self, calibration, base_channels=32, levels=4, adaptive_proxy=False, proxy_blur=5):
        super().__init__()
        if proxy_blur<1 or proxy_blur%2!=1:raise ValueError("proxy_blur must be a positive odd integer")
        self.proxy_blur=proxy_blur
        self.levels = levels
        self.adaptive_proxy = adaptive_proxy
        self.register_buffer("camera_matrix", torch.tensor(calibration["matrix"], dtype=torch.float32))
        self.register_buffer("camera_bias", torch.tensor(calibration["bias"], dtype=torch.float32))
        self.stem = nn.Sequential(nn.Conv2d(4, base_channels, 3, padding=1),
                                  nn.LeakyReLU(.1, inplace=True), ResidualBlock(base_channels))
        self.downs = nn.ModuleList();self.ups = nn.ModuleList()
        channels = base_channels
        for _ in range(levels):
            self.downs.append(nn.Sequential(nn.Conv2d(channels, channels*2, 3, stride=2, padding=1),
                                            nn.LeakyReLU(.1, inplace=True), ResidualBlock(channels*2)))
            channels *= 2
        self.bottleneck = ResidualBlock(channels)
        for _ in range(levels):
            self.ups.append(nn.Sequential(nn.Conv2d(channels, channels//2, 3, padding=1),
                                          nn.LeakyReLU(.1, inplace=True), ResidualBlock(channels//2)))
            channels //= 2
        self.head = nn.Conv2d(channels, 12, 3, padding=1)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)
        self.context_net = nn.Sequential(nn.Conv2d(4, 16, 3, stride=2, padding=1),nn.LeakyReLU(.1),
                                         nn.Conv2d(16, 32, 3, stride=2, padding=1),nn.LeakyReLU(.1),
                                         nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(32, 32),
                                         nn.LeakyReLU(.1),nn.Linear(32, 7 if adaptive_proxy else 6))
        nn.init.zeros_(self.context_net[-1].weight);nn.init.zeros_(self.context_net[-1].bias)

    def color_proxy(self, x, context):
        # Mild RAW-domain averaging gives a stable, colored initial prediction.
        radius=self.proxy_blur//2
        smooth = F.avg_pool2d(F.pad(x.float(), (radius,radius,radius,radius), mode="replicate"), self.proxy_blur, stride=1)
        rgb = torch.einsum("oc,bchw->bohw", self.camera_matrix.float(), smooth)
        rgb = rgb + self.camera_bias.float()[None,:,None,None]
        adjust = self.context_net(torch.log1p(context.clamp_min(0)) if self.adaptive_proxy else context)
        gain = ((2.0 if self.adaptive_proxy else .7) * torch.tanh(adjust[:,:3].float())).exp()[:,:,None,None]
        bias = (.05 * torch.tanh(adjust[:,3:6].float()))[:,:,None,None]
        rgb = linear_to_srgb(rgb * gain + bias)
        if self.adaptive_proxy:
            # The learned gate can suppress the noisy ISP prior on heavy-noise
            # frames. The U-Net still receives all (log-compressed) RAW values.
            gate=2*torch.sigmoid(adjust[:,6:7].float())[:,:,None,None]
            rgb=rgb.clamp(0,1)*gate
        return F.interpolate(rgb, scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, context=None):
        h,w = x.shape[-2:]
        if context is None: context=F.adaptive_avg_pool2d(x, (32,32))
        proxy=self.color_proxy(x,context)
        if self.adaptive_proxy:x=torch.log1p(x.clamp_min(0))
        multiple=2**self.levels
        ph,pw=(-h)%multiple,(-w)%multiple
        if ph or pw: x=F.pad(x,(0,pw,0,ph),mode="replicate")
        x=self.stem(x);skips=[x]
        for down in self.downs:
            x=down(x);skips.append(x)
        x=self.bottleneck(x)
        for i,up in enumerate(self.ups):
            x=F.interpolate(x,size=skips[-2-i].shape[-2:],mode="bilinear",align_corners=False)
            x=up(x)+skips[-2-i]
        residual=F.pixel_shuffle(self.head(x),2)[:,:,:2*h,:2*w]
        return proxy + .25*residual.float()


def build_model(cfg, manifest):
    return RawToRGB(manifest["camera_calibration"], **cfg["model"])
