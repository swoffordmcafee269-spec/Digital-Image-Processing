"""AVIF 图像压缩封装（基于 pillow-avif-plugin / libavif）。

AVIF 是 AV1 帧内编码的图片格式，率失真性能显著优于 JPEG，
属于传统（非神经）编码框架的第一梯队，用作"极致性能"对比方案。

统一接口（与 jpeg_dct 一致）：
    encode(img: np.uint8[H,W,3], quality=50) -> bytes
    decode(data: bytes) -> np.uint8[H,W,3]

quality：0~100，越大质量越高、码率越大（映射到 libavif 量化参数）。
"""

import io

import numpy as np
from PIL import Image

import pillow_avif  # noqa: F401  注册 AVIF 插件


def encode(img_rgb, quality=50):
    img = Image.fromarray(np.asarray(img_rgb, dtype=np.uint8))
    buf = io.BytesIO()
    # speed=8 兼顾编码速度；quality 控制量化
    img.save(buf, format="AVIF", quality=int(quality), speed=8)
    return buf.getvalue()


def decode(data):
    img = Image.open(io.BytesIO(bytes(data)))
    return np.array(img.convert("RGB"), dtype=np.uint8)
