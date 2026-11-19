# -*- coding: utf-8 -*-
"""通用工具: 支持中文路径的图像读写"""
import cv2
import numpy as np


def imread_u(path, flags=cv2.IMREAD_COLOR):
    """支持非 ASCII 路径的 imread"""
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_u(path, img):
    """支持非 ASCII 路径的 imwrite"""
    ext = "." + path.rsplit(".", 1)[-1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok
