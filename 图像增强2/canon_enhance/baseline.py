from __future__ import annotations

import cv2
import numpy as np

from .data import linear_to_srgb_np


def calibrated_baseline(packed,calibration):
    """Fast RAW denoise + train-calibrated camera transform + sRGB encoding.

    This is a calibrated classical baseline, not an unsupervised method.
    Parameters are fixed across all evaluation images; no GT is accepted.
    """
    channels=[]
    for plane in packed:
        channels.append(cv2.bilateralFilter(plane.astype(np.float32),7,.08,3))
    raw=np.stack(channels,axis=-1)
    matrix=np.asarray(calibration["matrix"],dtype=np.float32)
    bias=np.asarray(calibration["bias"],dtype=np.float32)
    linear=np.einsum("hwc,oc->hwo",raw,matrix)+bias
    rgb=linear_to_srgb_np(linear)
    rgb=cv2.resize(rgb,(packed.shape[2]*2,packed.shape[1]*2),interpolation=cv2.INTER_LINEAR)
    return np.rint(np.clip(rgb,0,1)*255).astype(np.uint8)
