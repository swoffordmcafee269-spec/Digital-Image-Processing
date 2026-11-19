# -*- coding: utf-8 -*-
"""训练日志与检查点工具。"""
import csv
import os

import torch


class CSVLogger:
    def __init__(self, path, columns):
        self.path = path
        self.columns = columns
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(columns)

    def log(self, **kwargs):
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([kwargs.get(c, "") for c in self.columns])


def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, model=None, optimizer=None, scheduler=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    if model is not None and "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("best_psnr", -1.0)


def get_device(preferred=None):
    if preferred:
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
