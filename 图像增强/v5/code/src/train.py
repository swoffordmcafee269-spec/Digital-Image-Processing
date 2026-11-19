# -*- coding: utf-8 -*-
"""训练入口。

用法（项目根目录）：
    python src/train.py
    python src/train.py --config configs/config.yaml --set data.dataset=Canon
    python src/train.py --set train.patch_size=256 train.batch_size=4
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config, parse_args  # noqa: E402
from src.data.dataset import PairedDataset  # noqa: E402
from src.models.losses import (SSIMLoss, charbonnier_loss,  # noqa: E402
                               chroma_loss, PerceptualLoss)
from src.models.unet import build_model  # noqa: E402
from src.utils.logger import (CSVLogger, get_device, load_checkpoint,  # noqa: E402
                              save_checkpoint)

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))              # …/v5/code/src
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_SRC_DIR)))  # 项目根


def resolve_dir(p):
    """相对路径按项目根解析（v5：保证产物落到 v5/ 下，不受 cwd 影响）。"""
    return p if os.path.isabs(p) else os.path.join(_PROJ_ROOT, p)


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filelist_path(cfg, name):
    return os.path.join(cfg.data.root, cfg.data.dataset, name)


def torch_psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 10.0 * math.log10(1.0 / (mse.item() + 1e-8))


def torch_ssim(pred, target):
    """验证用 SSIM（与评估口径一致：逐通道等权平均后取均值）。"""
    from src.models.losses import ssim
    return ssim(pred, target).item()


def make_scheduler(optimizer, cfg, steps_per_epoch):
    total = cfg.train.epochs * steps_per_epoch
    warmup = cfg.train.warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_loss(cfg, device):
    t = cfg.train
    ssim_loss = SSIMLoss() if t.ssim_weight > 0 else None
    perceptual = PerceptualLoss().to(device) if t.perceptual else None
    chroma_w = float(t.get("chroma_weight", 0.0))

    def loss_fn(pred, target):
        loss = t.l1_weight * charbonnier_loss(pred, target)
        if ssim_loss is not None:
            loss = loss + t.ssim_weight * ssim_loss(pred, target)
        if perceptual is not None:
            loss = loss + t.perceptual_weight * perceptual(pred, target)
        if chroma_w > 0:
            loss = loss + chroma_w * chroma_loss(pred, target)
        return loss

    return loss_fn


def main():
    args = parse_args("路线B 轻量 UNet 训练")
    cfg = load_config(args.config, args.set)
    set_seed(cfg.train.seed)
    device = get_device(args.device)
    print(f"使用设备: {device}")

    train_path = filelist_path(cfg, "train.txt")
    val_path = filelist_path(cfg, "val.txt")
    if not os.path.exists(train_path):
        print(f"[错误] 未找到训练 filelist: {train_path}")
        print("请先下载数据并执行: python scripts/make_filelist.py")
        sys.exit(1)

    train_ds = PairedDataset(cfg, train_path, patch_size=cfg.train.patch_size)
    val_ds = PairedDataset(cfg, val_path, patch_size=cfg.train.patch_size)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=cfg.train.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=cfg.train.num_workers, pin_memory=True)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = make_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)
    loss_fn = build_loss(cfg, device)

    start_epoch, best_psnr = 0, -1.0
    if cfg.train.resume and os.path.exists(cfg.train.resume):
        start_epoch, best_psnr = load_checkpoint(cfg.train.resume, model, optimizer, scheduler, scaler)
        print(f"从 checkpoint 续训，起始 epoch={start_epoch}，best PSNR={best_psnr:.2f}")

    # v5：日志增加 val_ssim；选模主指标可切换
    select_metric = str(cfg.train.get("select_metric", "psnr")).lower()
    best_score = -1.0
    log_dir = resolve_dir(cfg.train.log_dir)
    ckpt_dir = resolve_dir(cfg.train.ckpt_dir)
    logger = CSVLogger(os.path.join(log_dir, "train_log.csv"),
                       ["epoch", "lr", "train_loss", "val_psnr", "val_ssim"])
    print(f"[v5] WB={'on' if cfg.data.get('use_wb', False) else 'off'}  "
          f"chroma_weight={cfg.train.get('chroma_weight', 0.0)}  "
          f"select_metric={select_metric}")
    print(f"[v5] log_dir={log_dir}")
    print(f"[v5] ckpt_dir={ckpt_dir}")

    for epoch in range(start_epoch, cfg.train.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.train.epochs}")
        for low, gt in pbar:
            low, gt = low.to(device), gt.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=cfg.train.amp):
                pred = model(low)
            loss = loss_fn(pred.float(), gt)  # 损失在 fp32 下计算，避免 fp16 数值不稳定
            scaler.scale(loss).backward()
            if cfg.train.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # 验证集 PSNR + SSIM
        model.eval()
        val_psnr, val_ssim, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for low, gt in val_loader:
                low, gt = low.to(device), gt.to(device)
                pred = model(low)
                val_psnr += torch_psnr(pred, gt)
                val_ssim += torch_ssim(pred.float(), gt)
                n_val += 1
        val_psnr /= max(1, n_val)
        val_ssim /= max(1, n_val)

        avg_loss = total_loss / max(1, len(train_loader))
        lr = optimizer.param_groups[0]["lr"]
        logger.log(epoch=epoch + 1, lr=round(lr, 8), train_loss=round(avg_loss, 6),
                   val_psnr=round(val_psnr, 3), val_ssim=round(val_ssim, 4))
        print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}  val_PSNR={val_psnr:.2f}  "
              f"val_SSIM={val_ssim:.4f}  lr={lr:.2e}")

        save_checkpoint({"epoch": epoch + 1, "best_psnr": best_psnr,
                         "model": model.state_dict(),
                         "optimizer": optimizer.state_dict(),
                         "scheduler": scheduler.state_dict(),
                         "scaler": scaler.state_dict()},
                        os.path.join(ckpt_dir, "last.pt"))
        score = val_ssim if select_metric == "ssim" else val_psnr
        if score > best_score:
            best_score = score
            best_psnr = val_psnr if select_metric == "psnr" else best_psnr
            save_checkpoint({"epoch": epoch + 1, "best_psnr": best_psnr,
                             "best_score": best_score,
                             "select_metric": select_metric,
                             "model": model.state_dict()},
                            os.path.join(ckpt_dir, "best.pt"))
            print(f"  -> 新的最优模型已保存 ({select_metric}={best_score:.4f})")

    print(f"训练结束，最优 {select_metric} = {best_score:.4f}，模型保存在 {ckpt_dir}")


if __name__ == "__main__":
    main()
