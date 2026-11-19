from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F
from skimage.metrics import structural_similarity


def ssim_torch(pred, target, reduction=True):
    """RGB SSIM matching skimage defaults: 7x7, uniform, sample covariance.

    Valid pooling matches skimage's exclusion of its reflected 3-pixel border.
    Run in float32 even when training uses autocast.
    """
    with torch.autocast(device_type=pred.device.type, enabled=False):
        x,y=pred.float(),target.float()
        ux=F.avg_pool2d(x,7,1);uy=F.avg_pool2d(y,7,1)
        covariance=49/48
        vx=covariance*(F.avg_pool2d(x*x,7,1)-ux*ux)
        vy=covariance*(F.avg_pool2d(y*y,7,1)-uy*uy)
        vxy=covariance*(F.avg_pool2d(x*y,7,1)-ux*uy)
        result=((2*ux*uy+.01**2)*(2*vxy+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
        score=result.mean((1,2,3))
        return score.mean() if reduction else score


def chroma_loss(pred,target):
    with torch.autocast(device_type=pred.device.type,enabled=False):
        x=F.avg_pool2d(pred.float(),4,4);y=F.avg_pool2d(target.float(),4,4)
        matrix=x.new_tensor([[-.168736,-.331264,.5],[.5,-.418688,-.081312]])
        return torch.einsum("oc,bchw->bohw",matrix,x-y).abs().mean()


def restoration_loss(pred,target,ssim_weight=.1,chroma_weight=.05):
    diff=pred.float()-target.float()
    pixel=(diff*diff+1e-6).sqrt().mean()
    structure=1-ssim_torch(pred,target)
    chroma=chroma_loss(pred,target)
    loss=pixel+ssim_weight*structure+chroma_weight*chroma
    return loss,{"pixel":float(pixel.detach()),"structure":float(structure.detach()),"chroma":float(chroma.detach())}


def rgb_to_tensor(rgb,device):
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1).unsqueeze(0).to(device=device,dtype=torch.float32)/255


class MetricEvaluator:
    def __init__(self,device="cuda",with_lpips=True,ssim_backend="torch"):
        self.device=torch.device(device);self.lpips=None;self.ssim_backend=ssim_backend
        if with_lpips:
            import lpips
            self.lpips=lpips.LPIPS(net="alex",version="0.1").to(device).eval()
            for p in self.lpips.parameters():p.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self,pred_uint8,gt_uint8,color=True):
        if pred_uint8.dtype!=np.uint8 or gt_uint8.dtype!=np.uint8 or pred_uint8.shape!=gt_uint8.shape:
            raise ValueError("Metrics require same-size uint8 RGB images")
        pred=rgb_to_tensor(pred_uint8,self.device);target=rgb_to_tensor(gt_uint8,self.device)
        mse=(pred-target).square().double().mean().item()
        score={"psnr":float(-10*np.log10(mse)) if mse>0 else float("inf")}
        if self.ssim_backend=="torch":score["ssim"]=float(ssim_torch(pred,target))
        else:score["ssim"]=float(structural_similarity(pred_uint8.astype(np.float32)/255,gt_uint8.astype(np.float32)/255,channel_axis=2,data_range=1.0))
        if self.lpips is not None:score["lpips"]=float(self.lpips(pred*2-1,target*2-1))
        if color:
            # Secondary diagnostics use a stated 512-pixel-long-side area thumbnail.
            import cv2
            from skimage.color import rgb2lab,deltaE_ciede2000
            h,w=gt_uint8.shape[:2];scale=512/max(h,w);size=(round(w*scale),round(h*scale))
            a=cv2.resize(pred_uint8,size,interpolation=cv2.INTER_AREA).astype(np.float32)/255
            b=cv2.resize(gt_uint8,size,interpolation=cv2.INTER_AREA).astype(np.float32)/255
            score["delta_e_2000_512"]=float(deltaE_ciede2000(rgb2lab(a),rgb2lab(b)).mean())
            score["saturation_pred_512"]=float(cv2.cvtColor(a,cv2.COLOR_RGB2HSV)[...,1].mean())
            score["saturation_gt_512"]=float(cv2.cvtColor(b,cv2.COLOR_RGB2HSV)[...,1].mean())
        return score
