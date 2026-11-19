from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from .common import load_config,read_manifest,safe_output,configure_torch
from .data import read_raw,pack_raw,make_context
from .model import RawToRGB


@torch.inference_mode()
def predict_packed(model,packed,context,device="cuda",amp=True,tile_size=0,halo=192):
    """Full resolution by default. Tiles retain their halo until final cropping.

    Every tile receives the same full-frame context. Tile coordinates are
    multiples of 16 so the U-Net's sampling phase is preserved.
    """
    low=torch.from_numpy(np.ascontiguousarray(packed)).unsqueeze(0).to(device)
    ctx=torch.from_numpy(np.ascontiguousarray(context)).unsqueeze(0).to(device)
    def forward(x):
        with torch.autocast(device_type=torch.device(device).type,dtype=torch.float16,enabled=amp and str(device).startswith("cuda")):
            return model(x,ctx)
    if not tile_size:
        fallback=False
        try: output=forward(low)
        except torch.cuda.OutOfMemoryError:
            fallback=True
        if fallback:
            del low,ctx
            torch.cuda.empty_cache()
            return predict_packed(model,packed,context,device,amp,tile_size=512,halo=halo)
    else:
        h,w=low.shape[-2:];tile_size=max(16,(tile_size//16)*16);halo=((halo+15)//16)*16
        # Assemble on CPU so tiling also bounds device memory for the output.
        output=torch.empty((1,3,h*2,w*2),dtype=torch.float32,device="cpu")
        for y in range(0,h,tile_size):
            for x in range(0,w,tile_size):
                y1,x1=min(y+tile_size,h),min(x+tile_size,w)
                a,b=max(0,y-halo),max(0,x-halo);c,d=min(h,y1+halo),min(w,x1+halo)
                block=forward(low[:,:,a:c,b:d]).float().cpu()
                output[:,:,2*y:2*y1,2*x:2*x1]=block[:,:,2*(y-a):2*(y1-a),2*(x-b):2*(x1-b)]
    arr=output[0].float().clamp(0,1).cpu().permute(1,2,0).numpy()
    return np.rint(arr*255).clip(0,255).astype(np.uint8)


def load_checkpoint_model(checkpoint,device="cuda"):
    checkpoint=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=RawToRGB(checkpoint["calibration"],**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"]);model.to(device).eval()
    return model,checkpoint


def enhance(raw_path,checkpoint_path,device="cuda",tile_size=0):
    """Public inference API: RAW + checkpoint only; no GT or manifest required."""
    model,checkpoint=load_checkpoint_model(checkpoint_path,device)
    spec=checkpoint["input_spec"]
    raw=read_raw(raw_path,spec["height"],spec["width"])
    packed=pack_raw(raw,spec["black_level"],spec["scale"])
    return predict_packed(model,packed,make_context(packed),device,tile_size=tile_size)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("input");p.add_argument("--checkpoint",required=True)
    p.add_argument("--output",required=True);p.add_argument("--device",default="cuda");p.add_argument("--tile-size",type=int,default=0)
    a=p.parse_args();configure_torch();out=safe_output(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    rgb=enhance(a.input,a.checkpoint,a.device,a.tile_size);Image.fromarray(rgb).save(out,compress_level=1)
    print(json.dumps({"output":str(out),"shape":list(rgb.shape),"gt_required":False}))
