from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image,ImageDraw
from torch.utils.data import DataLoader

from .common import (load_config,read_manifest,seed_everything,configure_torch,
                     safe_output,atomic_torch_save,append_jsonl,write_json,worker_init)
from .data import TrainDataset,load_full_pair
from .model import build_model
from .metrics import restoration_loss,MetricEvaluator
from .inference import predict_packed
from .ema import ModelEMA


def checkpoint_payload(model,cfg,manifest,step,metrics,optimizer=None,scaler=None):
    result={"model":model.state_dict(),"model_config":cfg["model"],"calibration":manifest["camera_calibration"],
            "input_spec":{"height":cfg["height"],"width":cfg["width"],"black_level":cfg["black_level"],
                          "scale":manifest["normalization"]["scale"],"dtype":"little-endian uint16",
                          "packed_order":"top-left,top-right,bottom-left,bottom-right","output":"sRGB"},
            "manifest_fingerprint":manifest["fingerprint"],"step":step,"validation":metrics,
            "config":cfg,"pretrained":False}
    if optimizer is not None:result["optimizer"]=optimizer.state_dict()
    if scaler is not None:result["scaler"]=scaler.state_dict()
    return result


@torch.inference_mode()
def validate(model,cfg,manifest,evaluator,output_dir=None,limit=0):
    model.eval();items=[i for i in manifest["items"] if i["split"]=="val"]
    if limit:items=[items[i] for i in np.linspace(0,len(items)-1,min(limit,len(items)),dtype=int)]
    rows=[];panels=[];started=time.perf_counter()
    selected=set(np.linspace(0,len(items)-1,min(6,len(items)),dtype=int))
    for i,item in enumerate(items):
        packed,context,gt=load_full_pair(cfg,manifest,item)
        output=predict_packed(model,packed,context)
        row={"id":item["id"],**evaluator(output,gt)};rows.append(row)
        if i in selected:
            pred=Image.fromarray(output);target=Image.fromarray(gt)
            pred.thumbnail((480,270));target.thumbnail((480,270));panels.append((item["id"],pred,target,row))
    keys=[k for k in rows[0] if k!="id"]
    summary={k:float(np.mean([r[k] for r in rows])) for k in keys}
    summary.update({"images":len(rows),"seconds":time.perf_counter()-started})
    if output_dir:
        output_dir=safe_output(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        write_json(output_dir/"validation.json",{"summary":summary,"rows":rows})
        canvas=Image.new("RGB",(960,len(panels)*294),"white");draw=ImageDraw.Draw(canvas)
        for i,(name,pred,target,row) in enumerate(panels):
            canvas.paste(pred,(0,i*294));canvas.paste(target,(480,i*294))
            draw.text((6,i*294+273),f"ID {name} prediction | PSNR {row['psnr']:.2f} SSIM {row['ssim']:.3f} LPIPS {row.get('lpips',0):.3f}",fill="black")
            draw.text((486,i*294+273),"GT reference",fill="black")
        canvas.save(output_dir/"comparison.jpg",quality=92)
    return summary


def train(cfg,name="main",resume=None,stop_after=0,validation_limit=0):
    configure_torch(cfg["eval"]["cpu_threads"]);seed_everything(cfg["seed"])
    if not torch.cuda.is_available():raise RuntimeError("This 4K training configuration requires CUDA")
    manifest=read_manifest(cfg)
    out=safe_output(Path(cfg["artifact_root"])/"checkpoints"/name,cfg);out.mkdir(parents=True,exist_ok=True)
    logs=safe_output(Path(cfg["artifact_root"])/"logs"/name,cfg);logs.mkdir(parents=True,exist_ok=True)
    model=build_model(cfg,manifest).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["train"]["learning_rate"],weight_decay=cfg["train"]["weight_decay"])
    scaler=torch.amp.GradScaler("cuda",enabled=cfg["train"]["amp"])
    start_step=0;best_psnr=-float("inf");best_lpips=float("inf");latest={};ema_state=None
    if resume:
        ck=torch.load(resume,map_location="cpu",weights_only=True)
        if ck["manifest_fingerprint"]!=manifest["fingerprint"]:raise ValueError("Checkpoint belongs to a different data split/calibration")
        architecture_changed=ck["model_config"]!=cfg["model"]
        if architecture_changed:
            allowed=dict(ck["model_config"]);allowed["adaptive_proxy"]=True
            if allowed!=cfg["model"]:raise ValueError("Unsupported checkpoint architecture transfer")
            state=model.state_dict()
            for key,value in ck["model"].items():
                if key in state and state[key].shape==value.shape:state[key]=value
                elif key in ["context_net.8.weight","context_net.8.bias"] and state[key].shape[0]==7 and value.shape[0]==6:
                    state[key][:6]=value
                    state[key][:3]*=.7/2.0
            model.load_state_dict(state)
            print("Initialized adaptive proxy from main weights; optimizer reset for changed architecture.",flush=True)
        else:model.load_state_dict(ck["model"])
        start_step=int(ck["step"])
        if "optimizer" in ck and not architecture_changed:optimizer.load_state_dict(ck["optimizer"])
        if "scaler" in ck and not architecture_changed:scaler.load_state_dict(ck["scaler"])
        if not architecture_changed:ema_state=ck.get("ema")
        latest=ck.get("validation",{})
        # Always establish the resumed model on the requested validation subset.
    for filename,key in [("best_psnr.pt","psnr"),("best_lpips.pt","lpips")]:
        existing=out/filename
        if existing.exists():
            prior=torch.load(existing,map_location="cpu",weights_only=True)
            if prior["manifest_fingerprint"]!=manifest["fingerprint"]:
                raise ValueError("Choose a new run name after changing the data manifest")
            if not resume:raise FileExistsError("This training run already exists; use --resume or a new run name")
            if key=="psnr":best_psnr=prior["validation"][key]
            else:best_lpips=prior["validation"][key]
            del prior
    ema=ModelEMA(model,cfg["train"].get("ema_decay",0),ema_state) if cfg["train"].get("ema_decay",0)>0 else None
    del ema_state
    dataset=TrainDataset(cfg,manifest,cfg["train"]["patch_size"])
    loader=DataLoader(dataset,batch_size=cfg["train"]["batch_size"],shuffle=True,
                      num_workers=cfg["train"]["num_workers"],pin_memory=True,drop_last=True,
                      persistent_workers=cfg["train"]["num_workers"]>0,worker_init_fn=worker_init,
                      generator=torch.Generator().manual_seed(cfg["seed"]+start_step))
    evaluator=MetricEvaluator();started=time.perf_counter()
    write_json(logs/"run_config.json",{"config":cfg,"resume":str(resume) if resume else None,
                                      "manifest_fingerprint":manifest["fingerprint"],"parameters":sum(p.numel() for p in model.parameters())})
    print(f"TRAIN {name}: start={start_step}, planned={cfg['train']['steps']}, train_images={len(dataset)}, chroma={cfg['train']['chroma_weight']}",flush=True)

    def evaluate_save(step):
        nonlocal latest,best_psnr,best_lpips
        backup=ema.apply(model) if ema else None
        latest=validate(model,cfg,manifest,evaluator,logs/f"val_{step:06d}",validation_limit)
        append_jsonl(logs/"validation.jsonl",{"step":step,"elapsed_s":time.perf_counter()-started,**latest})
        payload=checkpoint_payload(model,cfg,manifest,step,latest)
        if latest["psnr"]>best_psnr:
            best_psnr=latest["psnr"];atomic_torch_save(payload,out/"best_psnr.pt")
        if latest["lpips"]<best_lpips:
            best_lpips=latest["lpips"];atomic_torch_save(payload,out/"best_lpips.pt")
        write_json(logs/"status.json",{"state":"training","step":step,"validation":latest,"elapsed_s":time.perf_counter()-started})
        print("VALIDATION "+json.dumps({"step":step,**latest}),flush=True)
        if backup is not None:model.load_state_dict(backup)
        model.train()

    def save_training(path,step):
        payload=checkpoint_payload(model,cfg,manifest,step,latest,optimizer,scaler)
        if ema:
            payload["ema"]=ema.state
            payload["validation_weights"]="EMA; inference best checkpoints contain these evaluated weights directly"
        atomic_torch_save(payload,path)

    evaluate_save(start_step)
    iterator=iter(loader);step=start_step;last_log=time.perf_counter();bucket=[]
    try:
        while step<cfg["train"]["steps"]:
            if stop_after and step>=stop_after:break
            if time.perf_counter()-started>cfg["train"]["max_hours"]*3600:break
            try:low,context,target=next(iterator)
            except StopIteration:iterator=iter(loader);low,context,target=next(iterator)
            low=low.cuda(non_blocking=True);context=context.cuda(non_blocking=True);target=target.cuda(non_blocking=True)
            step+=1
            warmup=cfg["train"]["warmup_steps"]
            if step<=warmup:multiplier=step/max(warmup,1)
            else:multiplier=.05+.95*.5*(1+math.cos(math.pi*(step-warmup)/max(cfg["train"]["steps"]-warmup,1)))
            for group in optimizer.param_groups:group["lr"]=cfg["train"]["learning_rate"]*multiplier
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16,enabled=cfg["train"]["amp"]):
                prediction=model(low,context)
                loss,parts=restoration_loss(prediction,target,cfg["train"]["ssim_weight"],cfg["train"]["chroma_weight"])
            perceptual_weight=cfg["train"].get("perceptual_weight",0.0)
            if perceptual_weight:
                # Frozen LPIPS parameters, but gradients propagate to prediction.
                with torch.autocast("cuda",enabled=False):
                    perceptual=evaluator.lpips(prediction.float().clamp(0,1)*2-1,target.float()*2-1).mean()
                loss=loss+perceptual_weight*perceptual
                parts["perceptual"]=float(perceptual.detach())
            if not torch.isfinite(loss):raise FloatingPointError(f"Non-finite training loss at step {step}")
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(optimizer);scaler.update()
            if ema:ema.update(model)
            bucket.append(float(loss.detach()))
            if step%100==0:
                now=time.perf_counter();row={"step":step,"loss":float(np.mean(bucket)),"lr":optimizer.param_groups[0]["lr"],
                                           "seconds_per_step":(now-last_log)/len(bucket),"elapsed_s":now-started,**parts}
                append_jsonl(logs/"train.jsonl",row);write_json(logs/"status.json",{"state":"training",**row})
                print("TRAIN "+json.dumps(row),flush=True);bucket=[];last_log=now
            if step%cfg["train"]["checkpoint_every"]==0:
                save_training(out/"last.pt",step)
            if step%cfg["train"]["validate_every"]==0:
                evaluate_save(step);last_log=time.perf_counter()
        if step%cfg["train"]["validate_every"]!=0 or step==start_step:evaluate_save(step)
        save_training(out/"last.pt",step)
        write_json(logs/"status.json",{"state":"completed","step":step,"elapsed_s":time.perf_counter()-started,
                                      "validation":latest,"best_psnr":best_psnr,"best_lpips":best_lpips})
    except BaseException as error:
        save_training(out/"interrupted.pt",step)
        write_json(logs/"status.json",{"state":"interrupted","step":step,"error":repr(error)})
        raise
    print(f"Training stage finished at step {step}; checkpoints: {out}",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/canon.json");p.add_argument("--name",default="main")
    p.add_argument("--resume");p.add_argument("--steps",type=int);p.add_argument("--stop-after",type=int,default=0)
    p.add_argument("--validation-limit",type=int,default=0);p.add_argument("--validate-every",type=int)
    p.add_argument("--chroma-weight",type=float);p.add_argument("--max-hours",type=float);p.add_argument("--workers",type=int)
    p.add_argument("--perceptual-weight",type=float);p.add_argument("--learning-rate",type=float)
    p.add_argument("--adaptive-proxy",action="store_true")
    p.add_argument("--ema-decay",type=float)
    a=p.parse_args();cfg=load_config(a.config)
    if a.adaptive_proxy:cfg["model"]["adaptive_proxy"]=True
    elif a.resume:
        prior=torch.load(a.resume,map_location="cpu",weights_only=True)
        cfg["model"]=prior["model_config"]
        del prior
    for key,value in [("steps",a.steps),("validate_every",a.validate_every),("chroma_weight",a.chroma_weight),("max_hours",a.max_hours),("num_workers",a.workers),("perceptual_weight",a.perceptual_weight),("learning_rate",a.learning_rate),("ema_decay",a.ema_decay)]:
        if value is not None:cfg["train"][key]=value
    train(cfg,a.name,a.resume,a.stop_after,a.validation_limit)
