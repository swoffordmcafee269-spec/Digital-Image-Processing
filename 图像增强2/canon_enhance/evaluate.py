from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_config,read_manifest,safe_output,write_json,configure_torch,file_sha256
from .data import load_full_pair,read_rgb
from .inference import load_checkpoint_model,predict_packed
from .baseline import calibrated_baseline
from .metrics import MetricEvaluator
from .prepare import verify_source_unchanged


def summarize(rows):
    keys=[k for k in ["psnr","ssim","lpips","delta_e_2000_512","saturation_pred_512","saturation_gt_512","inference_seconds","evaluation_seconds"] if k in rows[0]]
    result={}
    for split in ["all","train","val","test"]:
        selected=[r for r in rows if split=="all" or r["split"]==split]
        if not selected:continue
        result[split]={"count":len(selected),**{k:float(np.mean([float(r[k]) for r in selected])) for k in keys}}
    return result


def evaluate(cfg,checkpoint=None,split="all",name="final",baseline=False,limit=0,resume=False):
    configure_torch(cfg["eval"]["cpu_threads"]);manifest=read_manifest(cfg)
    out=safe_output(Path(cfg["artifact_root"])/"results"/name,cfg);images=out/"images";images.mkdir(parents=True,exist_ok=True)
    model=None;checkpoint_meta=None
    if not baseline:
        if not checkpoint:raise ValueError("A checkpoint is required")
        model,checkpoint_meta=load_checkpoint_model(checkpoint)
        if checkpoint_meta["manifest_fingerprint"]!=manifest["fingerprint"]:raise ValueError("Data manifest does not match checkpoint")
    evaluator=MetricEvaluator()
    items=[r for r in manifest["items"] if split=="all" or r["split"]==split]
    if limit:items=items[:limit]
    run_spec={"manifest_fingerprint":manifest["fingerprint"],"checkpoint":str(Path(checkpoint).resolve()) if checkpoint else None,
              "checkpoint_sha256":file_sha256(checkpoint) if checkpoint else None,
              "checkpoint_step":checkpoint_meta["step"] if checkpoint_meta else None,"baseline":baseline,"ids":[i["id"] for i in items],
              "metrics":{"domain":"saved/reloaded uint8 RGB PNG, original resolution","PSNR":"mean per-image dB",
                         "SSIM":"7x7 uniform window, sample covariance, RGB mean, data_range=1",
                         "LPIPS":"AlexNet v0.1, full resolution, FP32, input [-1,1]",
                         "color_diagnostics":"area thumbnail, longest side 512 pixels"}}
    run_file=out/"run.json"
    if resume and run_file.exists():
        prior=json.loads(run_file.read_text(encoding="utf-8"))
        if prior!=run_spec:raise ValueError("Cannot resume a different evaluation run into the same directory")
    elif (out/"metrics.csv").exists():raise FileExistsError("Use a new run name, or --resume with the same model")
    write_json(run_file,run_spec)
    rows=[];csv_path=out/"metrics.csv"
    if resume and csv_path.exists():
        rows=list(csv.DictReader(csv_path.open(encoding="utf-8")))
        if any(not (images/f"{r['id']}.png").exists() for r in rows):raise ValueError("A previously scored image is missing")
    done={r["id"] for r in rows};started=time.perf_counter()
    fields=["id","split","group","psnr","ssim","lpips","delta_e_2000_512","saturation_pred_512","saturation_gt_512","inference_seconds","evaluation_seconds"]
    with csv_path.open("a" if rows else "w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        if not rows:writer.writeheader()
        for item in items:
            if item["id"] in done:continue
            packed,context,gt=load_full_pair(cfg,manifest,item)
            t=time.perf_counter()
            output=calibrated_baseline(packed,manifest["camera_calibration"]) if baseline else predict_packed(model,packed,context)
            inference_seconds=time.perf_counter()-t
            path=images/f"{item['id']}.png";temp=path.with_suffix(".png.tmp")
            Image.fromarray(output).save(temp,format="PNG",compress_level=cfg["eval"]["png_compress_level"]);temp.replace(path)
            reloaded=read_rgb(path)
            if not np.array_equal(output,reloaded):raise RuntimeError("PNG round trip changed the pixels")
            t=time.perf_counter();metrics=evaluator(reloaded,gt)
            row={"id":item["id"],"split":item["split"],"group":item["group"],**metrics,
                 "inference_seconds":inference_seconds,"evaluation_seconds":time.perf_counter()-t}
            if not all(np.isfinite(float(row[k])) for k in ["psnr","ssim","lpips"]):raise RuntimeError(f"Invalid metrics: {row}")
            writer.writerow(row);handle.flush();rows.append(row)
            if len(rows)%25==0 or len(rows)==len(items):
                status={"state":"running","completed":len(rows),"total":len(items),"elapsed_s":time.perf_counter()-started,"summary":summarize(rows)}
                write_json(out/"status.json",status);print(json.dumps(status),flush=True)
    summary=summarize(rows);source_check=verify_source_unchanged(cfg)
    write_json(out/"summary.json",{"metrics":summary,"source_check":source_check,"complete":len(rows)==len(items),
                                   "elapsed_this_run_s":time.perf_counter()-started,"run":run_spec})
    write_json(out/"status.json",{"state":"completed","completed":len(rows),"total":len(items),"summary":summary})
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/canon.json");p.add_argument("--checkpoint")
    p.add_argument("--split",choices=["all","train","val","test"],default="all");p.add_argument("--name",default="final")
    p.add_argument("--baseline",action="store_true");p.add_argument("--limit",type=int,default=0);p.add_argument("--resume",action="store_true")
    a=p.parse_args();evaluate(load_config(a.config),a.checkpoint,a.split,a.name,a.baseline,a.limit,a.resume)
