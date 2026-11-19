"""Score already saved RGB PNGs against JPEG GT, without loading a model."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .common import configure_torch,safe_output,write_json
from .data import read_rgb
from .metrics import MetricEvaluator


def score_saved(images_dir,gt_dir,output_dir,device="cuda"):
    images=Path(images_dir).resolve();truth=Path(gt_dir).resolve();out=safe_output(output_dir)
    if out==images or out.is_relative_to(images) or out==truth or out.is_relative_to(truth):
        raise ValueError("Write scores into a separate project directory")
    files=sorted(images.glob("*.png"),key=lambda p:(0,int(p.stem)) if p.stem.isdigit() else (1,p.stem))
    if not files:raise FileNotFoundError(f"No PNG files found: {images}")
    missing=[p.stem for p in files if not (truth/f"{p.stem}.jpg").is_file()]
    if missing:raise FileNotFoundError(f"Missing JPEG references for IDs: {missing[:10]}")
    out.mkdir(parents=True,exist_ok=True)
    if (out/"metrics.csv").exists():raise FileExistsError("Choose a new score output directory")
    configure_torch();evaluator=MetricEvaluator(device);rows=[];started=time.perf_counter()
    keys=["psnr","ssim","lpips","delta_e_2000_512","saturation_pred_512","saturation_gt_512"]
    with (out/"metrics.csv").open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=["id",*keys]);writer.writeheader()
        for index,path in enumerate(files):
            prediction=read_rgb(path);target=read_rgb(truth/f"{path.stem}.jpg")
            metrics=evaluator(prediction,target)
            if not all(np.isfinite(metrics[k]) for k in keys):raise ValueError(f"Nonfinite score: {path.stem}")
            row={"id":path.stem,**metrics};writer.writerow(row);handle.flush();rows.append(row)
            if (index+1)%25==0:print(f"Scored {index+1}/{len(files)} saved PNGs",flush=True)
    summary={"images":len(rows),"metrics":{k:float(np.mean([r[k] for r in rows])) for k in keys},
             "elapsed_s":time.perf_counter()-started,"model_required":False,
             "prediction_dir":str(images),"gt_dir":str(truth),
             "protocol":"Saved uint8 RGB; full-resolution PSNR/SSIM/LPIPS Alex v0.1; per-image arithmetic mean; color diagnostics longest side 512"}
    write_json(out/"summary.json",summary);return summary


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("images_dir");p.add_argument("--gt-dir",required=True)
    p.add_argument("--output-dir",required=True);p.add_argument("--device",default="cuda")
    a=p.parse_args();print(json.dumps(score_saved(a.images_dir,a.gt_dir,a.output_dir,a.device),indent=2))
