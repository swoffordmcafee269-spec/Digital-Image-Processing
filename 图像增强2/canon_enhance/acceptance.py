"""Integration checks against the frozen, actually delivered checkpoint."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .common import configure_torch,load_config,read_manifest,write_json,file_sha256,source_paths
from .data import read_raw,read_rgb,pack_raw,make_context
from .inference import enhance,load_checkpoint_model,predict_packed
from .metrics import MetricEvaluator


def acceptance(cfg,name="final",ident="10"):
    configure_torch();root=Path(cfg["artifact_root"]);result=root/"results"/name
    run=json.loads((result/"run.json").read_text(encoding="utf-8"));checkpoint=run["checkpoint"]
    if file_sha256(checkpoint)!=run["checkpoint_sha256"]:raise AssertionError("Frozen checkpoint changed")
    manifest=read_manifest(cfg);items={i["id"]:i for i in manifest["items"]}
    raw_path,_=source_paths(cfg,items[ident])
    # This public call has no GT argument and does not require a manifest.
    independent=enhance(raw_path,checkpoint)
    saved=read_rgb(result/"images"/f"{ident}.png")
    error=np.abs(independent.astype(np.int16)-saved.astype(np.int16))
    if error.max()>1:raise AssertionError("Independent RAW-only inference differs from delivered PNG")
    model,ck=load_checkpoint_model(checkpoint);spec=ck["input_spec"]
    packed=pack_raw(read_raw(raw_path,spec["height"],spec["width"]),spec["black_level"],spec["scale"])
    tiled=predict_packed(model,packed,make_context(packed),tile_size=512)
    tile_error=np.abs(tiled.astype(np.int16)-independent.astype(np.int16))
    # AMP convolutions can differ slightly for different input tensor shapes.
    # Report the maximum and percentiles; material differences fail this check.
    if tile_error.mean()>.05 or np.percentile(tile_error,99)>1:
        raise AssertionError(f"Tiled inference differs materially: mean={tile_error.mean()}, max={tile_error.max()}")
    del model
    evaluator=MetricEvaluator();rows={r["id"]:r for r in csv.DictReader((result/"metrics.csv").open(encoding="utf-8"))}
    rechecks=[]
    for sample in ["10","100","798"]:
        _,gt_path=source_paths(cfg,items[sample])
        scores=evaluator(read_rgb(result/"images"/f"{sample}.png"),read_rgb(gt_path))
        differences={k:abs(scores[k]-float(rows[sample][k])) for k in ["psnr","ssim","lpips"]}
        if max(differences.values())>1e-4:raise AssertionError(f"Metric recheck mismatch: {sample}: {differences}")
        rechecks.append({"id":sample,"metric_absolute_differences":differences})
    payload={"passed":True,"checkpoint_sha256":run["checkpoint_sha256"],"raw_only_inference_id":ident,
             "raw_only_vs_delivered_max_uint8_error":int(error.max()),
             "tiled_vs_full_mean_uint8_error":float(tile_error.mean()),"tiled_vs_full_max_uint8_error":int(tile_error.max()),
             "tiled_vs_full_p99_uint8_error":float(np.percentile(tile_error,99)),"metric_rechecks":rechecks}
    write_json(result/"acceptance.json",payload);print(json.dumps(payload,indent=2));return payload


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--config",default="configs/canon.json")
    p.add_argument("--name",default="final");p.add_argument("--id",default="10")
    a=p.parse_args();acceptance(load_config(a.config),a.name,a.id)
