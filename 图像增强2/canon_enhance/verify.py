from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_config,read_manifest,write_json,file_sha256
from .prepare import verify_source_unchanged


def verify(cfg,name="final"):
    manifest=read_manifest(cfg);root=Path(cfg["artifact_root"]);out=root/"results"/name
    rows=list(csv.DictReader((out/"metrics.csv").open(encoding="utf-8")))
    expected={r["id"] for r in manifest["items"]};actual=[r["id"] for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise AssertionError("Metrics must contain each of the 1000 IDs exactly once")
    files={p.stem:p for p in (out/"images").glob("*.png")}
    if set(files)!=expected:raise AssertionError("Output PNG IDs do not exactly match the source IDs")
    for name_,path in files.items():
        with Image.open(path) as im:
            if im.mode!="RGB" or im.size!=(cfg["width"],cfg["height"]):raise AssertionError(f"Bad PNG {name_}")
            im.verify()
    for r in rows:
        if not all(np.isfinite(float(r[k])) for k in ["psnr","ssim","lpips"]):raise AssertionError(f"Non-finite score: {r['id']}")
    sets={s:{r["group"] for r in manifest["items"] if r["split"]==s} for s in ["train","val","test"]}
    for a,b in [("train","val"),("train","test"),("val","test")]:
        if sets[a]&sets[b]:raise AssertionError("A scene group crosses data splits")
    run=json.loads((out/"run.json").read_text(encoding="utf-8"))
    if run.get("checkpoint") and file_sha256(run["checkpoint"])!=run["checkpoint_sha256"]:
        raise AssertionError("The scored checkpoint was changed after evaluation")
    result={"passed":True,"images":len(files),"metric_rows":len(rows),"resolution":[cfg["width"],cfg["height"]],
            "mode":"RGB","groups_disjoint":True,"source_check":verify_source_unchanged(cfg),
            "checkpoint_hash_verified":bool(run.get("checkpoint")),"manifest_fingerprint":manifest["fingerprint"]}
    result["color_review_candidates"]=[r["id"] for r in rows if float(r["saturation_gt_512"])>.1 and float(r["saturation_pred_512"])<.02]
    result["mean_saturation_ratio"]=float(np.mean([float(r["saturation_pred_512"]) for r in rows])/max(np.mean([float(r["saturation_gt_512"]) for r in rows]),1e-8))
    result["note"]="Structural checks passed; image quality and target thresholds are reported separately. Saturation alone does not establish color correctness."
    write_json(out/"verification.json",result);print(json.dumps(result,ensure_ascii=False,indent=2));return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/canon.json");p.add_argument("--name",default="final")
    a=p.parse_args();verify(load_config(a.config),a.name)
