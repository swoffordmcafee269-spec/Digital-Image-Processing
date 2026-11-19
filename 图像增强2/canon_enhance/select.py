from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import torch

from .common import load_config,read_manifest,safe_output,write_json,file_sha256,configure_torch,atomic_torch_save
from .inference import load_checkpoint_model
from .metrics import MetricEvaluator
from .train import validate


def select(cfg,candidates):
    """Select only on the same full validation set; test remains untouched."""
    configure_torch(cfg["eval"]["cpu_threads"]);manifest=read_manifest(cfg);evaluator=MetricEvaluator();results=[]
    root=Path(cfg["artifact_root"])
    candidates=list(candidates)
    prior_selection=root/"selection.json"
    if prior_selection.exists():
        prior=json.loads(prior_selection.read_text(encoding="utf-8"))["selected"]
        path=Path(prior["path"])
        if path.exists() and file_sha256(path)==prior["sha256"] and str(path.resolve()) not in {str(Path(p).resolve()) for p in candidates}:
            candidates.append(str(path))
    def score(path,index):
        model,checkpoint=load_checkpoint_model(path)
        if checkpoint["manifest_fingerprint"]!=manifest["fingerprint"]:raise ValueError("Candidate was trained under a different split")
        metrics=validate(model,cfg,manifest,evaluator,root/"logs"/"selection"/f"candidate_{index}")
        if "inference_ablation" in checkpoint:
            checkpoint["validation"]=metrics
            atomic_torch_save(checkpoint,path)
        results.append({"path":str(Path(path).resolve()),"sha256":file_sha256(path),"step":checkpoint["step"],
                        "chroma_weight":checkpoint["config"]["train"]["chroma_weight"],
                        "model_config":checkpoint["model_config"],"metrics":metrics})
        del model
    for index,path in enumerate(candidates):score(path,index)
    # Bounded validation-only ablation: suppress noise in the explicit ISP
    # prior while leaving the learned RAW/detail branch intact. No GT enters
    # inference. Both kernel sizes are fixed before either is evaluated.
    leader=max(results,key=lambda r:r["metrics"]["psnr"])
    for kernel in [21,41]:
        checkpoint=torch.load(leader["path"],map_location="cpu",weights_only=True)
        if checkpoint["model_config"].get("proxy_blur",5)==kernel:continue
        checkpoint["model_config"]={**checkpoint["model_config"],"proxy_blur":kernel}
        checkpoint["config"]["model"]=checkpoint["model_config"]
        checkpoint["inference_ablation"]={"source_sha256":leader["sha256"],"proxy_blur":kernel,"selected_on":"validation only"}
        checkpoint.pop("optimizer",None);checkpoint.pop("scaler",None);checkpoint.pop("ema",None)
        destination=root/"checkpoints"/f"prior_blur_{kernel}"/f"from_{leader['sha256'][:12]}.pt"
        atomic_torch_save(checkpoint,destination);del checkpoint
        score(destination,len(results))
    # PSNR is primary; within 0.10 dB use LPIPS, then color error as tie breakers.
    maximum=max(r["metrics"]["psnr"] for r in results)
    shortlist=[r for r in results if r["metrics"]["psnr"]>=maximum-.10]
    chosen=min(shortlist,key=lambda r:(r["metrics"]["lpips"],r["metrics"]["delta_e_2000_512"]))
    destination=safe_output(root/"checkpoints"/"selected.pt",cfg)
    shutil.copyfile(chosen["path"],destination)
    write_json(root/"selection.json",{"criterion":"Highest full-val PSNR; candidates within 0.10dB use lowest LPIPS, then lowest color error",
                                      "candidates":results,"selected":chosen,"output":str(destination),"test_used":False})
    print(f"Selected {chosen['path']}")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/canon.json");p.add_argument("candidates",nargs='+')
    a=p.parse_args();select(load_config(a.config),a.candidates)
