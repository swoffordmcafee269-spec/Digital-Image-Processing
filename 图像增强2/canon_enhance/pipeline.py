from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .common import PROJECT_ROOT,load_config,write_json,read_manifest
from .prepare import verify_source_unchanged


def pipeline(config_path="configs/canon.json",wait_for_main=False):
    cfg=load_config(config_path);root=Path(cfg["artifact_root"]);logs=root/"logs";logs.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();deadline=started+24*3600

    def status(stage,state,**extra):
        payload={"stage":stage,"state":state,"elapsed_s":time.monotonic()-started,**extra}
        write_json(root/"pipeline_status.json",payload);print(json.dumps(payload),flush=True)

    def execute(stage,arguments):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError("24-hour pipeline budget exhausted")
        command=[sys.executable,"-X","utf8","-B",*arguments]
        status(stage,"running",command=command)
        with (logs/f"pipeline_{stage}.log").open("a",encoding="utf-8") as f:
            subprocess.run(command,cwd=PROJECT_ROOT,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remaining)
        status(stage,"completed")

    def stage_done(name,step):
        path=logs/name/"status.json"
        if not path.exists():return False
        obj=json.loads(path.read_text(encoding="utf-8"))
        return obj.get("state")=="completed" and obj.get("step",0)>=step

    try:
        execute("tests",["-m","unittest","discover","-s","tests","-v"])
        if not (root/"data"/"manifest.json").exists():execute("prepare",["-m","canon_enhance.prepare","--config",config_path])
        read_manifest(cfg);verify_source_unchanged(cfg)
        if wait_for_main:
            status("main","waiting_for_existing_training")
            while not stage_done("main",18000):
                path=logs/"main"/"status.json"
                if path.exists() and json.loads(path.read_text(encoding="utf-8")).get("state")=="interrupted":
                    raise RuntimeError("Existing main training was interrupted; resume it before continuing")
                if time.monotonic()>deadline:raise TimeoutError("Waiting for main training exceeded budget")
                time.sleep(10)
        elif not stage_done("main",18000):
            execute("main",["-m","canon_enhance.train","--config",config_path,"--name","main","--chroma-weight","0","--stop-after","18000"])
        for name,weight in [("chroma0","0"),("chroma005","0.05")]:
            if not stage_done(name,22000):
                execute(name,["-m","canon_enhance.train","--config",config_path,"--name",name,
                              "--resume",str(root/"checkpoints"/"main"/"last.pt"),"--chroma-weight",weight,"--ema-decay","0.999","--stop-after","22000"])
        candidates=[str(root/"checkpoints"/name/"best_psnr.pt") for name in ["main","chroma0","chroma005"]]
        if not stage_done("adaptive",24000):
            execute("adaptive",["-m","canon_enhance.train","--config",config_path,"--name","adaptive",
                                "--resume",str(root/"checkpoints"/"main"/"last.pt"),"--adaptive-proxy",
                                "--chroma-weight","0.05","--ema-decay","0.999","--learning-rate","0.0001","--stop-after","24000"])
        candidates.append(str(root/"checkpoints"/"adaptive"/"best_psnr.pt"))
        execute("selection",["-m","canon_enhance.select","--config",config_path,*candidates])
        choice=json.loads((root/"selection.json").read_text(encoding="utf-8"))["selected"]
        metrics=choice["metrics"]
        if metrics["psnr"]<24 or metrics["ssim"]<.82 or metrics["lpips"]>.35:
            # A bounded, low-weight perceptual experiment, selected on val only.
            target_step=choice["step"]+4000
            if not stage_done("perceptual",target_step):
                execute("perceptual",["-m","canon_enhance.train","--config",config_path,"--name","perceptual",
                                      "--resume",str(root/"checkpoints"/"selected.pt"),"--chroma-weight","0.05",
                                      "--perceptual-weight","0.02","--learning-rate","0.00005",
                                      "--ema-decay","0.999",
                                      "--steps",str(max(40000,target_step+2000)),"--stop-after",str(target_step)])
            candidates.extend([str(root/"checkpoints"/"perceptual"/"best_psnr.pt"),str(root/"checkpoints"/"perceptual"/"best_lpips.pt")])
            execute("selection_with_perceptual",["-m","canon_enhance.select","--config",config_path,*candidates])
        execute("final",["-m","canon_enhance.evaluate","--config",config_path,"--checkpoint",str(root/"checkpoints"/"selected.pt"),"--name","final","--resume"])
        execute("baseline",["-m","canon_enhance.evaluate","--config",config_path,"--baseline","--name","baseline","--resume"])
        for name in ["final","baseline"]:
            execute(f"verify_{name}",["-m","canon_enhance.verify","--config",config_path,"--name",name])
        execute("report",["-m","canon_enhance.report","--config",config_path])
        status("all","completed",source_check=verify_source_unchanged(cfg))
    except BaseException as error:
        status("pipeline","failed",error=repr(error));raise


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",default="configs/canon.json");p.add_argument("--wait-for-main",action="store_true")
    a=p.parse_args();pipeline(a.config,a.wait_for_main)
