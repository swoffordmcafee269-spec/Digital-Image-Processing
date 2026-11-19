"""Enhance a directory of headerless Canon RAW files without a GT path."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from .common import configure_torch,file_sha256,safe_output,source_signature,write_json
from .data import read_raw,pack_raw,make_context
from .inference import load_checkpoint_model,predict_packed


def enhance_directory(input_dir,checkpoint,output_dir,device="cuda",tile_size=0,resume=False):
    source=Path(input_dir).resolve();out=safe_output(output_dir)
    files=sorted(source.glob("*.raw"),key=lambda p:(0,int(p.stem)) if p.stem.isdigit() else (1,p.stem))
    if not files:raise FileNotFoundError(f"No RAW files found: {source}")
    if out==source or out.is_relative_to(source):raise ValueError("Output must be outside the RAW source directory")
    out.mkdir(parents=True,exist_ok=True)
    identity={"input_dir":str(source),"checkpoint_sha256":file_sha256(checkpoint),
              "device":device,"tile_size":tile_size,
              "sources":{p.name:source_signature(p) for p in files},"gt_required":False}
    metadata=out/"batch_run.json"
    if resume and metadata.exists():
        if json.loads(metadata.read_text(encoding="utf-8"))!=identity:
            raise ValueError("Resume requires unchanged RAW files, checkpoint and inference settings")
    elif metadata.exists() or any(out.glob("*.png")):
        raise FileExistsError("Choose an empty output directory or resume the same run")
    write_json(metadata,identity)
    configure_torch();model,ck=load_checkpoint_model(checkpoint,device);spec=ck["input_spec"]
    started=time.perf_counter();written=0;skipped=0
    for index,path in enumerate(files):
        target=out/f"{path.stem}.png"
        if resume and target.exists():
            with Image.open(target) as image:
                if image.mode!="RGB" or image.size!=(spec["width"],spec["height"]):
                    raise ValueError(f"Invalid previously written image: {target}")
                image.verify()
            skipped+=1;continue
        raw=read_raw(path,spec["height"],spec["width"])
        packed=pack_raw(raw,spec["black_level"],spec["scale"])
        rgb=predict_packed(model,packed,make_context(packed),device,tile_size=tile_size)
        temporary=target.with_suffix(".png.tmp")
        Image.fromarray(rgb).save(temporary,format="PNG",compress_level=1);temporary.replace(target)
        written+=1
        if (index+1)%25==0 or index+1==len(files):
            print(json.dumps({"completed":index+1,"total":len(files),"elapsed_s":time.perf_counter()-started}),flush=True)
    result={"state":"completed","total":len(files),"written":written,"skipped":skipped,
            "elapsed_s":time.perf_counter()-started,"gt_required":False}
    write_json(out/"batch_status.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("input_dir");p.add_argument("--checkpoint",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--device",default="cuda");p.add_argument("--tile-size",type=int,default=0);p.add_argument("--resume",action="store_true")
    a=p.parse_args();print(json.dumps(enhance_directory(a.input_dir,a.checkpoint,a.output_dir,a.device,a.tile_size,a.resume)))
