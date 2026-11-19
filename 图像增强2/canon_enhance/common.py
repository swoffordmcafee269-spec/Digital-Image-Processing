from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path="configs/canon.json"):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["data_root"] = str(Path(cfg["data_root"]).resolve())
    out = Path(cfg["artifact_root"])
    cfg["artifact_root"] = str((out if out.is_absolute() else PROJECT_ROOT / out).resolve())
    safe_output(cfg["artifact_root"], cfg)
    return cfg


def safe_output(path, cfg=None):
    p = Path(path).resolve()
    if not p.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"All experiment writes must stay inside the project: {p}")
    if cfg and p.is_relative_to(Path(cfg["data_root"]).resolve()):
        raise ValueError("The original dataset is read-only.")
    return p


def write_json(path, value):
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_torch(threads=8):
    import torch
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def read_manifest(cfg):
    p = Path(cfg["artifact_root"]) / "data" / "manifest.json"
    manifest=json.loads(p.read_text(encoding="utf-8"))
    if manifest.get("fingerprint")!=fingerprint({k:v for k,v in manifest.items() if k!="fingerprint"}):
        raise ValueError("Prepared manifest was changed; restore the original split/calibration or prepare a new experiment")
    for key in ["height","width","seed"]:
        if manifest[key]!=cfg[key]:raise ValueError(f"Configuration {key} differs from the prepared manifest")
    if manifest["normalization"]["black_level"]!=cfg["black_level"]:
        raise ValueError("Black level differs from the prepared calibration")
    return manifest


def source_paths(cfg, item):
    root = Path(cfg["data_root"])
    return root / item["raw"], root / item["gt"]


def source_signature(path):
    stat = Path(path).stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_torch_save(value, path):
    import torch
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def append_jsonl(path, value):
    path = safe_output(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, ensure_ascii=False) + "\n")


def worker_init(worker_id):
    import torch
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)
