from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

from .common import (load_config, safe_output, write_json, source_signature,
                     fingerprint, source_paths)
from .data import read_raw, pack_raw, srgb_to_linear_np


class UnionFind:
    def __init__(self, n): self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a, b)] = min(a, b)


def scene_groups(ids, thumbnails, features):
    """Group overlapping captures using ORB geometry and close thumbnails.

    Automatic grouping is a leakage precaution, not guaranteed scene labels.
    Persist contact sheets so ambiguous group boundaries can be inspected.
    """
    uf = UnionFind(len(ids))
    hashes, vectors = [], []
    for i, (name, rgb) in enumerate(zip(ids, thumbnails)):
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
        dct = cv2.dct(cv2.resize(gray, (32, 32)))[:8, :8].ravel()[1:]
        bits = dct > np.median(dct)
        hashes.append(sum(int(v) << k for k, v in enumerate(bits)))
        vec = cv2.resize(gray, (32, 18)).ravel(); vec -= vec.mean()
        vectors.append(vec / max(np.linalg.norm(vec), 1e-8))
    vectors = np.stack(vectors)
    links = []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    cv2.setRNGSeed(20261021)
    for i in range(len(ids)):
        for j in range(i):
            if uf.find(i) == uf.find(j): continue
            hd = (hashes[i] ^ hashes[j]).bit_count()
            if hd <= 8 and float(vectors[i] @ vectors[j]) >= 0.92:
                uf.union(i, j); links.append([ids[j], ids[i], "thumbnail"])
                continue
            if abs(int(ids[i])-int(ids[j])) > 12 and hd > 14: continue
            kp1,des1=features[i];kp2,des2=features[j]
            if des1 is None or des2 is None or min(len(des1),len(des2))<20:continue
            matches=matcher.knnMatch(des1,des2,k=2)
            good=[pair[0] for pair in matches if len(pair)==2 and pair[0].distance<.75*pair[1].distance]
            if len(good)<20:continue
            a=np.float32([kp1[m.queryIdx] for m in good]);b=np.float32([kp2[m.trainIdx] for m in good])
            matrix,mask=cv2.findHomography(a,b,cv2.RANSAC,3.0,maxIters=1000,confidence=.99)
            if matrix is None or mask is None:continue
            mask=mask.ravel().astype(bool);count=int(mask.sum())
            if count<18 or count/len(good)<.65:continue
            coverage=min(float(np.prod(np.ptp(a[mask],axis=0))),float(np.prod(np.ptp(b[mask],axis=0))))/(480*270)
            if coverage<.08:continue
            uf.union(i,j);links.append([ids[j],ids[i],"orb",count,round(coverage,3)])
    groups = {}
    for i in range(len(ids)): groups.setdefault(uf.find(i), []).append(i)
    return list(groups.values()), links


def split_groups(groups, n, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(groups)).tolist()
    assignment = ["train"] * n
    remaining = order[:]
    for split in ["test", "val"]:
        target, count = round(n * .1), 0
        while remaining:
            k = min(remaining, key=lambda g: abs(target - (count + len(groups[g]))))
            next_count = count + len(groups[k])
            if count and abs(target - next_count) >= abs(target - count): break
            remaining.remove(k)
            for idx in groups[k]: assignment[idx] = split
            count = next_count
            if count == target: break
    return assignment


def prepare(cfg, cache_gt=True):
    started = time.perf_counter()
    root = Path(cfg["data_root"])
    raw_dir, gt_dir = root / "Canon_0.0001" / "0.0001", root / "Canon_gt" / "gt"
    raw_files = {p.stem: p for p in raw_dir.glob("*.raw")}
    gt_files = {p.stem: p for p in gt_dir.glob("*.jpg")}
    if set(raw_files) != set(gt_files) or len(raw_files) != 1000:
        raise ValueError("Expected exactly 1000 complete Canon pairs")
    ids = sorted(raw_files, key=int)
    out = safe_output(Path(cfg["artifact_root"]) / "data", cfg); out.mkdir(parents=True, exist_ok=True)
    cache_dir = safe_output(Path(cfg["artifact_root"]) / "cache" / "gt", cfg)
    if cache_gt: cache_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    def inspect(name):
        raw = read_raw(raw_files[name], cfg["height"], cfg["width"])
        rng = np.random.default_rng(cfg["seed"] + int(name))
        sample = np.asarray(raw).reshape(-1)[rng.integers(0, raw.size, 4096)].astype(np.float32)
        packed = pack_raw(raw, cfg["black_level"], 1.0)
        small_raw = cv2.resize(packed.transpose(1, 2, 0), (128, 72), interpolation=cv2.INTER_AREA)
        with Image.open(gt_files[name]) as image:
            if image.size != (cfg["width"], cfg["height"]) or image.mode != "RGB":
                raise ValueError(f"Invalid GT image {name}")
            rgb = np.asarray(image).copy()
        cache = cache_dir / f"{name}.npy"
        if cache_gt and not cache.exists():
            temp = cache.with_suffix(".npy.tmp")
            with temp.open("wb") as f: np.save(f, rgb)
            temp.replace(cache)
        thumb = cv2.resize(rgb, (128, 72), interpolation=cv2.INTER_AREA)
        feature_image=cv2.cvtColor(cv2.resize(rgb,(480,270),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2GRAY)
        keypoints,descriptors=cv2.ORB_create(nfeatures=600,fastThreshold=10).detectAndCompute(feature_image,None)
        features=(np.asarray([p.pt for p in keypoints],dtype=np.float32),descriptors)
        # Average in linear light for camera calibration.
        linear = srgb_to_linear_np(rgb.astype(np.float32) / 255)
        small_linear = cv2.resize(linear, (128, 72), interpolation=cv2.INTER_AREA)
        return {"sample":sample, "raw_small":small_raw, "gt_linear":small_linear,
                "thumb":thumb, "features":features,
                "context":cv2.resize(packed.transpose(1,2,0),(32,32),interpolation=cv2.INTER_AREA).transpose(2,0,1),
                "minimum":int(raw.min()), "maximum":int(raw.max())}

    with ThreadPoolExecutor(max_workers=4) as pool:
        inspected = list(tqdm(pool.map(inspect, ids), total=len(ids), desc="Audit / cache"))
    groups, links = scene_groups(ids, [r["thumb"] for r in inspected], [r["features"] for r in inspected])
    split = split_groups(groups, len(ids), cfg["seed"])
    if min(split.count("train"), split.count("val"), split.count("test")) < 50:
        raise ValueError("Scene grouping left too few independent samples")
    group_id = {}
    for i, members in enumerate(groups):
        for j in members: group_id[j] = f"scene_{i:03d}"
    train_idx = [i for i, s in enumerate(split) if s == "train"]
    signal = np.concatenate([np.maximum(inspected[i]["sample"] - cfg["black_level"], 0) for i in train_idx])
    scale = float(max(np.percentile(signal, 99.5), 1.0))
    # Train-only least-squares camera mapping. No per-evaluation-image GT fit.
    xtx, xty = np.zeros((5, 5)), np.zeros((5, 3))
    for i in train_idx:
        x = inspected[i]["raw_small"].reshape(-1, 4).astype(np.float64) / scale
        y = inspected[i]["gt_linear"].reshape(-1, 3).astype(np.float64)
        keep = (y.max(1) < .98) & (y.mean(1) > .002)
        x, y = x[keep][::4], y[keep][::4]
        x = np.concatenate([x, np.ones((len(x), 1))], 1)
        xtx += x.T @ x; xty += x.T @ y
    ridge = np.eye(5) * max(float(np.trace(xtx[:4, :4])) * 1e-5, 1e-5)
    ridge[-1, -1] = 1e-3
    weights = np.linalg.solve(xtx + ridge, xty).T.astype(np.float32)
    contexts = np.stack([r["context"] / scale for r in inspected])
    np.save(out / "contexts.npy", contexts)
    items = []
    for i, name in enumerate(ids):
        items.append({"id":name, "index":i, "group":group_id[i], "split":split[i],
                      "raw":raw_files[name].relative_to(root).as_posix(),
                      "gt":gt_files[name].relative_to(root).as_posix(),
                      "raw_signature":source_signature(raw_files[name]),
                      "gt_signature":source_signature(gt_files[name])})
    manifest = {"version":1, "seed":cfg["seed"], "height":cfg["height"], "width":cfg["width"],
                "normalization":{"black_level":cfg["black_level"], "scale":scale,
                                 "method":"training RAW pooled signal 99.5 percentile; no upper clipping"},
                "camera_calibration":{"matrix":weights[:, :4].tolist(), "bias":weights[:, 4].tolist(),
                                      "fit_split":"train", "space":"linear-sRGB", "train_images":len(train_idx)},
                "grouping":{"method":"pHash<=8 and centered correlation>=0.92; or ORB ratio<0.75, RANSAC>=18 inliers, inlier ratio>=0.65, coverage>=0.08; candidates within 12 IDs or pHash<=14",
                            "groups":len(groups), "cross_block_links":links,
                            "limitation":"Heuristic scene grouping; not verified official scene labels."},
                "counts":{s:split.count(s) for s in ["train", "val", "test"]}, "items":items}
    manifest["fingerprint"] = fingerprint(manifest)
    write_json(out / "manifest.json", manifest)
    for s in ["train", "val", "test", "all"]:
        (out / f"{s}.txt").write_text("\n".join(r["id"] for r in items if s == "all" or r["split"] == s) + "\n", encoding="utf-8")
    # Compact review sheets for all validation/test images and one per scene.
    for name, chosen in [("val", [i for i,s in enumerate(split) if s=="val"]),
                         ("test", [i for i,s in enumerate(split) if s=="test"]),
                         ("groups", [g[0] for g in groups])]:
        cols, cellw, cellh = 8, 160, 110
        canvas = Image.new("RGB", (cols*cellw, ((len(chosen)+cols-1)//cols)*cellh), "white")
        draw = ImageDraw.Draw(canvas)
        for k,i in enumerate(chosen):
            x,y=(k%cols)*cellw,(k//cols)*cellh
            thumb=Image.fromarray(inspected[i]["thumb"]).resize((160,90))
            canvas.paste(thumb,(x,y));draw.text((x+3,y+92),f"{ids[i]} {split[i]} {group_id[i]}",fill="black")
        canvas.save(out / f"contact_{name}.jpg", quality=90)
    summary={"elapsed_s":time.perf_counter()-started, "pairs":len(items), "counts":manifest["counts"],
             "groups":len(groups), "scale":scale, "raw_min":min(r["minimum"] for r in inspected),
             "raw_max":max(r["maximum"] for r in inspected), "calibration":manifest["camera_calibration"],
             "source_read_only":True, "cache_gt":cache_gt,
             "pretrained_status":"Public source fetch timed out; no verified pretrained checkpoint used."}
    write_json(out / "audit.json", summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def verify_source_unchanged(cfg):
    from .common import read_manifest
    manifest = read_manifest(cfg); changed=[]
    for item in manifest["items"]:
        raw,gt=source_paths(cfg,item)
        for label,path in [("raw",raw),("gt",gt)]:
            if source_signature(path)!=item[f"{label}_signature"]:changed.append(str(path))
    if changed: raise RuntimeError(f"Source file metadata changed: {changed[:10]}")
    return {"checked_files":len(manifest["items"])*2,"changed":0,
            "check":"size and modification time against preparation manifest"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/canon.json")
    parser.add_argument("--no-cache",action="store_true");parser.add_argument("--verify-source",action="store_true")
    args=parser.parse_args();config=load_config(args.config)
    if args.verify_source: print(json.dumps(verify_source_unchanged(config)))
    else: prepare(config,not args.no_cache)
