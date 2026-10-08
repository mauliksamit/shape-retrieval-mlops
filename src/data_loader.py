"""Download ModelNet10 and convert OFF meshes -> fixed-size point clouds (.npz).

Usage:
    python -m src.data_loader --max-samples-per-class 200 --num-points 1024
"""
import argparse
import os
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.utils import load_config, set_seed, normalize_points, resample_points, ensure_dirs

MODELNET10_CLASSES = ["bathtub", "bed", "chair", "desk", "dresser",
                      "monitor", "night_stand", "sofa", "table", "toilet"]


def download_modelnet(url, raw_dir):
    import requests
    ensure_dirs(raw_dir)
    zip_path = os.path.join(raw_dir, "ModelNet10.zip")
    extract_dir = os.path.join(raw_dir, "ModelNet10")
    if os.path.isdir(extract_dir):
        print(f"Already extracted: {extract_dir}")
        return extract_dir
    if not os.path.exists(zip_path):
        print(f"Downloading ModelNet10 (~150MB) from {url} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True) as bar:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
    print("Extracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(raw_dir)
    return extract_dir


def sample_off_to_points(off_path, num_points, rng):
    """Load OFF mesh with trimesh, uniformly sample surface points."""
    import trimesh
    mesh = trimesh.load(off_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geoms = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geoms) if geoms else None
    if mesh is None or len(mesh.faces) == 0:
        pts = np.asarray(mesh.vertices if mesh is not None else np.zeros((0, 3)), dtype=np.float32)
    else:
        pts, _ = trimesh.sample.sample_surface(mesh, num_points * 4, seed=int(rng.integers(1e9)))
        pts = np.asarray(pts, dtype=np.float32)
    pts = resample_points(pts, num_points, rng)
    return normalize_points(pts)


def collect_off_files(extract_dir, split):
    """Return {class: [off paths]} for train/test split."""
    out = {}
    for cls in MODELNET10_CLASSES:
        d = Path(extract_dir) / cls / split
        if d.is_dir():
            files = sorted(d.glob("*.off"))
            if files:
                out[cls] = [str(f) for f in files]
    return out


def build_npz(extract_dir, out_path, num_points, max_samples_per_class, seed):
    rng = np.random.default_rng(seed)
    for split in ("train", "test"):
        files = collect_off_files(extract_dir, split)
        if not files:
            raise FileNotFoundError(f"No OFF files found under {extract_dir} — download may have failed.")
        pts_list, labels, ids = [], [], []
        for cls in MODELNET10_CLASSES:
            cls_files = files.get(cls, [])
            if max_samples_per_class:
                cls_files = cls_files[:max_samples_per_class]
            for f in tqdm(cls_files, desc=f"{split}/{cls}"):
                try:
                    pts = sample_off_to_points(f, num_points, rng)
                except Exception as e:
                    print(f"skip {f}: {e}")
                    continue
                pts_list.append(pts)
                labels.append(cls)
                ids.append(f"{split}/{cls}/{Path(f).stem}")
        arr = np.stack(pts_list).astype(np.float32)
        np.savez_compressed(out_path.replace("{split}", split),
                            points=arr,
                            labels=np.array(labels),
                            ids=np.array(ids))
        print(f"Wrote {split}: {arr.shape} -> {out_path.replace('{split}', split)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--num-points", type=int, default=None)
    ap.add_argument("--max-samples-per-class", type=int, default=None)
    ap.add_argument("--processed-pattern", default="data/processed/modelnet10_{split}.npz")
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc = cfg["data"]
    num_points = args.num_points or dc.get("num_points", 1024)
    max_spc = args.max_samples_per_class
    if max_spc is None:
        max_spc = dc.get("max_samples_per_class", 200)
    set_seed(dc.get("seed", 42))
    extract_dir = download_modelnet(dc["url"], dc["raw_dir"])
    ensure_dirs(dc["processed_dir"], "artifacts")
    build_npz(extract_dir, args.processed_pattern, num_points, max_spc, dc.get("seed", 42))


if __name__ == "__main__":
    main()
