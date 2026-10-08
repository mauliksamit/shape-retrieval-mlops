"""Preprocess step: verify/normalize point clouds or generate synthetic data.

- Normal path: ensure data/processed/modelnet10_{train,test}.npz exist (created by
  data_loader); re-normalize + validate shapes and write manifest.
- Synthetic path (--synthetic): generate random primitive-based clouds per class so
  the pipeline + CI run without downloading ModelNet10.

Usage:
    python -m src.preprocess [--synthetic --num-samples 120]
"""
import argparse
import json
import os

import numpy as np

from src.utils import load_config, set_seed, normalize_points, resample_points, ensure_dirs
from src.data_loader import MODELNET10_CLASSES

TRAIN_P = "data/processed/modelnet10_train.npz"
TEST_P = "data/processed/modelnet10_test.npz"


def make_synthetic_cloud(cls_idx, num_points, rng):
    """Distinct-but-confusable primitives: boxes (furniture) vs cylinders (toilet/bathtub)."""
    if cls_idx in (0, 9):  # bathtub / toilet — elongated + curved
        t = rng.uniform(0, 2 * np.pi, num_points)
        r = rng.normal(0.5, 0.08, num_points)
        pts = np.stack([np.cos(t) * r * 1.4, rng.normal(0, 0.25, num_points), np.sin(t) * r * 0.6], axis=1)
    elif cls_idx in (2, 8):  # chair / table — flat plane + legs
        n_top = num_points * 3 // 4
        top = np.stack([rng.uniform(-0.7, 0.7, n_top), np.zeros(n_top), rng.uniform(-0.7, 0.7, n_top)], 1)
        legs = np.stack([rng.uniform(-0.6, 0.6, num_points - n_top),
                         rng.uniform(-1, 0, num_points - n_top),
                         rng.uniform(-0.6, 0.6, num_points - n_top)], 1)
        pts = np.concatenate([top, legs], axis=0)
    else:  # boxy furniture — intentionally similar to induce desk/table-style confusion
        dims = np.array([0.6 + cls_idx * 0.03, 0.5 + (cls_idx % 3) * 0.1, 0.45 + (cls_idx % 4) * 0.05])
        pts = rng.uniform(-1, 1, (num_points, 3)) * dims
    pts += rng.normal(0, 0.02, pts.shape)
    return normalize_points(pts).astype(np.float32)


def write_synthetic(processed_dir, num_points, num_samples, seed):
    rng = np.random.default_rng(seed)
    ensure_dirs(processed_dir, "artifacts")
    per_split = {"train": int(num_samples * 0.8), "test": num_samples - int(num_samples * 0.8)}
    for split, n in per_split.items():
        pts, labels, ids = [], [], []
        for i in range(n):
            c = i % len(MODELNET10_CLASSES)
            pts.append(make_synthetic_cloud(c, num_points, rng))
            labels.append(MODELNET10_CLASSES[c])
            ids.append(f"{split}/synth_{i:04d}")
        arr = np.stack(pts)
        path = os.path.join(processed_dir, f"modelnet10_{split}.npz")
        np.savez_compressed(path, points=arr, labels=np.array(labels), ids=np.array(ids))
        print(f"Wrote synthetic {split}: {arr.shape} -> {path}")


def validate_and_manifest(processed_dir):
    manifest = {}
    for split in ("train", "test"):
        path = os.path.join(processed_dir, f"modelnet10_{split}.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing. Run `python -m src.data_loader` or `python -m src.preprocess --synthetic`.")
        d = np.load(path, allow_pickle=True)
        pts, labels = d["points"], [str(x) for x in d["labels"]]
        assert pts.ndim == 3 and pts.shape[2] == 3, f"bad points shape {pts.shape}"
        assert len(pts) == len(labels)
        # re-normalize defensively
        pts = np.stack([normalize_points(p) for p in pts]).astype(np.float32)
        np.savez_compressed(path, points=pts, labels=np.array(labels), ids=np.array([str(x) for x in d["ids"]]))
        counts = {c: int(np.sum(np.array(labels) == c)) for c in sorted(set(labels))}
        manifest[split] = {"path": path, "num_samples": len(pts), "num_points": int(pts.shape[1]),
                           "class_counts": counts}
        print(f"{split}: {pts.shape}, classes={counts}")
    with open(os.path.join(processed_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--num-points", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    dc = cfg["data"]
    set_seed(dc.get("seed", 42))
    num_points = args.num_points or dc.get("num_points", 1024)
    if args.synthetic:
        write_synthetic(dc["processed_dir"], num_points, args.num_samples, dc.get("seed", 42))
    validate_and_manifest(dc["processed_dir"])


if __name__ == "__main__":
    main()
