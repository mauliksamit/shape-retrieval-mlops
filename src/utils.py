"""Shared helpers: config, seed, point-cloud ops."""
import os
import random
import yaml
import numpy as np


def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def normalize_points(points: np.ndarray) -> np.ndarray:
    """Center to zero mean and scale to unit sphere. (N,3) -> (N,3)."""
    points = np.asarray(points, dtype=np.float32)
    points = points - points.mean(axis=0, keepdims=True)
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 1e-8:
        points = points / scale
    return points.astype(np.float32)


def resample_points(points: np.ndarray, num_points: int, rng=None) -> np.ndarray:
    """Resample (possibly variable-length) cloud to fixed size with replacement."""
    points = np.asarray(points, dtype=np.float32)
    n = len(points)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    rng = np.random.default_rng() if rng is None else rng
    if n >= num_points:
        idx = rng.choice(n, num_points, replace=False)
    else:
        idx = rng.choice(n, num_points, replace=True)
    return points[idx].astype(np.float32)


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
