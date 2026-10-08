"""Baseline: bag of geometric features + PCA -> fixed embedding.

Features per cloud (14-d): centroid (0 after norm, kept for robustness), std per axis,
bbox extents + volume + aspect ratios, mean/std radius, planarity-ish eigenvalue ratios
of the covariance. PCA reduces to `out_dim`. No gradient training; fits in seconds.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def geometric_features(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    centroid = p.mean(axis=0)
    std = p.std(axis=0)
    mins, maxs = p.min(axis=0), p.max(axis=0)
    extents = maxs - mins + 1e-8
    volume = float(np.prod(extents))
    aspects = np.array([extents[0] / extents[1], extents[1] / extents[2], extents[0] / extents[2]])
    radii = np.linalg.norm(p - centroid, axis=1)
    cov = np.cov(p.T) + np.eye(3) * 1e-9
    eig = np.sort(np.linalg.eigvalsh(cov))[::-1]  # l1 >= l2 >= l3
    eig_ratios = np.array([eig[1] / eig[0], eig[2] / eig[0]])
    return np.concatenate([std, extents, [volume], aspects, [radii.mean(), radii.std()], eig_ratios]).astype(np.float64)


class GeometricBaseline:
    def __init__(self, out_dim=32):
        self.out_dim = out_dim
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=out_dim)

    def _feats(self, clouds: np.ndarray) -> np.ndarray:
        return np.stack([geometric_features(p) for p in clouds])

    def fit(self, clouds: np.ndarray):
        feats = self._feats(clouds)
        self.scaler.fit(feats)
        self.pca.fit(self.scaler.transform(feats))
        return self

    def transform(self, clouds: np.ndarray) -> np.ndarray:
        feats = self._feats(np.asarray(clouds))
        emb = self.pca.transform(self.scaler.transform(feats))
        emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
        return emb.astype(np.float32)
