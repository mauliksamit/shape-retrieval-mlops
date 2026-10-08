"""Model + preprocessing unit tests (CPU, no data download)."""
import numpy as np

from src.utils import normalize_points, resample_points
from src.models.baseline import GeometricBaseline, geometric_features


def test_normalize_centers_and_scales():
    pts = np.array([[1, 0, 0], [3, 0, 0], [2, 0, 0]], dtype=np.float32)
    n = normalize_points(pts)
    assert abs(n.mean()) < 1e-5
    assert np.linalg.norm(n, axis=1).max() <= 1.0 + 1e-5


def test_resample_fixed_size():
    rng = np.random.default_rng(0)
    pts = rng.normal(size=(10, 3)).astype(np.float32)
    assert resample_points(pts, 1024, rng).shape == (1024, 3)
    assert resample_points(np.zeros((0, 3), dtype=np.float32), 8).shape == (8, 3)


def test_pointnet_forward_shape():
    torch = __import__("torch")
    from src.models.pointnet import PointNetClassifier
    m = PointNetClassifier(num_classes=10, embedding_dim=16)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(2, 32, 3))
        emb = m.encoder.embed(torch.randn(2, 32, 3))
    assert out.shape == (2, 10)
    assert emb.shape == (2, 16)


def test_baseline_features_and_pca():
    rng = np.random.default_rng(1)
    clouds = rng.normal(size=(20, 64, 3)).astype(np.float32)
    assert geometric_features(clouds[0]).shape[0] == 14
    base = GeometricBaseline(out_dim=5).fit(clouds)
    emb = base.transform(clouds)
    assert emb.shape == (20, 5)
