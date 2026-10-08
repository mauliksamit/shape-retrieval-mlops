"""FastAPI 3D shape retrieval service.

POST /retrieve {"points": [[x,y,z],...], "k": 5} -> top-k gallery shapes by FAISS.
Embeddings come from the trained PointNet encoder (or PCA baseline).
Index + metadata load once at startup; missing artifacts -> 503 with clear message.
"""
import json
import os
import pickle
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.utils import normalize_points, resample_points

app = FastAPI(title="3D Shape Retrieval", version="0.1.0")

INDEX_PATH = os.environ.get("INDEX_PATH", "artifacts/index.faiss")
METADATA_PATH = os.environ.get("METADATA_PATH", "artifacts/metadata.json")
ENCODER_PATH = os.environ.get("ENCODER_PATH", "artifacts/pointnet.pt")
BASELINE_PATH = os.environ.get("BASELINE_PATH", "artifacts/baseline.pkl")
MODEL = os.environ.get("MODEL", "pointnet")
NUM_POINTS = int(os.environ.get("NUM_POINTS", "1024"))

_state = {"index": None, "meta": None, "encoder": None, "baseline": None, "classes": []}


class RetrieveRequest(BaseModel):
    points: List[List[float]] = Field(..., description="Nx3 point cloud")
    k: int = Field(5, ge=1, le=50)
    model: Optional[str] = None


def _load():
    if _state["index"] is not None:
        return
    if not (os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)):
        raise RuntimeError(f"Missing {INDEX_PATH} or {METADATA_PATH}; run build_index first.")
    import faiss
    _state["index"] = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH) as f:
        _state["meta"] = json.load(f)
    _state["classes"] = sorted({m["label"] for m in _state["meta"]})
    if MODEL == "pointnet" and os.path.exists(ENCODER_PATH):
        import torch
        from src.models.pointnet import PointNetClassifier
        ckpt = torch.load(ENCODER_PATH, map_location="cpu")
        ncls = len(ckpt.get("classes", _state["classes"]) or [0] * 10)
        model = PointNetClassifier(num_classes=max(ncls, 2), embedding_dim=ckpt.get("embedding_dim", 64))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _state["encoder"] = model
    elif os.path.exists(BASELINE_PATH):
        with open(BASELINE_PATH, "rb") as f:
            _state["baseline"] = pickle.load(f)


def embed_query(points: np.ndarray) -> np.ndarray:
    pts = resample_points(np.asarray(points, dtype=np.float32), NUM_POINTS)
    pts = normalize_points(pts)[None, ...]  # (1, N, 3)
    if _state["encoder"] is not None:
        import torch
        with torch.no_grad():
            return _state["encoder"].encoder.embed(torch.from_numpy(pts)).numpy().astype(np.float32)
    if _state["baseline"] is not None:
        return _state["baseline"].transform(pts).astype(np.float32)
    raise RuntimeError("No encoder available: train a model first.")


@app.on_event("startup")
def startup():
    try:
        _load()
    except Exception as e:
        print(f"WARN: startup load deferred ({e})")


@app.get("/health")
def health():
    try:
        _load()
        return {"status": "ok", "indexed": _state["index"].ntotal, "model": MODEL}
    except Exception as e:
        raise HTTPException(503, str(e))


@app.get("/classes")
def classes():
    try:
        _load()
    except Exception as e:
        raise HTTPException(503, str(e))
    return {"classes": _state["classes"]}


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    try:
        _load()
    except Exception as e:
        raise HTTPException(503, str(e))
    if len(req.points) == 0:
        raise HTTPException(400, "points must be non-empty Nx3")
    try:
        q = embed_query(np.asarray(req.points, dtype=np.float32))
    except Exception as e:
        raise HTTPException(400, f"bad points: {e}")
    k = min(req.k, _state["index"].ntotal)
    dists, idxs = _state["index"].search(q, k)
    return {"results": [{"id": _state["meta"][i]["id"], "label": _state["meta"][i]["label"],
                         "distance": float(d)} for i, d in zip(idxs[0], dists[0])]}
