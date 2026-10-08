"""Build FAISS index over test-gallery embeddings + metadata.

Usage: python -m src.build_index --model pointnet
Produces artifacts/index.faiss + artifacts/metadata.json (+ model-specific variants).
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from src.utils import load_config, set_seed, ensure_dirs
from src.models.pointnet import PointNetClassifier
from src.data_loader import MODELNET10_CLASSES


def embed_gallery(model_name, pts):
    if model_name == "pointnet":
        ckpt = torch.load("artifacts/pointnet.pt", map_location="cpu")
        model = PointNetClassifier(num_classes=len(MODELNET10_CLASSES),
                                   embedding_dim=ckpt.get("embedding_dim", 64))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            return model.encoder.embed(torch.from_numpy(pts)).numpy().astype(np.float32)
    with open("artifacts/baseline.pkl", "rb") as f:
        return pickle.load(f).transform(pts).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--model", default=None, help="pointnet|baseline (default from config)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["data"].get("seed", 42))
    model_name = args.model or cfg["index"].get("model", "pointnet")
    d = np.load(os.path.join(cfg["data"]["processed_dir"], "modelnet10_test.npz"), allow_pickle=True)
    pts, labels, ids = d["points"].astype(np.float32), [str(x) for x in d["labels"]], [str(x) for x in d["ids"]]
    emb = embed_gallery(model_name, pts)
    import faiss
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    ensure_dirs("artifacts")
    faiss.write_index(index, "artifacts/index.faiss")
    with open("artifacts/metadata.json", "w") as f:
        json.dump([{"id": i, "label": l} for i, l in zip(ids, labels)], f)
    # model-specific copies so the API can switch
    faiss.write_index(index, f"artifacts/index_{model_name}.faiss")
    print(f"Built {index.ntotal}x{emb.shape[1]} {model_name} index -> artifacts/index.faiss")


if __name__ == "__main__":
    main()
