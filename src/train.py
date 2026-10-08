"""Train PointNet classifier (embeddings = encoder output) or fit PCA baseline.

Logs params + metrics + artifacts to MLflow.

Usage:
    python -m src.train --model pointnet --epochs 10
    python -m src.train --model baseline
"""
import argparse
import os
import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.utils import load_config, set_seed, ensure_dirs
from src.models.pointnet import PointNetClassifier
from src.models.baseline import GeometricBaseline
from src.data_loader import MODELNET10_CLASSES

CLS_TO_IDX = {c: i for i, c in enumerate(MODELNET10_CLASSES)}


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    return d["points"].astype(np.float32), [str(x) for x in d["labels"]]


def train_pointnet(train_pts, train_labels, test_pts, test_labels, cfg, args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    tr = cfg["train"]
    emb_dim = tr.get("embedding_dim", 64)
    model = PointNetClassifier(num_classes=len(MODELNET10_CLASSES), embedding_dim=emb_dim).to(device)
    ytr = torch.tensor([CLS_TO_IDX[l] for l in train_labels])
    yte = torch.tensor([CLS_TO_IDX[l] for l in test_labels])
    loader = DataLoader(TensorDataset(torch.from_numpy(train_pts), ytr),
                        batch_size=args.batch_size or tr.get("batch_size", 32), shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr or tr.get("lr", 1e-3))
    loss_fn = torch.nn.CrossEntropyLoss()
    epochs = args.epochs or tr.get("epochs", 10)
    model.train()
    for ep in range(epochs):
        total, correct, loss_sum = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(xb)
            total += len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
        print(f"epoch {ep+1}/{epochs} loss={loss_sum/total:.4f} acc={correct/total:.4f}")
    # test accuracy
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_pts).to(device))
        acc = (logits.argmax(1).cpu() == yte).float().mean().item()
    print(f"test_acc={acc:.4f}")
    ensure_dirs("artifacts")
    path = "artifacts/pointnet.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": emb_dim,
                "classes": MODELNET10_CLASSES}, path)
    return {"test_acc": acc, "artifact": path, "embedding_dim": emb_dim, "epochs": epochs}


def train_baseline(train_pts, train_labels):
    from src.utils import load_config as _lc  # local to keep import graph simple
    dim = 32
    try:
        dim = _lc()["train"].get("baseline_dim", 32)
    except Exception:
        pass
    base = GeometricBaseline(out_dim=min(dim, len(train_pts) - 1, 14))
    base.fit(train_pts)
    ensure_dirs("artifacts")
    path = "artifacts/baseline.pkl"
    with open(path, "wb") as f:
        pickle.dump(base, f)
    # pseudo train accuracy via nearest-centroid in embedding space
    emb = base.transform(train_pts)
    cents = np.stack([emb[np.array(train_labels) == c].mean(0) for c in MODELNET10_CLASSES])
    pred = [MODELNET10_CLASSES[int(np.argmin(((cents - e) ** 2).sum(1)))] for e in emb]
    acc = float(np.mean([p == t for p, t in zip(pred, train_labels)]))
    print(f"baseline train-centroid acc={acc:.4f}")
    return {"test_acc": acc, "artifact": path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pointnet", "baseline"], default="pointnet")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["data"].get("seed", 42))
    tr_pts, tr_labels = load_npz(os.path.join(cfg["data"]["processed_dir"], "modelnet10_train.npz"))
    te_pts, te_labels = load_npz(os.path.join(cfg["data"]["processed_dir"], "modelnet10_test.npz"))

    if args.model == "pointnet":
        stats = train_pointnet(tr_pts, tr_labels, te_pts, te_labels, cfg, args)
        params = {"model": "pointnet", "epochs": stats["epochs"], "embedding_dim": stats["embedding_dim"],
                  "lr": args.lr or cfg["train"]["lr"], "batch_size": args.batch_size or cfg["train"]["batch_size"]}
    else:
        stats = train_baseline(tr_pts, tr_labels)
        params = {"model": "baseline-pca"}

    # MLflow logging (best-effort: never fail training if server is down)
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name=f"{args.model}"):
            mlflow.log_params(params)
            mlflow.log_metric("test_acc", stats["test_acc"])
            mlflow.log_artifact(stats["artifact"])
            mlflow.log_dict(cfg, "config.yaml")
        print("Logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
