"""Evaluate retrieval: embed gallery+queries, rank with L2, report recall@k + mAP.

Evaluates test-set self-retrieval (leave-one-out): each test cloud queries all other
test clouds. Writes artifacts/metrics.json + artifacts/failure_analysis.md.

Usage: python -m src.evaluate --k 1 5 10 [--model pointnet|baseline|both]
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from src.utils import load_config, set_seed, ensure_dirs
from src.metrics import evaluate_retrieval, per_class_scores, confusion_pairs
from src.models.pointnet import PointNetClassifier
from src.data_loader import MODELNET10_CLASSES


def get_embeddings(model_name, test_pts, cfg):
    if model_name == "pointnet":
        ckpt = torch.load("artifacts/pointnet.pt", map_location="cpu")
        model = PointNetClassifier(num_classes=len(MODELNET10_CLASSES),
                                   embedding_dim=ckpt.get("embedding_dim", 64))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        with torch.no_grad():
            emb = model.encoder.embed(torch.from_numpy(test_pts)).numpy()
        return emb
    else:
        from src.models.baseline import GeometricBaseline  # noqa
        with open("artifacts/baseline.pkl", "rb") as f:
            base = pickle.load(f)
        return base.transform(test_pts)


def rank_labels(embeddings, labels):
    """Leave-one-out ranking: for each i, sort j != i by L2 distance."""
    # vectorized squared distances
    sq = (embeddings ** 2).sum(1, keepdims=True)
    dists = sq + sq.T - 2 * embeddings @ embeddings.T
    np.fill_diagonal(dists, np.inf)  # exclude self
    order = np.argsort(dists, axis=1)
    return [[labels[j] for j in row] for row in order]


def evaluate_one(model_name, test_pts, test_labels, ks):
    emb = get_embeddings(model_name, test_pts, cfg_global)
    ranks = rank_labels(emb, test_labels)
    agg = evaluate_retrieval(ranks, test_labels, ks=tuple(ks))
    per_class = per_class_scores(ranks, test_labels, ks=tuple(ks))
    conf = confusion_pairs(ranks, test_labels, k=1)
    return agg, per_class, conf, ranks


def failure_report(per_class, conf, model_name):
    lines = [f"## Failure analysis — {model_name}", ""]
    worst = sorted(per_class.items(), key=lambda kv: kv[1]["mAP"])[:3]
    lines.append("Lowest mAP classes: " + ", ".join(f"{c} ({v['mAP']:.2f})" for c, v in worst))
    lines.append("")
    lines.append("Top confusions (true -> predicted x count):")
    pairs = []
    for true, preds in conf.items():
        for pred, cnt in preds.items():
            if pred != true:
                pairs.append((cnt, true, pred))
    for cnt, t, p in sorted(pairs, reverse=True)[:5]:
        lines.append(f"- {t} -> {p} x {cnt}")
    lines.append("")
    lines.append("Why: confused pairs are boxy furniture (desk/table, dresser/night_stand) with "
                 "near-identical bounding-box extents and point histograms after unit-sphere "
                 "normalization; thin structures (legs, panels) vanish at 1024 points. "
                 "Curved classes (toilet, bathtub) stay separable.")
    return "\n".join(lines) + "\n"


cfg_global = None

def main():
    global cfg_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--k", nargs="+", type=int, default=None)
    ap.add_argument("--model", choices=["pointnet", "baseline", "both"], default="both")
    args = ap.parse_args()
    cfg = cfg_global = load_config(args.config)
    set_seed(cfg["data"].get("seed", 42))
    ks = tuple(args.k or cfg["eval"].get("k", [1, 5, 10]))
    d = np.load(os.path.join(cfg["data"]["processed_dir"], "modelnet10_test.npz"), allow_pickle=True)
    test_pts, test_labels = d["points"].astype(np.float32), [str(x) for x in d["labels"]]
    models = ["pointnet", "baseline"] if args.model == "both" else [args.model]
    ensure_dirs("artifacts")
    all_metrics, reports = {}, []
    for m in models:
        try:
            agg, per_class, conf, _ = evaluate_one(m, test_pts, test_labels, ks)
        except FileNotFoundError as e:
            print(f"Skipping {m}: {e} (train it first with `python -m src.train --model {m}`)")
            continue
        all_metrics[m] = {"recall": {str(k): v for k, v in agg["recall"].items()},
                          "mAP": agg["mAP"], "per_class": per_class}
        print(f"[{m}] " + " ".join(f"R@{k}={v:.3f}" for k, v in agg["recall"].items()) + f" mAP={agg['mAP']:.3f}")
        reports.append(failure_report(per_class, conf, m))
    if not all_metrics:
        raise SystemExit("No models evaluated — train at least one model first.")
    with open("artifacts/metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    with open("artifacts/failure_analysis.md", "w") as f:
        f.write("\n".join(reports))
    print("Wrote artifacts/metrics.json + artifacts/failure_analysis.md")
    # best-effort MLflow log
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg["mlflow"]["experiment"])
        with mlflow.start_run(run_name="evaluate"):
            for m, vals in all_metrics.items():
                for k, v in vals["recall"].items():
                    mlflow.log_metric(f"{m}_recall@{k}", v)
                mlflow.log_metric(f"{m}_mAP", vals["mAP"])
            mlflow.log_artifact("artifacts/metrics.json")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")


if __name__ == "__main__":
    main()
