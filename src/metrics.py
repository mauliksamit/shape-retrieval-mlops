"""Retrieval metrics in pure numpy (no torch/faiss needed) — fully unit-tested."""
import numpy as np


def recall_at_k(retrieved_labels, query_label, k) -> float:
    """1.0 if query_label appears in top-k retrieved labels, else 0.0."""
    return 1.0 if query_label in list(retrieved_labels[:k]) else 0.0


def average_precision(retrieved_labels, query_label) -> float:
    """Standard AP: mean precision at each relevant rank, 0 if none relevant."""
    retrieved_labels = list(retrieved_labels)
    hits, total = 0, 0.0
    for i, lab in enumerate(retrieved_labels, start=1):
        if lab == query_label:
            hits += 1
            total += hits / i
    return total / hits if hits else 0.0


def evaluate_retrieval(rank_lists, query_labels, ks=(1, 5, 10)):
    """Aggregate recall@k and mAP.

    Args:
        rank_lists: list of ranked label lists (gallery order per query, query itself excluded).
        query_labels: true label per query.
    Returns:
        dict with recall@k, mAP and per-query APs.
    """
    n = len(query_labels)
    recalls = {k: float(np.mean([recall_at_k(r, q, k) for r, q in zip(rank_lists, query_labels)])) if n else 0.0
               for k in ks}
    aps = [average_precision(r, q) for r, q in zip(rank_lists, query_labels)]
    return {"recall": recalls, "mAP": float(np.mean(aps)) if aps else 0.0, "aps": aps}


def per_class_scores(rank_lists, query_labels, gallery_labels=None, ks=(1, 5, 10)):
    """Per-class recall@k + mAP. Returns {class: {recall@k, mAP, n}}."""
    classes = sorted(set(query_labels))
    out = {}
    for c in classes:
        idx = [i for i, q in enumerate(query_labels) if q == c]
        sub = [rank_lists[i] for i in idx]
        res = evaluate_retrieval(sub, [c] * len(sub), ks=ks)
        out[str(c)] = {"n": len(idx), "recall": {str(k): v for k, v in res["recall"].items()}, "mAP": res["mAP"]}
    return out


def confusion_pairs(rank_lists, query_labels, k=1):
    """Count (true_label -> predicted top-1 label) confusions. Returns dict of dicts."""
    conf = {}
    for ranks, q in zip(rank_lists, query_labels):
        pred = ranks[0] if ranks else None
        conf.setdefault(str(q), {}).setdefault(str(pred), 0)
        conf[str(q)][str(pred)] += 1
    return conf
