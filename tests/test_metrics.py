"""Metrics unit tests: hand-computable cases for recall@k, AP, aggregation."""
from src.metrics import recall_at_k, average_precision, evaluate_retrieval, per_class_scores


def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a", "b", "c"], "b", 2) == 1.0
    assert recall_at_k(["a", "b", "c"], "c", 2) == 0.0
    assert recall_at_k(["a", "b", "c"], "a", 1) == 1.0


def test_average_precision_known_value():
    # relevant at ranks 1 and 3: (1/1 + 2/3) / 2 = 0.8333
    ap = average_precision(["a", "b", "a", "c"], "a")
    assert abs(ap - (1 + 2 / 3) / 2) < 1e-6
    assert average_precision(["b", "c"], "a") == 0.0


def test_evaluate_retrieval_aggregation():
    ranks = [["a", "b"], ["b", "a"]]
    res = evaluate_retrieval(ranks, ["a", "b"], ks=(1, 2))
    assert res["recall"][1] == 1.0
    assert res["recall"][2] == 1.0
    assert res["mAP"] == 1.0


def test_per_class_scores():
    ranks = [["a", "b"], ["a", "b"], ["b", "a"]]
    per = per_class_scores(ranks, ["a", "a", "b"], ks=(1,))
    assert per["a"]["n"] == 2 and per["b"]["n"] == 1
    assert per["b"]["recall"]["1"] == 1.0
