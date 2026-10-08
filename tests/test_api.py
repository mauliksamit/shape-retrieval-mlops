"""API test with a tiny fake FAISS index (no training needed)."""
import json
import numpy as np


def _setup_app(tmp_path, monkeypatch):
    import faiss
    import src.serving.app as app

    dim = 8
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(6, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = faiss.IndexFlatL2(dim)
    index.add(vecs)
    idx_p = str(tmp_path / "index.faiss")
    faiss.write_index(index, idx_p)
    meta = [{"id": f"test/chair/{i}", "label": "chair" if i < 3 else "table"} for i in range(6)]

    def fake_load():
        import faiss as _f
        app._state["index"] = _f.read_index(idx_p)
        app._state["meta"] = meta
        app._state["classes"] = ["chair", "table"]

    def fake_embed_query(points):
        v = np.ones((1, dim), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(app, "_load", fake_load)
    monkeypatch.setattr(app, "embed_query", fake_embed_query)
    app._state.update({"index": None, "meta": None, "classes": []})
    return app


def test_health_and_retrieve(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    app = _setup_app(tmp_path, monkeypatch)
    c = TestClient(app.app)
    assert c.get("/health").status_code == 200
    assert set(c.get("/classes").json()["classes"]) == {"chair", "table"}
    pts = np.random.default_rng(2).normal(size=(32, 3)).tolist()
    r = c.post("/retrieve", json={"points": pts, "k": 2})
    assert r.status_code == 200
    assert len(r.json()["results"]) == 2


def test_retrieve_rejects_empty(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    app = _setup_app(tmp_path, monkeypatch)
    c = TestClient(app.app)
    assert c.post("/retrieve", json={"points": [], "k": 2}).status_code == 400
