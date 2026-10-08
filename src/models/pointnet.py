"""Lightweight PointNet-style encoder (no T-Net) for laptop CPU training.

point MLP: 3 -> 64 -> 128 -> 256, max-pool over points -> 256-d global,
projection 256 -> embedding_dim, classifier head for supervised training.
Embeddings (L2-normalized projection) are used for FAISS retrieval.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetEncoder(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 128), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Linear(128, 256), nn.BatchNorm1d(256), nn.ReLU(),
        )
        self.proj = nn.Sequential(nn.Linear(256, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU())

    def forward(self, x):
        # x: (B, N, 3)
        b, n, _ = x.shape
        h = self.mlp(x.reshape(b * n, 3)).reshape(b, n, 256)
        global_feat = h.max(dim=1).values  # (B, 256) symmetric pooling
        return self.proj(global_feat)  # (B, embedding_dim)

    def embed(self, x):
        self.eval()
        with torch.no_grad():
            emb = self.forward(x)
            return F.normalize(emb, p=2, dim=1)


class PointNetClassifier(nn.Module):
    def __init__(self, num_classes=10, embedding_dim=64):
        super().__init__()
        self.encoder = PointNetEncoder(embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.head(self.encoder(x))
