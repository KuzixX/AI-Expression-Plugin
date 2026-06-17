"""
MuscleSkinningNet — основная архитектура (твоя), с DiffusionNet вместо GNN.

  Vertex features ──▶ DiffusionNet (vertex encoder) ──▶ vertex emb (V, D)
  Muscle features ──▶ Muscle MLP  ─────────────────────▶ muscle emb (M, D)
                                          │
  Pair priors (geo, align) ──────────────┤
                                          ▼
        Cross-attention with anatomical biases:
        scores[v,m] = (Q_v·K_m)/√D − α·geo[v,m] + β·align[v,m]
        W_pred[v,m] = sigmoid(scores)        # независимые веса по мышцам
                                              # столбец m = ID мышцы (структурно)

Выход: W_pred (V, M). Деформация считается отдельно (см. deformation/skinning).
"""
import math

import torch
import torch.nn as nn

from .diffusion_net import DiffusionNet


class MuscleEncoder(nn.Module):
    """Per-muscle MLP + обучаемый ID-эмбеддинг (привязка к конкретной мышце)."""

    def __init__(self, F_m, n_muscles, D, hidden=128):
        super().__init__()
        self.id_emb = nn.Embedding(n_muscles, D)
        self.mlp = nn.Sequential(
            nn.Linear(F_m, hidden), nn.ReLU(),
            nn.Linear(hidden, D))

    def forward(self, muscle_feats):
        # muscle_feats: (M, F_m)
        M = muscle_feats.shape[0]
        ids = torch.arange(M, device=muscle_feats.device)
        return self.mlp(muscle_feats) + self.id_emb(ids)      # (M, D)


class MuscleSkinningNet(nn.Module):
    def __init__(self, C_in, F_m, n_muscles, D=128,
                 diff_width=128, diff_blocks=4, use_wks=False):
        super().__init__()
        self.n_muscles = n_muscles
        self.D = D
        # vertex encoder (DiffusionNet) → embeddings of size D
        self.vertex_encoder = DiffusionNet(
            C_in=C_in, C_out=D, C_width=diff_width, N_block=diff_blocks)
        self.muscle_encoder = MuscleEncoder(F_m, n_muscles, D)
        # проекции Q/K для attention
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        # обучаемые скаляры anatomical bias
        self.alpha = nn.Parameter(torch.tensor(1.0))   # вес geodesic
        self.beta = nn.Parameter(torch.tensor(1.0))    # вес alignment

    def forward(self, feats, mass, evals, evecs, gradX, gradY,
                muscle_feats, geo, align):
        """
        feats: (V, C_in); ops; muscle_feats: (M, F_m);
        geo/align: (V, M). Возвращает W_pred (V, M)."""
        vemb = self.vertex_encoder(feats, mass, evals, evecs, gradX, gradY)
        memb = self.muscle_encoder(muscle_feats)              # (M, D)
        Q = self.q_proj(vemb)                                 # (V, D)
        K = self.k_proj(memb)                                 # (M, D)
        scores = (Q @ K.transpose(0, 1)) / math.sqrt(self.D)  # (V, M)
        scores = scores - self.alpha * geo + self.beta * align
        W = torch.sigmoid(scores)                             # (V, M) — независимо
        return W
