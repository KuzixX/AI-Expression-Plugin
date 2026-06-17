"""
DiffusionNet — compact self-contained implementation (Sharp et al., 2022).

Surface-based network: learned heat diffusion over the mesh + spatial gradient
features. Topology/discretization robust — ideal for heads with different
tessellation. Used here as the VERTEX ENCODER (replaces GNN/EdgeConv) in the
muscle-skinning architecture.

Operators (mass, evals, evecs, gradX, gradY) are precomputed per mesh — see
src/data/operators.py. This module only consumes them.

Reference: github.com/nmwsharp/diffusion-net
"""
import torch
import torch.nn as nn


class LearnedTimeDiffusion(nn.Module):
    """Diffuse each feature channel by a LEARNED time t, in the Laplacian
    spectral basis:  diffused = evecs @ (exp(-evals * t) * (evecs^T @ mass @ x)).

    Each channel gets its own learnable diffusion time (softplus-positive).
    """

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.diffusion_time = nn.Parameter(torch.empty(n_channels).normal_(
            mean=0.0, std=0.0001))

    def forward(self, x, mass, evals, evecs):
        # x: (V, C); mass: (V,); evals: (K,); evecs: (V, K)
        t = torch.clamp(self.diffusion_time, min=1e-8)          # (C,)
        # project to spectral basis (weighted by mass)
        x_spec = evecs.transpose(-2, -1) @ (mass.unsqueeze(-1) * x)  # (K, C)
        decay = torch.exp(-evals.unsqueeze(-1) * t.unsqueeze(0))     # (K, C)
        x_diff = evecs @ (decay * x_spec)                            # (V, C)
        return x_diff


class SpatialGradientFeatures(nn.Module):
    """Per-vertex spatial gradients (anisotropy): from gradX/gradY operators
    build complex tangent gradients, learn a per-channel rotation, take inner
    products → rotation-aware scalar features."""

    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.A = nn.Linear(n_channels, n_channels, bias=False)

    def forward(self, x, gradX, gradY):
        # x: (V, C); gradX/gradY: sparse (V, V)
        gx = gradX @ x                                  # (V, C)
        gy = gradY @ x                                  # (V, C)
        # learned mixing of the rotated gradient, dot with original gradient
        gxr = self.A(gx); gyr = self.A(gy)
        feat = gx * gxr + gy * gyr                      # (V, C)
        return torch.tanh(feat)


class MiniMLP(nn.Sequential):
    def __init__(self, layer_sizes, activation=nn.ReLU, last_act=False):
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i + 2 < len(layer_sizes) or last_act:
                layers.append(activation())
        super().__init__(*layers)


class DiffusionNetBlock(nn.Module):
    """One DiffusionNet block: learned diffusion → gradient features → MLP,
    with a residual connection."""

    def __init__(self, width, mlp_hidden=None, dropout=0.0):
        super().__init__()
        self.width = width
        self.diffusion = LearnedTimeDiffusion(width)
        self.gradient = SpatialGradientFeatures(width)
        hidden = mlp_hidden or [width]
        # input to MLP: [x, diffused, gradient_feat] = 3*width
        self.mlp = MiniMLP([3 * width] + hidden + [width])
        self.norm = nn.LayerNorm(width)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, mass, evals, evecs, gradX, gradY):
        diff = self.diffusion(x, mass, evals, evecs)
        grad = self.gradient(diff, gradX, gradY)
        cat = torch.cat([x, diff, grad], dim=-1)        # (V, 3*width)
        out = self.mlp(cat)
        out = self.drop(out)
        return self.norm(x + out)                       # residual + norm


class DiffusionNet(nn.Module):
    """Vertex encoder. Maps per-vertex input features (C_in) to per-vertex
    embeddings (C_out) via N_block DiffusionNet blocks.

    outputs_at='vertices' → returns (V, C_out).
    """

    def __init__(self, C_in, C_out, C_width=128, N_block=4,
                 mlp_hidden=None, dropout=0.0, last_activation=None,
                 outputs_at='vertices'):
        super().__init__()
        self.outputs_at = outputs_at
        self.last_activation = last_activation
        self.first = nn.Linear(C_in, C_width)
        self.blocks = nn.ModuleList([
            DiffusionNetBlock(C_width, mlp_hidden, dropout)
            for _ in range(N_block)])
        self.last = nn.Linear(C_width, C_out)

    def forward(self, x, mass, evals, evecs, gradX, gradY):
        h = self.first(x)
        for blk in self.blocks:
            h = blk(h, mass, evals, evecs, gradX, gradY)
        out = self.last(h)
        if self.last_activation is not None:
            out = self.last_activation(out)
        return out
