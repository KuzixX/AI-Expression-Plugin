"""
Компоненты лосса (твои):
  L_deform — ‖δ_pred − δ_target‖²       (главный сигнал)
  L_weight — ‖W − W_gt‖²                (warm start, опц., если есть GT-веса)
  L_smooth — ‖L · W‖²                   (гладкость весов по мешу)
  L_sparse — ‖W‖₁                       (разреженность: мало мышц на вершину)
"""
import torch


def loss_deform(delta_pred, delta_target):
    # (E, V, 3) both
    return ((delta_pred - delta_target) ** 2).sum(-1).mean()


def loss_weight(W, W_gt):
    return ((W - W_gt) ** 2).mean()


def loss_smooth(W, L_sparse_op):
    """‖L·W‖² — L_sparse_op это разреженный Лапласиан (V,V) torch sparse."""
    LW = torch.sparse.mm(L_sparse_op, W)         # (V, M)
    return (LW ** 2).mean()


def loss_sparse(W):
    return W.abs().mean()


def total_loss(delta_pred, delta_target, W, *, lap_op=None, W_gt=None,
               w_deform=10.0, w_weight=0.0, w_smooth=0.3, w_sparse=0.005):
    """Суммарный лосс + словарь компонент (для логов)."""
    comps = {}
    L = w_deform * loss_deform(delta_pred, delta_target)
    comps["deform"] = float(L.detach())
    if W_gt is not None and w_weight > 0:
        lw = w_weight * loss_weight(W, W_gt)
        L = L + lw; comps["weight"] = float(lw.detach())
    if lap_op is not None and w_smooth > 0:
        ls = w_smooth * loss_smooth(W, lap_op)
        L = L + ls; comps["smooth"] = float(ls.detach())
    if w_sparse > 0:
        lsp = w_sparse * loss_sparse(W)
        L = L + lsp; comps["sparse"] = float(lsp.detach())
    comps["total"] = float(L.detach())
    return L, comps
