"""
Дифференцируемый линейный скиннинг (blendshape-микс по мышцам):

    δ_pred[v] = Σ_m W[v, m] · activation[m] · direction[m]

Та же формула, что в reference (movement = base + Σ W·act·dir), но с
ПРЕДСКАЗАННЫМИ весами W. Полностью дифференцируема по W (и при желании по
direction). Для batch выражений сразу.
"""
import torch


def skin_deformation(W, activations, directions):
    """
      W           (V, M)   предсказанные веса
      activations (E, M)   активации мышц на каждое выражение (batch E)
      directions  (M, 3)   направление тяги мышцы (из рига)
    Возвращает δ_pred (E, V, 3)."""
    # вклад мышцы m в вершину v на выражении e:
    #   W[v,m] * act[e,m] * dir[m]
    # δ[e,v] = Σ_m W[v,m]*act[e,m]*dir[m]
    Wa = W.unsqueeze(0) * activations.unsqueeze(1)            # (E, V, M)
    delta = Wa @ directions                                  # (E, V, 3)
    return delta
