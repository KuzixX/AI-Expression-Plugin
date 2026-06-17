# Muscle-Skinning Net (DiffusionNet vertex encoder)

Нейросеть, предсказывающая **skinning weights** `W (N, M)` (вершины × мышцы рига)
по геометрии головы. Обучение self-supervised через дифференцируемый скиннинг:
веса прогоняются через `δ = Σ W·act·dir` и сравниваются с целевыми δ (перенос
из `motion_groups_v7`).

## Архитектура

```
Vertex features ─▶ DiffusionNet (vertex encoder) ─▶ vertex emb (N, D)
Muscle features ─▶ Muscle MLP + ID embedding ─────▶ muscle emb (M, D)
Pair priors (geodesic, alignment) ────────────────┐
                                                   ▼
   cross-attention + anatomical bias:
   scores[v,m] = (Q·K)/√D − α·geo + β·align
   W = sigmoid(scores)        # независимые веса по мышцам, столбец = ID мышцы
                                   │
   δ_pred = Σ_m W·act_m·dir_m  ◀───┘  (differentiable skinning)
                                   │
   loss = w·‖δ_pred−δ_target‖² + smooth(W) + sparse(W)
```

DiffusionNet заменяет GNN/EdgeConv — устойчив к разной топологии голов.

## Структура файлов

```
src/
  models/
    diffusion_net.py   — DiffusionNet (learned diffusion + spatial gradients)
    skinning_net.py    — MuscleSkinningNet (encoder + muscle MLP + attention)
  data/
    operators.py       — спектральные операторы (mass/evals/evecs/gradX/Y) + кеш
    features.py        — per-vertex фичи (xyz, normal, curvature, опц. WKS)
    muscles.py         — MuscleRig (origin/direction) + парные приоры
    dataset.py         — HeadDataset: читает transferred.h5 → головы
  deformation/
    skinning.py        — дифференцируемый линейный скиннинг δ = Σ W·act·dir
  training/
    losses.py          — L_deform / L_weight / L_smooth / L_sparse
    train.py           — тренировочный цикл (сплит по головам)
  inference/           — (TODO) применение модели к новой голове → W
  export/              — (TODO) экспорт весов в Unity-формат
```

## Запуск

```bash
cd python
source ../.venv/bin/activate
python -m src.training.train --data data/transferred.h5 --epochs 50
```

## ВАЖНО — что подставить из реального рига

Сейчас `muscles.make_dummy_rig` создаёт **заглушку** мышц (случайные origin/
direction), чтобы обкатать пайплайн. Для реального обучения нужно из Unity-рига
положить в reference HDF5 на каждую мышцу:
  • **origin** (точка крепления, 3),
  • **direction** (направление тяги, 3),
и читать настоящий `MuscleRig` в `train.build_rig`. Активации `(M,)` на
выражение уже есть.

## Зависимости

`torch`, `potpourri3d`, `robust_laplacian`, `h5py`, `scipy`, `numpy` — в .venv.
DiffusionNet реализован внутри проекта (`models/diffusion_net.py`), внешний
пакет не нужен.
