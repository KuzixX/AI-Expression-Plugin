# Motion Groups Transfer — версия **3.0** (locked 2026-05-25)

Топология-инвариантный перенос motion-groups (cluster-based мимической деформации)
между разными мешами: **FLAME ↔ FBX** разных топологий и пропорций.

## Файлы

| Файл | Назначение |
|---|---|
| [`debug_head1_pipeline.py`](./debug_head1_pipeline.py) | Главный debug-pipeline + GUI выбор режима матчинга |
| [`multi_anchor_motion_groups.py`](./multi_anchor_motion_groups.py) | Production transfer (Voronoi-режим без HKS/WKS/heat_*) |
| [`visualize_dumps.py`](./visualize_dumps.py) | Batch-визуализация дампов: heatmap, SVD, кластеры, delta (offline PNG) |
| [`align_heat_tables.py`](./align_heat_tables.py) | Выравнивание heat-таблиц HEAD1 ↔ FBX (методы A: NN, B: канонические) |

---

## Что в v3 vs v2

### Девять режимов матчинга кластеров на target

| Режим | Принцип | Когда хорошо |
|---|---|---|
| `voronoi` | Dijkstra от anchor-relative seed | Похожие меши; рабочая лошадка |
| `hks` | Heat Kernel Signature (scale-invariant) | Разные пропорции, похожая анатомия |
| `wks` | Wave Kernel Signature (band-pass) | HKS даёт "размытые" соответствия |
| `hybrid` | Voronoi + HKS взвешенно | Когда HKS бьётся с Voronoi |
| `heat_vec` | K-мерный heat-fingerprint per vertex vs cluster profile | Универсально, средне-точно |
| **`heat_align`** ⭐ | Per-vertex correspondence через heat-space + k-NN voting + mesh smooth | **Лучше всего на практике**; точнее границ |
| `heat_svd` | Joint SVD `[H_src \| H_tgt]` → общий r-мерный базис | Шумоподавление; анкеры скоррелированы |
| `heat_rank` | Per-anchor percentile (shape-invariant) | Сильно разные absolute heat-значения |
| `sinkhorn` | Optimal Transport (balanced) | Нужно равномерное распределение |
| `rbf` | Radial Basis Function warping | Когда геометрия похожа но смещена |

### Главное новое: **`heat_align`** (рекомендуемый режим)

Реализация идеи "взять heat-распределение HEAD1, центровать с FBX по тепловым отпечаткам,
наследовать кластеры FBX-вершинами от ближайших FLAME-вершин в heat-space":

1. Per-anchor max-normalize обеих heat-матриц
2. Для каждой FBX-вершины `j` в зоне anchor `a` находим **top-k** FLAME-вершин `i₁..iₖ`
   с минимальным `||h_fbx[:,j] - h_flame[:,iₘ]||`
3. Голосование за cluster label с весами `1/d²` → побеждает группа, не одиночка
4. Post-smoothing labels на mesh-графе FBX через majority vote 1-ring соседей
5. Стандартная polar-decomp трансформация → деформация FBX

**Параметры:**
- `heat_align_knn` (default 5) — 1=argmin, ≥3 anti-noise
- `heat_align_smooth` (default 2) — итерации mesh-graph smoothing

### `heat_svd` — совместный SVD heat-матриц

`H = [H_source | H_target]` shape `(K, N_s + N_t)`, `SVD → U·Σ·V^T`. `U` это общий
anchor-basis обеих мешей, `V_weighted` — vertex descriptors в одном базисе.
Дальше — heat_vec-style matching на r-мерных векторах.

- `n_svd` (default 0=auto, обычно 4-8) — сколько компонент держать

### Отдельный smoothing для FBX

В v2 был один `smooth_iters` для всех мешей → на крупном FBX (18k+ верт) 3 итерации
давали невидимый эффект. В v3:

- `smooth_iters` — для HEAD 1 (default 3)
- `smooth_iters_fbx` — для FBX (default 30)
- **Авто-скейл**: если FBX в N раз крупнее HEAD 1, итерации умножаются на √N

---

## Сценарии запуска

```bash
cd /Users/kuzix/Documents/GitHub/Muscle-autoskinner
source .venv/bin/activate

# Default GUI запуск (выбираешь FBX и режим через диалог)
python python/scripts/motion_groups_v3/debug_head1_pipeline.py

# CLI без GUI с конкретным режимом
python python/scripts/motion_groups_v3/debug_head1_pipeline.py \
  --no-gui --fbx-path "..." --assign-mode heat_align

# Batch-визуализация дампов последнего ран'а
python python/scripts/motion_groups_v3/visualize_dumps.py

# Выравнивание heat-таблиц HEAD1↔FBX (для оффлайн-анализа)
python python/scripts/motion_groups_v3/align_heat_tables.py
```

---

## Что выбирать

### Для большинства случаев — **`heat_align`** ⭐
Если меши разные топологически — это лучший выбор.

### Используй **`voronoi`** если:
- Меши почти идентичные (FLAME → FLAME shape variations)
- Нужна максимальная предсказуемость
- Стилизация настолько сильная что heat-сигнал ломается

### Используй **`heat_svd`** если:
- Анкеры скоррелированы (см. `heat_corr.png`)
- Нужно шумоподавление

### Сравнивай в дебаге
Запусти **2-3 раза** с одинаковыми anchor'ами, меняй только `--assign-mode`.
В логе появляется **ASSIGN signature** — если signature разный, кластеры разные.

---

## Output данные (CSV/JSON)

Папка: `python/scripts/debug_output/run_<TIMESTAMP>/`

```
run_<TS>/
├── head1/
│   ├── heat.csv                  # (N_flame × K_anchors)
│   ├── clusters.json             # полный JSON с R, S, μ, c_rest
│   ├── clusters_flat.csv         # плоская таблица vertex→cluster
│   ├── verts_rest.csv, faces.csv
│   ├── delta_native.csv          # из FLAME blendshape
│   ├── delta_recon.csv           # из кластеров
│   └── delta_smoothed.csv
├── fbx/
│   ├── heat.csv                  # (N_fbx × K_anchors)
│   ├── target_clusters.json      # перенесённые кластеры
│   ├── delta_raw.csv, delta_smoothed.csv
│   └── verts_*.csv
├── plots/                        # если запускал visualize_dumps.py
│   ├── head1/  fbx/  compare/
└── aligned/                      # если запускал align_heat_tables.py
    ├── method_A_correspondence.csv
    ├── method_A_heat_fbx_aligned.csv
    ├── method_A_quality.png
    └── method_B_canonical_*.csv
```

`metadata.json` содержит все параметры включая `assign_mode`, `n_eigs`, `n_scales`,
`n_svd`, `heat_align_knn`, `heat_align_smooth`, `smooth_iters_fbx`.

---

## Что в v4 (planned)

- **Soft cluster assignment** — вместо hard `argmax` голосовать softmax → плавные границы без post-smooth
- **Functional Maps (pyFM)** — формальная теория соответствий через спектральные базы
- **Spectrum caching** — eigendecomp HKS/WKS считается раз и кэшируется на диск
- **Pre-alignment FLAME→FBX** через FLAME fitting (sparse landmarks)
- **Auto anchor placement** — заменить ручной выбор через шифт-клик

---

## Производительность

| Операция | FLAME (5k) | FBX (8-18k) |
|---|---|---|
| Eigendecomp (k=128) | ~5-10 сек | ~30-60 сек |
| HKS/WKS computation | <1 сек | ~2 сек |
| heat_align (k_nn=5) | <0.5 сек | ~1-3 сек |
| heat_svd | <0.2 сек | <0.5 сек |
| Voronoi+Dijkstra | ~1 сек | ~5-10 сек |
| Laplacian smooth | <0.5 сек | ~2-5 сек на 30 iters |

## Зависимости

- numpy, scipy, open3d, trimesh, sklearn
- matplotlib + seaborn (для `visualize_dumps.py` — offline PNG)
- pandas (для `align_heat_tables.py`)
- assimp CLI (FBX → OBJ)
- tkinter (GUI; на macOS+pyenv нужен tcl-tk@8)
