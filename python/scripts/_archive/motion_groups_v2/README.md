# Motion Groups Transfer — версия 2

Топология-независимый перенос мимики через **motion-groups clustering** + **полярную декомпозицию** + **honest geodesic distance**.

## Файлы

| Файл | Назначение |
|---|---|
| [`multi_anchor_motion_groups.py`](./multi_anchor_motion_groups.py) | Production pipeline: HEAD 1 (FLAME) → HEAD 2 (FLAME/FBX), полный flow с анимацией |
| [`debug_head1_pipeline.py`](./debug_head1_pipeline.py) | **Debug/inspection tool** для HEAD 1 + опциональный transfer на FBX, с GUI и сохранением артефактов |

Оба скрипта реализуют **один и тот же** алгоритм, но с разным UI и уровнем визуализации/диагностики.

---

## Что нового в v2 vs v1

### Алгоритмические улучшения

| Аспект | v1 | v2 |
|---|---|---|
| **Geodesic limit** | Euclidean `\|v - c_centroid\| < σ·factor` | **Dijkstra по поверхности** до `σ·factor` |
| **Seed для геодезического обхода** | — | `find_nearest_vertex` от source centroid |
| **Adjacency для Dijkstra** | — | Edge list с длинами рёбер |
| **Поведение на стилизованных мешах со складками** | Может перепутать "близкие в xyz, далёкие по поверхности" вершины | Корректно обходит складки |
| **Mass-normalized initial condition** | ✅ есть | ✅ есть |

### UX / Tooling улучшения (debug_head1_pipeline.py)

- ✅ **Tkinter GUI** для всех параметров (требует tkinter — см. INSTALL.md)
- ✅ **Native macOS file picker** через osascript для выбора FBX (не виснет)
- ✅ **Pre-check window** — обе головы рядом перед всеми действиями для проверки ориентации
- ✅ **Сохранение всех артефактов** в `python/scripts/debug_output/run_<timestamp>/`
- ✅ **CSV/JSON dumps** на каждом шаге pipeline (см. ниже)
- ✅ **Поддержка до 20 anchor-точек** (раньше было 4)
- ✅ **Custom betas input** через ручной формат `300:8.0,302:-5.0`

---

## Алгоритм пошагово

### Стадия A — Подготовка

```
v_template, shapedirs, faces = load_flame(FLAME_PKL)
verts_1 = normalize_bbox(apply_betas(v_template, shapedirs, shape_betas))
verts_2 = normalize_bbox(load_fbx_or_apply_betas(...))     # FBX или FLAME-2
```

Обе головы нормализуются в bbox=1.0 для сопоставимости.

### Стадия B — Heat diffusion (на каждой голове)

```
для каждой anchor-точки src:
    A_src = mass[src]                            # площадь Voronoi-ячейки
    u_0[src] = 1 / A_src                          # mass-normalized δ-функция
    (M + dt·L) u_{n+1} = M · u_n                  # implicit Euler backward
    heat[anchor, v] = u_steps
```

`L` = cotangent Laplacian, `M` = diagonal mass matrix.

**Mass-normalized initial** гарантирует одинаковую полную тепловую массу `Σ u·A = 1` на обеих мешах → теплопотоки сопоставимы.

### Стадия C — Δ = blendshape на HEAD 1

```
head1_expr = FLAME(shape_1 + expr_betas)
δ_1 = head1_expr - verts_1                        # "истинная" мимика на HEAD 1
```

### Стадия D — Motion-groups clustering на HEAD 1

Для каждой anchor-зоны независимо:

1. **Активные вершины:** `heat_1[a, v] > 0.05 · max`
2. **K-means** в feature space:
   ```
   feature[v] = [ δ_1[v]/scale ,  (verts_1[v]-mean)/scale · position_weight ]
   ```
   - `position_weight=0` (default) → кластеризация **только по motion** (хорошо для асимметричных мимик)
   - `position_weight>0` → учёт пространственной локальности
3. **Полярная декомпозиция** на каждом кластере:
   ```
   F = (Σ w·q⊗p) · (Σ w·p⊗p)^(-1)                  # weighted deformation gradient
   U Σ V^T = SVD(F)
   R = U · diag(1,1,sign(det(U V^T))) · V^T         # rotation, det = +1
   S = R^T · F                                       # symmetric stretch
   stretches = eig(S)                                # principal stretch values
   ```

**Дескриптор кластера** (топология-независимый):
```python
{
  'anchor_idx':    int,
  'c_rest':        (3,),         # heat-weighted centroid
  'spatial_sigma': float,        # RMS spatial spread
  'μ':             (3,),         # translation
  'R':             (3, 3),       # rotation
  'S':             (3, 3),       # stretch
  'stretches':     (3,),         # eigenvalues
  'axes':          (3, 3),       # eigenvectors
  'F':             (3, 3),       # full deformation gradient
}
```

### Стадия E — Voronoi-разбиение HEAD 2 (с honest geodesic)

Для каждой anchor-зоны на HEAD 2:

1. **Активные вершины:** `heat_2[a, v] > 0.05 · max`
2. **Для каждого source-кластера в этом anchor**:
   ```
   seed = find_nearest_vertex(verts_2, source_cluster.c_rest)    # Euclidean argmin
   R_max = source_cluster.spatial_sigma · geodesic_factor
   geodesic_dist[v] = Dijkstra(adjacency_2, seed, R_max)         # по поверхности
   ```
3. **Каждая активная вершина** → к кластеру с **минимальным geodesic_dist** (не Euclidean)
4. Если все ∞ → вершина **не приписывается** (нет деформации)

### Стадия F — Применение трансформации

Для каждой target-вершины `v` в кластере с source `s`:
```
c_target = heat-weighted centroid of target cluster vertices
r = verts_2[v] - c_target
δ_2[v] += heat_2[v] · ( μ_s + (R_s · S_s - I) · r )
```

### Стадия G — Laplacian smoothing

```
W = neighbor-averaging sparse matrix (по рёбрам)
для smooth_iters итераций:
    δ_2 = (1 - α) · δ_2 + α · (W @ δ_2)
```

Сглаживает резкие границы между кластерами.

### Стадия H — Финальная деформация

```
head_2_deformed = verts_2 + δ_2
```

---

## Сценарии использования

### Production transfer (multi_anchor_motion_groups.py)

```bash
cd /Users/kuzix/Documents/GitHub/Muscle-autoskinner
source .venv/bin/activate

# FLAME → FLAME
python python/scripts/motion_groups_v2/multi_anchor_motion_groups.py

# FLAME → FBX
python python/scripts/motion_groups_v2/multi_anchor_motion_groups.py \
  --fbx "Muscle-autoskinner/Assets/Meshes/Reference/stylized_female_head_solid.fbx"
```

### Debug / inspection (debug_head1_pipeline.py)

```bash
# Только HEAD 1 — pipeline без transfer'а
python python/scripts/motion_groups_v2/debug_head1_pipeline.py

# HEAD 1 + transfer на FBX
python python/scripts/motion_groups_v2/debug_head1_pipeline.py \
  --fbx-path "Muscle-autoskinner/Assets/Meshes/Reference/stylized_female_head_solid.fbx"

# С конкретными параметрами заранее
python python/scripts/motion_groups_v2/debug_head1_pipeline.py \
  --fbx-path "..." \
  --time 0.005 --position-weight 0.5 --n-clusters 8 \
  --smooth-iters 5 --geodesic-factor 2.5
```

---

## CLI параметры

| Флаг | Default | Что |
|---|---|---|
| `-n N` | 20 | Макс. число anchor-точек |
| `--time` | 0.002 | Время диффузии (после bbox=1 норм) |
| `--steps` | 60 | Шагов implicit Euler |
| `--fps` | 24 | FPS для анимации диффузии |
| `--n-clusters` | 5 | Макс. число motion-groups в каждой anchor-зоне |
| `--heat-threshold` | 0.05 | Доля от heat.max() ниже которой вершина игнорится |
| `--position-weight` | 0.0 | Вес позиции vs motion в k-means (0=motion only) |
| `--smooth-iters` | 3 | Кол-во Laplacian smoothing итераций |
| `--smooth-alpha` | 0.5 | Сила сглаживания на итерацию |
| `--geodesic-factor` | 3.0 | Радиус Dijkstra = σ · этот множитель |
| `--fbx-path` / `--fbx` | None | Путь к FBX для HEAD 2 (опционально) |
| `--no-gui` | False | Использовать консольный ввод вместо tkinter GUI |

---

## Output структура (debug_head1_pipeline.py)

При каждом запуске создаётся `python/scripts/debug_output/run_<timestamp>/`:

```
run_YYYYMMDD_HHMMSS/
├── metadata.json                       # Параметры запуска, betas, anchor индексы
├── head1/
│   ├── verts_rest.csv                  # (N×3) исходные позиции
│   ├── faces.csv                       # (F×3) индексы вершин в гранях
│   ├── anchor_indices.csv              # Индексы выбранных anchor
│   ├── heat.csv                        # (N×K) heat-карты per-anchor
│   ├── verts_deformed_native.csv       # (N×3) позиции после блендшейпа
│   ├── delta_native.csv                # (N×3) δ блендшейпа
│   ├── clusters.json                   # Все motion-groups (μ, R, S, indices, ...)
│   ├── verts_deformed_recon.csv        # (N×3) reconstructed позиции
│   ├── delta_recon.csv                 # (N×3) reconstructed δ
│   ├── verts_deformed_smoothed.csv     # (N×3) после smoothing
│   └── delta_smoothed.csv              # (N×3) smoothed δ
└── fbx/                                # Только если --fbx-path задан
    ├── verts_rest.csv
    ├── faces.csv
    ├── anchor_indices.csv
    ├── heat.csv
    ├── target_clusters.json            # Voronoi-разбиение (source ↔ target)
    ├── delta_raw.csv                   # δ до сглаживания
    ├── delta_smoothed.csv
    └── verts_deformed.csv
```

### Формат clusters.json
```json
[
  {
    "anchor_idx": 0,
    "cluster_idx": 0,
    "n_verts": 87,
    "indices": [1234, 1235, ...],
    "heat_weights": [0.95, 0.87, ...],
    "c_rest": [0.001, -0.09, 0.13],
    "spatial_sigma": 0.025,
    "mu": [-0.003, 0.011, 0.008],
    "F": [[1.01, ...], ...],
    "R": [[0.99, ...], ...],
    "S": [[1.01, ...], ...],
    "stretches": [1.014, 1.002, 0.991],
    "axes": [[...], ...]
  }
]
```

### Формат target_clusters.json (FBX)
```json
[
  {
    "source_anchor_idx": 0,
    "source_mu":     [...],
    "source_R":      [...],
    "source_S":      [...],
    "source_c_rest": [...],
    "source_sigma":  0.025,
    "target_indices":[892, 1234, ...],
    "target_heat":   [0.92, 0.85, ...],
    "target_c":      [...],
    "n_target_verts": 73,
    "display_color": [0.85, 0.42, 0.17]
  }
]
```

---

## Окна визуализации (debug_head1_pipeline.py)

```
GUI Setup → START
↓
ОКНО 0     ← HEAD 1 (rest) | FBX (rest) — проверка ориентации (если FBX задан)
↓
Pick anchors on HEAD 1 (Shift+click)
↓
ОКНО 1     ← Анимация диффузии HEAD 1
↓
ОКНО 2     ← rest | native deformed
ОКНО 3     ← cluster colors + μ стрелки
ОКНО 4     ← rest | native | reconstructed
ОКНО 5     ← native | reconstructed | smoothed
↓
─── если FBX задан ───
Pick anchors on FBX
↓
ОКНО fbx-diffusion ← Анимация диффузии FBX
↓
ОКНО 6a    ← HEAD 1 clusters | FBX clusters (same palette)
ОКНО 6     ← HEAD 1 native | FBX rest | FBX deformed
```

---

## Сильные стороны v2

- ✅ **Honest geodesic** для Voronoi → корректно работает на мешах со складками
- ✅ **Полное полярное разложение** (μ + R + S) → захватывает translation + rotation + stretch
- ✅ **Топология-независимый дескриптор кластера**
- ✅ **Артефакты сохраняются** для офлайн-анализа (CSV/JSON)
- ✅ **GUI для всех параметров**
- ✅ **Pre-check ориентации** до начала pipeline

## Ограничения

- ❌ **Centroid-based seed** для Dijkstra использует Euclidean argmin (если xyz центроид кластера на HEAD 1 не соответствует анатомически той же точке на HEAD 2 — seed ложится не туда)
- ❌ **Bbox-нормализация ≠ анатомическое выравнивание** — главная фундаментальная проблема при разных пропорциях голов
- ❌ **K-means недетерминирован** (фиксирован `random_state=0`, но семантика может меняться от мелких возмущений)
- ❌ **Симметрия не различается** — для асимметричных мимик k-means с position_weight=0 хорошо разделяет; для симметричных может объединять L+R в один кластер

## Что в v3 (planned)

- **HKS/WKS-based assignment** — заменить Euclidean+Dijkstra на intrinsic descriptors (Heat Kernel Signature) для более семантически правильного матчинга
- **Functional Maps (pyFM)** — formal correspondence framework вместо ad-hoc Voronoi
- **FLAME-fitting pre-alignment** — натягивать FBX на FLAME-пространство через NRICP перед всеми операциями, чтобы анатомические координаты совпадали

---

## Зависимости

- `numpy`, `scipy`, `open3d`, `trimesh`, `sklearn`, `matplotlib`
- `assimp` CLI (для FBX → OBJ конвертации)
- `tkinter` (для GUI; на macOS+pyenv требует пересборки Python с tcl-tk, см. INSTALL.md)

## Структура файлов

```
motion_groups_v2/
├── README.md                          ← этот файл
├── multi_anchor_motion_groups.py     ← production transfer
└── debug_head1_pipeline.py           ← debug/inspection с GUI и CSV output
```
