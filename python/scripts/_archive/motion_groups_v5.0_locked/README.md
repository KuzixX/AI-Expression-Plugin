# Motion Groups Transfer — версия **5.0** (in progress)

После v4 проведён большой strip — убраны 7 неиспользуемых assign-режимов,
оставлены только **рабочие**, плюс добавлены 3 экспериментальных варианта
основанных на наблюдении: «когда heat-поля анкеров перекрываются, кластеры
ломаются; когда зоны независимы — работает».

## Файлы

| Файл | Назначение |
|---|---|
| [`debug_head1_pipeline.py`](./debug_head1_pipeline.py) | Главный pipeline с GUI |
| [`multi_anchor_motion_groups.py`](./multi_anchor_motion_groups.py) | Production transfer (legacy v2) |
| [`compare_heads.py`](./compare_heads.py) | Standalone: сравнение FLAME ↔ FBX через HKS/WKS |
| [`functional_map_fit.py`](./functional_map_fit.py) | Classical Functional Maps registration |
| [`align_heat_tables.py`](./align_heat_tables.py) | Выравнивание heat-таблиц через A/B методы |
| [`visualize_dumps.py`](./visualize_dumps.py) | Batch-визуализация дампов |
| [`METHOD.md`](./METHOD.md) | Описание метода heat_align (наследие v4) |

## Что выпилено vs v4

❌ Удалены assign modes: `voronoi`, `hks`, `wks`, `hybrid`, `heat_vec`,
   `heat_align`, `heat_svd`, `heat_rank`, `sinkhorn`, `rbf`

❌ Удалены GUI поля: `n_eigs/n_scales` (для удалённого HKS-matching), `n_svd`,
   `heat_align_*`, `w_geo/w_sig`, `sinkhorn_eps`, `rbf_kernel`

✅ Удалось сократить файл с **3857 → ~3050 строк** при том что добавлены
3 новых метода.

## 4 режима matching в v5

| Режим | Принцип | Когда хорошо |
|---|---|---|
| **`heat_zone_xyz`** ⭐ (default) | Per anchor zone: point-cloud alignment (centroid/scale/non_rigid) + xyz NN | Универсальный, рабочий |
| **`zonal_1d`** (B) | Hard argmax-partition → каждая вершина строго в один anchor → xyz NN внутри | Когда heat-зоны сильно перекрываются |
| **`sequential_anchor`** (C) | Anchor'ы обрабатываются по очереди, занятые вершины пропускаются | Контролируемое разрешение конфликтов |
| **`decorr_heat`** (D) | Gram-Schmidt orthogonalization heat-полей → matching на decorrelated heat | Математически устраняет перекрытие |

### Подробнее про варианты B/C/D

**B — `zonal_1d`** реализует идею: «для каждой вершины определяем её один
доминирующий anchor (argmax), и работаем только в этой зоне». Никаких смесей
heat от разных anchor'ов → matching внутри зоны делается по 3D xyz после
alignment.

**C — `sequential_anchor`** работает в стиле «жадного» назначения. Сортируем
anchor'ы (по `max heat` или `по index`), обрабатываем их по очереди. На каждом
шаге доступны только вершины которые ещё не были «забраны» предыдущими
anchor'ами. Конфликты разрешаются приоритетом порядка.

**D — `decorr_heat`** применяет Gram-Schmidt orthogonalization к K
heat-векторам как rows матрицы (N,K). На выходе K **некоррелированных**
полей-дескрипторов. Их подаём в стандартный heat_zone_xyz matching.
Перекрытие зон по построению устранено.

## Параметры (общие для всех 4 режимов)

| Параметр | Default | Что |
|---|---|---|
| `heat_zone_alignment_mode` | `scale` | `centroid` / `scale` / `non_rigid` |
| `heat_zone_rigid` | `True` | Включить scale alignment |
| `heat_zone_non_rigid_iters` | `2` | RBF iterations для non_rigid |
| `heat_zone_non_rigid_smoothing` | `0.01` | RBF smoothing |
| `heat_zone_use_anchor_align` | `True` | Anchor-based (off=centroid) |
| `heat_zone_use_rotation` | `False` | Procrustes pre-step |
| `heat_zone_smooth` | `2` | Label majority-vote iterations |
| `heat_zone_show_viz` | `True` | 3D окно ZONE-ALIGNMENT |
| `sequential_anchor_order` | `by_max_heat` | Только для variant C |

## Smoothing & Geo Filter

✅ Сохранены полностью:
- `smooth_iters` (HEAD 1 Laplacian)
- `smooth_iters_fbx` (FBX отдельный)
- `smooth_alpha` (сила сглаживания)
- `geo_filter_enable` + `geo_filter_tolerance` (sanity filter переноса)
- Mesh-graph majority vote для labels

## HKS/WKS Diagnostic (отдельный шаг)

✅ Сохранён — diagnostic clustering до выбора anchor'ов с показом
обоих мешей. Параметры: `viz_hks_*` (type, n_clusters, n_eigs, n_scales,
smoothing options, show flags).

## Запуск

```bash
cd /Users/kuzix/Documents/GitHub/Muscle-autoskinner
source .venv/bin/activate
python3 python/scripts/motion_groups_v5/debug_head1_pipeline.py
```

GUI → выбрать один из 4 assign modes → выбрать FBX → START.

## Связь с предыдущими версиями

- **v3** — frozen at `motion_groups_v3.0_locked/`
- **v4** — frozen at `motion_groups_v4.0_locked/` (содержит heat_align,
  heat_svd, и т.д. — на случай если что-то понадобится взять обратно)
- **v5** — текущая
