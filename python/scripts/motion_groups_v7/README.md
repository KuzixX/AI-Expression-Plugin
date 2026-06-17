# motion_groups_v7 — Батч-перенос группы выражений

Чистый перенос **группы выражений** на папку голов → один HDF5-датасет.
Без отладочных окон/визуализации. Переиспользует функции v6
(`../motion_groups_v6/debug_head1_pipeline.py`, `auto_anchors.py`).

## Поток данных

```
reference.h5 (нейтраль + N выражений)  ┐
                                       ├──►  transferred.h5
папка голов (.fbx/.obj/.ply/.stl)      ┘     (все головы × все выражения)
```

Для каждой головы: авто-анкоры (MediaPipe) → single-t диффузия → multi-t зоны →
кластеры → UV-развёртка (+relax) → перенос δ каждого выражения. δ FLAME-зон
переиспользуются (считаются один раз), на каждое выражение подставляется его δ.

## Схема HDF5

**Reference (вход):**
```
/neutral/verts   (N,3)   нейтраль
/neutral/faces   (K,3)
/muscle_names    (M,)    имена мышц (порядок activations), опц.
/expressions/<имя>/delta        (N,3)   [или /verts → delta = verts - neutral]
/expressions/<имя>/activations  (M,)    вектор активаций мышц рига (опц.)
```

**Выход:**
```
attrs: n_heads, n_expr, expr_names(json), has_activations
/muscle_names                  (M,)    общие, из reference
/expressions/<выраж>/activations (M,)  АКТИВАЦИИ — общие для всех голов,
                                        пишутся ОДИН раз (не дублируются)
/heads/<имя>/neutral           (Nh,3)
/heads/<имя>/faces             (Kh,3)
/heads/<имя>/expr/<выраж>/delta (Nh,3)  результат на этой голове
```

**Логика:** активация мышц = свойство ВЫРАЖЕНИЯ (одна на все головы), δ =
результат на конкретной голове. Связь по имени выражения. Для обучения скиннинга:
`activations` — вход рига, `delta` — целевой выход на каждой голове.

## Запуск (GUI)

```bash
cd /Users/kuzix/Documents/GitHub/Muscle-autoskinner
source .venv/bin/activate
python python/scripts/motion_groups_v7/transfer_gui.py
```

В окне (английский UI + тултипы по наведению): reference HDF5, папка голов,
выход HDF5 + все параметры (диффузия, кластеризация, сглаживание,
MediaPipe-анкоры, multi-t, UV+relax).

**Browse heads** — карусель из 5 голов внизу окна: центральная крупнее, по 2
слева/справа (если есть). Слайдер двигает центральную голову по всему датасету;
выпадающий список выбирает выражение (или «neutral»). Рендер кэшируется. После
переноса карусель загружается автоматически по готовому HDF5.

## Подготовка reference из FLAME betas

```bash
python python/scripts/motion_groups_v7/make_reference.py \
    --out data/reference.h5 \
    --expr "smile=308:8" --expr "brows=310:6,311:-4"
```

## Pipeline step debug

Кнопка **«🔬 Pipeline step debug»** в окне открывает пошаговый Open3D-вьювер
(reference-нейтраль + первая голова папки). Кнопка **«Next ▶»** показывает каждый
шаг с перекраской: anchors → diffusion → zones → clusters → transfer. Все
настройки берутся из текущих полей основного окна v7 (передаются через временный
JSON). Удобно отладить, как ложатся зоны/перенос перед батчем.

## Файлы
- `transfer_engine.py` — headless-движок (`run_transfer`, `read_reference`).
- `transfer_gui.py` — tkinter GUI (один экран, все параметры, пресеты, карусель).
- `pipeline_debug.py` — пошаговый Open3D-вьювер (параметры из основного окна).
- `make_reference.py` — генератор reference-HDF5 из FLAME выражений.
- `make_test_reference.py` — быстрый тестовый reference (6 эмоций, нулевые акт.).
