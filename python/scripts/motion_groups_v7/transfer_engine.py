#!/usr/bin/env python3
"""
motion_groups_v7 — батч-перенос ГРУППЫ выражений на папку голов → HDF5.

Вход:
  • reference HDF5 — нейтраль + N выражений (см. схему ниже);
  • папка с головами (.fbx/.obj/.ply/.stl).
Выход:
  • HDF5 со ВСЕМИ головами × ВСЕМИ выражениями (нейтраль + δ на каждое выраж.).

Переиспользует функции v6 (motion_groups_v6/debug_head1_pipeline.py) — без
элементов отладки/визуализации. Чистый перенос.

──────────────────────────────────────────────────────────────────────────────
СХЕМА reference HDF5 (нейтраль + выражения):
  /neutral/verts     (N,3)  нейтральная FLAME-голова (rest)
  /neutral/faces     (K,3)
  /expressions/<имя>/delta   (N,3)  смещение δ выражения (deformed - neutral)
  [или /expressions/<имя>/verts (N,3) — тогда delta = verts - neutral]
  attrs (опц.): описание

СХЕМА выходного HDF5:
  attrs: n_heads, n_expr, expr_names (json)
  /heads/<имя>/neutral   (Nh,3)  нейтраль головы
  /heads/<имя>/faces     (Kh,3)
  /heads/<имя>/expr/<имя_выраж>/delta  (Nh,3)  перенесённая деформация
──────────────────────────────────────────────────────────────────────────────
"""
import json
import sys
from pathlib import Path

import numpy as np

# импорт функций v6
_V6 = Path(__file__).resolve().parent.parent / "motion_groups_v6"
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))
import debug_head1_pipeline as pipe          # noqa: E402
import auto_anchors                          # noqa: E402


def _wks_fn():
    """Ленивая загрузка wks (spectral_descriptors тянет open3d.gui — не нужен
    в мультипроцесс-воркерах переноса, только для WKS-превью)."""
    import spectral_descriptors as _spectral
    return _spectral.wks


DEFAULT_PARAMS = dict(
    # диффузия (без анимационных полей fps/steps-анимации)
    time=0.002, steps=60,
    # кластеризация
    n_clusters=5, heat_threshold=0.05, position_weight=0.0,
    clustering_method='kmeans', cluster_similarity_threshold=0.3,
    # сглаживание δ
    smooth_iters=3, smooth_alpha=0.5,
    # сглаживание перенесённых групп (majority-vote)
    smooth_transferred_groups=True, group_smooth_iters=8,
    # multi-t
    multi_t_n_times=8, multi_t_n_eigs=80, multi_t_mask_by_single_t=True,
    # UV
    uv_flat=False, uv_world_orient=False,
    uv_align_pca_icp=False,
    uv_interp_delta=True,
    uv_relax_method='arap', uv_relax_iters=50,
    # boundary heat-warp (как в v6): тянуть границу острова FBX на ребро FLAME
    uv_warp_heat=False, uv_warp_heat_t=0.05, uv_warp_min_dist=0.0,
    # MediaPipe анкоры
    landmarks=(9, 4, 199),
)


# ── диффузия (статичная, без анимации) ──────────────────────────────────────

def _static_diffusion(verts, faces, L, MM, srcs, total_time, steps):
    from scipy.sparse.linalg import factorized
    n = len(srcs)
    dt = total_time / max(steps, 1)
    solve = factorized((MM + dt * L).tocsc())
    a_diag = np.array(MM.diagonal())
    u = np.zeros((n, L.shape[0]))
    for ai in range(n):
        u[ai, srcs[ai]] = 1.0 / max(a_diag[srcs[ai]], 1e-12)
    for _ in range(steps):
        for ai in range(n):
            u[ai] = solve(MM @ u[ai])
    return u


def _build_zones(verts, faces, anchors, delta_native, p):
    """single-t диффузия → multi-t зоны (+маска) → кластеры. Возвращает
    result_dict-совместимый dict для transfer_deformations_uv."""
    faces64 = faces.astype(np.int64)
    L, MM = pipe.build_operators(verts, faces64)
    heat = _static_diffusion(verts, faces64, L, MM, anchors,
                             p['time'], p['steps'])
    enr, _ = pipe.enrich_heat_multi_t(
        verts, faces64, list(anchors),
        n_times=p['multi_t_n_times'], n_eigs=p['multi_t_n_eigs'],
        smooth_iters=5, smooth_alpha=0.5, mesh_label="v7")
    if p.get('multi_t_mask_by_single_t', True):
        h1 = heat / heat.max(1, keepdims=True).clip(1e-12)
        active = h1.max(0) > p['heat_threshold']
        enr[:, ~active] = 0.0
    partition = pipe._argmax_partition(enr, threshold=p['heat_threshold'])
    n_anchors = len(anchors)
    vgid = -np.ones(len(verts), dtype=np.int64)
    vgw = np.zeros(len(verts)); gid = 0
    for a in range(n_anchors):
        masked = enr[a].copy(); masked[partition != a] = 0.0
        cls = pipe.cluster_zone(
            masked, delta_native, verts, anchor_idx=a,
            heat_threshold=p['heat_threshold'],
            n_clusters_max=p['n_clusters'],
            position_weight=p.get('position_weight', 0.0),
            clustering_method=p.get('clustering_method', 'kmeans'),
            similarity_threshold=p.get('cluster_similarity_threshold', 0.3),
            print_quality=False)
        for cl in cls:
            for j, vi in enumerate(cl['indices']):
                if cl['heat_weights'][j] > vgw[vi]:
                    vgw[vi] = cl['heat_weights'][j]; vgid[vi] = gid
            gid += 1
    return {'verts': verts, 'faces': faces64, 'partition': partition,
            'n_anchors': n_anchors, 'delta_native': delta_native,
            'vert_gcid': vgid, 'heat': enr}


def _relax_zones(res, p):
    """Релакс UV-островов зон головы (в формате zd для transfer)."""
    flat = "world" if p.get('uv_world_orient') else bool(p.get('uv_flat', False))
    zd = pipe.compute_zone_islands(res['verts'], res['faces'],
                                   res['partition'], res['n_anchors'], flat=flat)
    method = p.get('uv_relax_method', 'arap')
    iters = int(p.get('uv_relax_iters', 50))
    if iters > 0 and method != 'none':
        if bool(p.get('uv_relax_adaptive', False)):
            rounds = max(int(p.get('uv_relax_rounds', 8)), 1)
            target = float(p.get('uv_relax_target', 4.5))
            ipr = max(iters // rounds, 4)              # итераций на раунд
            zd = {a: (pipe.relax_uv_island_adaptive(
                        uv, F, res['verts'][gi], method=method,
                        iters_per_round=ipr, max_rounds=rounds,
                        target=target), F, gi)
                  for a, (uv, F, gi) in zd.items()}
        else:
            zd = {a: (pipe.relax_uv_island(uv, F, res['verts'][gi],
                                           method=method, iters=iters), F, gi)
                  for a, (uv, F, gi) in zd.items()}
    return zd


# ── reference HDF5 ──────────────────────────────────────────────────────────

def read_reference(ref_h5):
    """Читаем нейтраль + выражения из reference HDF5.

    Возвращает (neutral_verts, faces, exprs, acts, muscle_names), где:
      exprs        — {имя_выраж: delta (N,3)} геометрия выражения;
      acts         — {имя_выраж: activations (M,)} вектор активаций мышц
                     (общий для всех голов; None если в reference его нет);
      muscle_names — список имён мышц (порядок activations) или None."""
    import h5py
    with h5py.File(ref_h5, "r") as h:
        if "neutral" not in h or "expressions" not in h:
            raise RuntimeError(
                "reference HDF5 должен содержать /neutral и /expressions")
        neutral = h["neutral"]["verts"][:].astype(np.float64)
        faces = h["neutral"]["faces"][:].astype(np.int64)
        muscle_names = None
        if "muscle_names" in h:
            muscle_names = [m.decode() if isinstance(m, bytes) else str(m)
                            for m in h["muscle_names"][:]]
        exprs = {}
        acts = {}
        for name in h["expressions"]:
            g = h["expressions"][name]
            if "delta" in g:
                exprs[name] = g["delta"][:].astype(np.float64)
            elif "verts" in g:
                exprs[name] = g["verts"][:].astype(np.float64) - neutral
            else:
                continue
            if "activations" in g:               # n-мерный вектор активаций
                acts[name] = g["activations"][:].astype(np.float32)
    if not exprs:
        raise RuntimeError("в /expressions нет ни одного выражения с delta/verts")
    if not acts:
        acts = None
    return neutral, faces, exprs, acts, muscle_names


# ── превью голов из выходного HDF5 (для GUI) ────────────────────────────────

def list_output_heads(out_h5):
    """Имена голов в выходном HDF5."""
    import h5py
    with h5py.File(out_h5, "r") as h:
        return list(h["heads"].keys()) if "heads" in h else []


def get_head_flag(out_h5, head_name):
    """Метка качества переноса головы: True если помечена как 'bad'."""
    import h5py
    with h5py.File(out_h5, "r") as h:
        g = h["heads"].get(head_name)
        return bool(g.attrs.get("bad_transfer", False)) if g is not None else False


def set_head_flag(out_h5, head_name, bad):
    """Помечаем голову как плохой/нормальный перенос (атрибут bad_transfer).
    Метка хранится в самом датасете → попадает в обучение."""
    import h5py
    with h5py.File(out_h5, "a") as h:
        if "heads" in h and head_name in h["heads"]:
            h["heads"][head_name].attrs["bad_transfer"] = bool(bad)
            return True
    return False


def list_bad_heads(out_h5):
    """Список голов, помеченных как плохой перенос."""
    import h5py
    bad = []
    with h5py.File(out_h5, "r") as h:
        if "heads" in h:
            for nm in h["heads"]:
                if bool(h["heads"][nm].attrs.get("bad_transfer", False)):
                    bad.append(nm)
    return bad


_SPECTRUM_CACHE = {}                              # (h5, head, n_eigs) → (ev, evec)


def _head_descriptor(out_h5, head_name, verts, faces, n_eigs=80,
                     n_channels=60, wks_sigma=7.0, channel=None):
    """Скалярная per-vertex карта WKS на голове (нормирована в [0,1]).
    Спектр кешируется по (файл, голова, n_eigs) — считается один раз.

      n_eigs     — число собственных функций («разрешение» спектра);
      n_channels — число энергетических каналов WKS;
      wks_sigma  — ширина гауссова окна WKS;
      channel    — какой канал брать (None → средний)."""
    key = (out_h5, head_name, int(n_eigs))
    spec = _SPECTRUM_CACHE.get(key)
    if spec is None:
        ev, evec = pipe.compute_spectrum(verts, faces, n_eigs=int(n_eigs))
        _SPECTRUM_CACHE[key] = spec = (ev, evec)
    ev, evec = spec
    S, _ = _wks_fn()(ev, evec, n_e=int(n_channels),
                     sigma_scale=float(wks_sigma))
    ch = S.shape[1] // 2 if channel is None else int(
        np.clip(channel, 0, S.shape[1] - 1))
    v = S[:, ch].astype(np.float64)
    v = np.log1p(np.clip(v - v.min(), 0, None))
    rng = v.max() - v.min()
    return (v - v.min()) / rng if rng > 1e-12 else np.zeros_like(v)


def render_head_preview(out_h5, head_name, expr=None, res=256,
                        gain=1.0, colorize=True,
                        col_lo=(0.1, 0.2, 1.0), col_hi=(1.0, 0.1, 0.1),
                        descriptor=None, desc_n_eigs=80, desc_channels=60,
                        desc_channel=None, desc_wks_sigma=7.0,
                        show_landmarks=False, show_signature=False,
                        sig_min=0.33, sig_max=0.50, sig_dist=0.05,
                        sig_smooth=0, sig_counter=None):
    """Рендерим голову из выходного HDF5 в RGB-картинку (numpy HxWx3 uint8)
    орто-рейкастом спереди (без окон).

    descriptor='wks' → раскрашиваем голову WKS-дескриптором (CMAP_HEAT), имеет
    приоритет над colorize. Иначе:
      • colorize=True — величина |δ| градиентом col_lo→col_hi;
      • colorize=False — боковой свет (рельеф).
    show_signature=True → точки в центроидах групп вершин с WKS в [sig_min,
    sig_max], сгруппированных по расстоянию sig_dist (доля диагонали bbox)."""
    import h5py
    with h5py.File(out_h5, "r") as h:
        g = h["heads"][head_name]
        verts = g["neutral"][:].astype(np.float64)
        faces = g["faces"][:].astype(np.int64)
        delta = None
        if expr is not None and expr in g["expr"]:
            delta = g["expr"][expr][:].astype(np.float64)
            verts = verts + gain * delta
    (img, origins, ddir_vec, bbox2d, (a0, a1), ax,
     depth0, ddir, scene) = auto_anchors._shade_image(
        verts, faces, axis=2, sign=1.0, res=res)
    # WKS-карта нужна для раскраски ИЛИ для сигнатурных лендмарков — считаем раз
    dmap = None
    if descriptor == "wks" or show_signature:
        dmap = _head_descriptor(out_h5, head_name, verts, faces,
                                n_eigs=desc_n_eigs, n_channels=desc_channels,
                                wks_sigma=desc_wks_sigma, channel=desc_channel)
    if descriptor == "wks":
        img = _overlay_vertex_scalar(img, faces, dmap, res,
                                     a0, a1, bbox2d, depth0, ddir, ax, scene)
    elif colorize and delta is not None:
        img = _overlay_delta_color(img, verts, faces, delta, res,
                                   a0, a1, bbox2d, depth0, ddir, ax, scene,
                                   col_lo=col_lo, col_hi=col_hi)
    elif not colorize:
        # боковой свет (вместо фронтального шейда _shade_image) — рельеф виднее
        img = _relight_side(img, res, a0, a1, bbox2d, depth0, ddir, ax, scene)
    if show_landmarks:
        img = _draw_landmarks(img, res)            # MediaPipe-точки поверх
    if show_signature and dmap is not None:
        img, n_groups = _draw_signature_landmarks(
            img, verts, faces, dmap, res, a0, a1, bbox2d, ax, ddir,
            sig_min=sig_min, sig_max=sig_max, sig_dist=sig_dist,
            sig_smooth=sig_smooth)
        if sig_counter is not None:
            sig_counter.append(n_groups)
    # ВЕРНЫЕ ПРОПОРЦИИ: _shade_image растягивает bbox (W×H) на квадрат res×res.
    # Сжимаем по более узкой оси, чтобы геометрия не искажалась (letterbox).
    x0, x1, y0, y1 = bbox2d
    w = float(x1 - x0); h = float(y1 - y0)
    if w > 1e-9 and h > 1e-9 and abs(w - h) > 1e-6:
        img = _letterbox_aspect(img, w, h)
    return img


def _draw_landmarks(img, res):
    """Детектим MediaPipe Face Mesh на рендере и рисуем ВСЕ лендмарки зелёными
    точками поверх (на квадратной картинке res×res, до letterbox)."""
    lms = auto_anchors._detect_landmarks(np.ascontiguousarray(img))
    if lms is None:
        return img
    out = img.copy()
    for x, y in lms:                               # норм. [0,1], y вниз
        px = int(round(x * (res - 1)))
        py = int(round(y * (res - 1)))
        # маленький крест 3×3, чтобы точки были видны
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                xx, yy = px + dx, py + dy
                if 0 <= xx < res and 0 <= yy < res:
                    out[yy, xx] = (0, 255, 0)
    return out


def _vertex_normals(verts, faces):
    """Нормали по вершинам (площадь-взвешенное среднее нормалей граней)."""
    V = np.asarray(verts, np.float64); F = np.asarray(faces, np.int64)
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def _smooth_scalar(vals, faces, iters):
    """Сглаживание per-vertex скаляра по 1-кольцу (Laplacian-усреднение) — гасит
    шумовые/мусорные значения сигнатуры. iters=0 → без изменений."""
    vals = np.asarray(vals, np.float64)
    if int(iters) <= 0:
        return vals.copy()
    F = np.asarray(faces, np.int64)
    N = len(vals)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]],
                        F[:, [1, 0]], F[:, [2, 1]], F[:, [0, 2]]], axis=0)
    deg = np.zeros(N); np.add.at(deg, E[:, 0], 1.0); deg = np.maximum(deg, 1.0)
    y = vals.copy()
    for _ in range(int(iters)):
        acc = np.zeros(N); np.add.at(acc, E[:, 0], y[E[:, 1]])
        y = 0.5 * y + 0.5 * (acc / deg)
    return y


def _project_to_px(p, res, a0, a1, bbox2d):
    """3D-точка → пиксель квадратного рендера res×res (до letterbox), орто."""
    x0, x1, y0, y1 = bbox2d
    col = (p[a0] - x0) / (x1 - x0 + 1e-12) * (res - 1)
    row = (y1 - p[a1]) / (y1 - y0 + 1e-12) * (res - 1)
    return int(round(col)), int(round(row))


def _disk(img, px, py, res, r, color, ring=(0, 0, 0)):
    """Заливаем кружок радиуса r цветом color с тонкой обводкой ring."""
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            d2 = dx * dx + dy * dy
            if d2 > r * r:
                continue
            xx, yy = px + dx, py + dy
            if 0 <= xx < res and 0 <= yy < res:
                img[yy, xx] = color if d2 <= (r - 1) * (r - 1) else ring


def _draw_signature_landmarks(img, verts, faces, sigmap, res, a0, a1, bbox2d,
                              ax, ddir, sig_min=0.33, sig_max=0.50,
                              sig_dist=0.05, sig_smooth=0, color=(255, 210, 0)):
    """Точки в центроидах ГРУПП вершин с WKS-сигнатурой в [sig_min, sig_max].

    Группировка по расстоянию (single-linkage): две вершины в одной группе, если
    они ближе sig_dist·(диагональ bbox); связь транзитивна, т.е. группа растёт,
    пока есть соседи в радиусе — это связные компоненты радиус-графа. В центроиде
    каждой группы (→ ближайшая вершина) ставим точку.

      sig_min,sig_max — диапазон нормированной WKS [0..1] (в палитре hot
                        «красное» ≈ 0.33..0.5);
      sig_dist        — радиус группировки в долях диагонали bbox головы."""
    from scipy.spatial import cKDTree
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    V = np.asarray(verts, np.float64); F = np.asarray(faces, np.int64)
    sig = _smooth_scalar(sigmap, F, sig_smooth)         # гасим мусорные сигнатуры
    vn = _vertex_normals(V, F)
    front = (vn[:, ax] * ddir) < 0.05                  # фронт-видимые вершины
    sel = np.where((sig >= sig_min) & (sig <= sig_max) & front)[0]
    if len(sel) == 0:
        return img, 0
    P = V[sel]
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    r = max(sig_dist, 1e-4) * diag
    pairs = cKDTree(P).query_pairs(r, output_type='ndarray')   # пары в радиусе
    n = len(sel)
    if len(pairs):
        g = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                          shape=(n, n))
    else:
        g = sp.coo_matrix((n, n))
    ncomp, labels = connected_components(g, directed=False)
    out = img.copy()
    for c in range(ncomp):
        members = sel[labels == c]
        ctr = V[members].mean(0)                       # центроид группы
        med = members[int(np.argmin(((V[members] - ctr) ** 2).sum(1)))]
        px, py = _project_to_px(V[med], res, a0, a1, bbox2d)
        _disk(out, px, py, res, 4, color)
    return out, ncomp                                  # картинка + число групп


def solve_sig_dist(out_h5, head_name, target_n, sig_min=0.33, sig_max=0.50,
                   desc_n_eigs=80, desc_channels=60, desc_channel=None,
                   desc_wks_sigma=7.0, sig_smooth=0, iters=32):
    """Подбираем sig_dist∈[0,1] так, чтобы число групп = target_n.

    Число групп монотонно УБЫВАЕТ с ростом расстояния (больше радиус → больше
    слияний) → бинарный поиск. Возврат (dist, got) — найденное расстояние и
    фактически достигнутое число групп (может отличаться: функция ступенчатая)."""
    import h5py
    from scipy.spatial import cKDTree
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    with h5py.File(out_h5, "r") as h:
        g = h["heads"][head_name]
        V = g["neutral"][:].astype(np.float64)
        F = g["faces"][:].astype(np.int64)
    dmap = _head_descriptor(out_h5, head_name, V, F, n_eigs=desc_n_eigs,
                            n_channels=desc_channels, wks_sigma=desc_wks_sigma,
                            channel=desc_channel)
    # front-маска — как в render (тот же axis/ddir), shade один раз
    (_i, _o, _d, _b, (_a0, _a1), ax, _dp, ddir, _s) = auto_anchors._shade_image(
        V, F, axis=2, sign=1.0, res=64)
    dmap = _smooth_scalar(dmap, F, sig_smooth)          # как в _draw_signature
    vn = _vertex_normals(V, F)
    front = (vn[:, ax] * ddir) < 0.05
    sel = np.where((dmap >= sig_min) & (dmap <= sig_max) & front)[0]
    if len(sel) == 0:
        return 1.0, 0
    P = V[sel]
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))

    def count(frac):
        r = max(frac, 1e-6) * diag
        pairs = cKDTree(P).query_pairs(r, output_type='ndarray')
        n = len(sel)
        gm = (sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                            shape=(n, n)) if len(pairs) else sp.coo_matrix((n, n)))
        return connected_components(gm, directed=False)[0]

    lo, hi = 0.0, 1.0
    for _ in range(iters):                             # groups убывает с frac
        mid = 0.5 * (lo + hi)
        if count(mid) > target_n:
            lo = mid                                   # групп много → раздвинуть
        else:
            hi = mid
    cl, ch = count(lo), count(hi)                      # вернуть ближайший к N
    return (lo, cl) if abs(cl - target_n) < abs(ch - target_n) else (hi, ch)


def _solve_task(args):
    """Воркер пула: один вызов solve_sig_dist. Возврат (head, dist|None)."""
    (h5, nm, target, sig_min, sig_max, d_eigs, d_ch, d_chan, d_sig,
     sig_smooth) = args
    try:
        dist, _got = solve_sig_dist(
            h5, nm, target, sig_min=sig_min, sig_max=sig_max,
            desc_n_eigs=d_eigs, desc_channels=d_ch,
            desc_channel=d_chan, desc_wks_sigma=d_sig, sig_smooth=sig_smooth)
        return (nm, dist)
    except Exception:
        return (nm, None)


def solve_sig_dist_batch(out_h5, head_names, target, sig_min=0.33, sig_max=0.50,
                         desc_n_eigs=80, desc_channels=60, desc_channel=None,
                         desc_wks_sigma=7.0, sig_smooth=0, n_workers=1,
                         progress=None):
    """Подбор sig_dist под target групп для СПИСКА голов, по ядрам (spawn-пул).
    progress(done, total) — для прогресса в GUI. Возврат {head: dist}."""
    tasks = [(out_h5, nm, target, sig_min, sig_max, desc_n_eigs, desc_channels,
              desc_channel, desc_wks_sigma, sig_smooth) for nm in head_names]
    n = len(tasks)
    solved = {}
    nw = max(1, int(n_workers))
    if nw <= 1 or n <= 1:                               # последовательно
        for k, t in enumerate(tasks):
            nm, dist = _solve_task(t)
            if dist is not None:
                solved[nm] = dist
            if progress:
                progress(k + 1, n)
        return solved
    import multiprocessing as mp
    ctx = mp.get_context("spawn")                       # безопасно для Open3D
    with ctx.Pool(processes=min(nw, n)) as pool:
        done = 0
        for nm, dist in pool.imap_unordered(_solve_task, tasks):
            if dist is not None:
                solved[nm] = dist
            done += 1
            if progress:
                progress(done, n)
    return solved


def _wks_dmap_raw(verts, faces, n_eigs=80, n_channels=60, wks_sigma=7.0,
                  channel=None):
    """WKS per-vertex карта [0,1] прямо из verts/faces (без HDF5-кэша)."""
    ev, evec = pipe.compute_spectrum(verts, faces, n_eigs=int(n_eigs))
    S, _ = _wks_fn()(ev, evec, n_e=int(n_channels), sigma_scale=float(wks_sigma))
    ch = S.shape[1] // 2 if channel is None else int(
        np.clip(channel, 0, S.shape[1] - 1))
    v = S[:, ch].astype(np.float64)
    v = np.log1p(np.clip(v - v.min(), 0, None))
    rng = v.max() - v.min()
    return (v - v.min()) / rng if rng > 1e-12 else np.zeros_like(v)


def _signature_groups(verts, faces, sig_min=0.33, sig_max=0.50, sig_dist=0.05,
                      sig_smooth=0, n_eigs=80, n_channels=60, channel=None,
                      wks_sigma=7.0):
    """Группы вершин WKS-сигнатуры → список массивов глоб. индексов (членов
    группы). Полоса [sig_min,sig_max] → группировка по расстоянию sig_dist·diag.
    Без фронт-фильтра (на всю голову)."""
    from scipy.spatial import cKDTree
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    V = np.asarray(verts, np.float64); F = np.asarray(faces, np.int64)
    dmap = _wks_dmap_raw(V, F, n_eigs, n_channels, wks_sigma, channel)
    sig = _smooth_scalar(dmap, F, sig_smooth)
    sel = np.where((sig >= sig_min) & (sig <= sig_max))[0]
    if len(sel) == 0:
        return []
    P = V[sel]
    diag = float(np.linalg.norm(V.max(0) - V.min(0)))
    r = max(sig_dist, 1e-4) * diag
    pairs = cKDTree(P).query_pairs(r, output_type='ndarray')
    n = len(sel)
    g = (sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(n, n)) if len(pairs) else sp.coo_matrix((n, n)))
    ncomp, labels = connected_components(g, directed=False)
    return [sel[labels == c] for c in range(ncomp)]


def signature_landmark_verts(verts, faces, sig_min=0.33, sig_max=0.50,
                             sig_dist=0.05, sig_smooth=0, n_eigs=80,
                             n_channels=60, channel=None, wks_sigma=7.0):
    """WKS-сигнатурные лендмарки меша → ГЛОБАЛЬНЫЕ индексы вершин-медоидов групп
    (ближайшая к центроиду вершина). Для UV-матча переноса."""
    V = np.asarray(verts, np.float64)
    groups = _signature_groups(V, faces, sig_min, sig_max, sig_dist, sig_smooth,
                               n_eigs, n_channels, channel, wks_sigma)
    out = []
    for members in groups:
        ctr = V[members].mean(0)
        out.append(int(members[int(np.argmin(((V[members] - ctr) ** 2).sum(1)))]))
    return np.array(out, np.int64)


def signature_landmark_centroids_h5(out_h5, head_name, expr=None, gain=1.0,
                                    sig_min=0.33, sig_max=0.50, sig_dist=0.05,
                                    sig_smooth=0, n_eigs=80, n_channels=60,
                                    channel=None, wks_sigma=7.0):
    """Центроиды WKS-групп головы из HDF5 → позиции (N,3) на ОТОБРАЖАЕМОМ меше
    (нейтраль + gain·δ выражения). Сигнатура/группы считаются на нейтрали, а
    центр группы берётся на деформированных позициях — чтобы сфера села на меш,
    который видно в 3D."""
    import h5py
    with h5py.File(out_h5, "r") as h:
        g = h["heads"][head_name]
        neutral = g["neutral"][:].astype(np.float64)
        faces = g["faces"][:].astype(np.int64)
        disp = neutral.copy()
        if expr is not None and "expr" in g and expr in g["expr"]:
            disp = neutral + gain * g["expr"][expr][:].astype(np.float64)
    groups = _signature_groups(neutral, faces, sig_min, sig_max, sig_dist,
                               sig_smooth, n_eigs, n_channels, channel, wks_sigma)
    if not groups:
        return np.empty((0, 3), np.float64)
    return np.array([disp[m].mean(0) for m in groups], np.float64)


def _wks_landmarks_from_params(verts, faces, p):
    """WKS-лендмарки по параметрам из p (uv_wks_*). [] если выключено."""
    if not p.get('uv_wks_match'):
        return None
    return signature_landmark_verts(
        verts, faces,
        sig_min=float(p.get('wks_sig_min', 0.33)),
        sig_max=float(p.get('wks_sig_max', 0.50)),
        sig_dist=float(p.get('wks_sig_dist', 0.05)),
        sig_smooth=int(p.get('wks_sig_smooth', 0)),
        n_eigs=int(p.get('wks_desc_eigs', 80)),
        n_channels=int(p.get('wks_desc_channels', 60)),
        channel=p.get('wks_desc_channel'),
        wks_sigma=float(p.get('wks_desc_sigma', 7.0)))


def _letterbox_aspect(img, w, h):
    """Перемасштабируем квадратную картинку под реальные пропорции bbox w×h и
    вписываем в квадрат на чёрном фоне (без искажения геометрии)."""
    from PIL import Image
    res = img.shape[0]
    if w >= h:                                    # шире → сжать высоту
        new_w, new_h = res, max(1, int(round(res * h / w)))
    else:                                         # выше → сжать ширину
        new_w, new_h = max(1, int(round(res * w / h))), res
    pil = Image.fromarray(img).resize((new_w, new_h))
    canvas = Image.new("RGB", (res, res), (0, 0, 0))
    canvas.paste(pil, ((res - new_w) // 2, (res - new_h) // 2))
    return np.asarray(canvas)


def _relight_side(img, res, a0, a1, bbox2d, depth0, ddir, ax, scene):
    """Переосвещаем рендер боковым источником: повторяем рейкаст, берём нормали
    попаданий, считаем диффуз от света сбоку-сверху + ambient → рельеф виднее,
    чем при фронтальном шейде _shade_image."""
    import open3d as o3d
    x0, x1, y0, y1 = bbox2d
    us = np.linspace(x0, x1, res); vs = np.linspace(y1, y0, res)
    gu, gv = np.meshgrid(us, vs)
    origins = np.zeros((res * res, 3), dtype=np.float32)
    origins[:, a0] = gu.ravel(); origins[:, a1] = gv.ravel()
    origins[:, ax] = depth0
    dirs = np.zeros((res * res, 3), dtype=np.float32); dirs[:, ax] = ddir
    rays = np.concatenate([origins, dirs], axis=1)
    ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = ans['t_hit'].numpy()
    nrm = ans['primitive_normals'].numpy()
    hit = np.isfinite(t_hit).reshape(res, res)
    # источник света: сбоку-сверху, спереди (в экранных осях a0=гориз,a1=верт)
    L = np.zeros(3); L[a0] = 0.55; L[a1] = 0.55; L[ax] = -ddir * 0.62
    L = L / (np.linalg.norm(L) + 1e-9)
    diff = np.abs(nrm @ L).reshape(res, res)
    val = np.clip(0.22 + 0.78 * diff, 0, 1)        # ambient + diffuse
    base = np.array([0.82, 0.78, 0.74])            # тёплый тон кожи
    out = (val[:, :, None] * base * 255.0)
    out[~hit] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_vertex_scalar(img, faces, vscalar, res, a0, a1, bbox2d,
                           depth0, ddir, ax, scene):
    """Раскраска головы per-vertex скаляром vscalar∈[0,1] (CMAP_HEAT): для
    пикселя берём треугольник попадания и среднюю величину его вершин."""
    import open3d as o3d
    x0, x1, y0, y1 = bbox2d
    us = np.linspace(x0, x1, res); vs = np.linspace(y1, y0, res)
    gu, gv = np.meshgrid(us, vs)
    origins = np.zeros((res * res, 3), dtype=np.float32)
    origins[:, a0] = gu.ravel(); origins[:, a1] = gv.ravel()
    origins[:, ax] = depth0
    dirs = np.zeros((res * res, 3), dtype=np.float32); dirs[:, ax] = ddir
    rays = np.concatenate([origins, dirs], axis=1)
    ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = ans['t_hit'].numpy()
    tri_id = ans['primitive_ids'].numpy()
    hit = (np.isfinite(t_hit)
           & (tri_id != o3d.t.geometry.RaycastingScene.INVALID_ID)).reshape(res, res)
    F = np.asarray(faces)
    tri = np.clip(tri_id, 0, len(F) - 1)
    face_val = vscalar[F[tri]].mean(1).reshape(res, res)
    colors = pipe.to_colors(face_val.ravel(), pipe.CMAP_HEAT).reshape(res, res, 3)
    out = img.copy().astype(np.float32)
    val = (out.mean(2) / 255.0)                    # яркость рендера (рельеф)
    shaded = np.clip(0.35 + 0.65 * val[:, :, None], 0, 1) * colors * 255.0
    out[hit] = shaded[hit]
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_delta_color(img, verts, faces, delta, res, a0, a1, bbox2d,
                         depth0, ddir, ax, scene,
                         col_lo=(0.1, 0.2, 1.0), col_hi=(1.0, 0.1, 0.1)):
    """Раскрашиваем величину |δ| поверх рендера: для каждого пикселя берём
    треугольник попадания луча и среднюю |δ| его вершин → цвет, интерполируемый
    между col_lo (нет изменений) и col_hi (макс. изменение)."""
    import open3d as o3d
    mag = np.linalg.norm(delta, axis=1)
    mag = mag / max(mag.max(), 1e-9)
    x0, x1, y0, y1 = bbox2d
    us = np.linspace(x0, x1, res)
    vs = np.linspace(y1, y0, res)
    gu, gv = np.meshgrid(us, vs)
    origins = np.zeros((res * res, 3), dtype=np.float32)
    origins[:, a0] = gu.ravel(); origins[:, a1] = gv.ravel()
    origins[:, ax] = depth0
    dirs = np.zeros((res * res, 3), dtype=np.float32); dirs[:, ax] = ddir
    rays = np.concatenate([origins, dirs], axis=1)
    ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
    t_hit = ans['t_hit'].numpy()
    tri_id = ans['primitive_ids'].numpy()
    hit = np.isfinite(t_hit) & (tri_id != o3d.t.geometry.RaycastingScene.INVALID_ID)
    out = img.copy().astype(np.float32)
    F = np.asarray(faces)
    tri = np.clip(tri_id, 0, len(F) - 1)
    face_mag = mag[F[tri]].mean(1).reshape(res, res)   # средняя |δ| грани
    hitm = hit.reshape(res, res)
    # линейная интерполяция col_lo→col_hi по величине изменения
    lo = np.asarray(col_lo, np.float32); hi = np.asarray(col_hi, np.float32)
    t = face_mag[:, :, None]
    overlay = lo[None, None, :] * (1 - t) + hi[None, None, :] * t
    val = (out.mean(2) / 255.0)                        # яркость освещения
    blend = np.clip(0.35 + 0.65 * val[:, :, None], 0, 1) * overlay * 255.0
    out[hitm] = (0.4 * out[hitm] + 0.6 * blend[hitm])
    return np.clip(out, 0, 255).astype(np.uint8)


def export_head_obj(out_h5, head_name, expr=None, gain=1.0, out_obj=None,
                    descriptor=None, colorize=False,
                    col_lo=(0.1, 0.2, 1.0), col_hi=(1.0, 0.1, 0.1),
                    desc_n_eigs=80, desc_channels=60, desc_channel=None,
                    desc_wks_sigma=7.0):
    """Экспортируем голову из HDF5 в OBJ (нейтраль + gain·δ) для 3D-просмотра.

    descriptor='wks' → пишем per-vertex цвета по WKS (как в превью).
    colorize=True (и есть δ) → per-vertex цвета по |δ| градиентом col_lo→col_hi.
    Цвета вершин читает mesh_viewer (формат 'v x y z r g b').
    Возвращает путь к OBJ."""
    import h5py
    import tempfile
    with h5py.File(out_h5, "r") as h:
        g = h["heads"][head_name]
        neutral = g["neutral"][:].astype(np.float64)
        faces = g["faces"][:].astype(np.int64)
        delta = None
        if expr is not None and expr in g["expr"]:
            delta = g["expr"][expr][:].astype(np.float64)
        verts = neutral + (gain * delta if delta is not None else 0.0)

    colors = None
    if descriptor == "wks":
        dmap = _head_descriptor(out_h5, head_name, neutral, faces,
                                n_eigs=desc_n_eigs, n_channels=desc_channels,
                                wks_sigma=desc_wks_sigma, channel=desc_channel)
        colors = pipe.to_colors(dmap, pipe.CMAP_HEAT)
    elif colorize and delta is not None:
        mag = np.linalg.norm(delta, axis=1)
        mag = mag / max(mag.max(), 1e-9)
        lo = np.asarray(col_lo); hi = np.asarray(col_hi)
        colors = lo[None, :] * (1 - mag[:, None]) + hi[None, :] * mag[:, None]

    if out_obj is None:
        out_obj = tempfile.NamedTemporaryFile(suffix=".obj", delete=False).name
    pipe._write_obj(out_obj, verts, faces, C=colors)
    return out_obj


# ── головы из папки ─────────────────────────────────────────────────────────

def list_head_files(heads_dir):
    exts = (".fbx", ".obj", ".ply", ".stl")
    return sorted(f for f in Path(heads_dir).iterdir()
                  if f.suffix.lower() in exts)


# ── обработка одной головы (общая для seq и parallel) ───────────────────────

def _process_head(fpath, p, res_f0, zd_f, exprs, expr_names, landmarks,
                  wks_f=None):
    """Полная обработка ОДНОЙ головы: загрузка → анкоры → зоны → перенос всех
    выражений. Возвращает dict для записи в HDF5 или {'error': ...}.
    Не пишет в HDF5 (для мультипроцессинга).

    wks_f → WKS-лендмарки источника (FLAME, глоб. индексы); если задано и
    включён uv_wks_match — считаем лендмарки таргета и матчим в UV-warp."""
    try:
        v_raw, faces_x = pipe.load_custom_mesh(str(fpath))
        verts_x = pipe.normalize_bbox(v_raw)
        anch_x, dbg_x = auto_anchors.auto_anchors(
            verts_x, faces_x.astype(np.int64), landmark_indices=landmarks)
        if not dbg_x.get('ok') or len(anch_x) < 1:
            raise RuntimeError("лицо не найдено MediaPipe")
        res_x0 = _build_zones(verts_x, faces_x, anch_x,
                              np.zeros_like(verts_x), p)
        zd_x = _relax_zones(res_x0, p)
        faces64 = faces_x.astype(np.int64)
        # WKS-лендмарки таргета (если включён матч)
        wks_x = (_wks_landmarks_from_params(verts_x, faces_x, p)
                 if (wks_f is not None and p.get('uv_wks_match')) else None)
        deltas = {}
        for ename in expr_names:
            res_src = dict(res_f0)
            res_src['delta_native'] = exprs[ename]
            tr = pipe.transfer_deformations_uv(
                res_src, res_x0,
                flat=("world" if p.get('uv_world_orient')
                      else bool(p.get('uv_flat', False))),
                align_pca_icp=bool(p.get('uv_align_pca_icp', False)),
                # WKS-match сам включает warp (граница тянется в любом случае)
                warp_heat=(bool(p.get('uv_warp_heat', False))
                           or bool(p.get('uv_wks_match', False))),
                warp_heat_t=float(p.get('uv_warp_heat_t', 0.05)),
                warp_min_dist=float(p.get('uv_warp_min_dist', 0.0)),
                interp_delta=bool(p.get('uv_interp_delta', True)),
                zd_src=zd_f, zd_dst=zd_x,
                wks_src=wks_f, wks_dst=wks_x)
            if tr is None:
                raise RuntimeError(f"перенос '{ename}' вернул None")
            if p.get('smooth_transferred_groups', True):
                it = int(p.get('group_smooth_iters', 8))
                tr['gcid'] = pipe.smooth_labels(tr['gcid'], faces64, n_iter=it)
                tr['zone'] = pipe.smooth_labels(tr['zone'], faces64, n_iter=it)
            delta = pipe.smooth_delta(
                tr['delta'], faces64,
                n_iter=int(p.get('smooth_iters', 3)),
                alpha=float(p.get('smooth_alpha', 0.5)))
            deltas[ename] = delta.astype(np.float32)
        return {'name': fpath.stem, 'neutral': verts_x.astype(np.float32),
                'faces': faces_x.astype(np.int32), 'deltas': deltas}
    except Exception as e:
        return {'name': fpath.stem, 'error': str(e)}


# глобальное состояние воркера (устанавливается _pool_init один раз на процесс)
_WORKER = {}


def _pool_init(p, res_f0, zd_f, exprs, expr_names, landmarks, wks_f):
    _WORKER['p'] = p
    _WORKER['res_f0'] = res_f0
    _WORKER['zd_f'] = zd_f
    _WORKER['exprs'] = exprs
    _WORKER['expr_names'] = expr_names
    _WORKER['landmarks'] = landmarks
    _WORKER['wks_f'] = wks_f


def _pool_task(fpath):
    return _process_head(fpath, _WORKER['p'], _WORKER['res_f0'],
                         _WORKER['zd_f'], _WORKER['exprs'],
                         _WORKER['expr_names'], _WORKER['landmarks'],
                         _WORKER['wks_f'])


# ── основной батч-перенос ───────────────────────────────────────────────────

def run_transfer(ref_h5, heads_dir, out_h5, params=None, progress=None,
                 head_from=None, head_to=None, n_workers=1):
    """Перенос ВСЕХ выражений reference на головы папки → out_h5.

    head_from / head_to → срез голов по индексу (0-based, включительно). None →
    с начала / до конца. Берём только головы files[head_from:head_to+1].

    n_workers > 1 → мультипроцессинг по головам (головы независимы). Запись в
    HDF5 идёт в главном процессе по мере готовности (HDF5 не пишут параллельно).

    progress(stage, cur, total, msg) — колбэк прогресса.
    Возвращает (n_heads_ok, n_heads_total, n_expr)."""
    import h5py
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update(params)
    landmarks = list(p.get('landmarks') or DEFAULT_PARAMS['landmarks'])

    # 1. reference: нейтраль FLAME + выражения (+ активации мышц), зоны на
    # нейтрали считаем ОДИН раз
    neutral_f, faces_f, exprs, acts, muscle_names = read_reference(ref_h5)
    neutral_f = pipe.normalize_bbox(neutral_f)
    expr_names = sorted(exprs)
    n_expr = len(expr_names)
    if progress:
        progress("ref", 0, 1, f"reference: {n_expr} выражений")

    anch_f, dbg_f = auto_anchors.auto_anchors(
        neutral_f, faces_f.astype(np.int64), landmark_indices=landmarks)
    if not dbg_f.get('ok') or len(anch_f) < 1:
        raise RuntimeError("MediaPipe не нашёл лицо на reference-нейтрали")
    # зоны на reference строим один раз (δ=0 для зон, δ выражения подставляем
    # позже per-expression при переносе).
    res_f0 = _build_zones(neutral_f, faces_f, anch_f,
                          np.zeros_like(neutral_f), p)
    zd_f = _relax_zones(res_f0, p)

    # WKS-лендмарки источника (FLAME) — считаем ОДИН раз (для UV-WKS-match)
    wks_f = None
    if p.get('uv_wks_match'):
        wks_f = _wks_landmarks_from_params(neutral_f, faces_f, p)
        if progress:
            progress("ref", 1, 1,
                     f"WKS-лендмарков источника: {0 if wks_f is None else len(wks_f)}")

    files = list_head_files(heads_dir)
    # срез по диапазону голов (head_from..head_to включительно, 0-based)
    lo = 0 if head_from is None else max(int(head_from), 0)
    hi = len(files) if head_to is None else min(int(head_to) + 1, len(files))
    files = files[lo:hi]
    n_total = len(files)
    Path(out_h5).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with h5py.File(out_h5, "w") as out:
        out.attrs['n_expr'] = n_expr
        out.attrs['expr_names'] = json.dumps(expr_names)
        out.attrs['has_activations'] = bool(acts)
        out.attrs['schema'] = ("expressions/<выраж>/activations(M,) — общие "
                               "активации мышц; heads/<имя>/{neutral(N,3),"
                               "faces(K,3),expr/<выраж>/delta(N,3)}")
        # активации мышц — СВОЙСТВО ВЫРАЖЕНИЯ (общие для всех голов): пишем один
        # раз в /expressions, не дублируя по головам.
        eg_ref = out.create_group("expressions")
        if muscle_names is not None:
            dt = h5py.string_dtype(encoding="utf-8")
            out.create_dataset("muscle_names",
                               data=np.array(muscle_names, dtype=object),
                               dtype=dt)
        for ename in expr_names:
            ee = eg_ref.create_group(ename)
            if acts and ename in acts:
                ee.create_dataset("activations", data=acts[ename])
        def _write_result(fi, r):
            nonlocal n_ok
            hname = r['name']
            if 'error' in r:
                if progress:
                    progress("head_err", fi + 1, n_total,
                             f"{hname}: пропуск — {r['error']}")
                print(f"  [{fi+1}/{n_total}] {hname}: пропуск — {r['error']}")
                return
            g = out.create_group(f"heads/{hname}")
            g.create_dataset('neutral', data=r['neutral'])
            g.create_dataset('faces', data=r['faces'])
            g.attrs['n_verts'] = len(r['neutral'])
            ge = g.create_group('expr')
            for ename, delta in r['deltas'].items():
                ge.create_dataset(ename, data=delta)
            n_ok += 1
            if progress:
                progress("head_done", fi + 1, n_total,
                         f"{hname}: ✓ {n_expr} выраж.")

        nw = max(int(n_workers), 1)
        if nw <= 1 or n_total <= 1:
            # ── последовательно ──
            for fi, fpath in enumerate(files):
                if progress:
                    progress("head", fi, n_total, f"{fpath.stem}: обработка")
                r = _process_head(fpath, p, res_f0, zd_f, exprs, expr_names,
                                  landmarks, wks_f)
                _write_result(fi, r)
        else:
            # ── мультипроцессинг по головам ──
            import multiprocessing as mp
            ctx = mp.get_context("spawn")          # безопасно для Open3D/MediaPipe
            nw = min(nw, n_total)
            if progress:
                progress("head", 0, n_total, f"параллельно: {nw} процессов")
            with ctx.Pool(processes=nw, initializer=_pool_init,
                          initargs=(p, res_f0, zd_f, exprs, expr_names,
                                    landmarks, wks_f)) as pool:
                done = 0
                for r in pool.imap_unordered(_pool_task, files):
                    _write_result(done, r)
                    done += 1
        out.attrs['n_heads'] = n_ok
    return n_ok, n_total, n_expr
