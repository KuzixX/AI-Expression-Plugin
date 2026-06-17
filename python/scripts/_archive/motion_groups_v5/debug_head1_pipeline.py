"""
Motion-Groups Transfer Pipeline — VERSION 5.0

Debug pipeline для HEAD 1 (source FLAME) → HEAD 2 (target FBX) с GUI выбором
режима matching кластеров.

После v4 проведён большой strip — оставлены только рабочие методы плюс 3
экспериментальных варианта (B/C/D) на тему «как избежать перекрытия heat-зон».

4 РЕЖИМА МАТЧИНГА:
  heat_zone_xyz       — ⭐ Per anchor zone: point-cloud alignment (centroid/
                         scale/non_rigid) + xyz NN (рабочий по умолчанию)
  zonal_1d        (B) — Hard argmax-partition (каждая вершина строго в одной
                         anchor-зоне) → xyz NN внутри
  sequential_anchor (C) — Anchor'ы обрабатываются по очереди (sorted by max
                          heat или index), занятые вершины пропускаются
  decorr_heat     (D) — Gram-Schmidt orthogonalization heat-полей → matching
                         на decorrelated heat (устраняет перекрытие математически)

ПАРАМЕТРЫ alignment (общие для всех 4):
  heat_zone_alignment_mode  — 'centroid' / 'scale' / 'non_rigid'
  heat_zone_rigid           — scale-align ON/OFF
  heat_zone_non_rigid_iters — RBF iterations (для non_rigid)
  heat_zone_use_anchor_align — anchor-based pivot (off=centroid)
  heat_zone_use_rotation    — Procrustes pre-step
  heat_zone_smooth          — label majority-vote iterations

SMOOTHING & GEO FILTER:
  smooth_iters / smooth_iters_fbx — Laplacian δ-smoothing (отдельно head1/fbx)
  geo_filter_enable + tolerance   — sanity filter переноса по геодезии

HKS/WKS DIAGNOSTIC (отдельный шаг до anchor selection):
  viz_hks_enable + viz_hks_type (hks/wks/combined) + cluster params
  Показывает кластеризацию вершин по intrinsic сигнатурам — diagnostic only,
  не используется в matching.

DUMPS (в python/scripts/debug_output/run_*/):
  head1/heat.csv, clusters.json, clusters_flat.csv, delta_*.csv, verts_*.csv
  fbx/  heat.csv, target_clusters.json, delta_raw.csv, delta_smoothed.csv

СОПУТСТВУЮЩИЕ СКРИПТЫ:
  visualize_dumps.py    — batch-визуализация дампов (matplotlib offline)
  align_heat_tables.py  — выравнивание heat-таблиц HEAD1 ↔ FBX (методы A, B)
  compare_heads.py      — standalone сравнение мешей через HKS/WKS
  functional_map_fit.py — classical Functional Maps registration
"""

__version__ = "5.0"

import argparse
import datetime as _dt
import json
import pickle
import subprocess
import tempfile
import time as time_mod
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import open3d as o3d
from sklearn.cluster import KMeans


FLAME_PKL = ("Muscle-autoskinner/Assets/Meshes/FLAME/"
             "FLAME2023 Open for commercial use/flame2023_Open.pkl")
# Колормэпы вручную (без matplotlib — он крашит pipeline на macOS)

def _cmap_hot(v):
    """Hot colormap: black → red → yellow → white. v in [0, 1] shape (N,) → (N, 3)."""
    v = np.clip(v, 0, 1)
    rgb = np.zeros((len(v), 3))
    rgb[:, 0] = np.clip(v * 3, 0, 1)             # R: 0 → 1 в первой трети
    rgb[:, 1] = np.clip((v - 0.33) * 3, 0, 1)    # G: 0 → 1 во второй трети
    rgb[:, 2] = np.clip((v - 0.66) * 3, 0, 1)    # B: 0 → 1 в последней трети
    return rgb


def _cmap_cool(v):
    """Cool colormap: cyan → magenta. v in [0, 1] → (N, 3)."""
    v = np.clip(v, 0, 1)
    rgb = np.empty((len(v), 3))
    rgb[:, 0] = v                                  # R: 0 → 1
    rgb[:, 1] = 1.0 - v                            # G: 1 → 0
    rgb[:, 2] = 1.0                                # B: всегда 1
    return rgb


CMAP_HEAT = _cmap_hot
CMAP_DISP = _cmap_cool


# ── Загрузка / геометрия ─────────────────────────────────────────────────────

def load_flame(path):
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    def to_np(x):
        if hasattr(x, "r"): return np.array(x.r)
        if hasattr(x, "toarray"): return x.toarray()
        return np.array(x)
    return (to_np(d["v_template"]).astype(np.float64),
            to_np(d["shapedirs"]).astype(np.float64),
            to_np(d["f"]).astype(np.int64))


def apply_betas(v_t, sd, betas_dict):
    betas = np.zeros(sd.shape[2])
    for i, val in betas_dict.items(): betas[i] = val
    return v_t + np.einsum("ijk,k->ij", sd, betas)


def save_matrix_csv(path, array, header=None):
    """Сохраняет numpy матрицу в CSV. header — строка с именами колонок."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(array), delimiter=',',
                header=(header or ''), comments='')


def save_clusters_json(path, clusters_per_anchor):
    """Сохраняет кластерные дескрипторы (с μ, R, S, stretches, indices) в JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = []
    for a, cls in enumerate(clusters_per_anchor):
        for ci, cl in enumerate(cls):
            out.append({
                'anchor_idx':    int(cl['anchor_idx']),
                'cluster_idx':   ci,
                'n_verts':       int(len(cl['indices'])),
                'indices':       cl['indices'].tolist(),
                'heat_weights':  cl['heat_weights'].tolist(),
                'c_rest':        cl['c_rest'].tolist(),
                'spatial_sigma': float(cl['spatial_sigma']),
                'mu':            cl['mu'].tolist(),
                'F':             cl['F'].tolist(),
                'R':             cl['R'].tolist(),
                'S':             cl['S'].tolist(),
                'stretches':     cl['stretches'].tolist(),
                'axes':          cl['axes'].tolist(),
            })
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


def save_target_clusters_json(path, target_clusters, cluster_color_map=None):
    """Сохраняет result разбиения на target меш."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out = []
    for tc in target_clusters:
        s = tc['source']
        entry = {
            'source_anchor_idx':  int(s['anchor_idx']),
            'source_mu':          s['mu'].tolist(),
            'source_R':           s['R'].tolist(),
            'source_S':           s['S'].tolist(),
            'source_c_rest':      s['c_rest'].tolist(),
            'source_sigma':       float(s['spatial_sigma']),
            'target_indices':     tc['target_indices'].tolist(),
            'target_heat':        tc['target_heat'].tolist(),
            'target_c':           tc['c_target'].tolist(),
            'n_target_verts':     int(len(tc['target_indices'])),
        }
        if cluster_color_map is not None:
            col = cluster_color_map.get(id(s))
            if col is None:
                col = [0.5, 0.5, 0.5]
            elif hasattr(col, 'tolist'):
                col = col.tolist()
            entry['display_color'] = list(col)
        out.append(entry)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)


def save_metadata_json(path, params, shape, expr, n_anchors, src1, src_fbx=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'timestamp':       _dt.datetime.now().isoformat(),
        'shape_betas':     {str(k): float(v) for k, v in shape.items()},
        'expr_betas':      {str(k): float(v) for k, v in expr.items()},
        'n_anchors':       int(n_anchors),
        'src_head1':       [int(x) for x in src1],
        'src_fbx':         [int(x) for x in src_fbx] if src_fbx is not None else None,
        'params':          {k: (v if not isinstance(v, dict) else
                                 {str(kk): float(vv) for kk, vv in v.items()})
                             for k, v in params.items() if not k.startswith('_')},
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_custom_mesh(path):
    """FBX → OBJ через assimp → trimesh (process=True мержит UV-splits)."""
    import trimesh as _tm
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False); tmp.close()
    r = subprocess.run(["assimp", "export", path, tmp.name],
                       capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(r.stderr)
    m = _tm.load(tmp.name, force="mesh", process=False)
    Path(tmp.name).unlink(missing_ok=True)
    m = _tm.Trimesh(vertices=m.vertices, faces=m.faces, process=True)
    return (np.array(m.vertices, dtype=np.float64),
            np.array(m.faces, dtype=np.int64))


def normalize_bbox(v):
    v = v - v.mean(0)
    return v / (np.linalg.norm(v.max(0) - v.min(0)) + 1e-12)


def build_operators(verts, faces):
    N = len(verts)
    row, col, data = [], [], []
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        vi, vj, vk = verts[faces[:, i]], verts[faces[:, j]], verts[faces[:, k]]
        u, v = vi - vk, vj - vk
        cos_a = (u * v).sum(1)
        sin_a = np.linalg.norm(np.cross(u, v), axis=1).clip(1e-8)
        cot = cos_a / sin_a * 0.5
        fi, fj = faces[:, i], faces[:, j]
        row += fi.tolist(); col += fj.tolist(); data += cot.tolist()
        row += fj.tolist(); col += fi.tolist(); data += cot.tolist()
    W = sp.csr_matrix((data, (row, col)), shape=(N, N))
    L = (sp.diags(np.array(W.sum(1)).ravel()) - W).astype(np.float64)
    areas = np.zeros(N)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fa = np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1) / 6.0
    for i in range(3): np.add.at(areas, faces[:, i], fa)
    return L, sp.diags(areas)


# ── Polar decomposition ──────────────────────────────────────────────────────

def polar_decomposition(heat, delta, verts, eps=1e-8):
    w = np.clip(heat, 0, None)
    W = max(w.sum(), eps)
    mu = (w[:, None] * delta).sum(0) / W
    c_rest = (w[:, None] * verts).sum(0) / W
    p = verts - c_rest
    q = p + (delta - mu)
    wp = w[:, None] * p
    A = q.T @ wp
    B = p.T @ wp; B = 0.5 * (B + B.T)
    I3 = np.eye(3)
    F = np.linalg.solve(B + eps * I3, A.T).T
    U, sig, Vt = np.linalg.svd(F)
    det_sign = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, det_sign])
    R = U @ D @ Vt
    S = R.T @ F; S = 0.5 * (S + S.T)
    eigvals, eigvecs = np.linalg.eigh(S)
    stretches = eigvals[::-1]
    axes = eigvecs[:, ::-1]
    return {'mu': mu, 'F': F, 'R': R, 'S': S,
            'stretches': stretches, 'axes': axes, 'c_rest': c_rest}


def axis_angle_from_R(R):
    trace = np.trace(R)
    cos_a = np.clip((trace - 1) * 0.5, -1 + 1e-9, 1 - 1e-9)
    angle = np.arccos(cos_a)
    sin_a = max(np.sin(angle), 1e-9)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2 * sin_a)
    return axis, angle


# ── Кластеризация ────────────────────────────────────────────────────────────

def cluster_zone(heat, delta, verts, anchor_idx,
                  heat_threshold=0.05, n_clusters_max=5,
                  position_weight=1.5, min_cluster_size=4,
                  clustering_method='kmeans',
                  similarity_threshold=0.3,
                  print_quality=True):
    """Группировка вершин anchor-зоны в кластеры.

    clustering_method:
      'kmeans'        — фикс. число n_clusters_max (default, как раньше)
      'agglomerative' — порог similarity_threshold в нормированном feature-space
                        (Ward linkage; число кластеров определяется автоматически)

    similarity_threshold: distance в нормированном feature-пространстве.
      Малое (0.1) → больше мелких кластеров (строгое разделение)
      Большое (0.5) → меньше крупных кластеров (мягкое объединение)
      Эффективно работает с alignment_mode='agglomerative'.

    print_quality=True печатает silhouette score per zone (метрика чистоты).
    """
    heat_max = max(heat.max(), 1e-12)
    active_idx = np.where(heat > heat_threshold * heat_max)[0]
    if len(active_idx) < min_cluster_size * 2:
        return []
    a_verts = verts[active_idx]
    a_delta = delta[active_idx]
    a_heat = heat[active_idx]

    d_scale = max(np.linalg.norm(a_delta, axis=1).max(), 1e-8)
    p_mean = a_verts.mean(0)
    p_scale = max(np.linalg.norm(a_verts - p_mean, axis=1).max(), 1e-8)
    features = np.concatenate([
        a_delta / d_scale,
        (a_verts - p_mean) / p_scale * position_weight,
    ], axis=1)

    if clustering_method == 'agglomerative':
        from sklearn.cluster import AgglomerativeClustering
        # n_clusters=None + distance_threshold = автоматическое определение
        ac = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=similarity_threshold,
            linkage='ward')
        labels = ac.fit_predict(features)
        # ограничиваем сверху n_clusters_max если получилось больше
        n_clusters = labels.max() + 1
        if n_clusters > n_clusters_max:
            # перекластеризуем K-means'ом до n_clusters_max
            km = KMeans(n_clusters=n_clusters_max, n_init=8, random_state=0)
            labels = km.fit_predict(features)
            n_clusters = n_clusters_max
        elif n_clusters < 2:
            # если получился 1 кластер — форсим K-means с min(2, ...)
            n_clusters = min(2, n_clusters_max)
            km = KMeans(n_clusters=n_clusters, n_init=8, random_state=0)
            labels = km.fit_predict(features)
    else:
        # KMeans (default)
        n_clusters = max(2, min(n_clusters_max, len(active_idx) // 30))
        km = KMeans(n_clusters=n_clusters, n_init=8, random_state=0)
        labels = km.fit_predict(features)

    # Quality: silhouette score (от -1 до +1, чем выше тем чище разделение)
    if print_quality and n_clusters >= 2 and len(active_idx) > n_clusters + 1:
        try:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(features, labels)
            quality_label = ("отлично" if sil > 0.5 else
                              "хорошо" if sil > 0.25 else
                              "средне" if sil > 0.1 else
                              "слабо")
            print(f"      [zone a{anchor_idx}] method={clustering_method}, "
                  f"K={n_clusters}, N_active={len(active_idx)}, "
                  f"silhouette={sil:.3f} ({quality_label})")
        except Exception:
            pass

    clusters = []
    for c in range(n_clusters):
        mask = labels == c
        if mask.sum() < min_cluster_size: continue
        idx = active_idx[mask]
        cl_heat = a_heat[mask]
        cl_verts = a_verts[mask]
        cl_delta = a_delta[mask]
        polar = polar_decomposition(cl_heat, cl_delta, cl_verts)
        c_rest = polar['c_rest']
        rms = np.sqrt(
            (cl_heat * np.linalg.norm(cl_verts - c_rest, axis=-1) ** 2).sum()
            / max(cl_heat.sum(), 1e-12)
        )
        clusters.append({
            'anchor_idx': anchor_idx,
            'spatial_sigma': max(rms, 1e-4),
            'indices': idx,
            'heat_weights': cl_heat,
            **polar,
        })
    return clusters


# ── Spectral descriptors (HKS / WKS) ─────────────────────────────────────────

def compute_spectrum(verts, faces, n_eigs=128):
    """Generalised eigenproblem L·v = λ·M·v через scipy.sparse.linalg.eigsh.
    Возвращает (eigvals, eigvecs) для первых n_eigs мод (нижние частоты).
    """
    from scipy.sparse.linalg import eigsh
    N = len(verts)
    print(f"  Computing eigendecomposition (k={n_eigs}, mesh={N} verts)...")
    L_dense, M_diag = build_operators(verts, faces)
    L_sp = L_dense.tocsr() if hasattr(L_dense, 'tocsr') else sp.csr_matrix(L_dense)
    M_sp = M_diag if sp.issparse(M_diag) else sp.diags(np.asarray(M_diag).ravel())
    k = min(n_eigs, N - 2)
    try:
        eigvals, eigvecs = eigsh(L_sp.astype(np.float64), k=k, M=M_sp,
                                  sigma=-1e-6, which='LM')
    except Exception:
        eigvals, eigvecs = eigsh(L_sp.astype(np.float64), k=k, M=M_sp, which='SM')
    order = np.argsort(eigvals)
    eigvals = np.clip(eigvals[order], 0.0, None)
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def compute_anchor_heat_multi_t(eigvals, eigvecs, anchor_indices, n_times=8,
                                  t_min=None, t_max=None):
    """Multi-scale heat от anchor-точек через spectral expansion.

    heat(v, anchor_a, t) = Σ_k exp(-t·λ_k) · ψ_k(anchor_a) · ψ_k(v)

    Один eigendecomp → heat для любых (anchor, time) пар за O(N·k_eigs).

    Returns:
        H_multi: (K * T, N) — каждая строка одно scalar поле
                 порядок: [anchor_0/t_0, anchor_0/t_1, ..., anchor_K/t_T]
        times:   (T,) — log-spaced timesteps
    """
    K = len(anchor_indices)
    N = eigvecs.shape[0]

    # Авто-выбор диапазона t из спектра (как в HKS)
    if t_min is None or t_max is None:
        lam_min = max(float(eigvals[1]), 1e-6)        # пропускаем λ_0 ≈ 0
        lam_max = float(eigvals[-1])
        if t_min is None: t_min = 4 * np.log(10) / lam_max
        if t_max is None: t_max = 4 * np.log(10) / lam_min

    times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)

    H_multi = np.zeros((K * n_times, N), dtype=np.float64)
    for ki, a in enumerate(anchor_indices):
        psi_a = eigvecs[int(a), :]                    # (k_eigs,)
        for ti, t in enumerate(times):
            weights = np.exp(-t * eigvals) * psi_a    # (k_eigs,)
            H_multi[ki * n_times + ti] = eigvecs @ weights
    return H_multi, times


def cluster_zones_global(verts, delta, heat_per_anchor,
                          n_clusters_global=20,
                          position_weight=1.5,
                          heat_threshold=0.05,
                          min_cluster_size=4,
                          min_anchor_heat_share=0.05,
                          verbose=True):
    """ГЛОБАЛЬНАЯ кластеризация motion-групп на всём меше с multi-anchor views.

    Концепция:
        - Сначала запускаем K-means на ВСЕХ active вершинах (без разбиения
          на anchor-зоны)
        - Получаем `n_clusters_global` уникальных motion-групп
        - Каждую группу распределяем по anchor-зонам с heat-веcами:
          одна и та же группа может «принадлежать» нескольким anchor'ам
          с разной силой влияния (heat[anchor, v])

    Возвращает структуру clusters_per_anchor совместимую с downstream-кодом:
        clusters_per_anchor[a] = список views в anchor-зоне a
        Каждый view содержит:
            - 'indices'         — global vertex indices в этой motion-группе
            - 'heat_weights'    — heat[a, v] для каждой v (per-anchor view)
            - 'anchor_idx'      — a (как обычно)
            - 'global_group_id' — id мотейн-группы (общий между views)
            - 'anchor_share'    — доля тепла этого anchor'а в группе [0..1]
            - R, S, mu, c_rest, etc — ОБЩИЕ для всех views одной motion-группы
              (polar decomposition посчитан один раз на всех её вершинах)

    `min_anchor_heat_share`: если доля тепла anchor'а в группе ниже этого
    порога, view для этого anchor'а не создаётся (anchor не «владеет» этой
    группой).
    """
    K, N = heat_per_anchor.shape

    # Active mask: вершина active если ХОТЯ БЫ ОДИН anchor её нагрел
    heat_max_pa = heat_per_anchor.max(axis=1, keepdims=True).clip(min=1e-12)
    heat_norm = heat_per_anchor / heat_max_pa
    active_mask = heat_norm.max(axis=0) > heat_threshold
    active_idx = np.where(active_mask)[0]

    if len(active_idx) < min_cluster_size * 2:
        if verbose:
            print(f"  ⚠ global clustering: слишком мало active вершин "
                  f"({len(active_idx)})")
        return [[] for _ in range(K)]

    a_verts = verts[active_idx]
    a_delta = delta[active_idx]
    a_heat  = heat_per_anchor[:, active_idx]                          # (K, M)

    # K-means features
    d_scale = max(np.linalg.norm(a_delta, axis=1).max(), 1e-8)
    p_mean = a_verts.mean(0)
    p_scale = max(np.linalg.norm(a_verts - p_mean, axis=1).max(), 1e-8)
    features = np.concatenate([
        a_delta / d_scale,
        (a_verts - p_mean) / p_scale * position_weight,
    ], axis=1)

    n_clusters_actual = min(n_clusters_global, max(len(active_idx) // 10, 2))
    n_clusters_actual = max(n_clusters_actual, 2)

    km = KMeans(n_clusters=n_clusters_actual, n_init=10, random_state=0)
    labels = km.fit_predict(features)

    # Quality metric
    if verbose and n_clusters_actual >= 2 and len(active_idx) > n_clusters_actual + 1:
        try:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(features, labels)
            print(f"    silhouette={sil:.3f}")
        except Exception:
            pass

    clusters_per_anchor = [[] for _ in range(K)]
    n_views_total = 0
    group_stats = []

    for g in range(n_clusters_actual):
        g_mask = labels == g
        if g_mask.sum() < min_cluster_size: continue

        g_idx_active = np.where(g_mask)[0]
        g_global_idx = active_idx[g_idx_active]

        # Heat sum per anchor для этой motion-группы
        heat_for_g = a_heat[:, g_idx_active]                          # (K, |G|)
        anchor_heat_sums = heat_for_g.sum(axis=1)                     # (K,)
        total_heat = anchor_heat_sums.sum()
        if total_heat < 1e-12: continue
        anchor_shares = anchor_heat_sums / total_heat

        # Polar decomposition — ОДИН РАЗ для всей motion-группы.
        # Веса = max heat across anchors (анатомически важные точки сильнее)
        g_heat_max = heat_for_g.max(axis=0)
        g_verts_arr = verts[g_global_idx]
        g_delta_arr = delta[g_global_idx]
        polar = polar_decomposition(g_heat_max, g_delta_arr, g_verts_arr)
        c_rest = polar['c_rest']
        rms = float(np.sqrt(
            (g_heat_max * np.linalg.norm(g_verts_arr - c_rest, axis=-1) ** 2).sum()
            / max(g_heat_max.sum(), 1e-12)))
        spatial_sigma = max(rms, 1e-4)

        # Создаём views для каждого anchor'а у которого доля тепла достаточна
        anchors_with_views = []
        for a in range(K):
            if anchor_shares[a] < min_anchor_heat_share: continue
            view_heat = heat_for_g[a]
            view = {
                'anchor_idx':       int(a),
                'spatial_sigma':    spatial_sigma,
                'indices':          g_global_idx.copy(),
                'heat_weights':     view_heat.copy(),
                'global_group_id':  int(g),
                'anchor_share':     float(anchor_shares[a]),
                **polar,
            }
            clusters_per_anchor[a].append(view)
            n_views_total += 1
            anchors_with_views.append(a)

        group_stats.append({
            'global_group_id': g,
            'n_verts': int(g_mask.sum()),
            'anchors': anchors_with_views,
            'anchor_shares': anchor_shares.tolist(),
        })

    if verbose:
        print(f"  ── GLOBAL motion-groups clustering ──")
        print(f"    Active вершин: {len(active_idx)}/{N} "
              f"({100*len(active_idx)/N:.1f}%)")
        print(f"    Уникальных motion-групп: {n_clusters_actual}")
        print(f"    Anchor-views создано: {n_views_total}")
        print(f"    Группы распределены по anchor'ам "
              f"(min_share={min_anchor_heat_share:.0%}):")
        for a in range(K):
            n_views = len(clusters_per_anchor[a])
            print(f"      anchor {a}: {n_views} views этой anchor-зоны")

        # Сколько групп multi-anchor (присутствуют в >1 anchor-зоне)
        multi = sum(1 for s in group_stats if len(s['anchors']) > 1)
        single = sum(1 for s in group_stats if len(s['anchors']) == 1)
        print(f"    Motion-групп с multi-anchor влиянием: {multi}")
        print(f"    Motion-групп с single-anchor: {single}")

    return clusters_per_anchor


def merge_cross_anchor_motion_groups(
        clusters_per_anchor, verts, delta, heat_per_anchor,
        overlap_threshold=0.3,
        transform_sim_threshold=0.20,
        verbose=True,
        require_overlap=False):
    """ДЕДУПЛИКАЦИЯ motion-групп по идентичности трансформации.

    Алгоритм:
        1. Перебираем ВСЕ пары motion-групп (через все anchor-зоны)
        2. Сравниваем трансформации R и μ:
             ||R_A - R_B|| + ||μ_A - μ_B|| / ||μ_avg||  < sim_threshold
           → groups считаются идентичными (одна и та же мышца / движение)
        3. (опц., require_overlap=True) дополнительно требуем |A ∩ B| ≥ overlap×min
        4. Connected components на графе эквивалентности
        5. В каждом CC размером ≥ 2:
           - Union vertex indices (heat_weights = max per vertex)
           - Пересчитываем polar decomposition (R, S, μ, c) на merged данных
           - Primary anchor = тот у которого max heat-sum в merged зоне
           → один merged-cluster заменяет несколько duplicates

    Returns: new clusters_per_anchor с дедуплицированными группами.
             Merged clusters содержат поле merged_from_anchors.
    """
    K = len(clusters_per_anchor)
    all_cl = []
    for a_idx, cls in enumerate(clusters_per_anchor):
        for cl in cls:
            all_cl.append((a_idx, cl))
    N_cl = len(all_cl)
    if N_cl <= 1:
        return clusters_per_anchor

    # Union-find
    parent = list(range(N_cl))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py

    n_edges = 0
    for i in range(N_cl):
        a_i, cl_i = all_cl[i]
        idx_i = set(int(v) for v in cl_i['indices']) if require_overlap else None
        for j in range(i + 1, N_cl):
            a_j, cl_j = all_cl[j]
            # ВНИМАНИЕ: теперь дедупликация работает ПО ВСЕМ парам
            # (включая same-anchor) — критерий = только transform similarity.
            # Опционально require_overlap=True добавляет проверку пересечения.

            # Transform similarity
            R_diff = float(np.linalg.norm(cl_i['R'] - cl_j['R']))
            mu_norm = float(np.linalg.norm(cl_i['mu']) + np.linalg.norm(cl_j['mu']))
            mu_diff = float(np.linalg.norm(cl_i['mu'] - cl_j['mu']))
            mu_rel = mu_diff / max(mu_norm * 0.5, 1e-6)
            sim = R_diff + mu_rel
            if sim >= transform_sim_threshold: continue

            # Опциональная проверка overlap
            if require_overlap:
                idx_j = set(int(v) for v in cl_j['indices'])
                overlap = len(idx_i & idx_j)
                min_size = max(min(len(idx_i), len(idx_j)), 1)
                if overlap < overlap_threshold * min_size: continue

            union(i, j); n_edges += 1

    # Connected components
    from collections import defaultdict
    comp_map = defaultdict(list)
    for i in range(N_cl):
        comp_map[find(i)].append(i)

    new_cpa = [[] for _ in range(K)]
    n_merged_pairs = 0

    for root, comp in comp_map.items():
        if len(comp) == 1:
            a_i, cl_i = all_cl[comp[0]]
            new_cpa[a_i].append(cl_i)
            continue

        # Merge constituents into one cluster
        n_merged_pairs += 1
        heat_max_per_v = {}
        constituent_anchors = []
        for i in comp:
            a_i, cl_i = all_cl[i]
            constituent_anchors.append(a_i)
            for v, hw in zip(cl_i['indices'], cl_i['heat_weights']):
                v = int(v); hw = float(hw)
                heat_max_per_v[v] = max(heat_max_per_v.get(v, 0.0), hw)

        indices = np.array(sorted(heat_max_per_v.keys()), dtype=np.int64)
        heat_weights = np.array([heat_max_per_v[int(v)] for v in indices])
        cl_verts = verts[indices]
        cl_delta = delta[indices]

        polar = polar_decomposition(heat_weights, cl_delta, cl_verts)
        c_rest = polar['c_rest']
        rms = float(np.sqrt(
            (heat_weights * np.linalg.norm(cl_verts - c_rest, axis=-1) ** 2).sum()
            / max(heat_weights.sum(), 1e-12)))

        # Primary anchor: тот у которого наибольший heat sum в merged зоне
        anchor_scores = {}
        for a in set(constituent_anchors):
            anchor_scores[a] = float(heat_per_anchor[a, indices].sum())
        primary = max(anchor_scores, key=anchor_scores.get)

        new_cluster = {
            'anchor_idx': primary,
            'spatial_sigma': max(rms, 1e-4),
            'indices': indices,
            'heat_weights': heat_weights,
            'merged_from_anchors': sorted(set(constituent_anchors)),
            **polar,
        }
        new_cpa[primary].append(new_cluster)

    if verbose:
        n_total_after = sum(len(cls) for cls in new_cpa)
        print(f"  ── DEDUPLICATE motion-groups ──")
        scope = ("ALL pairs (включая same-anchor)" if not require_overlap
                  else f"pairs с overlap≥{overlap_threshold:.0%}")
        print(f"    Scope: {scope}")
        print(f"    Критерий: transform_sim ||R||+||μ_rel|| < {transform_sim_threshold}")
        print(f"    Найдено {n_edges} duplicate-связей → "
              f"{n_merged_pairs} merge-cluster'ов")
        print(f"    Всего групп: {N_cl} → {n_total_after} "
              f"(-{N_cl - n_total_after} удалено)")
        # Per-anchor breakdown
        for a in range(K):
            n_b = len(clusters_per_anchor[a])
            n_a = len(new_cpa[a])
            n_merged_here = sum(1 for cl in new_cpa[a]
                                 if 'merged_from_anchors' in cl
                                 and len(cl['merged_from_anchors']) > 1)
            mark = "" if n_b == n_a and n_merged_here == 0 else f" (+{n_merged_here} merged)"
            print(f"      anchor {a}: {n_b} → {n_a}{mark}")

    return new_cpa


def auto_pair_anchors(verts_src, anchor_indices_src,
                       verts_tgt, anchor_indices_tgt,
                       verbose=True):
    """Автоматическое сопоставление anchor'ов между двумя мешами через
    Hungarian matching по bbox-нормализованным 3D-позициям.

    Возвращает: (anchor_indices_tgt_reordered, match_info)
        match_info = list of (src_i, tgt_i_original, tgt_i_new, distance)

    Логика:
        1. Bbox-нормализуем оба меша (центр в 0, диагональ = 1)
        2. Для каждой пары (FLAME-anchor, FBX-anchor) считаем 3D-расстояние
        3. Hungarian matching → оптимальная 1-к-1 перестановка
        4. Если порядок изменился — выводим предупреждение
    """
    K = len(anchor_indices_src)
    if len(anchor_indices_tgt) != K:
        if verbose:
            print(f"  ⚠ auto-pair: разное число anchor'ов "
                  f"({K} vs {len(anchor_indices_tgt)}) — пропускаю")
        return list(anchor_indices_tgt), []

    pos_src = np.array([verts_src[int(a)] for a in anchor_indices_src])
    pos_tgt = np.array([verts_tgt[int(a)] for a in anchor_indices_tgt])

    # Bbox-нормализация на основе ПОЛНЫХ мешей (не только anchor'ов)
    def _norm(v_all, points):
        center = v_all.mean(0)
        scale = np.linalg.norm(v_all.max(0) - v_all.min(0)) + 1e-12
        return (points - center) / scale

    pos_src_n = _norm(verts_src, pos_src)
    pos_tgt_n = _norm(verts_tgt, pos_tgt)

    # Cost matrix
    cost = np.linalg.norm(pos_src_n[:, None, :] - pos_tgt_n[None, :, :], axis=-1)

    # Hungarian assignment
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
    except Exception as e:
        if verbose:
            print(f"  ⚠ auto-pair: Hungarian failed ({e}), оставляю исходный порядок")
        return list(anchor_indices_tgt), []

    new_tgt = [int(anchor_indices_tgt[col_ind[i]]) for i in range(K)]
    match_info = [(int(i), int(anchor_indices_tgt[i]), int(new_tgt[i]),
                   float(cost[i, col_ind[i]])) for i in range(K)]

    # Diagnostic
    changed = sum(1 for i in range(K) if col_ind[i] != i)
    if verbose:
        if changed == 0:
            print(f"  ✓ auto-pair: anchor-порядок UYZE верный, нет перестановок")
        else:
            print(f"  ⚠ auto-pair: переупорядочил {changed} из {K} anchor'ов FBX:")
            for i in range(K):
                if col_ind[i] != i:
                    print(f"      src anchor {i} ↔ tgt anchor {col_ind[i]} "
                          f"(было tgt anchor {i}, dist={cost[i, col_ind[i]]:.4f})")
                else:
                    print(f"      src anchor {i} ↔ tgt anchor {i} "
                          f"(совпало, dist={cost[i, col_ind[i]]:.4f})")
        mean_d = float(cost[np.arange(K), col_ind].mean())
        max_d = float(cost[np.arange(K), col_ind].max())
        print(f"    mean pair distance (bbox-norm): {mean_d:.4f}, "
              f"max: {max_d:.4f}")
        if max_d > 0.3:
            print(f"    ⚠ max pair distance > 0.3 — некоторые anchor'ы "
                  f"анатомически не совпадают даже после reorder. Возможно "
                  f"меши сильно разной анатомии.")

    return new_tgt, match_info


def enrich_heat_multi_t(verts, faces, anchor_indices,
                         n_times=8, n_eigs=80,
                         smooth_iters=5, smooth_alpha=0.5,
                         mesh_label="MESH"):
    """Универсальный multi-t enrichment с Laplacian smoothing.

    1. Eigendecomp k_eigs мод
    2. compute_anchor_heat_multi_t → (K*T, N)
    3. Per-row max-norm + reshape → (K, T, N) → L2-aggregate over T → (K, N)
    4. Laplacian smoothing (если smooth_iters > 0)

    Returns: (heat_enriched_smoothed (K,N), times_used (T,))
    """
    K = len(anchor_indices)
    N = len(verts)

    ev, ef = compute_spectrum(verts, faces, n_eigs=n_eigs)
    H_multi, times_used = compute_anchor_heat_multi_t(
        ev, ef, anchor_indices, n_times=n_times)

    # Per-row max-norm и L2-aggregate
    H_multi_n = H_multi / H_multi.max(axis=1, keepdims=True).clip(min=1e-12)
    H_3d = H_multi_n.reshape(K, n_times, -1)
    heat_enriched = np.sqrt((H_3d ** 2).mean(axis=1))                # (K, N)

    # Laplacian smoothing — границы зон становятся плавнее
    if smooth_iters > 0:
        W = build_neighbor_avg_matrix(N, faces)
        H_T = heat_enriched.T.copy()
        for _ in range(smooth_iters):
            H_T = (1 - smooth_alpha) * H_T + smooth_alpha * (W @ H_T)
        heat_enriched = H_T.T

    print(f"    [{mesh_label}] multi-t enrichment: K={K}, T={n_times}, "
          f"k_eigs={n_eigs}, smooth_iters={smooth_iters}, "
          f"times ∈ [{times_used[0]:.3g}, {times_used[-1]:.3g}]")
    return heat_enriched, times_used


def compute_hks(eigvals, eigvecs, t_values, scale_invariant=True):
    """Heat Kernel Signature: HKS(v,t) = Σ_k exp(-t·λ_k)·φ_k(v)². Returns (N, T).

    scale_invariant=True (default): делим на heat_trace(t) = Σ_k exp(-t·λ_k).
    Это делает HKS сопоставимым между мешами разного размера/плотности
    (Bronstein & Kokkinos 2010). КРИТИЧНО для cross-mesh матчинга!
    """
    decay = np.exp(-np.outer(t_values, eigvals))      # (T, K)
    hks = (eigvecs ** 2) @ decay.T                     # (N, T)
    if scale_invariant:
        heat_trace = decay.sum(axis=1).clip(min=1e-12) # (T,) — Σ_k exp(-t·λ_k)
        hks = hks / heat_trace[None, :]
    return hks


def compute_wks(eigvals, eigvecs, energies, sigma):
    """Wave Kernel Signature: WKS(v,e) = (1/C(e)) Σ_k exp(-(log λ_k - e)² / (2σ²)) φ_k(v)².
    Returns (N, E).
    """
    log_lam = np.log(eigvals.clip(min=1e-9))           # (K,)
    weights = np.exp(-((log_lam[None, :] - energies[:, None]) ** 2)
                      / (2.0 * sigma * sigma))         # (E, K)
    C = weights.sum(axis=1).clip(min=1e-12)
    return ((eigvecs ** 2) @ weights.T) / C[None, :]   # (N, E)


def default_hks_times(eigvals, n_scales=16):
    """Log-spaced t от t_min до t_max (Sun et al. 2009)."""
    eps = 1e-6
    lam_min = max(eigvals[1] if len(eigvals) > 1 else eps, eps)
    lam_max = max(eigvals[-1], lam_min * 10)
    t_min = 4 * np.log(10) / lam_max
    t_max = 4 * np.log(10) / lam_min
    return np.logspace(np.log10(t_min), np.log10(t_max), n_scales)


def default_wks_energies(eigvals, n_scales=16):
    """Log-spaced энергии для WKS (Aubry et al. 2011). Returns (energies, sigma)."""
    eps = 1e-6
    log_lam_min = np.log(max(eigvals[1] if len(eigvals) > 1 else eps, eps))
    log_lam_max = np.log(max(eigvals[-1], eps * 10))
    energies = np.linspace(log_lam_min, log_lam_max, n_scales)
    sigma = 7 * (log_lam_max - log_lam_min) / n_scales
    return energies, sigma


def cluster_signature_profile(source_cluster, sig_per_vertex):
    """Heat-weighted average descriptor по вершинам кластера. Returns (D,)."""
    idx = source_cluster['indices']
    w = source_cluster['heat_weights']
    W = max(w.sum(), 1e-12)
    return (w[:, None] * sig_per_vertex[idx]).sum(0) / W


def normalize_signature(sig):
    """L2-нормализация per-vertex (для cosine-like сравнения)."""
    norms = np.linalg.norm(sig, axis=-1, keepdims=True).clip(min=1e-12)
    return sig / norms


# ── Реконструкция δ из кластеров ─────────────────────────────────────────────

def build_vertex_adjacency(N, verts, faces):
    """Edges с длинами для Dijkstra по поверхности."""
    neighbors = [[] for _ in range(N)]
    seen = set()
    for f in faces:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(f[i]), int(f[j])
                key = (min(a, b), max(a, b))
                if key in seen: continue
                seen.add(key)
                d = float(np.linalg.norm(verts[a] - verts[b]))
                neighbors[a].append((b, d))
                neighbors[b].append((a, d))
    return neighbors


def geodesic_dijkstra(neighbors, src, max_dist):
    import heapq
    N = len(neighbors)
    dist = np.full(N, np.inf)
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > max_dist: continue
        if d > dist[u]: continue
        for v, w in neighbors[u]:
            nd = d + w
            if nd < dist[v] and nd <= max_dist:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def find_nearest_vertex(verts, point):
    return int(np.argmin(np.linalg.norm(verts - point, axis=-1)))


def compute_source_cluster_geo_radius(neighbors_source, src_cluster):
    """Геодезический "радиус" source-кластера = max geo-distance от seed-вершины
    (вершина с max heat_weight) до любой другой вершины кластера.

    Возвращает (seed_vertex_global, radius).
    """
    indices = np.asarray(src_cluster['indices'])
    weights = np.asarray(src_cluster['heat_weights'])
    if len(indices) == 0:
        return None, 0.0
    seed_idx_local = int(np.argmax(weights))
    seed = int(indices[seed_idx_local])
    # Dijkstra с max_dist=inf чтобы достать всех в кластере
    dist = geodesic_dijkstra(neighbors_source, seed, max_dist=np.inf)
    # max расстояние до членов кластера (игнорируя недостижимые)
    cluster_dists = dist[indices]
    finite = cluster_dists[np.isfinite(cluster_dists)]
    if len(finite) == 0:
        return seed, 0.0
    return seed, float(finite.max())


def filter_target_clusters_by_geodesic_radius(
        target_clusters, verts_target, faces_target,
        neighbors_source, tolerance_factor=1.2):
    """Sanity filter: для каждого target-кластера измеряем геодезическое
    расстояние от своего "seed"-вертекса (FBX вертекс с max target_heat)
    до остальных target-вершин. Если расстояние превышает
    source_radius * tolerance_factor — выпиливаем эту target-вершину.

    Source radius — геодезический radius соответствующего source-кластера
    на FLAME (max geo-distance внутри кластера).

    Returns: (filtered_target_clusters, n_removed)
    """
    if not target_clusters:
        return target_clusters, 0

    # Кешируем source radius (один на source-кластер — не пересчитываем
    # если несколько target указывают на один source)
    src_radius_cache = {}                          # id(src_cluster) → radius

    # Соседи на FBX (для Dijkstra)
    N_t = len(verts_target)
    neighbors_target = build_vertex_adjacency(N_t, verts_target, faces_target)

    n_removed_total = 0
    filtered = []
    for tc in target_clusters:
        src = tc['source']
        if id(src) not in src_radius_cache:
            _, r = compute_source_cluster_geo_radius(neighbors_source, src)
            src_radius_cache[id(src)] = r
        src_radius = src_radius_cache[id(src)]
        if src_radius <= 0:
            filtered.append(tc); continue

        max_allowed = src_radius * tolerance_factor

        t_indices = np.asarray(tc['target_indices'])
        t_heat    = np.asarray(tc['target_heat'])
        if len(t_indices) == 0:
            continue

        # Seed на FBX = вершина с max target_heat
        seed_local = int(np.argmax(t_heat))
        seed_global = int(t_indices[seed_local])

        # Dijkstra от seed на FBX, ограниченный max_allowed (отсекаем дальше)
        dist_t = geodesic_dijkstra(neighbors_target, seed_global,
                                     max_dist=max_allowed * 1.1)
        within_mask = np.isfinite(dist_t[t_indices]) & (dist_t[t_indices] <= max_allowed)

        n_kept = int(within_mask.sum())
        n_removed = len(t_indices) - n_kept
        n_removed_total += n_removed

        if n_kept == 0:
            # весь кластер выпал
            continue

        kept_indices = t_indices[within_mask]
        kept_heat    = t_heat[within_mask]
        W = max(kept_heat.sum(), 1e-12)
        c_target = (kept_heat[:, None] * verts_target[kept_indices]).sum(0) / W

        # Копируем tc с обновлёнными полями
        new_tc = dict(tc)
        new_tc['target_indices'] = kept_indices
        new_tc['target_heat']    = kept_heat
        new_tc['c_target']       = c_target
        new_tc['geo_radius_src'] = float(src_radius)
        new_tc['geo_radius_max'] = float(max_allowed)
        new_tc['n_removed']      = int(n_removed)
        filtered.append(new_tc)

    return filtered, n_removed_total


def compute_centroid_diff_diagnostic(target_clusters, anchor_pos_source,
                                       anchor_pos_target, verts_source, verts_target):
    """DIAGNOSTIC: для каждого target-кластера сравниваем относительное положение
    его centroid'а внутри anchor-зоны с тем же на source.

    Per cluster:
      offset_src = (c_src - anchor_src) / src_radius
      offset_tgt = (c_tgt - anchor_tgt) / tgt_radius
      diff_norm = ||offset_src - offset_tgt||  (в нормализованных единицах)

    Чем больше diff_norm — тем сильнее target-кластер «уехал» относительно
    своего anchor'а по сравнению с тем как было на source.

    Returns:
      per_vertex_diff: (N_target,) float — для каждой target-вершины присвоено
                       diff_norm её кластера; NaN если вершина не в кластере
      cluster_stats:   list of dicts с per-cluster метриками
    """
    N_t = len(verts_target)
    per_vertex_diff = np.full(N_t, np.nan)
    cluster_stats = []

    for tc in target_clusters:
        src = tc['source']
        a_idx = src['anchor_idx']
        anchor_src = anchor_pos_source[a_idx]
        anchor_tgt = anchor_pos_target[a_idx]

        # Source radius: max расстояние от anchor'а до любой вершины source-кластера
        src_idx = np.asarray(src['indices'])
        src_radius = float(np.linalg.norm(
            verts_source[src_idx] - anchor_src, axis=1).max())
        src_radius = max(src_radius, 1e-6)

        # Target radius: max расстояние от anchor'а до любой target-вершины
        tgt_idx = np.asarray(tc['target_indices'])
        if len(tgt_idx) == 0: continue
        tgt_radius = float(np.linalg.norm(
            verts_target[tgt_idx] - anchor_tgt, axis=1).max())
        tgt_radius = max(tgt_radius, 1e-6)

        # Normalized offsets
        c_src = np.asarray(src['c_rest'])
        c_tgt = np.asarray(tc['c_target'])
        offset_src = (c_src - anchor_src) / src_radius
        offset_tgt = (c_tgt - anchor_tgt) / tgt_radius
        diff_vec = offset_src - offset_tgt
        diff_norm = float(np.linalg.norm(diff_vec))

        # Запоминаем в стате
        cluster_stats.append({
            'source': src,
            'anchor_idx': a_idx,
            'src_radius': src_radius,
            'tgt_radius': tgt_radius,
            'offset_src': offset_src.tolist(),
            'offset_tgt': offset_tgt.tolist(),
            'diff_norm': diff_norm,
            'diff_pct': diff_norm * 100,
        })

        # Также записываем в per-vertex
        for v in tgt_idx:
            per_vertex_diff[int(v)] = diff_norm

    return per_vertex_diff, cluster_stats


# ── Mesh-graph label smoothing helpers (для heat_zone_xyz / zonal_1d / etc) ─

def _build_vertex_adjacency(N, faces):
    """1-ring adjacency list для меша. Возвращает list[set[int]]."""
    adj = [set() for _ in range(N)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    return adj


def _smooth_labels_on_mesh(labels, adj, n_iter=2):
    """Majority-vote по 1-ring соседям на mesh-графе. labels: dict {vert: cluster_id}.
    Возвращает обновлённый dict (только для вершин которые есть в labels)."""
    if n_iter <= 0: return labels
    cur = dict(labels)
    for _ in range(n_iter):
        new = {}
        for v, lab in cur.items():
            votes = {}
            votes[lab] = votes.get(lab, 0) + 2          # своё мнение чуть весомее
            for n in adj[v]:
                if n in cur:
                    votes[cur[n]] = votes.get(cur[n], 0) + 1
            new[v] = max(votes.items(), key=lambda kv: kv[1])[0]
        cur = new
    return cur


def assign_target_to_source_by_heat_zone(
        verts_target, faces_target,
        verts_source,
        heat_target_per_anchor, heat_source_per_anchor,
        src_clusters_list,
        heat_threshold=0.05,
        rigid_align=True, n_icp_iters=3,
        label_smooth_iters=2,
        collect_alignment_data=False,
        faces_source=None,
        alignment_mode='scale',         # 'centroid' | 'scale' | 'non_rigid'
        non_rigid_iters=2,
        non_rigid_smoothing=0.01,
        anchor_verts_source=None,        # list of K vertex indices (FLAME anchors)
        anchor_verts_target=None,        # list of K vertex indices (FBX anchors)
        use_anchor_align=True,           # alignment по anchor'у вместо centroid'а
        use_rotation=False,              # Procrustes rotation на pre-step
        hard_partition_zones=False):     # argmax-partition вместо threshold (нет перекрытий)
    """A4. HEAT-ZONE XYZ MATCHING — point-cloud alignment per anchor zone.

    НАПРАВЛЕНИЕ: подгоняем TARGET (FBX) → SOURCE (FLAME).
    SRC остаётся в своих координатах, TGT деформируется чтобы лечь поверх SRC.

    Концепция:
        1. Для каждой anchor-зоны берём облако точек на source (с уже известными
           cluster-labels) и облако на target (без labels).
        2. Центрируем оба облака в своих centroid'ах.
        3. (Опц.) Масштабируем TARGET под размер SOURCE (scale alignment).
        4. (Опц.) NON-RIGID: TPS-RBF деформация TARGET → SOURCE по NN-парам.
        5. ПОВОРОТЫ НЕ ПРИМЕНЯЮТСЯ — только translation + scale + (опц.) non-rigid.
        6. Для каждой target-вершины в зоне ищем ближайшую source-вершину
           по 3D-расстоянию в выровненном пространстве → наследуем cluster.
        7. Group + (опц.) label smoothing на mesh-графе.

    Параметр n_icp_iters игнорируется (оставлен для backwards compat).

    Отличие от heat_align:
        heat_align     — matching по K-мерному heat-vector (intrinsic, медленнее)
        heat_zone_xyz  — matching по 3D xyz после translation + scale (+ non_rigid)
                         (extrinsic, быстрее, лучше когда анатомия похожая)
    """
    K, N_s = heat_source_per_anchor.shape
    _, N_t = heat_target_per_anchor.shape

    # Reverse lookup: vertex_id → cluster_obj (только для clustered source-vertex)
    vertex_to_cluster = {}
    for s in src_clusters_list:
        for v_idx in s['indices']:
            vertex_to_cluster[int(v_idx)] = s

    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    fbx_to_cluster = {}
    n_total = 0
    alignment_data = []     # для опц. визуализации

    # Hard-partition: одна метка argmax на вершину (без перекрытия между зонами)
    if hard_partition_zones:
        tgt_partition = _argmax_partition(heat_target_per_anchor,
                                            threshold=heat_threshold)
        src_partition = _argmax_partition(heat_source_per_anchor,
                                            threshold=heat_threshold)

        # DIAGNOSTIC: размеры зон per anchor для обоих мешей
        N_src = heat_source_per_anchor.shape[1]
        N_tgt = heat_target_per_anchor.shape[1]
        print(f"    [hard_partition] Размеры зон по анкерам:")
        print(f"      HEAD 1 (N={N_src}):")
        for a in range(heat_source_per_anchor.shape[0]):
            c = int((src_partition == a).sum())
            print(f"        anchor {a}: {c} verts ({100*c/max(N_src,1):.1f}%)")
        unass_s = int((src_partition == -1).sum())
        print(f"        unassigned: {unass_s} ({100*unass_s/max(N_src,1):.1f}%)")
        print(f"      FBX (N={N_tgt}):")
        for a in range(heat_target_per_anchor.shape[0]):
            c = int((tgt_partition == a).sum())
            print(f"        anchor {a}: {c} verts ({100*c/max(N_tgt,1):.1f}%)")
        unass_t = int((tgt_partition == -1).sum())
        print(f"        unassigned: {unass_t} ({100*unass_t/max(N_tgt,1):.1f}%)")
        # Warning if biased
        tgt_counts = np.array([(tgt_partition == a).sum()
                                for a in range(heat_target_per_anchor.shape[0])])
        if tgt_counts.max() > 0.6 * N_tgt:
            print(f"      ⚠ FBX argmax-partition BIASED: anchor "
                  f"{int(np.argmax(tgt_counts))} захватил "
                  f"{100*tgt_counts.max()/N_tgt:.0f}% вершин! "
                  f"Heat-зоны на FBX неравномерные. Возможные причины: "
                  f"плохо расставлены anchor'ы (близко к одной точке) или "
                  f"топология FBX неравномерна (один anchor 'центральнее' других).")

    for a, src_list in src_by_anchor.items():
        # Source zone: вершины принадлежащие кластерам этого anchor'а
        src_vert_set = set()
        for s in src_list:
            src_vert_set.update(int(v) for v in s['indices'])
        src_zone_idx_clusters = np.array(sorted(src_vert_set), dtype=np.int64)
        if len(src_zone_idx_clusters) < 3: continue

        if hard_partition_zones:
            # Cross-reference: только те source-вершины кластеров anchor a которые
            # ТАКЖЕ попадают в hard-partition зону этого anchor (argmax == a)
            src_argmax_zone = (src_partition == a)
            mask = src_argmax_zone[src_zone_idx_clusters]
            src_zone_idx = src_zone_idx_clusters[mask]
            if len(src_zone_idx) < 3:
                src_zone_idx = src_zone_idx_clusters   # fallback
            # Target: argmax-partition (без overlap с другими anchor'ами)
            tgt_zone_idx = np.where(tgt_partition == a)[0]
        else:
            src_zone_idx = src_zone_idx_clusters
            # Target: heat > threshold (может перекрываться между anchor'ами)
            h_t = heat_target_per_anchor[a]
            h_t_max = max(h_t.max(), 1e-12)
            tgt_zone_idx = np.where(h_t > heat_threshold * h_t_max)[0]
        if len(tgt_zone_idx) < 1: continue

        # 3D облака точек
        P_src = verts_source[src_zone_idx].astype(np.float64)
        P_tgt = verts_target[tgt_zone_idx].astype(np.float64)

        # ── Anchor-based alignment: выравниваем по самой anchor-точке ────
        # Если есть точка anchor'а — она становится origin'ом обеих зон.
        # Иначе fallback на centroid.
        anchor_pos_src = None
        anchor_pos_tgt = None
        if (use_anchor_align and anchor_verts_source is not None
                and anchor_verts_target is not None
                and a < len(anchor_verts_source)):
            anchor_pos_src = verts_source[int(anchor_verts_source[a])]
            anchor_pos_tgt = verts_target[int(anchor_verts_target[a])]
            origin_src = anchor_pos_src
            origin_tgt = anchor_pos_tgt
        else:
            origin_src = P_src.mean(0)
            origin_tgt = P_tgt.mean(0)

        Q_src = P_src - origin_src                # SRC координаты от anchor/центра
        Q_tgt = P_tgt - origin_tgt                # TGT координаты от anchor/центра

        # Scale alignment: масштабируем TGT под размер SRC относительно anchor
        if alignment_mode in ('scale', 'non_rigid') and rigid_align:
            scale_src = np.linalg.norm(Q_src, axis=1).mean() + 1e-9
            scale_tgt = np.linalg.norm(Q_tgt, axis=1).mean() + 1e-9
            Q_tgt = Q_tgt * (scale_src / scale_tgt)

        # Procrustes rotation (опц.): для каждой Q_tgt[i] NN на Q_src[j],
        # потом SVD-Procrustes на парах → оптимальная ротация Q_tgt
        if use_rotation and alignment_mode != 'centroid':
            d_sq = ((Q_tgt[:, None, :] - Q_src[None, :, :]) ** 2).sum(-1)
            nn = np.argmin(d_sq, axis=1)
            Y = Q_src[nn]
            H = Q_tgt.T @ Y                                          # (3, 3)
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1; R = Vt.T @ U.T
            Q_tgt = Q_tgt @ R.T

        # NON-RIGID: TPS/RBF деформация TGT → SRC c anchor-pin
        # Anchor вершина TGT (она в origin = 0 после центрирования) должна
        # ОСТАТЬСЯ в 0 → добавляем (0, 0) как control point с zero smoothing
        if alignment_mode == 'non_rigid' and non_rigid_iters > 0:
            try:
                from scipy.interpolate import RBFInterpolator
                for it in range(non_rigid_iters):
                    # NN tgt → src в текущем выровненном пространстве
                    d_sq = ((Q_tgt[:, None, :] - Q_src[None, :, :]) ** 2).sum(-1)
                    nn = np.argmin(d_sq, axis=1)
                    Y = Q_src[nn]                                    # цели для tgt
                    # Подвыборка control points если зона большая (для скорости)
                    n_pts = len(Q_tgt)
                    if n_pts > 400:
                        sub = np.random.RandomState(42 + it).choice(
                            n_pts, 400, replace=False)
                        ctrl_X = Q_tgt[sub]; ctrl_Y = Y[sub]
                    else:
                        ctrl_X = Q_tgt.copy(); ctrl_Y = Y.copy()
                    # ANCHOR-PIN: добавляем дополнительные copies pин-точки
                    # вокруг origin'а чтобы заставить RBF фиксировать anchor.
                    # 10 копий (anchor=(0,0,0) ↔ anchor=(0,0,0)) с weight'ом
                    # доминируют локально вблизи origin'а.
                    if use_anchor_align and anchor_pos_src is not None:
                        pin_x = np.zeros((10, 3))
                        pin_y = np.zeros((10, 3))
                        ctrl_X = np.vstack([ctrl_X, pin_x])
                        ctrl_Y = np.vstack([ctrl_Y, pin_y])
                    # TPS RBF interpolant: tgt → src
                    rbf = RBFInterpolator(ctrl_X, ctrl_Y,
                                           kernel='thin_plate_spline',
                                           smoothing=non_rigid_smoothing)
                    Q_tgt = rbf(Q_tgt)
            except Exception as e:
                print(f"    ⚠ NON-RIGID не удался для anchor {a}: {e}, "
                      f"fallback на scale only")

        # Финальный NN: tgt → src в выровненном пространстве
        # (Q_tgt уже подогнан под Q_src; SRC остался в своих координатах)
        d_sq = ((Q_tgt[:, None, :] - Q_src[None, :, :]) ** 2).sum(-1)
        nn_tgt2src = np.argmin(d_sq, axis=1)
        nn_src_global = src_zone_idx[nn_tgt2src]

        for t_v, s_v in zip(tgt_zone_idx, nn_src_global):
            cl_obj = vertex_to_cluster.get(int(s_v))
            if cl_obj is not None:
                fbx_to_cluster[int(t_v)] = cl_obj
                n_total += 1

        if collect_alignment_data:
            # Submesh — только faces где все 3 вершины принадлежат зоне
            src_faces_local = None
            tgt_faces_local = None
            if faces_source is not None:
                mask_v_src = np.zeros(len(verts_source), dtype=bool)
                mask_v_src[src_zone_idx] = True
                src_face_mask = mask_v_src[faces_source].all(axis=1)
                remap_s = -np.ones(len(verts_source), dtype=np.int64)
                remap_s[src_zone_idx] = np.arange(len(src_zone_idx))
                src_faces_local = remap_s[faces_source[src_face_mask]]
            if faces_target is not None:
                mask_v_tgt = np.zeros(len(verts_target), dtype=bool)
                mask_v_tgt[tgt_zone_idx] = True
                tgt_face_mask = mask_v_tgt[faces_target].all(axis=1)
                remap_t = -np.ones(len(verts_target), dtype=np.int64)
                remap_t[tgt_zone_idx] = np.arange(len(tgt_zone_idx))
                tgt_faces_local = remap_t[faces_target[tgt_face_mask]]

            alignment_data.append({
                'anchor_idx': a,
                'src_zone_idx': src_zone_idx,
                'tgt_zone_idx': tgt_zone_idx,
                'P_src_orig':    P_src,                 # в world coords
                'P_tgt_orig':    P_tgt,                 # в world coords
                # ПЕРЕИМЕНОВАНО: src остаётся "просто центрирован",
                # tgt — "подогнан под src" (центрирован + scale + non-rigid)
                'P_src_aligned': Q_src,                 # src просто центрирован
                'P_tgt_centered': Q_tgt,                # tgt подогнан под src
                'src_faces_local': src_faces_local,     # reindexed faces для submesh
                'tgt_faces_local': tgt_faces_local,
                'src_cluster_objs': [vertex_to_cluster.get(int(v))
                                     for v in src_zone_idx],
            })

    # Post-smoothing labels на mesh-графе FBX (как в heat_align)
    if label_smooth_iters > 0 and faces_target is not None and fbx_to_cluster:
        id_to_cl = {id(cl): cl for cl in fbx_to_cluster.values()}
        labels_id = {v: id(cl) for v, cl in fbx_to_cluster.items()}
        adj = _build_vertex_adjacency(len(verts_target), faces_target)
        labels_id = _smooth_labels_on_mesh(labels_id, adj, n_iter=label_smooth_iters)
        fbx_to_cluster = {v: id_to_cl[lid] for v, lid in labels_id.items()}

    # Группировка
    cluster_to_targets = {}
    for t_v, cl_obj in fbx_to_cluster.items():
        cluster_to_targets.setdefault(id(cl_obj), (cl_obj, [])).__getitem__(1).append(t_v)

    target_clusters = []
    for _, (cl_obj, t_list) in cluster_to_targets.items():
        if not t_list: continue
        t_indices = np.array(sorted(set(t_list)), dtype=np.int64)
        a = cl_obj['anchor_idx']
        t_heat = heat_target_per_anchor[a][t_indices]
        W = max(t_heat.sum(), 1e-12)
        c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
        target_clusters.append({
            'source': cl_obj,
            'target_indices': t_indices,
            'target_heat': t_heat,
            'c_target': c_target,
        })

    print(f"    [heat_zone_xyz] {n_total} per-vertex NN matches "
          f"(mode={alignment_mode}, scale={rigid_align}) → "
          f"{len(target_clusters)} target clusters")
    if collect_alignment_data:
        return target_clusters, alignment_data
    return target_clusters


# ═══════════════════════════════════════════════════════════════════════════
# v5 EXPERIMENTAL VARIANTS B / C / D
# ═══════════════════════════════════════════════════════════════════════════

def _argmax_partition(heat_per_anchor, threshold=0.05):
    """Hard-partition вершин по argmax. Возвращает labels (N,) ∈ [0..K-1] для
    активных вершин, -1 для пассивных (heat ниже threshold у всех anchor'ов).
    """
    K, N = heat_per_anchor.shape
    H = heat_per_anchor / heat_per_anchor.max(axis=1, keepdims=True).clip(min=1e-12)
    active = H.max(axis=0) > threshold
    dom = np.argmax(H, axis=0)
    labels = np.where(active, dom, -1)
    return labels


def assign_target_to_source_zonal_1d(
        verts_target, faces_target, verts_source, faces_source,
        heat_target_per_anchor, heat_source_per_anchor,
        src_clusters_list,
        heat_threshold=0.05,
        label_smooth_iters=2,
        alignment_mode='scale',
        non_rigid_iters=2, non_rigid_smoothing=0.01,
        anchor_verts_source=None, anchor_verts_target=None,
        use_anchor_align=True, use_rotation=False,
        collect_alignment_data=False):
    """ВАРИАНТ B: Hard zone partition (argmax) + XYZ matching внутри зон.

    Каждая вершина приписывается СТРОГО к одному anchor'у (тому где её heat
    максимален). Это полностью устраняет перекрытие зон. Дальше внутри каждой
    непересекающейся зоны делаем 3D xyz-matching (как в heat_zone_xyz).

    Идея: heat-fingerprint становится тривиально 1-мерным (есть только одно
    значение для зоны), и matching не страдает от шумных смесей.
    """
    # Hard partition по argmax (одинаково для src и tgt)
    src_part = _argmax_partition(heat_source_per_anchor, threshold=heat_threshold)
    tgt_part = _argmax_partition(heat_target_per_anchor, threshold=heat_threshold)

    # Reverse lookup vertex → cluster (только clustered source-vertex)
    vertex_to_cluster = {}
    for s in src_clusters_list:
        for v_idx in s['indices']:
            vertex_to_cluster[int(v_idx)] = s

    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    fbx_to_cluster = {}
    n_total = 0
    alignment_data = []

    for a in src_by_anchor.keys():
        # Hard зоны по argmax (пересечений нет по построению)
        src_zone_idx = np.where(src_part == a)[0]
        tgt_zone_idx = np.where(tgt_part == a)[0]
        # Только source-вершины принадлежащие кластерам этого anchor'а
        src_zone_idx = np.array([v for v in src_zone_idx if int(v) in vertex_to_cluster])
        if len(src_zone_idx) < 3 or len(tgt_zone_idx) < 1:
            continue

        P_src = verts_source[src_zone_idx].astype(np.float64)
        P_tgt = verts_target[tgt_zone_idx].astype(np.float64)

        # Anchor-based alignment (как в heat_zone_xyz)
        anchor_pos_src = anchor_pos_tgt = None
        if (use_anchor_align and anchor_verts_source is not None
                and anchor_verts_target is not None
                and a < len(anchor_verts_source)):
            anchor_pos_src = verts_source[int(anchor_verts_source[a])]
            anchor_pos_tgt = verts_target[int(anchor_verts_target[a])]
            origin_src = anchor_pos_src; origin_tgt = anchor_pos_tgt
        else:
            origin_src = P_src.mean(0); origin_tgt = P_tgt.mean(0)
        Q_src = P_src - origin_src; Q_tgt = P_tgt - origin_tgt

        # Scale alignment
        if alignment_mode in ('scale', 'non_rigid'):
            ssrc = np.linalg.norm(Q_src, axis=1).mean() + 1e-9
            stgt = np.linalg.norm(Q_tgt, axis=1).mean() + 1e-9
            Q_tgt = Q_tgt * (ssrc / stgt)

        # Optional rotation
        if use_rotation and alignment_mode != 'centroid':
            d_sq = ((Q_tgt[:, None, :] - Q_src[None, :, :]) ** 2).sum(-1)
            nn = np.argmin(d_sq, axis=1); Y = Q_src[nn]
            H_ = Q_tgt.T @ Y
            U, _, Vt = np.linalg.svd(H_)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0: Vt[-1, :] *= -1; R = Vt.T @ U.T
            Q_tgt = Q_tgt @ R.T

        # Non-rigid RBF deformation tgt → src (с anchor-pin)
        if alignment_mode == 'non_rigid' and non_rigid_iters > 0:
            try:
                from scipy.interpolate import RBFInterpolator
                for it in range(non_rigid_iters):
                    d_sq = ((Q_tgt[:, None, :] - Q_src[None, :, :]) ** 2).sum(-1)
                    nn = np.argmin(d_sq, axis=1); Y = Q_src[nn]
                    if len(Q_tgt) > 400:
                        sub = np.random.RandomState(42 + it).choice(
                            len(Q_tgt), 400, replace=False)
                        cX, cY = Q_tgt[sub], Y[sub]
                    else:
                        cX, cY = Q_tgt.copy(), Y.copy()
                    if use_anchor_align and anchor_pos_src is not None:
                        pin = np.zeros((10, 3))
                        cX = np.vstack([cX, pin]); cY = np.vstack([cY, pin])
                    rbf = RBFInterpolator(cX, cY, kernel='thin_plate_spline',
                                          smoothing=non_rigid_smoothing)
                    Q_tgt = rbf(Q_tgt)
            except Exception as e:
                print(f"    ⚠ RBF не удался для anchor {a}: {e}")

        # NN: tgt → src в выровненном пространстве
        d_sq = ((Q_tgt[:, None, :] - Q_src[None, :, :]) ** 2).sum(-1)
        nn_tgt2src = np.argmin(d_sq, axis=1)
        nn_src_global = src_zone_idx[nn_tgt2src]

        for t_v, s_v in zip(tgt_zone_idx, nn_src_global):
            cl_obj = vertex_to_cluster.get(int(s_v))
            if cl_obj is not None:
                fbx_to_cluster[int(t_v)] = cl_obj
                n_total += 1

        if collect_alignment_data:
            src_faces_local = tgt_faces_local = None
            if faces_source is not None:
                mvs = np.zeros(len(verts_source), bool); mvs[src_zone_idx] = True
                rm = -np.ones(len(verts_source), np.int64)
                rm[src_zone_idx] = np.arange(len(src_zone_idx))
                src_faces_local = rm[faces_source[mvs[faces_source].all(1)]]
            if faces_target is not None:
                mvt = np.zeros(len(verts_target), bool); mvt[tgt_zone_idx] = True
                rm = -np.ones(len(verts_target), np.int64)
                rm[tgt_zone_idx] = np.arange(len(tgt_zone_idx))
                tgt_faces_local = rm[faces_target[mvt[faces_target].all(1)]]
            alignment_data.append({
                'anchor_idx': a,
                'src_zone_idx': src_zone_idx, 'tgt_zone_idx': tgt_zone_idx,
                'P_src_aligned': Q_src, 'P_tgt_centered': Q_tgt,
                'src_faces_local': src_faces_local,
                'tgt_faces_local': tgt_faces_local,
                'src_cluster_objs': [vertex_to_cluster.get(int(v))
                                     for v in src_zone_idx],
            })

    # Label smoothing
    if label_smooth_iters > 0 and faces_target is not None and fbx_to_cluster:
        id_to_cl = {id(cl): cl for cl in fbx_to_cluster.values()}
        labs = {v: id(cl) for v, cl in fbx_to_cluster.items()}
        adj = _build_vertex_adjacency(len(verts_target), faces_target)
        labs = _smooth_labels_on_mesh(labs, adj, n_iter=label_smooth_iters)
        fbx_to_cluster = {v: id_to_cl[l] for v, l in labs.items()}

    # Group
    g = {}
    for t_v, cl in fbx_to_cluster.items():
        g.setdefault(id(cl), (cl, [])).__getitem__(1).append(t_v)
    target_clusters = []
    for _, (cl, lst) in g.items():
        if not lst: continue
        t_idx = np.array(sorted(set(lst)), dtype=np.int64)
        a = cl['anchor_idx']
        th = heat_target_per_anchor[a][t_idx]
        W = max(th.sum(), 1e-12)
        c_t = (th[:, None] * verts_target[t_idx]).sum(0) / W
        target_clusters.append({
            'source': cl, 'target_indices': t_idx,
            'target_heat': th, 'c_target': c_t,
        })
    n_src_zone = (src_part >= 0).sum()
    n_tgt_zone = (tgt_part >= 0).sum()
    print(f"    [zonal_1d] hard-partition: src_active={n_src_zone}, "
          f"tgt_active={n_tgt_zone}, matched={n_total} → "
          f"{len(target_clusters)} clusters")
    if collect_alignment_data:
        return target_clusters, alignment_data
    return target_clusters


def assign_target_to_source_sequential_anchor(
        verts_target, faces_target, verts_source, faces_source,
        heat_target_per_anchor, heat_source_per_anchor,
        src_clusters_list,
        heat_threshold=0.05,
        label_smooth_iters=2,
        alignment_mode='scale',
        non_rigid_iters=2, non_rigid_smoothing=0.01,
        anchor_verts_source=None, anchor_verts_target=None,
        use_anchor_align=True, use_rotation=False,
        anchor_order='by_max_heat'):
    """ВАРИАНТ C: SEQUENTIAL anchor processing.

    Обрабатываем anchor'ы по очереди (sorted by heat max или просто по index'у).
    На каждом шаге:
      - active = вершины с heat > threshold для текущего anchor'а И
        ещё не назначенные предыдущим anchor'ом
      - matching как в heat_zone_xyz (анkor-align + scale + opt non-rigid)
      - вершины «помечаются как занятые» — больше не рассматриваются

    Идея: первый anchor берёт всё что ему «принадлежит» (его горячую зону),
    оставшиеся горячие точки достанутся следующим. Перекрытие учитывается
    «приоритетом» порядка.
    """
    K, N_s = heat_source_per_anchor.shape
    _, N_t = heat_target_per_anchor.shape

    vertex_to_cluster = {}
    for s in src_clusters_list:
        for v_idx in s['indices']:
            vertex_to_cluster[int(v_idx)] = s

    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    # Порядок обработки anchor'ов: по убыванию max heat (самые «уверенные» первые)
    anchor_ids = list(src_by_anchor.keys())
    if anchor_order == 'by_max_heat':
        anchor_ids.sort(key=lambda a: -heat_source_per_anchor[a].max())

    used_src = np.zeros(N_s, dtype=bool)
    used_tgt = np.zeros(N_t, dtype=bool)
    fbx_to_cluster = {}
    n_total = 0

    for a in anchor_ids:
        h_s = heat_source_per_anchor[a]
        h_t = heat_target_per_anchor[a]
        active_s = (h_s > heat_threshold * max(h_s.max(), 1e-12)) & (~used_src)
        active_t = (h_t > heat_threshold * max(h_t.max(), 1e-12)) & (~used_tgt)
        src_zone_idx = np.where(active_s)[0]
        src_zone_idx = np.array([v for v in src_zone_idx if int(v) in vertex_to_cluster])
        tgt_zone_idx = np.where(active_t)[0]
        if len(src_zone_idx) < 3 or len(tgt_zone_idx) < 1:
            continue

        # Тот же alignment-блок что в zonal_1d/heat_zone_xyz
        P_src = verts_source[src_zone_idx].astype(np.float64)
        P_tgt = verts_target[tgt_zone_idx].astype(np.float64)
        anchor_pos_src = anchor_pos_tgt = None
        if (use_anchor_align and anchor_verts_source is not None
                and anchor_verts_target is not None
                and a < len(anchor_verts_source)):
            anchor_pos_src = verts_source[int(anchor_verts_source[a])]
            anchor_pos_tgt = verts_target[int(anchor_verts_target[a])]
            origin_src = anchor_pos_src; origin_tgt = anchor_pos_tgt
        else:
            origin_src = P_src.mean(0); origin_tgt = P_tgt.mean(0)
        Q_src = P_src - origin_src; Q_tgt = P_tgt - origin_tgt

        if alignment_mode in ('scale', 'non_rigid'):
            ssrc = np.linalg.norm(Q_src, axis=1).mean() + 1e-9
            stgt = np.linalg.norm(Q_tgt, axis=1).mean() + 1e-9
            Q_tgt = Q_tgt * (ssrc / stgt)

        if use_rotation and alignment_mode != 'centroid':
            d_sq = ((Q_tgt[:, None, :] - Q_src[None, :, :]) ** 2).sum(-1)
            nn = np.argmin(d_sq, axis=1); Y = Q_src[nn]
            H_ = Q_tgt.T @ Y
            U, _, Vt = np.linalg.svd(H_)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0: Vt[-1, :] *= -1; R = Vt.T @ U.T
            Q_tgt = Q_tgt @ R.T

        if alignment_mode == 'non_rigid' and non_rigid_iters > 0:
            try:
                from scipy.interpolate import RBFInterpolator
                for it in range(non_rigid_iters):
                    d_sq = ((Q_tgt[:, None, :] - Q_src[None, :, :]) ** 2).sum(-1)
                    nn = np.argmin(d_sq, axis=1); Y = Q_src[nn]
                    if len(Q_tgt) > 400:
                        sub = np.random.RandomState(42 + it).choice(
                            len(Q_tgt), 400, replace=False)
                        cX, cY = Q_tgt[sub], Y[sub]
                    else:
                        cX, cY = Q_tgt.copy(), Y.copy()
                    if use_anchor_align and anchor_pos_src is not None:
                        pin = np.zeros((10, 3))
                        cX = np.vstack([cX, pin]); cY = np.vstack([cY, pin])
                    rbf = RBFInterpolator(cX, cY, kernel='thin_plate_spline',
                                           smoothing=non_rigid_smoothing)
                    Q_tgt = rbf(Q_tgt)
            except Exception as e:
                print(f"    ⚠ RBF не удался для anchor {a}: {e}")

        d_sq = ((Q_tgt[:, None, :] - Q_src[None, :, :]) ** 2).sum(-1)
        nn_tgt2src = np.argmin(d_sq, axis=1)
        nn_src_global = src_zone_idx[nn_tgt2src]

        for t_v, s_v in zip(tgt_zone_idx, nn_src_global):
            cl_obj = vertex_to_cluster.get(int(s_v))
            if cl_obj is not None:
                fbx_to_cluster[int(t_v)] = cl_obj
                used_tgt[int(t_v)] = True
                used_src[int(s_v)] = True
                n_total += 1

    # Smoothing
    if label_smooth_iters > 0 and faces_target is not None and fbx_to_cluster:
        id_to_cl = {id(cl): cl for cl in fbx_to_cluster.values()}
        labs = {v: id(cl) for v, cl in fbx_to_cluster.items()}
        adj = _build_vertex_adjacency(len(verts_target), faces_target)
        labs = _smooth_labels_on_mesh(labs, adj, n_iter=label_smooth_iters)
        fbx_to_cluster = {v: id_to_cl[l] for v, l in labs.items()}

    g = {}
    for t_v, cl in fbx_to_cluster.items():
        g.setdefault(id(cl), (cl, [])).__getitem__(1).append(t_v)
    target_clusters = []
    for _, (cl, lst) in g.items():
        if not lst: continue
        t_idx = np.array(sorted(set(lst)), dtype=np.int64)
        a = cl['anchor_idx']
        th = heat_target_per_anchor[a][t_idx]
        W = max(th.sum(), 1e-12)
        c_t = (th[:, None] * verts_target[t_idx]).sum(0) / W
        target_clusters.append({
            'source': cl, 'target_indices': t_idx,
            'target_heat': th, 'c_target': c_t,
        })
    print(f"    [sequential_anchor] order={anchor_order}, "
          f"matched={n_total} → {len(target_clusters)} clusters")
    return target_clusters


def gram_schmidt_decorrelate(heat_per_anchor):
    """Применяет Gram-Schmidt orthogonalization к heat-полям как rows матрицы.
    Каждая строка — heat от одного anchor'а, длина N. Orthogonal rows
    обеспечивают что новые "поля" не коррелированы между anchor'ами.

    Возвращает матрицу той же формы (K, N).
    """
    K, N = heat_per_anchor.shape
    # M-mass-weighted скалярное произведение можно использовать; пока обычное
    H = heat_per_anchor.astype(np.float64).copy()
    Q = np.zeros_like(H)
    for i in range(K):
        v = H[i].copy()
        for j in range(i):
            # Вычитаем проекцию на уже ортогональные ранее
            proj = (v @ Q[j]) / max(Q[j] @ Q[j], 1e-12)
            v = v - proj * Q[j]
        nrm = np.linalg.norm(v)
        Q[i] = v / nrm if nrm > 1e-9 else v
    # Возвращаем «положительную часть» (heat не может быть отрицательным,
    # но Gram-Schmidt даёт знакопеременные функции — клипаем в [0, max])
    return np.clip(Q, 0, None)


def assign_target_to_source_decorr_heat(
        verts_target, faces_target, verts_source, faces_source,
        heat_target_per_anchor, heat_source_per_anchor,
        src_clusters_list,
        heat_threshold=0.05,
        label_smooth_iters=2,
        alignment_mode='scale',
        non_rigid_iters=2, non_rigid_smoothing=0.01,
        anchor_verts_source=None, anchor_verts_target=None,
        use_anchor_align=True, use_rotation=False):
    """ВАРИАНТ D: DECORRELATED HEAT через Gram-Schmidt.

    Берём K heat-полей, ортогонализуем их (Gram-Schmidt) → получаем K новых
    полей которые НЕ КОРРЕЛИРОВАНЫ между собой. Это математически устраняет
    проблему «перекрывающихся зон».

    Дальше применяем стандартный heat_zone_xyz matching на ортогонализованных
    heat-полях.

    Замечание: Gram-Schmidt не сохраняет неотрицательность — heat может стать
    «отрицательным» в некоторых вершинах. Мы клипаем в [0, max].
    """
    print(f"    [decorr_heat] Gram-Schmidt orthogonalization on heat fields...")
    H_src_decorr = gram_schmidt_decorrelate(heat_source_per_anchor)
    H_tgt_decorr = gram_schmidt_decorrelate(heat_target_per_anchor)

    # Используем ортогонализованные heat'ы для зонирования.
    # Передаём в стандартный heat_zone_xyz matcher.
    return assign_target_to_source_by_heat_zone(
        verts_target=verts_target, faces_target=faces_target,
        verts_source=verts_source,
        heat_target_per_anchor=H_tgt_decorr,
        heat_source_per_anchor=H_src_decorr,
        src_clusters_list=src_clusters_list,
        heat_threshold=heat_threshold,
        rigid_align=True, n_icp_iters=0,
        label_smooth_iters=label_smooth_iters,
        collect_alignment_data=False,
        faces_source=faces_source,
        alignment_mode=alignment_mode,
        non_rigid_iters=non_rigid_iters,
        non_rigid_smoothing=non_rigid_smoothing,
        anchor_verts_source=anchor_verts_source,
        anchor_verts_target=anchor_verts_target,
        use_anchor_align=use_anchor_align,
        use_rotation=use_rotation,
    )


def assign_target_to_source_ring_match(
        verts_target, faces_target, verts_source, faces_source,
        heat_target_per_anchor, heat_source_per_anchor,
        src_clusters_list,
        anchor_verts_source, anchor_verts_target,
        heat_threshold=0.05,
        heat_tolerance=0.05,        # ширина "кольца" в % heat
        direction_weight=1.0,       # вес направления vs heat в score
        label_smooth_iters=2):
    """ВАРИАНТ E: RING-MATCH (polar coordinates around anchor).

    Концепция:
        Каждая вершина в зоне anchor a описывается ПОЛЯРНЫМИ координатами
        относительно anchor'а:
          - r (radial) = heat[a, v]  →  ~геодезическое расстояние
          - direction  = (v - anchor_pos) / ||v - anchor_pos||

        Вершины с близкими heat-значениями ("одно кольцо") находятся на
        одинаковом геодезическом расстоянии от anchor'а, но в РАЗНЫХ
        анатомических направлениях.

        Алгоритм matching:
          1. Per anchor: ранжируем вершины FLAME и FBX по heat
          2. Для каждой FBX-вершины находим её "кольцо" на FLAME
             (вершины с heat-значением в окне ±heat_tolerance)
          3. Внутри кольца ищем FLAME-вершину с максимально похожим
             направлением (cosine similarity)
          4. Inherit cluster label с этой FLAME-вершины
    """
    K, N_s = heat_source_per_anchor.shape
    _, N_t = heat_target_per_anchor.shape

    vertex_to_cluster = {}
    for s in src_clusters_list:
        for v in s['indices']:
            vertex_to_cluster[int(v)] = s

    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    fbx_to_cluster = {}
    n_total = 0
    n_no_ring = 0      # сколько раз кольцо пустое (fallback на heat-NN)

    for a in src_by_anchor.keys():
        # Anchor positions
        if a >= len(anchor_verts_source) or a >= len(anchor_verts_target):
            continue
        anchor_pos_src = verts_source[int(anchor_verts_source[a])]
        anchor_pos_tgt = verts_target[int(anchor_verts_target[a])]

        # Source zone (только clustered)
        src_vert_set = set()
        for s in src_by_anchor[a]:
            src_vert_set.update(int(v) for v in s['indices'])
        src_zone_idx = np.array(sorted(src_vert_set), dtype=np.int64)
        if len(src_zone_idx) < 3: continue

        # Target zone
        h_t_all = heat_target_per_anchor[a]
        h_t_max = max(h_t_all.max(), 1e-12)
        tgt_zone_idx = np.where(h_t_all > heat_threshold * h_t_max)[0]
        if len(tgt_zone_idx) < 1: continue

        # ── Полярные координаты для SOURCE ──────────────────────────────────
        # Per-anchor max-normalize → heat в [0,1]
        h_s_src = heat_source_per_anchor[a, src_zone_idx]
        h_s_max = max(h_s_src.max(), 1e-12)
        src_heat_norm = h_s_src / h_s_max                              # (S,)

        # Direction vectors (от anchor'а к вершине)
        src_offsets = verts_source[src_zone_idx] - anchor_pos_src      # (S, 3)
        src_dist = np.linalg.norm(src_offsets, axis=1)                 # (S,)
        # Защита от точки anchor'а самого себя (нулевой направляющий)
        valid_src_dir = src_dist > 1e-6
        src_dirs = np.zeros_like(src_offsets)
        src_dirs[valid_src_dir] = (src_offsets[valid_src_dir] /
                                     src_dist[valid_src_dir, None])

        # ── Полярные координаты для TARGET ──────────────────────────────────
        h_s_tgt = heat_target_per_anchor[a, tgt_zone_idx]
        h_t_max_zone = max(h_s_tgt.max(), 1e-12)
        tgt_heat_norm = h_s_tgt / h_t_max_zone                         # (M,)

        tgt_offsets = verts_target[tgt_zone_idx] - anchor_pos_tgt
        tgt_dist = np.linalg.norm(tgt_offsets, axis=1)
        valid_tgt_dir = tgt_dist > 1e-6
        tgt_dirs = np.zeros_like(tgt_offsets)
        tgt_dirs[valid_tgt_dir] = (tgt_offsets[valid_tgt_dir] /
                                     tgt_dist[valid_tgt_dir, None])

        # ── Matching ────────────────────────────────────────────────────────
        # Для каждой target-вершины:
        #   1. Найти SRC-вершины в её "кольце" (|heat_src - heat_tgt| < tolerance)
        #   2. Внутри кольца взять MAX cos_sim direction
        for ti in range(len(tgt_zone_idx)):
            t_v = int(tgt_zone_idx[ti])
            h_t = tgt_heat_norm[ti]
            d_t = tgt_dirs[ti]

            # Heat-ring filter
            heat_diffs = np.abs(src_heat_norm - h_t)
            ring_mask = heat_diffs < heat_tolerance

            if not ring_mask.any() or not valid_tgt_dir[ti]:
                # Fallback: ближайший по heat
                best_si = int(np.argmin(heat_diffs))
                n_no_ring += 1
            else:
                # Combined score: cos_sim - heat_penalty
                ring_indices = np.where(ring_mask)[0]
                cos_sim = src_dirs[ring_indices] @ d_t                # (R,)
                # Score = direction match minus heat penalty (тонкая поправка)
                heat_penalty = heat_diffs[ring_indices] / max(heat_tolerance, 1e-9)
                score = direction_weight * cos_sim - (1 - direction_weight) * heat_penalty
                best_local = int(np.argmax(score))
                best_si = int(ring_indices[best_local])

            s_v = int(src_zone_idx[best_si])
            cl_obj = vertex_to_cluster.get(s_v)
            if cl_obj is not None:
                fbx_to_cluster[t_v] = cl_obj
                n_total += 1

    # Label smoothing
    if label_smooth_iters > 0 and faces_target is not None and fbx_to_cluster:
        id_to_cl = {id(cl): cl for cl in fbx_to_cluster.values()}
        labs = {v: id(cl) for v, cl in fbx_to_cluster.items()}
        adj = _build_vertex_adjacency(len(verts_target), faces_target)
        labs = _smooth_labels_on_mesh(labs, adj, n_iter=label_smooth_iters)
        fbx_to_cluster = {v: id_to_cl[l] for v, l in labs.items()}

    g = {}
    for t_v, cl in fbx_to_cluster.items():
        g.setdefault(id(cl), (cl, [])).__getitem__(1).append(t_v)
    target_clusters = []
    for _, (cl, lst) in g.items():
        if not lst: continue
        t_idx = np.array(sorted(set(lst)), dtype=np.int64)
        a = cl['anchor_idx']
        th = heat_target_per_anchor[a][t_idx]
        W = max(th.sum(), 1e-12)
        c_t = (th[:, None] * verts_target[t_idx]).sum(0) / W
        target_clusters.append({
            'source': cl, 'target_indices': t_idx,
            'target_heat': th, 'c_target': c_t,
        })

    print(f"    [ring_match] heat_tolerance={heat_tolerance}, "
          f"direction_weight={direction_weight}")
    print(f"    [ring_match] matched={n_total}, no_ring_fallback={n_no_ring}, "
          f"→ {len(target_clusters)} target clusters")
    return target_clusters


def assign_target_to_source_tps_global(
        verts_target, faces_target, verts_source, faces_source,
        src_clusters_list,
        anchor_verts_source, anchor_verts_target,
        rbf_smoothing=0.001,
        rbf_kernel='thin_plate_spline',
        label_smooth_iters=2,
        verbose=True):
    """ВАРИАНТ F: TPS GLOBAL — anchor'ы как control points ОДНОГО RBF.

    Концепция:
        Берём K пар anchor'ов (FLAME ↔ FBX) как control points для
        ОДНОГО ГЛОБАЛЬНОГО non-rigid warp (TPS / RBF).

        RBF определяет гладкое преобразование которое:
          - Передвигает каждый anchor_FBX[i] ТОЧНО в anchor_FLAME[i]
          - Все остальные FBX-вершины интерполируются плавно
          - Никаких зон, никаких partition'ов, никаких швов

        После global warp: для каждой deformed FBX-вершины ищем
        ближайшую FLAME-вершину → inherit cluster label.

        Это **классический** подход к non-rigid mesh registration
        (NICP, TPS-RPM, ARAP с landmarks). Чем больше anchor'ов,
        тем точнее регистрация (а не «тем больше проблем»).
    """
    K = len(anchor_verts_source)
    assert K == len(anchor_verts_target), "src и tgt должны иметь одинаковое число anchor'ов"
    if K < 3:
        print(f"  ⚠ tps_global: K={K} anchor'ов недостаточно для TPS "
              f"(минимум 3). Fallback: identity warp.")

    # Control points
    src_anchors = np.array([verts_source[int(a)] for a in anchor_verts_source],
                            dtype=np.float64)
    tgt_anchors = np.array([verts_target[int(a)] for a in anchor_verts_target],
                            dtype=np.float64)

    # Build RBF: maps FBX anchor positions → FLAME anchor positions
    # Тогда RBF(verts_target) → deformed FBX в FLAME-coordinate space
    try:
        from scipy.interpolate import RBFInterpolator
        if K >= 3:
            rbf = RBFInterpolator(tgt_anchors, src_anchors,
                                   kernel=rbf_kernel,
                                   smoothing=rbf_smoothing)
            verts_target_warped = rbf(verts_target.astype(np.float64))
        else:
            # Fallback: translation alignment по средним anchor'ов
            shift = src_anchors.mean(0) - tgt_anchors.mean(0)
            verts_target_warped = verts_target + shift
    except Exception as e:
        print(f"  ⚠ RBF не удался: {e}, fallback на centroid alignment")
        shift = src_anchors.mean(0) - tgt_anchors.mean(0)
        verts_target_warped = verts_target + shift

    # Diagnostic: остаточная ошибка на control points (должна быть ~0
    # для exact interp, малая для smoothed)
    if verbose:
        try:
            tgt_anchors_warped = verts_target_warped[
                np.array([int(a) for a in anchor_verts_target])]
            residuals = np.linalg.norm(tgt_anchors_warped - src_anchors, axis=1)
            print(f"    [tps_global] RBF residuals on anchors: "
                  f"mean={residuals.mean():.6f}, max={residuals.max():.6f}")
        except Exception:
            pass

    # NN: warped_FBX[u] → nearest FLAME vertex
    from scipy.spatial import cKDTree
    tree = cKDTree(verts_source)
    nn_dist, nn_idx = tree.query(verts_target_warped, k=1)

    # Vertex → cluster lookup
    vertex_to_cluster = {}
    for s in src_clusters_list:
        for v in s['indices']:
            vertex_to_cluster[int(v)] = s

    # Inherit cluster labels для каждой FBX-вершины
    fbx_to_cluster = {}
    n_total = 0
    n_no_cluster = 0
    for u in range(len(verts_target)):
        s_v = int(nn_idx[u])
        cl_obj = vertex_to_cluster.get(s_v)
        if cl_obj is not None:
            fbx_to_cluster[u] = cl_obj
            n_total += 1
        else:
            n_no_cluster += 1

    if verbose:
        print(f"    [tps_global] NN matching: {n_total}/{len(verts_target)} верш. "
              f"в кластерах, {n_no_cluster} без cluster'а (не в active zones)")
        print(f"    [tps_global] NN distance: mean={nn_dist.mean():.4f}, "
              f"median={np.median(nn_dist):.4f}, max={nn_dist.max():.4f}")

    # Label smoothing
    if label_smooth_iters > 0 and faces_target is not None and fbx_to_cluster:
        id_to_cl = {id(cl): cl for cl in fbx_to_cluster.values()}
        labs = {v: id(cl) for v, cl in fbx_to_cluster.items()}
        adj = _build_vertex_adjacency(len(verts_target), faces_target)
        labs = _smooth_labels_on_mesh(labs, adj, n_iter=label_smooth_iters)
        fbx_to_cluster = {v: id_to_cl[l] for v, l in labs.items()}

    # Group
    g = {}
    for t_v, cl in fbx_to_cluster.items():
        g.setdefault(id(cl), (cl, [])).__getitem__(1).append(t_v)
    target_clusters = []
    for _, (cl, lst) in g.items():
        if not lst: continue
        t_idx = np.array(sorted(set(lst)), dtype=np.int64)
        # heat-weight для target — может не быть heat если cluster переносится
        # «через TPS», используем единичные веса
        a = cl['anchor_idx']
        # Если heat для этого anchor'а есть (он должен быть в общем pipeline) —
        # используем; иначе uniform
        try:
            # heat не передан как аргумент в эту функцию; для weighted centroid
            # просто берём uniform
            th = np.ones(len(t_idx))
        except Exception:
            th = np.ones(len(t_idx))
        W = max(th.sum(), 1e-12)
        c_t = (th[:, None] * verts_target[t_idx]).sum(0) / W
        target_clusters.append({
            'source': cl, 'target_indices': t_idx,
            'target_heat': th, 'c_target': c_t,
        })

    if verbose:
        print(f"    [tps_global] → {len(target_clusters)} target clusters")
    return target_clusters


def assign_target_to_source_direct_copy(
        verts_target, faces_target, verts_source, faces_source,
        src_clusters_list,
        delta_source,                    # δ_FLAME (известный per-vertex)
        anchor_verts_source, anchor_verts_target,
        rbf_smoothing=0.001,
        rbf_kernel='thin_plate_spline',
        scale_mode='bbox',               # 'bbox' | 'anchor' | 'none'
        label_smooth_iters=2,
        verbose=True):
    """ВАРИАНТ G: DIRECT VERTEX COPY — обход polar decomposition.

    Диагностический режим. Алгоритм:
        1. TPS warp FBX → FLAME coordinate space (через anchor-pairs)
        2. Для каждой FBX-вершины u находим ближайшую FLAME-вершину v*
        3. δ_FBX[u] = δ_FLAME[v*] * scale_factor
                   ↑ НЕ через polar decomposition (R, S, μ)
        4. Используется delta_override в target_clusters

    Зачем это:
        - Если result читаемый → polar decomp ломает correct matching
        - Если result мусор → проблема в самом matching'е (TPS warp плохой)
        - Промежуточный → проблема комбинированная

    scale_mode:
        'bbox'   — отношение bbox-диагоналей мешей (best для разных размеров)
        'anchor' — отношение avg distances между anchor'ами
        'none'   — без масштабирования (FLAME и FBX в одних единицах)
    """
    from scipy.interpolate import RBFInterpolator
    from scipy.spatial import cKDTree

    K = len(anchor_verts_source)
    assert K == len(anchor_verts_target)
    src_anchors = np.array([verts_source[int(a)] for a in anchor_verts_source],
                            dtype=np.float64)
    tgt_anchors = np.array([verts_target[int(a)] for a in anchor_verts_target],
                            dtype=np.float64)

    # Scale factor
    if scale_mode == 'bbox':
        s_src = float(np.linalg.norm(verts_source.max(0) - verts_source.min(0)))
        s_tgt = float(np.linalg.norm(verts_target.max(0) - verts_target.min(0)))
        scale_factor = s_tgt / max(s_src, 1e-9)
    elif scale_mode == 'anchor':
        d_src = np.linalg.norm(src_anchors[:, None] - src_anchors[None, :], axis=-1)
        d_tgt = np.linalg.norm(tgt_anchors[:, None] - tgt_anchors[None, :], axis=-1)
        scale_factor = float(d_tgt.mean() / max(d_src.mean(), 1e-9))
    else:
        scale_factor = 1.0

    if verbose:
        print(f"    [direct_copy] scale_mode={scale_mode}, factor={scale_factor:.4f}")

    # TPS warp FBX → FLAME space для correspondence
    try:
        if K >= 3:
            rbf = RBFInterpolator(tgt_anchors, src_anchors,
                                   kernel=rbf_kernel,
                                   smoothing=rbf_smoothing)
            verts_target_warped = rbf(verts_target.astype(np.float64))
        else:
            shift = src_anchors.mean(0) - tgt_anchors.mean(0)
            verts_target_warped = verts_target + shift
    except Exception as e:
        print(f"    ⚠ RBF failed: {e}, fallback на centroid shift")
        shift = src_anchors.mean(0) - tgt_anchors.mean(0)
        verts_target_warped = verts_target + shift

    # NN: для каждой warped FBX-вершины → ближайшая FLAME
    tree = cKDTree(verts_source)
    nn_dist, nn_idx = tree.query(verts_target_warped, k=1)

    if verbose:
        print(f"    [direct_copy] NN distance: mean={nn_dist.mean():.4f}, "
              f"median={np.median(nn_dist):.4f}, max={nn_dist.max():.4f}")
        delta_mag_src = np.linalg.norm(delta_source, axis=1)
        print(f"    [direct_copy] |δ_FLAME|: mean={delta_mag_src.mean():.5f}, "
              f"max={delta_mag_src.max():.5f}")

    # Прямое копирование δ_FLAME[v*] * scale_factor
    delta_target_per_vertex = delta_source[nn_idx] * scale_factor

    if verbose:
        delta_mag_tgt = np.linalg.norm(delta_target_per_vertex, axis=1)
        print(f"    [direct_copy] |δ_FBX| (после copy + scale): "
              f"mean={delta_mag_tgt.mean():.5f}, max={delta_mag_tgt.max():.5f}")

    # Vertex → cluster lookup
    vertex_to_cluster = {}
    for s in src_clusters_list:
        for v in s['indices']:
            vertex_to_cluster[int(v)] = s

    # Группируем FBX-вершины по их FLAME source-cluster'у
    # (для палитры в визуализации; сам delta уже посчитан)
    fbx_by_cluster = {}
    n_with_cluster = 0
    for u in range(len(verts_target)):
        s_v = int(nn_idx[u])
        cl_obj = vertex_to_cluster.get(s_v)
        if cl_obj is None: continue
        key = id(cl_obj)
        if key not in fbx_by_cluster:
            fbx_by_cluster[key] = (cl_obj, [], [])
        fbx_by_cluster[key][1].append(u)
        fbx_by_cluster[key][2].append(delta_target_per_vertex[u])
        n_with_cluster += 1

    if verbose:
        print(f"    [direct_copy] {n_with_cluster}/{len(verts_target)} верш. "
              f"имеют FLAME-cluster соответствие")

    target_clusters = []
    for key, (cl_obj, idx_list, delta_list) in fbx_by_cluster.items():
        t_idx = np.array(idx_list, dtype=np.int64)
        d_arr = np.array(delta_list, dtype=np.float64)
        th = np.ones(len(t_idx))
        c_t = verts_target[t_idx].mean(0)
        target_clusters.append({
            'source':           cl_obj,
            'target_indices':   t_idx,
            'target_heat':      th,
            'c_target':         c_t,
            'delta_override':   d_arr,    # ← вот ключевое: pre-computed delta
        })

    # Label smoothing — для палитры (cluster IDs), не для delta
    if label_smooth_iters > 0 and faces_target is not None and fbx_by_cluster:
        # Build labels dict
        id_to_cl = {key: cl_obj for key, (cl_obj, _, _) in fbx_by_cluster.items()}
        labs = {}
        for key, (_, idx_list, _) in fbx_by_cluster.items():
            for u in idx_list:
                labs[u] = key
        adj = _build_vertex_adjacency(len(verts_target), faces_target)
        labs = _smooth_labels_on_mesh(labs, adj, n_iter=label_smooth_iters)
        # Re-group with smoothed labels
        regrouped = {}
        for u, key in labs.items():
            cl_obj = id_to_cl[key]
            if key not in regrouped:
                regrouped[key] = (cl_obj, [], [])
            regrouped[key][1].append(u)
            regrouped[key][2].append(delta_target_per_vertex[u])
        target_clusters = []
        for key, (cl_obj, idx_list, delta_list) in regrouped.items():
            t_idx = np.array(idx_list, dtype=np.int64)
            d_arr = np.array(delta_list, dtype=np.float64)
            th = np.ones(len(t_idx))
            c_t = verts_target[t_idx].mean(0)
            target_clusters.append({
                'source': cl_obj,
                'target_indices': t_idx,
                'target_heat': th,
                'c_target': c_t,
                'delta_override': d_arr,
            })

    if verbose:
        print(f"    [direct_copy] → {len(target_clusters)} target clusters "
              f"(with delta_override)")
    return target_clusters


def assign_target_to_source_flame_fit(
        verts_target, faces_target, faces_source,
        v_template_flame, shapedirs_flame,
        src_clusters_list,
        n_betas=100, fit_iters=30,
        learning_rate=0.5, beta_reg=0.001,
        label_smooth_iters=2,
        show_fit_viz=True,
        verbose=True):
    """ВАРИАНТ H: FLAME-FIT — ICP-fit FLAME shape betas к FBX geometry, потом NN.

    1. Bbox-normalize FLAME neutral и FBX
    2. ICP fit FLAME shape betas (β) чтобы FLAME-mesh совпал по форме с FBX:
         minimize Σ ||FLAME[v] - FBX[NN(FLAME[v])]||² + λ·||β||²
    3. После fit'а: каждая FBX-вершина имеет ближайшую FLAME-вершину
       как анатомически correct correspondence (т.к. shapes теперь близки)
    4. Inherit cluster label от FLAME correspondent
    5. Build target_clusters → дальше standard polar decomp transfer

    Это альтернатива всем нашим procedural matching методам (heat_zone, tps_global,
    ring_match и т.д.) которые проваливаются на multi-anchor scenarios.
    """
    from scipy.spatial import cKDTree

    def _norm_bbox(v):
        c = v.mean(0); vc = v - c
        s = float(np.linalg.norm(vc.max(0) - vc.min(0))) + 1e-12
        return vc / s, c, s

    # Normalize both
    v_temp_n, c_f, s_f = _norm_bbox(v_template_flame.astype(np.float64))
    verts_tgt_n, c_t, s_t = _norm_bbox(verts_target.astype(np.float64))
    shapedirs_n = shapedirs_flame.astype(np.float64) / s_f

    # ICP fit
    n_betas_use = min(n_betas, shapedirs_n.shape[2])
    sd = shapedirs_n[:, :, :n_betas_use]
    sd_flat = sd.reshape(-1, n_betas_use)
    reg_I = beta_reg * np.eye(n_betas_use)

    tree_fbx = cKDTree(verts_tgt_n)
    betas = np.zeros(n_betas_use)

    if verbose:
        print(f"    [flame_fit] ICP fit: {n_betas_use} betas, "
              f"{fit_iters} iters, lr={learning_rate}, reg={beta_reg}")

    for it in range(fit_iters):
        V = v_temp_n + np.einsum('vdb,b->vd', sd, betas)
        nn_dist, nn_idx = tree_fbx.query(V, k=1)
        targets = verts_tgt_n[nn_idx]
        rhs_v = (targets - v_temp_n).reshape(-1)
        A_norm = sd_flat.T @ sd_flat + reg_I
        b_norm = sd_flat.T @ rhs_v
        beta_target = np.linalg.solve(A_norm, b_norm)
        betas = (1 - learning_rate) * betas + learning_rate * beta_target
        if verbose and (it == 0 or it == fit_iters // 2 or it == fit_iters - 1):
            print(f"      iter {it+1}/{fit_iters}: mean_NN={nn_dist.mean():.5f}, "
                  f"max_NN={nn_dist.max():.5f}, |β|={np.linalg.norm(betas):.2f}")

    V_fitted = v_temp_n + np.einsum('vdb,b->vd', sd, betas)
    if verbose:
        # Final correspondence quality
        final_nn_dist, _ = tree_fbx.query(V_fitted, k=1)
        print(f"    [flame_fit] FINAL: mean NN dist FLAME→FBX = "
              f"{final_nn_dist.mean():.5f}")

    # ── 3D viz: FBX | FLAME-neutral | FLAME-fitted ───────────────────────────
    if show_fit_viz:
        print(f"\n  >>> ОКНО FLAME-FIT: показываю как подогналась голова "
              f"(FBX синий | FLAME neutral серый | FLAME fitted красный, "
              f"OVERLAY) Q→продолжить <<<")
        try:
            # Все три меша в нормализованном пространстве (как fit делал)
            # Для viz: shift каждый меш горизонтально + overlay-вариант
            geoms = []
            x_cursor = 0.0

            # bbox для горизонтального шага
            bb_w = (verts_tgt_n.max(0) - verts_tgt_n.min(0))[0] * 1.3

            # 1. FBX (синий)
            tgt_shifted = verts_tgt_n.copy()
            tgt_shifted[:, 0] += x_cursor
            m_tgt = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(tgt_shifted),
                o3d.utility.Vector3iVector(faces_target))
            m_tgt.compute_vertex_normals()
            m_tgt.paint_uniform_color([0.20, 0.40, 0.85])    # синий
            geoms.append(m_tgt)
            x_cursor += bb_w

            # 2. FLAME neutral (серый)
            neu_shifted = v_temp_n.copy()
            neu_shifted[:, 0] += x_cursor
            m_neu = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(neu_shifted),
                o3d.utility.Vector3iVector(faces_source))
            m_neu.compute_vertex_normals()
            m_neu.paint_uniform_color([0.70, 0.70, 0.70])    # серый
            geoms.append(m_neu)
            x_cursor += bb_w

            # 3. FLAME fitted (красный)
            fit_shifted = V_fitted.copy()
            fit_shifted[:, 0] += x_cursor
            m_fit = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(fit_shifted),
                o3d.utility.Vector3iVector(faces_source))
            m_fit.compute_vertex_normals()
            m_fit.paint_uniform_color([0.85, 0.25, 0.20])    # красный
            geoms.append(m_fit)
            x_cursor += bb_w

            # 4. OVERLAY: FBX (синий полупрозрачный) + FLAME fitted (красный wireframe)
            # OpenCV / open3d не имеет alpha в обычной триангль mesh,
            # но можно сделать через line set для wireframe.
            # Простой overlay: оба меша на одном месте
            tgt_overlay = verts_tgt_n.copy()
            tgt_overlay[:, 0] += x_cursor
            m_tgt_o = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(tgt_overlay),
                o3d.utility.Vector3iVector(faces_target))
            m_tgt_o.compute_vertex_normals()
            m_tgt_o.paint_uniform_color([0.20, 0.40, 0.85])
            geoms.append(m_tgt_o)

            # FLAME fitted как WIREFRAME на overlay
            fit_overlay = V_fitted.copy()
            fit_overlay[:, 0] += x_cursor
            edges_set = set()
            for f in faces_source:
                a, b, c = int(f[0]), int(f[1]), int(f[2])
                edges_set.add((min(a, b), max(a, b)))
                edges_set.add((min(b, c), max(b, c)))
                edges_set.add((min(a, c), max(a, c)))
            edges_arr = np.array(list(edges_set), dtype=np.int64)
            line_set = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(fit_overlay),
                o3d.utility.Vector2iVector(edges_arr))
            line_set.colors = o3d.utility.Vector3dVector(
                np.tile([0.85, 0.25, 0.20], (len(edges_arr), 1)))
            geoms.append(line_set)

            print(f"    Layout: [FBX] [FLAME neutral] [FLAME fitted] [OVERLAY: FBX + FLAME-fitted wireframe]")
            print(f"    Финальный mean_NN = {final_nn_dist.mean():.5f}, "
                  f"max_NN = {final_nn_dist.max():.5f}")

            vis = o3d.visualization.Visualizer()
            if vis.create_window(
                    window_name=f"FLAME-FIT: FBX | neutral | fitted | overlay  "
                                 f"(mean_NN={final_nn_dist.mean():.4f})  Q→продолжить",
                    width=1900, height=800):
                for g in geoms: vis.add_geometry(g)
                opt = vis.get_render_option()
                opt.mesh_show_back_face = True
                opt.background_color = np.array([0.96, 0.96, 0.96])
                opt.line_width = 1.5
                vis.poll_events(); vis.update_renderer()
                vis.run(); vis.destroy_window()
                print(f"  → окно FLAME-FIT закрыто")
        except Exception as e:
            print(f"  ⚠ Не удалось открыть окно flame_fit viz: {e}")
            import traceback; traceback.print_exc()

    # Correspondence: FBX-vertex → FLAME-vertex (в fitted space)
    tree_flame = cKDTree(V_fitted)
    nn_dist_corr, fbx_to_flame = tree_flame.query(verts_tgt_n, k=1)
    if verbose:
        print(f"    [flame_fit] FBX→FLAME correspondence: "
              f"mean={nn_dist_corr.mean():.5f}, median={np.median(nn_dist_corr):.5f}, "
              f"max={nn_dist_corr.max():.5f}")

    # Vertex → cluster lookup
    vertex_to_cluster = {}
    for s in src_clusters_list:
        for v in s['indices']:
            vertex_to_cluster[int(v)] = s

    # Group FBX-вершины по their FLAME-cluster
    fbx_to_cluster = {}
    n_total = 0
    for u in range(len(verts_target)):
        s_v = int(fbx_to_flame[u])
        cl_obj = vertex_to_cluster.get(s_v)
        if cl_obj is not None:
            fbx_to_cluster[u] = cl_obj
            n_total += 1

    if verbose:
        print(f"    [flame_fit] {n_total}/{len(verts_target)} FBX-вершин "
              f"получили FLAME-cluster")

    # Label smoothing на mesh-графе FBX
    if label_smooth_iters > 0 and faces_target is not None and fbx_to_cluster:
        id_to_cl = {id(cl): cl for cl in fbx_to_cluster.values()}
        labs = {v: id(cl) for v, cl in fbx_to_cluster.items()}
        adj = _build_vertex_adjacency(len(verts_target), faces_target)
        labs = _smooth_labels_on_mesh(labs, adj, n_iter=label_smooth_iters)
        fbx_to_cluster = {v: id_to_cl[l] for v, l in labs.items()}

    # Group
    g = {}
    for t_v, cl in fbx_to_cluster.items():
        g.setdefault(id(cl), (cl, [])).__getitem__(1).append(t_v)
    target_clusters = []
    for _, (cl, lst) in g.items():
        if not lst: continue
        t_idx = np.array(sorted(set(lst)), dtype=np.int64)
        # Uniform heat weights (мы не используем heat для FLAME-fit matching)
        th = np.ones(len(t_idx))
        W = th.sum()
        c_t_centroid = verts_target[t_idx].mean(0)
        target_clusters.append({
            'source': cl, 'target_indices': t_idx,
            'target_heat': th, 'c_target': c_t_centroid,
        })

    if verbose:
        print(f"    [flame_fit] → {len(target_clusters)} target clusters")
    return target_clusters


def apply_target_clusters_transfer(verts_target, target_clusters):
    """Применяет (μ, R, S) source кластеров к target вершинам.
    δ[v] = μ_s + (R_s S_s - I)(verts_target[v] - c_target).

    Если target_cluster содержит поле 'delta_override' (array of per-vertex
    delta) — используем его НАПРЯМУЮ, обходя (R, S, μ, c) формулу.
    Это нужно для direct_copy mode (variant G).
    """
    N = verts_target.shape[0]
    delta = np.zeros((N, 3))
    weight = np.zeros(N)
    I3 = np.eye(3)
    n_overrides = 0
    n_standard = 0
    for tc in target_clusters:
        # ── Direct delta override (variant G: direct_copy) ──────────────
        if 'delta_override' in tc and tc['delta_override'] is not None:
            override = tc['delta_override']
            for j, v_idx in enumerate(tc['target_indices']):
                d = override[j]
                w = tc['target_heat'][j]
                delta[v_idx] += w * d
                weight[v_idx] += w
            n_overrides += 1
            continue

        # ── Стандартная (R, S, μ, c) формула ────────────────────────────
        s = tc['source']
        c_t = tc['c_target']
        RS = s['R'] @ s['S']
        for j, v_idx in enumerate(tc['target_indices']):
            r = verts_target[v_idx] - c_t
            d = s['mu'] + (RS - I3) @ r
            w = tc['target_heat'][j]
            delta[v_idx] += w * d
            weight[v_idx] += w
        n_standard += 1
    if n_overrides > 0:
        print(f"  [apply_transfer] {n_overrides} clusters with delta_override, "
              f"{n_standard} standard")
    valid = weight > 1e-12
    delta[valid] = delta[valid] / weight[valid, None]
    return delta


def reconstruct_delta_from_clusters(verts, N, clusters_per_anchor):
    """Восстанавливает δ применяя линейную трансформацию каждого кластера к
    его вершинам: δ[v] = μ + (RS - I)(verts[v] - c_rest)
    Для пересечений между кластерами — взвешенное среднее.
    """
    delta = np.zeros((N, 3))
    weight = np.zeros(N)
    I3 = np.eye(3)
    for cls in clusters_per_anchor:
        for cl in cls:
            indices = cl['indices']
            heat_w = cl['heat_weights']
            c_rest = cl['c_rest']
            RS = cl['R'] @ cl['S']
            for j, v_idx in enumerate(indices):
                r = verts[v_idx] - c_rest
                d = cl['mu'] + (RS - I3) @ r
                w = heat_w[j]
                delta[v_idx] += w * d
                weight[v_idx] += w
    valid = weight > 1e-12
    delta[valid] = delta[valid] / weight[valid, None]
    return delta


# ── Laplacian smoothing ──────────────────────────────────────────────────────

def build_neighbor_avg_matrix(N, faces):
    rows, cols = [], []
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        rows += [a, a, b, b, c, c]
        cols += [b, c, a, c, a, b]
    edges = set(zip(rows, cols))
    rows = np.array([r for r, _ in edges])
    cols = np.array([c for _, c in edges])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    row_sums = np.array(A.sum(1)).ravel().clip(min=1)
    return (sp.diags(1.0 / row_sums) @ A).tocsr()


def smooth_delta(delta, faces, n_iter=50, alpha=0.5):
    if n_iter <= 0: return delta.copy()
    N = len(delta)
    W = build_neighbor_avg_matrix(N, faces)
    d = delta.copy()
    for _ in range(n_iter):
        d = (1 - alpha) * d + alpha * (W @ d)
    return d


# ── Open3D ────────────────────────────────────────────────────────────────────

def to_colors(values, cmap):
    """Применяет cmap (наша функция) к values. Возвращает (N, 3) RGB."""
    v = np.clip(np.asarray(values, dtype=np.float64), 0, None)
    if v.max() > 0: v = v / v.max()
    out = cmap(v)
    # На случай если cmap вернула (N, 4) — обрежем до RGB
    if out.ndim == 2 and out.shape[1] == 4:
        out = out[:, :3]
    return out


def o3d_mesh(verts, faces, colors=None):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                   o3d.utility.Vector3iVector(faces))
    m.compute_vertex_normals()
    if colors is not None: m.vertex_colors = o3d.utility.Vector3dVector(colors)
    else: m.paint_uniform_color([0.85, 0.75, 0.68])
    return m


def make_cluster_palette(n):
    import colorsys
    if n == 0: return np.zeros((0, 3))
    h = (np.arange(n) * 0.61803398875) % 1.0
    s = 0.75 + 0.25 * (np.arange(n) % 2)
    v = 0.85 + 0.15 * ((np.arange(n) // 2) % 2)
    return np.array([colorsys.hsv_to_rgb(hi, si, vi) for hi, si, vi in zip(h, s, v)])


def make_arrow(p0, p1, color, radius=0.001):
    """Простая линия через LineSet."""
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array([p0, p1]))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    ls.colors = o3d.utility.Vector3dVector(np.array([color]))
    return ls


def pick_vertices(verts, faces, head_name, max_n=20, point_size=6.0):
    """Selection с densified click-coverage.

    Поверх меша рисуется DENSIFIED point cloud:
      - все mesh-вершины (yellow)
      - центроиды face'ов (orange — заполняют площади между вершинами)
      - середины рёбер (light orange — заполняют edges)

    Любая picked точка маппится в ближайшую mesh-вершину через KDTree.
    Эффект: можно кликать почти В ЛЮБУЮ ТОЧКУ ПОВЕРХНОСТИ — она автоматически
    привязывается к ближайшему mesh-вертексу.
    """
    from scipy.spatial import cKDTree

    print(f"\n[{head_name}] Shift+клик до {max_n} точек, Q закроет.")
    print(f"  Densified клик-зона: вершины + центры фейсов + середины рёбер.")
    print(f"  Кликни в любую точку рядом с нужным местом — ближайший вертекс "
          f"будет выбран автоматически.")

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(f"Выбери точки — {head_name}", 1000, 800)

    # Меш — для визуализации формы
    mesh = o3d_mesh(verts, faces)
    vis.add_geometry(mesh)

    # ── DENSIFIED POINT CLOUD: верт. + face-centroids + edge-midpoints ──────
    extra_pts_list = [verts]                     # mesh vertices
    extra_colors_list = [np.tile([0.95, 0.7, 0.15], (len(verts), 1))]  # янтарные

    # Face centroids
    if len(faces) > 0:
        face_centroids = verts[faces].mean(axis=1)               # (F, 3)
        extra_pts_list.append(face_centroids)
        extra_colors_list.append(
            np.tile([0.85, 0.55, 0.10], (len(face_centroids), 1)))

    # Edge midpoints (dedup)
    edge_set = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            e = (min(a, b), max(a, b))
            edge_set.add(e)
    if edge_set:
        edges_arr = np.array(list(edge_set), dtype=np.int64)
        edge_midpoints = 0.5 * (verts[edges_arr[:, 0]] + verts[edges_arr[:, 1]])
        extra_pts_list.append(edge_midpoints)
        extra_colors_list.append(
            np.tile([0.95, 0.75, 0.30], (len(edge_midpoints), 1)))

    all_clickable_pts = np.vstack(extra_pts_list).astype(np.float64)
    all_clickable_cols = np.vstack(extra_colors_list)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_clickable_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_clickable_cols)
    vis.add_geometry(pcd)

    print(f"  ({len(verts)} вершин + {len(faces)} face-centroids + "
          f"{len(edge_set)} edge-midpoints = "
          f"{len(all_clickable_pts)} clickable точек)")

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True

    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()

    # NN-маппинг: любая picked-точка → nearest mesh vertex
    tree = cKDTree(verts)
    chosen, seen = [], set()
    for p in picked:
        # Сначала пробуем .coord (xyz)
        idx = None
        if hasattr(p, 'coord') and p.coord is not None:
            try:
                xyz = np.asarray(p.coord, dtype=np.float64).reshape(3)
                _, nn_idx = tree.query(xyz, k=1)
                idx = int(nn_idx)
            except Exception:
                idx = None
        # Если .coord не сработал → используем index в densified pcd и
        # маппим через xyz из этого pcd
        if idx is None and hasattr(p, 'index') and p.index is not None:
            pcd_idx = int(p.index)
            if 0 <= pcd_idx < len(all_clickable_pts):
                xyz = all_clickable_pts[pcd_idx]
                _, nn_idx = tree.query(xyz, k=1)
                idx = int(nn_idx)
        if idx is None:
            continue
        if 0 <= idx < len(verts) and idx not in seen:
            chosen.append(idx)
            seen.add(idx)
        if len(chosen) >= max_n:
            break

    print(f"  Выбрано {len(chosen)} точек: {chosen}")
    return chosen




def animate_diffusion(verts, faces, L, MM, srcs, total_time, steps, fps=24,
                       stop_on_overlap=False,
                       overlap_threshold=0.05,
                       overlap_fraction=0.02):
    """Heat diffusion с опциональной авто-остановкой при overlap зон.

    stop_on_overlap=True:
        На каждом шаге считаем сколько вершин активны (heat > threshold*max)
        от ≥ 2 anchor'ов одновременно.
        Если эта доля превышает overlap_fraction — откатываемся на
        ПРЕДЫДУЩИЙ шаг (где overlap ещё не было) и останавливаемся.

    Параметры:
        overlap_threshold — доля от max heat для определения "active" вершины
        overlap_fraction  — макс доля overlapping вершин до остановки (0.02 = 2%)
    """
    N = len(srcs)
    dt = total_time / steps
    solve = spla.factorized((MM + dt * L).tocsc())
    A_diag = np.array(MM.diagonal())
    u = np.zeros((N, L.shape[0]))
    for ai in range(N):
        u[ai, srcs[ai]] = 1.0 / max(A_diag[srcs[ai]], 1e-12)

    mesh = o3d_mesh(verts, faces)
    spheres = []
    for s in srcs:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        sph.translate(verts[s]); sph.paint_uniform_color([1, 0, 0])
        sph.compute_vertex_normals(); spheres.append(sph)

    title = (f"ОКНО 1: Diffusion {N} sources, t={total_time}  (Q)"
              + (f"  [auto-stop on overlap≥{overlap_fraction:.0%}]"
                 if stop_on_overlap else ""))
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, 1200, 800)
    for g in [mesh, *spheres]: vis.add_geometry(g)
    vis.get_render_option().mesh_show_back_face = True

    frame_dt = 1.0 / fps
    u_prev = u.copy()                   # snapshot для отката при overlap
    stopped_early = False
    stop_step = steps
    for step_idx in range(steps):
        t0 = time_mod.perf_counter()
        # Запоминаем previous state перед шагом
        u_prev = u.copy()
        for ai in range(N):
            u[ai] = solve(MM @ u[ai])

        # Проверка overlap (после шага)
        if stop_on_overlap:
            # Per-anchor max-norm → "active" если heat > threshold*max
            u_max = u.max(axis=1, keepdims=True).clip(min=1e-12)
            u_norm = u / u_max                                  # (N, V)
            active = u_norm > overlap_threshold                 # (N, V) bool
            n_active_per_vertex = active.sum(axis=0)            # (V,)
            n_overlap = int((n_active_per_vertex >= 2).sum())
            V_total = u.shape[1]
            overlap_frac = n_overlap / max(V_total, 1)
            if overlap_frac > overlap_fraction:
                print(f"\n⚠ AUTO-STOP: overlap={overlap_frac:.2%} > "
                      f"{overlap_fraction:.0%} на шаге {step_idx+1}/{steps}.")
                print(f"  Откат на шаг {step_idx} (overlap ещё малый), "
                      f"эффективный t={dt * step_idx:.5f}")
                u = u_prev                              # rollback
                stopped_early = True
                stop_step = step_idx
                # Обновим viz на последнем валидном состоянии
                mesh.vertex_colors = o3d.utility.Vector3dVector(
                    to_colors(u.sum(0), CMAP_HEAT))
                vis.update_geometry(mesh)
                vis.poll_events(); vis.update_renderer()
                break

        mesh.vertex_colors = o3d.utility.Vector3dVector(to_colors(u.sum(0), CMAP_HEAT))
        vis.update_geometry(mesh)
        if not vis.poll_events(): break
        vis.update_renderer()
        wait = frame_dt - (time_mod.perf_counter() - t0)
        if wait > 0: time_mod.sleep(wait)
    if stopped_early:
        print(f"Диффузия остановлена ДО overlap'а: {stop_step}/{steps} шагов "
              f"(эфф. t={dt * stop_step:.5f} вместо {total_time:.5f}). "
              f"Q — продолжить.")
    else:
        print(f"Диффузия завершена ({steps}/{steps} шагов). Q — продолжить.")
    while vis.poll_events(): vis.update_renderer()
    vis.destroy_window()
    return np.clip(u, 0, None)


# ── Пресеты ──────────────────────────────────────────────────────────────────

SHAPE_PRESETS = {
    0: ("Нейтральная", {}), 1: ("Широкое лицо", {0: 2.5, 1: -1.5}),
    2: ("Узкое вытянутое", {0: -2.0, 1: 2.0}), 3: ("Крупная голова", {0: 3.0, 2: 1.5}),
    4: ("Детское лицо", {1: -2.0, 2: -1.5, 4: -1.0}),
}
EXPR_PRESETS = {
    0: ("Без экспрессии", {}),
    1: ("Экспрессия A (300)", {300: 8.0}),
    2: ("Экспрессия B (301)", {301: 8.0}),
    3: ("Экспрессия C (302)", {302: 8.0}),
    4: ("Экспрессия D (303)", {303: 8.0}),
    5: ("Mix A+B", {300: 5.0, 301: 5.0}),
    6: ("Mix C+D", {302: 5.0, 303: 5.0}),
    7: ("Отриц. A (-300)", {300: -8.0}),
}


def ask_preset(presets, title):
    print(f"\n  {title}")
    for k, (n, _) in presets.items(): print(f"   {k}. {n}")
    print(f"   9. Ввести вручную (idx:val,idx:val)")
    while True:
        raw = input("Номер: ").strip()
        try: x = int(raw)
        except ValueError: continue
        if x in presets: return dict(presets[x][1])
        if x == 9:
            spec = input("  betas (idx:val,...): ").strip()
            out = {}
            for part in spec.split(","):
                p = part.strip()
                if not p: continue
                i, val = p.split(":")
                out[int(i)] = float(val)
            return out


def parse_betas_string(s):
    """'300:8.0,302:-5.0' → {300: 8.0, 302: -5.0}. Пустая строка → {}."""
    out = {}
    s = s.strip()
    if not s: return out
    for part in s.split(","):
        p = part.strip()
        if not p: continue
        i, val = p.split(":")
        out[int(i)] = float(val)
    return out


def console_setup_dialog(defaults):
    """Console-based "form": показывает все параметры разом, пользователь
    редактирует только нужное (Enter — оставить default).
    """
    print("\n" + "═" * 70)
    print("  SETUP — Параметры pipeline (Enter оставит default)")
    print("═" * 70)

    # 1. Shape preset
    print("\nФорма головы (shape preset):")
    for k, (n, _) in SHAPE_PRESETS.items(): print(f"   {k}. {n}")
    shape_in = input(f"Номер [0]: ").strip()
    shape_idx = int(shape_in) if shape_in else 0
    shape = dict(SHAPE_PRESETS[shape_idx][1])

    # 2. Expression
    print("\nЭкспрессия (expression preset):")
    for k, (n, _) in EXPR_PRESETS.items(): print(f"   {k}. {n}")
    print(f"   9. Custom betas вручную")
    expr_in = input(f"Номер [1]: ").strip()
    expr_choice = int(expr_in) if expr_in else 1
    if expr_choice == 9:
        spec = input("  Custom betas (300:8.0,302:-5.0): ").strip()
        expr = parse_betas_string(spec)
    else:
        expr = dict(EXPR_PRESETS.get(expr_choice, (None, {}))[1])

    # 3. Numeric params
    def ask(key, label, cast):
        v_in = input(f"{label} [{defaults[key]}]: ").strip()
        return cast(v_in) if v_in else defaults[key]

    print("\n── Числовые параметры ──")
    params = {
        'shape':            shape,
        'expr':             expr,
        'time':             ask('time',            'Diffusion time',  float),
        'steps':            ask('steps',           'Diffusion steps', int),
        'fps':              ask('fps',             'Animation FPS',   int),
        'position_weight':  ask('position_weight', 'position_weight', float),
        'n_clusters':       ask('n_clusters',      'n_clusters max',  int),
        'heat_threshold':   ask('heat_threshold',  'heat_threshold',  float),
        'smooth_iters':     ask('smooth_iters',    'smooth_iters',    int),
        'smooth_alpha':     ask('smooth_alpha',    'smooth_alpha',    float),
        'smooth_iters_fbx': ask('smooth_iters_fbx','smooth_iters_fbx', int) if 'smooth_iters_fbx' in defaults else 30,
        'n_anchors':        ask('n_anchors',       'max anchor pts',  int),
        'fbx_path':         input(f"FBX path (Enter = skip transfer) [{defaults.get('fbx_path','')}]: ").strip() or defaults.get('fbx_path', ''),
        'geodesic_factor':  ask('geodesic_factor', 'geodesic_factor', float),
        'assign_mode':      (input(f"assign_mode (heat_zone_xyz/zonal_1d/sequential_anchor/decorr_heat) [{defaults.get('assign_mode','heat_zone_xyz')}]: ").strip() or defaults.get('assign_mode', 'heat_zone_xyz')),
        'heat_zone_rigid':    defaults.get('heat_zone_rigid', True),
        'heat_zone_icp_iters': 0,   # повороты отключены
        'heat_zone_alignment_mode': defaults.get('heat_zone_alignment_mode', 'scale'),
        'heat_zone_non_rigid_iters': defaults.get('heat_zone_non_rigid_iters', 2),
        'heat_zone_non_rigid_smoothing': defaults.get('heat_zone_non_rigid_smoothing', 0.01),
        'heat_zone_use_anchor_align':    defaults.get('heat_zone_use_anchor_align', True),
        'heat_zone_use_rotation':        defaults.get('heat_zone_use_rotation', False),
        'heat_zone_hard_partition':      defaults.get('heat_zone_hard_partition', False),
        'geo_filter_enable':             defaults.get('geo_filter_enable', True),
        'geo_filter_tolerance':          defaults.get('geo_filter_tolerance', 1.2),
        'centroid_diff_diagnostic':      defaults.get('centroid_diff_diagnostic', False),
        'auto_pair_anchors':             defaults.get('auto_pair_anchors', True),
        'merge_cross_anchor':            defaults.get('merge_cross_anchor', False),
        'global_n_clusters':             defaults.get('global_n_clusters', 20),
        'global_min_anchor_share':       defaults.get('global_min_anchor_share', 0.05),
        'stop_diffusion_on_overlap':     defaults.get('stop_diffusion_on_overlap', False),
        'diffusion_overlap_threshold':   defaults.get('diffusion_overlap_threshold', 0.05),
        'diffusion_overlap_fraction':    defaults.get('diffusion_overlap_fraction', 0.02),
        'multi_t_enable':                defaults.get('multi_t_enable', False),
        'multi_t_n_times':               defaults.get('multi_t_n_times', 8),
        'multi_t_n_eigs':                defaults.get('multi_t_n_eigs', 80),
        'multi_t_mask_by_single_t':      defaults.get('multi_t_mask_by_single_t', True),
        'heat_zone_smooth':   ask('heat_zone_smooth', 'heat_zone_smooth', int) if 'heat_zone_smooth' in defaults else 2,
        'heat_zone_show_viz': defaults.get('heat_zone_show_viz', True),
        'viz_hks_enable':     defaults.get('viz_hks_enable', False),
        'viz_hks_type':       defaults.get('viz_hks_type', 'hks'),
        'viz_hks_n_clusters': defaults.get('viz_hks_n_clusters', 15),
        'viz_hks_n_eigs':     defaults.get('viz_hks_n_eigs', 100),
        'viz_hks_n_scales':   defaults.get('viz_hks_n_scales', 20),
        'viz_hks_sig_smooth_iters':   defaults.get('viz_hks_sig_smooth_iters', 0),
        'viz_hks_label_smooth_iters': defaults.get('viz_hks_label_smooth_iters', 0),
        'viz_hks_show_xfer':          defaults.get('viz_hks_show_xfer', False),
        'viz_hks_show_similarity':    defaults.get('viz_hks_show_similarity', False),
        'sequential_anchor_order':    defaults.get('sequential_anchor_order', 'by_max_heat'),
        'ring_heat_tolerance':           defaults.get('ring_heat_tolerance', 0.05),
        'ring_direction_weight':         defaults.get('ring_direction_weight', 1.0),
        'tps_smoothing':                 defaults.get('tps_smoothing', 0.001),
        'tps_kernel':                    defaults.get('tps_kernel', 'thin_plate_spline'),
        'direct_copy_scale_mode':        defaults.get('direct_copy_scale_mode', 'bbox'),
        'direct_copy_smoothing':         defaults.get('direct_copy_smoothing', 0.001),
        'flame_fit_n_betas':             defaults.get('flame_fit_n_betas', 100),
        'flame_fit_iters':               defaults.get('flame_fit_iters', 30),
        'flame_fit_lr':                  defaults.get('flame_fit_lr', 0.5),
        'flame_fit_reg':                 defaults.get('flame_fit_reg', 0.001),
        'flame_fit_show_viz':            defaults.get('flame_fit_show_viz', True),
        'smoothing_method':           defaults.get('smoothing_method', 'both'),
        'clustering_method':          defaults.get('clustering_method', 'kmeans'),
        'cluster_similarity_threshold': defaults.get('cluster_similarity_threshold', 0.3),
        '_ok': True,
    }
    print("═" * 70)
    return params


def gui_setup_dialog(defaults):
    """Tkinter диалог. При отсутствии _tkinter → auto fallback на console."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog, simpledialog
    except ImportError:
        print("⚠ tkinter недоступен в этом Python (pyenv обычно без _tkinter).")
        print("  Использую console-режим. Чтобы починить GUI:")
        print("    brew install tcl-tk")
        print("    pyenv uninstall 3.11.9 && pyenv install 3.11.9")
        return console_setup_dialog(defaults)

    root = tk.Tk()
    root.title("Debug Pipeline — Setup")
    root.geometry("640x720")
    root.resizable(True, True)
    root.minsize(560, 400)

    # Container с Canvas + Scrollbar для прокрутки длинной формы
    container = tk.Frame(root)
    container.pack(fill='both', expand=True)

    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)

    # Внутренний frame — туда будут грид'иться все элементы.
    # Padding от краёв окна.
    frame = tk.Frame(canvas, padx=20, pady=15)
    frame.columnconfigure(1, weight=1)

    window_id = canvas.create_window((0, 0), window=frame, anchor='nw')

    def _on_frame_configure(_event):
        canvas.configure(scrollregion=canvas.bbox('all'))
    frame.bind('<Configure>', _on_frame_configure)

    def _on_canvas_configure(event):
        # Растянуть внутренний frame до ширины canvas (минус scrollbar)
        canvas.itemconfig(window_id, width=event.width)
    canvas.bind('<Configure>', _on_canvas_configure)

    # Мышиное колесо — прокрутка (macOS-friendly: delta не делится на 120)
    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * event.delta), 'units')
    # bind_all чтобы прокручивалось из любой части окна
    canvas.bind_all('<MouseWheel>', _on_mousewheel)

    # При закрытии окна — отвязать bind_all чтобы не висели обработчики
    def _on_destroy(_event):
        try: canvas.unbind_all('<MouseWheel>')
        except Exception: pass
    root.bind('<Destroy>', _on_destroy)

    vars_ = {}

    def add_label_entry(label, default, row, width=28):
        tk.Label(frame, text=label, anchor='w', justify='left').grid(
            row=row, column=0, sticky='w', padx=(0, 12), pady=4)
        v = tk.StringVar(value=str(default))
        tk.Entry(frame, textvariable=v, width=width).grid(
            row=row, column=1, padx=(0, 0), pady=4, sticky='ew')
        return v

    def add_label_combo(label, options, default_idx, row, width=28):
        tk.Label(frame, text=label, anchor='w').grid(
            row=row, column=0, sticky='w', padx=(0, 12), pady=4)
        v = tk.StringVar(value=options[default_idx])
        cb = ttk.Combobox(frame, textvariable=v, values=options, width=width,
                           state='readonly')
        cb.grid(row=row, column=1, padx=(0, 0), pady=4, sticky='ew')
        return v

    row = 0
    def section(text):
        nonlocal row
        tk.Label(frame, text=text, font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(12, 4))
        row += 1

    # ── PRESETS (в самом верху, всегда виден) ──────────────────────────────
    PRESETS_DIR = (Path(__file__).resolve().parent / "presets")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    LAST_USED = "_last_used"

    def _list_presets():
        return [""] + sorted([p.stem for p in PRESETS_DIR.glob("*.json")
                                if p.stem != LAST_USED])

    def _do_save_preset(name):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip("_")
        if not safe: return None
        path = PRESETS_DIR / f"{safe}.json"
        values = {k: v.get() for k, v in vars_.items()}
        with open(path, 'w') as f:
            json.dump(values, f, indent=2, ensure_ascii=False)
        return safe

    def _on_save_preset():
        name = simpledialog.askstring("Save preset", "Имя пресета:", parent=root)
        if not name: return
        saved = _do_save_preset(name)
        if saved:
            preset_combo['values'] = _list_presets()
            preset_var.set(saved)
            messagebox.showinfo("Saved", f"Preset '{saved}' сохранён в:\n{PRESETS_DIR}")

    def _on_load_preset():
        name = preset_var.get()
        if not name: return
        path = PRESETS_DIR / f"{name}.json"
        if not path.exists():
            messagebox.showerror("Error", f"Preset {name} не найден")
            return
        try:
            with open(path) as f: values = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        n_applied = 0
        for k, val in values.items():
            if k in vars_:
                try: vars_[k].set(val); n_applied += 1
                except Exception: pass
        print(f"  Loaded preset '{name}' — {n_applied} полей восстановлено")

    def _on_delete_preset():
        name = preset_var.get()
        if not name: return
        if not messagebox.askyesno("Delete", f"Удалить пресет '{name}'?"):
            return
        (PRESETS_DIR / f"{name}.json").unlink(missing_ok=True)
        preset_combo['values'] = _list_presets()
        preset_var.set("")

    def _auto_apply_preset(_event=None):
        """Автозагрузка при выборе из dropdown."""
        if preset_var.get():
            _on_load_preset()

    section("── Пресеты настроек ──")
    preset_row = tk.Frame(frame)
    preset_row.grid(row=row, column=0, columnspan=2, sticky='ew', pady=4)
    preset_row.columnconfigure(1, weight=1)
    tk.Label(preset_row, text="Preset:").grid(row=0, column=0, padx=(0, 6))
    preset_var = tk.StringVar()
    preset_combo = ttk.Combobox(preset_row, textvariable=preset_var,
                                  values=_list_presets(), state='readonly')
    preset_combo.grid(row=0, column=1, sticky='ew', padx=(0, 4))
    preset_combo.bind("<<ComboboxSelected>>", _auto_apply_preset)
    tk.Button(preset_row, text="Load",      command=_on_load_preset,
              width=6).grid(row=0, column=2, padx=2)
    tk.Button(preset_row, text="Save as…",  command=_on_save_preset,
              width=10).grid(row=0, column=3, padx=2)
    tk.Button(preset_row, text="✕",         command=_on_delete_preset,
              width=2).grid(row=0, column=4, padx=2)
    row += 1

    section("── Форма головы ──")
    shape_opts = [f"{k}: {n}" for k, (n, _) in SHAPE_PRESETS.items()]
    vars_['shape'] = add_label_combo("Shape preset:", shape_opts, 0, row); row += 1

    section("── Экспрессия ──")
    expr_opts = [f"{k}: {n}" for k, (n, _) in EXPR_PRESETS.items()]
    vars_['expr'] = add_label_combo("Expression preset:", expr_opts, 1, row); row += 1
    vars_['custom_betas'] = add_label_entry(
        "Custom betas (overrides preset)\n  '300:8.0,302:-5.0'",
        defaults.get('custom_betas', ''), row); row += 1

    section("── Диффузия ──")
    vars_['time']  = add_label_entry("Diffusion time:",  defaults['time'], row);  row += 1
    vars_['steps'] = add_label_entry("Diffusion steps:", defaults['steps'], row); row += 1
    vars_['fps']   = add_label_entry("Animation FPS:",   defaults['fps'], row);   row += 1

    section("── Кластеризация ──")
    vars_['position_weight'] = add_label_entry(
        "Position weight (0=motion only):", defaults['position_weight'], row); row += 1
    vars_['n_clusters'] = add_label_entry(
        "N clusters max per zone:", defaults['n_clusters'], row); row += 1
    vars_['heat_threshold'] = add_label_entry(
        "Heat threshold:", defaults['heat_threshold'], row); row += 1
    _cl_methods = ['kmeans', 'agglomerative']
    _cl_def = defaults.get('clustering_method', 'kmeans')
    _cl_def_idx = _cl_methods.index(_cl_def) if _cl_def in _cl_methods else 0
    vars_['clustering_method'] = add_label_combo(
        "Clustering method:", _cl_methods,
        default_idx=_cl_def_idx, row=row); row += 1
    vars_['cluster_similarity_threshold'] = add_label_entry(
        "  similarity threshold (для agglomerative, 0.1-0.5):",
        defaults.get('cluster_similarity_threshold', 0.3), row); row += 1

    section("── Diffusion Auto-stop ──")
    vars_['stop_diffusion_on_overlap'] = tk.BooleanVar(
        value=bool(defaults.get('stop_diffusion_on_overlap', False)))
    tk.Checkbutton(frame,
                    text="Auto-stop диффузии когда зоны начинают перекрываться",
                    variable=vars_['stop_diffusion_on_overlap']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['diffusion_overlap_threshold'] = add_label_entry(
        "  active threshold (heat > X * max):",
        defaults.get('diffusion_overlap_threshold', 0.05), row); row += 1
    vars_['diffusion_overlap_fraction'] = add_label_entry(
        "  max overlap fraction до остановки (0.02 = 2%):",
        defaults.get('diffusion_overlap_fraction', 0.02), row); row += 1

    section("── Сглаживание ──")
    _sm_methods = ['laplacian', 'sparse', 'both', 'none']
    _sm_def = defaults.get('smoothing_method', 'both')
    _sm_def_idx = _sm_methods.index(_sm_def) if _sm_def in _sm_methods else 2
    vars_['smoothing_method'] = add_label_combo(
        "Smoothing method:", _sm_methods,
        default_idx=_sm_def_idx, row=row); row += 1
    vars_['smooth_iters'] = add_label_entry(
        "Laplacian: smooth iters (HEAD 1):", defaults['smooth_iters'], row); row += 1
    vars_['smooth_alpha'] = add_label_entry(
        "Laplacian: smooth alpha (0..1):", defaults['smooth_alpha'], row); row += 1
    vars_['smooth_iters_fbx'] = add_label_entry(
        "Laplacian: smooth iters FBX (крупнее → больше):",
        defaults.get('smooth_iters_fbx', 30), row); row += 1

    section("── Anchor-точки ──")
    vars_['n_anchors'] = add_label_entry(
        "Max anchor points:", defaults['n_anchors'], row); row += 1

    # ── FBX TRANSFER ──
    section("── Перенос на FBX (опционально) ──")
    vars_['fbx_path'] = tk.StringVar(value=defaults.get('fbx_path', ''))
    tk.Label(frame, text="FBX path:", anchor='w').grid(
        row=row, column=0, sticky='w', padx=(0, 12), pady=4)
    fbx_row = tk.Frame(frame)
    fbx_row.grid(row=row, column=1, sticky='ew', pady=4)
    fbx_row.columnconfigure(0, weight=1)
    fbx_entry = tk.Entry(fbx_row, textvariable=vars_['fbx_path'])
    fbx_entry.grid(row=0, column=0, sticky='ew', padx=(0, 5))

    def browse_fbx():
        """Запускает осascript-picker в фоновом потоке, чтобы не блокировать
        tkinter mainloop. После выбора файла обновляет поле через root.after()."""
        import platform, threading
        if platform.system() != "Darwin":
            # На Linux/Windows — обычный filedialog
            try:
                root.update_idletasks()
                path = filedialog.askopenfilename(
                    parent=root, title="Выбери mesh-файл",
                    filetypes=[("Mesh files", "*.fbx *.obj *.ply"),
                               ("All files", "*.*")])
                if path: vars_['fbx_path'].set(path)
            except Exception as e:
                print(f"Picker error: {e}")
            return

        def worker():
            # Тащим dialog на передний план через активацию Finder.
            # Без этого osascript-окно прячется за tkinter и пользователь
            # не видит куда тыкать.
            project_root = "/Users/kuzix/Documents/GitHub/Muscle-autoskinner"
            script = f'''
                tell application "Finder" to activate
                delay 0.15
                set theFile to choose file with prompt "Выбери mesh-файл" ¬
                    of type {{"fbx","obj","ply"}} ¬
                    default location POSIX file "{project_root}"
                return POSIX path of theFile
            '''
            try:
                r = subprocess.run(["osascript", "-e", script],
                                    capture_output=True, text=True, timeout=600)
                if r.returncode == 0 and r.stdout.strip():
                    path = r.stdout.strip()
                    root.after(0, lambda p=path: vars_['fbx_path'].set(p))
                # На отмене osascript кидает non-zero — игнорим
            except subprocess.TimeoutExpired:
                pass
            except Exception as e:
                print(f"Picker worker error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def clear_fbx():
        vars_['fbx_path'].set("")

    tk.Button(fbx_row, text="Browse...", command=browse_fbx,
              width=10).grid(row=0, column=1, padx=(0, 3))
    tk.Button(fbx_row, text="✕", command=clear_fbx,
              width=2).grid(row=0, column=2)
    row += 1

    vars_['geodesic_factor'] = add_label_entry(
        "geodesic_factor (FBX):", defaults.get('geodesic_factor', 3.0), row); row += 1

    # ── РЕЖИМ МАТЧИНГА (v3) ──
    section("── Режим матчинга кластеров (FBX) ──")
    tk.Label(frame, text="Assign mode:", anchor='w').grid(
        row=row, column=0, sticky='w', padx=(0, 12), pady=4)
    vars_['assign_mode'] = tk.StringVar(value=defaults.get('assign_mode', 'heat_zone_xyz'))
    mode_frame = tk.Frame(frame)
    mode_frame.grid(row=row, column=1, sticky='w', pady=4)
    for mode_val, mode_label in [
        ('heat_zone_xyz',     'Heat-ZONE XYZ (point-cloud align + NN) ⭐'),
        ('zonal_1d',          'B: ZONAL-1D (hard argmax partition + NN)'),
        ('sequential_anchor', 'C: SEQUENTIAL ANCHOR (process in order)'),
        ('decorr_heat',       'D: DECORR-HEAT (Gram-Schmidt orthog.)'),
        ('ring_match',        'E: RING-MATCH (polar heat-rings + direction)'),
        ('tps_global',        'F: TPS-GLOBAL (anchors = control points одного RBF) 🌍'),
        ('direct_copy',       'G: DIRECT-COPY (skip polar, δ_FLAME[v*] напрямую) 🔬'),
        ('flame_fit',         'H: FLAME-FIT (ICP shape-fit + NN correspondence) ⭐⭐'),
    ]:
        tk.Radiobutton(mode_frame, text=mode_label, variable=vars_['assign_mode'],
                        value=mode_val, anchor='w').pack(anchor='w')
    row += 1

    _hz_modes = ['centroid', 'scale', 'non_rigid']
    _hz_def = defaults.get('heat_zone_alignment_mode', 'scale')
    _hz_def_idx = _hz_modes.index(_hz_def) if _hz_def in _hz_modes else 1
    vars_['heat_zone_alignment_mode'] = add_label_combo(
        "heat_zone alignment mode:", _hz_modes,
        default_idx=_hz_def_idx, row=row); row += 1
    vars_['heat_zone_rigid'] = tk.BooleanVar(
        value=bool(defaults.get('heat_zone_rigid', True)))
    tk.Checkbutton(frame, text="  scale-align (применяется для 'scale' и 'non_rigid')",
                    variable=vars_['heat_zone_rigid']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['heat_zone_non_rigid_iters'] = add_label_entry(
        "  non_rigid: RBF iters (1-3):",
        defaults.get('heat_zone_non_rigid_iters', 2), row); row += 1
    vars_['heat_zone_non_rigid_smoothing'] = add_label_entry(
        "  non_rigid: RBF smoothing (0..1):",
        defaults.get('heat_zone_non_rigid_smoothing', 0.01), row); row += 1
    vars_['heat_zone_use_anchor_align'] = tk.BooleanVar(
        value=bool(defaults.get('heat_zone_use_anchor_align', True)))
    tk.Checkbutton(frame, text="  anchor-based alignment (off=centroid)",
                    variable=vars_['heat_zone_use_anchor_align']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['heat_zone_use_rotation'] = tk.BooleanVar(
        value=bool(defaults.get('heat_zone_use_rotation', False)))
    tk.Checkbutton(frame, text="  use rotation (Procrustes pre-step)",
                    variable=vars_['heat_zone_use_rotation']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['heat_zone_hard_partition'] = tk.BooleanVar(
        value=bool(defaults.get('heat_zone_hard_partition', False)))
    tk.Checkbutton(frame, text="  hard-partition zones (argmax, без перекрытий) — auto-ON для multi-t",
                    variable=vars_['heat_zone_hard_partition']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['heat_zone_smooth'] = add_label_entry(
        "heat_zone_xyz label smooth iters:",
        defaults.get('heat_zone_smooth', 2), row); row += 1
    vars_['heat_zone_show_viz'] = tk.BooleanVar(
        value=bool(defaults.get('heat_zone_show_viz', True)))
    tk.Checkbutton(frame, text="heat_zone_xyz: показать 3D окно совмещения зон",
                    variable=vars_['heat_zone_show_viz']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1

    section("── Multi-t Heat Enrichment ──")
    vars_['multi_t_enable'] = tk.BooleanVar(
        value=bool(defaults.get('multi_t_enable', False)))
    tk.Checkbutton(frame,
                    text="Multi-t enrichment: L2-aggregate heat over T temporal scales",
                    variable=vars_['multi_t_enable']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['multi_t_n_times'] = add_label_entry(
        "  T (число temporal scales):",
        defaults.get('multi_t_n_times', 8), row); row += 1
    vars_['multi_t_n_eigs'] = add_label_entry(
        "  k_eigs (для spectral expansion):",
        defaults.get('multi_t_n_eigs', 80), row); row += 1
    vars_['multi_t_mask_by_single_t'] = tk.BooleanVar(
        value=bool(defaults.get('multi_t_mask_by_single_t', True)))
    tk.Checkbutton(frame,
                    text="  Маскировать multi-t зоны по single-t heat (форма от multi-t, область от single-t)",
                    variable=vars_['multi_t_mask_by_single_t']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1

    section("── Geodesic Filter (sanity check переноса) ──")
    vars_['geo_filter_enable'] = tk.BooleanVar(
        value=bool(defaults.get('geo_filter_enable', True)))
    tk.Checkbutton(frame,
                    text="Geo-filter: выпиливать target-вершины дальше source_radius",
                    variable=vars_['geo_filter_enable']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['geo_filter_tolerance'] = add_label_entry(
        "  geo_filter tolerance (× source radius):",
        defaults.get('geo_filter_tolerance', 1.2), row); row += 1
    vars_['centroid_diff_diagnostic'] = tk.BooleanVar(
        value=bool(defaults.get('centroid_diff_diagnostic', False)))
    tk.Checkbutton(frame,
                    text="DIAGNOSTIC: показать относительное смещение centroid'ов (FBX heatmap)",
                    variable=vars_['centroid_diff_diagnostic']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['auto_pair_anchors'] = tk.BooleanVar(
        value=bool(defaults.get('auto_pair_anchors', True)))
    tk.Checkbutton(frame,
                    text="Auto-pair anchor'ы FBX↔FLAME (Hungarian по bbox-позициям)",
                    variable=vars_['auto_pair_anchors']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1

    section("── Global Motion-Groups (one group → multi-anchor views) ──")
    vars_['merge_cross_anchor'] = tk.BooleanVar(
        value=bool(defaults.get('merge_cross_anchor', False)))
    tk.Checkbutton(frame,
                    text="Global K-means (один cluster → несколько anchor-зон с heat-весами)",
                    variable=vars_['merge_cross_anchor']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['global_n_clusters'] = add_label_entry(
        "  global_n_clusters (всего motion-групп на меше):",
        defaults.get('global_n_clusters', 20), row); row += 1
    vars_['global_min_anchor_share'] = add_label_entry(
        "  min_anchor_share (доля тепла, ≥ → view создаётся):",
        defaults.get('global_min_anchor_share', 0.05), row); row += 1

    # ── Sequential anchor (variant C) order ──
    _sa_orders = ['by_max_heat', 'by_index']
    _sa_def = defaults.get('sequential_anchor_order', 'by_max_heat')
    _sa_def_idx = _sa_orders.index(_sa_def) if _sa_def in _sa_orders else 0
    vars_['sequential_anchor_order'] = add_label_combo(
        "sequential_anchor order (C):", _sa_orders,
        default_idx=_sa_def_idx, row=row); row += 1
    vars_['ring_heat_tolerance'] = add_label_entry(
        "ring_match heat tolerance (E, % ширина кольца):",
        defaults.get('ring_heat_tolerance', 0.05), row); row += 1
    vars_['ring_direction_weight'] = add_label_entry(
        "ring_match direction weight (E, 0..1):",
        defaults.get('ring_direction_weight', 1.0), row); row += 1
    vars_['tps_smoothing'] = add_label_entry(
        "tps_global RBF smoothing (F, 0=exact, >0=relax):",
        defaults.get('tps_smoothing', 0.001), row); row += 1
    _tps_kernels = ['thin_plate_spline', 'multiquadric', 'gaussian',
                     'inverse_multiquadric', 'cubic']
    _tps_def = defaults.get('tps_kernel', 'thin_plate_spline')
    _tps_def_idx = _tps_kernels.index(_tps_def) if _tps_def in _tps_kernels else 0
    vars_['tps_kernel'] = add_label_combo(
        "tps_global RBF kernel (F):", _tps_kernels,
        default_idx=_tps_def_idx, row=row); row += 1
    _dc_scale_modes = ['bbox', 'anchor', 'none']
    _dc_def = defaults.get('direct_copy_scale_mode', 'bbox')
    _dc_def_idx = _dc_scale_modes.index(_dc_def) if _dc_def in _dc_scale_modes else 0
    vars_['direct_copy_scale_mode'] = add_label_combo(
        "direct_copy scale mode (G):", _dc_scale_modes,
        default_idx=_dc_def_idx, row=row); row += 1
    vars_['direct_copy_smoothing'] = add_label_entry(
        "direct_copy RBF smoothing (G):",
        defaults.get('direct_copy_smoothing', 0.001), row); row += 1
    vars_['flame_fit_n_betas'] = add_label_entry(
        "flame_fit n_betas (H, 50-300):",
        defaults.get('flame_fit_n_betas', 100), row); row += 1
    vars_['flame_fit_iters'] = add_label_entry(
        "flame_fit iters (H, 20-50):",
        defaults.get('flame_fit_iters', 30), row); row += 1
    vars_['flame_fit_lr'] = add_label_entry(
        "flame_fit learning rate (H, 0.3-0.7):",
        defaults.get('flame_fit_lr', 0.5), row); row += 1
    vars_['flame_fit_reg'] = add_label_entry(
        "flame_fit β regularization (H, 0.0001-0.01):",
        defaults.get('flame_fit_reg', 0.001), row); row += 1
    vars_['flame_fit_show_viz'] = tk.BooleanVar(
        value=bool(defaults.get('flame_fit_show_viz', True)))
    tk.Checkbutton(frame,
                    text="  flame_fit: показать 3D окно после fit'а "
                         "(FBX | neutral | fitted | overlay)",
                    variable=vars_['flame_fit_show_viz']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1

    # ── HKS/WKS DIAGNOSTIC CLUSTERING (отдельный шаг до anchor selection) ─────
    section("── HKS/WKS Spectral Clustering (DIAGNOSTIC) ──")
    vars_['viz_hks_enable'] = tk.BooleanVar(value=bool(defaults.get('viz_hks_enable', False)))
    tk.Checkbutton(frame, text="Кластеризовать вершины по HKS/WKS сигнатурам",
                    variable=vars_['viz_hks_enable']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    _hks_choices = ['hks', 'wks', 'combined']
    _hks_def_idx = (_hks_choices.index(defaults.get('viz_hks_type', 'hks'))
                     if defaults.get('viz_hks_type', 'hks') in _hks_choices else 0)
    vars_['viz_hks_type'] = add_label_combo(
        "  Тип сигнатуры:", _hks_choices,
        default_idx=_hks_def_idx, row=row); row += 1
    vars_['viz_hks_n_clusters'] = add_label_entry(
        "  Число групп (K-means):",
        defaults.get('viz_hks_n_clusters', 15), row); row += 1
    vars_['viz_hks_n_eigs'] = add_label_entry(
        "  n_eigs (мод спектра):",
        defaults.get('viz_hks_n_eigs', 100), row); row += 1
    vars_['viz_hks_n_scales'] = add_label_entry(
        "  n_scales (размер signature):",
        defaults.get('viz_hks_n_scales', 20), row); row += 1
    vars_['viz_hks_sig_smooth_iters'] = add_label_entry(
        "  Лапласиан смус сигнатур (iters, 0=off):",
        defaults.get('viz_hks_sig_smooth_iters', 0), row); row += 1
    vars_['viz_hks_label_smooth_iters'] = add_label_entry(
        "  «Спрас»/Majority-vote labels (iters, 0=off):",
        defaults.get('viz_hks_label_smooth_iters', 0), row); row += 1
    vars_['viz_hks_show_xfer'] = tk.BooleanVar(
        value=bool(defaults.get('viz_hks_show_xfer', False)))
    tk.Checkbutton(frame, text="  Показать FBX через HEAD1 centroids (cross-transfer test)",
                    variable=vars_['viz_hks_show_xfer']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['viz_hks_show_similarity'] = tk.BooleanVar(
        value=bool(defaults.get('viz_hks_show_similarity', False)))
    tk.Checkbutton(frame, text="  Показать SIMILARITY мешы (зелёный=ок, красный=расхождение)",
                    variable=vars_['viz_hks_show_similarity']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1

    result = {}

    def on_start():
        try:
            shape_idx = int(vars_['shape'].get().split(':')[0])
            result['shape'] = dict(SHAPE_PRESETS[shape_idx][1])

            custom = vars_['custom_betas'].get().strip()
            if custom:
                result['expr'] = parse_betas_string(custom)
            else:
                expr_idx = int(vars_['expr'].get().split(':')[0])
                result['expr'] = dict(EXPR_PRESETS[expr_idx][1])

            result['time']            = float(vars_['time'].get())
            result['steps']           = int(vars_['steps'].get())
            result['fps']             = int(vars_['fps'].get())
            result['position_weight'] = float(vars_['position_weight'].get())
            result['n_clusters']      = int(vars_['n_clusters'].get())
            result['clustering_method'] = vars_['clustering_method'].get()
            result['cluster_similarity_threshold'] = float(vars_['cluster_similarity_threshold'].get())
            result['heat_threshold']  = float(vars_['heat_threshold'].get())
            result['smooth_iters']    = int(vars_['smooth_iters'].get())
            result['smooth_alpha']    = float(vars_['smooth_alpha'].get())
            result['smooth_iters_fbx'] = int(vars_['smooth_iters_fbx'].get())
            result['n_anchors']       = int(vars_['n_anchors'].get())
            result['fbx_path']        = vars_['fbx_path'].get().strip()
            result['geodesic_factor'] = float(vars_['geodesic_factor'].get())
            result['assign_mode']     = vars_['assign_mode'].get()
            result['heat_zone_rigid']    = bool(vars_['heat_zone_rigid'].get())
            result['heat_zone_icp_iters'] = 0   # повороты отключены
            result['heat_zone_smooth']   = int(vars_['heat_zone_smooth'].get())
            result['heat_zone_alignment_mode'] = vars_['heat_zone_alignment_mode'].get()
            result['heat_zone_non_rigid_iters'] = int(vars_['heat_zone_non_rigid_iters'].get())
            result['heat_zone_non_rigid_smoothing'] = float(vars_['heat_zone_non_rigid_smoothing'].get())
            result['heat_zone_use_anchor_align'] = bool(vars_['heat_zone_use_anchor_align'].get())
            result['heat_zone_use_rotation']     = bool(vars_['heat_zone_use_rotation'].get())
            result['heat_zone_hard_partition']   = bool(vars_['heat_zone_hard_partition'].get())
            result['heat_zone_show_viz'] = bool(vars_['heat_zone_show_viz'].get())
            result['geo_filter_enable']    = bool(vars_['geo_filter_enable'].get())
            result['geo_filter_tolerance'] = float(vars_['geo_filter_tolerance'].get())
            result['centroid_diff_diagnostic'] = bool(vars_['centroid_diff_diagnostic'].get())
            result['auto_pair_anchors']        = bool(vars_['auto_pair_anchors'].get())
            result['merge_cross_anchor']         = bool(vars_['merge_cross_anchor'].get())
            result['global_n_clusters']          = int(vars_['global_n_clusters'].get())
            result['global_min_anchor_share']    = float(vars_['global_min_anchor_share'].get())
            result['stop_diffusion_on_overlap']  = bool(vars_['stop_diffusion_on_overlap'].get())
            result['diffusion_overlap_threshold'] = float(vars_['diffusion_overlap_threshold'].get())
            result['diffusion_overlap_fraction'] = float(vars_['diffusion_overlap_fraction'].get())
            result['multi_t_enable']           = bool(vars_['multi_t_enable'].get())
            result['multi_t_n_times']          = int(vars_['multi_t_n_times'].get())
            result['multi_t_n_eigs']           = int(vars_['multi_t_n_eigs'].get())
            result['multi_t_mask_by_single_t'] = bool(vars_['multi_t_mask_by_single_t'].get())
            result['viz_hks_enable']    = bool(vars_['viz_hks_enable'].get())
            result['viz_hks_type']      = vars_['viz_hks_type'].get()
            result['viz_hks_n_clusters'] = int(vars_['viz_hks_n_clusters'].get())
            result['viz_hks_n_eigs']    = int(vars_['viz_hks_n_eigs'].get())
            result['viz_hks_n_scales']  = int(vars_['viz_hks_n_scales'].get())
            result['viz_hks_sig_smooth_iters']   = int(vars_['viz_hks_sig_smooth_iters'].get())
            result['viz_hks_label_smooth_iters'] = int(vars_['viz_hks_label_smooth_iters'].get())
            result['viz_hks_show_xfer']          = bool(vars_['viz_hks_show_xfer'].get())
            result['viz_hks_show_similarity']    = bool(vars_['viz_hks_show_similarity'].get())
            result['sequential_anchor_order'] = vars_['sequential_anchor_order'].get()
            result['ring_heat_tolerance']     = float(vars_['ring_heat_tolerance'].get())
            result['ring_direction_weight']   = float(vars_['ring_direction_weight'].get())
            result['tps_smoothing']           = float(vars_['tps_smoothing'].get())
            result['tps_kernel']              = vars_['tps_kernel'].get()
            result['direct_copy_scale_mode']  = vars_['direct_copy_scale_mode'].get()
            result['direct_copy_smoothing']   = float(vars_['direct_copy_smoothing'].get())
            result['flame_fit_n_betas']       = int(vars_['flame_fit_n_betas'].get())
            result['flame_fit_iters']         = int(vars_['flame_fit_iters'].get())
            result['flame_fit_lr']            = float(vars_['flame_fit_lr'].get())
            result['flame_fit_reg']           = float(vars_['flame_fit_reg'].get())
            result['flame_fit_show_viz']      = bool(vars_['flame_fit_show_viz'].get())
            result['smoothing_method']        = vars_['smoothing_method'].get()
            # Auto-save последних настроек для удобства следующего запуска
            try:
                _do_save_preset(LAST_USED)
            except Exception as e:
                print(f"  ⚠ Не удалось auto-save _last_used: {e}")
            result['_ok'] = True
            root.destroy()
        except Exception as e:
            messagebox.showerror("Ошибка ввода", f"Неверное значение: {e}")

    def on_cancel():
        root.destroy()

    row += 1
    # Большая подсказка чтобы пользователь не закрывал GUI без START
    tk.Label(frame, text="↓ Нажми START чтобы запустить pipeline ↓",
              fg="#2E7D32", font=("Arial", 11, "bold")).grid(
        row=row, column=0, columnspan=2, pady=(15, 5))
    row += 1
    btn_frame = tk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=2, pady=(5, 15))
    tk.Button(btn_frame, text="START ▶", command=on_start, bg="#2E7D32", fg="white",
              font=("Arial", 14, "bold"), width=14, height=2).pack(side='left', padx=10)
    tk.Button(btn_frame, text="Отмена", command=on_cancel,
              font=("Arial", 11), width=10, height=2).pack(side='left', padx=10)

    # Закрытие окна через ✕ → трактуем как START (а не как cancel).
    # Это удобнее: даже если случайно закрыл — pipeline запускается.
    # Cancel только через явную кнопку "Отмена".
    root.protocol("WM_DELETE_WINDOW", on_start)

    # Авто-загрузка последних использованных настроек (если есть)
    last_used_path = PRESETS_DIR / f"{LAST_USED}.json"
    if last_used_path.exists():
        try:
            with open(last_used_path) as f: last = json.load(f)
            for k, val in last.items():
                if k in vars_:
                    try: vars_[k].set(val)
                    except Exception: pass
            print(f"  ✓ Auto-loaded предыдущие настройки из {last_used_path}")
        except Exception as e:
            print(f"  ⚠ Не удалось auto-load _last_used: {e}")

    root.mainloop()
    return result if result.get('_ok') else None


def ask_float(prompt, default):
    while True:
        r = input(f"{prompt} [{default}]: ").strip()
        if not r: return default
        try: return float(r)
        except ValueError: pass


def ask_int(prompt, lo, hi, default):
    while True:
        r = input(f"{prompt} [{default}]: ").strip()
        if not r: return default
        try:
            v = int(r)
            if lo <= v <= hi: return v
        except ValueError: pass


def show_meshes_side_by_side(meshes_with_colors_titles, gap_factor=1.3,
                              extra_geometries=None, window_title="Сравнение"):
    """meshes_with_colors_titles: список (verts, faces, colors, label) — рисуем в ряд.

    Использует explicit Visualizer вместо draw_geometries — последний может
    зависать на macOS после нескольких окон подряд.
    """
    geoms = []
    bx = meshes_with_colors_titles[0][0][:, 0].max() - meshes_with_colors_titles[0][0][:, 0].min()
    gap = bx * gap_factor
    for i, (v, f, c, _label) in enumerate(meshes_with_colors_titles):
        v_show = v.copy()
        v_show[:, 0] += gap * i
        geoms.append(o3d_mesh(v_show, f, c))
    if extra_geometries:
        for i, extras in enumerate(extra_geometries):
            for g in extras:
                if hasattr(g, 'points'):
                    pts = np.asarray(g.points)
                    pts_shifted = pts.copy()
                    pts_shifted[:, 0] += gap * i
                    g.points = o3d.utility.Vector3dVector(pts_shifted)
                geoms.append(g)

    # Explicit Visualizer: надёжнее на macOS
    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(window_name=window_title, width=1600, height=800)
    if not ok:
        print(f"  ⚠ Не удалось создать окно '{window_title}'")
        return
    for g in geoms:
        vis.add_geometry(g)
    vis.get_render_option().mesh_show_back_face = True
    # Принудительно обновляем рендер
    vis.poll_events(); vis.update_renderer()
    vis.run()
    vis.destroy_window()
    # Дополнительный flush для macOS
    try:
        for _ in range(3):
            vis.poll_events()
    except Exception: pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flame", default=FLAME_PKL)
    ap.add_argument("--no-gui", action="store_true",
                    help="Старый консольный ввод вместо GUI диалога")
    # Дефолты — также используются как initial values в GUI
    DEFAULTS = {
        'n_anchors':       20,
        'time':            0.002,
        'steps':           60,
        'fps':             24,
        'n_clusters':      5,
        'heat_threshold':  0.05,
        'position_weight': 0.0,
        'smooth_iters':    3,
        'smooth_alpha':    0.5,
        'smooth_iters_fbx': 30,   # отдельно для FBX (он крупнее → нужно больше)
        'custom_betas':    '',
        'fbx_path':        '',
        'geodesic_factor': 3.0,
        # ── v5: 4 режима matching (после strip) ──
        'assign_mode':     'heat_zone_xyz',  # 'heat_zone_xyz' | 'zonal_1d' | 'sequential_anchor' | 'decorr_heat'
        'heat_zone_rigid':   True,        # heat_zone_xyz: scale-alignment
        'heat_zone_icp_iters': 0,         # ПОВОРОТЫ ОТКЛЮЧЕНЫ
        'heat_zone_alignment_mode': 'scale',     # 'centroid' | 'scale' | 'non_rigid'
        'heat_zone_non_rigid_iters': 2,          # RBF iterations
        'heat_zone_non_rigid_smoothing': 0.01,   # RBF smoothing (0=interp, >0=approx)
        'heat_zone_use_anchor_align':    True,   # выравнивать по anchor (а не centroid)
        'heat_zone_use_rotation':        False,  # Procrustes rotation pre-step
        'heat_zone_hard_partition':      False,  # argmax-partition зоны (auto-ON при multi-t)
        'geo_filter_enable':             True,   # выпиливать target-вершины слишком далеко
        'geo_filter_tolerance':          1.2,    # × source radius — max разрешённое расстояние
        'centroid_diff_diagnostic':      False,  # показать FBX heatmap относительного смещения
        'auto_pair_anchors':             True,   # Hungarian переупорядочивание anchor'ов FBX
        'merge_cross_anchor':            False,  # global K-means (multi-anchor views per group)
        'global_n_clusters':             20,     # всего motion-групп на меше при global mode
        'global_min_anchor_share':       0.05,   # min доля тепла anchor'а для создания view
        'stop_diffusion_on_overlap':     False,  # авто-стоп диффузии при overlap зон
        'diffusion_overlap_threshold':   0.05,   # heat > X*max → active
        'diffusion_overlap_fraction':    0.02,   # max доля overlap-вершин до стопа
        'multi_t_enable':                False,  # multi-t heat enrichment (L2-агрегация T scales)
        'multi_t_n_times':               8,      # сколько temporal scales
        'multi_t_n_eigs':                80,     # сколько собств. мод для spectral expansion
        'multi_t_mask_by_single_t':      True,   # маскировать multi-t зоны по single-t heat reach
        'heat_zone_smooth':  2,           # heat_zone_xyz: label smoothing iters
        'heat_zone_show_viz': True,       # heat_zone_xyz: 3D окно совмещения зон
        # ── HKS/WKS diagnostic clustering (отдельный шаг до выбора anchor'ов)
        'viz_hks_enable':    False,       # включить чекбокс
        'viz_hks_type':      'hks',       # 'hks' | 'wks'
        'viz_hks_n_clusters': 15,         # сколько групп по сигнатурам
        'viz_hks_n_eigs':    100,         # сколько мод спектра использовать
        'viz_hks_n_scales':  20,          # размерность signature вектора
        'viz_hks_sig_smooth_iters': 0,    # Лапласиан смус сигнатур ДО K-means (0=off)
        'viz_hks_label_smooth_iters': 0,  # Majority-vote labels ПОСЛЕ K-means (0=off)
        'viz_hks_show_xfer':       False, # показать "FBX via HEAD1 centroids" (cross-transfer)
        'viz_hks_show_similarity': False, # показать SIMILARITY мешы (зелёный/красный)
        'sequential_anchor_order': 'by_max_heat',   # variant C: порядок (by_max_heat | by_index)
        'ring_heat_tolerance':           0.05,   # variant E: ширина кольца (% heat)
        'ring_direction_weight':         1.0,    # variant E: вес direction vs heat (0..1)
        'tps_smoothing':                 0.001,  # variant F: RBF smoothing (0=exact interp)
        'tps_kernel':                    'thin_plate_spline',  # variant F: RBF kernel
        'direct_copy_scale_mode':        'bbox',  # variant G: bbox|anchor|none
        'direct_copy_smoothing':         0.001,   # variant G: RBF smoothing for warp
        'flame_fit_n_betas':             100,     # variant H: число shape betas
        'flame_fit_iters':               30,      # variant H: ICP iters
        'flame_fit_lr':                  0.5,     # variant H: learning rate
        'flame_fit_reg':                 0.001,   # variant H: β regularization
        'flame_fit_show_viz':            True,    # variant H: 3D окно после fit'а
        'smoothing_method': 'both',       # 'laplacian' | 'sparse' | 'both' | 'none'
        'clustering_method': 'kmeans',    # 'kmeans' (fixed K) | 'agglomerative' (threshold)
        'cluster_similarity_threshold': 0.3,  # для agglomerative: меньше=мельче, больше=крупнее
    }
    # Override через CLI (приоритет над GUI defaults, но не выше пользовательского ввода в GUI)
    for k, default in DEFAULTS.items():
        if k == 'custom_betas': continue
        ap.add_argument(f"--{k.replace('_','-')}", type=type(default), default=None)
    args = ap.parse_args()

    # Применяем CLI overrides к дефолтам
    for k in DEFAULTS:
        cli_v = getattr(args, k, None)
        if cli_v is not None:
            DEFAULTS[k] = cli_v

    print(f"\n╔═══════════════════════════════════════════════════════════════╗")
    print(  f"║  DEBUG PIPELINE для ГОЛОВЫ 1 (source)                          ║")
    print(  f"╚═══════════════════════════════════════════════════════════════╝")

    print(f"\nЗагружаю FLAME: {args.flame}")
    v_t, sd, faces = load_flame(args.flame)
    print(f"  {len(v_t)} вершин, {len(faces)} граней")

    # ── Создаём output директорию для дампов CSV/JSON ────────────────────────
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = Path("python/scripts/debug_output") / f"run_{ts}"
    OUT_HEAD1 = OUT_DIR / "head1"
    OUT_FBX = OUT_DIR / "fbx"
    OUT_HEAD1.mkdir(parents=True, exist_ok=True)
    print(f"  Сохраняю промежуточные данные в: {OUT_DIR}/")

    # ── GUI SETUP DIALOG ──────────────────────────────────────────────────────
    if args.no_gui:
        # Old console path
        shape    = ask_preset(SHAPE_PRESETS, "Форма головы")
        expr     = ask_preset(EXPR_PRESETS, "Экспрессия")
        params   = dict(DEFAULTS)
        params['time']  = ask_float("Время диффузии t", params['time'])
        params['steps'] = ask_int("Шагов", 10, 300, params['steps'])
    else:
        params = gui_setup_dialog(DEFAULTS)
        if params is None:
            print("Setup cancelled.")
            return
        shape = params['shape']
        expr  = params['expr']

    n_anchors_max = params['n_anchors']
    t_anim        = params['time']
    num_steps     = params['steps']
    fps           = params['fps']
    n_clusters    = params['n_clusters']
    heat_thresh   = params['heat_threshold']
    pos_weight    = params['position_weight']
    smooth_iters  = params['smooth_iters']
    smooth_alpha  = params['smooth_alpha']
    smooth_iters_fbx = params.get('smooth_iters_fbx', 30)

    # ── Применяем smoothing_method (laplacian | sparse | both | none) ─────────
    # Дропдаун управляет тем какие смуса работают:
    #   laplacian → только δ-Laplacian, label majority-vote OFF
    #   sparse    → только label majority-vote, δ-Laplacian OFF
    #   both      → оба активны (default)
    #   none      → ничего не сглаживается
    sm_method = params.get('smoothing_method', 'both')
    if sm_method == 'laplacian':
        params['heat_zone_smooth'] = 0      # отключаем sparse везде
    elif sm_method == 'sparse':
        smooth_iters = 0
        smooth_iters_fbx = 0
    elif sm_method == 'none':
        smooth_iters = 0
        smooth_iters_fbx = 0
        params['heat_zone_smooth'] = 0
    # 'both' → ничего не меняем, оба активны
    print(f"  smoothing method = '{sm_method}' "
          f"(Laplacian iters: head1={smooth_iters}, fbx={smooth_iters_fbx}; "
          f"sparse: heat_zone_smooth={params.get('heat_zone_smooth', 0)})")

    print(f"\nПараметры:")
    print(f"  shape betas:        {shape}")
    print(f"  expr  betas:        {expr}")
    print(f"  diffusion: t={t_anim}, steps={num_steps}")
    print(f"  clustering: pw={pos_weight}, n_clusters={n_clusters}, "
          f"heat_thresh={heat_thresh}")
    print(f"  smoothing: iters={smooth_iters}, alpha={smooth_alpha}")

    # ── 1. SHAPE ─────────────────────────────────────────────────────────────
    v_raw = apply_betas(v_t, sd, shape)
    verts = normalize_bbox(v_raw)
    N_verts = len(verts)

    # ── 1.5. PRE-CHECK: загружаем FBX заранее (если задан) и показываем рядом
    #        с FLAME-головой для визуальной проверки ориентации до anchor pick
    fbx_path_early = params.get('fbx_path', '').strip()
    v_fbx_raw = None
    verts_fbx = None
    faces_fbx = None
    if fbx_path_early:
        print(f"\n── PRE-CHECK: загружаю FBX заранее: {fbx_path_early}")
        try:
            v_fbx_raw, faces_fbx = load_custom_mesh(fbx_path_early)
            verts_fbx = normalize_bbox(v_fbx_raw)
            print(f"  FBX: {len(verts_fbx)} вершин, {len(faces_fbx)} граней")
            print(f"  HEAD 1 bbox: min={verts.min(0).round(3)}  max={verts.max(0).round(3)}")
            print(f"  FBX    bbox: min={verts_fbx.min(0).round(3)}  max={verts_fbx.max(0).round(3)}")
            print("\n  ОКНО 0: HEAD 1 (rest) | FBX (rest) — проверь ориентацию")
            print("          XYZ оси: красная=X, зелёная=Y, синяя=Z")
            print("          Зелёный шарик — центр меша (после bbox-нормализации)")

            bx_h1 = verts[:, 0].max() - verts[:, 0].min()
            gap_check = bx_h1 * 1.3
            axes_h1  = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            axes_fbx = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            axes_fbx.translate([gap_check, 0, 0])
            sph_h1  = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sph_h1.paint_uniform_color([0.2, 0.8, 0.2]); sph_h1.compute_vertex_normals()
            sph_fbx = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sph_fbx.translate([gap_check, 0, 0])
            sph_fbx.paint_uniform_color([0.2, 0.8, 0.2]); sph_fbx.compute_vertex_normals()

            show_meshes_side_by_side([
                (verts,     faces,     np.tile([0.85, 0.75, 0.68], (N_verts, 1)),
                 "head1_flame"),
                (verts_fbx, faces_fbx,
                 np.tile([0.65, 0.78, 0.85], (len(verts_fbx), 1)), "fbx"),
            ], extra_geometries=[[axes_h1, sph_h1], [axes_fbx, sph_fbx]],
               window_title="ОКНО 0: HEAD 1 | FBX (rest) — проверь ориентацию (Q → продолжить)")
            print("  ✓ Ориентация подтверждена. Продолжаю с anchor pick...")
        except Exception as e:
            print(f"  ⚠ Ошибка загрузки FBX (продолжаю без переноса): {e}")
            verts_fbx = None
            faces_fbx = None
            fbx_path_early = ''

    # ── 1.5 (опционально) HKS/WKS SPECTRAL CLUSTERING DIAGNOSTIC ─────────────
    if params.get('viz_hks_enable', False):
        sig_kind = params.get('viz_hks_type', 'hks')
        n_clusters_viz = int(params.get('viz_hks_n_clusters', 15))
        n_eigs_viz = int(params.get('viz_hks_n_eigs', 100))
        n_scales_viz = int(params.get('viz_hks_n_scales', 20))

        print(f"\n── ШАГ 1.5 — DIAGNOSTIC: {sig_kind.upper()} spectral clustering ──")
        print(f"  n_eigs={n_eigs_viz}, n_scales={n_scales_viz}, "
              f"n_clusters={n_clusters_viz}")
        print(f"  Это чистый intrinsic анализ — НЕ зависит от anchor'ов")

        try:
            # ── Bbox-нормализация (КРИТИЧНО для cross-mesh сравнения)
            # Без неё спектр FBX (если он в другом масштабе) даёт сигнатуры
            # на совершенно другой шкале, и similarity оказывается мусором.
            def _norm_bbox(v):
                c = v.mean(0); vc = v - c
                s = np.linalg.norm(vc.max(0) - vc.min(0)) + 1e-12
                return vc / s
            verts_h1_n = _norm_bbox(verts)
            print(f"  Bbox-нормализация HEAD 1")

            ev_h1, ef_h1 = compute_spectrum(verts_h1_n, faces, n_eigs=n_eigs_viz)

            def build_signature(eigvals, eigvecs, kind, n_scales):
                """hks | wks | combined ([hks_L2norm | wks_L2norm])"""
                if kind == 'hks':
                    t_vals = default_hks_times(eigvals, n_scales=n_scales)
                    return compute_hks(eigvals, eigvecs, t_vals, scale_invariant=True)
                elif kind == 'wks':
                    energies, sigma = default_wks_energies(eigvals, n_scales=n_scales)
                    return compute_wks(eigvals, eigvecs, energies, sigma)
                elif kind == 'combined':
                    # Считаем оба + per-half L2-норм + конкат
                    t_vals = default_hks_times(eigvals, n_scales=n_scales)
                    hks_part = compute_hks(eigvals, eigvecs, t_vals, scale_invariant=True)
                    energies, sigma = default_wks_energies(eigvals, n_scales=n_scales)
                    wks_part = compute_wks(eigvals, eigvecs, energies, sigma)
                    # Per-half L2-норм per vertex → каждая часть вносит равный вклад
                    hks_n = hks_part / np.linalg.norm(hks_part, axis=1, keepdims=True).clip(min=1e-12)
                    wks_n = wks_part / np.linalg.norm(wks_part, axis=1, keepdims=True).clip(min=1e-12)
                    return np.concatenate([hks_n, wks_n], axis=1)        # (N, 2*n_scales)
                else:
                    raise ValueError(f"Unknown sig kind: {kind}")

            sig_h1 = build_signature(ev_h1, ef_h1, sig_kind, n_scales_viz)
            print(f"  {sig_kind.upper()} signature shape: {sig_h1.shape}, "
                  f"λ range [{ev_h1[0]:.3g}, {ev_h1[-1]:.3g}]")
            if sig_kind == 'combined':
                print(f"  [combined] = HKS({n_scales_viz}) + WKS({n_scales_viz}) "
                      f"= {sig_h1.shape[1]}-мерный вектор (per-half L2-нормализован)")

            # ── ЛАПЛАСИАН СМУС СИГНАТУР (опционально, до K-means) ────────────
            sig_smooth_n = int(params.get('viz_hks_sig_smooth_iters', 0))
            if sig_smooth_n > 0:
                print(f"  Лапласиан смус сигнатур HEAD 1 ({sig_smooth_n} iters)...")
                W_h1 = build_neighbor_avg_matrix(len(verts), faces)
                for _ in range(sig_smooth_n):
                    sig_h1 = 0.5 * sig_h1 + 0.5 * (W_h1 @ sig_h1)

            # L2-normalize signatures (для cosine-like clustering)
            sig_norm = sig_h1 / np.linalg.norm(sig_h1, axis=1, keepdims=True).clip(min=1e-12)

            # K-means clustering
            print(f"  K-means clustering на {len(sig_norm)} вершин в "
                  f"{sig_norm.shape[1]}-мерном signature space...")
            km = KMeans(n_clusters=n_clusters_viz, n_init=10, random_state=42)
            labels = km.fit_predict(sig_norm)

            # «СПРАС»/MAJORITY-VOTE smoothing labels (опц., после K-means) ───
            label_smooth_n = int(params.get('viz_hks_label_smooth_iters', 0))
            if label_smooth_n > 0:
                print(f"  «Спрас» (majority-vote) smoothing labels HEAD 1 ({label_smooth_n} iters)...")
                adj_h1 = _build_vertex_adjacency(len(verts), faces)
                lab_dict = {int(v): int(labels[v]) for v in range(len(labels))}
                lab_dict = _smooth_labels_on_mesh(lab_dict, adj_h1, n_iter=label_smooth_n)
                labels = np.array([lab_dict[int(v)] for v in range(len(labels))])

            unique, counts = np.unique(labels, return_counts=True)
            print(f"  Получено {len(unique)} групп. Размеры: "
                  f"min={counts.min()}, max={counts.max()}, mean={counts.mean():.0f}")

            # Сохраняем дамп
            try:
                save_matrix_csv(OUT_HEAD1 / f"viz_{sig_kind}_signatures.csv",
                                 sig_h1, header=",".join([f"s{i}" for i in range(n_scales_viz)]))
                save_matrix_csv(OUT_HEAD1 / f"viz_{sig_kind}_labels.csv",
                                 labels[:, None], header="cluster_id")
                print(f"  → saved viz_{sig_kind}_signatures.csv + viz_{sig_kind}_labels.csv")
            except Exception as e:
                print(f"  ⚠ Не удалось сохранить CSV: {e}")

            # Раскраска
            palette_viz = make_cluster_palette(n_clusters_viz)
            colors_viz = palette_viz[labels]

            # Симметричный меш FBX (если есть) — рисуем тоже
            extra_meshes = [(verts, faces, colors_viz,
                              f"HEAD 1 — {sig_kind.upper()} clustering "
                              f"({n_clusters_viz} groups)")]

            if verts_fbx is not None and faces_fbx is not None:
                print(f"  Считаю {sig_kind.upper()} и для FBX (для сравнения)...")
                verts_fbx_n = _norm_bbox(verts_fbx)
                print(f"  Bbox-нормализация FBX")
                ev_fbx, ef_fbx = compute_spectrum(verts_fbx_n, faces_fbx,
                                                    n_eigs=n_eigs_viz)
                print(f"  FBX λ range [{ev_fbx[0]:.3g}, {ev_fbx[-1]:.3g}]")
                sig_fbx_viz = build_signature(ev_fbx, ef_fbx, sig_kind, n_scales_viz)

                # Лапласиан смус сигнатур FBX (опц., до K-means)
                if sig_smooth_n > 0:
                    print(f"  Лапласиан смус сигнатур FBX ({sig_smooth_n} iters)...")
                    W_fbx = build_neighbor_avg_matrix(len(verts_fbx), faces_fbx)
                    for _ in range(sig_smooth_n):
                        sig_fbx_viz = 0.5 * sig_fbx_viz + 0.5 * (W_fbx @ sig_fbx_viz)

                sig_fbx_norm = sig_fbx_viz / np.linalg.norm(
                    sig_fbx_viz, axis=1, keepdims=True).clip(min=1e-12)

                # ── (1) FBX через HEAD1-centroids (cross-mesh transfer)
                centroids = km.cluster_centers_                       # (K, D)
                a_sq = (sig_fbx_norm ** 2).sum(1, keepdims=True)
                b_sq = (centroids ** 2).sum(1, keepdims=True).T
                cross = sig_fbx_norm @ centroids.T
                D2 = np.maximum(a_sq + b_sq - 2 * cross, 0)
                labels_fbx_xfer = np.argmin(D2, axis=1)
                colors_fbx_xfer = palette_viz[labels_fbx_xfer]

                # ── (2) НЕЗАВИСИМЫЙ K-means на FBX
                print(f"  Независимый K-means на FBX ({len(sig_fbx_norm)} вершин)...")
                km_fbx = KMeans(n_clusters=n_clusters_viz, n_init=10, random_state=42)
                labels_fbx_own = km_fbx.fit_predict(sig_fbx_norm)

                # Label smoothing (опц., после K-means)
                if label_smooth_n > 0:
                    print(f"  «Спрас» smoothing labels FBX ({label_smooth_n} iters)...")
                    adj_fbx = _build_vertex_adjacency(len(verts_fbx), faces_fbx)
                    lab_dict = {int(v): int(labels_fbx_own[v]) for v in range(len(labels_fbx_own))}
                    lab_dict = _smooth_labels_on_mesh(lab_dict, adj_fbx, n_iter=label_smooth_n)
                    labels_fbx_own = np.array([lab_dict[int(v)] for v in range(len(labels_fbx_own))])

                centroids_fbx = km_fbx.cluster_centers_

                # ── Сопоставление кластеров FBX↔HEAD1 для единой цветовой схемы
                # Hungarian-like greedy matching по centroid-distance в signature space
                K_cl = n_clusters_viz
                cost = np.zeros((K_cl, K_cl))
                for i in range(K_cl):
                    for j in range(K_cl):
                        cost[i, j] = np.linalg.norm(centroids_fbx[i] - centroids[j])
                try:
                    from scipy.optimize import linear_sum_assignment
                    row_ind, col_ind = linear_sum_assignment(cost)
                    # remap[fbx_cluster_id] = head1_cluster_id
                    remap = dict(zip(row_ind, col_ind))
                except Exception:
                    # fallback: greedy
                    remap = {}
                    used = set()
                    order = np.argsort(cost.min(axis=1))
                    for i in order:
                        sorted_j = np.argsort(cost[i])
                        for j in sorted_j:
                            if j not in used:
                                remap[int(i)] = int(j); used.add(j); break

                labels_fbx_own_remapped = np.array([remap[int(l)] for l in labels_fbx_own])
                colors_fbx_own = palette_viz[labels_fbx_own_remapped]

                # Качество сопоставления — оценим средний centroid-distance
                avg_match_dist = cost[list(remap.keys()), list(remap.values())].mean()
                print(f"  Кластер-к-кластеру matching: mean centroid distance = "
                      f"{avg_match_dist:.4f} (чем меньше — тем лучше анатомическое "
                      f"соответствие)")

                # FBX own K-means (всегда показываем)
                extra_meshes.append((verts_fbx, faces_fbx, colors_fbx_own,
                                      f"FBX — {sig_kind.upper()}  "
                                      f"(own K-means, matched palette)"))

                # Опционально: FBX через HEAD1 centroids (cross-transfer test)
                if params.get('viz_hks_show_xfer', False):
                    extra_meshes.append((verts_fbx, faces_fbx, colors_fbx_xfer,
                                          f"FBX — {sig_kind.upper()}  "
                                          f"(via HEAD1 centroids)"))

                # ── (3) КАРТА ПОХОЖЕСТИ СИГНАТУР ──────────────────────────
                # Для каждой вершины ищем nearest neighbor на ДРУГОМ меше
                # в signature space → расстояние = "несовпадение"
                print(f"  Считаю карту похожести сигнатур (NN distance)...")
                # FBX → HEAD1: для каждой FBX-вершины ближайшая FLAME
                a_sq = (sig_fbx_norm ** 2).sum(1, keepdims=True)
                b_sq = (sig_norm ** 2).sum(1, keepdims=True).T
                cross = sig_fbx_norm @ sig_norm.T
                D2_fbx2h1 = np.maximum(a_sq + b_sq - 2 * cross, 0)
                nn_dist_fbx = np.sqrt(D2_fbx2h1.min(axis=1))            # (N_fbx,)
                # HEAD1 → FBX: для каждой FLAME-вершины ближайшая FBX
                nn_dist_h1  = np.sqrt(D2_fbx2h1.min(axis=0))            # (N_h1,)

                # Статистика
                print(f"  ── Похожесть сигнатур (L2 в normalized signature space):")
                print(f"     HEAD1 → FBX: mean={nn_dist_h1.mean():.4f}, "
                      f"median={np.median(nn_dist_h1):.4f}, max={nn_dist_h1.max():.4f}")
                print(f"     FBX → HEAD1: mean={nn_dist_fbx.mean():.4f}, "
                      f"median={np.median(nn_dist_fbx):.4f}, max={nn_dist_fbx.max():.4f}")
                avg_sim = (nn_dist_h1.mean() + nn_dist_fbx.mean()) / 2
                quality = ("ОТЛИЧНО ✓" if avg_sim < 0.05 else
                            "ХОРОШО"    if avg_sim < 0.15 else
                            "СРЕДНЕ"    if avg_sim < 0.30 else
                            "ПЛОХО ✗")
                print(f"     >>> Качество соответствия сигнатур: {quality}  "
                      f"(avg={avg_sim:.4f}) <<<")

                # Раскраска: зелёный=маленькое расстояние (хорошо), красный=большое (плохо)
                def similarity_colors(dists, max_clip=None):
                    if max_clip is None:
                        max_clip = np.percentile(dists, 95)
                    t = np.clip(dists / max(max_clip, 1e-9), 0, 1)        # 0=ок, 1=плохо
                    cols = np.zeros((len(dists), 3))
                    cols[:, 0] = t                                          # R растёт с плохим
                    cols[:, 1] = np.clip(1.5 * (1 - t), 0, 1)               # G падает
                    cols[:, 2] = 0.1
                    return cols

                # Используем общий max-clip чтобы цвета сравнимы между мешами
                shared_max = max(np.percentile(nn_dist_h1, 95),
                                  np.percentile(nn_dist_fbx, 95))
                col_h1_sim  = similarity_colors(nn_dist_h1,  max_clip=shared_max)
                col_fbx_sim = similarity_colors(nn_dist_fbx, max_clip=shared_max)

                # Опционально: SIMILARITY-мешы (зелёный/красный)
                if params.get('viz_hks_show_similarity', False):
                    extra_meshes.append((verts, faces, col_h1_sim,
                                          f"HEAD1 SIMILARITY (mean={nn_dist_h1.mean():.3f})"))
                    extra_meshes.append((verts_fbx, faces_fbx, col_fbx_sim,
                                          f"FBX SIMILARITY (mean={nn_dist_fbx.mean():.3f})"))

                # Save dumps
                try:
                    save_matrix_csv(OUT_FBX / f"viz_{sig_kind}_labels_xfer.csv",
                                     labels_fbx_xfer[:, None],
                                     header="cluster_id_from_head1_centroids")
                    save_matrix_csv(OUT_FBX / f"viz_{sig_kind}_labels_own.csv",
                                     labels_fbx_own_remapped[:, None],
                                     header="cluster_id_own_remapped_to_head1")
                    save_matrix_csv(OUT_FBX / f"viz_{sig_kind}_signatures.csv",
                                     sig_fbx_viz,
                                     header=",".join([f"s{i}" for i in range(n_scales_viz)]))
                    save_matrix_csv(OUT_FBX / f"viz_{sig_kind}_similarity.csv",
                                     nn_dist_fbx[:, None],
                                     header="nn_distance_to_head1_signatures")
                    save_matrix_csv(OUT_HEAD1 / f"viz_{sig_kind}_similarity.csv",
                                     nn_dist_h1[:, None],
                                     header="nn_distance_to_fbx_signatures")
                    print(f"  → saved FBX viz_{sig_kind}_* CSV (4 файла) + "
                          f"similarity для HEAD1")
                except Exception as e:
                    print(f"  ⚠ Не удалось сохранить FBX CSV: {e}")

            print(f"\n  >>> ОКНО {sig_kind.upper()} CLUSTERING "
                  f"(Q → продолжить к anchor selection) <<<")
            # Подкорректируем title с меткой качества (если есть)
            quality_suffix = ""
            try:
                quality_suffix = f"  [{quality} avg_sim={avg_sim:.3f}]"
            except NameError:
                pass
            show_meshes_side_by_side(
                extra_meshes, gap_factor=1.3,
                window_title=f"DIAGNOSTIC: {sig_kind.upper()} spectral clustering  "
                              f"K={n_clusters_viz} groups{quality_suffix}  Q→продолжить")
            print(f"  → окно закрыто")
        except Exception as e:
            print(f"  ⚠ Ошибка HKS/WKS clustering: {e}")
            import traceback; traceback.print_exc()

    # ── 2. ANCHOR POINTS ─────────────────────────────────────────────────────
    print("\n── ШАГ 2 — Выбор anchor-точек (Open3D окно, Shift+click) ──")
    src = pick_vertices(verts, faces, "Голова 1", max_n=n_anchors_max)
    if len(src) == 0:
        print("Не выбрано ни одной точки — выход.")
        return
    N_anchors = len(src)

    # Сохраняем base data головы 1
    save_matrix_csv(OUT_HEAD1 / "verts_rest.csv", verts, header="x,y,z")
    save_matrix_csv(OUT_HEAD1 / "faces.csv", faces, header="v0,v1,v2")
    save_matrix_csv(OUT_HEAD1 / "anchor_indices.csv",
                     np.array(src).reshape(-1, 1), header="vertex_index")

    print("\nСтрою Laplacian...")
    L, MM = build_operators(verts, faces)

    # ── 4. ANIMATED DIFFUSION ────────────────────────────────────────────────
    print("\n── ОКНО 1: анимация диффузии ──")
    stop_overlap = bool(params.get('stop_diffusion_on_overlap', False))
    overlap_frac = float(params.get('diffusion_overlap_fraction', 0.02))
    overlap_thr  = float(params.get('diffusion_overlap_threshold', 0.05))
    heat = animate_diffusion(
        verts, faces, L, MM, src, t_anim, num_steps, fps=fps,
        stop_on_overlap=stop_overlap,
        overlap_threshold=overlap_thr,
        overlap_fraction=overlap_frac,
    )
    print(f"  heat shape: {heat.shape}, max per anchor: {heat.max(axis=1).round(4)}")

    # Сохраняем heat-карты (N×K, по колонке на anchor)
    heat_columns = ",".join([f"anchor_{a}" for a in range(N_anchors)])
    save_matrix_csv(OUT_HEAD1 / "heat.csv", heat.T, header=heat_columns)
    print(f"  → saved {OUT_HEAD1/'heat.csv'} ({heat.shape[1]} verts × {N_anchors} anchors)")

    # ── MULTI-T ENRICHMENT для HEAD 1 (ПЕРЕД clustering'ом) ─────────────────
    # Заменяем single-t heat → clustering, polar decomp, ВСЁ downstream
    # будет работать на multi-t enriched зонах.
    heat_h1_before_multi_t = heat.copy()                # сохраним для viz
    if bool(params.get('multi_t_enable', False)):
        mt_n_times = int(params.get('multi_t_n_times', 8))
        mt_n_eigs  = int(params.get('multi_t_n_eigs', 80))
        print(f"\n  ── MULTI-T ENRICHMENT HEAD 1 (T={mt_n_times}, k_eigs={mt_n_eigs}) ──")
        heat_enriched_h1, _times_mt_h1 = enrich_heat_multi_t(
            verts=verts, faces=faces,
            anchor_indices=list(np.asarray(src).ravel()),
            n_times=mt_n_times, n_eigs=mt_n_eigs,
            smooth_iters=5, smooth_alpha=0.5,
            mesh_label="HEAD 1",
        )
        # Заменяем — clusters_per_anchor будут построены на multi-t зонах
        heat = heat_enriched_h1

        # ── Optional: маскируем multi-t heat по СУММЕ single-t зон ──────
        # Multi-t определяет ФОРМУ зон (argmax distribution), single-t —
        # ОБЛАСТЬ где зоны существуют (там где есть тепло от anchor'ов
        # за time t). Вершины вне single-t-reach получают heat=0 во всех
        # multi-t полях.
        if bool(params.get('multi_t_mask_by_single_t', True)):
            print(f"\n  Маскирую multi-t зоны по single-t reach HEAD 1...")
            heat_thresh_mask = float(params.get('heat_threshold', 0.05))
            # Union: вершина "активна" если ХОТЬ ОДИН anchor её достиг в single-t
            h1_norm = heat_h1_before_multi_t / heat_h1_before_multi_t.max(
                axis=1, keepdims=True).clip(min=1e-12)
            active_mask_h1 = h1_norm.max(axis=0) > heat_thresh_mask    # (N,)
            n_active = int(active_mask_h1.sum())
            n_total = len(active_mask_h1)
            print(f"    Single-t active zone HEAD 1: {n_active}/{n_total} верш. "
                  f"({100*n_active/n_total:.1f}%) — вне зоны heat → 0")
            heat[:, ~active_mask_h1] = 0.0

        try:
            save_matrix_csv(OUT_HEAD1 / "heat_multi_t_enriched.csv",
                             heat.T, header=heat_columns)
            save_matrix_csv(OUT_HEAD1 / "heat_multi_t_times.csv",
                             _times_mt_h1[:, None], header="t")
            print(f"    → saved HEAD 1 heat_multi_t_enriched.csv")
        except Exception as e:
            print(f"    ⚠ Не удалось сохранить multi-t dumps: {e}")
        print(f"    ✓ HEAD 1 heat заменён → clustering пойдёт на multi-t зонах")

    def normalized_expr(v_raw_rest, betas_full):
        v_raw_e = apply_betas(v_t, sd, betas_full)
        m = v_raw_rest.mean(0)
        d = np.linalg.norm((v_raw_rest - m).max(0) - (v_raw_rest - m).min(0))
        return (v_raw_e - m) / (d + 1e-12)

    head_expr = normalized_expr(v_raw, {**shape, **expr})
    delta_native = head_expr - verts
    print(f"  max ||δ_native|| = {np.linalg.norm(delta_native, axis=1).max():.4f}")
    print(f"  mean ||δ_native|| = {np.linalg.norm(delta_native, axis=1).mean():.4f}")

    save_matrix_csv(OUT_HEAD1 / "verts_deformed_native.csv", head_expr, header="x,y,z")
    save_matrix_csv(OUT_HEAD1 / "delta_native.csv", delta_native, header="dx,dy,dz")
    print(f"  → saved delta_native.csv & verts_deformed_native.csv")

    # Сохраняем metadata pipeline-параметров
    save_metadata_json(OUT_DIR / "metadata.json", params, shape, expr, N_anchors, src)

    # ── 6. ОКНО 2: rest | native deformed ────────────────────────────────────
    print("\n── ШАГ 6 — ОКНО 2: rest vs native deformed ──")
    col_native = to_colors(np.linalg.norm(delta_native, axis=1), CMAP_DISP)
    show_meshes_side_by_side([
        (verts, faces, np.tile([0.85, 0.75, 0.68], (N_verts, 1)), "rest"),
        (head_expr, faces, col_native, "native deformed"),
    ], window_title="ОКНО 2: rest | native deformed (Q → продолжить)")

    # ── 7. CLUSTERING ────────────────────────────────────────────────────────
    print("\n── ШАГ 7 — Кластеризация ──")
    use_global_clustering = bool(params.get('merge_cross_anchor', False))
    use_multi_t_zones     = bool(params.get('multi_t_enable', False))

    if use_global_clustering:
        # GLOBAL motion-groups: K-means на ВСЕХ active вершинах, затем
        # распределяем группы по anchor-зонам с heat-весами (одна группа
        # может присутствовать в нескольких anchor-зонах с разной силой)
        n_global   = int(params.get('global_n_clusters', N_anchors * n_clusters))
        min_share  = float(params.get('global_min_anchor_share', 0.05))
        print(f"  Режим: GLOBAL motion-groups (n_global={n_global}, "
              f"min_anchor_share={min_share:.0%})")
        clusters_per_anchor = cluster_zones_global(
            verts=verts, delta=delta_native, heat_per_anchor=heat,
            n_clusters_global=n_global,
            position_weight=pos_weight,
            heat_threshold=heat_thresh,
            min_cluster_size=4,
            min_anchor_heat_share=min_share,
            verbose=True,
        )
        # Per-anchor сводка
        for a in range(N_anchors):
            print(f"\n  Anchor #{a}: {len(clusters_per_anchor[a])} views (global motion-groups)")
            for ci, cl in enumerate(clusters_per_anchor[a]):
                ax, ang = axis_angle_from_R(cl['R'])
                share = cl.get('anchor_share', 1.0)
                gid = cl.get('global_group_id', -1)
                print(f"    [{ci}] g={gid}, share={share:.2f}, "
                      f"{len(cl['indices']):4d} verts  "
                      f"μ={cl['mu'].round(4)} (|μ|={np.linalg.norm(cl['mu']):.4f})  "
                      f"rot={np.degrees(ang):.1f}°  "
                      f"stretch={cl['stretches'].round(3)}")
    else:
        cluster_method = params.get('clustering_method', 'kmeans')
        sim_thresh     = float(params.get('cluster_similarity_threshold', 0.3))

        # MULTI-T MODE: кластеризация в HARD-PARTITION zones
        # (как в окне MULTI-T ZONES — каждая вершина строго в одной зоне argmax)
        if use_multi_t_zones:
            print(f"  Режим: PER-ANCHOR clustering на multi-t HARD-PARTITION zones")
            print(f"  Метод: {cluster_method}, max_clusters per zone={n_clusters}")
            partition = _argmax_partition(heat, threshold=heat_thresh)

            # Print zone sizes
            print(f"  Размеры hard-partition зон HEAD 1:")
            for a in range(N_anchors):
                n_in_zone = int((partition == a).sum())
                pct = 100 * n_in_zone / max(len(verts), 1)
                print(f"    anchor {a}: {n_in_zone} verts ({pct:.1f}%)")
            unass = int((partition == -1).sum())
            print(f"    unassigned: {unass} ({100*unass/max(len(verts),1):.1f}%)")

            clusters_per_anchor = []
            for a in range(N_anchors):
                # Mask: оставляем heat только для вершин из hard-partition зоны
                # этого anchor'а. Остальные → 0 → cluster_zone их отсеет threshold'ом.
                zone_mask = (partition == a)
                masked_heat = heat[a].copy()
                masked_heat[~zone_mask] = 0.0

                cls = cluster_zone(
                    masked_heat, delta_native, verts, anchor_idx=a,
                    heat_threshold=heat_thresh,
                    n_clusters_max=n_clusters,
                    position_weight=pos_weight,
                    clustering_method=cluster_method,
                    similarity_threshold=sim_thresh,
                )
                clusters_per_anchor.append(cls)
                print(f"\n  Anchor #{a}: {len(cls)} motion-groups (in hard-partition zone)")
                for ci, cl in enumerate(cls):
                    ax, ang = axis_angle_from_R(cl['R'])
                    print(f"    [{ci}] {len(cl['indices']):4d} verts  "
                          f"μ={cl['mu'].round(4)} (|μ|={np.linalg.norm(cl['mu']):.4f})  "
                          f"rot={np.degrees(ang):.1f}°  "
                          f"stretch={cl['stretches'].round(3)}")
        else:
            print(f"  Метод: {cluster_method}"
                  + (f", similarity_threshold={sim_thresh}" if cluster_method == 'agglomerative' else "")
                  + f", max_clusters per zone={n_clusters}")
            clusters_per_anchor = []
            for a in range(N_anchors):
                cls = cluster_zone(
                    heat[a], delta_native, verts, anchor_idx=a,
                    heat_threshold=heat_thresh,
                    n_clusters_max=n_clusters,
                    position_weight=pos_weight,
                    clustering_method=cluster_method,
                    similarity_threshold=sim_thresh,
                )
                clusters_per_anchor.append(cls)
                print(f"\n  Anchor #{a}: {len(cls)} motion-groups")
                for ci, cl in enumerate(cls):
                    ax, ang = axis_angle_from_R(cl['R'])
                    print(f"    [{ci}] {len(cl['indices']):4d} verts  "
                          f"μ={cl['mu'].round(4)} (|μ|={np.linalg.norm(cl['mu']):.4f})  "
                          f"rot={np.degrees(ang):.1f}°  "
                          f"stretch={cl['stretches'].round(3)}")

    # Сохраняем все кластеры
    save_clusters_json(OUT_HEAD1 / "clusters.json", clusters_per_anchor)
    print(f"  → saved clusters.json ({sum(len(c) for c in clusters_per_anchor)} clusters)")

    # Плоская CSV-таблица: одна строка = одна вершина → её anchor/cluster
    # (удобно для анализа в Excel / pandas)
    flat_rows = []
    global_cid = 0
    for a_idx, cls in enumerate(clusters_per_anchor):
        for local_cid, cl in enumerate(cls):
            for v_idx, hw in zip(cl['indices'], cl['heat_weights']):
                flat_rows.append((int(v_idx), a_idx, local_cid, global_cid, float(hw)))
            global_cid += 1
    if flat_rows:
        flat_arr = np.array(flat_rows, dtype=np.float64)
        save_matrix_csv(OUT_HEAD1 / "clusters_flat.csv", flat_arr,
                         header="vertex_idx,anchor_idx,local_cluster_id,global_cluster_id,heat_weight")
        print(f"  → saved clusters_flat.csv ({len(flat_rows)} rows = vertex→cluster assignments)")

    # ── 8. ОКНО 3: cluster colors + μ arrows ─────────────────────────────────
    print("\n── ШАГ 8 — ОКНО 3: cluster colors + стрелки μ ──")
    vert_colors = np.tile([0.7, 0.7, 0.7], (N_verts, 1))
    vert_weight = np.zeros(N_verts)
    total_clusters = sum(len(cls) for cls in clusters_per_anchor)
    palette = make_cluster_palette(max(total_clusters, 1))
    color_idx = 0
    cluster_color_map = {}
    for a, cls in enumerate(clusters_per_anchor):
        for cl in cls:
            cluster_color_map[id(cl)] = palette[color_idx]
            for j, v_idx in enumerate(cl['indices']):
                w = cl['heat_weights'][j]
                if w > vert_weight[v_idx]:
                    vert_weight[v_idx] = w
                    vert_colors[v_idx] = palette[color_idx]
            color_idx += 1

    # Стрелки μ
    MU_SCALE = 3.0   # длина стрелки = |μ| · MU_SCALE
    arrows = []
    for a, cls in enumerate(clusters_per_anchor):
        for cl in cls:
            col = cluster_color_map[id(cl)]
            p0 = cl['c_rest']
            p1 = cl['c_rest'] + cl['mu'] * MU_SCALE
            arrows.append(make_arrow(p0, p1, color=[0, 0, 0]))    # чёрная стрелка
            # маленький шар на центроиде
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            sph.translate(cl['c_rest']); sph.paint_uniform_color(col.tolist())
            sph.compute_vertex_normals()
            arrows.append(sph)

    # Шарики на anchors
    for s in src:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
        sph.translate(verts[s]); sph.paint_uniform_color([1, 0, 0])
        sph.compute_vertex_normals(); arrows.append(sph)

    show_meshes_side_by_side([
        (verts, faces, vert_colors, "clusters_rest"),
        (head_expr, faces, vert_colors, "clusters_deformed"),
    ], extra_geometries=[arrows, []],
       window_title="ОКНО 3: кластеры + μ стрелки (чёрные) (Q → продолжить)")

    # ── 9. RECONSTRUCT δ FROM CLUSTERS ───────────────────────────────────────
    print("\n── ШАГ 9 — Реконструкция δ из кластеров ──")
    delta_recon = reconstruct_delta_from_clusters(verts, N_verts, clusters_per_anchor)
    head_recon = verts + delta_recon
    err_recon = np.linalg.norm(delta_recon - delta_native, axis=1)
    print(f"  max ||δ_recon - δ_native|| = {err_recon.max():.4f}")
    print(f"  mean ||δ_recon - δ_native|| = {err_recon.mean():.4f}")

    save_matrix_csv(OUT_HEAD1 / "delta_recon.csv", delta_recon, header="dx,dy,dz")
    save_matrix_csv(OUT_HEAD1 / "verts_deformed_recon.csv", head_recon, header="x,y,z")
    print(f"  → saved delta_recon.csv & verts_deformed_recon.csv")

    # ── 10. ОКНО 4: rest | native | reconstructed ────────────────────────────
    print("\n── ШАГ 10 — ОКНО 4: native vs reconstructed ──")
    col_recon = to_colors(np.linalg.norm(delta_recon, axis=1), CMAP_DISP)
    show_meshes_side_by_side([
        (verts, faces, np.tile([0.85, 0.75, 0.68], (N_verts, 1)), "rest"),
        (head_expr, faces, col_native, "δ_native"),
        (head_recon, faces, col_recon, "δ_reconstructed"),
    ], window_title="ОКНО 4: rest | native | reconstructed  (Q → продолжить)")

    # ── 11. SMOOTH RECONSTRUCTED δ ────────────────────────────────────────────
    print("\n── ШАГ 11 — Сглаживание реконструированной δ ──")
    delta_smooth = smooth_delta(delta_recon, faces,
                                 n_iter=smooth_iters,
                                 alpha=smooth_alpha)
    head_smooth = verts + delta_smooth
    max_smoothed = np.linalg.norm(delta_smooth, axis=1).max()
    print(f"  iters={smooth_iters}, α={smooth_alpha}")
    print(f"  max ||δ_smooth|| = {max_smoothed:.4f} "
          f"(было {np.linalg.norm(delta_recon, axis=1).max():.4f}, потеря "
          f"{(1 - max_smoothed/np.linalg.norm(delta_recon,axis=1).max())*100:.1f}%)")

    save_matrix_csv(OUT_HEAD1 / "delta_smoothed.csv", delta_smooth, header="dx,dy,dz")
    save_matrix_csv(OUT_HEAD1 / "verts_deformed_smoothed.csv",
                     head_smooth, header="x,y,z")
    print(f"  → saved delta_smoothed.csv & verts_deformed_smoothed.csv")

    # ── 12. ОКНО 5: native | recon | smoothed ────────────────────────────────
    print("\n── ШАГ 12 — ОКНО 5: native | recon | smoothed ──")
    col_smooth = to_colors(np.linalg.norm(delta_smooth, axis=1), CMAP_DISP)
    show_meshes_side_by_side([
        (head_expr, faces, col_native, "δ_native"),
        (head_recon, faces, col_recon, "δ_recon"),
        (head_smooth, faces, col_smooth, "δ_smoothed"),
    ], window_title="ОКНО 5: native | reconstructed | smoothed  (Q → выход)")

    # ── 13. ПЕРЕНОС НА FBX (если задан путь) ─────────────────────────────────
    if verts_fbx is not None and faces_fbx is not None:
        print("\n" + "═" * 70)
        print(f"  ШАГ 13 — Перенос на FBX: {fbx_path_early}")
        print("═" * 70)

        OUT_FBX.mkdir(parents=True, exist_ok=True)
        save_matrix_csv(OUT_FBX / "verts_rest.csv", verts_fbx, header="x,y,z")
        save_matrix_csv(OUT_FBX / "faces.csv", faces_fbx, header="v0,v1,v2")

        # Anchor selection на FBX (столько же сколько на голове 1)
        print(f"\n  Открываю окно выбора {N_anchors} anchor-точек на FBX...")
        print(f"  Ставь в ТОЙ ЖЕ анатомической последовательности что на голове 1.")
        src_fbx = pick_vertices(verts_fbx, faces_fbx, "FBX target", max_n=N_anchors)
        while len(src_fbx) < N_anchors:
            s = input(f"  Нужно ещё {N_anchors - len(src_fbx)} точек. "
                      f"Индекс #{len(src_fbx)+1}: ").strip()
            try:
                v = int(s)
                if 0 <= v < len(verts_fbx): src_fbx.append(v)
            except ValueError: print("  Целое число")

        # ── AUTO-PAIR anchor'ов FBX к FLAME по bbox-нормализованным позициям ──
        # Если пользователь кликнул в неправильном порядке — Hungarian
        # переупорядочит src_fbx чтобы anchor i на обоих мешах был
        # анатомически тем же ориентиром.
        if bool(params.get('auto_pair_anchors', True)):
            print(f"\n  ── AUTO-PAIR anchor'ов ──")
            new_src_fbx, _match_info = auto_pair_anchors(
                verts_src=verts,        anchor_indices_src=src,
                verts_tgt=verts_fbx,    anchor_indices_tgt=src_fbx,
                verbose=True,
            )
            src_fbx = new_src_fbx
        else:
            print(f"  Auto-pair OFF (assuming anchor'ы уже в правильном порядке)")

        # Laplacian на FBX
        print("\n  Строю Laplacian для FBX...")
        L_fbx, MM_fbx = build_operators(verts_fbx, faces_fbx)

        # Сохраняем anchor-индексы FBX
        save_matrix_csv(OUT_FBX / "anchor_indices.csv",
                         np.array(src_fbx).reshape(-1, 1), header="vertex_index")

        # Heat diffusion на FBX (анимация)
        print("  Анимирую диффузию на FBX...")
        heat_fbx = animate_diffusion(
            verts_fbx, faces_fbx, L_fbx, MM_fbx, src_fbx,
            t_anim, num_steps, fps=fps,
            stop_on_overlap=stop_overlap,
            overlap_threshold=overlap_thr,
            overlap_fraction=overlap_frac,
        )

        heat_fbx_columns = ",".join([f"anchor_{a}" for a in range(N_anchors)])
        save_matrix_csv(OUT_FBX / "heat.csv", heat_fbx.T, header=heat_fbx_columns)
        print(f"  → saved FBX heat.csv ({heat_fbx.shape[1]} verts × {N_anchors})")

        # ── MULTI-T ENRICHMENT для FBX (heat_fbx уже посчитан, обогащаем) ────
        # HEAD 1 уже обогащён ранее (перед clustering'ом), heat сейчас на multi-t.
        if bool(params.get('multi_t_enable', False)):
            mt_n_times = int(params.get('multi_t_n_times', 8))
            mt_n_eigs  = int(params.get('multi_t_n_eigs', 80))
            print(f"\n  ── MULTI-T ENRICHMENT FBX (T={mt_n_times}, k_eigs={mt_n_eigs}) ──")
            heat_fbx_before = heat_fbx.copy()
            heat_enriched_fbx, _times_mt_fbx = enrich_heat_multi_t(
                verts=verts_fbx, faces=faces_fbx,
                anchor_indices=list(np.asarray(src_fbx).ravel()),
                n_times=mt_n_times, n_eigs=mt_n_eigs,
                smooth_iters=5, smooth_alpha=0.5,
                mesh_label="FBX",
            )
            heat_fbx = heat_enriched_fbx

            # ── Optional: маскируем multi-t FBX по СУММЕ single-t зон ────
            if bool(params.get('multi_t_mask_by_single_t', True)):
                print(f"\n  Маскирую multi-t зоны по single-t reach FBX...")
                heat_thresh_mask = float(params.get('heat_threshold', 0.05))
                fbx_norm = heat_fbx_before / heat_fbx_before.max(
                    axis=1, keepdims=True).clip(min=1e-12)
                active_mask_fbx = fbx_norm.max(axis=0) > heat_thresh_mask
                n_active = int(active_mask_fbx.sum())
                n_total = len(active_mask_fbx)
                print(f"    Single-t active zone FBX: {n_active}/{n_total} верш. "
                      f"({100*n_active/n_total:.1f}%) — вне зоны heat → 0")
                heat_fbx[:, ~active_mask_fbx] = 0.0

            try:
                save_matrix_csv(OUT_FBX / "heat_multi_t_enriched.csv",
                                 heat_fbx.T, header=heat_fbx_columns)
                print(f"    → saved FBX heat_multi_t_enriched.csv")
            except Exception as e:
                print(f"    ⚠ Не удалось сохранить FBX multi-t dump: {e}")
            print(f"    ✓ FBX heat заменён → matching пойдёт на multi-t зонах")

            # ── ОКНО MULTI-T ZONES: single-t vs multi-t для обеих голов ──────
            print(f"\n  >>> ОКНО MULTI-T ZONES (single-t vs multi-t)  Q→продолжить <<<")
            try:
                palette_mt = make_cluster_palette(N_anchors)

                def hard_argmax_colors(H_per_anchor, palette):
                    H = H_per_anchor / H_per_anchor.max(axis=1, keepdims=True).clip(min=1e-12)
                    dom = np.argmax(H, axis=0)
                    cols = palette[dom]
                    overall = H.max(axis=0)
                    fade = np.clip(overall / 0.3, 0.2, 1.0)
                    return cols * fade[:, None]

                col_h1_single  = hard_argmax_colors(heat_h1_before_multi_t,  palette_mt)
                col_h1_multi   = hard_argmax_colors(heat,                    palette_mt)  # уже multi-t
                col_fbx_single = hard_argmax_colors(heat_fbx_before,         palette_mt)
                col_fbx_multi  = hard_argmax_colors(heat_fbx,                palette_mt)  # уже multi-t

                # Diff metrics
                dom_h1_s  = np.argmax(heat_h1_before_multi_t, axis=0)
                dom_h1_m  = np.argmax(heat,                   axis=0)
                dom_fbx_s = np.argmax(heat_fbx_before,        axis=0)
                dom_fbx_m = np.argmax(heat_fbx,               axis=0)
                n_diff_h1  = int((dom_h1_s  != dom_h1_m).sum())
                n_diff_fbx = int((dom_fbx_s != dom_fbx_m).sum())
                pct_h1  = 100 * n_diff_h1  / max(len(verts), 1)
                pct_fbx = 100 * n_diff_fbx / max(len(verts_fbx), 1)
                print(f"  [multi-t diff] HEAD 1: {n_diff_h1}/{len(verts)} "
                      f"({pct_h1:.1f}%) сменили dominant anchor")
                print(f"  [multi-t diff] FBX:    {n_diff_fbx}/{len(verts_fbx)} "
                      f"({pct_fbx:.1f}%) сменили dominant anchor")

                show_meshes_side_by_side(
                    [
                        (verts,     faces,     col_h1_single,
                         f"HEAD 1 single-t (зоны до multi-t)"),
                        (verts,     faces,     col_h1_multi,
                         f"HEAD 1 multi-t T={mt_n_times} ({pct_h1:.1f}% изм.) → используется"),
                        (verts_fbx, faces_fbx, col_fbx_single,
                         f"FBX single-t (зоны до multi-t)"),
                        (verts_fbx, faces_fbx, col_fbx_multi,
                         f"FBX multi-t T={mt_n_times} ({pct_fbx:.1f}% изм.) → используется"),
                    ],
                    gap_factor=1.25,
                    window_title=f"MULTI-T ZONES: эти зоны используются в clustering+matching+transfer  Q→продолжить",
                )
                print(f"  → окно MULTI-T ZONES закрыто")
            except Exception as e:
                print(f"  ⚠ Не удалось открыть окно multi-t zones: {e}")
                import traceback; traceback.print_exc()

        # ── Выбор стратегии разбиения target меша (v5: 4 mode) ──────────────
        src_flat = [cl for cls in clusters_per_anchor for cl in cls]
        assign_mode = params.get('assign_mode', 'heat_zone_xyz')

        # Anchor positions для anchor-relative offset (фикс v3)
        anchor_pos_source = verts[src]
        anchor_pos_target = verts_fbx[src_fbx]

        # Общие параметры alignment (для всех 4 режимов)
        rigid = bool(params.get('heat_zone_rigid', True))
        ls_iters = int(params.get('heat_zone_smooth', 2))
        show_alignment = bool(params.get('heat_zone_show_viz', True))
        align_mode = params.get('heat_zone_alignment_mode', 'scale')
        nr_iters = int(params.get('heat_zone_non_rigid_iters', 2))
        nr_smooth = float(params.get('heat_zone_non_rigid_smoothing', 0.01))
        use_anchor = bool(params.get('heat_zone_use_anchor_align', True))
        use_rot = bool(params.get('heat_zone_use_rotation', False))
        # Hard-partition зоны (без перекрытий). Авто-ON при multi-t.
        hard_zones = bool(params.get('heat_zone_hard_partition',
                                       bool(params.get('multi_t_enable', False))))
        # Multi-t auto-override: hard zones + non_rigid alignment
        if bool(params.get('multi_t_enable', False)):
            if not hard_zones:
                print(f"  ℹ multi-t enabled → автоматически включаю hard-partition zones")
                hard_zones = True
            if align_mode != 'non_rigid':
                print(f"  ℹ multi-t enabled → автоматически переключаю alignment_mode "
                      f"'{align_mode}' → 'non_rigid' (per-zone non-rigid alignment)")
                align_mode = 'non_rigid'
        anchor_idx_src = list(np.asarray(src).ravel())
        anchor_idx_tgt = list(np.asarray(src_fbx).ravel())

        if assign_mode == 'heat_zone_xyz':
            print(f"\n  HEAT-ZONE XYZ matching (mode={align_mode}, scale={rigid}, "
                  f"smooth={ls_iters}, hard_zones={hard_zones})")
            result = assign_target_to_source_by_heat_zone(
                verts_target=verts_fbx, faces_target=faces_fbx,
                verts_source=verts,
                heat_target_per_anchor=heat_fbx,
                heat_source_per_anchor=heat,
                src_clusters_list=src_flat,
                heat_threshold=heat_thresh,
                rigid_align=rigid, n_icp_iters=0,
                label_smooth_iters=ls_iters,
                collect_alignment_data=show_alignment,
                faces_source=faces,
                alignment_mode=align_mode,
                non_rigid_iters=nr_iters,
                non_rigid_smoothing=nr_smooth,
                anchor_verts_source=anchor_idx_src,
                anchor_verts_target=anchor_idx_tgt,
                use_anchor_align=use_anchor,
                use_rotation=use_rot,
                hard_partition_zones=hard_zones,
            )
            if show_alignment:
                target_clusters, alignment_data = result
            else:
                target_clusters = result
                alignment_data = None

        elif assign_mode == 'zonal_1d':
            print(f"\n  ZONAL-1D matching (variant B: hard argmax partition)")
            res = assign_target_to_source_zonal_1d(
                verts_fbx, faces_fbx, verts, faces,
                heat_fbx, heat, src_flat,
                heat_threshold=heat_thresh,
                label_smooth_iters=ls_iters,
                alignment_mode=align_mode,
                non_rigid_iters=nr_iters,
                non_rigid_smoothing=nr_smooth,
                anchor_verts_source=anchor_idx_src,
                anchor_verts_target=anchor_idx_tgt,
                use_anchor_align=use_anchor,
                use_rotation=use_rot,
                collect_alignment_data=show_alignment,
            )
            if show_alignment and isinstance(res, tuple):
                target_clusters, alignment_data = res
            else:
                target_clusters = res; alignment_data = None

        elif assign_mode == 'sequential_anchor':
            order = params.get('sequential_anchor_order', 'by_max_heat')
            print(f"\n  SEQUENTIAL ANCHOR matching (variant C: order={order})")
            target_clusters = assign_target_to_source_sequential_anchor(
                verts_fbx, faces_fbx, verts, faces,
                heat_fbx, heat, src_flat,
                heat_threshold=heat_thresh,
                label_smooth_iters=ls_iters,
                alignment_mode=align_mode,
                non_rigid_iters=nr_iters,
                non_rigid_smoothing=nr_smooth,
                anchor_verts_source=anchor_idx_src,
                anchor_verts_target=anchor_idx_tgt,
                use_anchor_align=use_anchor,
                use_rotation=use_rot,
                anchor_order=order,
            )
            alignment_data = None

        elif assign_mode == 'decorr_heat':
            print(f"\n  DECORR-HEAT matching (variant D: Gram-Schmidt orthogonalization)")
            target_clusters = assign_target_to_source_decorr_heat(
                verts_fbx, faces_fbx, verts, faces,
                heat_fbx, heat, src_flat,
                heat_threshold=heat_thresh,
                label_smooth_iters=ls_iters,
                alignment_mode=align_mode,
                non_rigid_iters=nr_iters,
                non_rigid_smoothing=nr_smooth,
                anchor_verts_source=anchor_idx_src,
                anchor_verts_target=anchor_idx_tgt,
                use_anchor_align=use_anchor,
                use_rotation=use_rot,
            )
            alignment_data = None

        elif assign_mode == 'ring_match':
            ring_tol = float(params.get('ring_heat_tolerance', 0.05))
            dir_w    = float(params.get('ring_direction_weight', 1.0))
            print(f"\n  RING-MATCH matching (variant E: polar heat-rings + direction)")
            target_clusters = assign_target_to_source_ring_match(
                verts_fbx, faces_fbx, verts, faces,
                heat_fbx, heat, src_flat,
                anchor_verts_source=anchor_idx_src,
                anchor_verts_target=anchor_idx_tgt,
                heat_threshold=heat_thresh,
                heat_tolerance=ring_tol,
                direction_weight=dir_w,
                label_smooth_iters=ls_iters,
            )
            alignment_data = None

        elif assign_mode == 'tps_global':
            tps_smoothing = float(params.get('tps_smoothing', 0.001))
            tps_kernel    = params.get('tps_kernel', 'thin_plate_spline')
            print(f"\n  TPS-GLOBAL matching (variant F: anchor'ы = control points "
                  f"одного RBF, kernel={tps_kernel}, smoothing={tps_smoothing})")
            target_clusters = assign_target_to_source_tps_global(
                verts_fbx, faces_fbx, verts, faces,
                src_flat,
                anchor_verts_source=anchor_idx_src,
                anchor_verts_target=anchor_idx_tgt,
                rbf_smoothing=tps_smoothing,
                rbf_kernel=tps_kernel,
                label_smooth_iters=ls_iters,
            )
            alignment_data = None

        elif assign_mode == 'direct_copy':
            dc_smoothing = float(params.get('direct_copy_smoothing', 0.001))
            dc_scale     = params.get('direct_copy_scale_mode', 'bbox')
            print(f"\n  DIRECT-COPY matching (variant G: skip polar decomp, "
                  f"copy δ_FLAME[v*] напрямую, scale_mode={dc_scale})")
            target_clusters = assign_target_to_source_direct_copy(
                verts_fbx, faces_fbx, verts, faces,
                src_flat,
                delta_source=delta_native,
                anchor_verts_source=anchor_idx_src,
                anchor_verts_target=anchor_idx_tgt,
                rbf_smoothing=dc_smoothing,
                rbf_kernel='thin_plate_spline',
                scale_mode=dc_scale,
                label_smooth_iters=ls_iters,
            )
            alignment_data = None

        elif assign_mode == 'flame_fit':
            ff_n_betas = int(params.get('flame_fit_n_betas', 100))
            ff_iters   = int(params.get('flame_fit_iters', 30))
            ff_lr      = float(params.get('flame_fit_lr', 0.5))
            ff_reg     = float(params.get('flame_fit_reg', 0.001))
            ff_show_viz = bool(params.get('flame_fit_show_viz', True))
            print(f"\n  FLAME-FIT matching (variant H: ICP fit FLAME shape betas, "
                  f"NN correspondence)")
            target_clusters = assign_target_to_source_flame_fit(
                verts_fbx, faces_fbx, faces_source=faces,
                v_template_flame=v_t, shapedirs_flame=sd,
                src_clusters_list=src_flat,
                n_betas=ff_n_betas, fit_iters=ff_iters,
                learning_rate=ff_lr, beta_reg=ff_reg,
                label_smooth_iters=ls_iters,
                show_fit_viz=ff_show_viz,
                verbose=True,
            )
            alignment_data = None

        else:
            raise ValueError(f"Unknown assign_mode: {assign_mode}. v5 supports: "
                              f"heat_zone_xyz / zonal_1d / sequential_anchor / "
                              f"decorr_heat / ring_match / tps_global / direct_copy / "
                              f"flame_fit")

        # ── 3D Visualization of zone alignment (если показ включён) ─────────
        if show_alignment and alignment_data:
            print(f"\n  >>> ОКНО ZONE-ALIGNMENT: {len(alignment_data)} зон  "
                  f"(каждая пара raw цветом anchor'а) Q→продолжить <<<")
            print(f"      ВНИМАНИЕ: для каждой пары zone цвет одинаковый — "
                  f"src ярче, tgt чуть тусклее. Anchor i на FLAME должен быть "
                  f"анатомически тем же что anchor i на FBX!")
            # Печатаем reference таблицу anchor'ов: на каких world-позициях они
            # на каждом меше — пользователь может проверить пары визуально
            print(f"\n      Проверка пар anchor'ов (i: FLAME_vert@xyz  vs  FBX_vert@xyz):")
            for i in range(min(N_anchors, len(anchor_idx_src))):
                fp = verts[int(anchor_idx_src[i])]
                tp = verts_fbx[int(anchor_idx_tgt[i])]
                print(f"        anchor {i}: FLAME[{anchor_idx_src[i]}] = "
                      f"({fp[0]:+.3f}, {fp[1]:+.3f}, {fp[2]:+.3f})  |  "
                      f"FBX[{anchor_idx_tgt[i]}] = "
                      f"({tp[0]:+.3f}, {tp[1]:+.3f}, {tp[2]:+.3f})")
            print(f"      Если эти пары НЕ соответствуют анатомически — порядок "
                  f"выбора anchor'ов нарушен. Пересними anchor'ы в том же порядке "
                  f"на обоих мешах.\n")

            try:
                geoms = []
                x_cursor = 0.0
                palette_align = make_cluster_palette(max(N_anchors, len(alignment_data)))

                for ad in alignment_data:
                    a_idx = ad['anchor_idx']
                    base_col = palette_align[a_idx % len(palette_align)]
                    COLOR_SRC = np.clip(base_col * 0.85, 0, 1)       # ярче
                    COLOR_TGT = np.clip(base_col * 0.55 + 0.25, 0, 1) # светлее (mixed white)

                    all_pts = np.vstack([ad['P_src_aligned'], ad['P_tgt_centered']])
                    width = (all_pts.max(0) - all_pts.min(0))[0] + 1e-6

                    src_pts = ad['P_src_aligned'].copy(); src_pts[:,0] += x_cursor
                    sf = ad.get('src_faces_local')
                    if sf is not None and len(sf) > 0:
                        m = o3d.geometry.TriangleMesh(
                            o3d.utility.Vector3dVector(src_pts),
                            o3d.utility.Vector3iVector(sf))
                        m.compute_vertex_normals()
                        m.paint_uniform_color(COLOR_SRC.tolist())
                        geoms.append(m)
                    tgt_pts = ad['P_tgt_centered'].copy(); tgt_pts[:,0] += x_cursor
                    tf = ad.get('tgt_faces_local')
                    if tf is not None and len(tf) > 0:
                        m = o3d.geometry.TriangleMesh(
                            o3d.utility.Vector3dVector(tgt_pts),
                            o3d.utility.Vector3iVector(tf))
                        m.compute_vertex_normals()
                        m.paint_uniform_color(COLOR_TGT.tolist())
                        geoms.append(m)

                    # Большой шарик-маркер ANCHOR'а — в его палитра-цвете
                    sphere_anchor = o3d.geometry.TriangleMesh.create_sphere(
                        radius=max(width*0.04, 0.005))
                    sphere_anchor.compute_vertex_normals()
                    sphere_anchor.paint_uniform_color(base_col.tolist())
                    sphere_anchor.translate([x_cursor + width*0.5,
                                              all_pts.max(0)[1]*1.15, 0])
                    geoms.append(sphere_anchor)

                    # Маленький белый цифровой ID (через стопку шариков =
                    # биты a_idx + 1) НАД anchor sphere
                    n_pop = (a_idx + 1)  # 1..K
                    for k in range(n_pop):
                        s = o3d.geometry.TriangleMesh.create_sphere(
                            radius=max(width*0.012, 0.0015))
                        s.compute_vertex_normals()
                        s.paint_uniform_color([0.1, 0.1, 0.1])
                        s.translate([x_cursor + width*0.5 + k*width*0.04,
                                      all_pts.max(0)[1]*1.32, 0])
                        geoms.append(s)

                    print(f"    anchor {a_idx} [{n_pop} dots]: "
                          f"src {len(ad['P_src_aligned'])}v (color "
                          f"{[round(c,2) for c in COLOR_SRC]}), "
                          f"tgt {len(ad['P_tgt_centered'])}v (lighter)")
                    x_cursor += width * 1.3

                vis = o3d.visualization.Visualizer()
                if vis.create_window(
                        window_name=f"{assign_mode} alignment per anchor "
                                     f"(цвет=anchor_id, src ярче, dot count = anchor_id+1)  "
                                     f"Q→продолжить",
                        width=1800, height=900):
                    for g in geoms: vis.add_geometry(g)
                    opt = vis.get_render_option()
                    opt.mesh_show_back_face = True
                    opt.background_color = np.array([0.95,0.95,0.95])
                    vis.poll_events(); vis.update_renderer()
                    vis.run(); vis.destroy_window()
                    print(f"  → окно ZONE-ALIGNMENT закрыто")
            except Exception as e:
                print(f"  ⚠ Не удалось открыть окно alignment: {e}")
                import traceback; traceback.print_exc()

        print(f"  Получено {len(target_clusters)} target-кластеров из {len(src_flat)} source")
        # Диагностика: показать "отпечаток" assignment'а (сумма target-индексов на cluster)
        sig = 0
        for tc in target_clusters:
            sig = (sig * 31 + int(tc['target_indices'].sum())) % (10**9)
        print(f"  ASSIGN signature (для сравнения методов): {sig}")
        print(f"  Total target verts assigned: "
              f"{sum(len(tc['target_indices']) for tc in target_clusters)}")

        # ── Geodesic sanity-filter ────────────────────────────────────────────
        if bool(params.get('geo_filter_enable', True)):
            tol = float(params.get('geo_filter_tolerance', 1.2))
            print(f"\n  ── Geo-filter переноса (tolerance × source radius = {tol:.2f}) ──")
            print(f"    Строю edge-graph HEAD 1 для измерения source-radius'ов...")
            neighbors_src = build_vertex_adjacency(len(verts), verts, faces)
            before_total = sum(len(tc['target_indices']) for tc in target_clusters)
            target_clusters, n_removed = filter_target_clusters_by_geodesic_radius(
                target_clusters, verts_fbx, faces_fbx,
                neighbors_source=neighbors_src,
                tolerance_factor=tol,
            )
            after_total = sum(len(tc['target_indices']) for tc in target_clusters)
            print(f"    Выпилено {n_removed}/{before_total} вершин "
                  f"({100*n_removed/max(before_total,1):.1f}%) — "
                  f"осталось {after_total} в {len(target_clusters)} кластерах")
        else:
            print(f"  Geo-filter выключен (geo_filter_enable=False)")

        # ── DIAGNOSTIC: centroid offset diff (FBX heatmap) ───────────────────
        if bool(params.get('centroid_diff_diagnostic', False)):
            print(f"\n  ── DIAGNOSTIC: относительное смещение centroid'ов кластеров ──")
            per_vertex_diff, cluster_stats = compute_centroid_diff_diagnostic(
                target_clusters,
                anchor_pos_source=anchor_pos_source,
                anchor_pos_target=anchor_pos_target,
                verts_source=verts,
                verts_target=verts_fbx,
            )
            # Печатаем top-10 worst (худшие совпадения)
            sorted_stats = sorted(cluster_stats, key=lambda s: -s['diff_norm'])
            print(f"    Всего кластеров проверено: {len(cluster_stats)}")
            if cluster_stats:
                diffs = np.array([s['diff_norm'] for s in cluster_stats])
                print(f"    Diff stats: mean={diffs.mean()*100:.1f}%, "
                      f"median={np.median(diffs)*100:.1f}%, "
                      f"max={diffs.max()*100:.1f}%")
                print(f"    TOP-10 worst clusters (наибольший mismatch):")
                for i, s in enumerate(sorted_stats[:10]):
                    print(f"      [{i+1}] anchor={s['anchor_idx']} "
                          f"diff={s['diff_pct']:.1f}%, "
                          f"offset_src={[round(x,2) for x in s['offset_src']]}, "
                          f"offset_tgt={[round(x,2) for x in s['offset_tgt']]}")

            # Save dump
            try:
                save_matrix_csv(OUT_FBX / "centroid_diff_per_vertex.csv",
                                 per_vertex_diff[:, None], header="diff_norm")
                import json as _json
                (OUT_FBX / "centroid_diff_per_cluster.json").write_text(
                    _json.dumps([{k: v for k, v in s.items() if k != 'source'}
                                 for s in cluster_stats], indent=2))
                print(f"    → saved centroid_diff_per_vertex.csv + per_cluster.json")
            except Exception as e:
                print(f"    ⚠ не удалось сохранить: {e}")

            # ── Heatmap mesh: красный = большой mismatch, зелёный = маленький
            print(f"\n  >>> ОКНО CENTROID-DIFF heatmap (FBX) — Q→продолжить <<<")
            try:
                col_fbx_diff = np.tile([0.4, 0.4, 0.4], (len(verts_fbx), 1))  # серый
                if cluster_stats:
                    diffs = np.array([s['diff_norm'] for s in cluster_stats])
                    vmax = max(np.percentile(diffs, 95), 0.05)
                    for v_idx in range(len(verts_fbx)):
                        d = per_vertex_diff[v_idx]
                        if np.isnan(d):
                            continue
                        t = np.clip(d / vmax, 0, 1)
                        col_fbx_diff[v_idx] = [t, np.clip(1.5*(1-t), 0, 1), 0.1]

                show_meshes_side_by_side(
                    [
                        (verts_fbx, faces_fbx, col_fbx_diff,
                         f"FBX centroid-diff heatmap  "
                         f"(red=mismatch up to {vmax*100:.1f}%)"),
                    ],
                    gap_factor=1.3,
                    window_title=f"DIAGNOSTIC: relative centroid offset — Q→продолжить",
                )
                print(f"  → окно centroid-diff закрыто")
            except Exception as e:
                print(f"  ⚠ Не удалось открыть centroid-diff окно: {e}")
                import traceback; traceback.print_exc()

        # ── ОКНО 6a: разбиение FBX на кластеры (по той же палитре что HEAD 1) ──
        print("\n  ОКНО 6a: разбиение FBX на кластеры (цвет = source cluster)")
        vert_colors_fbx = np.tile([0.7, 0.7, 0.7], (len(verts_fbx), 1))   # серый default
        vert_weight_fbx = np.zeros(len(verts_fbx))
        for tc in target_clusters:
            col = cluster_color_map.get(id(tc['source']),
                                          np.array([0.5, 0.5, 0.5]))
            for j, v_idx in enumerate(tc['target_indices']):
                w = tc['target_heat'][j]
                if w > vert_weight_fbx[v_idx]:
                    vert_weight_fbx[v_idx] = w
                    vert_colors_fbx[v_idx] = col

        # Дополнительно — центроиды target-кластеров + стрелки source-μ
        fbx_extras = []
        for tc in target_clusters:
            col = cluster_color_map.get(id(tc['source']),
                                          np.array([0.5, 0.5, 0.5])).tolist()
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            sph.translate(tc['c_target']); sph.paint_uniform_color(col)
            sph.compute_vertex_normals(); fbx_extras.append(sph)
            # стрелка μ от source-кластера, нарисованная в позиции target-центроида
            p0 = tc['c_target']
            p1 = tc['c_target'] + tc['source']['mu'] * 3.0
            fbx_extras.append(make_arrow(p0, p1, color=[0, 0, 0]))

        # Anchor-точки на FBX (красные)
        fbx_src_extras = []
        for s in src_fbx:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
            sph.translate(verts_fbx[s]); sph.paint_uniform_color([1, 0, 0])
            sph.compute_vertex_normals(); fbx_src_extras.append(sph)

        # Сравнение бок-о-бок: HEAD 1 clusters | FBX clusters
        # (используем уже посчитанные vert_colors из ШАГа 8)
        print(f"  → открываю ОКНО 6a — закрой Q чтобы продолжить к деформации")
        show_meshes_side_by_side([
            (verts,     faces,     vert_colors,     "head1_clusters_rest"),
            (verts_fbx, faces_fbx, vert_colors_fbx, "fbx_clusters_rest"),
        ], extra_geometries=[arrows, fbx_extras + fbx_src_extras],
           window_title="ОКНО 6a: HEAD 1 clusters | FBX clusters (Q → продолжить)")
        print(f"  → ОКНО 6a закрыто, продолжаю...")

        # Сохраняем результат Voronoi-разбиения (с защитой от ошибок)
        try:
            save_target_clusters_json(
                OUT_FBX / "target_clusters.json", target_clusters,
                cluster_color_map=cluster_color_map)
            print(f"  → saved FBX target_clusters.json ({len(target_clusters)} clusters)")
        except Exception as e:
            print(f"  ⚠ Не удалось сохранить target_clusters.json: {e}")
            import traceback; traceback.print_exc()

        print(f"  → Применяю трансформации (apply_target_clusters_transfer)...")
        delta_fbx_raw = apply_target_clusters_transfer(verts_fbx, target_clusters)
        print(f"  max ||δ_fbx (raw)|| = {np.linalg.norm(delta_fbx_raw, axis=1).max():.4f}")

        try:
            save_matrix_csv(OUT_FBX / "delta_raw.csv", delta_fbx_raw, header="dx,dy,dz")
            print(f"  → saved FBX delta_raw.csv")
        except Exception as e:
            print(f"  ⚠ Не удалось сохранить delta_raw.csv: {e}")

        # Сглаживание на FBX (отдельный параметр, т.к. FBX обычно крупнее)
        # Адаптивно увеличиваем если меш сильно больше HEAD 1
        n_iter_fbx = smooth_iters_fbx
        ratio = len(verts_fbx) / max(len(verts), 1)
        if ratio > 1.5:
            scale = int(np.ceil(np.sqrt(ratio)))
            n_iter_fbx_auto = max(smooth_iters_fbx, smooth_iters * scale)
            if n_iter_fbx_auto > n_iter_fbx:
                print(f"  → FBX в {ratio:.1f}× крупнее HEAD 1 → auto-scale smooth iters: {n_iter_fbx} → {n_iter_fbx_auto}")
                n_iter_fbx = n_iter_fbx_auto
        if n_iter_fbx > 0:
            print(f"  → Сглаживание FBX ({n_iter_fbx} iters, α={smooth_alpha}) — может занять секунды...")
            max_raw = np.linalg.norm(delta_fbx_raw, axis=1).max()
            delta_fbx = smooth_delta(delta_fbx_raw, faces_fbx,
                                      n_iter=n_iter_fbx,
                                      alpha=smooth_alpha)
            max_sm = np.linalg.norm(delta_fbx, axis=1).max()
            print(f"  max ||δ_fbx (raw)||      = {max_raw:.4f}")
            print(f"  max ||δ_fbx (smoothed)|| = {max_sm:.4f}  (ослабление: {(1 - max_sm/max(max_raw,1e-9))*100:.1f}%)")
        else:
            print(f"  → Сглаживание FBX выключено (smooth_iters_fbx=0)")
            delta_fbx = delta_fbx_raw

        head_fbx_def = verts_fbx + delta_fbx
        col_fbx = to_colors(np.linalg.norm(delta_fbx, axis=1), CMAP_DISP)

        try:
            save_matrix_csv(OUT_FBX / "delta_smoothed.csv", delta_fbx, header="dx,dy,dz")
            save_matrix_csv(OUT_FBX / "verts_deformed.csv", head_fbx_def, header="x,y,z")
            print(f"  → saved FBX delta_smoothed.csv & verts_deformed.csv")
        except Exception as e:
            print(f"  ⚠ Не удалось сохранить delta CSV: {e}")

        try:
            save_metadata_json(OUT_DIR / "metadata.json", params, shape, expr,
                                N_anchors, src, src_fbx=src_fbx)
        except Exception as e:
            print(f"  ⚠ Не удалось сохранить metadata: {e}")

        # ОКНО 6: HEAD 1 (native blendshape) | FBX rest | FBX deformed
        print("\n  >>> ОКНО 6: HEAD 1 (native) | FBX rest | FBX deformed (Q → выход) <<<")
        print("  Открываю окно деформированных мешей...")
        try:
            show_meshes_side_by_side([
                (head_expr, faces, col_native, "head1_native"),
                (verts_fbx, faces_fbx, np.tile([0.85, 0.75, 0.68], (len(verts_fbx), 1)),
                 "fbx_rest"),
                (head_fbx_def, faces_fbx, col_fbx, "fbx_deformed"),
            ], window_title="ОКНО 6: HEAD 1 (native) | FBX rest | FBX deformed")
            print("  → ОКНО 6 закрыто")
        except Exception as e:
            print(f"  ⚠ Не удалось открыть ОКНО 6: {e}")
            import traceback; traceback.print_exc()

    print("\n✓ Pipeline завершён.")
    print(f"  Все данные сохранены в: {OUT_DIR}/")


if __name__ == "__main__":
    main()
