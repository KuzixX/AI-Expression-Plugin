"""
Motion-Groups Transfer Pipeline — VERSION 3.0  (locked 2026-05-25)

Debug pipeline для HEAD 1 (source FLAME) → HEAD 2 (target FBX) с GUI выбором
режима матчинга кластеров.

ШАГИ:
  1. Загружаем FLAME, выбираем shape preset
  2. Выбираем N anchor-точек (Shift+клик)
  3. Указываем время/шаги диффузии
  4. ОКНО 1: анимация диффузии (видим как тепло расползается)
  5. Применяем блендшейп → δ_native (правильная мимика)
  6. ОКНО 2: rest и deformed head (родная экспрессия)
  7. Кластеризуем зоны (motion-groups) + полярная декомпозиция
  8. ОКНО 3: head с раскраской по кластерам + стрелки μ
  9. Реконструируем δ из кластеров (линейная аппроксимация)
 10. ОКНО 4: rest | δ_native | δ_reconstructed (side-by-side)
 11. Сглаживаем δ Laplacian'ом
 12. ОКНО 5: δ_native | δ_reconstructed | δ_smoothed
 13. (если есть FBX) перенос кластеров → δ_fbx → отдельный smooth → ОКНО 6

РЕЖИМЫ МАТЧИНГА (9 шт):
  voronoi    — Dijkstra от anchor-relative seed (рабочая лошадка)
  hks        — Heat Kernel Signature (scale-invariant intrinsic)
  wks        — Wave Kernel Signature (band-pass)
  hybrid     — Voronoi + HKS (взвешенное расстояние)
  heat_vec   — K-мерный heat-fingerprint per vertex, match с cluster-профилями
  heat_align — ⭐ per-vertex correspondence через heat-space (k-NN + mesh smooth)
  heat_svd   — Joint SVD heat-матриц (общий базис) → r-мерные дескрипторы
  heat_rank  — Per-anchor percentile (shape-invariant)
  sinkhorn   — Optimal Transport (balanced)
  rbf        — Radial Basis Function interpolation (anchor warping)

КЛЮЧЕВЫЕ ПАРАМЕТРЫ:
  smooth_iters       — Laplacian iters для HEAD 1 (default 3)
  smooth_iters_fbx   — отдельно для FBX (default 30, авто-скейл по размеру)
  heat_align_knn     — k-NN voting в heat_align (default 5)
  heat_align_smooth  — mesh-graph label smoothing (default 2)
  n_svd              — компоненты для heat_svd (0 = auto)

DUMPS (в python/scripts/debug_output/run_*/):
  head1/heat.csv, clusters.json, clusters_flat.csv, delta_*.csv, verts_*.csv
  fbx/  heat.csv, target_clusters.json, delta_raw.csv, delta_smoothed.csv

СОПУТСТВУЮЩИЕ СКРИПТЫ:
  visualize_dumps.py    — батч-генерация графиков по дампам (matplotlib offline)
  align_heat_tables.py  — выравнивание heat-таблиц HEAD1 ↔ FBX (методы A, B)
"""

__version__ = "3.0"

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
                  position_weight=1.5, min_cluster_size=4):
    """K-means в [δ, position] для одной anchor-зоны."""
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

    n_clusters = max(2, min(n_clusters_max, len(active_idx) // 30))
    km = KMeans(n_clusters=n_clusters, n_init=8, random_state=0)
    labels = km.fit_predict(features)

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


def assign_target_to_source_clusters(verts_target, faces_target,
                                      heat_target_per_anchor,
                                      src_clusters_list,
                                      anchor_pos_source=None,
                                      anchor_pos_target=None,
                                      heat_threshold=0.05,
                                      geodesic_factor=3.0):
    """Voronoi через Dijkstra по поверхности + ANCHOR-RELATIVE seed (фикс v3).

    КРИТИЧНОЕ ИЗМЕНЕНИЕ vs v2:
      Раньше seed для Dijkstra искался через
      find_nearest_vertex(verts_target, source.c_rest) — xyz-центроид кластера
      HEAD 1 использовался ПРЯМО как координата на target. Это ломалось на
      FBX с другими пропорциями: тот же xyz → другая анатомия.

      Если переданы anchor_pos_source/anchor_pos_target:
        offset    = source.c_rest - anchor_pos_source[anchor_idx]
        target_xy = anchor_pos_target[anchor_idx] + offset
        seed      = find_nearest_vertex(verts_target, target_xy)
      Это работает корректно если anchor-точки выбраны анатомически идентично
      (что и так требуется пользователем при Shift+click).

    Иначе — fallback на абсолютные xyz (legacy, OK для FLAME→FLAME).
    """
    N_t = verts_target.shape[0]
    print(f"  Строю Dijkstra adjacency на target меше ({N_t} верт)...")
    neighbors = build_vertex_adjacency(N_t, verts_target, faces_target)

    use_anchor_relative = (anchor_pos_source is not None and
                            anchor_pos_target is not None)
    if use_anchor_relative:
        print("  Mode: ANCHOR-RELATIVE seed (фикс для cross-mesh пропорций)")
    else:
        print("  Mode: абсолютный xyz centroid (legacy)")

    target_clusters = []
    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    for a, src_list in src_by_anchor.items():
        heat_a = heat_target_per_anchor[a]
        heat_max = max(heat_a.max(), 1e-12)
        active = heat_a > heat_threshold * heat_max
        active_idx = np.where(active)[0]
        if not len(active_idx): continue

        K = len(src_list)
        geo_dists = np.full((N_t, K), np.inf)
        for k, s in enumerate(src_list):
            if use_anchor_relative:
                offset = s['c_rest'] - anchor_pos_source[s['anchor_idx']]
                target_xyz = anchor_pos_target[s['anchor_idx']] + offset
            else:
                target_xyz = s['c_rest']
            seed = find_nearest_vertex(verts_target, target_xyz)
            R_max = s['spatial_sigma'] * geodesic_factor
            if not np.isfinite(R_max): R_max = float('inf')
            geo_dists[:, k] = geodesic_dijkstra(neighbors, seed, R_max)

        dists_active = geo_dists[active_idx]
        nearest = np.argmin(dists_active, axis=1)
        reachable = ~np.all(np.isinf(dists_active), axis=1)

        for k, s in enumerate(src_list):
            mask = (nearest == k) & reachable
            if not mask.any(): continue
            t_indices = active_idx[mask]
            t_heat = heat_a[t_indices]
            W = max(t_heat.sum(), 1e-12)
            c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
            target_clusters.append({
                'source': s,
                'target_indices': t_indices,
                'target_heat': t_heat,
                'c_target': c_target,
                'geo_dists': dists_active[mask, k],
            })
    return target_clusters


def compute_heat_percentiles(heat_per_anchor, active_only=True, threshold=0.05):
    """Percentile ranks в [0, 1] per-anchor.

    active_only=True (default, фикс v3.1):
        Ranks считаются ТОЛЬКО среди active вершин (heat > threshold·max).
        Неактивные вершины получают rank = 0 (фиксированное, не шум).
        Это устраняет проблему: на 18000 вершин 17800 имеют heat ≈ 0,
        и их ranks распределяются 0..0.99 по численному шуму → загрязняют профили.

    active_only=False: ranks по всем вершинам (legacy, шумно).
    """
    K, N = heat_per_anchor.shape
    percentiles = np.zeros_like(heat_per_anchor, dtype=np.float64)
    for a in range(K):
        h = heat_per_anchor[a]
        if active_only:
            h_max = max(h.max(), 1e-12)
            active_mask = h > threshold * h_max
            active_idx = np.where(active_mask)[0]
            if len(active_idx) < 2: continue
            # Ranks только внутри active зоны
            sort_idx = np.argsort(h[active_idx])     # ascending
            ranks_within = np.empty(len(active_idx), dtype=np.float64)
            ranks_within[sort_idx] = np.arange(len(active_idx)) / max(len(active_idx) - 1, 1)
            percentiles[a, active_idx] = ranks_within
            # Остальные вершины: rank = 0 (по умолчанию из np.zeros_like)
        else:
            sort_idx = np.argsort(h)
            ranks = np.empty(N, dtype=np.float64)
            ranks[sort_idx] = np.arange(N) / max(N - 1, 1)
            percentiles[a] = ranks
    return percentiles


def assign_target_to_source_by_heat_rank(
        verts_target, heat_target_per_anchor, src_clusters_list,
        heat_source_per_anchor,
        heat_threshold=0.05, normalize=True):
    """HEAT-RANK MATCHING — invariant к shape абсолютных heat-значений.

    Ключевая идея (под вопрос пользователя):
      Heat decay shapes похожи между мешами разной топологии, но абсолютные
      значения (max) различаются из-за плотности вершин. Если конвертировать
      heat в percentile rank per-anchor, обе меши получают одинаковые
      [0, 1] распределения → матчинг становится возможен по rank-vector.

    Per vertex:  rank_vec[v] = (rank в anchor 0, rank в anchor 1, ..., rank в anchor K-1)
    Per cluster: profile = heat-weighted mean rank_vec над c.indices
    Assignment:  argmin ||rank_vec_target[v] - profile||²

    Anatomical correspondence через rank даёт максимальную устойчивость
    к разности mesh density, при сохранённой форме heat decay.
    """
    perc_t = compute_heat_percentiles(heat_target_per_anchor)         # (K, N_t)
    perc_s = compute_heat_percentiles(heat_source_per_anchor)         # (K, N_s)

    rv_target = perc_t.T                                              # (N_t, K)
    rv_source = perc_s.T                                              # (N_s, K)

    if normalize:
        rv_target_n = normalize_signature(rv_target)
        rv_source_n = normalize_signature(rv_source)
    else:
        rv_target_n = rv_target
        rv_source_n = rv_source

    target_clusters = []
    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    for a, src_list in src_by_anchor.items():
        heat_a = heat_target_per_anchor[a]
        heat_max = max(heat_a.max(), 1e-12)
        active = heat_a > heat_threshold * heat_max
        active_idx = np.where(active)[0]
        if not len(active_idx): continue

        # Per-cluster rank-profile
        profiles = []
        for s in src_list:
            w = s['heat_weights']
            W = max(w.sum(), 1e-12)
            prof = (w[:, None] * rv_source_n[s['indices']]).sum(0) / W
            profiles.append(prof)
        profiles = np.stack(profiles)
        if normalize:
            profiles = normalize_signature(profiles)

        active_feat = rv_target_n[active_idx]
        dists = ((active_feat[:, None, :] - profiles[None, :, :]) ** 2).sum(-1)
        nearest = np.argmin(dists, axis=1)

        for k, s in enumerate(src_list):
            mask = nearest == k
            if not mask.any(): continue
            t_indices = active_idx[mask]
            t_heat = heat_a[t_indices]
            W = max(t_heat.sum(), 1e-12)
            c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
            target_clusters.append({
                'source': s,
                'target_indices': t_indices,
                'target_heat': t_heat,
                'c_target': c_target,
                'rank_dist': float(dists[mask, k].mean()),
            })
    return target_clusters


def assign_target_to_source_by_heat_vector(
        verts_target, heat_target_per_anchor, src_clusters_list,
        heat_source_per_anchor,
        heat_threshold=0.05, normalize=True, per_anchor_max_norm=True):
    """A. HEAT-VECTOR MATCHING — каждой вершине свой N-мерный "адрес" по heat
    от всех anchor-точек.

    ФИКС v3.1 — per_anchor_max_norm=True (default):
        Сначала heat per-anchor нормируется на свой max → heat[a,v] / max(heat[a])
        Это делает heat в [0,1] per anchor → comparable между мешами с разной
        плотностью сетки (max heat зависит от mesh density).
        Без этого holost heat absolute = разный → matching ломается.

    После per-anchor нормировки → формируем per-vertex K-мерный вектор →
    L2-нормируем per vertex для cosine-like matching.
    """
    heat_source = heat_source_per_anchor.copy()
    heat_target = heat_target_per_anchor.copy()

    if per_anchor_max_norm:
        # Per-anchor max normalize: heat[a,v] / max(heat[a]) → [0, 1]
        h_max_s = heat_source.max(axis=1, keepdims=True).clip(min=1e-12)
        h_max_t = heat_target.max(axis=1, keepdims=True).clip(min=1e-12)
        heat_source = heat_source / h_max_s
        heat_target = heat_target / h_max_t

    # h_vec — per-vertex N_anchors-мерный вектор
    h_vec_target = heat_target.T                                     # (N_t, K_anchors)
    h_vec_source = heat_source.T                                     # (N_s, K_anchors)

    if normalize:
        h_vec_target_n = normalize_signature(h_vec_target)
        h_vec_source_n = normalize_signature(h_vec_source)
    else:
        h_vec_target_n = h_vec_target
        h_vec_source_n = h_vec_source

    target_clusters = []
    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    for a, src_list in src_by_anchor.items():
        heat_a = heat_target_per_anchor[a]
        heat_max = max(heat_a.max(), 1e-12)
        active = heat_a > heat_threshold * heat_max
        active_idx = np.where(active)[0]
        if not len(active_idx): continue

        # Cluster profiles в heat-vector space
        profiles = []
        for s in src_list:
            w = s['heat_weights']
            W = max(w.sum(), 1e-12)
            prof = (w[:, None] * h_vec_source_n[s['indices']]).sum(0) / W
            profiles.append(prof)
        profiles = np.stack(profiles)                                # (K, N_anchors)
        if normalize:
            profiles = normalize_signature(profiles)

        active_feat = h_vec_target_n[active_idx]                     # (M, N_anchors)
        dists = ((active_feat[:, None, :] - profiles[None, :, :]) ** 2).sum(-1)
        nearest = np.argmin(dists, axis=1)

        for k, s in enumerate(src_list):
            mask = nearest == k
            if not mask.any(): continue
            t_indices = active_idx[mask]
            t_heat = heat_a[t_indices]
            W = max(t_heat.sum(), 1e-12)
            c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
            target_clusters.append({
                'source': s,
                'target_indices': t_indices,
                'target_heat': t_heat,
                'c_target': c_target,
                'heat_vec_dist': float(dists[mask, k].mean()),
            })
    return target_clusters


def _build_vertex_adjacency(N, faces):
    """1-ring adjacency list для меша. Возвращает list[set[int]]."""
    adj = [set() for _ in range(N)]
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].update((b, c)); adj[b].update((a, c)); adj[c].update((a, b))
    return adj


def _smooth_labels_on_mesh(labels, adj, n_iter=2):
    """Majority-vote по 1-ring соседям на mesh-графе. labels: dict {vert: cluster_id}.
    Возвращает обновлённый dict (только для вершин, которые есть в labels)."""
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
            # выбираем самый частый
            new[v] = max(votes.items(), key=lambda kv: kv[1])[0]
        cur = new
    return cur


def assign_target_to_source_by_heat_align(
        verts_target, heat_target_per_anchor, src_clusters_list,
        heat_source_per_anchor,
        heat_threshold=0.05,
        k_nn=5, label_smooth_iters=2, faces_target=None):
    """A3. HEAT-ALIGN MATCHING — per-vertex correspondence через heat-space.

    Концепция:
        1. Per-anchor max-normalize обеих heat-матриц
        2. Для каждой target вершины j (в зоне anchor a) находим k_nn ближайших
           source-вершин i₁..iₖ (в зоне a) в K-мерном heat-space
        3. Голосуем за cluster label с весами 1/d² → берём кластер-победитель
           (k_nn=1 → классический argmin; больше → подавление шума)
        4. (опц.) post-smoothing labels на mesh-графе FBX через majority vote
           1-ring соседей — убирает спекл-шум на границах кластеров
        5. Группируем target вершины по cluster

    Отличие от heat_vec:
        heat_vec   — сравнивает target-вершину с heat-weighted СРЕДНИМ профилем
                     каждого кластера → могут быть размытые границы
        heat_align — сравнивает target-вершину с КАЖДОЙ source-вершиной отдельно,
                     наследует label ближайшей → точнее, плотнее

    Это прямое использование "канонической корреспонденции" из align_heat_tables.py
    """
    heat_source = heat_source_per_anchor.copy().astype(np.float64)
    heat_target = heat_target_per_anchor.copy().astype(np.float64)
    K, N_s = heat_source.shape
    _, N_t = heat_target.shape

    # Per-anchor max-norm — обязательно для cross-mesh
    h_max_s = heat_source.max(axis=1, keepdims=True).clip(min=1e-12)
    h_max_t = heat_target.max(axis=1, keepdims=True).clip(min=1e-12)
    heat_source_n = heat_source / h_max_s                              # (K, N_s)
    heat_target_n = heat_target / h_max_t                              # (K, N_t)

    # Per-vertex K-мерные heat-векторы (транспонируем)
    H_s = heat_source_n.T                                              # (N_s, K)
    H_t = heat_target_n.T                                              # (N_t, K)

    # Reverse lookup: для каждой FLAME-вершины i → её cluster (с привязкой к anchor)
    # vertex_to_cluster[i] = (cluster_obj, anchor_idx)
    vertex_to_cluster = {}
    for s in src_clusters_list:
        a = s['anchor_idx']
        for v_idx in s['indices']:
            vertex_to_cluster[int(v_idx)] = (s, a)

    # Группируем source по anchor (для зональной фильтрации)
    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    # ── Основной цикл по anchor'ам ────────────────────────────────────────────
    # Для каждой active target-вершины ищем nearest source-вершину
    # в той же anchor-зоне
    fbx_to_cluster = {}                                                # {fbx_v: src_cluster_obj}
    n_total_aligned = 0

    for a, src_list in src_by_anchor.items():
        # Source: все вершины принадлежащие кластерам этого anchor'а
        src_vert_idx_set = set()
        for s in src_list:
            src_vert_idx_set.update(int(v) for v in s['indices'])
        src_vert_idx = np.array(sorted(src_vert_idx_set), dtype=np.int64)
        if not len(src_vert_idx): continue

        # Target: active в зоне anchor a
        heat_a_t = heat_target_per_anchor[a]
        h_max = max(heat_a_t.max(), 1e-12)
        active_t = heat_a_t > heat_threshold * h_max
        active_t_idx = np.where(active_t)[0]
        if not len(active_t_idx): continue

        # Pairwise distance: target_active × source_clustered (in K-space)
        H_t_zone = H_t[active_t_idx]                                   # (M, K)
        H_s_zone = H_s[src_vert_idx]                                   # (S, K)

        # || a - b ||² = ||a||² + ||b||² - 2 a·b
        a_sq = (H_t_zone ** 2).sum(1, keepdims=True)
        b_sq = (H_s_zone ** 2).sum(1, keepdims=True).T
        cross = H_t_zone @ H_s_zone.T
        D2 = np.maximum(a_sq + b_sq - 2 * cross, 0)

        # k-NN majority vote (вместо одного argmin → подавляет случайные перескоки)
        k_eff = min(k_nn, D2.shape[1])
        if k_eff <= 1:
            # fallback на argmin
            nn = np.argmin(D2, axis=1)
            for t_v, s_idx in zip(active_t_idx, nn):
                s_v = int(src_vert_idx[s_idx])
                cl_obj, _ = vertex_to_cluster.get(s_v, (None, None))
                if cl_obj is not None:
                    fbx_to_cluster[int(t_v)] = cl_obj
                    n_total_aligned += 1
        else:
            # топ-k индексов источников per target-вершина
            topk_idx = np.argpartition(D2, k_eff - 1, axis=1)[:, :k_eff]   # (M, k)
            topk_d2 = np.take_along_axis(D2, topk_idx, axis=1)             # (M, k)
            # веса = 1 / (d² + eps) — голос пропорционален близости
            weights = 1.0 / (topk_d2 + 1e-9)

            for m in range(len(active_t_idx)):
                # голосование за cluster-id с весами
                votes = {}
                for j in range(k_eff):
                    s_v = int(src_vert_idx[topk_idx[m, j]])
                    cl_obj, _ = vertex_to_cluster.get(s_v, (None, None))
                    if cl_obj is None: continue
                    key = id(cl_obj)
                    if key in votes:
                        votes[key] = (votes[key][0] + weights[m, j], cl_obj)
                    else:
                        votes[key] = (weights[m, j], cl_obj)
                if not votes: continue
                # выбираем кластер с максимальной суммой весов
                best_cl = max(votes.values(), key=lambda x: x[0])[1]
                fbx_to_cluster[int(active_t_idx[m])] = best_cl
                n_total_aligned += 1

    # ── Post-smoothing labels на mesh-графе FBX (убирает спекл-шум на границах)
    if label_smooth_iters > 0 and faces_target is not None and len(fbx_to_cluster) > 0:
        # labels как dict {vert: cluster_obj} → переводим к {vert: id} для majority vote
        id_to_cl = {id(cl): cl for cl in fbx_to_cluster.values()}
        labels_id = {v: id(cl) for v, cl in fbx_to_cluster.items()}
        adj = _build_vertex_adjacency(len(verts_target), faces_target)
        labels_id = _smooth_labels_on_mesh(labels_id, adj, n_iter=label_smooth_iters)
        # обратно
        fbx_to_cluster = {v: id_to_cl[lid] for v, lid in labels_id.items()}
        print(f"    [heat_align] post-smoothing labels ({label_smooth_iters} iters majority-vote)")

    # ── Группируем target-вершины по cluster ──────────────────────────────────
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

    print(f"    [heat_align] {n_total_aligned} per-vertex correspondences "
          f"(k_nn={k_nn}, smooth={label_smooth_iters}) → "
          f"{len(target_clusters)} target clusters")
    return target_clusters


def assign_target_to_source_by_heat_svd(
        verts_target, heat_target_per_anchor, src_clusters_list,
        heat_source_per_anchor,
        heat_threshold=0.05, n_components=0, normalize=True):
    """A2. HEAT-SVD MATCHING — совместный SVD heat-матриц обоих мешей.

    Идея: heat_source (K, N_s) и heat_target (K, N_t) делят общую размерность
    K (anchor'ы парные). Стэкуем горизонтально:
        H = [H_s | H_t]   shape (K, N_s + N_t)
    SVD: H = U · Σ · V^T,  где
        U ∈ R^{K×r}        — общий "anchor-basis"
        V ∈ R^{(N_s+N_t)×r} — vertex descriptors В ОДНОМ И ТОМ ЖЕ базисе
    Разрезаем V обратно: V_s = V[:N_s], V_t = V[N_s:].

    Эти r-мерные descriptors напрямую сравнимы (один базис), без Procrustes.
    Дальше — heat_vec-style nearest-neighbor matching на r-мерных векторах.

    Плюсы vs heat_vec:
        - Шумоподавление (отбрасываем малые сингулярные значения)
        - Компактнее (r << K)
        - Учитывает корреляции между anchor'ами

    Args:
        n_components: 0 → auto (min(K, 16)); иначе фиксированное число компонент.
    """
    heat_source = heat_source_per_anchor.copy().astype(np.float64)
    heat_target = heat_target_per_anchor.copy().astype(np.float64)
    K, N_s = heat_source.shape
    _, N_t = heat_target.shape

    # Per-anchor max-normalization (как в heat_vec) — выравнивает абсолютные шкалы
    h_max_s = heat_source.max(axis=1, keepdims=True).clip(min=1e-12)
    h_max_t = heat_target.max(axis=1, keepdims=True).clip(min=1e-12)
    heat_source /= h_max_s
    heat_target /= h_max_t

    # Стэкуем по второй оси: (K, N_s + N_t)
    H = np.concatenate([heat_source, heat_target], axis=1)

    # Совместный SVD
    r = n_components if n_components > 0 else min(K, 16)
    r = min(r, K, N_s + N_t)

    # Полный SVD маленькой матрицы (K мало, обычно <=20)
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    # Берём top-r
    U_r  = U[:, :r]                                                   # (K, r)
    S_r  = S[:r]                                                      # (r,)
    Vt_r = Vt[:r, :]                                                  # (r, N_s+N_t)

    # Vertex descriptors — взвешиваем сингулярными значениями,
    # чтобы более "важные" модусы доминировали в L2-расстояниях
    V_weighted = (Vt_r * S_r[:, None]).T                              # (N_s+N_t, r)
    V_s = V_weighted[:N_s]                                            # (N_s, r)
    V_t = V_weighted[N_s:]                                            # (N_t, r)

    if normalize:
        V_s = normalize_signature(V_s)
        V_t = normalize_signature(V_t)

    energy_kept = (S_r**2).sum() / max((S**2).sum(), 1e-12)
    print(f"    [heat_svd] r={r}/{len(S)} components, "
          f"energy kept = {energy_kept*100:.1f}%, "
          f"singular values: {[f'{s:.3g}' for s in S_r[:min(r,8)]]}")

    target_clusters = []
    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    for a, src_list in src_by_anchor.items():
        heat_a = heat_target_per_anchor[a]
        heat_max = max(heat_a.max(), 1e-12)
        active = heat_a > heat_threshold * heat_max
        active_idx = np.where(active)[0]
        if not len(active_idx): continue

        # Cluster profiles в r-мерном SVD-пространстве
        profiles = []
        for s in src_list:
            w = s['heat_weights']
            W = max(w.sum(), 1e-12)
            prof = (w[:, None] * V_s[s['indices']]).sum(0) / W
            profiles.append(prof)
        profiles = np.stack(profiles)                                 # (K_clusters, r)
        if normalize:
            profiles = normalize_signature(profiles)

        active_feat = V_t[active_idx]                                 # (M, r)
        dists = ((active_feat[:, None, :] - profiles[None, :, :]) ** 2).sum(-1)
        nearest = np.argmin(dists, axis=1)

        for k, s in enumerate(src_list):
            mask = nearest == k
            if not mask.any(): continue
            t_indices = active_idx[mask]
            t_heat = heat_a[t_indices]
            W = max(t_heat.sum(), 1e-12)
            c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
            target_clusters.append({
                'source': s,
                'target_indices': t_indices,
                'target_heat': t_heat,
                'c_target': c_target,
                'svd_dist': float(dists[mask, k].mean()),
            })
    return target_clusters


def assign_target_to_source_by_sinkhorn(
        verts_target, heat_target_per_anchor, src_clusters_list,
        heat_source_per_anchor,
        heat_threshold=0.05, epsilon=0.05, n_iter=200):
    """B. SINKHORN OPTIMAL TRANSPORT — balanced assignment.

    Для каждой anchor-зоны:
      - Source: K кластеров с marginal = sum of heat_weights
      - Target: M active вершин с marginal = heat values
      - Cost: distance в heat-vector space
      - Solve regularized OT (Sinkhorn): transport plan T[k, j]
      - Assignment: argmax_k T[k, j] для каждого target j

    Главное свойство: КАЖДЫЙ source кластер получает target вершин
    ПРОПОРЦИОНАЛЬНО своему размеру. Лечит проблему "большие кластеры
    съели все маленькие".

    epsilon: регуляризация (меньше → острее распределение, дороже вычисления)
    """
    h_vec_target = heat_target_per_anchor.T
    h_vec_source = heat_source_per_anchor.T

    target_clusters = []
    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    for a, src_list in src_by_anchor.items():
        heat_a = heat_target_per_anchor[a]
        heat_max = max(heat_a.max(), 1e-12)
        active = heat_a > heat_threshold * heat_max
        active_idx = np.where(active)[0]
        if not len(active_idx): continue

        K = len(src_list)
        M = len(active_idx)

        # Source profiles & marginals
        profiles = []
        marginals_src = []
        for s in src_list:
            w = s['heat_weights']
            W = max(w.sum(), 1e-12)
            prof = (w[:, None] * h_vec_source[s['indices']]).sum(0) / W
            profiles.append(prof)
            marginals_src.append(W)
        profiles = np.stack(profiles)                                # (K, N_anchors)
        marginals_src = np.array(marginals_src)
        marginals_src = marginals_src / max(marginals_src.sum(), 1e-12)

        # Target features & marginals
        target_feat = h_vec_target[active_idx]                       # (M, N_anchors)
        target_mass = heat_a[active_idx]
        marginals_tgt = target_mass / max(target_mass.sum(), 1e-12)

        # Cost matrix (K, M)
        cost = ((profiles[:, None, :] - target_feat[None, :, :]) ** 2).sum(-1)
        cost = cost / max(cost.max(), 1e-12)                         # нормировка для стабильности

        # Sinkhorn iterations
        K_mat = np.exp(-cost / max(epsilon, 1e-6))
        u = np.ones(K)
        v = np.ones(M)
        for _ in range(n_iter):
            v = marginals_tgt / (K_mat.T @ u + 1e-12)
            u = marginals_src / (K_mat @ v + 1e-12)
        T = u[:, None] * K_mat * v[None, :]                          # (K, M) transport plan

        # Каждая target вершина → cluster с максимальной массой
        assignment = np.argmax(T, axis=0)

        for k, s in enumerate(src_list):
            mask = assignment == k
            if not mask.any(): continue
            t_indices = active_idx[mask]
            t_heat = heat_a[t_indices]
            W = max(t_heat.sum(), 1e-12)
            c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
            target_clusters.append({
                'source': s,
                'target_indices': t_indices,
                'target_heat': t_heat,
                'c_target': c_target,
                'transport_mass': float(T[k, mask].sum()),
            })
    return target_clusters


def assign_target_via_rbf(
        verts_target, faces_target, heat_target_per_anchor,
        src_clusters_list, anchor_pos_source, anchor_pos_target,
        heat_threshold=0.05, geodesic_factor=3.0,
        rbf_kernel='thin_plate_spline'):
    """C. RBF INTERPOLATION — anchor-пары как control points для warping.

    Используем известные соответствия anchor[i]_source ↔ anchor[i]_target
    для построения **нелинейного отображения** xyz_source → xyz_target
    через Radial Basis Function интерполяцию.

    Для каждого source-кластера:
        target_pos = rbf(cluster.c_rest)        # warped position на FBX
        seed = nearest vertex
        Dijkstra по поверхности до σ·factor

    Учитывает локальные деформации между мешами лучше чем простой
    anchor-relative offset (который использует только ОДИН ближайший anchor).
    """
    from scipy.interpolate import RBFInterpolator
    if len(anchor_pos_source) < 2:
        raise ValueError("RBF требует минимум 2 anchor-пары")

    # smoothing=0 → точная интерполяция через все anchor-пары
    rbf = RBFInterpolator(anchor_pos_source, anchor_pos_target,
                            kernel=rbf_kernel, smoothing=0)
    print(f"  RBF kernel: {rbf_kernel}, anchor pairs: {len(anchor_pos_source)}")

    N_t = verts_target.shape[0]
    neighbors = build_vertex_adjacency(N_t, verts_target, faces_target)

    target_clusters = []
    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    for a, src_list in src_by_anchor.items():
        heat_a = heat_target_per_anchor[a]
        heat_max = max(heat_a.max(), 1e-12)
        active = heat_a > heat_threshold * heat_max
        active_idx = np.where(active)[0]
        if not len(active_idx): continue

        K = len(src_list)
        geo_dists = np.full((N_t, K), np.inf)
        for k, s in enumerate(src_list):
            # Маппим source-центроид на target меш через RBF warp
            target_pos = rbf(s['c_rest'][None, :])[0]
            seed = find_nearest_vertex(verts_target, target_pos)
            R_max = s['spatial_sigma'] * geodesic_factor
            if not np.isfinite(R_max): R_max = float('inf')
            geo_dists[:, k] = geodesic_dijkstra(neighbors, seed, R_max)

        dists_active = geo_dists[active_idx]
        nearest = np.argmin(dists_active, axis=1)
        reachable = ~np.all(np.isinf(dists_active), axis=1)

        for k, s in enumerate(src_list):
            mask = (nearest == k) & reachable
            if not mask.any(): continue
            t_indices = active_idx[mask]
            t_heat = heat_a[t_indices]
            W = max(t_heat.sum(), 1e-12)
            c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
            target_clusters.append({
                'source': s,
                'target_indices': t_indices,
                'target_heat': t_heat,
                'c_target': c_target,
                'geo_dists': dists_active[mask, k],
            })
    return target_clusters


def assign_target_to_source_hybrid(
        verts_target, faces_target, heat_target_per_anchor,
        src_clusters_list, sig_target, sig_source,
        anchor_pos_source, anchor_pos_target,
        w_geo=1.0, w_sig=1.0,
        heat_threshold=0.05, geodesic_factor=3.0):
    """ГИБРИДНЫЙ режим: Voronoi-geodesic + signature similarity.

    Для каждой active target вершины:
      score_k = w_geo · (geo_dist[v,k] / max_geo) + w_sig · (sig_dist[v,k] / max_sig)
      assign к argmin(score)

    Геодезик защищает от "семантически похожих но далёких" вершин
    (например симметричная сторона лица).
    Сигнатура защищает от "близких по xyz но анатомически разных"
    (например складки век, носогубная борозда).
    """
    N_t = verts_target.shape[0]
    print(f"  HYBRID: Dijkstra adjacency + signature matching")
    neighbors = build_vertex_adjacency(N_t, verts_target, faces_target)

    sig_t = normalize_signature(sig_target)
    sig_s = normalize_signature(sig_source)

    target_clusters = []
    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    for a, src_list in src_by_anchor.items():
        heat_a = heat_target_per_anchor[a]
        heat_max = max(heat_a.max(), 1e-12)
        active = heat_a > heat_threshold * heat_max
        active_idx = np.where(active)[0]
        if not len(active_idx): continue

        K = len(src_list)

        # Geodesic distances (с anchor-relative seed)
        geo_dists = np.full((N_t, K), np.inf)
        for k, s in enumerate(src_list):
            offset = s['c_rest'] - anchor_pos_source[s['anchor_idx']]
            target_xyz = anchor_pos_target[s['anchor_idx']] + offset
            seed = find_nearest_vertex(verts_target, target_xyz)
            R_max = s['spatial_sigma'] * geodesic_factor
            if not np.isfinite(R_max): R_max = float('inf')
            geo_dists[:, k] = geodesic_dijkstra(neighbors, seed, R_max)

        # Signature distances
        profiles = np.stack([
            (lambda s: (s['heat_weights'][:, None] *
                        sig_s[s['indices']]).sum(0) /
                       max(s['heat_weights'].sum(), 1e-12))(s)
            for s in src_list
        ])
        profiles = normalize_signature(profiles)
        sig_dists = ((sig_t[:, None, :] - profiles[None, :, :]) ** 2).sum(-1)  # (N, K)

        # Нормируем расстояния перед взвешенной суммой
        geo_clip = np.where(np.isinf(geo_dists), 1e10, geo_dists)
        geo_norm = geo_clip / max(geo_clip[geo_clip < 1e10].max(), 1e-12)
        sig_norm = sig_dists / max(sig_dists.max(), 1e-12)

        # Combined score (∞ для геодезически недостижимых)
        combined = w_geo * geo_norm + w_sig * sig_norm
        combined = np.where(np.isinf(geo_dists), np.inf, combined)

        dists_active = combined[active_idx]
        nearest = np.argmin(dists_active, axis=1)
        reachable = ~np.all(np.isinf(dists_active), axis=1)

        for k, s in enumerate(src_list):
            mask = (nearest == k) & reachable
            if not mask.any(): continue
            t_indices = active_idx[mask]
            t_heat = heat_a[t_indices]
            W = max(t_heat.sum(), 1e-12)
            c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
            target_clusters.append({
                'source': s,
                'target_indices': t_indices,
                'target_heat': t_heat,
                'c_target': c_target,
                'combined_score': dists_active[mask, k],
            })
    return target_clusters


def assign_target_to_source_by_signature(
        verts_target, heat_target_per_anchor, src_clusters_list,
        sig_target, sig_source, heat_threshold=0.05,
        normalize=True):
    """Voronoi-разбиение по INTRINSIC СИГНАТУРЕ (HKS или WKS).

    Алгоритм:
      Для каждой anchor-зоны на target меше:
        1. Активные вершины: heat_target[a, v] > threshold (geodesic-aware фильтр)
        2. Для каждого source-кластера c в этом anchor:
             cluster_profile_c = heat-weighted average of sig_source over c.indices
        3. Для каждой активной вершины v:
             assign к кластеру с min ||sig_target[v] - cluster_profile_c||²
             (или cosine similarity если normalize=True)

    Симметрия снимается через heat-зону (вершина должна быть в anchor'е).
    """
    target_clusters = []
    src_by_anchor = {}
    for s in src_clusters_list:
        src_by_anchor.setdefault(s['anchor_idx'], []).append(s)

    sig_t = normalize_signature(sig_target) if normalize else sig_target
    sig_s = normalize_signature(sig_source) if normalize else sig_source

    for a, src_list in src_by_anchor.items():
        heat_a = heat_target_per_anchor[a]
        heat_max = max(heat_a.max(), 1e-12)
        active = heat_a > heat_threshold * heat_max
        active_idx = np.where(active)[0]
        if not len(active_idx): continue

        # Профили source-кластеров в signature space
        profiles = np.stack([
            (lambda s: (s['heat_weights'][:, None] *
                        sig_s[s['indices']]).sum(0) /
                       max(s['heat_weights'].sum(), 1e-12))(s)
            for s in src_list
        ])                                                              # (K, D)
        if normalize:
            profiles = normalize_signature(profiles)

        active_sig = sig_t[active_idx]                                  # (M, D)
        # L2 distance в нормированном пространстве ≈ 2(1 - cosine)
        dists = ((active_sig[:, None, :] - profiles[None, :, :]) ** 2).sum(-1)
        nearest = np.argmin(dists, axis=1)

        for k, s in enumerate(src_list):
            mask = nearest == k
            if not mask.any(): continue
            t_indices = active_idx[mask]
            t_heat = heat_a[t_indices]
            W = max(t_heat.sum(), 1e-12)
            c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
            target_clusters.append({
                'source':         s,
                'target_indices': t_indices,
                'target_heat':    t_heat,
                'c_target':       c_target,
                'sig_dist':       float(dists[mask, k].mean()),
            })
    return target_clusters


def apply_target_clusters_transfer(verts_target, target_clusters):
    """Применяет (μ, R, S) source кластеров к target вершинам.
    δ[v] = μ_s + (R_s S_s - I)(verts_target[v] - c_target).
    """
    N = verts_target.shape[0]
    delta = np.zeros((N, 3))
    weight = np.zeros(N)
    I3 = np.eye(3)
    for tc in target_clusters:
        s = tc['source']
        c_t = tc['c_target']
        RS = s['R'] @ s['S']
        for j, v_idx in enumerate(tc['target_indices']):
            r = verts_target[v_idx] - c_t
            d = s['mu'] + (RS - I3) @ r
            w = tc['target_heat'][j]
            delta[v_idx] += w * d
            weight[v_idx] += w
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
    """Selection с увеличенной target-площадью.

    Поверх меша рисуется point cloud вершин с большим point_size, так что
    Shift+клик не нужно делать прямо в вертекс — целишься в "толстую точку"
    которая занимает несколько пикселей вокруг реального вертекса.

    После выбора каждая picked точка маппится в ближайшую mesh-вершину
    через nearest-neighbor (на случай если клик попал в other geometry).
    """
    from scipy.spatial import cKDTree

    print(f"\n[{head_name}] Shift+клик до {max_n} точек, Q закроет.")
    print(f"  Каждая вершина = точка размером {point_size}px → можно не "
          f"целиться прямо в неё.")

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(f"Выбери точки — {head_name}", 1000, 800)

    # Меш — для визуализации формы
    mesh = o3d_mesh(verts, faces)
    vis.add_geometry(mesh)

    # Point cloud — большие точки на вершинах = "удобные target'ы"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile([0.95, 0.7, 0.15], (len(verts), 1))   # янтарные точки
    )
    vis.add_geometry(pcd)

    # Размер точек в пикселях — увеличивает clickable area
    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.mesh_show_back_face = True

    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()

    # Маппинг picked в индексы mesh-вершин:
    # picked.coord (xyz) → nearest-neighbor в исходных верт. меша
    tree = cKDTree(verts)
    chosen, seen = [], set()
    for p in picked:
        # PickedPoint может иметь .coord (некоторые версии Open3D) или только .index
        if hasattr(p, 'coord') and p.coord is not None:
            try:
                xyz = np.asarray(p.coord, dtype=np.float64).reshape(3)
                _, idx = tree.query(xyz, k=1)
                idx = int(idx)
            except Exception:
                idx = int(p.index)
        else:
            idx = int(p.index)
        # PointCloud и Mesh имеют разные internal index spaces, но
        # nearest-neighbor по coord даёт корректный mesh-vertex.
        # На всякий случай: clamp в диапазон и дедуплицируем.
        if 0 <= idx < len(verts) and idx not in seen:
            chosen.append(idx)
            seen.add(idx)
        if len(chosen) >= max_n:
            break

    print(f"  Выбрано {len(chosen)} точек: {chosen}")
    return chosen




def animate_diffusion(verts, faces, L, MM, srcs, total_time, steps, fps=24):
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

    vis = o3d.visualization.Visualizer()
    vis.create_window(f"ОКНО 1: Diffusion {N} sources, t={total_time}  (Q)", 1200, 800)
    for g in [mesh, *spheres]: vis.add_geometry(g)
    vis.get_render_option().mesh_show_back_face = True

    frame_dt = 1.0 / fps
    for _ in range(steps):
        t0 = time_mod.perf_counter()
        for ai in range(N):
            u[ai] = solve(MM @ u[ai])
        mesh.vertex_colors = o3d.utility.Vector3dVector(to_colors(u.sum(0), CMAP_HEAT))
        vis.update_geometry(mesh)
        if not vis.poll_events(): break
        vis.update_renderer()
        wait = frame_dt - (time_mod.perf_counter() - t0)
        if wait > 0: time_mod.sleep(wait)
    print("Диффузия завершена. Q — продолжить.")
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
        'assign_mode':      (input(f"assign_mode (voronoi/hks/wks/hybrid/heat_vec/heat_align/heat_svd/heat_rank/sinkhorn/rbf) [{defaults.get('assign_mode','voronoi')}]: ").strip() or defaults.get('assign_mode', 'voronoi')),
        'n_eigs':           ask('n_eigs', 'n_eigs (HKS/WKS)', int),
        'n_scales':         ask('n_scales', 'n_scales (HKS/WKS)', int),
        'n_svd':            ask('n_svd', 'n_svd (heat_svd, 0=auto)', int) if 'n_svd' in defaults else 0,
        'heat_align_knn':   ask('heat_align_knn', 'heat_align_knn', int) if 'heat_align_knn' in defaults else 5,
        'heat_align_smooth': ask('heat_align_smooth', 'heat_align_smooth', int) if 'heat_align_smooth' in defaults else 2,
        'w_geo':            ask('w_geo', 'w_geo (hybrid)', float),
        'w_sig':            ask('w_sig', 'w_sig (hybrid)', float),
        'sinkhorn_eps':     ask('sinkhorn_eps', 'sinkhorn epsilon', float),
        'rbf_kernel':       (input(f"rbf_kernel [{defaults.get('rbf_kernel','thin_plate_spline')}]: ").strip() or defaults.get('rbf_kernel', 'thin_plate_spline')),
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
        "N clusters max:", defaults['n_clusters'], row); row += 1
    vars_['heat_threshold'] = add_label_entry(
        "Heat threshold:", defaults['heat_threshold'], row); row += 1

    section("── Сглаживание ──")
    vars_['smooth_iters'] = add_label_entry(
        "Smooth iters:", defaults['smooth_iters'], row); row += 1
    vars_['smooth_alpha'] = add_label_entry(
        "Smooth alpha (0..1):", defaults['smooth_alpha'], row); row += 1
    vars_['smooth_iters_fbx'] = add_label_entry(
        "Smooth iters FBX (крупнее → больше):",
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
    vars_['assign_mode'] = tk.StringVar(value=defaults.get('assign_mode', 'voronoi'))
    mode_frame = tk.Frame(frame)
    mode_frame.grid(row=row, column=1, sticky='w', pady=4)
    for mode_val, mode_label in [
        ('voronoi',   'Voronoi+Dijkstra (anchor-relative)'),
        ('hks',       'HKS (Heat Kernel Sig., scale-invariant)'),
        ('wks',       'WKS (Wave Kernel Sig.)'),
        ('hybrid',    'Hybrid (Voronoi + HKS)'),
        ('heat_vec',   'Heat-vector (anchor distances)'),
        ('heat_align', 'Heat-ALIGN (per-vertex correspondence) ⭐'),
        ('heat_svd',   'Heat-SVD (joint matrix decomposition)'),
        ('heat_rank', 'Heat-RANK percentile (shape-invariant)'),
        ('sinkhorn',  'Sinkhorn Optimal Transport (balanced)'),
        ('rbf',       'RBF interpolation (anchor warping)'),
    ]:
        tk.Radiobutton(mode_frame, text=mode_label, variable=vars_['assign_mode'],
                        value=mode_val, anchor='w').pack(anchor='w')
    row += 1

    vars_['n_eigs'] = add_label_entry(
        "n_eigs (для HKS/WKS):", defaults.get('n_eigs', 128), row); row += 1
    vars_['n_scales'] = add_label_entry(
        "n_scales (для HKS/WKS):", defaults.get('n_scales', 16), row); row += 1
    vars_['n_svd'] = add_label_entry(
        "n_svd (heat_svd, 0=auto):", defaults.get('n_svd', 0), row); row += 1
    vars_['heat_align_knn'] = add_label_entry(
        "heat_align k_nn (1=argmin, ≥3 anti-noise):",
        defaults.get('heat_align_knn', 5), row); row += 1
    vars_['heat_align_smooth'] = add_label_entry(
        "heat_align mesh smooth iters:",
        defaults.get('heat_align_smooth', 2), row); row += 1
    vars_['w_geo'] = add_label_entry(
        "w_geo (hybrid):", defaults.get('w_geo', 1.0), row); row += 1
    vars_['w_sig'] = add_label_entry(
        "w_sig (hybrid):", defaults.get('w_sig', 1.0), row); row += 1
    vars_['sinkhorn_eps'] = add_label_entry(
        "sinkhorn epsilon:", defaults.get('sinkhorn_eps', 0.05), row); row += 1
    vars_['rbf_kernel'] = add_label_combo(
        "RBF kernel:",
        ['thin_plate_spline', 'multiquadric', 'gaussian',
         'inverse_multiquadric', 'cubic', 'quintic', 'linear'],
        default_idx=0, row=row); row += 1

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
            result['heat_threshold']  = float(vars_['heat_threshold'].get())
            result['smooth_iters']    = int(vars_['smooth_iters'].get())
            result['smooth_alpha']    = float(vars_['smooth_alpha'].get())
            result['smooth_iters_fbx'] = int(vars_['smooth_iters_fbx'].get())
            result['n_anchors']       = int(vars_['n_anchors'].get())
            result['fbx_path']        = vars_['fbx_path'].get().strip()
            result['geodesic_factor'] = float(vars_['geodesic_factor'].get())
            result['assign_mode']     = vars_['assign_mode'].get()
            result['n_eigs']          = int(vars_['n_eigs'].get())
            result['n_scales']        = int(vars_['n_scales'].get())
            result['n_svd']           = int(vars_['n_svd'].get())
            result['heat_align_knn']  = int(vars_['heat_align_knn'].get())
            result['heat_align_smooth'] = int(vars_['heat_align_smooth'].get())
            result['w_geo']           = float(vars_['w_geo'].get())
            result['w_sig']           = float(vars_['w_sig'].get())
            result['sinkhorn_eps']    = float(vars_['sinkhorn_eps'].get())
            result['rbf_kernel']      = vars_['rbf_kernel'].get()
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
        # ── v3: режим матчинга на target меше ──
        'assign_mode':     'voronoi',     # 'voronoi' | 'hks' | 'wks' | 'hybrid'
                                          #          | 'heat_vec' | 'sinkhorn' | 'rbf'
        'n_eigs':          128,           # сколько собств. мод считать для HKS/WKS
        'n_scales':        16,            # сколько t-значений / energy levels
        'n_svd':           0,             # heat_svd: 0=auto (min(K,16))
        'heat_align_knn':  5,             # heat_align: 1=argmin, 3-10=anti-noise
        'heat_align_smooth': 2,           # heat_align: post-smoothing iters на mesh-графе
        'w_geo':           1.0,           # вес geodesic в hybrid режиме
        'w_sig':           1.0,           # вес signature в hybrid режиме
        'sinkhorn_eps':    0.05,          # регуляризация Sinkhorn
        'rbf_kernel':      'thin_plate_spline',  # ядро для RBF
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
    heat = animate_diffusion(verts, faces, L, MM, src, t_anim, num_steps, fps=fps)
    print(f"  heat shape: {heat.shape}, max per anchor: {heat.max(axis=1).round(4)}")

    # Сохраняем heat-карты (N×K, по колонке на anchor)
    heat_columns = ",".join([f"anchor_{a}" for a in range(N_anchors)])
    save_matrix_csv(OUT_HEAD1 / "heat.csv", heat.T, header=heat_columns)
    print(f"  → saved {OUT_HEAD1/'heat.csv'} ({heat.shape[1]} verts × {N_anchors} anchors)")

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
    clusters_per_anchor = []
    for a in range(N_anchors):
        cls = cluster_zone(
            heat[a], delta_native, verts, anchor_idx=a,
            heat_threshold=heat_thresh,
            n_clusters_max=n_clusters,
            position_weight=pos_weight,
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
            t_anim, num_steps, fps=fps)

        heat_fbx_columns = ",".join([f"anchor_{a}" for a in range(N_anchors)])
        save_matrix_csv(OUT_FBX / "heat.csv", heat_fbx.T, header=heat_fbx_columns)
        print(f"  → saved FBX heat.csv ({heat_fbx.shape[1]} verts × {N_anchors})")

        # ── Выбор стратегии разбиения target меша ────────────────────────────
        src_flat = [cl for cls in clusters_per_anchor for cl in cls]
        assign_mode = params.get('assign_mode', 'voronoi')
        n_eigs = int(params.get('n_eigs', 128))
        n_scales = int(params.get('n_scales', 16))

        # Anchor positions для anchor-relative offset (главный фикс v3)
        anchor_pos_source = verts[src]                # (N_anchors, 3) на HEAD 1
        anchor_pos_target = verts_fbx[src_fbx]        # (N_anchors, 3) на FBX

        # Pre-compute spectra если нужно (hks/wks/hybrid)
        sig_source = sig_target = None
        if assign_mode in ('hks', 'wks', 'hybrid'):
            # Eigendecomp на HEAD 1
            t0 = time_mod.perf_counter()
            print(f"\n  [HEAD 1] eigendecomp (n_eigs={n_eigs})...")
            ev1, ef1 = compute_spectrum(verts, faces, n_eigs=n_eigs)
            print(f"           {time_mod.perf_counter()-t0:.1f}s, "
                  f"λ range [{ev1[0]:.4f}, {ev1[-1]:.2f}]")
            # Eigendecomp на FBX
            t0 = time_mod.perf_counter()
            print(f"  [FBX]    eigendecomp...")
            ev2, ef2 = compute_spectrum(verts_fbx, faces_fbx, n_eigs=n_eigs)
            print(f"           {time_mod.perf_counter()-t0:.1f}s, "
                  f"λ range [{ev2[0]:.4f}, {ev2[-1]:.2f}]")

            sig_kind = 'wks' if assign_mode == 'wks' else 'hks'
            if sig_kind == 'hks':
                ts1 = default_hks_times(ev1, n_scales=n_scales)
                ts2 = default_hks_times(ev2, n_scales=n_scales)
                ts = np.sqrt(ts1 * ts2)
                # scale_invariant=True → cross-mesh сопоставимо (фикс v3)
                sig_source = compute_hks(ev1, ef1, ts, scale_invariant=True)
                sig_target = compute_hks(ev2, ef2, ts, scale_invariant=True)
                print(f"  HKS computed: t range [{ts[0]:.6f}, {ts[-1]:.4f}], "
                      f"scale_invariant=True, shapes "
                      f"{sig_source.shape}/{sig_target.shape}")
            else:
                en1, sg1 = default_wks_energies(ev1, n_scales=n_scales)
                en2, sg2 = default_wks_energies(ev2, n_scales=n_scales)
                en_min = max(en1[0], en2[0])
                en_max = min(en1[-1], en2[-1])
                en = np.linspace(en_min, en_max, n_scales)
                sg = (sg1 + sg2) * 0.5
                sig_source = compute_wks(ev1, ef1, en, sg)
                sig_target = compute_wks(ev2, ef2, en, sg)
                print(f"  WKS computed: energy range [{en[0]:.4f}, {en[-1]:.4f}], "
                      f"sigma={sg:.4f}, shapes "
                      f"{sig_source.shape}/{sig_target.shape}")

        if assign_mode == 'voronoi':
            print(f"\n  Voronoi+Dijkstra (geodesic_factor={params['geodesic_factor']})...")
            target_clusters = assign_target_to_source_clusters(
                verts_fbx, faces_fbx, heat_fbx, src_flat,
                anchor_pos_source=anchor_pos_source,        # ФИКС v3
                anchor_pos_target=anchor_pos_target,
                heat_threshold=heat_thresh,
                geodesic_factor=params['geodesic_factor'],
            )

        elif assign_mode in ('hks', 'wks'):
            print(f"\n  Pure {assign_mode.upper()} matching")
            target_clusters = assign_target_to_source_by_signature(
                verts_fbx, heat_fbx, src_flat,
                sig_target=sig_target, sig_source=sig_source,
                heat_threshold=heat_thresh, normalize=True,
            )

        elif assign_mode == 'hybrid':
            w_geo = float(params.get('w_geo', 1.0))
            w_sig = float(params.get('w_sig', 1.0))
            print(f"\n  HYBRID: Voronoi + HKS (w_geo={w_geo}, w_sig={w_sig})")
            target_clusters = assign_target_to_source_hybrid(
                verts_fbx, faces_fbx, heat_fbx, src_flat,
                sig_target=sig_target, sig_source=sig_source,
                anchor_pos_source=anchor_pos_source,
                anchor_pos_target=anchor_pos_target,
                w_geo=w_geo, w_sig=w_sig,
                heat_threshold=heat_thresh,
                geodesic_factor=params['geodesic_factor'],
            )

        elif assign_mode == 'heat_vec':
            print(f"\n  HEAT-VECTOR matching (anchor distance encoding)")
            target_clusters = assign_target_to_source_by_heat_vector(
                verts_fbx, heat_fbx, src_flat,
                heat_source_per_anchor=heat,                 # heat HEAD 1
                heat_threshold=heat_thresh,
                normalize=True,
            )

        elif assign_mode == 'heat_align':
            k_nn = int(params.get('heat_align_knn', 5))
            ls_iters = int(params.get('heat_align_smooth', 2))
            print(f"\n  HEAT-ALIGN matching (per-vertex, k_nn={k_nn}, smooth={ls_iters})")
            target_clusters = assign_target_to_source_by_heat_align(
                verts_fbx, heat_fbx, src_flat,
                heat_source_per_anchor=heat,
                heat_threshold=heat_thresh,
                k_nn=k_nn,
                label_smooth_iters=ls_iters,
                faces_target=faces_fbx,
            )

        elif assign_mode == 'heat_svd':
            n_svd = int(params.get('n_svd', 0))
            print(f"\n  HEAT-SVD matching (joint SVD descriptors, n_components={n_svd or 'auto'})")
            target_clusters = assign_target_to_source_by_heat_svd(
                verts_fbx, heat_fbx, src_flat,
                heat_source_per_anchor=heat,
                heat_threshold=heat_thresh,
                n_components=n_svd,
                normalize=True,
            )

        elif assign_mode == 'heat_rank':
            print(f"\n  HEAT-RANK matching (per-anchor percentile, shape-invariant)")
            target_clusters = assign_target_to_source_by_heat_rank(
                verts_fbx, heat_fbx, src_flat,
                heat_source_per_anchor=heat,
                heat_threshold=heat_thresh,
                normalize=True,
            )

        elif assign_mode == 'sinkhorn':
            sinkhorn_eps = float(params.get('sinkhorn_eps', 0.05))
            print(f"\n  SINKHORN Optimal Transport (eps={sinkhorn_eps})")
            target_clusters = assign_target_to_source_by_sinkhorn(
                verts_fbx, heat_fbx, src_flat,
                heat_source_per_anchor=heat,
                heat_threshold=heat_thresh,
                epsilon=sinkhorn_eps, n_iter=200,
            )

        elif assign_mode == 'rbf':
            rbf_kernel = params.get('rbf_kernel', 'thin_plate_spline')
            print(f"\n  RBF interpolation (kernel={rbf_kernel}, "
                  f"geodesic_factor={params['geodesic_factor']})")
            target_clusters = assign_target_via_rbf(
                verts_fbx, faces_fbx, heat_fbx, src_flat,
                anchor_pos_source=anchor_pos_source,
                anchor_pos_target=anchor_pos_target,
                heat_threshold=heat_thresh,
                geodesic_factor=params['geodesic_factor'],
                rbf_kernel=rbf_kernel,
            )

        else:
            raise ValueError(f"Unknown assign_mode: {assign_mode}. Use one of: "
                              f"voronoi / hks / wks / hybrid / heat_vec / heat_align "
                              f"/ heat_svd / heat_rank / sinkhorn / rbf")

        print(f"  Получено {len(target_clusters)} target-кластеров из {len(src_flat)} source")
        # Диагностика: показать "отпечаток" assignment'а (сумма target-индексов на cluster)
        sig = 0
        for tc in target_clusters:
            sig = (sig * 31 + int(tc['target_indices'].sum())) % (10**9)
        print(f"  ASSIGN signature (для сравнения методов): {sig}")
        print(f"  Total target verts assigned: "
              f"{sum(len(tc['target_indices']) for tc in target_clusters)}")

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
