"""
Motion-Groups — VERSION 6.0  (HEAD 1 only / source analysis)

После v5 проведён радикальный strip: УБРАНО ВСЁ что связано с FBX,
matching'ом и переносом деформаций. v6 — это чистый анализ source-головы
FLAME: diffusion → multi-t enrichment → clustering motion-групп.

ЧТО ОСТАЛОСЬ (перенесено из v5):
  • Пресеты настроек (save/load)
  • Форма головы (shape betas preset)
  • Экспрессия (expression preset + custom betas)
  • Heat-диффузия (+ авто-стоп при overlap зон)
  • Кластеризация (kmeans / agglomerative)
  • Сглаживание (Laplacian δ-smoothing)
  • Anchor-точки (Shift+click pick)
  • Multi-t enrichment (+ маскировка single-t reach'ем)

ПАЙПЛАЙН v6:
  1. Выбираем anchor-точки (heat sources)
  2. Если multi_t ВЫКЛ → кластеризуем всю голову (per-anchor zone по heat-порогу)
  3. Если multi_t ВКЛ  → кластеризуем ТОЛЬКО внутри зоны ответственности каждого
     anchor'а (hard argmax-partition), маскированной single-t reach'ем
  4. Финальное окно:
       multi_t ВКЛ → замаскированные hard-partition зоны
       multi_t ВЫКЛ → кластеры, окрашенные смесью single-t от каждого anchor'а

  БЕЗ матчинга голов и без переноса деформаций.

ЗАПУСК:
  cd /Users/kuzix/Documents/GitHub/Muscle-autoskinner
  source .venv/bin/activate
  python3 python/scripts/motion_groups_v6/debug_head1_pipeline.py
"""

__version__ = "6.0"

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


# Абсолютный путь к FLAME (привязан к корню репо через __file__) — работает из
# любого cwd. Этот файл лежит в <repo>/python/scripts/motion_groups_v6/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
FLAME_PKL = str(_REPO_ROOT / "Assets/Meshes/FLAME/"
                             "FLAME2023 Open for commercial use/flame2023_Open.pkl")


# ── Колормэпы (без matplotlib — он крашит pipeline на macOS) ──────────────────

def _cmap_hot(v):
    """Hot colormap: black → red → yellow → white. v in [0, 1] → (N, 3)."""
    v = np.clip(v, 0, 1)
    rgb = np.zeros((len(v), 3))
    rgb[:, 0] = np.clip(v * 3, 0, 1)
    rgb[:, 1] = np.clip((v - 0.33) * 3, 0, 1)
    rgb[:, 2] = np.clip((v - 0.66) * 3, 0, 1)
    return rgb


def _cmap_cool(v):
    """Cool colormap: cyan → magenta. v in [0, 1] → (N, 3)."""
    v = np.clip(v, 0, 1)
    rgb = np.empty((len(v), 3))
    rgb[:, 0] = v
    rgb[:, 1] = 1.0 - v
    rgb[:, 2] = 1.0
    return rgb


CMAP_HEAT = _cmap_hot
CMAP_DISP = _cmap_cool


# ── Загрузка / геометрия ─────────────────────────────────────────────────────

def _install_chumpy_shim():
    """FLAME .pkl сериализован с chumpy. Регистрируем заглушку, чтобы pickle
    мог распаковать без установки самого chumpy (он ломается на новых numpy)."""
    import sys, types
    if 'chumpy' in sys.modules:
        return
    m = types.ModuleType('chumpy')

    class Ch:
        def __setstate__(self, state):
            self.__dict__.update(state if isinstance(state, dict) else {})
        @property
        def r(self):
            return getattr(self, 'x', None)

    m.Ch = Ch
    ch_mod = types.ModuleType('chumpy.ch')
    ch_mod.Ch = Ch
    m.ch = ch_mod
    sys.modules['chumpy'] = m
    sys.modules['chumpy.ch'] = ch_mod


def load_flame(path):
    _install_chumpy_shim()
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


def load_custom_mesh(path):
    """FBX/OBJ/PLY → trimesh через assimp (process=True мержит UV-splits)."""
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


def build_operators(verts, faces, clamp_cot=False):
    """Котангенс-Лапласиан L и диагональная масс-матрица MM (барицентр.).

    clamp_cot=True → отрицательные котангенсы (тупые углы / тонкие треугольники)
    клампятся в 0. Это сохраняет принцип максимума при диффузии (тепло не
    «проваливается» в ноль рядом с горячей границей) — нужно для равномерного
    теплового поля и устойчивого heat-warp на кривой/неровной развёртке."""
    N = len(verts)
    row, col, data = [], [], []
    for i, j, k in [(0, 1, 2), (1, 2, 0), (2, 0, 1)]:
        vi, vj, vk = verts[faces[:, i]], verts[faces[:, j]], verts[faces[:, k]]
        u, v = vi - vk, vj - vk
        cos_a = (u * v).sum(1)
        sin_a = np.linalg.norm(np.cross(u, v), axis=1).clip(1e-8)
        cot = cos_a / sin_a * 0.5
        if clamp_cot:
            cot = np.clip(cot, 0.0, None)
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


# ── Сохранение дампов ─────────────────────────────────────────────────────────

def save_matrix_csv(path, array, header=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(array), delimiter=',',
                header=(header or ''), comments='')


def save_clusters_json(path, clusters_per_anchor):
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


def save_metadata_json(path, params, shape, expr, n_anchors, src1):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    meta = {
        'version':     __version__,
        'timestamp':   _dt.datetime.now().isoformat(),
        'shape_betas': {str(k): float(v) for k, v in shape.items()},
        'expr_betas':  {str(k): float(v) for k, v in expr.items()},
        'n_anchors':   int(n_anchors),
        'src_head1':   [int(x) for x in src1],
        'params':      {k: v for k, v in params.items()
                        if not k.startswith('_') and not isinstance(v, dict)},
    }
    with open(path, 'w') as f:
        json.dump(meta, f, indent=2)


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
    """Группировка вершин anchor-зоны в motion-кластеры.

    clustering_method:
      'kmeans'        — авто-K по числу активных вершин (≤ n_clusters_max)
      'agglomerative' — порог similarity_threshold в нормированном feature-space
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
        ac = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=similarity_threshold,
            linkage='ward')
        labels = ac.fit_predict(features)
        n_clusters = labels.max() + 1
        if n_clusters > n_clusters_max:
            km = KMeans(n_clusters=n_clusters_max, n_init=8, random_state=0)
            labels = km.fit_predict(features)
            n_clusters = n_clusters_max
        elif n_clusters < 2:
            n_clusters = min(2, n_clusters_max)
            km = KMeans(n_clusters=n_clusters, n_init=8, random_state=0)
            labels = km.fit_predict(features)
    else:
        n_clusters = max(2, min(n_clusters_max, len(active_idx) // 30))
        km = KMeans(n_clusters=n_clusters, n_init=8, random_state=0)
        labels = km.fit_predict(features)

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


# ── Spectral / multi-t enrichment ────────────────────────────────────────────

def compute_spectrum(verts, faces, n_eigs=128):
    """Generalised eigenproblem L·v = λ·M·v. Возвращает (eigvals, eigvecs)."""
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
    """Multi-scale heat от anchor-точек через spectral expansion:
    heat(v, anchor_a, t) = Σ_k exp(-t·λ_k) · ψ_k(anchor_a) · ψ_k(v)

    Returns: H_multi (K*T, N), times (T,)
    """
    K = len(anchor_indices)
    N = eigvecs.shape[0]
    if t_min is None or t_max is None:
        lam_min = max(float(eigvals[1]), 1e-6)
        lam_max = float(eigvals[-1])
        if t_min is None: t_min = 4 * np.log(10) / lam_max
        if t_max is None: t_max = 4 * np.log(10) / lam_min
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_times)
    H_multi = np.zeros((K * n_times, N), dtype=np.float64)
    for ki, a in enumerate(anchor_indices):
        psi_a = eigvecs[int(a), :]
        for ti, t in enumerate(times):
            weights = np.exp(-t * eigvals) * psi_a
            H_multi[ki * n_times + ti] = eigvecs @ weights
    return H_multi, times


def enrich_heat_multi_t(verts, faces, anchor_indices,
                         n_times=8, n_eigs=80,
                         smooth_iters=5, smooth_alpha=0.5,
                         mesh_label="MESH"):
    """Multi-t enrichment + Laplacian smoothing.
    Returns: (heat_enriched (K,N), times_used (T,))
    """
    K = len(anchor_indices)
    N = len(verts)
    ev, ef = compute_spectrum(verts, faces, n_eigs=n_eigs)
    H_multi, times_used = compute_anchor_heat_multi_t(
        ev, ef, anchor_indices, n_times=n_times)
    H_multi_n = H_multi / H_multi.max(axis=1, keepdims=True).clip(min=1e-12)
    H_3d = H_multi_n.reshape(K, n_times, -1)
    heat_enriched = np.sqrt((H_3d ** 2).mean(axis=1))                # (K, N)
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


def _argmax_partition(heat_per_anchor, threshold=0.05):
    """Hard-partition по argmax. labels (N,) ∈ [0..K-1], -1 если ниже threshold."""
    K, N = heat_per_anchor.shape
    H = heat_per_anchor / heat_per_anchor.max(axis=1, keepdims=True).clip(min=1e-12)
    active = H.max(axis=0) > threshold
    dom = np.argmax(H, axis=0)
    labels = np.where(active, dom, -1)
    return labels


# ── Реконструкция / сглаживание ──────────────────────────────────────────────

def reconstruct_delta_from_clusters(verts, N, clusters_per_anchor):
    """δ[v] = μ + (RS - I)(verts[v] - c_rest), взвешенное среднее на пересечениях."""
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


def smooth_labels(labels, faces, n_iter=3):
    """Сглаживание ДИСКРЕТНЫХ меток (групп/зон) по сетке — majority-vote (мода)
    в 1-кольце соседей за n_iter итераций. Аналог Laplacian-смуза для меток:
    убирает «крапинки» на швах зон после UV-NN переноса.

    labels: (N,) int, -1 = непокрытая вершина (не голосует и не меняется).
    Возвращает сглаженные labels (copy)."""
    labels = np.asarray(labels).copy()
    N = len(labels)
    if n_iter <= 0 or N == 0:
        return labels
    F = np.asarray(faces)
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    A = ((A + A.T) > 0).astype(np.float64) + sp.identity(N)  # +self-loop
    valid = labels >= 0
    if not valid.any():
        return labels
    for _ in range(n_iter):
        uniq = np.unique(labels[labels >= 0])
        best = labels.copy()
        best_cnt = np.zeros(N)
        for u in uniq:
            cnt = A @ (labels == u).astype(np.float64)
            upd = cnt > best_cnt
            best[upd] = u
            best_cnt[upd] = cnt[upd]
        labels = np.where(valid, best, labels)  # непокрытые остаются -1
    return labels


def smooth_delta(delta, faces, n_iter=50, alpha=0.5):
    if n_iter <= 0: return delta.copy()
    N = len(delta)
    W = build_neighbor_avg_matrix(N, faces)
    d = delta.copy()
    for _ in range(n_iter):
        d = (1 - alpha) * d + alpha * (W @ d)
    return d


# ── Open3D helpers ────────────────────────────────────────────────────────────

def to_colors(values, cmap):
    v = np.clip(np.asarray(values, dtype=np.float64), 0, None)
    if v.max() > 0: v = v / v.max()
    out = cmap(v)
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
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array([p0, p1]))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]]))
    ls.colors = o3d.utility.Vector3dVector(np.array([color]))
    return ls


def pick_vertices(verts, faces, head_name, max_n=20, point_size=6.0):
    """Selection с densified click-coverage (вершины + face-centroids + edge-mid).
    Любая picked точка маппится в ближайшую mesh-вершину через KDTree."""
    from scipy.spatial import cKDTree

    print(f"\n[{head_name}] Shift+клик до {max_n} точек, Q закроет.")
    print(f"  Кликни в любую точку рядом с нужным местом — ближайший вертекс "
          f"будет выбран автоматически.")

    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(f"Выбери точки — {head_name}", 1000, 800)

    mesh = o3d_mesh(verts, faces)
    vis.add_geometry(mesh)

    extra_pts_list = [verts]
    extra_colors_list = [np.tile([0.95, 0.7, 0.15], (len(verts), 1))]

    if len(faces) > 0:
        face_centroids = verts[faces].mean(axis=1)
        extra_pts_list.append(face_centroids)
        extra_colors_list.append(
            np.tile([0.85, 0.55, 0.10], (len(face_centroids), 1)))

    edge_set = set()
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            edge_set.add((min(a, b), max(a, b)))
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

    tree = cKDTree(verts)
    chosen, seen = [], set()
    for p in picked:
        idx = None
        if hasattr(p, 'coord') and p.coord is not None:
            try:
                xyz = np.asarray(p.coord, dtype=np.float64).reshape(3)
                _, nn_idx = tree.query(xyz, k=1)
                idx = int(nn_idx)
            except Exception:
                idx = None
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
    """Heat diffusion с опциональной авто-остановкой при overlap зон."""
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
    u_prev = u.copy()
    stopped_early = False
    stop_step = steps
    for step_idx in range(steps):
        t0 = time_mod.perf_counter()
        u_prev = u.copy()
        for ai in range(N):
            u[ai] = solve(MM @ u[ai])

        if stop_on_overlap:
            u_max = u.max(axis=1, keepdims=True).clip(min=1e-12)
            u_norm = u / u_max
            active = u_norm > overlap_threshold
            n_active_per_vertex = active.sum(axis=0)
            n_overlap = int((n_active_per_vertex >= 2).sum())
            V_total = u.shape[1]
            overlap_frac = n_overlap / max(V_total, 1)
            if overlap_frac > overlap_fraction:
                print(f"\n⚠ AUTO-STOP: overlap={overlap_frac:.2%} > "
                      f"{overlap_fraction:.0%} на шаге {step_idx+1}/{steps}.")
                print(f"  Откат на шаг {step_idx} (overlap ещё малый), "
                      f"эффективный t={dt * step_idx:.5f}")
                u = u_prev
                stopped_early = True
                stop_step = step_idx
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


def show_meshes_side_by_side(meshes_with_colors_titles, gap_factor=1.3,
                              extra_geometries=None, window_title="Сравнение"):
    """meshes_with_colors_titles: список (verts, faces, colors, label) — рисуем в ряд."""
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

    vis = o3d.visualization.Visualizer()
    ok = vis.create_window(window_name=window_title, width=1600, height=800)
    if not ok:
        print(f"  ⚠ Не удалось создать окно '{window_title}'")
        return
    for g in geoms:
        vis.add_geometry(g)
    vis.get_render_option().mesh_show_back_face = True
    vis.poll_events(); vis.update_renderer()
    vis.run()
    vis.destroy_window()
    try:
        for _ in range(3):
            vis.poll_events()
    except Exception: pass


# ── Пресеты betas ─────────────────────────────────────────────────────────────

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
            return parse_betas_string(spec)


def console_setup_dialog(defaults):
    print("\n" + "═" * 70)
    print("  SETUP — Параметры pipeline (Enter оставит default)")
    print("═" * 70)
    shape = ask_preset(SHAPE_PRESETS, "Форма головы")
    expr  = ask_preset(EXPR_PRESETS, "Экспрессия")
    params = dict(defaults)

    def ask(key, label, cast):
        v_in = input(f"{label} [{defaults[key]}]: ").strip()
        return cast(v_in) if v_in else defaults[key]

    params['n_anchors']      = ask('n_anchors', "Max anchor points", int)
    params['time']           = ask('time', "Diffusion time", float)
    params['steps']          = ask('steps', "Diffusion steps", int)
    params['n_clusters']     = ask('n_clusters', "N clusters max per zone", int)
    params['heat_threshold'] = ask('heat_threshold', "Heat threshold", float)
    params['multi_t_enable'] = bool(int(ask('multi_t_enable',
                                            "Multi-t (1/0)", lambda s: int(s))))
    params['uv_flat'] = bool(int(ask('uv_flat',
                                     "UV flat-проекция (1/0)", lambda s: int(s))))
    params['uv_world_orient'] = bool(int(ask(
        'uv_world_orient',
        "UV world-flat: проекция вдоль нормали, Y вверх (1/0)",
        lambda s: int(s))))
    params['uv_overlay_scale_match'] = bool(int(ask(
        'uv_overlay_scale_match',
        "UV overlay scale-fit (1/0)", lambda s: int(s))))
    params['uv_align_pca_icp'] = bool(int(ask(
        'uv_align_pca_icp',
        "UV PCA+ICP выравнивание островов (1/0)", lambda s: int(s))))
    params['uv_warp_heat'] = bool(int(ask(
        'uv_warp_heat',
        "UV boundary heat-warp границы FBX→FLAME (1/0)", lambda s: int(s))))
    params['uv_warp_heat_t'] = ask(
        'uv_warp_heat_t', "  heat-warp t (время диффузии)", float)
    params['uv_warp_min_dist'] = ask(
        'uv_warp_min_dist',
        "  макс. зазор точка→ребро после подгонки (0=точно на ребро)", float)
    params['uv_warp_line_step'] = int(ask(
        'uv_warp_line_step',
        "  показ линий warp: шаг (0=выкл,1=все,6=каждая 6-я)", lambda s: int(s)))
    params['uv_show_boundary'] = bool(int(ask(
        'uv_show_boundary',
        "Подсветка крайних рёбер островов (FLAME пурпур / FBX жёлт.) (1/0)",
        lambda s: int(s))))
    params['uv_show_heat'] = bool(int(ask(
        'uv_show_heat',
        "Панель распределения тепла границы на подгоняемом меше (1/0)",
        lambda s: int(s))))
    params['uv_export_obj'] = bool(int(ask(
        'uv_export_obj',
        "Экспорт UV-развёрток в OBJ (1/0)", lambda s: int(s))))
    params['transfer_to_fbx'] = bool(int(ask(
        'transfer_to_fbx',
        "Перенос δ FLAME→FBX по UV-NN (1/0)", lambda s: int(s))))
    params['uv_interp_delta'] = bool(int(ask(
        'uv_interp_delta',
        "Барицентрическая интерполяция δ при переносе (1=гладко, 0=NN)",
        lambda s: int(s))))
    params['smooth_transferred_groups'] = bool(int(ask(
        'smooth_transferred_groups',
        "Сглаживать перенесённые группы (1/0)", lambda s: int(s))))
    params['group_smooth_iters'] = int(ask(
        'group_smooth_iters',
        "Итераций сглаживания групп", lambda s: int(s)))
    params['shape'] = shape
    params['expr']  = expr
    params['_ok'] = True
    return params


# ── GUI ────────────────────────────────────────────────────────────────────────

def gui_setup_dialog(defaults):
    """Tkinter диалог. При отсутствии _tkinter → fallback на console."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, simpledialog
    except ImportError:
        print("⚠ tkinter недоступен — использую console-режим.")
        return console_setup_dialog(defaults)

    root = tk.Tk()
    root.title(f"Motion-Groups v{__version__} — Setup (HEAD 1)")
    root.geometry("600x720")
    root.resizable(True, True)
    root.minsize(520, 400)

    container = tk.Frame(root)
    container.pack(fill='both', expand=True)
    canvas = tk.Canvas(container, highlightthickness=0)
    scrollbar = tk.Scrollbar(container, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)

    frame = tk.Frame(canvas, padx=20, pady=15)
    frame.columnconfigure(1, weight=1)
    window_id = canvas.create_window((0, 0), window=frame, anchor='nw')

    def _on_frame_configure(_e):
        canvas.configure(scrollregion=canvas.bbox('all'))
    frame.bind('<Configure>', _on_frame_configure)

    def _on_canvas_configure(event):
        canvas.itemconfig(window_id, width=event.width)
    canvas.bind('<Configure>', _on_canvas_configure)

    def _on_mousewheel(event):
        canvas.yview_scroll(int(-1 * event.delta), 'units')
    canvas.bind_all('<MouseWheel>', _on_mousewheel)

    def _on_destroy(_e):
        try: canvas.unbind_all('<MouseWheel>')
        except Exception: pass
    root.bind('<Destroy>', _on_destroy)

    vars_ = {}

    def add_label_entry(label, default, row, width=26):
        tk.Label(frame, text=label, anchor='w', justify='left').grid(
            row=row, column=0, sticky='w', padx=(0, 12), pady=4)
        v = tk.StringVar(value=str(default))
        tk.Entry(frame, textvariable=v, width=width).grid(
            row=row, column=1, pady=4, sticky='ew')
        return v

    def add_label_combo(label, options, default_idx, row, width=26):
        tk.Label(frame, text=label, anchor='w').grid(
            row=row, column=0, sticky='w', padx=(0, 12), pady=4)
        v = tk.StringVar(value=options[default_idx])
        ttk.Combobox(frame, textvariable=v, values=options, width=width,
                      state='readonly').grid(row=row, column=1, pady=4, sticky='ew')
        return v

    row = 0
    def section(text):
        nonlocal row
        tk.Label(frame, text=text, font=("Arial", 11, "bold")).grid(
            row=row, column=0, columnspan=2, sticky='w', pady=(12, 4))
        row += 1

    # ── Пресеты настроек ──
    PRESETS_DIR = (Path(__file__).resolve().parent / "presets")
    PRESETS_DIR.mkdir(parents=True, exist_ok=True)
    LAST_USED = "_last_used"

    def _list_presets():
        return [""] + sorted([p.stem for p in PRESETS_DIR.glob("*.json")
                                if p.stem != LAST_USED])

    def _do_save_preset(name):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name).strip("_")
        if not safe: return None
        with open(PRESETS_DIR / f"{safe}.json", 'w') as f:
            json.dump({k: v.get() for k, v in vars_.items()}, f,
                      indent=2, ensure_ascii=False)
        return safe

    def _on_save_preset():
        name = simpledialog.askstring("Save preset", "Имя пресета:", parent=root)
        if not name: return
        saved = _do_save_preset(name)
        if saved:
            preset_combo['values'] = _list_presets()
            preset_var.set(saved)
            messagebox.showinfo("Saved", f"Preset '{saved}' сохранён")

    def _on_load_preset():
        name = preset_var.get()
        if not name: return
        path = PRESETS_DIR / f"{name}.json"
        if not path.exists():
            messagebox.showerror("Error", f"Preset {name} не найден"); return
        try:
            with open(path) as f: values = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", str(e)); return
        for k, val in values.items():
            if k in vars_:
                try: vars_[k].set(val)
                except Exception: pass

    def _on_delete_preset():
        name = preset_var.get()
        if not name: return
        if not messagebox.askyesno("Delete", f"Удалить пресет '{name}'?"): return
        (PRESETS_DIR / f"{name}.json").unlink(missing_ok=True)
        preset_combo['values'] = _list_presets()
        preset_var.set("")

    def _auto_apply_preset(_e=None):
        if preset_var.get(): _on_load_preset()

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
    tk.Button(preset_row, text="Load", command=_on_load_preset,
              width=6).grid(row=0, column=2, padx=2)
    tk.Button(preset_row, text="Save as…", command=_on_save_preset,
              width=10).grid(row=0, column=3, padx=2)
    tk.Button(preset_row, text="✕", command=_on_delete_preset,
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

    section("── Кластеризация ──")
    vars_['position_weight'] = add_label_entry(
        "Position weight (0=motion only):", defaults['position_weight'], row); row += 1
    vars_['n_clusters'] = add_label_entry(
        "N clusters max per zone:", defaults['n_clusters'], row); row += 1
    vars_['heat_threshold'] = add_label_entry(
        "Heat threshold:", defaults['heat_threshold'], row); row += 1
    _cl_methods = ['kmeans', 'agglomerative']
    _cl_def = defaults.get('clustering_method', 'kmeans')
    vars_['clustering_method'] = add_label_combo(
        "Clustering method:", _cl_methods,
        _cl_methods.index(_cl_def) if _cl_def in _cl_methods else 0, row); row += 1
    vars_['cluster_similarity_threshold'] = add_label_entry(
        "  similarity threshold (agglomerative, 0.1-0.5):",
        defaults.get('cluster_similarity_threshold', 0.3), row); row += 1

    section("── Сглаживание ──")
    vars_['smooth_iters'] = add_label_entry(
        "Laplacian: smooth iters:", defaults['smooth_iters'], row); row += 1
    vars_['smooth_alpha'] = add_label_entry(
        "Laplacian: smooth alpha (0..1):", defaults['smooth_alpha'], row); row += 1

    section("── Anchor-точки ──")
    vars_['n_anchors'] = add_label_entry(
        "Max anchor points:", defaults['n_anchors'], row); row += 1

    # ── FBX (опционально) — тот же heat/зональный анализ, без матчинга ──
    section("── FBX голова (опц., heat-зоны) ──")
    vars_['fbx_path'] = tk.StringVar(value=defaults.get('fbx_path', ''))
    tk.Label(frame, text="FBX path:", anchor='w').grid(
        row=row, column=0, sticky='w', padx=(0, 12), pady=4)
    fbx_row = tk.Frame(frame)
    fbx_row.grid(row=row, column=1, sticky='ew', pady=4)
    fbx_row.columnconfigure(0, weight=1)
    tk.Entry(fbx_row, textvariable=vars_['fbx_path']).grid(
        row=0, column=0, sticky='ew', padx=(0, 5))

    def make_browse(var):
        def browse():
            import platform, threading
            if platform.system() != "Darwin":
                try:
                    from tkinter import filedialog
                    root.update_idletasks()
                    path = filedialog.askopenfilename(
                        parent=root, title="Выбери mesh-файл",
                        filetypes=[("Mesh files", "*.fbx *.obj *.ply"),
                                   ("All files", "*.*")])
                    if path: var.set(path)
                except Exception as e:
                    print(f"Picker error: {e}")
                return

            def worker():
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
                        root.after(0, lambda p=path: var.set(p))
                except Exception as e:
                    print(f"Picker worker error: {e}")

            threading.Thread(target=worker, daemon=True).start()
        return browse

    tk.Button(fbx_row, text="Browse...", command=make_browse(vars_['fbx_path']),
              width=10).grid(row=0, column=1, padx=(0, 3))
    tk.Button(fbx_row, text="✕", command=lambda: vars_['fbx_path'].set(""),
              width=2).grid(row=0, column=2)
    row += 1

    # доп. FBX-источники 2 и 3 (только для Pipeline viewer: перенос на 3 головы)
    for slot in (2, 3):
        key = f'fbx_path{slot}'
        vars_[key] = tk.StringVar(value=defaults.get(key, ''))
        tk.Label(frame, text=f"FBX {slot} (для viewer):", anchor='w').grid(
            row=row, column=0, sticky='w', padx=(0, 12), pady=4)
        rw = tk.Frame(frame); rw.grid(row=row, column=1, sticky='ew', pady=4)
        rw.columnconfigure(0, weight=1)
        tk.Entry(rw, textvariable=vars_[key]).grid(
            row=0, column=0, sticky='ew', padx=(0, 5))
        tk.Button(rw, text="Browse...", command=make_browse(vars_[key]),
                  width=10).grid(row=0, column=1, padx=(0, 3))
        tk.Button(rw, text="✕", command=lambda k=key: vars_[k].set(""),
                  width=2).grid(row=0, column=2)
        row += 1

    def open_spectral():
        import threading, subprocess, sys
        here = Path(__file__).resolve().parent
        flame = vars_['flame_path'].get().strip() if 'flame_path' in vars_ \
            else FLAME_PKL
        fbx = vars_['fbx_path'].get().strip()
        cmd = [sys.executable, str(here / "spectral_descriptors.py"),
               "--flame", flame]
        if fbx:
            cmd += ["--fbx", fbx]

        def worker():
            try:
                subprocess.run(cmd, cwd=str(here.parents[2]))
            except Exception as e:
                print(f"Spectral viewer error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    tk.Button(frame, text="🔬 Spectral descriptors (HKS / WKS)",
              command=open_spectral, bg="#1565C0", fg="white",
              font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=2, sticky='we', pady=(2, 2)); row += 1

    def open_pipeline_viewer():
        import threading, subprocess, sys
        here = Path(__file__).resolve().parent
        # до 3 FBX-источников (fbx_path, fbx_path2, fbx_path3)
        fbx_list = []
        for key in ('fbx_path', 'fbx_path2', 'fbx_path3'):
            if key in vars_:
                v = vars_[key].get().strip()
                if v:
                    fbx_list.append(v)
        custom = vars_['custom_betas'].get().strip() if 'custom_betas' in vars_ \
            else ""
        # форма головы (shape preset) → строка "idx:val,..." для --shape
        shape_str = ""
        if 'shape' in vars_:
            try:
                sidx = int(vars_['shape'].get().split(':')[0])
                sdict = SHAPE_PRESETS.get(sidx, ('', {}))[1]
                shape_str = ",".join(f"{k}:{v}" for k, v in sdict.items())
            except Exception:
                shape_str = ""
        cmd = [sys.executable, str(here / "pipeline_viewer.py"),
               "--flame", FLAME_PKL]
        for f in fbx_list:
            cmd += ["--fbx", f]
        if custom:
            cmd += ["--expr", custom]
        if shape_str:
            cmd += ["--shape", shape_str]

        def worker():
            try:
                subprocess.run(cmd, cwd=str(here.parents[2]))
            except Exception as e:
                print(f"Pipeline viewer error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    tk.Button(frame, text="▶ Pipeline viewer (пошагово, перенос)",
              command=open_pipeline_viewer, bg="#6A1B9A", fg="white",
              font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=2, sticky='we', pady=(2, 2)); row += 1

    # ── Батч-перенос: папка FBX → HDF5 (data/dataset.h5) ──
    vars_['batch_fbx_dir'] = tk.StringVar(value=defaults.get('batch_fbx_dir', ''))
    brow = tk.Frame(frame); brow.grid(row=row, column=1, sticky='ew', pady=2)
    brow.columnconfigure(0, weight=1)
    tk.Label(frame, text="Папка FBX (батч):", anchor='w').grid(
        row=row, column=0, sticky='w', padx=(0, 12))
    tk.Entry(brow, textvariable=vars_['batch_fbx_dir']).grid(
        row=0, column=0, sticky='ew', padx=(0, 5))

    def browse_batch_dir():
        from tkinter import filedialog
        d = filedialog.askdirectory(parent=root, title="Папка с FBX-головами")
        if d:
            vars_['batch_fbx_dir'].set(d)
    tk.Button(brow, text="...", width=3, command=browse_batch_dir).grid(
        row=0, column=1)
    row += 1

    def run_batch_transfer():
        import threading, subprocess, sys
        here = Path(__file__).resolve().parent
        fbx_dir = vars_['batch_fbx_dir'].get().strip()
        if not fbx_dir:
            print("Батч: укажи папку FBX.")
            return
        expr = vars_['custom_betas'].get().strip() if 'custom_betas' in vars_ \
            else ""
        if not expr:
            print("Батч: впиши эмоцию в Custom betas (напр. 300:8.0).")
            return
        shape_str = ""
        if 'shape' in vars_:
            try:
                sidx = int(vars_['shape'].get().split(':')[0])
                sdict = SHAPE_PRESETS.get(sidx, ('', {}))[1]
                shape_str = ",".join(f"{k}:{v}" for k, v in sdict.items())
            except Exception:
                shape_str = ""
        cmd = [sys.executable, str(here / "batch_transfer.py"),
               "--fbx-dir", fbx_dir, "--expr", expr,
               "--out", "data/dataset.h5", "--flame", FLAME_PKL]
        if shape_str:
            cmd += ["--shape", shape_str]

        def worker():
            print(f"Батч-перенос: {fbx_dir} → data/dataset.h5 ...")
            try:
                subprocess.run(cmd, cwd=str(here.parents[2]))
            except Exception as e:
                print(f"Batch transfer error: {e}")
        threading.Thread(target=worker, daemon=True).start()

    tk.Button(frame, text="📦 Батч-перенос папки FBX → data/dataset.h5",
              command=run_batch_transfer, bg="#37474F", fg="white",
              font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=2, sticky='we', pady=(2, 6)); row += 1

    def open_head_generator():
        import threading, subprocess, sys
        here = Path(__file__).resolve().parent
        cmd = [sys.executable, str(here / "head_generator.py")]

        def worker():
            try:
                subprocess.run(cmd, cwd=str(here.parents[2]))
            except Exception as e:
                print(f"Head generator error: {e}")
        threading.Thread(target=worker, daemon=True).start()

    tk.Button(frame, text="🧬 Генератор голов (N случайных FLAME)",
              command=open_head_generator, bg="#00695C", fg="white",
              font=("Arial", 10, "bold")).grid(
        row=row, column=0, columnspan=2, sticky='we', pady=(2, 6)); row += 1

    section("── Multi-t Heat Enrichment ──")
    vars_['multi_t_enable'] = tk.BooleanVar(
        value=bool(defaults.get('multi_t_enable', False)))
    tk.Checkbutton(frame,
                    text="Multi-t: кластеризовать в hard-partition зонах (argmax)",
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
                    text="  Маскировать multi-t зоны по single-t reach",
                    variable=vars_['multi_t_mask_by_single_t']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1

    section("── UV settings ──")
    vars_['uv_flat'] = tk.BooleanVar(value=bool(defaults.get('uv_flat', False)))
    tk.Checkbutton(frame,
                    text="Flat: разворачивать зону планарной проекцией "
                         "(как Flat в Cinema4D)",
                    variable=vars_['uv_flat']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['uv_world_orient'] = tk.BooleanVar(
        value=bool(defaults.get('uv_world_orient', False)))
    tk.Checkbutton(frame,
                    text="World-flat: проекция вдоль нормали зоны, мировой Y "
                         "вверх (детерминир. ориентация)",
                    variable=vars_['uv_world_orient']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['uv_overlay_scale_match'] = tk.BooleanVar(
        value=bool(defaults.get('uv_overlay_scale_match', True)))
    tk.Checkbutton(frame,
                    text="Overlay scale-fit: подгонять масштаб зон 2-й головы "
                         "к 1-й (3-я панель)",
                    variable=vars_['uv_overlay_scale_match']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['uv_align_pca_icp'] = tk.BooleanVar(
        value=bool(defaults.get('uv_align_pca_icp', False)))
    tk.Checkbutton(frame,
                    text="PCA+ICP выравнивание островов 2-й головы к 1-й "
                         "(overlay + перенос)",
                    variable=vars_['uv_align_pca_icp']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['uv_warp_heat'] = tk.BooleanVar(
        value=bool(defaults.get('uv_warp_heat', False)))
    tk.Checkbutton(frame,
                    text="Boundary heat-warp: тянуть границу острова FBX к "
                         "границе FLAME (тепловое поле)",
                    variable=vars_['uv_warp_heat']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['uv_warp_heat_t'] = add_label_entry(
        "  heat-warp t (время диффузии влияния):",
        defaults.get('uv_warp_heat_t', 0.05), row); row += 1
    vars_['uv_warp_min_dist'] = add_label_entry(
        "  макс. зазор точка→ребро после подгонки (0=точно на ребро):",
        defaults.get('uv_warp_min_dist', 0.0), row); row += 1
    vars_['uv_warp_line_step'] = add_label_entry(
        "  показ линий warp: шаг (0=выкл, 1=все, 6=каждая 6-я):",
        defaults.get('uv_warp_line_step', 0), row); row += 1
    vars_['uv_show_boundary'] = tk.BooleanVar(
        value=bool(defaults.get('uv_show_boundary', True)))
    tk.Checkbutton(frame,
                    text="Подсветка крайних рёбер островов (FLAME пурпур / FBX жёлт.)",
                    variable=vars_['uv_show_boundary']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['uv_show_heat'] = tk.BooleanVar(
        value=bool(defaults.get('uv_show_heat', False)))
    tk.Checkbutton(frame,
                    text="Панель распределения тепла границы на подгоняемом меше (FBX)",
                    variable=vars_['uv_show_heat']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['uv_export_obj'] = tk.BooleanVar(
        value=bool(defaults.get('uv_export_obj', True)))
    tk.Checkbutton(frame,
                    text="Экспорт UV-развёрток зон в OBJ (плоский меш, цвет по зоне)",
                    variable=vars_['uv_export_obj']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['transfer_to_fbx'] = tk.BooleanVar(
        value=bool(defaults.get('transfer_to_fbx', True)))
    tk.Checkbutton(frame,
                    text="Перенос δ FLAME→FBX по UV-NN (нужен FBX + multi-t)",
                    variable=vars_['transfer_to_fbx']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['smooth_transferred_groups'] = tk.BooleanVar(
        value=bool(defaults.get('smooth_transferred_groups', True)))
    tk.Checkbutton(frame,
                    text="Сглаживать перенесённые группы (majority-vote)",
                    variable=vars_['smooth_transferred_groups']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['uv_interp_delta'] = tk.BooleanVar(
        value=bool(defaults.get('uv_interp_delta', True)))
    tk.Checkbutton(frame,
                    text="Барицентрическая интерполяция δ при переносе "
                         "(гладко внутри зоны; иначе UV-NN)",
                    variable=vars_['uv_interp_delta']).grid(
        row=row, column=0, columnspan=2, sticky='w', pady=2); row += 1
    vars_['group_smooth_iters'] = add_label_entry(
        "  итераций сглаживания групп:",
        defaults.get('group_smooth_iters', 8), row); row += 1

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
            result['n_anchors']       = int(vars_['n_anchors'].get())
            result['fbx_path']        = vars_['fbx_path'].get().strip()
            result['stop_diffusion_on_overlap']   = bool(vars_['stop_diffusion_on_overlap'].get())
            result['diffusion_overlap_threshold'] = float(vars_['diffusion_overlap_threshold'].get())
            result['diffusion_overlap_fraction']  = float(vars_['diffusion_overlap_fraction'].get())
            result['multi_t_enable']           = bool(vars_['multi_t_enable'].get())
            result['multi_t_n_times']          = int(vars_['multi_t_n_times'].get())
            result['multi_t_n_eigs']           = int(vars_['multi_t_n_eigs'].get())
            result['multi_t_mask_by_single_t'] = bool(vars_['multi_t_mask_by_single_t'].get())
            result['uv_flat']                  = bool(vars_['uv_flat'].get())
            result['uv_world_orient']          = bool(
                vars_['uv_world_orient'].get())
            result['uv_overlay_scale_match']   = bool(
                vars_['uv_overlay_scale_match'].get())
            result['uv_align_pca_icp']         = bool(
                vars_['uv_align_pca_icp'].get())
            result['uv_warp_heat']             = bool(
                vars_['uv_warp_heat'].get())
            result['uv_warp_heat_t']           = float(
                vars_['uv_warp_heat_t'].get())
            result['uv_warp_min_dist']         = float(
                vars_['uv_warp_min_dist'].get())
            result['uv_warp_line_step']        = int(
                vars_['uv_warp_line_step'].get())
            result['uv_show_boundary']         = bool(
                vars_['uv_show_boundary'].get())
            result['uv_show_heat']             = bool(
                vars_['uv_show_heat'].get())
            result['uv_export_obj']            = bool(
                vars_['uv_export_obj'].get())
            result['transfer_to_fbx']          = bool(
                vars_['transfer_to_fbx'].get())
            result['uv_interp_delta']          = bool(
                vars_['uv_interp_delta'].get())
            result['smooth_transferred_groups'] = bool(
                vars_['smooth_transferred_groups'].get())
            result['group_smooth_iters']       = int(
                vars_['group_smooth_iters'].get())
            try: _do_save_preset(LAST_USED)
            except Exception as e: print(f"  ⚠ auto-save _last_used: {e}")
            result['_ok'] = True
            root.destroy()
        except Exception as e:
            messagebox.showerror("Ошибка ввода", f"Неверное значение: {e}")

    def on_cancel():
        root.destroy()

    row += 1
    tk.Label(frame, text="↓ Нажми START чтобы запустить pipeline ↓",
              fg="#2E7D32", font=("Arial", 11, "bold")).grid(
        row=row, column=0, columnspan=2, pady=(15, 5)); row += 1
    btn_frame = tk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=2, pady=(5, 15))
    tk.Button(btn_frame, text="START ▶", command=on_start, bg="#2E7D32", fg="white",
              font=("Arial", 14, "bold"), width=14, height=2).pack(side='left', padx=10)
    tk.Button(btn_frame, text="Отмена", command=on_cancel,
              font=("Arial", 11), width=10, height=2).pack(side='left', padx=10)

    root.protocol("WM_DELETE_WINDOW", on_start)

    last_used_path = PRESETS_DIR / f"{LAST_USED}.json"
    if last_used_path.exists():
        try:
            with open(last_used_path) as f: last = json.load(f)
            for k, val in last.items():
                if k in vars_:
                    try: vars_[k].set(val)
                    except Exception: pass
            print(f"  ✓ Auto-loaded предыдущие настройки")
        except Exception as e:
            print(f"  ⚠ auto-load _last_used: {e}")

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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flame", default=FLAME_PKL)
    ap.add_argument("--no-gui", action="store_true",
                    help="Консольный ввод вместо GUI")
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
        'custom_betas':    '',
        'fbx_path':        '',
        'fbx_path2':       '',
        'fbx_path3':       '',
        'batch_fbx_dir':   '',
        'clustering_method': 'kmeans',
        'cluster_similarity_threshold': 0.3,
        'stop_diffusion_on_overlap':   False,
        'diffusion_overlap_threshold': 0.05,
        'diffusion_overlap_fraction':  0.02,
        'multi_t_enable':              False,
        'multi_t_n_times':             8,
        'multi_t_n_eigs':              80,
        'multi_t_mask_by_single_t':    True,
        'uv_flat':                     False,
        'uv_world_orient':             False,
        'uv_overlay_scale_match':      True,
        'uv_align_pca_icp':            False,
        'uv_warp_heat':                False,
        'uv_warp_heat_t':              0.05,
        'uv_warp_min_dist':            0.0,
        'uv_warp_line_step':           0,
        'uv_show_boundary':            True,
        'uv_show_heat':                False,
        'uv_export_obj':               True,
        'transfer_to_fbx':             True,
        'uv_interp_delta':             True,
        'smooth_transferred_groups':   True,
        'group_smooth_iters':          8,
    }
    for k, default in DEFAULTS.items():
        if k == 'custom_betas': continue
        ap.add_argument(f"--{k.replace('_','-')}", type=type(default), default=None)
    args = ap.parse_args()
    for k in DEFAULTS:
        cli_v = getattr(args, k, None)
        if cli_v is not None:
            DEFAULTS[k] = cli_v

    print("\n╔═══════════════════════════════════════════════════════════════╗")
    print(f"║  MOTION-GROUPS v{__version__} — HEAD 1 (source) analysis          ║")
    print("╚═══════════════════════════════════════════════════════════════╝")

    print(f"\nЗагружаю FLAME: {args.flame}")
    v_t, sd, faces = load_flame(args.flame)
    print(f"  {len(v_t)} вершин, {len(faces)} граней")

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    OUT_DIR = Path("python/scripts/debug_output") / f"run_v6_{ts}"
    OUT_HEAD1 = OUT_DIR / "head1"
    OUT_HEAD1.mkdir(parents=True, exist_ok=True)
    print(f"  Сохраняю промежуточные данные в: {OUT_DIR}/")

    # ── SETUP ─────────────────────────────────────────────────────────────────
    if args.no_gui:
        params = console_setup_dialog(DEFAULTS)
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

    print(f"\nПараметры:")
    print(f"  shape betas:  {shape}")
    print(f"  expr  betas:  {expr}")
    print(f"  diffusion:    t={t_anim}, steps={num_steps}")
    print(f"  clustering:   pw={pos_weight}, n_clusters={n_clusters}, "
          f"heat_thresh={heat_thresh}")
    print(f"  smoothing:    iters={smooth_iters}, alpha={smooth_alpha}")
    print(f"  multi-t:      {params.get('multi_t_enable', False)}")

    # ── HEAD 1 (FLAME source) ─────────────────────────────────────────────
    v_raw = apply_betas(v_t, sd, shape)
    verts = normalize_bbox(v_raw)

    print("\n── ВЫБОР anchor-точек HEAD 1 (Open3D, Shift+click) ──")
    src = pick_vertices(verts, faces, "Голова 1 (FLAME)", max_n=n_anchors_max)
    if len(src) == 0:
        print("Не выбрано ни одной точки — выход.")
        return

    def normalized_expr(v_raw_rest, betas_full):
        v_raw_e = apply_betas(v_t, sd, betas_full)
        m = v_raw_rest.mean(0)
        d = np.linalg.norm((v_raw_rest - m).max(0) - (v_raw_rest - m).min(0))
        return (v_raw_e - m) / (d + 1e-12)

    head_expr = normalized_expr(v_raw, {**shape, **expr})
    delta_native = head_expr - verts
    print(f"\n  max ||δ_native|| = {np.linalg.norm(delta_native, axis=1).max():.4f}")
    print(f"  mean ||δ_native|| = {np.linalg.norm(delta_native, axis=1).mean():.4f}")
    save_metadata_json(OUT_DIR / "metadata.json", params, shape, expr, len(src), src)

    use_multi_t = bool(params.get('multi_t_enable', False))
    results = []
    res_head1 = run_zone_analysis(verts, faces, src, params, OUT_HEAD1,
                                  "HEAD 1 (FLAME)",
                                  delta_native=delta_native, head_expr=head_expr)
    results.append(res_head1)

    # ── FBX (опционально) — тот же heat/зональный анализ, БЕЗ матчинга ──────
    fbx_path = params.get('fbx_path', '').strip()
    if fbx_path:
        print("\n" + "═" * 70)
        print(f"  FBX АНАЛИЗ (heat-зоны, без матчинга/переноса): {fbx_path}")
        print("═" * 70)
        verts_fbx = faces_fbx = None
        try:
            v_fbx_raw, faces_fbx = load_custom_mesh(fbx_path)
            verts_fbx = normalize_bbox(v_fbx_raw)
            print(f"  FBX: {len(verts_fbx)} вершин, {len(faces_fbx)} граней")
        except Exception as e:
            print(f"  ⚠ Ошибка загрузки FBX (пропускаю): {e}")
        if verts_fbx is not None:
            OUT_FBX = OUT_DIR / "fbx"
            OUT_FBX.mkdir(parents=True, exist_ok=True)
            print("\n── ВЫБОР anchor-точек FBX (Open3D, Shift+click) ──")
            src_fbx = pick_vertices(verts_fbx, faces_fbx, "FBX target",
                                     max_n=n_anchors_max)
            if len(src_fbx) == 0:
                print("  FBX: anchor'ы не выбраны — пропускаю FBX.")
            else:
                # FBX не имеет δ (нет модели экспрессии) → анализируем только
                # heat-зоны. delta_native=None → кластеризация по heat+позиции.
                res_fbx = run_zone_analysis(verts_fbx, faces_fbx, src_fbx,
                                            params, OUT_FBX, "FBX",
                                            delta_native=None, head_expr=None)
                results.append(res_fbx)

    # ── ОБЩЕЕ ФИНАЛЬНОЕ ОКНО (все головы) ──────────────────────────────────
    show_final_combined(results, use_multi_t)

    # Перенос δ FLAME→FBX считаем ЗАРАНЕЕ (нужен и для 4-й UV-панели, и для
    # окна deformed) — только при multi-t, наличии FBX и включённом переносе.
    do_transfer = (use_multi_t and len(results) >= 2
                   and params.get('transfer_to_fbx', True))
    tr = None
    if do_transfer:
        try:
            tr = transfer_deformations_uv(
                results[0], results[1],
                flat=_uv_mode(params),
                align_pca_icp=bool(params.get('uv_align_pca_icp', False)),
                warp_heat=bool(params.get('uv_warp_heat', False)),
                warp_heat_t=float(params.get('uv_warp_heat_t', 0.05)),
                warp_min_dist=float(params.get('uv_warp_min_dist', 0.0)),
                interp_delta=bool(params.get('uv_interp_delta', True)))
            # Сглаживание перенесённых ГРУПП (дискретные метки → majority-vote),
            # чтобы убрать крапинки на швах зон (видно в 4-й UV-панели и дампе).
            if tr is not None and params.get('smooth_transferred_groups', True):
                it = int(params.get('group_smooth_iters', 8))
                print(f"  Сглаживание перенесённых групп "
                      f"(majority-vote, iters={it})...")
                tr['gcid'] = smooth_labels(tr['gcid'], results[1]['faces'],
                                           n_iter=it)
                tr['zone'] = smooth_labels(tr['zone'], results[1]['faces'],
                                           n_iter=it)
        except Exception:
            import traceback
            print("  ⚠ Ошибка в переносе FLAME→FBX (расчёт):")
            traceback.print_exc()

    # ── UV-РАЗВЁРТКА masked-зон (только multi-t) ───────────────────────────
    if use_multi_t:
        if params.get('uv_export_obj', True):
            try:
                export_uv_layouts_obj(
                    results, OUT_DIR / "uv",
                    flat=_uv_mode(params))
            except Exception:
                import traceback
                print("  ⚠ Ошибка экспорта UV в OBJ:")
                traceback.print_exc()
        try:
            show_uv_unwrap_combined(
                results, flat=_uv_mode(params),
                overlay_scale_match=bool(
                    params.get('uv_overlay_scale_match', True)),
                transfer=tr, transfer_dst_index=1,
                align_pca_icp=bool(params.get('uv_align_pca_icp', False)),
                warp_line_step=int(params.get('uv_warp_line_step', 0)),
                warp_min_dist=float(params.get('uv_warp_min_dist', 0.0)),
                show_boundary=bool(params.get('uv_show_boundary', True)),
                show_heat=bool(params.get('uv_show_heat', False)),
                warp_heat_t=float(params.get('uv_warp_heat_t', 0.05)),
                warp_heat=bool(params.get('uv_warp_heat', False)))
        except Exception:
            import traceback
            print("  ⚠ Ошибка в UV-развёртке:")
            traceback.print_exc()
    else:
        print("\n(UV-развёртка показывается только при включённом multi-t.)")

    # ── ПЕРЕНОС ДЕФОРМАЦИЙ FLAME → FBX: применение + окно (multi-t + FBX) ────
    if do_transfer:
        try:
            transfer_and_show_fbx(
                results[0], results[1], OUT_DIR / "fbx",
                flat=_uv_mode(params),
                smooth_iters=int(params.get('smooth_iters', 3)),
                smooth_alpha=float(params.get('smooth_alpha', 0.5)), tr=tr,
                align_pca_icp=bool(params.get('uv_align_pca_icp', False)))
        except Exception:
            import traceback
            print("  ⚠ Ошибка в переносе FLAME→FBX:")
            traceback.print_exc()
    elif not use_multi_t and params.get('fbx_path', '').strip():
        print("\n(Перенос δ на FBX доступен только при включённом multi-t.)")

    print("\n✓ Готово. Все дампы в:", OUT_DIR)


def run_zone_analysis(verts, faces, src, params, out_dir, label,
                       delta_native=None, head_expr=None):
    """Полный heat/зональный анализ одного меша (HEAD 1 или FBX).

    delta_native is not None (HEAD 1) → есть деформация:
        показываем ОКНО rest|deformed, ОКНО native|recon|smoothed,
        кластеризация по motion (δ).
    delta_native is None (FBX) → деформации нет:
        кластеризация по heat+позиции, показываем только диффузию + финал зоны.
    """
    N_verts = len(verts)
    N_anchors = len(src)
    has_deform = delta_native is not None
    if not has_deform:
        delta_native = np.zeros((N_verts, 3))
        head_expr = verts

    t_anim       = params['time']
    num_steps    = params['steps']
    fps          = params['fps']
    n_clusters   = params['n_clusters']
    heat_thresh  = params['heat_threshold']
    pos_weight   = params['position_weight']
    smooth_iters = params['smooth_iters']
    smooth_alpha = params['smooth_alpha']
    use_multi_t  = bool(params.get('multi_t_enable', False))

    save_matrix_csv(out_dir / "verts_rest.csv", verts, header="x,y,z")
    save_matrix_csv(out_dir / "faces.csv", faces, header="v0,v1,v2")
    save_matrix_csv(out_dir / "anchor_indices.csv",
                     np.array(src).reshape(-1, 1), header="vertex_index")
    if has_deform:
        save_matrix_csv(out_dir / "delta_native.csv", delta_native, header="dx,dy,dz")

    print(f"\nСтрою Laplacian для {label}...")
    L, MM = build_operators(verts, faces)

    # ── ДИФФУЗИЯ (ОКНО 1) ─────────────────────────────────────────────────────
    print(f"\n── [{label}] ОКНО 1: анимация диффузии ──")
    heat = animate_diffusion(
        verts, faces, L, MM, src, t_anim, num_steps, fps=fps,
        stop_on_overlap=bool(params.get('stop_diffusion_on_overlap', False)),
        overlap_threshold=float(params.get('diffusion_overlap_threshold', 0.05)),
        overlap_fraction=float(params.get('diffusion_overlap_fraction', 0.02)),
    )
    print(f"  heat shape: {heat.shape}, max per anchor: {heat.max(axis=1).round(4)}")
    heat_columns = ",".join([f"anchor_{a}" for a in range(N_anchors)])
    save_matrix_csv(out_dir / "heat.csv", heat.T, header=heat_columns)

    # ── MULTI-T ENRICHMENT (+ single-t маскировка) ────────────────────────────
    heat_single_t = heat.copy()
    if use_multi_t:
        mt_n_times = int(params.get('multi_t_n_times', 8))
        mt_n_eigs  = int(params.get('multi_t_n_eigs', 80))
        print(f"\n  ── [{label}] MULTI-T ENRICHMENT (T={mt_n_times}, k_eigs={mt_n_eigs}) ──")
        heat_enriched, _times_mt = enrich_heat_multi_t(
            verts=verts, faces=faces,
            anchor_indices=list(np.asarray(src).ravel()),
            n_times=mt_n_times, n_eigs=mt_n_eigs,
            smooth_iters=5, smooth_alpha=0.5, mesh_label=label,
        )
        heat = heat_enriched
        if bool(params.get('multi_t_mask_by_single_t', True)):
            print(f"  Маскирую multi-t зоны по single-t reach...")
            h1_norm = heat_single_t / heat_single_t.max(axis=1, keepdims=True).clip(min=1e-12)
            active_mask = h1_norm.max(axis=0) > heat_thresh
            n_active = int(active_mask.sum())
            print(f"    Single-t active zone: {n_active}/{N_verts} верш. "
                  f"({100*n_active/N_verts:.1f}%) — вне зоны heat → 0")
            heat[:, ~active_mask] = 0.0
        try:
            save_matrix_csv(out_dir / "heat_multi_t_enriched.csv",
                             heat.T, header=heat_columns)
        except Exception as e:
            print(f"    ⚠ Не удалось сохранить multi-t dump: {e}")

    # ── ОКНО 2: rest | native deformed (только если есть δ) ────────────────────
    col_native = to_colors(np.linalg.norm(delta_native, axis=1), CMAP_DISP)
    if has_deform:
        print(f"\n── [{label}] ОКНО 2: rest vs native deformed ──")
        show_meshes_side_by_side([
            (verts, faces, np.tile([0.85, 0.75, 0.68], (N_verts, 1)), "rest"),
            (head_expr, faces, col_native, "native deformed"),
        ], window_title=f"[{label}] rest | native deformed (Q → продолжить)")

    # ── КЛАСТЕРИЗАЦИЯ ─────────────────────────────────────────────────────────
    print(f"\n── [{label}] Кластеризация ──")
    cluster_method = params.get('clustering_method', 'kmeans')
    sim_thresh     = float(params.get('cluster_similarity_threshold', 0.3))
    partition = None

    if use_multi_t:
        print(f"  Режим: PER-ANCHOR clustering в multi-t HARD-PARTITION zones")
        partition = _argmax_partition(heat, threshold=heat_thresh)
        for a in range(N_anchors):
            n_in_zone = int((partition == a).sum())
            print(f"    anchor {a}: {n_in_zone} verts "
                  f"({100*n_in_zone/max(N_verts,1):.1f}%)")
        unass = int((partition == -1).sum())
        print(f"    unassigned: {unass} ({100*unass/max(N_verts,1):.1f}%)")
        clusters_per_anchor = []
        for a in range(N_anchors):
            masked_heat = heat[a].copy()
            masked_heat[partition != a] = 0.0
            cls = cluster_zone(
                masked_heat, delta_native, verts, anchor_idx=a,
                heat_threshold=heat_thresh, n_clusters_max=n_clusters,
                position_weight=pos_weight,
                clustering_method=cluster_method, similarity_threshold=sim_thresh,
            )
            clusters_per_anchor.append(cls)
            print(f"  Anchor #{a}: {len(cls)} групп (в hard-partition зоне)")
    else:
        print(f"  Режим: PER-ANCHOR clustering всей головы (single-t зоны)")
        clusters_per_anchor = []
        for a in range(N_anchors):
            cls = cluster_zone(
                heat[a], delta_native, verts, anchor_idx=a,
                heat_threshold=heat_thresh, n_clusters_max=n_clusters,
                position_weight=pos_weight,
                clustering_method=cluster_method, similarity_threshold=sim_thresh,
            )
            clusters_per_anchor.append(cls)
            print(f"  Anchor #{a}: {len(cls)} групп")

    save_clusters_json(out_dir / "clusters.json", clusters_per_anchor)
    n_total = sum(len(c) for c in clusters_per_anchor)
    print(f"  → saved clusters.json ({n_total} clusters)")

    flat_rows = []
    gcid = 0
    for a_idx, cls in enumerate(clusters_per_anchor):
        for lcid, cl in enumerate(cls):
            for v_idx, hw in zip(cl['indices'], cl['heat_weights']):
                flat_rows.append((int(v_idx), a_idx, lcid, gcid, float(hw)))
            gcid += 1
    if flat_rows:
        save_matrix_csv(out_dir / "clusters_flat.csv",
                         np.array(flat_rows, dtype=np.float64),
                         header="vertex_idx,anchor_idx,local_cluster_id,global_cluster_id,heat_weight")

    # ── ОКНО 3: реконструкция + сглаживание (только если есть δ) ───────────────
    if has_deform:
        print(f"\n── [{label}] Реконструкция δ из кластеров + сглаживание ──")
        delta_recon = reconstruct_delta_from_clusters(verts, N_verts, clusters_per_anchor)
        delta_smooth = smooth_delta(delta_recon, faces, n_iter=smooth_iters, alpha=smooth_alpha)
        head_recon  = verts + delta_recon
        head_smooth = verts + delta_smooth
        err_recon = np.linalg.norm(delta_recon - delta_native, axis=1)
        print(f"  max  ||δ_recon - δ_native|| = {err_recon.max():.4f}")
        print(f"  mean ||δ_recon - δ_native|| = {err_recon.mean():.4f}")
        save_matrix_csv(out_dir / "delta_recon.csv", delta_recon, header="dx,dy,dz")
        save_matrix_csv(out_dir / "delta_smoothed.csv", delta_smooth, header="dx,dy,dz")
        print(f"\n── [{label}] ОКНО 3: native | reconstructed | smoothed ──")
        col_recon  = to_colors(np.linalg.norm(delta_recon, axis=1), CMAP_DISP)
        col_smooth = to_colors(np.linalg.norm(delta_smooth, axis=1), CMAP_DISP)
        show_meshes_side_by_side([
            (head_expr,  faces, col_native, "δ_native"),
            (head_recon, faces, col_recon,  "δ_recon"),
            (head_smooth, faces, col_smooth, "δ_smoothed"),
        ], window_title=f"[{label}] native | reconstructed | smoothed (Q → продолжить)")

    # ── ФИНАЛЬНЫЕ ЦВЕТА (окно рисуется один раз в main, для всех голов) ─────────
    print(f"\n── [{label}] подготовка финальных цветов ──")
    if use_multi_t:
        anchor_palette = make_cluster_palette(max(N_anchors, 1))
        final_colors = np.tile([0.6, 0.6, 0.6], (N_verts, 1))
        for v_idx in range(N_verts):
            a = partition[v_idx]
            if a >= 0:
                final_colors[v_idx] = anchor_palette[a]
    else:
        palette = make_cluster_palette(max(n_total, 1))
        final_colors = np.tile([0.6, 0.6, 0.6], (N_verts, 1))
        vert_weight = np.zeros(N_verts)
        color_idx = 0
        for a, cls in enumerate(clusters_per_anchor):
            for cl in cls:
                for j, v_idx in enumerate(cl['indices']):
                    w = cl['heat_weights'][j]
                    if w > vert_weight[v_idx]:
                        vert_weight[v_idx] = w
                        final_colors[v_idx] = palette[color_idx]
                color_idx += 1

    # μ-стрелки + шары на anchor'ах (μ нулевые для FBX — стрелки не рисуются)
    extras = []
    MU_SCALE = 3.0
    cl_palette = make_cluster_palette(max(n_total, 1))
    ci = 0
    for a, cls in enumerate(clusters_per_anchor):
        for cl in cls:
            if has_deform:
                p0 = cl['c_rest']; p1 = cl['c_rest'] + cl['mu'] * MU_SCALE
                extras.append(make_arrow(p0, p1, color=[0, 0, 0]))
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.004)
            sph.translate(cl['c_rest'])
            sph.paint_uniform_color(cl_palette[ci].tolist())
            sph.compute_vertex_normals(); extras.append(sph)
            ci += 1
    for s in src:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.007)
        sph.translate(verts[s]); sph.paint_uniform_color([1, 0, 0])
        sph.compute_vertex_normals(); extras.append(sph)

    # Per-vertex доминирующий global cluster id (по макс. heat-весу) — для
    # UV-NN переноса кластеров на FBX. -1 = вершина ни в одном кластере.
    vert_gcid = -np.ones(N_verts, dtype=np.int64)
    vgw = np.zeros(N_verts)
    gcid = 0
    for a, cls in enumerate(clusters_per_anchor):
        for cl in cls:
            for j, v_idx in enumerate(cl['indices']):
                w = cl['heat_weights'][j]
                if w > vgw[v_idx]:
                    vgw[v_idx] = w; vert_gcid[v_idx] = gcid
            gcid += 1

    return {
        'label':        label,
        'verts':        verts,
        'faces':        faces,
        'head_expr':    head_expr,
        'final_colors': final_colors,
        'extras':       extras,
        'has_deform':   has_deform,
        'use_multi_t':  use_multi_t,
        'partition':    partition,      # (N,) argmax-зоны или None (single-t)
        'n_anchors':    N_anchors,
        'delta_native': delta_native,   # (N,3) деформация (нули для FBX)
        'vert_gcid':    vert_gcid,      # (N,) глобальный cluster id на вершину
    }


def show_final_combined(results, use_multi_t):
    """Одно общее финальное окно для всех проанализированных голов.

    multi-t ВКЛ  → показываем ВСЕ головы рядом с heat-зонами
                   (hard-partition, маскированными по single-t reach).
    single-t     → показываем только ПЕРВУЮ голову с кластерами по single-t зоне.
    """
    if not results:
        return
    print(f"\n── ФИНАЛЬНОЕ ОКНО (все головы) ──")
    if use_multi_t:
        meshes = []
        extra_list = []
        for r in results:
            meshes.append((r['verts'], r['faces'], r['final_colors'],
                           f"[{r['label']}] зоны"))
            extra_list.append(r['extras'])
        labels = " | ".join(r['label'] for r in results)
        show_meshes_side_by_side(
            meshes, extra_geometries=extra_list,
            window_title=f"ФИНАЛ: heat-зоны (multi-t, masked by single-t) — "
                         f"{labels} (Q → выход)")
    else:
        r = results[0]
        title = (f"[{r['label']}] ФИНАЛ: кластеры (single-t зоны) (Q → выход)")
        if r['has_deform']:
            show_meshes_side_by_side([
                (r['verts'],     r['faces'], r['final_colors'], "rest"),
                (r['head_expr'], r['faces'], r['final_colors'], "deformed"),
            ], extra_geometries=[r['extras'], []], window_title=title)
        else:
            show_meshes_side_by_side([
                (r['verts'], r['faces'], r['final_colors'], "zones"),
            ], extra_geometries=[r['extras']], window_title=title)


# ── UV-параметризация одной зоны: 1 остров (Tutte + ARAP релакс) ──────────────

def _zone_largest_cc(V, F):
    """Оставляем только КРУПНЕЙШУЮ связную компоненту зоны (остальные мелкие
    несвязные ошмётки argmax-разбиения отбрасываем → один цельный остров)."""
    import scipy.sparse.csgraph as csg
    n = len(V)
    if len(F) == 0:
        return V, F, 1, np.arange(n)
    rows = np.concatenate([F[:, 0], F[:, 1], F[:, 2]])
    cols = np.concatenate([F[:, 1], F[:, 2], F[:, 0]])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    ncomp, lab = csg.connected_components(A, directed=False)
    if ncomp <= 1:
        return V, F, 1, np.arange(n)
    face_lab = lab[F[:, 0]]
    best = int(np.argmax(np.bincount(face_lab)))
    Fb = F[face_lab == best]
    keep = np.unique(Fb)
    remap = -np.ones(n, dtype=np.int64); remap[keep] = np.arange(len(keep))
    return V[keep], remap[Fb], ncomp, keep


def _boundary_loop(F):
    """Упорядоченный граничный цикл (рёбра, входящие лишь в один треугольник)."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for tri in F:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            cnt[(min(a, b), max(a, b))] += 1
    bedges = [e for e, c in cnt.items() if c == 1]
    if len(bedges) < 3:
        return None
    nbr = defaultdict(list)
    for a, b in bedges:
        nbr[a].append(b); nbr[b].append(a)
    start = bedges[0][0]
    loop = [start]; prev = None; cur = start
    while True:
        nxts = [x for x in nbr[cur] if x != prev]
        if not nxts:
            break
        nxt = nxts[0]
        if nxt == start:
            break
        loop.append(nxt); prev, cur = cur, nxt
        if len(loop) > len(bedges) + 2:
            break
    return loop if len(loop) >= 3 else None


def _square_boundary(V, loop):
    """Раскладываем граничный цикл по периметру квадрата [0,1]^2 (по 3D-длине)."""
    pts = V[loop]
    d = np.linalg.norm(np.diff(np.vstack([pts, pts[:1]]), axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(d)])
    total = max(cum[-1], 1e-12)
    t = cum[:-1] / total * 4.0
    uv = np.zeros((len(loop), 2))
    for i, ti in enumerate(t):
        if ti < 1:   uv[i] = [ti, 0]
        elif ti < 2: uv[i] = [1, ti - 1]
        elif ti < 3: uv[i] = [3 - ti, 1]
        else:        uv[i] = [0, 4 - ti]
    return uv


def _cot_laplacian_clamped(V, F):
    """Котангенс-Лапласиан с клампом весов > 0 (Tutte → без переворотов)."""
    n = len(V); I, J, W = [], [], []
    for tri in F:
        P = [V[tri[0]], V[tri[1]], V[tri[2]]]
        for e, (a, b, c) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
            u = P[a] - P[c]; v = P[b] - P[c]
            sn = np.linalg.norm(np.cross(u, v))
            cot = (u @ v) / sn if sn > 1e-12 else 0.0
            w = max(0.5 * cot, 1e-4)
            i, j = int(tri[a]), int(tri[b])
            I += [i, j]; J += [j, i]; W += [w, w]
    Aw = sp.csr_matrix((W, (I, J)), shape=(n, n))
    deg = np.asarray(Aw.sum(1)).ravel()
    return sp.diags(deg) - Aw


def _tutte_uv(V, F):
    """Tutte/гармоническая инициализация: граница на квадрат, интерьер solve."""
    loop = _boundary_loop(F)
    if loop is None:
        return None
    n = len(V)
    uvb = _square_boundary(V, loop)
    L = _cot_laplacian_clamped(V, F)
    is_b = np.zeros(n, bool); is_b[loop] = True
    interior = np.where(~is_b)[0]
    uv = np.zeros((n, 2)); uv[loop] = uvb
    if len(interior) > 0:
        Lii = L[interior][:, interior].tocsc()
        rhs = -(L[interior][:, loop] @ uvb)
        try:
            lu = spla.splu(Lii)
            uv[interior, 0] = lu.solve(rhs[:, 0])
            uv[interior, 1] = lu.solve(rhs[:, 1])
        except Exception:
            return None
    return uv


def _arap_relax(V, F, uv, iters=8):
    """ARAP-параметризация (Liu 2008): релаксируем Tutte-инициализацию к
    натуральной форме (минимум искажений), оставаясь одним островом."""
    n = len(uv); m = len(F)
    eidx = [(0, 1), (1, 2), (2, 0)]
    # rest 2D (изометрическое разворачивание каждого треугольника) + котан-веса
    tri_x = np.zeros((m, 3, 2)); tri_w = np.zeros((m, 3))
    for ti, tri in enumerate(F):
        P = [V[tri[0]], V[tri[1]], V[tri[2]]]
        e1 = P[1] - P[0]; l1 = np.linalg.norm(e1)
        if l1 < 1e-12:
            continue
        x1 = e1 / l1; v2 = P[2] - P[0]
        a = v2 @ x1; h = np.linalg.norm(v2 - a * x1)
        tri_x[ti] = [[0, 0], [l1, 0], [a, h]]
        for e, (p, q, r) in enumerate([(0, 1, 2), (1, 2, 0), (2, 0, 1)]):
            u = P[p] - P[r]; vv = P[q] - P[r]
            sn = np.linalg.norm(np.cross(u, vv))
            tri_w[ti, e] = 0.5 * ((u @ vv) / sn) if sn > 1e-12 else 0.0
    # постоянная LHS (котан-Лапласиан), pin вершины 0
    I, J, W = [], [], []
    for ti, tri in enumerate(F):
        for e, (a, b) in enumerate(eidx):
            i, j = int(tri[a]), int(tri[b]); w = tri_w[ti, e]
            I += [i, i, j, j]; J += [i, j, j, i]; W += [w, -w, w, -w]
    L = sp.csr_matrix((W, (I, J)), shape=(n, n)).tolil()
    pin = 0
    L[pin, :] = 0; L[pin, pin] = 1.0
    try:
        lu = spla.splu(L.tocsc())
    except Exception:
        return uv
    Fa = np.asarray(F)
    for _ in range(iters):
        # local: оптимальные повороты R_t (2x2) на треугольник
        S = np.zeros((m, 2, 2))
        for e, (a, b) in enumerate(eidx):
            du = uv[Fa[:, a]] - uv[Fa[:, b]]
            dx = tri_x[:, a] - tri_x[:, b]
            S += tri_w[:, e][:, None, None] * np.einsum('ni,nj->nij', du, dx)
        U, _s, Vt = np.linalg.svd(S)
        R = U @ Vt
        flip = np.linalg.det(R) < 0
        U[flip, :, -1] *= -1
        R = U @ Vt
        # global: solve L uv = b
        b = np.zeros((n, 2))
        for e, (a, bb) in enumerate(eidx):
            dx = tri_x[:, a] - tri_x[:, bb]
            rot = np.einsum('nij,nj->ni', R, dx)
            contrib = tri_w[:, e][:, None] * rot
            np.add.at(b, Fa[:, a], contrib)
            np.add.at(b, Fa[:, bb], -contrib)
        b[pin] = uv[pin]
        uv_new = np.column_stack([lu.solve(b[:, 0]), lu.solve(b[:, 1])])
        if not np.all(np.isfinite(uv_new)):
            break
        uv = uv_new
    return uv


def _laplacian_relax_uv(uv, F, iters=10, alpha=0.5):
    """Лаплас-сглаживание UV-острова: интерьер тянется к среднему соседей,
    граница ЗАФИКСИРОВАНА (форма острова не съёживается). Быстро убирает
    дёрганость/перехлёсты сетки, но может усилить искажение площадей."""
    uv = np.asarray(uv, dtype=np.float64).copy()
    n = len(uv)
    bnd = set(_boundary_vertices(F).tolist())
    interior = np.array([i for i in range(n) if i not in bnd], dtype=np.int64)
    if len(interior) == 0:
        return uv
    W = build_neighbor_avg_matrix(n, F)          # строки нормированы (среднее)
    for _ in range(iters):
        avg = W @ uv
        uv[interior] = (1 - alpha) * uv[interior] + alpha * avg[interior]
    return uv


def _spring_relax_uv(uv, F, V3d, iters=20, alpha=0.3):
    """Mass-spring релаксация: каждое ребро тянется к своей ДЛИНЕ НА 3D-меше
    (изометрия рёбер), граница свободна, но центр фиксируется от дрейфа.
    Сохраняет относительные длины рёбер → меньше искажение, чем у Лапласа."""
    uv = np.asarray(uv, dtype=np.float64).copy()
    V3d = np.asarray(V3d, dtype=np.float64)
    F = np.asarray(F)
    # уникальные рёбра + их 3D-длины (целевые)
    eset = {}
    for tri in F:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(tri[a]), int(tri[b])
            e = (min(i, j), max(i, j))
            if e not in eset:
                eset[e] = np.linalg.norm(V3d[i] - V3d[j])
    edges = np.array(list(eset.keys()), dtype=np.int64)
    rest = np.array(list(eset.values()), dtype=np.float64)
    n = len(uv)
    deg = np.bincount(edges.ravel(), minlength=n).clip(1)
    for _ in range(iters):
        disp = np.zeros((n, 2))
        d = uv[edges[:, 1]] - uv[edges[:, 0]]
        L = np.linalg.norm(d, axis=1).clip(1e-12)
        dirn = d / L[:, None]
        corr = (L - rest)[:, None] * dirn * 0.5   # к целевой длине ребра
        np.add.at(disp, edges[:, 0], corr)
        np.add.at(disp, edges[:, 1], -corr)
        uv += alpha * disp / deg[:, None]
        uv -= uv.mean(0)                          # против дрейфа центра
    return uv


def relax_uv_island(uv, F, V3d, method="arap", iters=10, align_world=True):
    """Диспетчер релаксации одного UV-острова.
      method="arap"      → ARAP (минимум искажений, лучший по качеству);
      method="laplacian" → Лаплас-сглаживание (граница фикс., быстро);
      method="spring"    → mass-spring к 3D-длинам рёбер (изометрия рёбер).

    align_world=True → после релаксации остров доворачивается по мировым осям
    (мировой Y → +V), чтобы не «съезжал» (ARAP/spring свободно вращают остров).
    Возвращает новый uv (Nx2). При ошибке — исходный uv."""
    try:
        if method == "laplacian":
            out = _laplacian_relax_uv(uv, F, iters=iters)
        elif method == "spring":
            out = _spring_relax_uv(uv, F, V3d, iters=iters)
        else:
            out = _arap_relax(V3d, F, np.asarray(uv, float).copy(), iters=iters)
        if align_world:
            out = align_uv_to_world(out, V3d, F)
        return out
    except Exception:
        return uv


def uv_island_distortion(uv, V3d, F):
    """Дисторсия UV-острова: per-triangle symmetric Dirichlet D = σ₁²+σ₂²+
    σ₁⁻²+σ₂⁻² (σ — сингулярные числа отображения 3D-треуг.→UV). Минимум 4 при
    изометрии; растёт на растяжении И сжатии; → ∞ на вырожденных/перевёрнутых.

    Возвращает (mean_area_weighted, p95, max, n_flips). mean — для оценки/цикла."""
    uv = np.asarray(uv, np.float64); V = np.asarray(V3d, np.float64)
    F = np.asarray(F)
    if len(F) == 0:
        return (4.0, 4.0, 4.0, 0)
    P0, P1, P2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    Q0, Q1, Q2 = uv[F[:, 0]], uv[F[:, 1]], uv[F[:, 2]]
    e1 = P1 - P0; e2 = P2 - P0
    l1 = np.linalg.norm(e1, axis=1)
    u = e1 / np.maximum(l1[:, None], 1e-12)
    x2 = (e2 * u).sum(1)
    y2 = np.linalg.norm(e2 - x2[:, None] * u, axis=1)      # высота треуг.
    detA = np.where(np.abs(l1 * y2) < 1e-12, 1e-12, l1 * y2)
    dQ1 = Q1 - Q0; dQ2 = Q2 - Q0
    # J = D_Q · inv(D_A), D_A=[[l1,x2],[0,y2]] → inv = 1/detA·[[y2,-x2],[0,l1]]
    a = (dQ1[:, 0] * y2) / detA
    b = (dQ1[:, 0] * (-x2) + dQ2[:, 0] * l1) / detA
    c = (dQ1[:, 1] * y2) / detA
    d = (dQ1[:, 1] * (-x2) + dQ2[:, 1] * l1) / detA
    fro2 = a * a + b * b + c * c + d * d                   # σ₁²+σ₂²
    det = a * d - b * c                                    # σ₁σ₂ (знак=ориент.)
    det2 = np.maximum(det * det, 1e-12)
    D = fro2 + fro2 / det2                                 # symmetric Dirichlet
    area = 0.5 * np.abs(l1 * y2)
    w = area / np.maximum(area.sum(), 1e-12)
    return (float((D * w).sum()), float(np.percentile(D, 95)),
            float(D.max()), int((det < 0).sum()))


def relax_uv_island_adaptive(uv, F, V3d, method="arap", iters_per_round=8,
                             max_rounds=8, target=4.5, tol=0.01,
                             align_world=True):
    """Адаптивный, ФЛИП-АВАРНЫЙ релакс острова: релаксим раундами, выбирая лучший
    по (число складок, дисторсия). Если складки (закрученный/перехлёстнутый
    остров) остаются — ARAP их не развернёт, поэтому переинициализируем через
    Tutte (гарантированно БЕЗ складок для диск-острова) + ARAP.

    «Лучше» = меньше флипов, при равных — меньше mean-дисторсии."""
    F = np.asarray(F)

    def score(u):
        m, _, _, nf = uv_island_distortion(u, V3d, F)
        return (int(nf), float(m))
    best = np.asarray(uv, np.float64).copy()
    bf, bd = score(best)
    cur = best
    stall = 0
    for _ in range(int(max_rounds)):
        if bf == 0 and bd <= target:
            break
        cur = relax_uv_island(cur, F, V3d, method=method,
                              iters=int(iters_per_round), align_world=False)
        f, d = score(cur)
        if (f, d) < (bf, bd - tol):                        # лучше по (флипы,дист.)
            bf, bd = f, d; best = np.asarray(cur, np.float64).copy(); stall = 0
        else:
            stall += 1
            if bf == 0 and stall >= 2:                     # без складок и плато
                break
    # фолбэк: остались СКЛАДКИ (или дисторсия застряла высоко) → Tutte + ARAP
    if bf > 0 or bd > max(2.5 * target, 10.0):
        try:
            t = _tutte_uv(V3d, F)                          # без складок (bijective)
            if t is not None:
                t = _arap_relax(V3d, F, np.asarray(t, np.float64).copy(),
                                iters=int(iters_per_round) * int(max_rounds))
                f2, d2 = score(t)
                if (f2, d2) < (bf, bd):                    # берём, если лучше
                    best = np.asarray(t, np.float64).copy(); bf, bd = f2, d2
        except Exception:
            pass
    if align_world:
        best = align_uv_to_world(best, V3d, F)
    return best


def _orient_uv_canonical(uv, V3d, up=(0.0, 1.0, 0.0), right=(1.0, 0.0, 0.0)):
    """Каноническая ориентация острова: мировое «вверх» (Y) → +V, «вправо»
    (X) → +U (с фиксацией отражения). Поскольку обе головы нормализованы в
    одном мировом фрейме, одинаковые зоны на РАЗНЫХ головах получают
    одинаковую ориентацию → острова совпадают и сравнимы напрямую.
    """
    uc = uv - uv.mean(0)
    Vc = V3d - V3d.mean(0)
    # линейная аппроксимация отображения 3D→UV: Vc @ M ≈ uc, M=(3,2)
    M, *_ = np.linalg.lstsq(Vc, uc, rcond=None)
    up = np.asarray(up, float); right = np.asarray(right, float)
    up_uv = up @ M
    if np.linalg.norm(up_uv) < 1e-9:
        return uv
    # поворот, переводящий up_uv → +V (ось (0,1))
    alpha = np.arctan2(up_uv[1], up_uv[0])
    phi = np.pi / 2.0 - alpha
    c, s = np.cos(phi), np.sin(phi)
    Rphi = np.array([[c, -s], [s, c]])
    uc = uc @ Rphi.T
    # фиксация отражения: «вправо» должно идти в +U
    right_uv = (right @ M) @ Rphi.T
    if right_uv[0] < 0:
        uc[:, 0] = -uc[:, 0]
    return uc


def _flat_projection_uv(V3d):
    """Планарная проекция (как режим Flat в Cinema4D): проецируем вершины зоны
    на их плоскость наилучшего приближения (2 главные компоненты PCA).
    Один остров, без параметризации. Может давать перехлёсты там, где
    поверхность загибается — это ожидаемо для flat-проекции."""
    c = V3d.mean(0)
    X = V3d - c
    # главные оси через SVD ковариации
    _u, _s, Wt = np.linalg.svd(X, full_matrices=False)
    return X @ Wt[:2].T


def _world_flat_projection_uv(V3d, F=None, world_up=(0.0, 1.0, 0.0)):
    """Планарная проекция вдоль НОРМАЛИ зоны с фиксированной мировой
    ориентацией: мировой Y → +V (вверх в UV), (n × up) → +U.

    Базис строится из мировых направлений, НЕ из формы зоны → ориентация
    детерминирована и одинакова на разных головах (обе в общем normalize_bbox-
    кадре): зоны автоматически согласованы, без нестабильного lstsq.

    Нормаль зоны n — площадь-взвешенная средняя нормаль граней (или PCA-нормаль,
    если граней нет). Для почти горизонтальных зон (Y∥n, темя) up вырождается →
    fallback: up из мирового Z."""
    V3d = np.asarray(V3d, dtype=np.float64)
    c = V3d.mean(0)
    X = V3d - c
    up_w = np.asarray(world_up, dtype=np.float64)

    # нормаль зоны
    n = None
    if F is not None and len(F) > 0:
        F = np.asarray(F)
        v0, v1, v2 = V3d[F[:, 0]], V3d[F[:, 1]], V3d[F[:, 2]]
        fn = np.cross(v1 - v0, v2 - v0)          # нормали*2площади (взвеш.)
        s = fn.sum(0)
        if np.linalg.norm(s) > 1e-12:
            n = s / np.linalg.norm(s)
    if n is None:                                # PCA-нормаль (мин. ось разброса)
        _u, _s, Wt = np.linalg.svd(X, full_matrices=False)
        n = Wt[-1]

    # up = мировой Y, спроецированный на плоскость зоны (⊥ n)
    up = up_w - (up_w @ n) * n
    if np.linalg.norm(up) < 1e-6:                # зона ⟂ Y (темя/подбородок)
        alt = np.array([0.0, 0.0, 1.0])          # fallback: мировой Z
        up = alt - (alt @ n) * n
    up = up / np.clip(np.linalg.norm(up), 1e-12, None)
    right = np.cross(n, up)
    right = right / np.clip(np.linalg.norm(right), 1e-12, None)

    u = X @ right                                # +U
    v = X @ up                                   # +V (мировой Y вверх)
    return np.column_stack([u, v])


def align_uv_to_world(uv, V3d, F=None, world_up=(0.0, 1.0, 0.0)):
    """Довернуть/отразить готовый UV-остров так, чтобы мировой Y снова смотрел
    в +V (как в world-flat). Нужно после релаксации (ARAP/spring свободно
    вращают остров → мировая ориентация теряется, остров «съезжает»).

    Находим оптимальное жёсткое преобразование (поворот ± отражение, 2D
    Procrustes), переводящее текущий uv в эталонную world-flat проекцию тех же
    вершин. Масштаб/форму релаксации СОХРАНЯЕМ — применяем только R к uv."""
    uv = np.asarray(uv, dtype=np.float64)
    ref = _world_flat_projection_uv(V3d, F, world_up=world_up)  # эталон (N,2)
    A = uv - uv.mean(0)
    B = ref - ref.mean(0)
    # ортогональный Procrustes: R = argmin ‖A R − B‖ (с допуском отражения)
    H = A.T @ B
    U, _s, Vt = np.linalg.svd(H)
    R = U @ Vt                                   # 2x2 ортогональная
    out = A @ R
    return out + uv.mean(0)                       # центр острова не двигаем


def _uv_mode(params):
    """Значение аргумента flat по параметрам: "world" если uv_world_orient,
    иначе bool(uv_flat). Единая точка для всех вызовов UV-функций."""
    if bool(params.get('uv_world_orient', False)):
        return "world"
    return bool(params.get('uv_flat', False))


def parameterize_zone_single(V, F, arap_iters=8, flat=False):
    """Одна зона → один UV-остров (канонически ориентированный).

    flat=False    → Tutte-инициализация + ARAP-релакс (низкие искажения).
    flat=True     → планарная проекция на плоскость best-fit (как Flat в C4D).
    flat="world"  → планарная проекция вдоль нормали зоны с мировым Y вверх
                    (детерминированная ориентация, без _orient_uv_canonical).

    Возвращает (uv Nx2, F Kx3, ncomp, keep) или None.
    keep — индексы оставленных вершин в исходном V (для проброса глоб. индексов)."""
    Vk, Fk, ncomp, keep = _zone_largest_cc(np.asarray(V), np.asarray(F))
    if Fk is None or len(Fk) < 1 or len(Vk) < 3:
        return None
    if isinstance(flat, str) and flat == "world":
        # мировая ориентация: Y вверх; ориентация уже задана, не канонизируем
        uv = _world_flat_projection_uv(Vk, Fk)
        return uv, Fk, ncomp, keep
    if flat:
        uv = _flat_projection_uv(Vk)
    else:
        uv0 = _tutte_uv(Vk, Fk)
        if uv0 is None:
            uv = _flat_projection_uv(Vk)          # фолбэк: нет границы
        else:
            try:
                uv = _arap_relax(Vk, Fk, uv0.copy(), iters=arap_iters)
                if not np.all(np.isfinite(uv)):
                    uv = uv0
            except Exception:
                uv = uv0
    uv = _orient_uv_canonical(uv, Vk)
    return uv, Fk, ncomp, keep


def compute_zone_islands(verts, faces, partition, n_anchors, flat=False):
    """Параметризация каждой зоны в ОДИН UV-остров (без раскладки в сетку).

    Возвращает dict {anchor: (uv Nx2, F Kx3, gidx N)} — канонически
    ориентированные острова. gidx — глобальные индексы вершин меша, отвечающие
    строкам uv (для UV-NN переноса). Разделено с упаковкой, чтобы одни и те же
    острова можно было укладывать в разные сетки (напр., общую для overlay).
    """
    faces = np.asarray(faces)
    out = {}
    for a in range(n_anchors):
        vmask = (partition == a)
        if int(vmask.sum()) < 3:
            continue
        fmask = vmask[faces].all(axis=1)
        Fz = faces[fmask]
        if len(Fz) == 0:
            continue
        uniq, inv = np.unique(Fz.reshape(-1), return_inverse=True)
        res = parameterize_zone_single(verts[uniq], inv.reshape(-1, 3), flat=flat)
        if res is None:
            print(f"    zone {a}: не удалось развернуть (нет границы) — пропуск")
            continue
        uv, Fk, ncomp, keep = res
        out[a] = (np.asarray(uv, dtype=np.float64),
                  np.asarray(Fk, dtype=np.int64),
                  np.asarray(uniq[keep], dtype=np.int64))
        extra = f", откинул {ncomp-1} мелких куск." if ncomp > 1 else ""
        print(f"    zone {a}: {len(Fz)} граней → 1 остров, "
              f"UV {uv.shape[0]} верш.{extra}")
    return out


def _umeyama_sim_2d(src, dst):
    """Оптимальное similarity-преобразование (R 2x2, scale, t), переводящее
    src→dst для СООТВЕТСТВУЮЩИХ пар точек (Umeyama 1991). Без отражения
    (det(R)=+1) — острова уже канонически ориентированы."""
    mu_s = src.mean(0); mu_d = dst.mean(0)
    Xs = src - mu_s; Xd = dst - mu_d
    Sigma = (Xd.T @ Xs) / max(len(src), 1)
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_s = (Xs ** 2).sum() / max(len(src), 1)
    scale = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    t = mu_d - scale * (R @ mu_s)
    return R, scale, t


def _align_island_pca_icp(mov, ref, icp_iters=12):
    """Подгоняем остров `mov` (Nx2) к `ref` (Mx2) similarity-преобразованием
    (поворот+масштаб+сдвиг), даже при разной топологии/числе вершин.

    1. центр + равномерный масштаб по RMS;
    2. PCA: совмещаем главные оси (перебор 4 знаков осей, лучший по NN-дист.);
    3. ICP-уточнение: NN-соответствия → Umeyama-similarity, несколько итераций.

    Возвращает преобразованный `mov` в системе `ref`.
    """
    from scipy.spatial import cKDTree
    mov = np.asarray(mov, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if len(mov) < 2 or len(ref) < 2:
        return mov
    rc = ref.mean(0)
    R0 = ref - rc                                   # ref центрирован
    mc = mov.mean(0)
    mr = np.sqrt(((mov - mc) ** 2).sum(1).mean())
    rr = np.sqrt((R0 ** 2).sum(1).mean())
    P = (mov - mc) * (rr / max(mr, 1e-9))           # центр + RMS-масштаб

    def major_axes(X):
        C = (X.T @ X) / max(len(X), 1)
        _w, V = np.linalg.eigh(C)                    # по возрастанию
        return V[:, ::-1]                            # главная ось первой
    Vm = major_axes(P); Vr = major_axes(R0)

    tree = cKDTree(R0)
    best_Q = P; best_d = np.inf
    for sx in (1.0, -1.0):                           # перебор знаков осей
        for sy in (1.0, -1.0):
            Rrot = Vr @ (Vm * np.array([sx, sy])).T
            Q = P @ Rrot.T
            d, _ = tree.query(Q)
            if d.mean() < best_d:
                best_d = d.mean(); best_Q = Q
    Q = best_Q

    for _ in range(max(int(icp_iters), 0)):          # ICP-уточнение
        d, idx = tree.query(Q)
        R, s, t = _umeyama_sim_2d(Q, R0[idx])
        Q = s * (Q @ R.T) + t
    return Q + rc


def _grid_dims(n_cells):
    cols = int(np.ceil(np.sqrt(max(n_cells, 1))))
    rows = int(np.ceil(n_cells / cols))
    cellw = 1.0 / cols; cellh = 1.0 / rows; margin = 0.07
    avail = min(cellw, cellh) * (1.0 - 2 * margin)
    return cols, rows, cellw, cellh, avail


def _place_zone_in_cell(uv, a, cols, rows, cellw, cellh, avail):
    """Нормируем остров по своему bbox и центрируем в ячейке `a`.
    Возвращает placed Nx2 (uniform-scale, aspect сохранён)."""
    mn = uv.min(0); ext = uv.max(0) - mn
    s = max(float(ext.max()), 1e-9)
    un = (uv - mn) / s
    iw = ext[0] / s * avail; ih = ext[1] / s * avail
    col = a % cols; row = a // cols
    x0 = col * cellw + (cellw - iw) / 2.0
    y0 = (rows - 1 - row) * cellh + (cellh - ih) / 2.0  # зона 0 — сверху
    P = un * avail; P[:, 0] += x0; P[:, 1] += y0
    return P


def _pack_islands_to_grid(zone_dict, n_cells, color, ref_placed=None,
                          align_pca_icp=False):
    """Укладываем острова по ячейкам сетки в единичный квадрат.

    Зона `a` → ячейка `a` (одна и та же для всех голов).
    color: палитра (n_cells,3) [индекс по anchor], один RGB на всё, либо
        dict {a: (N,3)} — цвет на вершину острова (для per-vertex раскраски,
        напр. перенесёнными группами).
    ref_placed: опц. dict {a: placed Nx2} — эталонная раскладка (другая
        голова). Если задан, остров подгоняется по масштабу/центру к эталону
        той же зоны (совпадение центроида + RMS-радиуса) — scale-коррекция
        для overlay. Возвращает (V Nx3, F Kx3, C Nx3, placed_dict) или None.
    """
    if not zone_dict:
        return None
    is_dict = isinstance(color, dict)
    if not is_dict:
        color = np.asarray(color, dtype=np.float64)
        per_anchor = (color.ndim == 2)
    cols, rows, cellw, cellh, avail = _grid_dims(n_cells)

    allV, allF, allC = [], [], []
    placed = {}
    off = 0
    for a in sorted(zone_dict):
        uv, F = zone_dict[a][0], zone_dict[a][1]
        P = _place_zone_in_cell(uv, a, cols, rows, cellw, cellh, avail)
        if ref_placed is not None and a in ref_placed:
            R = ref_placed[a]
            if align_pca_icp:
                P = _align_island_pca_icp(P, R)     # PCA+ICP к эталону
            else:
                # Подгонка по ГАБАРИТНОМУ радиусу (97-й перцентиль расстояния до
                # центроида) — симметрична: больший остров сжимается, меньший
                # растёт, и FBX не торчит за FLAME. RMS зависел бы от формы и мог
                # раздуть остров другой формы.
                rc = R.mean(0); pc = P.mean(0)
                rr = np.percentile(np.linalg.norm(R - rc, axis=1), 97)
                pr = np.percentile(np.linalg.norm(P - pc, axis=1), 97)
                sc = rr / max(pr, 1e-9)
                P = (P - pc) * sc + rc              # центр + габаритный радиус
        placed[a] = P
        Vz = np.zeros((len(P), 3)); Vz[:, :2] = P
        allV.append(Vz)
        allF.append(F + off)
        if is_dict:
            allC.append(np.asarray(color[a], dtype=np.float64))
        else:
            c = color[a] if per_anchor else color
            allC.append(np.tile(c, (len(P), 1)))
        off += len(P)
    if not allV:
        return None
    return np.vstack(allV), np.vstack(allF), np.vstack(allC), placed


def build_zone_uv_layout(verts, faces, partition, n_anchors, flat=False):
    """Острова зон + раскладка в сетку (цвет = anchor). Возвращает
    (V,F,C) или None. Тонкая обёртка над compute_zone_islands+_pack."""
    zd = compute_zone_islands(verts, faces, partition, n_anchors, flat=flat)
    if not zd:
        return None
    pal = make_cluster_palette(max(n_anchors, 1))
    res = _pack_islands_to_grid(zd, n_anchors, pal)
    if res is None:
        return None
    return res[0], res[1], res[2]


def _write_obj(path, V, F, C=None):
    """Пишем меш в OBJ. V (N,3), F (K,3, 0-based). C (N,3) опц. — цвет вершин
    (расширение `v x y z r g b`, читают MeshLab/Blender). Грани 1-based."""
    import os
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    has_c = C is not None and len(C) == len(V)
    if has_c:
        C = np.clip(np.asarray(C, dtype=np.float64), 0.0, 1.0)
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# UV-развёртка masked-зон (motion_groups_v6)\n")
        for i in range(len(V)):
            x, y, z = V[i]
            if has_c:
                r, g, b = C[i]
                f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r:.4f} {g:.4f} {b:.4f}\n")
            else:
                f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for tri in F:
            f.write(f"f {tri[0]+1} {tri[1]+1} {tri[2]+1}\n")
    return path


def export_deformation(out_base, verts_rest, faces, delta, label="head"):
    """Сохраняем перенесённую деформацию двумя файлами:

      <out_base>.fbx — деформированный меш (verts_rest + delta) через assimp
                       (obj → fbx). Если assimp недоступен — останется .obj.
      <out_base>.h5  — HDF5-таблица per-vertex: vertex_index, direction (единич.
                       вектор сдвига), magnitude (сила сдвига |δ|), плюс
                       raw delta-вектор и rest/deformed позиции.

    Возвращает dict с путями записанных файлов."""
    import os
    import subprocess
    import tempfile

    verts_rest = np.asarray(verts_rest, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    delta = np.asarray(delta, dtype=np.float64)
    deformed = verts_rest + delta
    out_base = str(out_base)
    os.makedirs(os.path.dirname(out_base) or ".", exist_ok=True)
    written = {}

    # ── 1. деформированный меш → FBX (через obj + assimp) ──
    tmp_obj = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    tmp_obj.close()
    _write_obj(tmp_obj.name, deformed, faces)
    fbx_path = out_base + ".fbx"
    ok = False
    try:
        r = subprocess.run(["assimp", "export", tmp_obj.name, fbx_path],
                           capture_output=True, text=True, timeout=120)
        ok = (r.returncode == 0 and os.path.exists(fbx_path))
    except Exception as e:
        print(f"  ⚠ assimp export не удался: {e}")
    if ok:
        written['fbx'] = fbx_path
    else:                                    # фолбэк: оставляем obj
        obj_path = out_base + ".obj"
        _write_obj(obj_path, deformed, faces)
        written['obj'] = obj_path
        print("  ⚠ FBX не записан (нет assimp?) — сохранён OBJ.")
    try:
        os.unlink(tmp_obj.name)
    except OSError:
        pass

    # ── 2. per-vertex таблица деформации ──
    mag = np.linalg.norm(delta, axis=1)                  # сила сдвига
    # направление: ТОЧНО единичный вектор там, где есть сдвиг; ноль, где δ≈0.
    # (нельзя нормировать нулевой вектор; clip-делитель искажал короткие δ).
    direction = np.zeros_like(delta)
    nz = mag > 1e-9
    direction[nz] = delta[nz] / mag[nz, None]

    # CSV — ВСЕГДА: vertex_index, направление (единичн.), сила, raw δ
    csv_path = out_base + ".csv"
    rows = np.column_stack([np.arange(len(verts_rest)), direction, mag, delta])
    save_matrix_csv(csv_path, rows,
                    header="vertex_index,dir_x,dir_y,dir_z,magnitude,"
                           "delta_x,delta_y,delta_z")
    written['csv'] = csv_path

    # HDF5 — если есть h5py (компактнее + хранит rest/deformed/faces)
    h5_path = out_base + ".h5"
    try:
        import h5py
        with h5py.File(h5_path, "w") as h:
            h.attrs['label'] = label
            h.attrs['n_verts'] = len(verts_rest)
            h.attrs['schema'] = ("per-vertex deformation: vertex_index, "
                                 "direction(3, unit), magnitude(|delta|)")
            h.create_dataset('vertex_index',
                             data=np.arange(len(verts_rest), dtype=np.int64))
            h.create_dataset('direction', data=direction)        # (N,3)
            h.create_dataset('magnitude', data=mag)              # (N,)
            h.create_dataset('delta', data=delta)                # (N,3) raw
            h.create_dataset('rest', data=verts_rest)            # (N,3)
            h.create_dataset('deformed', data=deformed)          # (N,3)
            h.create_dataset('faces', data=faces)                # (K,3)
        written['h5'] = h5_path
    except Exception as e:
        print(f"  ⚠ HDF5 не записан ({e}) — есть CSV.")

    print(f"  Деформация сохранена: {', '.join(written.values())}")
    return written


def export_uv_layouts_obj(results, out_dir, flat=False):
    """Экспорт UV-развёрток зон каждой головы в OBJ (плоский меш z=0, цвет по
    зоне). Один файл на голову: <out_dir>/uv_<label>.obj. Возвращает список
    записанных путей."""
    from pathlib import Path
    out_dir = Path(out_dir)
    written = []
    for r in results:
        if r.get('partition') is None:
            continue
        zd = compute_zone_islands(
            r['verts'], r['faces'], r['partition'], r['n_anchors'], flat=flat)
        if not zd:
            continue
        pal = make_cluster_palette(max(r['n_anchors'], 1))
        packed = _pack_islands_to_grid(zd, r['n_anchors'], pal)
        if packed is None:
            continue
        V, F, C, _ = packed
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_"
                       for ch in str(r['label']))
        p = out_dir / f"uv_{safe}.obj"
        _write_obj(p, V, F, C)
        written.append(p)
        print(f"  UV → OBJ: {p}  ({len(V)} верш., {len(F)} граней)")
    return written


def _uv_grid_lines(n_anchors):
    """Серая сетка ячеек + рамка единичного квадрата (Open3D LineSet)."""
    cols = int(np.ceil(np.sqrt(n_anchors)))
    rows = int(np.ceil(n_anchors / cols))
    pts, lines = [], []
    def add(p0, p1):
        i = len(pts); pts.append(p0); pts.append(p1); lines.append([i, i + 1])
    for c in range(cols + 1):
        x = c / cols; add([x, 0, 0], [x, 1, 0])
    for r in range(rows + 1):
        y = r / rows; add([0, y, 0], [1, y, 0])
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile([0.5, 0.5, 0.5], (len(lines), 1)))
    return ls


def _boundary_lineset(zone_dict, placed, color, z=0.0):
    """LineSet всех КРАЙНИХ (граничных) рёбер островов в раскладке `placed`
    (dict {зона: Nx2}), окрашенных одним цветом. `z` — подъём над плоскостью
    (чтобы линия не z-fight'ила с заливкой/каркасом). Возвращает LineSet/None."""
    pts, lines = [], []
    for a in sorted(zone_dict):
        if a not in placed:
            continue
        be = _boundary_edges(zone_dict[a][1])    # граничные рёбра острова
        if len(be) == 0:
            continue
        P = placed[a]
        for e in be:
            i = len(pts)
            pts.append([P[e[0], 0], P[e[0], 1], z])
            pts.append([P[e[1], 0], P[e[1], 1], z])
            lines.append([i, i + 1])
    if not lines:
        return None
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.array(pts, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.array(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1)))
    return ls


# Цвета подсветки крайних рёбер (контрастные к заливкам/каркасу overlay).
BND_COL_A = [1.0, 0.0, 1.0]   # голова 1 (FLAME) — пурпурный
BND_COL_B = [1.0, 0.85, 0.0]  # голова 2 (FBX)   — жёлтый


def show_uv_unwrap_combined(results, flat=False, overlay_scale_match=True,
                            transfer=None, transfer_dst_index=1,
                            align_pca_icp=False, warp_line_step=0,
                            warp_min_dist=0.0, show_boundary=True,
                            show_heat=False, warp_heat_t=0.05,
                            warp_heat=False):
    """Новое окно: UV-развёртки masked-зон всех голов рядом (multi-t).

    Каждая голова → зоны развёрнуты по одному острову (Tutte+ARAP, либо
    flat-проекция), упакованы по сетке в единичный квадрат. Острова одной
    зоны у разных голов — в одной ячейке для прямого сравнения.

    transfer: опц. результат transfer_deformations_uv (dict с 'gcid'). Если
        задан, добавляется 4-я панель: развёртка FBX (head с индексом
        transfer_dst_index среди голов с partition), раскрашенная
        ПЕРЕНЕСЁННЫМИ группами (global cluster id из FLAME).
    """
    mode = "flat-проекция" if flat else "Tutte+ARAP"
    print(f"\n── UV-РАЗВЁРТКА masked-зон (multi-t, {mode}) ──")

    # Параметризуем острова по каждой голове (отдельно от упаковки).
    heads = []   # (label, n_anchors, zone_dict, result)
    for r in results:
        if r.get('partition') is None:
            continue
        print(f"  [{r['label']}] строю UV-острова...")
        zd = compute_zone_islands(
            r['verts'], r['faces'], r['partition'], r['n_anchors'], flat=flat)
        if zd:
            heads.append((r['label'], r['n_anchors'], zd, r))

    if not heads:
        print("  UV-развёртка: нет зон для отображения "
              "(partition пуст? multi-t выключен?).")
        return

    # Сопоставление зон FBX(голова 2)→FLAME(голова 1) по ПОЗИЦИИ (не по индексу
    # пика) → острова одной зоны ложатся в одну ячейку и красятся одинаково.
    zmap = match_zones_by_position(heads[0][3], heads[1][3]) \
        if len(heads) >= 2 else {}

    def _remap_zd(zd, mapping):
        return {mapping[a]: isl for a, isl in zd.items() if a in mapping}

    # Панели: по одной на голову (цвет = anchor) + overlay (если ≥2 головы).
    panels = []   # (title, [geoms])
    for hi, (label, na, zd, r) in enumerate(heads):
        zd_use, na_use = zd, na
        if hi >= 1 and zmap:                 # головы 2+ → ячейки/цвета FLAME
            zd_use = _remap_zd(zd, zmap); na_use = heads[0][1]
        pal = make_cluster_palette(max(na_use, 1))
        res_lay = _pack_islands_to_grid(zd_use, na_use, pal)
        if res_lay is None:
            continue
        V, F, C, placed_h = res_lay
        geoms_h = [o3d_mesh(V, F, C), _uv_grid_lines(na_use)]
        if show_boundary:                    # крайние рёбра подсвечиваем
            bcol = BND_COL_A if hi == 0 else BND_COL_B
            bls = _boundary_lineset(zd_use, placed_h, bcol, z=0.01)
            if bls is not None:
                geoms_h.append(bls)
        panels.append((label, geoms_h))

    OVA_COL = [0.93, 0.40, 0.18]   # голова 1 — тёплый (заливка)
    OVB_COL = [0.15, 0.45, 0.95]   # голова 2 — холодный (каркас поверх)
    if len(heads) >= 2:
        (l0, na0, zd0, _r0), (l1, na1, zd1, _r1) = heads[0], heads[1]
        zd1 = _remap_zd(zd1, zmap) if zmap else zd1   # FBX зоны → ячейки FLAME
        n_ov = na0                                    # ячейки по FLAME
        geoms_ov = [_uv_grid_lines(n_ov)]
        # Голова 1 — эталон раскладки.
        A = _pack_islands_to_grid(zd0, n_ov, OVA_COL)
        ref_placed = None
        if A is not None:
            geoms_ov.append(o3d_mesh(A[0], A[1], A[2]))
            ref_placed = A[3]
        # Голова 2 — подгоняем к зонам головы 1 (overlay): PCA+ICP либо
        # центр+габаритный-масштаб (scale-fit).
        scale_match = bool(overlay_scale_match)
        use_ref = ref_placed if (scale_match or align_pca_icp) else None
        B = _pack_islands_to_grid(
            zd1, n_ov, OVB_COL,
            ref_placed=use_ref, align_pca_icp=align_pca_icp)
        b_placed = None
        if B is not None:
            b_placed = B[3]
            Vb = B[0]
            # Если включён heat-warp — реально деформируем остров FBX на FLAME
            # прямо в overlay: граница тянется на ребро, нутро следует за теплом.
            # Тогда наложенный шелл совпадает, а зелёные линии становятся ~0.
            if warp_heat and ref_placed is not None:
                Vparts = []
                for a in sorted(zd1):
                    P = b_placed[a]
                    if a in ref_placed and a in zd0:
                        P = _warp_island_heat(
                            P, zd1[a][1], ref_placed[a], zd0[a][1],
                            t=warp_heat_t, min_dist=warp_min_dist)
                        b_placed[a] = P          # обновляем (для линий/границ)
                    Vz = np.zeros((len(P), 3)); Vz[:, :2] = P
                    Vparts.append(Vz)
                Vb = np.vstack(Vparts) if Vparts else B[0]
            bm = o3d_mesh(Vb, B[1], B[2])
            wire = o3d.geometry.LineSet.create_from_triangle_mesh(bm)
            wire.paint_uniform_color(OVB_COL)
            geoms_ov.append(wire)
        # Крайние (граничные) рёбра обоих мешей — разным цветом:
        # FLAME = пурпурный, FBX = жёлтый. Видно, где границы расходятся.
        if show_boundary:
            # разный z: FLAME ниже, FBX выше каркаса — обе границы видны, не
            # z-fight'ят ни с заливкой/каркасом, ни друг с другом.
            if ref_placed is not None:
                bls_a = _boundary_lineset(zd0, ref_placed, BND_COL_A, z=0.02)
                if bls_a is not None:
                    geoms_ov.append(bls_a)
            if b_placed is not None:
                bls_b = _boundary_lineset(zd1, b_placed, BND_COL_B, z=0.03)
                if bls_b is not None:
                    geoms_ov.append(bls_b)
        # Линии связи границы: каждая граничная точка острова FBX (голова 2) →
        # её проекция на ребро границы острова FLAME (голова 1), В СИСТЕМЕ УЖЕ
        # ПОДОГНАННЫХ островов overlay-панели. step — шаг показа линий.
        if (warp_line_step and int(warp_line_step) > 0
                and ref_placed is not None and B is not None):
            step = max(int(warp_line_step), 1)
            b_placed = B[3]
            lp, ll = [], []
            for a in sorted(zd1):
                if a not in zd0 or a not in ref_placed or a not in b_placed:
                    continue
                Fk_s = zd0[a][1]; Fk_d = zd1[a][1]
                bnd_d = _boundary_vertices(Fk_d)
                be_s = _boundary_edges(Fk_s)        # ВСЕ граничные рёбра FLAME
                if len(bnd_d) < 3 or len(be_s) < 3:
                    continue
                rp = ref_placed[a]
                src = b_placed[a][bnd_d][::step]
                if len(src) == 0:
                    continue
                dst = _project_to_segments(src, rp[be_s[:, 0]], rp[be_s[:, 1]])
                dst = _cap_targets_min_dist(src, dst, warp_min_dist)
                for s, d in zip(src, dst):
                    i = len(lp)
                    lp.append([s[0], s[1], 0.0]); lp.append([d[0], d[1], 0.0])
                    ll.append([i, i + 1])
            if ll:
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(
                    np.array(lp, dtype=np.float64))
                ls.lines = o3d.utility.Vector2iVector(
                    np.array(ll, dtype=np.int32))
                ls.colors = o3d.utility.Vector3dVector(
                    np.tile([0.10, 0.85, 0.20], (len(ll), 1)))
                geoms_ov.append(ls)
                print(f"  Overlay: {len(ll)} линий связи границы FBX→FLAME "
                      f"(step={step}).")
        sm = ("PCA+ICP" if align_pca_icp
              else ("scale-fit" if scale_match else "as-is"))
        panels.append(
            (f"overlay: {l0} (залив.) + {l1} (каркас, {sm})", geoms_ov))

    # 4-я панель: развёртка FBX, раскрашенная ПЕРЕНЕСЁННЫМИ группами.
    if (transfer is not None and 'gcid' in transfer
            and len(heads) > transfer_dst_index):
        l_dst, na_dst, zd_dst, _r_dst = heads[transfer_dst_index]
        if transfer_dst_index >= 1 and zmap:        # ячейки FLAME (как overlay)
            zd_dst = _remap_zd(zd_dst, zmap); na_dst = heads[0][1]
        gcid = np.asarray(transfer['gcid'])
        ng = int(gcid.max()) + 1 if gcid.max() >= 0 else 1
        pal_g = make_cluster_palette(max(ng, 1))
        GREY = np.array([0.6, 0.6, 0.6])
        color_dict = {}
        for a in zd_dst:
            gidx = zd_dst[a][2]
            g = gcid[gidx]
            cols_v = np.where((g >= 0)[:, None],
                              pal_g[np.clip(g, 0, ng - 1)], GREY)
            color_dict[a] = cols_v
        res_tr = _pack_islands_to_grid(zd_dst, na_dst, color_dict)
        if res_tr is not None:
            V, F, C, _ = res_tr
            panels.append((f"{l_dst}: перенесённые группы",
                           [o3d_mesh(V, F, C), _uv_grid_lines(na_dst)]))

    # Панель: распределение ТЕПЛА на меше, который подгоняем (FBX, голова
    # transfer_dst_index). Тепло «впрыснуто» по границе и диффундирует внутрь —
    # это поле и определяет, как внутренние точки следуют за границей при warp.
    if show_heat and len(heads) > transfer_dst_index:
        l_h, na_h, zd_h, _r_h = heads[transfer_dst_index]
        if transfer_dst_index >= 1 and zmap:        # ячейки FLAME (как overlay)
            zd_h = _remap_zd(zd_h, zmap); na_h = heads[0][1]
        hc = _zone_heat_color_dict(zd_h, warp_heat_t)
        res_h = _pack_islands_to_grid(zd_h, na_h, hc)
        if res_h is not None:
            V, F, C, _ = res_h
            panels.append((f"{l_h}: тепло границы (warp, t={warp_heat_t})",
                           [o3d_mesh(V, F, C), _uv_grid_lines(na_h)]))

    # Раскладываем панели в ряд по X.
    gap = 1.3
    all_geoms = []
    for i, (title, geoms) in enumerate(panels):
        dx = gap * i
        for g in geoms:
            if hasattr(g, 'vertices') and len(np.asarray(g.vertices)):
                v = np.asarray(g.vertices).copy(); v[:, 0] += dx
                g.vertices = o3d.utility.Vector3dVector(v)
                g.compute_vertex_normals()
            elif hasattr(g, 'points') and len(np.asarray(g.points)):
                p = np.asarray(g.points).copy(); p[:, 0] += dx
                g.points = o3d.utility.Vector3dVector(p)
            all_geoms.append(g)
        print(f"    панель {i}: {title}")

    print(f"  Открываю окно UV-развёрток ({len(panels)} панелей)... "
          f"закрой клавишей Q")
    try:
        o3d.visualization.draw_geometries(
            all_geoms,
            window_name=f"UV-развёртки masked-зон ({mode}) — "
                        "головы + overlay + перенос групп (Q → выход)",
            width=1700, height=800, mesh_show_back_face=True)
    except Exception:
        import traceback
        print("  ⚠ Не удалось показать окно UV (draw_geometries):")
        traceback.print_exc()


def _boundary_vertices(F):
    """Уникальные граничные вершины острова (рёбра, входящие в 1 треугольник).
    Индексы — локальные (в системе строк острова, как uv/F)."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for tri in F:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            cnt[(min(a, b), max(a, b))] += 1
    bset = set()
    for (a, b), c in cnt.items():
        if c == 1:
            bset.add(a); bset.add(b)
    return np.array(sorted(bset), dtype=np.int64)


def _boundary_edges(F):
    """ВСЕ граничные рёбра острова как пары локальных индексов (рёбра, входящие
    лишь в 1 треугольник). В отличие от `_boundary_loop` не теряет доли при
    защемлениях (apex-вершина на границе дважды) и не создаёт фиктивных хорд."""
    from collections import defaultdict
    cnt = defaultdict(int)
    for tri in F:
        for k in range(3):
            a, b = int(tri[k]), int(tri[(k + 1) % 3])
            cnt[(min(a, b), max(a, b))] += 1
    be = [e for e, c in cnt.items() if c == 1]
    return np.array(be, dtype=np.int64) if be else np.zeros((0, 2), np.int64)


def _project_to_segments(pts, A, B):
    """Для каждой точки pts (M,2) — ближайшая точка на НАБОРЕ отрезков
    [A_i, B_i] (K отрезков). Проекция на реальные граничные рёбра (без
    упорядоченного цикла → без пропусков долей и фиктивных хорд)."""
    A = np.asarray(A, dtype=np.float64); B = np.asarray(B, dtype=np.float64)
    if len(A) == 0:
        return np.asarray(pts, dtype=np.float64).copy()
    AB = B - A
    ab2 = np.clip((AB * AB).sum(1), 1e-12, None)
    out = np.zeros_like(pts, dtype=np.float64)
    for i, q in enumerate(pts):
        tl = np.clip(((q - A) * AB).sum(1) / ab2, 0.0, 1.0)
        proj = A + tl[:, None] * AB             # ближайшая точка на каждом ребре
        j = int(np.argmin(((proj - q) ** 2).sum(1)))
        out[i] = proj[j]
    return out


def _heat_influence_weights(uv, F, bnd_idx, t):
    """Тепловое поле влияния граничных точек на остальные вершины острова.

    Решаем неявное тепло (MM + t·L) H = MM·E (multi-RHS, E — one-hot по границе)
    в UV-домене острова; строки нормируем → partition-of-unity. Граничные строки
    пиним в one-hot, чтобы граничные точки садились ТОЧНО на цели.

    Возвращает W (N, nb): вклад каждой граничной точки в смещение вершины."""
    N = len(uv)
    uv3 = np.column_stack([uv, np.zeros(N)])
    L, MM = build_operators(uv3, F, clamp_cot=True)   # робастно к тонким треуг.
    nb = len(bnd_idx)
    E = np.zeros((N, nb))
    E[bnd_idx, np.arange(nb)] = 1.0
    A = (MM + float(t) * L).tocsc()
    H = np.asarray(spla.spsolve(A, MM @ E))
    if H.ndim == 1:
        H = H[:, None]
    H = np.clip(H, 0.0, None)
    W = H / np.clip(H.sum(1, keepdims=True), 1e-12, None)
    W[bnd_idx] = 0.0
    W[bnd_idx, np.arange(nb)] = 1.0             # граница → точно на цель
    return W


def _boundary_heat_scalar(uv, F, bnd_idx, t):
    """Скалярное тепловое поле острова: тепло «впрыснуто» по всей границе
    (источник=1 на крайних вершинах) и неявно диффундирует внутрь —
    (MM + t·L)·h = MM·e. Граница ПИНИТСЯ в максимум (как в warp: каждая краевая
    вершина — полноценный источник), внутренние точки спадают к центру.
    Возвращает h в [0,1], длина N (по вершинам uv)."""
    N = len(uv)
    uv3 = np.column_stack([uv, np.zeros(N)])
    L, MM = build_operators(uv3, F, clamp_cot=True)   # робастно к тонким треуг.
    e = np.zeros(N)
    e[bnd_idx] = 1.0
    A = (MM + float(t) * L).tocsc()
    h = np.asarray(spla.spsolve(A, MM @ e)).ravel()
    h = np.clip(h, 0.0, None)
    rng = h.max() - h.min()
    h = (h - h.min()) / rng if rng > 1e-12 else np.zeros(N)
    h[bnd_idx] = 1.0                 # все краевые вершины — горячие (как в warp)
    return h


def _zone_heat_color_dict(zone_dict, t):
    """Цвет вершин каждого острова по тепловому полю границы (CMAP_HEAT).
    Возвращает dict {зона: (N,3)} для _pack_islands_to_grid."""
    cd = {}
    for a in zone_dict:
        uv, F = zone_dict[a][0], zone_dict[a][1]
        bnd = _boundary_vertices(F)
        if len(bnd) < 3:
            cd[a] = np.tile([0.2, 0.2, 0.2], (len(uv), 1))
            continue
        h = _boundary_heat_scalar(uv, F, bnd, t)
        cd[a] = to_colors(h, CMAP_HEAT)
    return cd


def _cap_targets_min_dist(src, targets, min_dist):
    """Ограничиваем подтяжку границы: после подгонки расстояние «точка → цель
    на ребре» должно быть НЕ БОЛЬШЕ min_dist (ближе/в ноль — допустимо).

    Точки дальше min_dist подтягиваем ровно до min_dist от ребра; точки уже
    ближе — оставляем на месте. min_dist<=0 → точное совпадение (как было).
    Возвращает фактические цели (куда реально встанет граница)."""
    if min_dist is None or float(min_dist) <= 0.0:
        return targets
    gap = targets - src
    dist = np.linalg.norm(gap, axis=1)
    scale = np.where(dist > min_dist, 1.0 - min_dist / np.clip(dist, 1e-12, None), 0.0)
    return src + gap * scale[:, None]


def _warp_island_heat(nd, F_d, ns, F_s, t=0.05, min_dist=0.0, return_corr=False,
                      lm_local=None, lm_targets=None):
    """Нежёсткая деформация острова FBX (nd) на остров FLAME (ns).

    Граничные точки FBX тянем к ближайшей точке на граничной ЛОМАНОЙ FLAME
    (проекция на отрезки, не на вершины); внутренние точки следуют за границей
    по тепловому полю влияния. Возвращает деформированный nd (для UV-NN).

    min_dist > 0 → после подгонки граница остаётся НЕ ДАЛЬШЕ min_dist от ребра
    (см. _cap_targets_min_dist); min_dist = 0 → точное совпадение.

    lm_local / lm_targets → ДОПОЛНИТЕЛЬНЫЕ якоря: локальные индексы вершин FBV
    (WKS-лендмарки) и их целевые UV-позиции на острове FLAME. Тянутся вместе с
    границей (одно тепловое поле). Если None/пусто → только граница.

    return_corr=True → также возвращает соответствие границы
    {'src': точки границы FBX до warp, 'dst': цели на ребре FLAME} (для виза)."""
    def _none_corr():
        return (nd, None) if return_corr else nd
    bnd_d = _boundary_vertices(F_d)
    if len(bnd_d) < 3:
        return _none_corr()
    be_s = _boundary_edges(F_s)                 # ВСЕ граничные рёбра FLAME
    if len(be_s) < 3:
        return _none_corr()
    src = nd[bnd_d].copy()
    targets = _project_to_segments(src, ns[be_s[:, 0]], ns[be_s[:, 1]])
    targets = _cap_targets_min_dist(src, targets, min_dist)
    # объединяем якоря границы с WKS-лендмарками (если есть)
    anchors = bnd_d
    src_pts = src
    tgt_pts = targets
    if lm_local is not None and len(lm_local):
        lm_local = np.asarray(lm_local, dtype=np.int64)
        keep = ~np.isin(lm_local, bnd_d)        # не дублируем граничные вершины
        lm_local = lm_local[keep]
        lm_t = np.asarray(lm_targets, dtype=np.float64)[keep]
        if len(lm_local):
            anchors = np.concatenate([bnd_d, lm_local])
            src_pts = np.concatenate([src, nd[lm_local]])
            tgt_pts = np.concatenate([targets, lm_t])
    disp = tgt_pts - src_pts                     # смещения всех якорей
    W = _heat_influence_weights(nd, F_d, anchors, t)
    warped = nd + W @ disp
    if return_corr:
        return warped, {'src': src, 'dst': targets}   # corr — только граница
    return warped


def _barycentric_interp(query, src_uv, src_F, values, k=12):
    """Барицентрическая интерполяция `values` (Ns, d) с вершин треугольной сетки
    (src_uv, src_F) в точки `query` (M, 2) в UV-домене.

    Для каждой точки ищем СОДЕРЖАЩИЙ её треугольник (через kNN по центроидам) и
    интерполируем значения по барицентрическим координатам. Если точка вне сетки
    — берём ближайший треугольник и клампим барикоорды (= ближайшая точка на
    треугольнике). Возвращает (M, d). Гладко внутри зоны (нет NN-ступенек)."""
    from scipy.spatial import cKDTree
    src_uv = np.asarray(src_uv, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    F = np.asarray(src_F)
    A = src_uv[F[:, 0]]; B = src_uv[F[:, 1]]; C = src_uv[F[:, 2]]
    cent = (A + B + C) / 3.0
    tree = cKDTree(cent)
    k = int(min(k, len(F)))
    _, cand = tree.query(query, k=k)
    if cand.ndim == 1:
        cand = cand[:, None]
    # предрасчёт базиса треугольников для барикоординат
    v0 = B - A; v1 = C - A
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    denom = d00 * d11 - d01 * d01
    denom = np.where(np.abs(denom) < 1e-20, 1e-20, denom)
    out = np.zeros((len(query), values.shape[1]), dtype=np.float64)
    for i, q in enumerate(query):
        best = None; best_err = np.inf
        for t in cand[i]:
            v2 = q - A[t]
            d20 = v0[t] @ v2; d21 = v1[t] @ v2
            v = (d11[t] * d20 - d01[t] * d21) / denom[t]
            w = (d00[t] * d21 - d01[t] * d20) / denom[t]
            u = 1.0 - v - w
            bc = np.array([u, v, w])
            if (bc >= -1e-6).all():                  # точка внутри треугольника
                best = (t, bc); best_err = 0.0
                break
            err = -min(bc.min(), 0.0)                # насколько вышли наружу
            if err < best_err:
                best_err = err; best = (t, np.clip(bc, 0.0, None))
        t, bc = best
        bc = bc / max(bc.sum(), 1e-12)               # ренормировка
        tri = F[t]
        out[i] = bc[0] * values[tri[0]] + bc[1] * values[tri[1]] + bc[2] * values[tri[2]]
    return out


def _normalize_island_uv(uv):
    """Центрируем остров и нормируем по RMS-радиусу → острова одной зоны на
    разных головах сравнимы в общем UV-фрейме (для UV-NN)."""
    c = uv.mean(0)
    r = np.sqrt(np.mean(np.sum((uv - c) ** 2, 1)))
    return (uv - c) / max(r, 1e-9)


def _zone_centroids(verts, partition, n_anchors):
    """Центроид каждой зоны (среднее её вершин) в нормализованном кадре."""
    verts = np.asarray(verts); partition = np.asarray(partition)
    cents = {}
    for a in range(int(n_anchors)):
        msk = (partition == a)
        if msk.any():
            cents[a] = verts[msk].mean(0)
    return cents


def match_zones_by_position(res_src, res_dst):
    """Сопоставляем зоны FBX→FLAME по БЛИЖАЙШЕМУ ЦЕНТРОИДУ зоны в общем
    нормализованном кадре (обе головы прошли normalize_bbox). One-to-one через
    венгерский алгоритм (linear_sum_assignment). НЕ зависит от порядка пика
    anchor'ов → глаз садится на глаз, нос на нос.

    Возвращает dict {fbx_zone_id: flame_zone_id}."""
    from scipy.optimize import linear_sum_assignment
    cs = _zone_centroids(res_src['verts'], res_src['partition'],
                         res_src['n_anchors'])
    cd = _zone_centroids(res_dst['verts'], res_dst['partition'],
                         res_dst['n_anchors'])
    if not cs or not cd:
        return {}
    src_ids = sorted(cs); dst_ids = sorted(cd)
    Cs = np.array([cs[i] for i in src_ids])
    Cd = np.array([cd[j] for j in dst_ids])
    D = np.linalg.norm(Cd[:, None, :] - Cs[None, :, :], axis=2)   # (dst, src)
    ri, ci = linear_sum_assignment(D)
    mapping = {dst_ids[r]: src_ids[c] for r, c in zip(ri, ci)}
    pairs = ", ".join(f"FBX{d}→FLAME{s} ({D[list(dst_ids).index(d),list(src_ids).index(s)]:.3f})"
                      for d, s in sorted(mapping.items()))
    print(f"  Сопоставление зон по позиции: {pairs}")
    return mapping


def transfer_deformations_uv(res_src, res_dst, flat=False, align_pca_icp=False,
                             warp_heat=False, warp_heat_t=0.05,
                             warp_min_dist=0.0, interp_delta=True,
                             zd_src=None, zd_dst=None,
                             wks_src=None, wks_dst=None):
    """UV-NN перенос деформаций (δ) и кластеров с res_src (FLAME) на res_dst
    (FBX).

    Для каждой зоны, общей для обеих голов: разворачиваем зону в UV-остров на
    каждой голове, нормируем оба острова (центроид + RMS-радиус) в общий фрейм,
    затем для каждой вершины FBX в зоне берём БЛИЖАЙШУЮ вершину FLAME в UV
    (cKDTree) и копируем её δ, global cluster id и зону.

    align_pca_icp=True → перед NN остров FBX дополнительно подгоняется к острову
    FLAME через PCA+ICP (точнее соответствие при разной форме развёртки).

    zd_src / zd_dst → уже готовые UV-острова {зона: (uv, F, gidx)} (напр.,
    зарелаксенные во вьюере). Если заданы — НЕ пересчитываем развёртку, работаем
    с ними. ВАЖНО: zd_dst должен быть в ИНДЕКСАЦИИ ЗОН res_dst (а не ремапнут к
    src), т.к. zmap считается заново ниже.

    Возвращает dict {delta (Nd,3), zone (Nd,), gcid (Nd,), covered (Nd bool),
    или None, если у какой-то головы нет partition (не multi-t).
    """
    from scipy.spatial import cKDTree
    if res_src.get('partition') is None or res_dst.get('partition') is None:
        print("  ⚠ Перенос невозможен: нет partition (требуется multi-t).")
        return None
    flame_flat = bool(flat)
    zd_s = zd_src if zd_src is not None else compute_zone_islands(
        res_src['verts'], res_src['faces'],
        res_src['partition'], res_src['n_anchors'], flat=flame_flat)
    zd_d = zd_dst if zd_dst is not None else compute_zone_islands(
        res_dst['verts'], res_dst['faces'],
        res_dst['partition'], res_dst['n_anchors'], flat=flame_flat)
    Nd = len(res_dst['verts'])
    delta = np.zeros((Nd, 3))
    zone = -np.ones(Nd, dtype=np.int64)
    gcid = -np.ones(Nd, dtype=np.int64)
    covered = np.zeros(Nd, dtype=bool)
    delta_src = np.asarray(res_src['delta_native'])
    gcid_src = np.asarray(res_src['vert_gcid'])
    # Пары зон FBX→FLAME по ПОЗИЦИИ (а не по индексу пика) — иначе глаз мог
    # лечь на нос при разном порядке кликов.
    zmap = match_zones_by_position(res_src, res_dst)
    for a in sorted(zd_d):
        a_src = zmap.get(a)
        if a_src is None or a_src not in zd_s:
            print(f"    zone FBX {a}: нет пары на FLAME — пропуск")
            continue
        uv_s, Fk_s, gi_s = zd_s[a_src]
        uv_d, Fk_d, gi_d = zd_d[a]
        ns = _normalize_island_uv(uv_s)
        nd = _normalize_island_uv(uv_d)
        if align_pca_icp:
            nd = _align_island_pca_icp(nd, ns)      # подгонка FBX→FLAME
        if warp_heat:
            # WKS-лендмарки этого острова → доп. якоря warp (нос↔нос и т.п.)
            lm_local = lm_targets = None
            if wks_src is not None and wks_dst is not None:
                s_loc = np.where(np.isin(gi_s, wks_src))[0]   # лендмарки FLAME
                d_loc = np.where(np.isin(gi_d, wks_dst))[0]   # лендмарки FBX
                if len(s_loc) and len(d_loc):
                    # каждый лендмарк FBX → ближайший лендмарк FLAME в UV
                    _, mi = cKDTree(ns[s_loc]).query(nd[d_loc])
                    lm_local = d_loc
                    lm_targets = ns[s_loc[mi]]
                    print(f"      WKS-match зона {a}: {len(d_loc)} FBX ↔ "
                          f"{len(s_loc)} FLAME лендмарков")
            nd = _warp_island_heat(nd, Fk_d, ns, Fk_s, t=warp_heat_t,
                                   min_dist=warp_min_dist,
                                   lm_local=lm_local, lm_targets=lm_targets)
        _, idx = cKDTree(ns).query(nd)
        src_g = gi_s[idx]
        # δ — БАРИЦЕНТРИЧЕСКАЯ интерполяция по треугольникам FLAME в UV (гладко
        # внутри зоны, без NN-ступенек). Метки gcid/zone дискретны → берём NN.
        if interp_delta:
            d_local = _barycentric_interp(nd, ns, Fk_s, delta_src[gi_s])
        else:
            d_local = delta_src[src_g]
        delta[gi_d] = d_local
        gcid[gi_d] = gcid_src[src_g]
        zone[gi_d] = a_src          # помечаем зоной FLAME (семантика групп)
        covered[gi_d] = True
        mode = "barycentric" if interp_delta else "UV-NN"
        print(f"    zone FBX {a} ← FLAME {a_src}: {len(gi_d)} верш. FBX ← "
              f"{len(gi_s)} верш. FLAME (δ {mode})")
    n_cov = int(covered.sum())
    print(f"  Перенос завершён: покрыто {n_cov}/{Nd} верш. FBX "
          f"({100*n_cov/max(Nd,1):.1f}%); непокрытые → δ=0.")
    return {'delta': delta, 'zone': zone, 'gcid': gcid, 'covered': covered}


def transfer_and_show_fbx(res_src, res_dst, out_fbx, flat=False,
                          smooth_iters=3, smooth_alpha=0.5, tr=None,
                          align_pca_icp=False):
    """Переносим δ FLAME→FBX по UV-NN, применяем к FBX, сглаживаем перенесённое
    поле δ (Laplacian δ-smoothing), дампим и показываем окно
    rest | deformed | deformed-smoothed.

    tr: опц. уже посчитанный результат transfer_deformations_uv (чтобы не
        считать перенос дважды — он же используется для 4-й UV-панели)."""
    print("\n" + "═" * 70)
    print("  ПЕРЕНОС ДЕФОРМАЦИЙ FLAME → FBX (UV nearest-neighbor)")
    print("═" * 70)
    if tr is None:
        tr = transfer_deformations_uv(res_src, res_dst, flat=flat,
                                      align_pca_icp=align_pca_icp)
    if tr is None:
        return
    verts_dst = np.asarray(res_dst['verts'])
    faces_dst = np.asarray(res_dst['faces'])
    delta = tr['delta']
    deformed = verts_dst + delta
    disp = np.linalg.norm(delta, axis=1)
    print(f"  max ||δ_transferred||  = {disp.max():.4f}")
    print(f"  mean ||δ_transferred|| = {disp.mean():.4f}")

    # Сглаживание перенесённого поля δ (UV-NN даёт ступеньки на швах зон) —
    # тот же Laplacian δ-smoothing, что и для HEAD 1.
    print(f"\n  Laplacian δ-smoothing FBX (iters={smooth_iters}, "
          f"alpha={smooth_alpha})...")
    delta_sm = smooth_delta(delta, faces_dst, n_iter=smooth_iters,
                            alpha=smooth_alpha)
    deformed_sm = verts_dst + delta_sm
    disp_sm = np.linalg.norm(delta_sm, axis=1)
    print(f"  max ||δ_smoothed||  = {disp_sm.max():.4f}")

    out_fbx.mkdir(parents=True, exist_ok=True)
    save_matrix_csv(out_fbx / "delta_transferred.csv", delta,
                    header="dx,dy,dz")
    save_matrix_csv(out_fbx / "deformed_transferred.csv", deformed,
                    header="x,y,z")
    save_matrix_csv(out_fbx / "delta_transferred_smoothed.csv", delta_sm,
                    header="dx,dy,dz")
    save_matrix_csv(out_fbx / "deformed_transferred_smoothed.csv", deformed_sm,
                    header="x,y,z")
    rows = np.column_stack([np.arange(len(verts_dst)), tr['zone'],
                            tr['gcid'], tr['covered'].astype(int)])
    save_matrix_csv(out_fbx / "clusters_transferred.csv",
                    rows.astype(np.float64),
                    header="vertex_idx,zone,global_cluster_id,covered")
    print(f"  Дампы переноса → {out_fbx}/ (delta_transferred[_smoothed], "
          f"deformed_transferred[_smoothed], clusters_transferred)")

    col_def = to_colors(disp, CMAP_DISP)
    col_sm = to_colors(disp_sm, CMAP_DISP)
    rest_col = np.tile([0.85, 0.75, 0.68], (len(verts_dst), 1))
    print("\n── ОКНО: FBX rest | deformed | deformed-smoothed (перенос δ) ──")
    show_meshes_side_by_side([
        (verts_dst, faces_dst, rest_col, "FBX rest"),
        (deformed, faces_dst, col_def, "FBX deformed (перенос)"),
        (deformed_sm, faces_dst, col_sm, "FBX deformed (smoothed)"),
    ], window_title="FBX: rest | deformed | smoothed — перенос δ FLAME→FBX "
                    "(Q → выход)")


if __name__ == "__main__":
    main()
