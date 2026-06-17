"""
Multi-anchor с MOTION GROUPS clustering для переноса деформации.

Идея:
  - Внутри каждой нагретой зоны кластеризуем вершины на motion-groups
    (группы где смещение, поворот и сжатие похожи)
  - Используем k-means на (3 dim motion = δ) + (3 dim position, с весом) для
    учёта одновременно похожести движения И геометрической локальности.
    Position-weight играет роль приближённой геодезической константы:
    вершины далеко по мешу не попадут в одну группу даже если их δ похожи.
  - Для каждой группы делаем полярное разложение (μ, R, S)
  - Переносим эти трансформации на голову 2 (с той же топологией)

В отличие от ws-переноса (одна стрелка на всю зону), motion-groups дают
несколько локальных трансформаций → лучшее сохранение внутренних паттернов.

Usage:
    python multi_anchor_motion_groups.py
"""

import argparse
import pickle
import subprocess
import sys
import tempfile
import time as time_mod
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


FLAME_PKL = ("Muscle-autoskinner/Assets/Meshes/FLAME/"
             "FLAME2023 Open for commercial use/flame2023_Open.pkl")
CMAP_HEAT = plt.get_cmap("hot")
CMAP_DISP = plt.get_cmap("cool")


# ── Базовые функции (из multi_anchor_flame.py) ───────────────────────────────

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


def load_custom_mesh(path):
    """FBX → OBJ через assimp → trimesh (process=True мержит UV-splits)."""
    import trimesh as _tm
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False); tmp.close()
    r = subprocess.run(["assimp", "export", path, tmp.name],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
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


def diffuse_static(L, M, src, total_time, steps=60):
    dt = total_time / steps
    solve = spla.factorized((M + dt * L).tocsc())
    A_src = float(M.diagonal()[src])
    u = np.zeros(L.shape[0]); u[src] = 1.0 / max(A_src, 1e-12)
    for _ in range(steps): u = solve(M @ u)
    return np.clip(u, 0, None)


def to_colors(values, cmap):
    v = np.clip(values, 0, None)
    if v.max() > 0: v = v / v.max()
    return cmap(v)[:, :3]


def o3d_mesh(verts, faces, colors=None):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                   o3d.utility.Vector3iVector(faces))
    m.compute_vertex_normals()
    if colors is not None: m.vertex_colors = o3d.utility.Vector3dVector(colors)
    else: m.paint_uniform_color([0.85, 0.75, 0.68])
    return m


def pick_vertices_up_to(verts, faces, head_name, max_n):
    """Shift+клик до max_n точек. Q закрывает — что выбрано, то и берём."""
    print(f"\n[{head_name}] Shift+клик до {max_n} точек тепла, затем Q.")
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(f"Выбери до {max_n} точек — {head_name}  (Q когда готово)",
                      1000, 800)
    vis.add_geometry(o3d_mesh(verts, faces))
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()
    chosen = [p.index for p in picked][:max_n]
    print(f"  Выбрано {len(chosen)} из ≤{max_n}")
    if len(chosen) < 1:
        s = input(f"  Не выбрано ни одной точки. Введи индекс вершины: ").strip()
        try:
            v = int(s)
            if 0 <= v < len(verts): chosen.append(v)
        except ValueError: pass
    return chosen


def pick_exact_n_vertices(verts, faces, head_name, n_required):
    """Shift+клик на ровно n_required точек. Если меньше — добор через терминал."""
    print(f"\n[{head_name}] Shift+клик на {n_required} точках тепла, затем Q.")
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(f"Выбери {n_required} точек — {head_name}", 1000, 800)
    vis.add_geometry(o3d_mesh(verts, faces))
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()
    chosen = [p.index for p in picked][:n_required]
    print(f"  Выбрано {len(chosen)} из {n_required}")
    while len(chosen) < n_required:
        s = input(f"  Добавить #{len(chosen)+1}: ").strip()
        try:
            v = int(s)
            if 0 <= v < len(verts): chosen.append(v)
        except ValueError: print("  Целое число")
    return chosen


# ── Polar decomposition (numpy) ──────────────────────────────────────────────

def polar_decomposition_analysis(heat, delta, verts, eps=1e-8):
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
    B_reg = B + eps * I3
    F = np.linalg.solve(B_reg, A.T).T
    U, sigma, Vt = np.linalg.svd(F)
    det_sign = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1.0, 1.0, det_sign])
    R = U @ D @ Vt
    S = R.T @ F; S = 0.5 * (S + S.T)
    eigvals, eigvecs = np.linalg.eigh(S)
    stretches = eigvals[::-1]
    axes = eigvecs[:, ::-1]
    delta_pred = mu[None] + p @ (F - I3).T
    residual = float(np.sqrt((w[:, None] * (delta - delta_pred) ** 2).sum() / W))
    return {'mu': mu, 'F': F, 'R': R, 'S': S, 'stretches': stretches,
            'axes': axes, 'residual': residual, 'c_rest': c_rest}


# ── Motion-group clustering ──────────────────────────────────────────────────

def cluster_zone_motion(heat, delta, verts, anchor_idx,
                          heat_threshold=0.05,
                          n_clusters_max=5,
                          position_weight=1.5,
                          min_cluster_size=4):
    """Кластеризует вершины в зоне нагрева на motion-groups.

    Возвращает топология-независимые дескрипторы кластеров:
        anchor_idx     : индекс anchor-зоны (для матчинга на target меше)
        c_rest         : центроид (3,) в норм. пространстве
        spatial_sigma  : пространственный разброс (1,)
        μ, R, S, ...   : полярная декомпозиция

    indices/heat_weights сохраняются ТОЛЬКО для визуализации на голове 1.
    Перенос работает БЕЗ них.
    """
    heat_max = max(heat.max(), 1e-12)
    active_mask = heat > heat_threshold * heat_max
    active_idx = np.where(active_mask)[0]
    if len(active_idx) < min_cluster_size * 2:
        return []

    a_verts = verts[active_idx]
    a_delta = delta[active_idx]
    a_heat  = heat[active_idx]

    # Features: motion + position
    d_scale = max(np.linalg.norm(a_delta, axis=1).max(), 1e-8)
    p_mean  = a_verts.mean(0)
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
        c_heat  = a_heat[mask]
        c_verts = a_verts[mask]
        c_delta = a_delta[mask]

        polar = polar_decomposition_analysis(c_heat, c_delta, c_verts)
        # Пространственный разброс кластера (RMS вокруг центроида)
        c_rest = polar['c_rest']
        rms = np.sqrt(
            (c_heat * np.linalg.norm(c_verts - c_rest, axis=-1) ** 2).sum()
            / max(c_heat.sum(), 1e-12)
        )
        clusters.append({
            'anchor_idx':    anchor_idx,
            'spatial_sigma': max(rms, 1e-4),     # для Гауссиана membership на target
            **polar,
            # ↓ для визуализации на голове 1, не используется в переносе:
            '_indices':      active_idx[mask],
            '_heat_weights': c_heat,
        })
    return clusters


def assign_target_to_source_clusters(verts_target, faces_target,
                                       heat_target_per_anchor,
                                       src_clusters_list,
                                       heat_threshold=0.05,
                                       geodesic_factor=3.0):
    """Voronoi-разбиение зон головы 2 по центроидам source-кластеров +
    ЧЕСТНЫЙ геодезический лимит через Dijkstra по поверхности.

    Алгоритм для каждой anchor-зоны:
      1) Berlonging to zone — heat_target[anchor, v] > heat_threshold · max
         (geodesic-aware через диффузию)
      2) Для каждого source-кластера в этом anchor'е:
         a. Находим seed-вершину на target меше (ближайшую к c_rest_source)
         b. Запускаем Dijkstra от seed по рёбрам меша, ограниченный
            R_max = σ_source · geodesic_factor (в норм. единицах bbox)
         c. Получаем geodesic_distance[v] для всех вершин в радиусе
      3) Каждая active вершина приписывается к кластеру с МИНИМАЛЬНОЙ
         геодезической дистанцией. Если все ∞ — пропускается.

    geodesic_factor:
      3.0 (default) — мягкий лимит, кластер ~3σ по поверхности
      2.0 — строже
      ∞ — без лимита (все вершины распределяются по ближайшему центроиду
            по xyz — fallback на старое поведение)
    """
    N_target = verts_target.shape[0]

    # Строим adjacency один раз (используется всеми кластерами)
    print("  Строю adjacency для honest geodesic (Dijkstra)...")
    neighbors = build_vertex_adjacency(N_target, verts_target, faces_target)

    target_clusters = []

    # Группируем source по anchor
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

        # Для каждого source-кластера: запускаем Dijkstra на target меше
        geo_dists = np.full((N_target, K), np.inf)                     # (N, K)
        for k, s in enumerate(src_list):
            seed = find_nearest_vertex(verts_target, s['c_rest'])
            R_max = s['spatial_sigma'] * geodesic_factor
            if not np.isfinite(R_max):
                R_max = float('inf')
            dists_k = geodesic_distance_dijkstra(neighbors, seed, R_max)
            geo_dists[:, k] = dists_k

        # Активные вершины: ближайший достижимый кластер по геодезик-дистанции
        dists_active = geo_dists[active_idx]                           # (M, K)
        nearest = np.argmin(dists_active, axis=1)                      # (M,)
        reachable = ~np.all(np.isinf(dists_active), axis=1)

        for k, s in enumerate(src_list):
            mask = (nearest == k) & reachable
            if not mask.any(): continue
            t_indices = active_idx[mask]
            t_heat    = heat_a[t_indices]
            W = max(t_heat.sum(), 1e-12)
            c_target = (t_heat[:, None] * verts_target[t_indices]).sum(0) / W
            target_clusters.append({
                'source':         s,
                'target_indices': t_indices,
                'target_heat':    t_heat,
                'c_target':       c_target,
                'geo_dists':      dists_active[mask, k],
            })

    return target_clusters


def build_vertex_adjacency(N, verts, faces):
    """Возвращает (neighbors, edge_lengths) для BFS по поверхности.

    neighbors[v] = list of (neighbor_idx, edge_length) tuples
    Используем edge_length чтобы BFS давал ИСТИННОЕ геодезическое расстояние
    (по ломаной из рёбер меша), а не количество хопов.
    """
    neighbors = [[] for _ in range(N)]
    seen_edges = set()
    for f in faces:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = int(f[i]), int(f[j])
                key = (min(a, b), max(a, b))
                if key in seen_edges: continue
                seen_edges.add(key)
                d = float(np.linalg.norm(verts[a] - verts[b]))
                neighbors[a].append((b, d))
                neighbors[b].append((a, d))
    return neighbors


def geodesic_distance_dijkstra(neighbors, source_idx, max_distance):
    """Dijkstra от source_idx по графу рёбер, ограниченный max_distance.

    Возвращает массив (N,) с расстояниями. Вершины дальше max_distance — np.inf.
    Использует heapq для эффективности (O(E log V)).
    """
    import heapq
    N = len(neighbors)
    dist = np.full(N, np.inf)
    dist[source_idx] = 0.0
    pq = [(0.0, source_idx)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > max_distance: continue
        if d > dist[u]: continue
        for v, w in neighbors[u]:
            nd = d + w
            if nd < dist[v] and nd <= max_distance:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def find_nearest_vertex(verts, point):
    """Индекс вершины ближайшей к point (Euclidean) — для seed'а Dijkstra."""
    return int(np.argmin(np.linalg.norm(verts - point, axis=-1)))


def build_avg_neighbor_matrix(N, faces):
    """Sparse матрица W: (W @ δ)[v] = среднее значение δ среди соседей v."""
    rows, cols = [], []
    for f in faces:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        rows += [a, a, b, b, c, c]
        cols += [b, c, a, c, a, b]
    # Удалим дубликаты
    edges = set(zip(rows, cols))
    rows = np.array([r for r, _ in edges])
    cols = np.array([c for _, c in edges])
    data = np.ones(len(rows))
    A = sp.csr_matrix((data, (rows, cols)), shape=(N, N))
    row_sums = np.array(A.sum(axis=1)).ravel().clip(min=1.0)
    D_inv = sp.diags(1.0 / row_sums)
    return (D_inv @ A).tocsr()


def smooth_delta_field(delta, faces, n_iter=5, alpha=0.5):
    """Лапласовское сглаживание векторного поля δ через усреднение соседей.

    Для каждой итерации:
        δ_new[v] = (1 - α) · δ[v] + α · mean(δ[neighbors of v])

    n_iter: количество итераций сглаживания (0 = выкл)
    alpha:  per-iteration blend (0 = без изменений, 1 = полное усреднение)

    Возвращает сглаженный δ той же формы (N, 3).
    """
    if n_iter <= 0:
        return delta.copy()
    N = len(delta)
    W = build_avg_neighbor_matrix(N, faces)
    d = delta.copy()
    for _ in range(n_iter):
        d_avg = W @ d                                                  # (N, 3)
        d = (1 - alpha) * d + alpha * d_avg
    return d


def apply_target_clusters_transfer(verts_target, target_clusters):
    """Применяет (μ, R, S) с source кластера к назначенным target вершинам.

    Формула: δ[v] = μ_s + (R_s S_s − I) · (verts_target[v] − c_target)
    где c_target — heat-weighted центроид присвоенных target вершин.
    """
    N = verts_target.shape[0]
    delta = np.zeros((N, 3))
    weight = np.zeros(N)
    I3 = np.eye(3)

    for tc in target_clusters:
        s    = tc['source']
        c_t  = tc['c_target']
        RS   = s['R'] @ s['S']
        for j, v_idx in enumerate(tc['target_indices']):
            r = verts_target[v_idx] - c_t
            d = s['mu'] + (RS - I3) @ r
            w = tc['target_heat'][j]
            delta[v_idx] += w * d
            weight[v_idx] += w

    valid = weight > 1e-12
    delta[valid] = delta[valid] / weight[valid, None]
    return delta


def axis_angle_from_R(R):
    trace = np.trace(R)
    cos_a = np.clip((trace - 1) * 0.5, -1 + 1e-9, 1 - 1e-9)
    angle = np.arccos(cos_a)
    sin_a = max(np.sin(angle), 1e-9)
    axis = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]]) / (2 * sin_a)
    return axis, angle


# ── Палитра цветов для кластеров ──────────────────────────────────────────────

def make_cluster_palette(n):
    """HSV golden-ratio sampling для различимых цветов."""
    import colorsys
    if n == 0: return np.zeros((0, 3))
    h = (np.arange(n) * 0.61803398875) % 1.0
    s = 0.75 + 0.25 * (np.arange(n) % 2)
    v = 0.85 + 0.15 * ((np.arange(n) // 2) % 2)
    return np.array([colorsys.hsv_to_rgb(hi, si, vi) for hi, si, vi in zip(h, s, v)])


# ── Анимация диффузии ───────────────────────────────────────────────────────

def animate_multi_source_diffusion(verts1, faces1, verts2, faces2,
                                    L1, MM1, L2, MM2,
                                    srcs1, srcs2, t1, t2, steps, gap, fps=24):
    N = len(srcs1)
    dt1, dt2 = t1 / steps, t2 / steps
    solve1 = spla.factorized((MM1 + dt1 * L1).tocsc())
    solve2 = spla.factorized((MM2 + dt2 * L2).tocsc())
    A1 = np.array(MM1.diagonal()); A2 = np.array(MM2.diagonal())
    u1 = np.zeros((N, L1.shape[0])); u2 = np.zeros((N, L2.shape[0]))
    for ai in range(N):
        u1[ai, srcs1[ai]] = 1.0 / max(A1[srcs1[ai]], 1e-12)
        u2[ai, srcs2[ai]] = 1.0 / max(A2[srcs2[ai]], 1e-12)
    verts2_show = verts2.copy(); verts2_show[:, 0] += gap
    mesh1 = o3d_mesh(verts1, faces1); mesh2 = o3d_mesh(verts2_show, faces2)
    spheres = []
    for s in srcs1:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        sph.translate(verts1[s]); sph.paint_uniform_color([1, 0, 0])
        sph.compute_vertex_normals(); spheres.append(sph)
    for s in srcs2:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        sph.translate(verts2_show[s]); sph.paint_uniform_color([1, 0, 0])
        sph.compute_vertex_normals(); spheres.append(sph)
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"Diffusion {N} sources  (Q)", 1400, 750)
    for g in [mesh1, mesh2, *spheres]: vis.add_geometry(g)
    vis.get_render_option().mesh_show_back_face = True
    frame_dt = 1.0 / fps
    for _ in range(steps):
        t0 = time_mod.perf_counter()
        for ai in range(N):
            u1[ai] = solve1(MM1 @ u1[ai])
            u2[ai] = solve2(MM2 @ u2[ai])
        mesh1.vertex_colors = o3d.utility.Vector3dVector(to_colors(u1.sum(0), CMAP_HEAT))
        mesh2.vertex_colors = o3d.utility.Vector3dVector(to_colors(u2.sum(0), CMAP_HEAT))
        vis.update_geometry(mesh1); vis.update_geometry(mesh2)
        if not vis.poll_events(): break
        vis.update_renderer()
        wait = frame_dt - (time_mod.perf_counter() - t0)
        if wait > 0: time_mod.sleep(wait)
    print("Диффузия завершена. Q.")
    while vis.poll_events(): vis.update_renderer()
    vis.destroy_window()
    return np.clip(u1, 0, None), np.clip(u2, 0, None)


# ── Пресеты ──────────────────────────────────────────────────────────────────

SHAPE_PRESETS = {
    0: ("Нейтральная", {}), 1: ("Широкое лицо", {0: 2.5, 1: -1.5}),
    2: ("Узкое вытянутое", {0: -2.0, 1: 2.0}), 3: ("Крупная голова", {0: 3.0, 2: 1.5}),
    4: ("Детское лицо", {1: -2.0, 2: -1.5, 4: -1.0}),
    5: ("Угловатое мужское", {0: -1.5, 3: 2.5, 5: -1.0}),
    6: ("Округлое мягкое", {1: -2.0, 2: -2.0, 5: 1.5}),
}
EXPR_PRESETS = {
    0: ("Без экспрессии", {}), 1: ("Экспрессия A", {300: 8.0}),
    2: ("Экспрессия B", {301: 8.0}), 3: ("Экспрессия C", {302: 8.0}),
    4: ("Экспрессия D", {303: 8.0}), 5: ("Mix A+B", {300: 5.0, 301: 5.0}),
    6: ("Mix C+D", {302: 5.0, 303: 5.0}),
}


def ask_preset(presets, title):
    print(f"\n  {title}")
    for k, (n, _) in presets.items(): print(f"   {k}. {n}")
    while True:
        try:
            x = int(input("Номер: ").strip())
            if x in presets: return dict(presets[x][1])
        except ValueError: pass


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
    ap.add_argument("-n", "--n", type=int, default=20,
                    help="макс. число точек на голове 1 (Q закрывает раньше). "
                         "На голове 2 будет столько же сколько поставил на 1й.")
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--n-clusters", type=int, default=5,
                    help="макс. число motion-groups в каждой зоне")
    ap.add_argument("--heat-threshold", type=float, default=0.05,
                    help="доля от heat.max() ниже которой вершина игнорится")
    ap.add_argument("--position-weight", type=float, default=1.5,
                    help="вес позиции vs motion в feature space")
    ap.add_argument("--smooth-iters", type=int, default=50,
                    help="Laplacian smoothing итераций после переноса (0 = выкл)")
    ap.add_argument("--smooth-alpha", type=float, default=0.5,
                    help="сила сглаживания на итерацию (0..1)")
    ap.add_argument("--fbx", default=None,
                    help="Путь к FBX для головы 2 (вместо второй FLAME формы)")
    ap.add_argument("--geodesic-factor", type=float, default=3.0,
                    help="радиус досягаемости кластера = σ · этот множитель "
                         "(3=мягко, 1.5=строго, inf=без лимита)")
    args = ap.parse_args()

    N = args.n
    print(f"Загружаю FLAME: {args.flame}")
    v_t, sd, faces1 = load_flame(args.flame)

    # ── Голова 1 — всегда FLAME ──────────────────────────────────────────────
    shape1 = ask_preset(SHAPE_PRESETS, "ШАГ 1 — Форма головы 1 (FLAME)")
    v1_raw = apply_betas(v_t, sd, shape1)
    verts1 = normalize_bbox(v1_raw)

    # ── Голова 2 — FLAME или FBX ─────────────────────────────────────────────
    if args.fbx:
        print(f"Загружаю FBX (голова 2): {args.fbx}")
        v2_raw, faces2 = load_custom_mesh(args.fbx)
        verts2 = normalize_bbox(v2_raw)
        is_custom = True
        head2_name = "FBX"
        shape2 = None
        print(f"  FBX меш: {len(v2_raw)} вершин, {len(faces2)} граней")
    else:
        shape2 = ask_preset(SHAPE_PRESETS, "ШАГ 2 — Форма головы 2 (FLAME)")
        v2_raw = apply_betas(v_t, sd, shape2)
        verts2 = normalize_bbox(v2_raw)
        faces2 = faces1
        is_custom = False
        head2_name = "FLAME-B"

    # Голова 1: до N точек, Q закрывает
    src1 = pick_vertices_up_to(verts1, faces1, "Голова 1 (FLAME-A)", N)
    n_actual = len(src1)
    print(f"\n→ На голове 2 нужно поставить ровно {n_actual} точек\n")
    # Голова 2: ровно столько же сколько было на 1й
    src2 = pick_exact_n_vertices(verts2, faces2, f"Голова 2 ({head2_name})", n_actual)
    N = n_actual

    expr = ask_preset(EXPR_PRESETS, "ШАГ 3 — Экспрессия (применяется к ГОЛОВЕ 1)")

    def normalized_expr(v_raw_rest, betas_full):
        v_raw_e = apply_betas(v_t, sd, betas_full)
        m = v_raw_rest.mean(0)
        d = np.linalg.norm((v_raw_rest - m).max(0) - (v_raw_rest - m).min(0))
        return (v_raw_e - m) / (d + 1e-12)

    head1_expr = normalized_expr(v1_raw, {**shape1, **expr})
    delta1 = head1_expr - verts1
    # head2_expr и delta2_native НЕ ВЫЧИСЛЯЕМ:
    # перенос идёт от source-кластеров через Voronoi, родная экспрессия головы 2 не нужна.

    t_anim = args.time if args.time is not None else ask_float(
        "Время диффузии t", 0.002)
    num_steps = args.steps if args.steps is not None else ask_int(
        "Шагов диффузии", 10, 300, 60)

    print("\nСтрою Laplacian-ы...")
    L1, MM1 = build_operators(verts1, faces1)
    L2, MM2 = build_operators(verts2, faces2)

    bx = verts1[:, 0].max() - verts1[:, 0].min()
    gap = bx * 1.3
    print("Анимирую диффузию...")
    heat1, heat2 = animate_multi_source_diffusion(
        verts1, faces1, verts2, faces2, L1, MM1, L2, MM2,
        src1, src2, t_anim, t_anim, num_steps, gap, fps=args.fps)

    # ── MOTION GROUPS: кластеризуем каждую зону на голове 1 ──────────────────
    print("\n" + "═"*72)
    print("  MOTION GROUPS CLUSTERING на голове 1 (топология-независимые дескрипторы)")
    print("═"*72)
    clusters_per_anchor = []
    for a in range(N):
        cls = cluster_zone_motion(
            heat1[a], delta1, verts1, anchor_idx=a,
            heat_threshold=args.heat_threshold,
            n_clusters_max=args.n_clusters,
            position_weight=args.position_weight,
        )
        clusters_per_anchor.append(cls)
        print(f"\n  Anchor #{a}: {len(cls)} motion-groups")
        for ci, cl in enumerate(cls):
            ax, ang = axis_angle_from_R(cl['R'])
            print(f"    [{ci}] {len(cl['_indices']):4d} verts  "
                  f"μ={cl['mu'].round(4)} (|μ|={np.linalg.norm(cl['mu']):.4f})  "
                  f"rot={np.degrees(ang):.1f}°  "
                  f"stretch={cl['stretches'].round(3)}  "
                  f"σ_spatial={cl['spatial_sigma']:.4f}  "
                  f"resid={cl['residual']:.4f}")

    # ── ПРИСВОЕНИЕ вершин головы 2 к source-кластерам ────────────────────────
    # Каждая вершина в anchor-зоне на голове 2 идёт к ближайшему source-кластеру
    # (того же anchor'а) по евклидову расстоянию до его центроида.
    # Никакого независимого k-means на голове 2 — разбиение определяется source.
    print("\n" + "═"*72)
    print("  ПРИСВОЕНИЕ вершин головы 2 к source-кластерам (Voronoi)")
    print("═"*72)
    src_flat = [cl for cls in clusters_per_anchor for cl in cls]
    target_clusters = assign_target_to_source_clusters(
        verts2, faces2, heat2, src_flat,
        heat_threshold=args.heat_threshold,
        geodesic_factor=args.geodesic_factor,
    )
    print(f"  Получили {len(target_clusters)} target-кластеров из {len(src_flat)} source")
    for tc in target_clusters:
        s = tc['source']
        a = s['anchor_idx']
        ax, ang = axis_angle_from_R(s['R'])
        print(f"    anchor #{a}: target c={tc['c_target'].round(3)}  "
              f"src c={s['c_rest'].round(3)}  "
              f"verts: {len(tc['target_indices'])} (target) / "
              f"{len(s['_indices'])} (source)")

    # ── Применяем source трансформации к назначенным target вершинам ─────────
    delta2_raw = apply_target_clusters_transfer(verts2, target_clusters)
    print(f"\n  max ||δ2 (raw)|| = {np.linalg.norm(delta2_raw, axis=1).max():.4f}")

    # ── Сглаживание векторного поля δ_2 (Laplacian smoothing) ────────────────
    if args.smooth_iters > 0:
        delta2 = smooth_delta_field(
            delta2_raw, faces2,
            n_iter=args.smooth_iters,
            alpha=args.smooth_alpha,
        )
        print(f"  Laplacian smoothing: {args.smooth_iters} iters, α={args.smooth_alpha}")
        print(f"  max ||δ2 (smoothed)|| = {np.linalg.norm(delta2, axis=1).max():.4f}")
        # На сколько усреднилось
        diff_smooth = float(np.linalg.norm(delta2 - delta2_raw, axis=1).max())
        print(f"  max сдвиг от сглаживания = {diff_smooth:.4f}")
    else:
        delta2 = delta2_raw

    head2_def = verts2 + delta2

    # ── Палитра цветов на каждый source-кластер ──────────────────────────────
    total_clusters = sum(len(cls) for cls in clusters_per_anchor)
    palette = make_cluster_palette(max(total_clusters, 1))
    # Привяжем уникальный цвет к каждому source-кластеру (по id объекта)
    source_color = {}
    color_idx = 0
    for cls in clusters_per_anchor:
        for cl in cls:
            source_color[id(cl)] = palette[color_idx]
            color_idx += 1

    # ── ОКНО 1: morph rest → deformed на обеих головах ───────────────────────
    print("\nОкно 1: morph (head1=блендшейп, head2=cluster-transfer)")
    target1 = head1_expr
    target2 = head2_def

    head2_show   = verts2.copy();   head2_show[:, 0]   += gap
    target2_show = target2.copy(); target2_show[:, 0] += gap

    col1_morph = to_colors(np.linalg.norm(delta1, axis=1), CMAP_DISP)
    col2_morph = to_colors(np.linalg.norm(delta2, axis=1), CMAP_DISP)

    mesh1 = o3d_mesh(verts1.copy(), faces1, col1_morph)
    mesh2 = o3d_mesh(head2_show.copy(), faces2, col2_morph)
    spheres = []
    for s in src1:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        sph.translate(verts1[s]); sph.paint_uniform_color([1, 0, 0])
        sph.compute_vertex_normals(); spheres.append(sph)
    for s in src2:
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
        sph.translate(head2_show[s]); sph.paint_uniform_color([1, 0, 0])
        sph.compute_vertex_normals(); spheres.append(sph)

    vis = o3d.visualization.Visualizer()
    vis.create_window("Morph rest → deformed  (Q)", 1400, 750)
    for g in [mesh1, mesh2, *spheres]: vis.add_geometry(g)
    vis.get_render_option().mesh_show_back_face = True
    frames = 40; fps = 30; frame_dt = 1.0 / fps
    for f in range(frames + 1):
        t0 = time_mod.perf_counter()
        a_ = f / frames
        v1 = (1 - a_) * verts1 + a_ * target1
        v2 = (1 - a_) * head2_show + a_ * target2_show
        mesh1.vertices = o3d.utility.Vector3dVector(v1)
        mesh2.vertices = o3d.utility.Vector3dVector(v2)
        mesh1.compute_vertex_normals(); mesh2.compute_vertex_normals()
        vis.update_geometry(mesh1); vis.update_geometry(mesh2)
        if not vis.poll_events(): break
        vis.update_renderer()
        wait = frame_dt - (time_mod.perf_counter() - t0)
        if wait > 0: time_mod.sleep(wait)
    print("Морф готов. Q.")
    while vis.poll_events(): vis.update_renderer()
    vis.destroy_window()

    # ── ОКНО 2: motion-groups на ОБЕИХ головах (в самом конце) ───────────────
    print("\nОкно 2: MOTION GROUPS на голове 1 (source) и голове 2 (target по Voronoi)")

    # Цвета вершин головы 1: по принадлежности к source-кластеру (по максимальному heat)
    vert_colors1 = np.tile([0.85, 0.75, 0.68], (len(verts1), 1))
    vert_weight1 = np.zeros(len(verts1))
    for cls in clusters_per_anchor:
        for cl in cls:
            col = source_color[id(cl)]
            for j, v_idx in enumerate(cl['_indices']):
                w = cl['_heat_weights'][j]
                if w > vert_weight1[v_idx]:
                    vert_weight1[v_idx] = w
                    vert_colors1[v_idx] = col

    # Цвета вершин головы 2: по принадлежности к target-кластеру → source-color матча
    vert_colors2 = np.tile([0.85, 0.75, 0.68], (len(verts2), 1))
    vert_weight2 = np.zeros(len(verts2))
    for tc in target_clusters:
        col = source_color[id(tc['source'])]
        for j, v_idx in enumerate(tc['target_indices']):
            w = tc['target_heat'][j]
            if w > vert_weight2[v_idx]:
                vert_weight2[v_idx] = w
                vert_colors2[v_idx] = col

    # Сглаживаем цвета головы 2 теми же параметрами что и δ_2 —
    # чтобы визуально граница между моушн-группами совпадала с реальной
    # post-smoothing структурой деформации.
    if args.smooth_iters > 0:
        vert_colors2 = smooth_delta_field(
            vert_colors2, faces2,
            n_iter=args.smooth_iters,
            alpha=args.smooth_alpha,
        )
        vert_colors2 = np.clip(vert_colors2, 0, 1)
        print(f"  Цвета головы 2 сглажены теми же {args.smooth_iters} итерациями "
              f"(α={args.smooth_alpha})")

    # Показываем обе головы рядом, в deformed состоянии (для наглядности)
    head1_def_show = head1_expr.copy()
    head2_def_show = head2_def.copy(); head2_def_show[:, 0] += gap

    o3d.visualization.draw_geometries(
        [
            o3d_mesh(head1_def_show, faces1, vert_colors1),
            o3d_mesh(head2_def_show, faces2, vert_colors2),
        ],
        window_name="Motion-groups: HEAD1 (source) | HEAD2 (Voronoi target) — цвета матчатся",
        width=1400, height=750, mesh_show_back_face=True,
    )


if __name__ == "__main__":
    main()
