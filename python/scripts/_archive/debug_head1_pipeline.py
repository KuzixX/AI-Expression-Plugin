"""
Debug pipeline для ГОЛОВЫ 1 (source) — пошаговая визуализация.

Шаги:
  1. Загружаем FLAME, выбираем shape preset
  2. Выбираем N anchor-точек (Shift+клик)
  3. Указываем время/шаги диффузии
  4. ОКНО 1: анимация диффузии (видим как тепло расползается)
  5. Применяем блендшейп → δ_native (правильная мимика)
  6. ОКНО 2: rest и deformed head (родная экспрессия для сравнения)
  7. Кластеризуем зоны (motion-groups) + полярная декомпозиция
  8. ОКНО 3: head с раскраской по кластерам + стрелки μ
  9. Реконструируем δ из кластеров (линейная аппроксимация)
 10. ОКНО 4: rest | δ_native | δ_reconstructed (side-by-side)
 11. Сглаживаем δ_reconstructed Laplacian'ом
 12. ОКНО 5: δ_native | δ_reconstructed | δ_smoothed (3 в ряд)

Это позволяет визуально проверить:
  - Heat зоны корректные?
  - K-means разделил движения по анатомии?
  - Полярная декомпозиция захватила движение (μ, R, S)?
  - Линейная реконструкция близка к оригиналу?
  - Сглаживание не убило сигнал?
"""

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
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


FLAME_PKL = ("Muscle-autoskinner/Assets/Meshes/FLAME/"
             "FLAME2023 Open for commercial use/flame2023_Open.pkl")
CMAP_HEAT = plt.get_cmap("hot")
CMAP_DISP = plt.get_cmap("cool")


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
    """Сохраняет result Voronoi-разбиения на target меш."""
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
            entry['display_color'] = cluster_color_map.get(
                id(s), [0.5, 0.5, 0.5]).tolist() \
                if hasattr(cluster_color_map.get(id(s)), 'tolist') \
                else list(cluster_color_map.get(id(s), [0.5, 0.5, 0.5]))
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
                                      heat_threshold=0.05,
                                      geodesic_factor=3.0):
    """Voronoi-разбиение target меша по геодезик-радиусам source кластеров."""
    N_t = verts_target.shape[0]
    print(f"  Строю Dijkstra adjacency на target меше ({N_t} верт)...")
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
            seed = find_nearest_vertex(verts_target, s['c_rest'])
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


def pick_vertices(verts, faces, head_name, max_n=20):
    print(f"\n[{head_name}] Shift+клик до {max_n} точек, Q закроет.")
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(f"Выбери точки — {head_name}", 1000, 800)
    vis.add_geometry(o3d_mesh(verts, faces))
    vis.run()
    picked = vis.get_picked_points()
    vis.destroy_window()
    chosen = [p.index for p in picked][:max_n]
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
        'n_anchors':        ask('n_anchors',       'max anchor pts',  int),
        'fbx_path':         input(f"FBX path (Enter = skip transfer) [{defaults.get('fbx_path','')}]: ").strip() or defaults.get('fbx_path', ''),
        'geodesic_factor':  ask('geodesic_factor', 'geodesic_factor', float),
        '_ok': True,
    }
    print("═" * 70)
    return params


def gui_setup_dialog(defaults):
    """Tkinter диалог. При отсутствии _tkinter → auto fallback на console."""
    try:
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
    except ImportError:
        print("⚠ tkinter недоступен в этом Python (pyenv обычно без _tkinter).")
        print("  Использую console-режим. Чтобы починить GUI:")
        print("    brew install tcl-tk")
        print("    pyenv uninstall 3.11.9 && pyenv install 3.11.9")
        return console_setup_dialog(defaults)

    root = tk.Tk()
    root.title("Debug Pipeline — Setup")
    root.geometry("620x720")
    root.resizable(True, True)
    root.minsize(540, 600)

    # Внешняя рамка с отступами от края окна
    frame = tk.Frame(root, padx=20, pady=15)
    frame.pack(fill='both', expand=True)
    # Растянем колонку для entry
    frame.columnconfigure(1, weight=1)

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
            result['n_anchors']       = int(vars_['n_anchors'].get())
            result['fbx_path']        = vars_['fbx_path'].get().strip()
            result['geodesic_factor'] = float(vars_['geodesic_factor'].get())
            result['_ok'] = True
            root.destroy()
        except Exception as e:
            messagebox.showerror("Ошибка ввода", f"Неверное значение: {e}")

    def on_cancel():
        root.destroy()

    row += 1
    btn_frame = tk.Frame(frame)
    btn_frame.grid(row=row, column=0, columnspan=2, pady=(20, 5))
    tk.Button(btn_frame, text="START ▶", command=on_start, bg="#2E7D32", fg="white",
              font=("Arial", 12, "bold"), width=12, height=2).pack(side='left', padx=10)
    tk.Button(btn_frame, text="Отмена", command=on_cancel,
              font=("Arial", 11), width=10, height=2).pack(side='left', padx=10)

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
                # сдвигаем дополнительные геометрии на тот же offset что меш
                if hasattr(g, 'points'):
                    pts = np.asarray(g.points)
                    pts_shifted = pts.copy()
                    pts_shifted[:, 0] += gap * i
                    g.points = o3d.utility.Vector3dVector(pts_shifted)
                geoms.append(g)
    o3d.visualization.draw_geometries(geoms, window_name=window_title,
                                       width=1600, height=800, mesh_show_back_face=True)


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
        'custom_betas':    '',
        'fbx_path':        '',
        'geodesic_factor': 3.0,
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

        # Перенос source-кластеров на FBX через Voronoi + Dijkstra
        print(f"\n  Voronoi-разбиение FBX (geodesic_factor={params['geodesic_factor']})...")
        src_flat = [cl for cls in clusters_per_anchor for cl in cls]
        target_clusters = assign_target_to_source_clusters(
            verts_fbx, faces_fbx, heat_fbx, src_flat,
            heat_threshold=heat_thresh,
            geodesic_factor=params['geodesic_factor'],
        )
        print(f"  Получено {len(target_clusters)} target-кластеров из {len(src_flat)} source")

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
        show_meshes_side_by_side([
            (verts,     faces,     vert_colors,     "head1_clusters_rest"),
            (verts_fbx, faces_fbx, vert_colors_fbx, "fbx_clusters_rest"),
        ], extra_geometries=[arrows, fbx_extras + fbx_src_extras],
           window_title="ОКНО 6a: HEAD 1 clusters | FBX clusters (Q → продолжить)")

        # Сохраняем результат Voronoi-разбиения
        save_target_clusters_json(
            OUT_FBX / "target_clusters.json", target_clusters,
            cluster_color_map=cluster_color_map)
        print(f"  → saved FBX target_clusters.json ({len(target_clusters)} clusters)")

        delta_fbx_raw = apply_target_clusters_transfer(verts_fbx, target_clusters)
        print(f"  max ||δ_fbx (raw)|| = {np.linalg.norm(delta_fbx_raw, axis=1).max():.4f}")

        save_matrix_csv(OUT_FBX / "delta_raw.csv", delta_fbx_raw, header="dx,dy,dz")
        print(f"  → saved FBX delta_raw.csv")

        # Сглаживание на FBX теми же параметрами
        if smooth_iters > 0:
            delta_fbx = smooth_delta(delta_fbx_raw, faces_fbx,
                                      n_iter=smooth_iters,
                                      alpha=smooth_alpha)
            print(f"  max ||δ_fbx (smoothed)|| = {np.linalg.norm(delta_fbx, axis=1).max():.4f}")
        else:
            delta_fbx = delta_fbx_raw

        head_fbx_def = verts_fbx + delta_fbx
        col_fbx = to_colors(np.linalg.norm(delta_fbx, axis=1), CMAP_DISP)

        save_matrix_csv(OUT_FBX / "delta_smoothed.csv", delta_fbx, header="dx,dy,dz")
        save_matrix_csv(OUT_FBX / "verts_deformed.csv", head_fbx_def, header="x,y,z")
        print(f"  → saved FBX delta_smoothed.csv & verts_deformed.csv")

        # Обновляем metadata добавляя FBX anchors
        save_metadata_json(OUT_DIR / "metadata.json", params, shape, expr,
                            N_anchors, src, src_fbx=src_fbx)

        # ОКНО 6: HEAD 1 (native blendshape) | FBX rest | FBX deformed
        print("\n  ОКНО 6: HEAD 1 (native) | FBX rest | FBX deformed (Q → выход)")
        show_meshes_side_by_side([
            (head_expr, faces, col_native, "head1_native"),
            (verts_fbx, faces_fbx, np.tile([0.85, 0.75, 0.68], (len(verts_fbx), 1)),
             "fbx_rest"),
            (head_fbx_def, faces_fbx, col_fbx, "fbx_deformed"),
        ], window_title="ОКНО 6: HEAD 1 (native) | FBX rest | FBX deformed")

    print("\n✓ Pipeline завершён.")
    print(f"  Все данные сохранены в: {OUT_DIR}/")


if __name__ == "__main__":
    main()
