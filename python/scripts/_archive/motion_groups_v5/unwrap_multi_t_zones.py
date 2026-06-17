"""
unwrap_multi_t_zones.py — Polar UV unwrap для multi-t hard-partition зон.

Алгоритм:
1. Берём heat (multi-t enriched или обычный single-t) из run-output
2. Hard partition вершин по argmax → каждая вершина в одной anchor-зоне
3. Для каждой зоны:
   - Строим tangent frame в anchor'е (через vertex normal)
   - Polar coords:
       r = 1 - heat[a, v] / max(heat[a])   ← 0 в anchor'е, 1 на границе
       θ = atan2(local_y, local_x)         ← угол в tangent plane
4. Atlas packing: K disks в [0,1]²
5. Сохраняем mesh.obj + UV CSV + preview PNG + 3D окно

Использование:
    python python/scripts/motion_groups_v5/unwrap_multi_t_zones.py
        → берёт самый свежий run

    python python/scripts/motion_groups_v5/unwrap_multi_t_zones.py \
        --run <path> --use-multi-t --atlas-mode grid
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latest_run(base="python/scripts/debug_output"):
    runs = sorted(Path(base).glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"Нет run_* в {base}/")
    return runs[-1]


def load_csv(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def compute_vertex_normals(verts, faces):
    """Усреднённые normals по face-normals."""
    N = len(verts)
    normals = np.zeros((N, 3))
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True).clip(min=1e-12)
    # Аккумулируем на 3 вершины каждого треугольника
    for i in range(3):
        np.add.at(normals, faces[:, i], face_normals)
    norms = np.linalg.norm(normals, axis=1, keepdims=True).clip(min=1e-12)
    return normals / norms


def build_tangent_frame(normal):
    """Построить (t1, t2) ортонормированный базис в касательной плоскости."""
    n = normal / max(np.linalg.norm(normal), 1e-12)
    # Выбираем не-параллельный вектор
    if abs(n[0]) < 0.9:
        tmp = np.array([1.0, 0.0, 0.0])
    else:
        tmp = np.array([0.0, 1.0, 0.0])
    t1 = np.cross(n, tmp)
    t1 /= max(np.linalg.norm(t1), 1e-12)
    t2 = np.cross(n, t1)
    t2 /= max(np.linalg.norm(t2), 1e-12)
    return t1, t2


def compute_polar_uv_per_zone(verts, heat_per_anchor, partition,
                                 anchor_indices, vert_normals):
    """Polar UV для каждой зоны: r = 1 - heat_norm, θ = direction angle.

    Returns:
      uv_local: (N, 2) — UV в локальной zone-disk system [-1,1]²
                          UV в зоне = polar (r, θ); вершины вне зон → (0,0)
      zone_id:  (N,)   — id зоны для каждой вершины (-1 если unassigned)
    """
    N = len(verts)
    K = heat_per_anchor.shape[0]
    uv_local = np.zeros((N, 2))
    zone_id = np.full(N, -1, dtype=np.int64)

    for a in range(K):
        zone_mask = (partition == a)
        if zone_mask.sum() < 3: continue

        anchor_pos = verts[anchor_indices[a]]
        n = vert_normals[anchor_indices[a]]
        t1, t2 = build_tangent_frame(n)

        # Heat в зоне
        zone_verts_idx = np.where(zone_mask)[0]
        h_zone = heat_per_anchor[a, zone_verts_idx]
        h_max = max(h_zone.max(), 1e-12)

        # Polar coords
        offsets = verts[zone_verts_idx] - anchor_pos
        x_local = offsets @ t1
        y_local = offsets @ t2
        theta = np.arctan2(y_local, x_local)
        r = 1.0 - h_zone / h_max               # 0 в anchor'е, 1 на границе

        uv_local[zone_verts_idx, 0] = r * np.cos(theta)
        uv_local[zone_verts_idx, 1] = r * np.sin(theta)
        zone_id[zone_verts_idx] = a

    return uv_local, zone_id


def compute_uv_at(v_idx, zone, verts, heat_per_anchor, anchor_indices,
                   vert_normals, zone_heat_max):
    """Polar UV для произвольной вершины v в системе координат zone'ы a.
    Используется для boundary-вершин которые ФИЗИЧЕСКИ принадлежат другой зоне
    но рисуются на границе этой зоны (через UV-seam)."""
    anchor_pos = verts[anchor_indices[zone]]
    n = vert_normals[anchor_indices[zone]]
    t1, t2 = build_tangent_frame(n)
    offset = verts[v_idx] - anchor_pos
    x_local = offset @ t1
    y_local = offset @ t2
    theta = np.arctan2(y_local, x_local)
    # Heat в этой зоне для этой вершины (может быть малое, т.к. вершина не в зоне)
    h = max(float(heat_per_anchor[zone, v_idx]), 0.0)
    r = 1.0 - h / max(zone_heat_max[zone], 1e-12)
    r = min(max(r, 0.0), 1.0)            # clamp в [0, 1]
    return (r * np.cos(theta), r * np.sin(theta))


def determine_face_zones(faces, partition, heat_per_anchor):
    """Для каждой face определяем «owning zone» через majority vote vertex zones.
    При tie — выбираем zone с большим суммарным heat на face-вершинах.

    Returns: (N_faces,) с zone_id (-1 если все 3 vertex unassigned)
    """
    from collections import Counter
    F = len(faces)
    face_zones = np.full(F, -1, dtype=np.int64)
    for fi in range(F):
        f = faces[fi]
        zs = [int(partition[v]) for v in f if partition[v] >= 0]
        if not zs:
            face_zones[fi] = -1
            continue
        cnt = Counter(zs)
        most_common = cnt.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            face_zones[fi] = most_common[0][0]
        else:
            # Tie: max sum of heat среди tied zones
            tied = [z for z, c in most_common if c == most_common[0][1]]
            best_z, best_score = tied[0], -1.0
            for z in tied:
                score = sum(float(heat_per_anchor[z, int(v)]) for v in f)
                if score > best_score:
                    best_score = score; best_z = z
            face_zones[fi] = best_z
    return face_zones


def relax_uv_islands(vt_local, vt_zones, face_vt, vt_to_vertex,
                       faces, partition, anchor_indices, verts,
                       n_iter=20, alpha=0.5, mode='anchor_only',
                       verbose=True):
    """Relaxation UV islands. Несколько режимов:

    mode='anchor_only' (DEFAULT):
        Fixed: только anchor vt каждой зоны (центр диска, r=0) + trash
        Free: всё остальное включая outer boundary
        Метод: Laplacian smoothing.
        → Boundary свободно расходится в естественную форму, не растягивая
          швы (главная проблема fixed_boundary режима).

    mode='fixed_boundary':
        Fixed: vts соответствующие zone-boundary mesh-вершинам + trash
        Free: только interior
        Метод: Laplacian smoothing.
        → Швы остаются на месте; interior разглаживается. Но при
          неравномерной 3D-геометрии boundary может быть растянут.

    mode='springs':
        Fixed: только anchor vt + trash
        Метод: pull-spring — edges тянут друг к другу с rest_length
        пропорциональным 3D-длине ребра.
        → UV-длины пропорциональны 3D-длинам → метрические искажения малы.

    Returns: relaxed vt_local (M, 2)
    """
    M = len(vt_local)
    N = len(partition)

    # vt → vt adjacency через face edges
    adj_sets = [set() for _ in range(M)]
    for fvt in face_vt:
        ta, tb, tc = fvt
        for u, v in [(ta, tb), (tb, tc), (tc, ta)]:
            if u != v:
                adj_sets[u].add(v); adj_sets[v].add(u)
    adj = [list(s) for s in adj_sets]

    # is_fixed: зависит от mode
    is_fixed = np.zeros(M, dtype=bool)

    if mode == 'fixed_boundary':
        # mesh adjacency для определения zone-boundary вершин
        mesh_adj = [set() for _ in range(N)]
        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            mesh_adj[a].update((b, c)); mesh_adj[b].update((a, c))
            mesh_adj[c].update((a, b))
        is_zone_boundary_v = np.zeros(N, dtype=bool)
        for v in range(N):
            z = int(partition[v])
            if z < 0:
                is_zone_boundary_v[v] = True; continue
            for n in mesh_adj[v]:
                if int(partition[n]) != z:
                    is_zone_boundary_v[v] = True; break
        for i in range(M):
            if vt_zones[i] < 0: is_fixed[i] = True; continue
            mv = int(vt_to_vertex[i])
            if mv < 0: is_fixed[i] = True; continue
            if is_zone_boundary_v[mv]:
                is_fixed[i] = True
    else:  # 'anchor_only' or 'springs'
        # Fix только anchor vt каждой зоны + trash
        anchor_set = set(int(a) for a in anchor_indices)
        for i in range(M):
            if vt_zones[i] < 0:
                is_fixed[i] = True; continue
            mv = int(vt_to_vertex[i])
            if mv < 0: is_fixed[i] = True; continue
            # Проверяем что эта vt — anchor для своей зоны
            z = int(vt_zones[i])
            if z < len(anchor_indices) and mv == int(anchor_indices[z]):
                is_fixed[i] = True

    n_fixed = int(is_fixed.sum())
    n_free = M - n_fixed
    if verbose:
        print(f"    mode={mode}: {n_fixed} fixed + {n_free} free vts")
        if n_free == 0:
            print(f"    ⚠ Нечего relax'ить, skip")
            return vt_local.copy()

    uv = vt_local.copy()

    if mode == 'springs':
        # Pre-compute rest lengths по 3D + adjacency
        edges_unique = set()
        for fvt in face_vt:
            ta, tb, tc = fvt
            for u, v in [(ta, tb), (tb, tc), (tc, ta)]:
                if u != v:
                    edges_unique.add((min(u, v), max(u, v)))
        edges_list = list(edges_unique)
        rest_lengths = np.zeros(len(edges_list))
        for ei, (u, v) in enumerate(edges_list):
            mu, mv_ = int(vt_to_vertex[u]), int(vt_to_vertex[v])
            if mu >= 0 and mv_ >= 0:
                rest_lengths[ei] = float(np.linalg.norm(verts[mu] - verts[mv_]))
        valid = rest_lengths > 0
        if valid.any():
            # Нормируем чтобы avg 3D edge соответствовал avg current UV edge.
            # Это даёт корректные target длины не разнося UV.
            avg_3d = rest_lengths[valid].mean()
            avg_uv = float(np.linalg.norm(
                uv[np.array([u for u,_ in edges_list])] -
                uv[np.array([v for _,v in edges_list])],
                axis=1).mean())
            scale_factor = avg_uv / max(avg_3d, 1e-12)
            rest_lengths *= scale_factor

        # Vertex degree (для нормировки force per vertex)
        degrees = np.zeros(M, dtype=np.int64)
        for u, v in edges_list:
            degrees[u] += 1; degrees[v] += 1
        degrees = np.maximum(degrees, 1)

        # Очень маленький step + clamp для стабильности
        step = alpha * 0.1
        for it in range(n_iter):
            force = np.zeros_like(uv)
            for ei, (u, v) in enumerate(edges_list):
                if rest_lengths[ei] <= 0: continue
                d = uv[v] - uv[u]
                dist = float(np.linalg.norm(d))
                if dist < 1e-9: continue
                # spring force: PROPORTIONAL to delta_length, capped
                delta = dist - rest_lengths[ei]
                # tanh-clip force to prevent runaway
                f_mag = np.tanh(delta / max(rest_lengths[ei], 1e-9)) * 0.5
                f = f_mag * (d / dist)
                force[u] += f
                force[v] -= f
            # Average force per vertex by degree
            force /= degrees[:, None]
            # Apply
            new_uv = uv + step * force
            new_uv[is_fixed] = uv[is_fixed]
            # Clamp UV в [-3, 3] чтоб не улетало в космос
            new_uv = np.clip(new_uv, -3.0, 3.0)
            if verbose and (it == 0 or it == n_iter - 1):
                disp = float(np.linalg.norm(new_uv - uv, axis=1).mean())
                max_uv = float(np.abs(new_uv).max())
                print(f"    iter {it+1}/{n_iter}: mean disp = {disp:.6f}, "
                      f"max |UV| = {max_uv:.3f}")
            uv = new_uv
    else:
        # Laplacian (fixed_boundary or anchor_only)
        for it in range(n_iter):
            new_uv = uv.copy()
            for i in range(M):
                if is_fixed[i]: continue
                if not adj[i]: continue
                avg = uv[adj[i]].mean(axis=0)
                new_uv[i] = (1 - alpha) * uv[i] + alpha * avg
            if verbose and (it == 0 or it == n_iter - 1):
                disp = float(np.linalg.norm(new_uv - uv, axis=1).mean())
                print(f"    iter {it+1}/{n_iter}: mean disp = {disp:.6f}")
            uv = new_uv

    return uv


def build_islands_with_seams(verts, faces, partition, heat_per_anchor,
                               anchor_indices, vert_normals):
    """Строим UV-острова с швами по краям heat-зон.

    На границе зон вершина появляется НЕСКОЛЬКО раз как разные `vt` индексы —
    по одному на каждую зону которую она touches через faces.

    Returns:
      vt_list:    list of (uv_x, uv_y) в локальных zone-disk coords
      vt_zones:   list of int (zone_id для каждого vt entry — для atlas packing)
      face_vt:    list of (vt_a, vt_b, vt_c) — индексы vt для каждой face
      face_zones: list of zone_id per face (-1 = unassigned/skipped)
    """
    K = heat_per_anchor.shape[0]

    # Per-zone max heat (для radial coord нормализации)
    zone_heat_max = np.zeros(K)
    for a in range(K):
        zm = (partition == a)
        if zm.any():
            zone_heat_max[a] = float(heat_per_anchor[a, zm].max())
        else:
            zone_heat_max[a] = 1e-12

    # Определяем зону каждой face
    face_zones = determine_face_zones(faces, partition, heat_per_anchor)

    # (v, zone) → vt_idx
    vt_map = {}
    vt_list = []
    vt_zones = []
    vt_to_vertex = []      # для каждого vt — соответствующий mesh vertex (-1 если trash)
    face_vt = []

    for fi in range(len(faces)):
        f = faces[fi]
        z = int(face_zones[fi])
        if z < 0:
            # Все 3 vertex unassigned — face идёт в "trash" zone
            trash_key = ('trash',)
            if trash_key not in vt_map:
                vt_list.append((0.0, 0.0))
                vt_zones.append(-1)
                vt_to_vertex.append(-1)
                vt_map[trash_key] = len(vt_list) - 1
            ti = vt_map[trash_key]
            face_vt.append((ti, ti, ti))
            continue

        face_vts = []
        for vi in f:
            vi = int(vi)
            key = (vi, z)
            if key not in vt_map:
                uv = compute_uv_at(vi, z, verts, heat_per_anchor,
                                    anchor_indices, vert_normals, zone_heat_max)
                vt_list.append(uv)
                vt_zones.append(z)
                vt_to_vertex.append(vi)
                vt_map[key] = len(vt_list) - 1
            face_vts.append(vt_map[key])
        face_vt.append(tuple(face_vts))

    return (np.array(vt_list, dtype=np.float64),
            np.array(vt_zones, dtype=np.int64),
            face_vt, face_zones,
            np.array(vt_to_vertex, dtype=np.int64))


def atlas_pack_grid(vt_local, vt_zones, K):
    """Атлас: K disks в grid layout [0,1]².

    vt_local: (M, 2) — локальные polar UV per vt entry
    vt_zones: (M,)   — zone_id per vt entry (-1 = trash)
    """
    M = len(vt_local)
    cols = int(np.ceil(np.sqrt(K)))
    rows = int(np.ceil(K / cols))
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    disk_r = min(cell_w, cell_h) * 0.42

    final_uv = np.zeros((M, 2), dtype=np.float64)
    for i in range(M):
        z = int(vt_zones[i])
        if z < 0:
            final_uv[i] = (0.99, 0.99)
            continue
        col = z % cols
        row = z // cols
        cx = (col + 0.5) * cell_w
        cy = (row + 0.5) * cell_h
        final_uv[i] = (cx + disk_r * vt_local[i, 0],
                        cy + disk_r * vt_local[i, 1])
    return final_uv


def atlas_pack_circular(vt_local, vt_zones, K):
    """Атлас: 1 центральный disk + (K-1) по кругу."""
    M = len(vt_local)
    if K <= 1:
        final_uv = vt_local.copy() * 0.45 + np.array([0.5, 0.5])
        final_uv[vt_zones < 0] = (0.99, 0.99)
        return final_uv

    final_uv = np.zeros((M, 2), dtype=np.float64)
    n_outer = K - 1
    R_outer = 0.32
    r_center = 0.18
    r_outer = 0.12

    for i in range(M):
        z = int(vt_zones[i])
        if z < 0:
            final_uv[i] = (0.99, 0.99); continue
        if z == 0:
            cx, cy, rd = 0.5, 0.5, r_center
        else:
            phi = 2 * np.pi * (z - 1) / n_outer
            cx = 0.5 + R_outer * np.cos(phi)
            cy = 0.5 + R_outer * np.sin(phi)
            rd = r_outer
        final_uv[i] = (cx + rd * vt_local[i, 0],
                        cy + rd * vt_local[i, 1])
    return final_uv


def save_obj_with_islands(path, verts, faces, vt_uvs, face_vt, vt_zones=None,
                            zone_id=None):
    """Сохраняем OBJ с UV-островами (швы по границам heat-зон).

    verts:    (N, 3)
    faces:    (F, 3)  — vertex indices
    vt_uvs:   (M, 2)  — UV coords per vt entry (M может быть > N из-за seam'ов)
    face_vt:  list of (vt_a, vt_b, vt_c)  — vt index per face vertex
    vt_zones: (M,)    — zone per vt entry (для комментария)
    zone_id:  (N,)    — zone per vertex (для комментария)
    """
    N = len(verts); M = len(vt_uvs); F = len(faces)
    with open(path, "w") as f:
        f.write(f"# Unwrapped mesh with multi-t zone UV islands (seams at zone borders)\n")
        f.write(f"# {N} vertices, {M} vt entries, {F} faces\n\n")
        for i, v in enumerate(verts):
            zstr = f"  # zone={zone_id[i]}" if zone_id is not None else ""
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}{zstr}\n")
        for i, uv in enumerate(vt_uvs):
            zstr = f"  # zone={vt_zones[i]}" if vt_zones is not None else ""
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}{zstr}\n")
        for fi, face in enumerate(faces):
            va, vb, vc = int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
            ta, tb, tc = face_vt[fi]
            ta += 1; tb += 1; tc += 1
            f.write(f"f {va}/{ta} {vb}/{tb} {vc}/{tc}\n")


def make_palette(K):
    import colorsys
    return np.array([colorsys.hsv_to_rgb(i / K, 0.7, 0.95) for i in range(K)])


def plot_uv_preview(uvs, zone_id, K, out_path, title="UV atlas",
                     faces=None, face_vt=None, face_zones=None):
    """Plot UV scatter + (опц.) edges из faces для visual seams."""
    palette = make_palette(K)
    colors = np.zeros((len(uvs), 3))
    for v in range(len(uvs)):
        z = int(zone_id[v]) if zone_id is not None else 0
        if z >= 0:
            colors[v] = palette[z % K]
        else:
            colors[v] = [0.6, 0.6, 0.6]

    fig, ax = plt.subplots(figsize=(9, 9))
    # Edges: для каждой face рисуем 3 ребра между её vt indices
    if faces is not None and face_vt is not None:
        for fi, vts in enumerate(face_vt):
            ta, tb, tc = vts
            for u, v in [(ta, tb), (tb, tc), (tc, ta)]:
                ax.plot([uvs[u, 0], uvs[v, 0]], [uvs[u, 1], uvs[v, 1]],
                         color='black', linewidth=0.15, alpha=0.4)
    ax.scatter(uvs[:, 0], uvs[:, 1], c=colors, s=4, alpha=0.85, zorder=3)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('u'); ax.set_ylabel('v')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def show_3d_mesh_colored(verts, faces, zone_id, K):
    try:
        import open3d as o3d
    except ImportError:
        print("⚠ open3d не установлен — пропускаю 3D окно")
        return

    palette = make_palette(K)
    colors = np.zeros((len(verts), 3))
    for v in range(len(verts)):
        z = zone_id[v]
        if z >= 0:
            colors[v] = palette[z % K]
        else:
            colors[v] = [0.5, 0.5, 0.5]   # серый = unassigned

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(verts.astype(np.float64)),
        o3d.utility.Vector3iVector(faces.astype(np.int64)))
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    vis = o3d.visualization.Visualizer()
    if vis.create_window("Multi-t zones (3D)  Q→выход", 1200, 800):
        vis.add_geometry(mesh)
        vis.get_render_option().mesh_show_back_face = True
        vis.poll_events(); vis.update_renderer()
        vis.run(); vis.destroy_window()


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                     help="путь к run_*/; default = самый свежий")
    ap.add_argument("--base", type=str, default="python/scripts/debug_output")
    ap.add_argument("--use-multi-t", action="store_true",
                     help="использовать heat_multi_t_enriched.csv (если есть)")
    ap.add_argument("--atlas-mode", choices=["grid", "circular"], default="grid",
                     help="layout atlas'а: grid (default) или circular")
    ap.add_argument("--heat-threshold", type=float, default=0.05,
                     help="порог heat для active partition (по умолчанию 0.05)")
    ap.add_argument("--no-viz", action="store_true",
                     help="без 3D Open3D окна")
    ap.add_argument("--relax-iters", type=int, default=20,
                     help="число итераций relax (0=off, default 20)")
    ap.add_argument("--relax-alpha", type=float, default=0.5,
                     help="α сглаживания (0..1, default 0.5)")
    ap.add_argument("--relax-mode",
                     choices=["anchor_only", "fixed_boundary", "springs"],
                     default="anchor_only",
                     help="режим: anchor_only (default, only center fixed) | "
                          "fixed_boundary (old, may stretch seams) | "
                          "springs (edge-length preserving)")
    args = ap.parse_args()

    run_dir = Path(args.run) if args.run else latest_run(args.base)
    run_dir = run_dir.resolve()
    print(f"● Run: {run_dir}")
    if not run_dir.exists():
        print(f"✗ {run_dir} не существует"); sys.exit(1)

    head1 = run_dir / "head1"
    if not head1.exists():
        print(f"✗ {head1}/ не существует"); sys.exit(1)

    # ── Загрузка mesh данных ──────────────────────────────────────────────────
    print(f"\n● Load mesh data из {head1}/")
    verts = load_csv(head1 / "verts_rest.csv").astype(np.float64)
    faces = load_csv(head1 / "faces.csv").astype(np.int64)
    print(f"  {len(verts)} вершин, {len(faces)} faces")

    # ── Загрузка heat ─────────────────────────────────────────────────────────
    multi_t_path = head1 / "heat_multi_t_enriched.csv"
    if args.use_multi_t and multi_t_path.exists():
        print(f"  Использую multi-t enriched heat: {multi_t_path.name}")
        heat = load_csv(multi_t_path).T                  # (K, N)
    else:
        if args.use_multi_t:
            print(f"  ⚠ {multi_t_path.name} не найден, fallback на heat.csv")
        print(f"  Использую single-t heat: heat.csv")
        heat = load_csv(head1 / "heat.csv").T

    K = heat.shape[0]
    print(f"  heat: {K} anchors × {heat.shape[1]} вершин")

    # ── Anchor indices ────────────────────────────────────────────────────────
    anchor_indices = load_csv(head1 / "anchor_indices.csv").astype(np.int64).flatten()
    print(f"  anchors: {list(anchor_indices)}")

    # ── Hard partition ────────────────────────────────────────────────────────
    print(f"\n● Hard partition (argmax, threshold={args.heat_threshold})")
    heat_norm = heat / heat.max(axis=1, keepdims=True).clip(min=1e-12)
    active = heat_norm.max(axis=0) > args.heat_threshold
    partition = np.where(active, np.argmax(heat_norm, axis=0), -1)
    for a in range(K):
        n_in = int((partition == a).sum())
        print(f"  zone {a}: {n_in} вершин ({100*n_in/len(verts):.1f}%)")
    n_unass = int((partition == -1).sum())
    print(f"  unassigned: {n_unass} ({100*n_unass/len(verts):.1f}%)")

    # ── Vertex normals (для tangent frames) ───────────────────────────────────
    print(f"\n● Compute vertex normals")
    vert_normals = compute_vertex_normals(verts, faces)

    # ── Per-vertex zone UV (для CSV / single-UV preview) ──────────────────────
    print(f"\n● Polar UV per vertex (own zone)")
    uv_local_per_v, zone_per_v = compute_polar_uv_per_zone(
        verts, heat, partition, anchor_indices, vert_normals)

    # ── UV islands с seams по краям зон (face-aware) ──────────────────────────
    print(f"\n● Building UV islands with seams по краям heat-зон")
    vt_local, vt_zones, face_vt, face_zones, vt_to_vertex = build_islands_with_seams(
        verts, faces, partition, heat, anchor_indices, vert_normals)
    print(f"  vt entries: {len(vt_local)} (vs {len(verts)} вершин)")
    print(f"  extra-seam vt entries: {len(vt_local) - len(verts)} "
          f"(boundary vertices duplicated)")
    from collections import Counter
    fz_counts = Counter(int(z) for z in face_zones)
    print(f"  Faces per zone: {dict(sorted(fz_counts.items()))}")

    # Сохраняем pre-relax UV для preview
    vt_local_before = vt_local.copy()

    # ── Relaxation per island ─────────────────────────────────────────────────
    if args.relax_iters > 0:
        print(f"\n● Relaxing UV islands (mode={args.relax_mode}, "
              f"{args.relax_iters} iters, α={args.relax_alpha})")
        vt_local = relax_uv_islands(
            vt_local, vt_zones, face_vt, vt_to_vertex,
            faces, partition, anchor_indices, verts,
            n_iter=args.relax_iters, alpha=args.relax_alpha,
            mode=args.relax_mode,
            verbose=True)
    else:
        print(f"\n● Relaxation skipped (--relax-iters 0)")

    # ── Atlas packing ─────────────────────────────────────────────────────────
    print(f"\n● Atlas packing (mode={args.atlas_mode})")
    if args.atlas_mode == "grid":
        final_vt_uv = atlas_pack_grid(vt_local, vt_zones, K)
    else:
        final_vt_uv = atlas_pack_circular(vt_local, vt_zones, K)
    print(f"  final UV range: u∈[{final_vt_uv[:,0].min():.3f}, "
          f"{final_vt_uv[:,0].max():.3f}], "
          f"v∈[{final_vt_uv[:,1].min():.3f}, {final_vt_uv[:,1].max():.3f}]")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir = run_dir / "uv"
    out_dir.mkdir(exist_ok=True)
    obj_path = out_dir / "mesh_unwrapped.obj"
    print(f"\n● Saving outputs to {out_dir}/")
    save_obj_with_islands(obj_path, verts, faces, final_vt_uv, face_vt,
                            vt_zones=vt_zones, zone_id=zone_per_v)
    print(f"  → {obj_path.name}")

    # CSV: per-vertex (только своя зона)
    import pandas as pd
    df_v = pd.DataFrame({
        "vertex_idx": np.arange(len(verts)),
        "uv_local_x": uv_local_per_v[:, 0],
        "uv_local_y": uv_local_per_v[:, 1],
        "zone_id":    zone_per_v,
    })
    df_v.to_csv(out_dir / "uv_per_vertex.csv", index=False)
    print(f"  → uv_per_vertex.csv ({len(df_v)} entries, по 1 на вершину)")

    # CSV: per-vt (все entries в т.ч. boundary-duplicates)
    df_vt = pd.DataFrame({
        "vt_idx":     np.arange(len(vt_local)),
        "uv_local_x": vt_local[:, 0],
        "uv_local_y": vt_local[:, 1],
        "atlas_u":    final_vt_uv[:, 0],
        "atlas_v":    final_vt_uv[:, 1],
        "zone_id":    vt_zones,
    })
    df_vt.to_csv(out_dir / "uv_per_vt.csv", index=False)
    print(f"  → uv_per_vt.csv ({len(df_vt)} entries — с дубликатами на швах)")

    # Preview PNG c visible wireframe (видно острова!)
    plot_uv_preview(final_vt_uv, vt_zones, K,
                     out_dir / "preview_atlas.png",
                     title=f"UV atlas with seams ({args.atlas_mode}, "
                            f"K={K} islands, {len(vt_local)} vt entries"
                            + (f", relax={args.relax_iters}" if args.relax_iters > 0 else "")
                            + ")",
                     faces=faces, face_vt=face_vt)

    # local preview без atlas packing
    local_normalized = vt_local * 0.5 + 0.5
    plot_uv_preview(local_normalized, vt_zones, K,
                     out_dir / "preview_local.png",
                     title=f"UV local (per-zone polar"
                            + (f", после relax {args.relax_iters} iters" if args.relax_iters > 0 else ", без relax")
                            + ")",
                     faces=faces, face_vt=face_vt)

    # Before/after preview (если relax был)
    if args.relax_iters > 0:
        before_norm = vt_local_before * 0.5 + 0.5
        plot_uv_preview(before_norm, vt_zones, K,
                         out_dir / "preview_before_relax.png",
                         title=f"UV local PRE-RELAX (raw polar)",
                         faces=faces, face_vt=face_vt)
        print(f"  → preview_atlas.png + preview_local.png + preview_before_relax.png")
    else:
        print(f"  → preview_atlas.png + preview_local.png")

    # 3D viz
    if not args.no_viz:
        print(f"\n● 3D viz: mesh раскрашенный по zone_id (Q→выход)")
        show_3d_mesh_colored(verts, faces, zone_id, K)

    print(f"\n✓ Готово. Atlas mesh в {obj_path}")


if __name__ == "__main__":
    main()
