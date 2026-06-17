"""
compare_heads.py — Сравнение двух голов (FLAME ↔ FBX) через HKS/WKS сигнатуры.

Standalone-тулза. Не требует pipeline'а, anchor'ов и т.д. — просто берёт
два меша и показывает насколько они похожи в spectral signature space.

Использование:
    python python/scripts/motion_groups_v4/compare_heads.py \
        --flame "Muscle-autoskinner/Assets/Meshes/FLAME/.../flame2023.pkl" \
        --fbx "Muscle-autoskinner/Assets/Meshes/Reference/stylized_female_head_solid.fbx" \
        [--sig hks|wks]  [--k-clusters 15]  [--n-eigs 100]  [--n-scales 20]

Output (в python/scripts/debug_output/compare_<TS>/):
    head1_signatures.csv, head1_labels.csv, head1_similarity.csv
    fbx_signatures.csv,   fbx_labels.csv,   fbx_similarity.csv
    cluster_match.csv      — sопоставление кластеров FBX↔HEAD1 + средние centroid-расстояния
    summary.json           — главные метрики
    Plus Open3D окно с 5 мешами для визуального сравнения.

5 мешей в окне:
  1. HEAD 1 own K-means
  2. FBX via HEAD1 centroids (cross-mesh transfer signatures)
  3. FBX own K-means (matched palette через Hungarian)
  4. HEAD 1 SIMILARITY (per-vertex distance to nearest FBX signature, green=ок/red=плохо)
  5. FBX SIMILARITY (per-vertex distance to nearest HEAD1 signature)
"""

from __future__ import annotations
import argparse, json, pickle, subprocess, sys, tempfile, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import datetime as _dt


# ── MESH I/O ─────────────────────────────────────────────────────────────────

def _chumpy_shim():
    import sys, types
    if "chumpy" in sys.modules: return
    chumpy = types.ModuleType("chumpy")
    class Ch:
        def __init__(self, *a, **kw): pass
        def __setstate__(self, state):
            if isinstance(state, dict):
                self.r = np.array(state.get("x", state.get("_x", None)))
            else: self.r = np.array(state)
        def __reduce__(self): return (Ch, ())
    chumpy.Ch = Ch; chumpy.ch = Ch
    sys.modules["chumpy"] = chumpy; sys.modules["chumpy.ch"] = chumpy


def load_flame(path):
    _chumpy_shim()
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    def to_np(x):
        if hasattr(x, "r"): return np.array(x.r)
        if hasattr(x, "toarray"): return x.toarray()
        return np.array(x)
    return (to_np(d["v_template"]).astype(np.float64),
            to_np(d["f"]).astype(np.int64))


def load_fbx(path):
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


# ── LAPLACE-BELTRAMI + SIGNATURES ────────────────────────────────────────────

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
    return L, sp.diags(areas.clip(min=1e-12))


def compute_spectrum(L, M, k=100):
    print(f"  Eigendecomp (k={k}, N={L.shape[0]})...", end=" ", flush=True)
    t0 = time.time()
    try:
        ev, ef = spla.eigsh(L, k=min(k, L.shape[0] - 2), M=M, sigma=-1e-6, which='LM')
    except Exception:
        ev, ef = spla.eigsh(L, k=min(k, L.shape[0] - 2), M=M, which='SM')
    order = np.argsort(ev)
    ev = np.clip(ev[order], 0, None); ef = ef[:, order]
    print(f"done in {time.time()-t0:.1f}s, λ range [{ev[0]:.3g}, {ev[-1]:.3g}]")
    return ev, ef


def compute_hks(eigvals, eigvecs, n_scales=20):
    lam_min = max(float(eigvals[1]), 1e-6)
    lam_max = float(eigvals[-1])
    t_min = 4 * np.log(10) / lam_max
    t_max = 4 * np.log(10) / lam_min
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_scales)
    decay = np.exp(-np.outer(times, eigvals))                # (T, K)
    hks = (eigvecs ** 2) @ decay.T                            # (N, T)
    heat_trace = decay.sum(axis=1).clip(min=1e-12)            # scale-invariant
    return hks / heat_trace[None, :]


def compute_wks(eigvals, eigvecs, n_scales=20):
    log_lam = np.log(eigvals.clip(min=1e-9))
    e_min, e_max = log_lam[1], log_lam[-1]
    energies = np.linspace(e_min, e_max, n_scales)
    sigma = 7 * (e_max - e_min) / n_scales
    weights = np.exp(-((log_lam[None, :] - energies[:, None]) ** 2)
                      / (2.0 * sigma * sigma))                # (E, K)
    C = weights.sum(axis=1).clip(min=1e-12)
    wks = ((eigvecs ** 2) @ weights.T) / C[None, :]
    return wks


# ── UTIL ─────────────────────────────────────────────────────────────────────

def make_cluster_palette(n):
    import colorsys
    return np.array([colorsys.hsv_to_rgb(i / n, 0.7, 0.95) for i in range(n)])


def save_csv(path: Path, array, header):
    np.savetxt(path, np.asarray(array), delimiter=",", header=header,
                comments="", fmt="%.6e" if np.issubdtype(np.asarray(array).dtype,
                                                          np.floating) else "%d")


def show_meshes_side_by_side(meshes, gap=1.3, window_title="Compare heads"):
    """meshes: list of (verts, faces, colors, label)"""
    try:
        import open3d as o3d
    except ImportError:
        print("  ⚠ open3d не установлен")
        return

    geoms = []
    x_cursor = 0.0
    for v, f, colors, label in meshes:
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
        v_c = v - v.mean(0)
        width = v_c.max(0)[0] - v_c.min(0)[0]
        v_s = v_c.copy(); v_s[:, 0] += x_cursor + width / 2
        x_cursor += width * gap

        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v_s),
                                       o3d.utility.Vector3iVector(f))
        m.compute_vertex_normals()
        m.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        geoms.append(m)
        print(f"    [{label}]")

    print(f"\n  >>> Открываю окно '{window_title}'  (Q→выход) <<<")
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name=window_title, width=1800, height=900):
        print("  ⚠ Не удалось создать окно"); return
    for g in geoms: vis.add_geometry(g)
    vis.get_render_option().mesh_show_back_face = True
    vis.poll_events(); vis.update_renderer()
    vis.run(); vis.destroy_window()
    print(f"  → окно закрыто")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flame", type=str,
                     default="Muscle-autoskinner/Assets/Meshes/FLAME/FLAME2023 nonommercial use/flame2023.pkl",
                     help="путь к FLAME .pkl")
    ap.add_argument("--fbx", type=str, required=True,
                     help="путь к FBX (или OBJ/STL — assimp пропустит)")
    ap.add_argument("--sig", choices=["hks", "wks", "combined"], default="hks")
    ap.add_argument("--k-clusters", type=int, default=15)
    ap.add_argument("--n-eigs", type=int, default=100)
    ap.add_argument("--n-scales", type=int, default=20)
    ap.add_argument("--no-viz", action="store_true",
                     help="без 3D окна — только CSV+JSON")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(
        f"python/scripts/debug_output/compare_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"● Output: {out_dir}/")

    # ── Load meshes ──────────────────────────────────────────────────────────
    print(f"\n● Loading meshes")
    v_h1_orig,  f_h1  = load_flame(args.flame)
    print(f"  FLAME: {len(v_h1_orig)} verts, {len(f_h1)} faces")
    v_fbx_orig, f_fbx = load_fbx(args.fbx)
    print(f"  FBX:   {len(v_fbx_orig)} verts, {len(f_fbx)} faces")

    # Нормализуем bbox обоих — критично! Без этого HKS/WKS считаются на
    # сетках разного абсолютного масштаба и сигнатуры не сравнимы.
    def normalize_bbox(v):
        c = v.mean(0)
        v_c = v - c
        scale = np.linalg.norm(v_c.max(0) - v_c.min(0)) + 1e-12
        return (v_c / scale, c, scale)
    v_h1,  c_h1,  s_h1  = normalize_bbox(v_h1_orig)
    v_fbx, c_fbx, s_fbx = normalize_bbox(v_fbx_orig)
    print(f"  Bbox-нормализация: FLAME scale={s_h1:.4f}, FBX scale={s_fbx:.4f}  "
          f"(ratio {s_fbx/s_h1:.2f}×)")

    # ── Spectrum + Signatures ────────────────────────────────────────────────
    print(f"\n● Computing operators + spectrum + {args.sig.upper()}")
    print(f"  HEAD 1:")
    L_h1, M_h1 = build_operators(v_h1, f_h1)
    ev_h1, ef_h1 = compute_spectrum(L_h1, M_h1, k=args.n_eigs)
    def _build_sig(ev, ef, kind, n_scales):
        if kind == "hks":
            return compute_hks(ev, ef, n_scales=n_scales)
        elif kind == "wks":
            return compute_wks(ev, ef, n_scales=n_scales)
        elif kind == "combined":
            h = compute_hks(ev, ef, n_scales=n_scales)
            w = compute_wks(ev, ef, n_scales=n_scales)
            hn = h / np.linalg.norm(h, axis=1, keepdims=True).clip(min=1e-12)
            wn = w / np.linalg.norm(w, axis=1, keepdims=True).clip(min=1e-12)
            return np.concatenate([hn, wn], axis=1)
        raise ValueError(kind)

    sig_h1 = _build_sig(ev_h1, ef_h1, args.sig, args.n_scales)
    print(f"  HEAD 1 {args.sig.upper()} shape: {sig_h1.shape}")

    print(f"  FBX:")
    L_fbx, M_fbx = build_operators(v_fbx, f_fbx)
    ev_fbx, ef_fbx = compute_spectrum(L_fbx, M_fbx, k=args.n_eigs)
    sig_fbx = _build_sig(ev_fbx, ef_fbx, args.sig, args.n_scales)
    print(f"  FBX    {args.sig.upper()} shape: {sig_fbx.shape}")

    # L2-нормируем signature-вектора (cosine-like sравнение)
    sig_h1_n  = sig_h1  / np.linalg.norm(sig_h1,  axis=1, keepdims=True).clip(min=1e-12)
    sig_fbx_n = sig_fbx / np.linalg.norm(sig_fbx, axis=1, keepdims=True).clip(min=1e-12)

    # ── Clustering ───────────────────────────────────────────────────────────
    from sklearn.cluster import KMeans
    print(f"\n● K-means clustering (K={args.k_clusters})")
    print(f"  HEAD 1...", end=" ", flush=True); t0 = time.time()
    km_h1 = KMeans(n_clusters=args.k_clusters, n_init=10, random_state=42)
    lab_h1 = km_h1.fit_predict(sig_h1_n)
    print(f"done in {time.time()-t0:.1f}s")
    print(f"  FBX...   ", end=" ", flush=True); t0 = time.time()
    km_fbx = KMeans(n_clusters=args.k_clusters, n_init=10, random_state=42)
    lab_fbx_own = km_fbx.fit_predict(sig_fbx_n)
    print(f"done in {time.time()-t0:.1f}s")

    # FBX через HEAD1 centroids
    centroids = km_h1.cluster_centers_
    a_sq = (sig_fbx_n ** 2).sum(1, keepdims=True)
    b_sq = (centroids ** 2).sum(1, keepdims=True).T
    cross = sig_fbx_n @ centroids.T
    D2 = np.maximum(a_sq + b_sq - 2 * cross, 0)
    lab_fbx_xfer = np.argmin(D2, axis=1)

    # Hungarian matching FBX-centroids ↔ HEAD1-centroids
    cost = np.linalg.norm(km_fbx.cluster_centers_[:, None, :]
                          - centroids[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        remap = dict(zip(row_ind, col_ind))
    except Exception:
        remap = {i: int(np.argmin(cost[i])) for i in range(args.k_clusters)}
    lab_fbx_own_remapped = np.array([remap[int(l)] for l in lab_fbx_own])
    avg_match_dist = cost[list(remap.keys()), list(remap.values())].mean()

    # ── Similarity maps ──────────────────────────────────────────────────────
    print(f"\n● Computing pairwise signature similarity")
    # FBX vertex → nearest HEAD1 vertex
    a_sq = (sig_fbx_n ** 2).sum(1, keepdims=True)
    b_sq = (sig_h1_n ** 2).sum(1, keepdims=True).T
    cross = sig_fbx_n @ sig_h1_n.T
    D2_full = np.maximum(a_sq + b_sq - 2 * cross, 0)
    nn_fbx = np.sqrt(D2_full.min(axis=1))                    # (N_fbx,)
    nn_h1  = np.sqrt(D2_full.min(axis=0))                    # (N_h1,)

    # Summary
    summary = {
        "flame_path": args.flame,
        "fbx_path": args.fbx,
        "signature": args.sig,
        "n_eigs": args.n_eigs,
        "n_scales": args.n_scales,
        "k_clusters": args.k_clusters,
        "n_head1_verts": int(len(v_h1)),
        "n_fbx_verts": int(len(v_fbx)),
        "cluster_centroid_distance_mean": float(avg_match_dist),
        "head1_to_fbx_nn_mean":    float(nn_h1.mean()),
        "head1_to_fbx_nn_median":  float(np.median(nn_h1)),
        "head1_to_fbx_nn_max":     float(nn_h1.max()),
        "fbx_to_head1_nn_mean":    float(nn_fbx.mean()),
        "fbx_to_head1_nn_median":  float(np.median(nn_fbx)),
        "fbx_to_head1_nn_max":     float(nn_fbx.max()),
    }
    print(f"\n── РЕЗУЛЬТАТЫ ──")
    for k, v in summary.items():
        if isinstance(v, float): print(f"  {k:35s} = {v:.4f}")
        elif isinstance(v, int):  print(f"  {k:35s} = {v}")

    quality = ("ОТЛИЧНО ✓" if summary["fbx_to_head1_nn_mean"] < 0.05 else
                "ХОРОШО"   if summary["fbx_to_head1_nn_mean"] < 0.15 else
                "СРЕДНЕ"   if summary["fbx_to_head1_nn_mean"] < 0.30 else
                "ПЛОХО ✗")
    print(f"\n  >>> Качество соответствия сигнатур: {quality} <<<")

    # ── Save CSV/JSON ────────────────────────────────────────────────────────
    print(f"\n● Saving dumps to {out_dir}/")
    sig_header = ",".join([f"s{i}" for i in range(args.n_scales)])
    save_csv(out_dir / "head1_signatures.csv", sig_h1, sig_header)
    save_csv(out_dir / "fbx_signatures.csv",   sig_fbx, sig_header)
    save_csv(out_dir / "head1_labels.csv",     lab_h1[:, None], "cluster_id")
    save_csv(out_dir / "fbx_labels_xfer.csv",  lab_fbx_xfer[:, None],
              "cluster_id_via_head1_centroids")
    save_csv(out_dir / "fbx_labels_own.csv",   lab_fbx_own_remapped[:, None],
              "cluster_id_own_remapped")
    save_csv(out_dir / "head1_similarity.csv", nn_h1[:, None],
              "nn_distance_to_fbx_signatures")
    save_csv(out_dir / "fbx_similarity.csv",   nn_fbx[:, None],
              "nn_distance_to_head1_signatures")
    save_csv(out_dir / "cluster_match.csv",
              np.array([[k, v, float(cost[k, v])] for k, v in remap.items()]),
              "fbx_cluster,head1_cluster,centroid_distance")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  ✓ 7 CSV + summary.json")

    # ── Visualization ────────────────────────────────────────────────────────
    if not args.no_viz:
        palette = make_cluster_palette(args.k_clusters)
        col_h1       = palette[lab_h1]
        col_fbx_xfer = palette[lab_fbx_xfer]
        col_fbx_own  = palette[lab_fbx_own_remapped]

        # similarity colors
        shared_max = max(np.percentile(nn_h1, 95), np.percentile(nn_fbx, 95))
        def sim_col(d):
            t = np.clip(d / max(shared_max, 1e-9), 0, 1)
            c = np.zeros((len(d), 3))
            c[:, 0] = t                                            # R
            c[:, 1] = np.clip(1.5 * (1 - t), 0, 1)                  # G
            c[:, 2] = 0.1
            return c
        col_h1_sim  = sim_col(nn_h1)
        col_fbx_sim = sim_col(nn_fbx)

        # Для визуализации используем оригинальные (не bbox-нормализованные)
        # верты, иначе FBX будет огромным и FLAME крошечным в окне.
        # Но т.к. show_meshes_side_by_side сам центрирует и горизонтально
        # раскладывает — нормализованные тоже подойдут (даже лучше).
        show_meshes_side_by_side([
            (v_h1,  f_h1,  col_h1,       f"HEAD1  {args.sig.upper()}  own K-means"),
            (v_fbx, f_fbx, col_fbx_xfer, f"FBX    {args.sig.upper()}  via HEAD1 centroids"),
            (v_fbx, f_fbx, col_fbx_own,  f"FBX    {args.sig.upper()}  own K-means (matched)"),
            (v_h1,  f_h1,  col_h1_sim,
             f"HEAD1 SIMILARITY (mean={summary['head1_to_fbx_nn_mean']:.3f})"),
            (v_fbx, f_fbx, col_fbx_sim,
             f"FBX SIMILARITY   (mean={summary['fbx_to_head1_nn_mean']:.3f})"),
        ], gap=1.25,
           window_title=f"Compare heads — {args.sig.upper()} signatures  "
                        f"({quality})")

    print(f"\n✓ Готово. Всё в: {out_dir}/")


if __name__ == "__main__":
    main()
