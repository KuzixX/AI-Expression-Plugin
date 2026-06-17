"""
functional_map_fit.py — Классический Functional Maps registration
(Ovsjanikov et al. 2012) + ZoomOut refinement (Melzi et al. 2019).

Задача: найти корреспонденцию вершин между двумя мешами разных топологий
(FLAME source ↔ FBX target) и "натянуть" FBX на FLAME через эту
корреспонденцию.

Pipeline:
  1. Загружаем FLAME (.pkl) и FBX (через assimp → trimesh)
  2. Нормализуем bbox обоих
  3. Cotangent-Laplacian + mass matrix для обоих мешей
  4. Generalized eigendecomp L·ψ = λ·M·ψ (k=80 мод)
  5. Multi-scale HKS descriptors per vertex
  6. Solve functional map: C = argmin ||C·B - A||² + α·||C·Λ_B - Λ_A·C||²
  7. ZoomOut refinement: iteratively увеличиваем k, пересчитываем C
  8. Recover point-to-point map через nearest-neighbor в spectral space
  9. Apply deformation FBX → FLAME (или FLAME → FBX) + Laplacian smoothing
 10. Save: correspondence.csv, deformed mesh OBJ, viz PNG

Использование:
    python python/scripts/motion_groups_v4/functional_map_fit.py \
        --fbx <path> [--flame <path>] [--direction fbx2flame | flame2fbx]
        [--k 30] [--zoomout-steps 5] [--out <run_dir>]

Зависимости: numpy, scipy, trimesh, assimp CLI, open3d (опционально для viz)
"""

from __future__ import annotations
import argparse, json, pickle, subprocess, tempfile, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── MESH I/O ─────────────────────────────────────────────────────────────────

def _install_chumpy_shim():
    """FLAME pkl содержит chumpy-объекты; если chumpy не установлена —
    подсовываем пустые шимы достаточные для unpickling."""
    import sys, types
    if "chumpy" in sys.modules: return
    chumpy = types.ModuleType("chumpy")
    class Ch:
        def __init__(self, *a, **kw): pass
        def __setstate__(self, state):
            # chumpy.Ch хранит value в state['x'] или state['_dirty_vars']
            if isinstance(state, dict):
                self.r = np.array(state.get("x", state.get("_x", None)))
            else: self.r = np.array(state)
        def __reduce__(self): return (Ch, ())
    chumpy.Ch = Ch
    chumpy.ch = Ch
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = chumpy


def load_flame(path: str):
    """FLAME .pkl → (verts, faces). Использует v_template (нейтральный shape)."""
    _install_chumpy_shim()
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    def to_np(x):
        if hasattr(x, "r"): return np.array(x.r)
        if hasattr(x, "toarray"): return x.toarray()
        return np.array(x)
    v = to_np(d["v_template"]).astype(np.float64)
    fa = to_np(d["f"]).astype(np.int64)
    return v, fa


def load_fbx(path: str):
    """FBX → OBJ через assimp → trimesh (process=True мержит UV-splits)."""
    import trimesh as _tm
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False); tmp.close()
    r = subprocess.run(["assimp", "export", path, tmp.name],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"assimp failed: {r.stderr}")
    m = _tm.load(tmp.name, force="mesh", process=False)
    Path(tmp.name).unlink(missing_ok=True)
    m = _tm.Trimesh(vertices=m.vertices, faces=m.faces, process=True)
    return (np.array(m.vertices, dtype=np.float64),
            np.array(m.faces, dtype=np.int64))


def normalize_bbox(v: np.ndarray):
    """Центрируем в 0, масштабируем под диагональ bbox=1."""
    v = v - v.mean(0)
    return v / (np.linalg.norm(v.max(0) - v.min(0)) + 1e-12)


def save_obj(path: Path, verts: np.ndarray, faces: np.ndarray):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ── LAPLACE-BELTRAMI ─────────────────────────────────────────────────────────

def build_operators(verts: np.ndarray, faces: np.ndarray):
    """Cotangent Laplacian L и diagonal mass matrix M (Voronoi area).
    L·ψ = λ·M·ψ — generalized eigenproblem."""
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
    for i in range(3):
        np.add.at(areas, faces[:, i], fa)
    M = sp.diags(areas.clip(min=1e-12))
    return L, M


def compute_spectrum(L, M, k: int = 80):
    """Решаем generalized eigenproblem L·ψ = λ·M·ψ.
    Возвращает (eigvals (k,), eigvecs (N, k)) от младших к старшим."""
    print(f"  Eigendecomp (k={k}, N={L.shape[0]})...")
    t0 = time.time()
    # sigma=0 + which='LM' = shift-invert → находит eigvals возле 0 (нам нужны
    # самые маленькие)
    eigvals, eigvecs = spla.eigsh(L, k=k, M=M, sigma=0, which='LM')
    # Сортируем по возрастанию
    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Normalize: <ψ_i, M ψ_j> = δ_ij
    norms = np.sqrt(np.einsum('ij,ij->j', eigvecs, M @ eigvecs)).clip(min=1e-12)
    eigvecs = eigvecs / norms[None, :]
    print(f"    done in {time.time()-t0:.1f}s, λ range [{eigvals[0]:.3g}, {eigvals[-1]:.3g}]")
    return eigvals, eigvecs


# ── HKS DESCRIPTOR ───────────────────────────────────────────────────────────

def compute_hks(eigvals: np.ndarray, eigvecs: np.ndarray, n_scales: int = 16):
    """Heat Kernel Signature: hks(v, t) = Σ_k exp(-t·λ_k) · ψ_k(v)²
    Возвращает (N, n_scales) — multi-scale дескриптор per vertex."""
    # Log-spaced times от t_min до t_max
    lam_min = max(eigvals[1], 1e-6)            # пропускаем λ_0=0
    lam_max = eigvals[-1]
    t_min = 4 * np.log(10) / lam_max
    t_max = 4 * np.log(10) / lam_min
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_scales)

    # hks[v, t] = Σ_k exp(-t·λ_k) · ψ_k(v)²
    # vectorized: exp(-t λ)_k * ψ²(v,k) — outer product над k
    psi_sq = eigvecs ** 2                       # (N, K)
    hks = np.zeros((eigvecs.shape[0], n_scales))
    for ti, t in enumerate(times):
        weights = np.exp(-t * eigvals)          # (K,)
        hks[:, ti] = psi_sq @ weights           # (N,)
    # Scale-invariant normalization: делим на heat-trace
    heat_trace = np.exp(-times[None, :] * eigvals[:, None]).sum(0)  # (T,)
    hks = hks / heat_trace[None, :].clip(min=1e-12)
    return hks                                  # (N, T)


# ── FUNCTIONAL MAP ───────────────────────────────────────────────────────────

def solve_functional_map(desc_A: np.ndarray, desc_B: np.ndarray,
                          eigvecs_A: np.ndarray, eigvecs_B: np.ndarray,
                          eigvals_A: np.ndarray, eigvals_B: np.ndarray,
                          mass_A: sp.spmatrix, mass_B: sp.spmatrix,
                          k: int, alpha_comm: float = 1e-2):
    """Solve C ∈ ℝ^(k×k) такое что:
        || C·B - A ||²_F + α · || C·Λ_B - Λ_A·C ||²_F

    где A, B — спектральные коэффициенты дескрипторов на каждом меше,
    Λ — diag(eigvals).

    Returns: C, A, B
    """
    # Spectral coefficients of descriptors: <ψ_k, M·g_i>
    Phi_A = eigvecs_A[:, :k]
    Phi_B = eigvecs_B[:, :k]

    # A_coef = Φ_A^T · M_A · desc_A → (k, n_descr)
    A_coef = Phi_A.T @ (mass_A @ desc_A)        # (k, n_descr)
    B_coef = Phi_B.T @ (mass_B @ desc_B)        # (k, n_descr)

    # Data term: minimize ||C·B - A||² → C = A · B^+
    # С регуляризацией коммутативности:
    #   solve (B B^T + α·D²) · C^T = B A^T
    # где D[i,j] = λ_A[i] - λ_B[j] (структурный оператор)

    Lam_A = eigvals_A[:k]
    Lam_B = eigvals_B[:k]

    # Решаем построчно: для каждой строки i из C → C[i, :]
    BBt = B_coef @ B_coef.T                     # (k, k)
    BAt = B_coef @ A_coef.T                     # (k, k)   — note: A·B^T transposed

    C = np.zeros((k, k))
    for i in range(k):
        D_diag = (Lam_A[i] - Lam_B) ** 2        # (k,)
        # (BB^T + α·diag(D²)) · c_i^T = (B·A^T)[:, i]
        lhs = BBt + alpha_comm * np.diag(D_diag)
        rhs = BAt[:, i]
        C[i, :] = np.linalg.solve(lhs, rhs)
    return C, A_coef, B_coef


def recover_p2p(C: np.ndarray, eigvecs_A: np.ndarray, eigvecs_B: np.ndarray, k: int):
    """Из functional map C восстанавливаем point-to-point φ: V_B → V_A
    через nearest neighbor в spectral space.

    Для каждой target-вершины u ∈ B:
        её spectral coord = Φ_B[u, :k]
        C·Φ_B[u, :k] = ожидаемая spectral coord на A
        φ(u) = argmin_v ||Φ_A[v, :k] - C·Φ_B[u, :k]||
    """
    Phi_A = eigvecs_A[:, :k]                    # (N_A, k)
    Phi_B = eigvecs_B[:, :k]                    # (N_B, k)

    # Pullback: spectral coords B → spectral coords A через C^T (transposed!)
    # Если f_A = C · f_B (для функций), то для точечных delta-функций:
    # Φ_A[φ(u), :].T ≈ C · Φ_B[u, :].T
    # → ищем v так что Φ_A[v, :] ≈ (C @ Φ_B[u, :].T).T
    targets_in_A_space = Phi_B @ C.T            # (N_B, k)

    # NN via squared L2 (BLAS-style)
    a_sq = (Phi_A ** 2).sum(1, keepdims=True)   # (N_A, 1)
    b_sq = (targets_in_A_space ** 2).sum(1, keepdims=True).T  # (1, N_B)
    cross = Phi_A @ targets_in_A_space.T        # (N_A, N_B)
    D2 = np.maximum(a_sq + b_sq - 2 * cross, 0)
    p2p = np.argmin(D2, axis=0)                 # (N_B,) → индексы вершин A
    return p2p


def zoomout_refine(eigvecs_A: np.ndarray, eigvecs_B: np.ndarray,
                    mass_A: sp.spmatrix, mass_B: sp.spmatrix,
                    p2p_init: np.ndarray, k_init: int, k_final: int,
                    step: int = 5):
    """ZoomOut: iteratively увеличиваем k, пересчитывая C из p2p и обратно.
    Каждый шаг улучшает корреспонденцию через high-frequency моды."""
    print(f"  ZoomOut refinement: k={k_init} → {k_final} (step={step})")
    p2p = p2p_init.copy()
    K_max = min(eigvecs_A.shape[1], eigvecs_B.shape[1])
    k_final = min(k_final, K_max)

    for k in range(k_init, k_final + 1, step):
        # C ← convert p2p to C: C = Φ_A^T · M_A · Π · Φ_B (где Π — permutation
        # matrix, но в нашем случае Π·Φ_B = Φ_B[p2p_inverse], сложно. Проще:
        # Из формулы f_A = C·f_B и f_A = f_B ∘ φ →
        # C ≈ Φ_A[p2p, :]^T · M_? · Φ_B ... но это не строго.
        # Стандартное упрощение: C = pinv(Φ_A[p2p, :k]) · Φ_B[:, :k]
        # → лучше: C = Φ_B[:, :k]^T · M_B · Φ_A[p2p, :k]  (transpose semantics)
        Phi_A_k = eigvecs_A[:, :k]
        Phi_B_k = eigvecs_B[:, :k]
        # C[i, j] = <ψ_A^i ∘ φ, ψ_B^j> в L²(B)
        # = Σ_u ψ_A^i(φ(u)) · ψ_B^j(u) · M_B(u, u)
        # = (Φ_A[p2p, i])^T · M_B · Φ_B[:, j]
        M_B_diag = mass_B.diagonal()
        C = (Phi_A_k[p2p, :] * M_B_diag[:, None]).T @ Phi_B_k  # (k, k)

        # Пересчёт p2p из нового C
        p2p = recover_p2p(C, eigvecs_A, eigvecs_B, k)
        print(f"    k={k}: refined p2p")

    return p2p, C


# ── DEFORMATION (apply correspondence) ───────────────────────────────────────

def deform_via_correspondence(verts_src: np.ndarray, p2p: np.ndarray,
                                verts_tgt: np.ndarray, faces_src: np.ndarray,
                                smooth_iters: int = 50, alpha: float = 0.5):
    """Применяем p2p: deform src так чтобы src[u] ≈ tgt[p2p[u]].

    p2p: (N_src,) → каждый src vertex u получает target position verts_tgt[p2p[u]]
    Затем Laplacian smoothing для удаления discontinuities.
    """
    target_positions = verts_tgt[p2p]            # (N_src, 3) — куда «двигать»
    delta = target_positions - verts_src

    # Laplacian smoothing на edge-графе src
    N = len(verts_src)
    rows, cols = [], []
    for f in faces_src:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        rows += [a, a, b, b, c, c]
        cols += [b, c, a, c, a, b]
    edges = set(zip(rows, cols))
    rows = np.array([r for r, _ in edges])
    cols = np.array([c for _, c in edges])
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(N, N))
    rs = np.array(A.sum(1)).ravel().clip(min=1)
    W = (sp.diags(1.0 / rs) @ A).tocsr()

    print(f"  Smoothing deformation ({smooth_iters} iters, α={alpha})...")
    d = delta.copy()
    for _ in range(smooth_iters):
        d = (1 - alpha) * d + alpha * (W @ d)
    return verts_src + d, d


# ── VISUALIZATION ────────────────────────────────────────────────────────────

def plot_fmap(C: np.ndarray, out_path: Path):
    fig, ax = plt.subplots(figsize=(6, 5))
    vmax = np.abs(C).max()
    im = ax.imshow(C, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("eigenfunction index (B)")
    ax.set_ylabel("eigenfunction index (A)")
    ax.set_title(f"Functional Map matrix C (k={C.shape[0]})")
    plt.colorbar(im, ax=ax)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def show_meshes_3d(meshes, window_title: str = "Result"):
    """Открывает Open3D окно с несколькими мешами side-by-side.
    meshes: list of (verts, faces, vertex_colors, label) — vertex_colors может
    быть (N,3) RGB в [0,1] или None (тогда uniform-color)."""
    try:
        import open3d as o3d
    except ImportError:
        print("  ⚠ open3d не установлен — пропускаю 3D-просмотр")
        return

    # Раскладываем меши горизонтально с зазором
    geoms = []
    x_cursor = 0.0
    gap_factor = 1.3
    for i, (v, f, colors, label) in enumerate(meshes):
        v = np.asarray(v, dtype=np.float64)
        f = np.asarray(f, dtype=np.int64)
        # Центрируем по своему bbox
        v_centered = v - v.mean(0)
        width = (v_centered.max(0) - v_centered.min(0))[0]
        v_shifted = v_centered.copy()
        v_shifted[:, 0] += x_cursor + width / 2
        x_cursor += width * gap_factor

        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v_shifted),
            o3d.utility.Vector3iVector(f))
        m.compute_vertex_normals()
        if colors is not None and len(colors) == len(v):
            m.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
        else:
            m.paint_uniform_color([0.85, 0.75, 0.68])
        geoms.append(m)

        # Подпись через bbox (не каждый billboard-text возможен — кладём axes
        # на позиции лейбла для ориентации)
        print(f"    [{i}] {label}: {len(v)} verts at x≈{v_shifted[:,0].mean():.3f}")

    print(f"\n  >>> Открываю Open3D окно '{window_title}' (Q → выход) <<<")
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name=window_title, width=1600, height=800):
        print("  ⚠ Не удалось создать Open3D окно")
        return
    for g in geoms:
        vis.add_geometry(g)
    vis.get_render_option().mesh_show_back_face = True
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()
    print(f"  → Окно закрыто")


def plot_correspondence_quality(verts_A: np.ndarray, p2p: np.ndarray,
                                  verts_B: np.ndarray, out_path: Path):
    """3D scatter верт. меша B, цвет = ||deformed_B - tgt_B||."""
    target_pos = verts_A[p2p]
    err = np.linalg.norm(target_pos - verts_B, axis=1)
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(verts_B[:, 0], verts_B[:, 1], verts_B[:, 2],
                     c=err, cmap='RdYlGn_r', s=4, alpha=0.7,
                     vmin=0, vmax=np.percentile(err, 95))
    ax.set_title("Correspondence quality (per target vertex)\n"
                  "green=close to source target, red=far")
    plt.colorbar(sc, ax=ax, label="distance", shrink=0.6)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flame", type=str,
                     default="Muscle-autoskinner/Assets/Meshes/FLAME/FLAME2023 nonommercial use/flame2023.pkl",
                     help="путь к FLAME .pkl")
    ap.add_argument("--fbx", type=str, required=True,
                     help="путь к FBX-файлу")
    ap.add_argument("--direction", choices=["fbx2flame", "flame2fbx"],
                     default="fbx2flame",
                     help="что во что деформируем")
    ap.add_argument("--k", type=int, default=30,
                     help="начальное k для functional map")
    ap.add_argument("--k-max", type=int, default=80,
                     help="максимальное k для ZoomOut")
    ap.add_argument("--zoomout-step", type=int, default=10,
                     help="шаг увеличения k в ZoomOut")
    ap.add_argument("--n-scales", type=int, default=16,
                     help="HKS time-scales")
    ap.add_argument("--alpha-comm", type=float, default=1e-2,
                     help="вес коммутативности с Laplacian")
    ap.add_argument("--smooth-iters", type=int, default=50,
                     help="Laplacian smoothing iterations на финальной деформации")
    ap.add_argument("--smooth-alpha", type=float, default=0.5)
    ap.add_argument("--out", type=str, default=None,
                     help="выходная папка (default = python/scripts/debug_output/fmap_run_<ts>)")
    ap.add_argument("--no-viz", action="store_true",
                     help="не открывать Open3D окно в конце (только сохранить файлы)")
    args = ap.parse_args()

    # ── Output setup ─────────────────────────────────────────────────────────
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(
        f"python/scripts/debug_output/fmap_run_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"● Output: {out_dir}/")

    # ── Load meshes ──────────────────────────────────────────────────────────
    print(f"\n● Load FLAME: {args.flame}")
    v_flame, f_flame = load_flame(args.flame)
    print(f"  {len(v_flame)} verts, {len(f_flame)} faces")

    print(f"\n● Load FBX: {args.fbx}")
    v_fbx, f_fbx = load_fbx(args.fbx)
    print(f"  {len(v_fbx)} verts, {len(f_fbx)} faces")

    # ── Normalize ────────────────────────────────────────────────────────────
    v_flame_n = normalize_bbox(v_flame)
    v_fbx_n   = normalize_bbox(v_fbx)

    # Решаем кто source (A) и кто target (B)
    if args.direction == "fbx2flame":
        # деформируем FBX так чтобы он стал похож на FLAME
        v_A, f_A, label_A = v_flame_n, f_flame, "FLAME"
        v_B, f_B, label_B = v_fbx_n,   f_fbx,   "FBX"
        v_src_world, v_tgt_world = v_fbx, v_flame
        f_src = f_fbx
    else:
        v_A, f_A, label_A = v_fbx_n,   f_fbx,   "FBX"
        v_B, f_B, label_B = v_flame_n, f_flame, "FLAME"
        v_src_world, v_tgt_world = v_flame, v_fbx
        f_src = f_flame

    print(f"\n  Direction: {args.direction}  (source=A={label_A}, target=B={label_B})")
    # Хотим деформировать SRC так чтобы он лёг на TGT.
    # SRC — это меш чьи topology сохраним; для каждой src-вершины ищем
    # tgt-вершину куда её сдвинуть.
    # → p2p должно отображать src → tgt
    #   В терминах FM: SRC = B (target в FM-смысле), TGT = A (source в FM-смысле)
    # Меняем семантику:
    if args.direction == "fbx2flame":
        # SRC = FBX, TGT = FLAME
        v_src_n, f_src_n = v_fbx_n,   f_fbx
        v_tgt_n, f_tgt_n = v_flame_n, f_flame
    else:
        v_src_n, f_src_n = v_flame_n, f_flame
        v_tgt_n, f_tgt_n = v_fbx_n,   f_fbx
    src_label = "FBX" if args.direction == "fbx2flame" else "FLAME"
    tgt_label = "FLAME" if args.direction == "fbx2flame" else "FBX"
    print(f"  SRC ({src_label}) → TGT ({tgt_label}): для каждой SRC-вершины ищем TGT-вершину")

    # ── Build Laplacian operators ────────────────────────────────────────────
    print(f"\n● Cotangent Laplacian + mass matrix")
    print(f"  {src_label}...")
    L_src, M_src = build_operators(v_src_n, f_src_n)
    print(f"  {tgt_label}...")
    L_tgt, M_tgt = build_operators(v_tgt_n, f_tgt_n)

    # ── Eigendecomposition ───────────────────────────────────────────────────
    print(f"\n● Eigendecomposition (k={args.k_max})")
    print(f"  {src_label}:")
    eig_src_v, eig_src_f = compute_spectrum(L_src, M_src, k=args.k_max)
    print(f"  {tgt_label}:")
    eig_tgt_v, eig_tgt_f = compute_spectrum(L_tgt, M_tgt, k=args.k_max)

    # ── HKS descriptors ──────────────────────────────────────────────────────
    print(f"\n● HKS descriptors (n_scales={args.n_scales})")
    hks_src = compute_hks(eig_src_v, eig_src_f, n_scales=args.n_scales)
    hks_tgt = compute_hks(eig_tgt_v, eig_tgt_f, n_scales=args.n_scales)
    print(f"  HKS shape: src={hks_src.shape}, tgt={hks_tgt.shape}")

    # ── Solve functional map ─────────────────────────────────────────────────
    # FM map: для функции f на TGT, её pullback на SRC = C · f_spectral
    # Семантика: A = TGT (где живёт source descriptor); B = SRC
    # Поэтому в solve_functional_map:
    #   desc_A = hks_tgt, eigvecs_A = eig_tgt_f, ...
    # И recover_p2p вернёт φ: V_SRC → V_TGT
    print(f"\n● Solve functional map (k={args.k}, α={args.alpha_comm})")
    C_init, _, _ = solve_functional_map(
        desc_A=hks_tgt, desc_B=hks_src,
        eigvecs_A=eig_tgt_f, eigvecs_B=eig_src_f,
        eigvals_A=eig_tgt_v, eigvals_B=eig_src_v,
        mass_A=M_tgt, mass_B=M_src,
        k=args.k, alpha_comm=args.alpha_comm,
    )
    print(f"  C shape: {C_init.shape}, ||C||_F = {np.linalg.norm(C_init):.4f}")

    plot_fmap(C_init, out_dir / "C_initial.png")
    print(f"  → saved C_initial.png")

    # ── Recover initial p2p ──────────────────────────────────────────────────
    print(f"\n● Recover point-to-point (k={args.k})")
    p2p_init = recover_p2p(C_init, eig_tgt_f, eig_src_f, args.k)
    print(f"  p2p shape: {p2p_init.shape} → indices в {tgt_label}")

    # ── ZoomOut refinement ───────────────────────────────────────────────────
    p2p_final, C_final = zoomout_refine(
        eigvecs_A=eig_tgt_f, eigvecs_B=eig_src_f,
        mass_A=M_tgt, mass_B=M_src,
        p2p_init=p2p_init,
        k_init=args.k, k_final=args.k_max,
        step=args.zoomout_step,
    )
    plot_fmap(C_final, out_dir / "C_refined.png")
    print(f"  → saved C_refined.png")

    # ── Save correspondence ──────────────────────────────────────────────────
    import pandas as pd
    corr_df = pd.DataFrame({
        f"{src_label.lower()}_vertex": np.arange(len(p2p_final)),
        f"{tgt_label.lower()}_vertex": p2p_final,
    })
    corr_df.to_csv(out_dir / "correspondence.csv", index=False)
    print(f"\n● Saved correspondence: {len(p2p_final)} mappings → correspondence.csv")

    # ── Deform SRC → TGT ─────────────────────────────────────────────────────
    print(f"\n● Deform {src_label} → {tgt_label}")
    # Используем world-coordinate меши (не нормализованные) для финальной геометрии
    src_world = v_fbx if args.direction == "fbx2flame" else v_flame
    tgt_world = v_flame if args.direction == "fbx2flame" else v_fbx
    f_src_world = f_fbx if args.direction == "fbx2flame" else f_flame

    deformed, delta = deform_via_correspondence(
        verts_src=src_world,
        p2p=p2p_final,
        verts_tgt=tgt_world,
        faces_src=f_src_world,
        smooth_iters=args.smooth_iters,
        alpha=args.smooth_alpha,
    )
    print(f"  max ||δ|| = {np.linalg.norm(delta, axis=1).max():.4f}")
    print(f"  mean ||δ|| = {np.linalg.norm(delta, axis=1).mean():.4f}")

    # ── Save OBJ ─────────────────────────────────────────────────────────────
    obj_path = out_dir / f"{src_label.lower()}_deformed_to_{tgt_label.lower()}.obj"
    save_obj(obj_path, deformed, f_src_world)
    print(f"  → saved {obj_path.name}")
    save_obj(out_dir / f"{src_label.lower()}_rest.obj", src_world, f_src_world)
    save_obj(out_dir / f"{tgt_label.lower()}_rest.obj", tgt_world,
              f_flame if args.direction == "fbx2flame" else f_fbx)

    # ── Visualization ────────────────────────────────────────────────────────
    plot_correspondence_quality(
        verts_A=tgt_world, p2p=p2p_final, verts_B=src_world,
        out_path=out_dir / "correspondence_quality.png",
    )
    print(f"  → saved correspondence_quality.png")

    # ── Metadata ─────────────────────────────────────────────────────────────
    meta = {
        "flame_path": args.flame,
        "fbx_path": args.fbx,
        "direction": args.direction,
        "k_init": args.k,
        "k_max": args.k_max,
        "zoomout_step": args.zoomout_step,
        "n_scales": args.n_scales,
        "alpha_comm": args.alpha_comm,
        "smooth_iters": args.smooth_iters,
        "smooth_alpha": args.smooth_alpha,
        "n_src": len(src_world),
        "n_tgt": len(tgt_world),
        "src_label": src_label,
        "tgt_label": tgt_label,
        "max_delta": float(np.linalg.norm(delta, axis=1).max()),
        "mean_delta": float(np.linalg.norm(delta, axis=1).mean()),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"  → saved metadata.json")

    print(f"\n✓ Готово. Всё в: {out_dir}/")
    print(f"  Открыть OBJ: open {obj_path}")

    # ── 3D Visualization (Open3D) ────────────────────────────────────────────
    if not args.no_viz:
        print(f"\n● 3D просмотр (Open3D)")
        # Цветим деформированный меш по ||δ|| (зелёный=мало, красный=много)
        delta_mag = np.linalg.norm(delta, axis=1)
        dmax = max(delta_mag.max(), 1e-9)
        t = np.clip(delta_mag / dmax, 0, 1)
        # heat-like colormap (green → yellow → red)
        col_deformed = np.zeros((len(deformed), 3))
        col_deformed[:, 0] = t                                     # R
        col_deformed[:, 1] = np.clip(2 * (1 - t), 0, 1) * 0.7 + 0.3  # G
        col_deformed[:, 2] = 0.2 * (1 - t)                          # B

        # SRC rest — нейтральный
        col_src = np.tile([0.7, 0.75, 0.85], (len(src_world), 1))
        # TGT rest — слегка розовый чтоб отличался
        col_tgt = np.tile([0.85, 0.7, 0.7], (len(tgt_world), 1))

        f_tgt_world = f_flame if args.direction == "fbx2flame" else f_fbx
        show_meshes_3d(
            [
                (src_world, f_src_world, col_src,
                 f"{src_label} rest"),
                (tgt_world, f_tgt_world, col_tgt,
                 f"{tgt_label} rest (target)"),
                (deformed,  f_src_world, col_deformed,
                 f"{src_label} → {tgt_label} (colored by ||δ||)"),
            ],
            window_title=f"Functional Map: {src_label}→{tgt_label}  |  "
                          f"Q→выход",
        )


if __name__ == "__main__":
    main()
