"""
flame_fit_transfer.py — FLAME shape-fitting to FBX + direct expression transfer.

Идея (вместо procedural matching):
1. Загружаем FLAME (v_template + shapedirs) и FBX
2. ICP-fit: подгоняем FLAME shape betas чтобы FLAME-mesh совпал по форме с FBX
   (~300 shape parameters → большое разнообразие face shapes)
3. После fit'а: каждая FBX-вершина имеет ближайшую FLAME-вершину как ground truth
   correspondence
4. Любую FLAME-deformation (blendshape expression) можно перенести через эту
   correspondence: δ_FBX[u] = δ_FLAME[NN(u)]

Это альтернатива всем нашим procedural matching методам (heat_zone, tps_global,
direct_copy и т.д.) которые не работают на multi-anchor сценарии.

Это упрощённая offline-версия идеи MICA/DECA/FLAME-tracker (без deep learning).
Если нужен industrial-quality fit — можно подменить ICP-fit на pretrained MICA.

Использование:
    python python/scripts/motion_groups_v5/flame_fit_transfer.py \
        --fbx <path/to/FBX> \
        [--expression-betas "300:8.0,302:-5.0"]
        [--fit-iters 30] [--scale-mode bbox|none]

Output (в python/scripts/debug_output/fit_run_<TS>/):
    fitted_flame.obj            — FLAME-mesh подогнанный под форму FBX
    fbx_to_flame_correspondence.csv  — table fbx_vert → flame_vert + distance
    fbx_deformed.obj            — FBX с применённой FLAME-deformation
    delta_per_fbx_vertex.csv    — δ для каждой FBX-вершины
    metadata.json               — параметры fit'а
"""

from __future__ import annotations
import argparse, json, pickle, subprocess, tempfile, time
from pathlib import Path
import numpy as np

# ── FLAME loader (chumpy shim как в других скриптах) ──────────────────────────

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
    """Returns (v_template, shapedirs, faces, exprdirs).
    v_template: (5023, 3) — neutral mesh
    shapedirs:  (5023, 3, N_shape) — shape blendshape directions
    exprdirs:   (5023, 3, N_expr)  — expression blendshape directions (если есть)
    faces:      (F, 3)
    """
    _chumpy_shim()
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    def to_np(x):
        if hasattr(x, "r"): return np.array(x.r)
        if hasattr(x, "toarray"): return x.toarray()
        return np.array(x)
    v_template = to_np(d["v_template"]).astype(np.float64)
    shapedirs  = to_np(d["shapedirs"]).astype(np.float64)
    faces = to_np(d["f"]).astype(np.int64)
    # FLAME 2023 splits shape and expression betas in shapedirs:
    # обычно shape = первые 300, expression = последующие
    # Здесь возвращаем shapedirs целиком, разделение делает caller
    return v_template, shapedirs, faces


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


# ── Rigid alignment ──────────────────────────────────────────────────────────

def bbox_normalize(v):
    """Returns (v_normalized, center, scale)."""
    center = v.mean(0)
    vc = v - center
    scale = np.linalg.norm(vc.max(0) - vc.min(0)) + 1e-12
    return vc / scale, center, scale


def rigid_align_via_procrustes(src, tgt):
    """Rigid Procrustes без scale (только R + t)."""
    src_c = src.mean(0); tgt_c = tgt.mean(0)
    A = src - src_c; B = tgt - tgt_c
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1; R = Vt.T @ U.T
    t = tgt_c - R @ src_c
    return R, t


# ── FLAME shape fitting via ICP ──────────────────────────────────────────────

def fit_flame_shape_to_fbx(v_template, shapedirs, faces_flame,
                             verts_fbx, n_betas=100,
                             n_iter=30, learning_rate=0.5,
                             beta_reg=0.001, verbose=True):
    """ICP-fit FLAME shape parameters к FBX geometry.

    Минимизирует Σ ||FLAME[v] - FBX[NN_FBX(FLAME[v])]||² + λ·||β||²

    n_betas: число shape betas использовать (FLAME ~300, можно меньше для скорости)
    learning_rate: 0..1, насколько сильно применять каждый β-update
    beta_reg: regularization on β (не давать выходить за reasonable shape space)
    """
    from scipy.spatial import cKDTree

    N_v = v_template.shape[0]
    N_betas_total = shapedirs.shape[2]
    n_betas = min(n_betas, N_betas_total)
    sd = shapedirs[:, :, :n_betas]                 # (V, 3, n_betas)
    sd_flat = sd.reshape(-1, n_betas)              # (V*3, n_betas)

    # Pre-build для regularization
    if beta_reg > 0:
        # Solve (A^T A + λI) β = A^T b
        reg_I = beta_reg * np.eye(n_betas)
    else:
        reg_I = None

    # Pre-compute FBX KD-tree
    tree_fbx = cKDTree(verts_fbx)

    # Initial β = zeros (neutral FLAME)
    betas = np.zeros(n_betas)
    V = v_template.copy()

    if verbose:
        print(f"  Fit: {n_betas}/{N_betas_total} betas, "
              f"{n_iter} iters, lr={learning_rate}, reg={beta_reg}")

    history = []
    for it in range(n_iter):
        # Current FLAME mesh
        V = v_template + np.einsum('vdb,b->vd', sd, betas)

        # NN: для каждой FLAME-вершины найти ближайшую FBX
        nn_dist, nn_idx = tree_fbx.query(V, k=1)
        targets = verts_fbx[nn_idx]                # (V, 3)

        # Solve: sd_flat @ β_new ≈ (targets - v_template).flatten()
        rhs_v = (targets - v_template).reshape(-1)
        if reg_I is not None:
            A_norm = sd_flat.T @ sd_flat + reg_I
            b_norm = sd_flat.T @ rhs_v
            beta_target = np.linalg.solve(A_norm, b_norm)
        else:
            beta_target, *_ = np.linalg.lstsq(sd_flat, rhs_v, rcond=None)

        # Step with learning rate
        betas = (1 - learning_rate) * betas + learning_rate * beta_target

        if verbose and (it < 3 or it % 5 == 0 or it == n_iter - 1):
            mean_d = float(nn_dist.mean())
            max_d  = float(nn_dist.max())
            beta_norm = float(np.linalg.norm(betas))
            print(f"  iter {it+1}/{n_iter}: NN_dist mean={mean_d:.5f}, "
                  f"max={max_d:.5f}, |β|={beta_norm:.2f}")
        history.append({'iter': it+1, 'mean_nn': float(nn_dist.mean()),
                         'max_nn': float(nn_dist.max())})

    V_final = v_template + np.einsum('vdb,b->vd', sd, betas)
    return betas, V_final, history


# ── Save OBJ ─────────────────────────────────────────────────────────────────

def save_obj(path, verts, faces, comment=""):
    with open(path, "w") as f:
        if comment: f.write(f"# {comment}\n")
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flame", type=str,
                     default="Muscle-autoskinner/Assets/Meshes/FLAME/FLAME2023 nonommercial use/flame2023.pkl",
                     help="path to FLAME .pkl")
    ap.add_argument("--fbx", type=str, required=True,
                     help="path to FBX (или OBJ) face mesh")
    ap.add_argument("--n-betas", type=int, default=100,
                     help="сколько shape betas использовать (default 100)")
    ap.add_argument("--fit-iters", type=int, default=30,
                     help="число ICP iterations для shape fit")
    ap.add_argument("--learning-rate", type=float, default=0.5,
                     help="step size для каждой iter (0..1)")
    ap.add_argument("--beta-reg", type=float, default=0.001,
                     help="L2 reg on β (не выходить за reasonable shape space)")
    ap.add_argument("--scale-mode", choices=["bbox", "none"], default="bbox",
                     help="bbox-normalize meshes перед fit'ом")
    ap.add_argument("--expression-betas", type=str, default="",
                     help="expression to transfer, format 'idx:val,idx:val' "
                          "(например '300:8,302:-5')")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    # ── Output dir ───────────────────────────────────────────────────────────
    import datetime as _dt
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(
        f"python/scripts/debug_output/fit_run_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"● Output: {out_dir}/")

    # ── Load meshes ──────────────────────────────────────────────────────────
    print(f"\n● Load FLAME: {args.flame}")
    v_template, shapedirs, faces_flame = load_flame(args.flame)
    print(f"  v_template: {v_template.shape}, shapedirs: {shapedirs.shape}, "
          f"faces: {faces_flame.shape}")

    print(f"\n● Load FBX: {args.fbx}")
    verts_fbx, faces_fbx = load_fbx(args.fbx)
    print(f"  {len(verts_fbx)} verts, {len(faces_fbx)} faces")

    # ── Normalize ────────────────────────────────────────────────────────────
    v_template_orig = v_template.copy()
    verts_fbx_orig  = verts_fbx.copy()

    if args.scale_mode == "bbox":
        v_template, c_f, s_f = bbox_normalize(v_template)
        verts_fbx, c_fbx, s_fbx = bbox_normalize(verts_fbx)
        # Re-scale shapedirs тоже
        shapedirs = shapedirs / s_f
        print(f"  Normalized: FLAME scale={s_f:.4f}, FBX scale={s_fbx:.4f}")
    else:
        c_f = np.zeros(3); s_f = 1.0
        c_fbx = np.zeros(3); s_fbx = 1.0

    # Rigid pre-alignment via Procrustes (на centroids; уже совпадают после bbox)
    # Skip — bbox normalize уже центрирует обе

    # ── Fit FLAME shape ──────────────────────────────────────────────────────
    print(f"\n● Fitting FLAME shape to FBX geometry...")
    t0 = time.time()
    betas, V_fitted, history = fit_flame_shape_to_fbx(
        v_template, shapedirs, faces_flame, verts_fbx,
        n_betas=args.n_betas,
        n_iter=args.fit_iters,
        learning_rate=args.learning_rate,
        beta_reg=args.beta_reg,
        verbose=True,
    )
    fit_time = time.time() - t0
    print(f"  ✓ fit completed in {fit_time:.1f}s")
    print(f"  final mean NN dist: {history[-1]['mean_nn']:.5f}, "
          f"max: {history[-1]['max_nn']:.5f}")

    # ── Save fitted FLAME mesh ───────────────────────────────────────────────
    V_fitted_world = V_fitted * s_fbx + c_fbx       # back to FBX world coords
    save_obj(out_dir / "fitted_flame.obj", V_fitted_world, faces_flame,
              comment=f"FLAME-shape fitted to {Path(args.fbx).name}")
    print(f"  → fitted_flame.obj saved")

    # ── Correspondence: FBX_vertex → FLAME_vertex ────────────────────────────
    from scipy.spatial import cKDTree
    tree_flame = cKDTree(V_fitted)
    nn_dist, fbx_to_flame = tree_flame.query(verts_fbx, k=1)
    print(f"\n● Correspondence FBX → FLAME:")
    print(f"  mean NN dist: {nn_dist.mean():.5f}")
    print(f"  median NN dist: {np.median(nn_dist):.5f}")
    print(f"  max NN dist: {nn_dist.max():.5f}")

    import pandas as pd
    df = pd.DataFrame({
        "fbx_vertex": np.arange(len(verts_fbx)),
        "flame_vertex": fbx_to_flame,
        "nn_distance": nn_dist,
    })
    df.to_csv(out_dir / "fbx_to_flame_correspondence.csv", index=False)
    print(f"  → fbx_to_flame_correspondence.csv saved")

    # ── Apply expression (если задан) и transfer ─────────────────────────────
    if args.expression_betas:
        print(f"\n● Apply expression and transfer to FBX")
        # Parse expression
        expr_dict = {}
        for part in args.expression_betas.split(","):
            p = part.strip()
            if not p: continue
            try:
                idx, val = p.split(":")
                expr_dict[int(idx)] = float(val)
            except Exception:
                print(f"  ⚠ bad expression spec '{p}', skipping")

        if expr_dict:
            # Compute FLAME deformed mesh
            betas_expr = np.zeros(shapedirs.shape[2])
            for idx, val in expr_dict.items():
                if 0 <= idx < shapedirs.shape[2]:
                    betas_expr[idx] = val
            V_with_expr = v_template + np.einsum('vdb,b->vd', shapedirs, betas_expr)
            delta_flame = V_with_expr - v_template      # FLAME deformation (in normalized space)

            # Transfer to FBX via correspondence
            # δ_FBX[u] = δ_FLAME[fbx_to_flame[u]] (scale already in shapedirs)
            delta_fbx = delta_flame[fbx_to_flame]
            verts_fbx_deformed = verts_fbx + delta_fbx

            # Back to world coords
            verts_fbx_deformed_world = verts_fbx_deformed * s_fbx + c_fbx
            save_obj(out_dir / "fbx_deformed.obj",
                      verts_fbx_deformed_world, faces_fbx,
                      comment=f"FBX with FLAME expression {expr_dict}")

            delta_fbx_world = delta_fbx * s_fbx
            df_delta = pd.DataFrame({
                "fbx_vertex": np.arange(len(verts_fbx)),
                "dx": delta_fbx_world[:, 0],
                "dy": delta_fbx_world[:, 1],
                "dz": delta_fbx_world[:, 2],
                "magnitude": np.linalg.norm(delta_fbx_world, axis=1),
            })
            df_delta.to_csv(out_dir / "delta_per_fbx_vertex.csv", index=False)
            print(f"  expression betas: {expr_dict}")
            print(f"  |δ_FLAME| max: {np.linalg.norm(delta_flame, axis=1).max():.5f}")
            print(f"  |δ_FBX|   max: {np.linalg.norm(delta_fbx_world, axis=1).max():.5f}")
            print(f"  → fbx_deformed.obj + delta_per_fbx_vertex.csv saved")
    else:
        print(f"\n● No expression specified (--expression-betas пустой)")
        print(f"  Только fit + correspondence сохранены. Можно использовать для batch generate.")

    # ── Save metadata ────────────────────────────────────────────────────────
    metadata = {
        "flame_path": args.flame,
        "fbx_path": args.fbx,
        "n_betas": args.n_betas,
        "fit_iters": args.fit_iters,
        "learning_rate": args.learning_rate,
        "beta_reg": args.beta_reg,
        "scale_mode": args.scale_mode,
        "n_fbx_verts": int(len(verts_fbx)),
        "n_flame_verts": int(len(v_template)),
        "fit_history_mean_nn": [h['mean_nn'] for h in history],
        "fit_history_max_nn": [h['max_nn'] for h in history],
        "final_mean_nn": float(history[-1]['mean_nn']),
        "final_max_nn": float(history[-1]['max_nn']),
        "fbx_to_flame_nn_mean": float(nn_dist.mean()),
        "fbx_to_flame_nn_median": float(np.median(nn_dist)),
        "fitted_betas_norm": float(np.linalg.norm(betas)),
        "fit_time_seconds": fit_time,
        "expression_betas": args.expression_betas,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\n  → metadata.json saved")
    print(f"\n✓ Done. Outputs in: {out_dir}/")
    print(f"\n  Open fitted_flame.obj in Blender/MeshLab to see FLAME shape fit.")
    if args.expression_betas:
        print(f"  Open fbx_deformed.obj to see expression transfer result.")


if __name__ == "__main__":
    main()
