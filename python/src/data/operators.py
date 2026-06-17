"""
Спектральные операторы для DiffusionNet: mass, eigenvalues, eigenvectors,
gradX, gradY. Считаются ОДИН раз на меш (зависят только от геометрии нейтрали)
и кешируются на диск — это самый дорогой предпросчёт.

Использует robust_laplacian (устойчив к плохой триангуляции) + potpourri3d
для градиентов в касательной плоскости.
"""
import hashlib
from pathlib import Path

import numpy as np
import scipy.sparse
import scipy.sparse.linalg as sla


def _mesh_hash(verts, faces):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(verts, np.float64).tobytes())
    h.update(np.ascontiguousarray(faces, np.int64).tobytes())
    return h.hexdigest()[:16]


def compute_operators(verts, faces, k_eig=128):
    """Возвращает dict с numpy/scipy операторами:
       mass (V,), evals (k,), evecs (V,k), gradX (V,V sparse), gradY (V,V sparse).

    gradX/gradY — операторы тангенциального градиента (комплексные → берём
    действительную и мнимую части как две вещественные оси)."""
    import robust_laplacian
    import potpourri3d as pp3d

    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)

    # котангенс-Лапласиан + масса (robust к вырожденным треугольникам)
    L, M = robust_laplacian.mesh_laplacian(verts, faces)
    massvec = np.asarray(M.diagonal()).ravel()

    # обобщённая задача L v = λ M v → k низких мод
    k = min(int(k_eig), len(verts) - 2)
    M_mat = scipy.sparse.diags(massvec)
    try:
        evals, evecs = sla.eigsh(L, k=k, M=M_mat, sigma=1e-8, which="LM")
    except Exception:
        evals, evecs = sla.eigsh(L + 1e-8 * scipy.sparse.eye(len(verts)),
                                 k=k, M=M_mat, which="SM")
    order = np.argsort(evals)
    evals = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]

    # тангенциальный градиент через potpourri3d (комплексный)
    solver = pp3d.MeshHeatMethodDistanceSolver  # noqa: F841 (touch import)
    gradX, gradY = _build_gradients(verts, faces, massvec)

    return {
        "mass": massvec.astype(np.float32),
        "evals": evals.astype(np.float32),
        "evecs": evecs.astype(np.float32),
        "gradX": gradX.astype(np.float32),
        "gradY": gradY.astype(np.float32),
    }


def _build_gradients(verts, faces, massvec):
    """Грубые, но рабочие операторы градиента по граням, усреднённые на вершины.
    Возвращает (gradX, gradY) как scipy.sparse (V, V). Две ортогональные
    касательные оси на вершину (через локальный базис от нормали)."""
    V = len(verts)
    # нормали вершин (усреднение нормалей граней)
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12)
    vn = np.zeros((V, 3))
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)
    # локальный касательный базис (e1, e2) на вершину
    ref = np.tile([1.0, 0.0, 0.0], (V, 1))
    bad = np.abs((vn * ref).sum(1)) > 0.9
    ref[bad] = [0.0, 1.0, 0.0]
    e1 = ref - (ref * vn).sum(1, keepdims=True) * vn
    e1 /= (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12)
    e2 = np.cross(vn, e1)

    # для каждого ребра (i,j) вклад в градиент по проекции на e1/e2
    rows, cols, dx, dy = [], [], [], []
    deg = np.zeros(V)
    for tri in faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(tri[a]), int(tri[b])
            d = verts[j] - verts[i]
            rows += [i, i]; cols += [j, i]
            dxi = d @ e1[i]; dyi = d @ e2[i]
            dx += [dxi, -dxi]; dy += [dyi, -dyi]
            deg[i] += 1
    gx = scipy.sparse.csr_matrix((dx, (rows, cols)), shape=(V, V))
    gy = scipy.sparse.csr_matrix((dy, (rows, cols)), shape=(V, V))
    inv = scipy.sparse.diags(1.0 / np.clip(deg, 1, None))
    return inv @ gx, inv @ gy


# ── кеш на диск ──────────────────────────────────────────────────────────────

def get_operators(verts, faces, k_eig=128, cache_dir=None):
    """Считает операторы с кешированием по хешу геометрии. cache_dir=None →
    без кеша (всегда пересчёт)."""
    if cache_dir is None:
        return compute_operators(verts, faces, k_eig)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"{_mesh_hash(verts, faces)}_k{k_eig}.npz"
    p = cache_dir / key
    if p.exists():
        d = np.load(p, allow_pickle=True)
        return _unpack_npz(d)
    ops = compute_operators(verts, faces, k_eig)
    _save_npz(p, ops)
    return ops


def _save_npz(path, ops):
    np.savez(path,
             mass=ops["mass"], evals=ops["evals"], evecs=ops["evecs"],
             gradX_data=ops["gradX"].data, gradX_idx=ops["gradX"].indices,
             gradX_ptr=ops["gradX"].indptr, gradX_shape=ops["gradX"].shape,
             gradY_data=ops["gradY"].data, gradY_idx=ops["gradY"].indices,
             gradY_ptr=ops["gradY"].indptr, gradY_shape=ops["gradY"].shape)


def _unpack_npz(d):
    gx = scipy.sparse.csr_matrix(
        (d["gradX_data"], d["gradX_idx"], d["gradX_ptr"]),
        shape=tuple(d["gradX_shape"]))
    gy = scipy.sparse.csr_matrix(
        (d["gradY_data"], d["gradY_idx"], d["gradY_ptr"]),
        shape=tuple(d["gradY_shape"]))
    return {"mass": d["mass"], "evals": d["evals"], "evecs": d["evecs"],
            "gradX": gx, "gradY": gy}
