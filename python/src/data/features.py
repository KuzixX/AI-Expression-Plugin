"""
Per-vertex input features for the network: position, normal, curvature,
optional spectral descriptors (WKS). These go into the DiffusionNet encoder.
"""
import numpy as np


def vertex_normals(verts, faces):
    fn = np.cross(verts[faces[:, 1]] - verts[faces[:, 0]],
                  verts[faces[:, 2]] - verts[faces[:, 0]])
    vn = np.zeros_like(verts)
    for k in range(3):
        np.add.at(vn, faces[:, k], fn)
    n = np.linalg.norm(vn, axis=1, keepdims=True)
    return vn / np.clip(n, 1e-12, None)


def mean_curvature_proxy(verts, faces, normals):
    """Дешёвый proxy кривизны: средняя проекция рёбер 1-кольца на нормаль."""
    V = len(verts)
    acc = np.zeros(V); deg = np.zeros(V)
    for tri in faces:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            i, j = int(tri[a]), int(tri[b])
            d = verts[j] - verts[i]
            acc[i] += d @ normals[i]; deg[i] += 1
    return acc / np.clip(deg, 1, None)


def wks_feature(evals, evecs, n_e=16, sigma_scale=7.0):
    """Компактный WKS как доп. фичи (n_e каналов)."""
    lam = np.clip(evals, 0.0, None)
    nz = lam > 1e-8
    lam = lam[nz]; phi = evecs[:, nz]
    log_lam = np.log(lam)
    e = np.linspace(log_lam.min(), log_lam.max(), n_e)
    sigma = sigma_scale * (e[1] - e[0] if n_e > 1 else 1.0)
    G = np.exp(-((e[:, None] - log_lam[None, :]) ** 2) / (2 * sigma ** 2))
    norm = G.sum(1).clip(1e-12)
    return ((phi ** 2) @ G.T) / norm[None, :]            # (V, n_e)


def build_vertex_features(verts, faces, ops=None, use_wks=False, wks_n=16):
    """Собираем per-vertex фичи. Возвращает (V, C_in) float32.

    Базовые: xyz(3) + normal(3) + curvature(1) = 7.
    + use_wks → ещё wks_n каналов (нужны ops с evals/evecs)."""
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    nrm = vertex_normals(verts, faces)
    curv = mean_curvature_proxy(verts, faces, nrm)[:, None]
    feats = [verts, nrm, curv]
    if use_wks and ops is not None:
        feats.append(wks_feature(ops["evals"], ops["evecs"], n_e=wks_n))
    return np.concatenate(feats, axis=1).astype(np.float32)


def feature_dim(use_wks=False, wks_n=16):
    return 7 + (wks_n if use_wks else 0)
