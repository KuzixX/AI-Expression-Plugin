"""
Инференс: грузит чекпойнт + голову из HDF5 → предсказанные skinning weights
W (N, M). Используется визуализатором и экспортом.

  from src.inference.predict import load_model, predict_weights
"""
import sys
from pathlib import Path

import numpy as np
import torch

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

from src.data.operators import get_operators                  # noqa: E402
from src.data.features import build_vertex_features           # noqa: E402
from src.data.muscles import make_dummy_rig                   # noqa: E402
from src.models.skinning_net import MuscleSkinningNet         # noqa: E402


def load_model(ckpt_path, device=None, F_m=None, use_wks=False):
    device = device or ("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available()
                        else "cpu")
    ck = torch.load(ckpt_path, map_location=device)
    model = MuscleSkinningNet(
        C_in=ck["C_in"], F_m=ck.get("F_m", F_m or 7),
        n_muscles=ck["n_muscles"], D=ck.get("dim", 128),
        diff_width=ck.get("width", 128),
        diff_blocks=ck.get("blocks", 4)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck["n_muscles"], device


def predict_weights(model, n_muscles, device, h5_path, head_name,
                    k_eig=128, use_wks=False, cache_dir=None, rig=None):
    """Возвращает (verts (N,3), faces (K,3), W (N, M) numpy)."""
    import h5py
    with h5py.File(h5_path, "r") as h:
        g = h["heads"][head_name]
        verts = g["neutral"][:].astype(np.float64)
        faces = g["faces"][:].astype(np.int64)
    cache_dir = cache_dir or (Path(h5_path).parent / "op_cache")
    ops = get_operators(verts, faces, k_eig, cache_dir=cache_dir)
    feats = build_vertex_features(verts, faces, ops, use_wks)
    if rig is None:
        rig = make_dummy_rig(n_muscles, verts)
    mf = torch.from_numpy(rig.muscle_features()).to(device)
    geo, align = rig.pair_priors(verts)
    with torch.no_grad():
        W = model(
            torch.from_numpy(feats).to(device),
            torch.from_numpy(ops["mass"]).to(device),
            torch.from_numpy(ops["evals"]).to(device),
            torch.from_numpy(ops["evecs"]).to(device),
            _sp2t(ops["gradX"], device), _sp2t(ops["gradY"], device),
            mf, torch.from_numpy(geo).to(device),
            torch.from_numpy(align).to(device))
    return verts, faces, W.cpu().numpy()


def _sp2t(m, device):
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    val = torch.from_numpy(m.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, m.shape).coalesce().to(device)
