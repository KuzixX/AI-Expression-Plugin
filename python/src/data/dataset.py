"""
Dataset: читает transferred.h5 (выход motion_groups_v7) и отдаёт головы для
обучения. Единица датасета = ГОЛОВА (с её нейтралью, операторами, фичами и
всеми δ выражений + активациями мышц).

Схема входного HDF5 (см. motion_groups_v7):
  /muscle_names                       (M,)
  /expressions/<выраж>/activations    (M,)   — общие для всех голов
  /heads/<имя>/neutral                (N,3)
  /heads/<имя>/faces                  (K,3)
  /heads/<имя>/expr/<выраж>/delta      (N,3)
  /heads/<имя>.attrs['bad_transfer']  bool   — помеченные плохими (исключаем)
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .operators import get_operators
from .features import build_vertex_features, feature_dim


class HeadDataset(Dataset):
    def __init__(self, h5_path, k_eig=128, use_wks=False, wks_n=16,
                 cache_dir=None, skip_bad=True, head_names=None):
        self.h5_path = str(h5_path)
        self.k_eig = k_eig
        self.use_wks = use_wks
        self.wks_n = wks_n
        self.cache_dir = cache_dir or (Path(h5_path).parent / "op_cache")
        import h5py
        with h5py.File(self.h5_path, "r") as h:
            self.expr_names = json.loads(h.attrs["expr_names"])
            self.has_acts = bool(h.attrs.get("has_activations", False))
            self.muscle_names = ([m.decode() if isinstance(m, bytes) else str(m)
                                  for m in h["muscle_names"][:]]
                                 if "muscle_names" in h else None)
            # активации выражений (M,) — общие
            self.activations = {}
            if "expressions" in h:
                for en in h["expressions"]:
                    g = h["expressions"][en]
                    if "activations" in g:
                        self.activations[en] = g["activations"][:].astype(
                            np.float32)
            # список голов
            allh = list(h["heads"].keys())
            if skip_bad:
                allh = [n for n in allh
                        if not bool(h["heads"][n].attrs.get("bad_transfer",
                                                            False))]
            self.heads = head_names if head_names is not None else allh
        self.n_muscles = (len(self.muscle_names) if self.muscle_names
                          else (len(next(iter(self.activations.values())))
                                if self.activations else 0))

    def __len__(self):
        return len(self.heads)

    def feature_dim(self):
        return feature_dim(self.use_wks, self.wks_n)

    def __getitem__(self, idx):
        import h5py
        name = self.heads[idx]
        with h5py.File(self.h5_path, "r") as h:
            g = h["heads"][name]
            verts = g["neutral"][:].astype(np.float64)
            faces = g["faces"][:].astype(np.int64)
            deltas = {en: g["expr"][en][:].astype(np.float32)
                      for en in g["expr"]}
        ops = get_operators(verts, faces, self.k_eig, cache_dir=self.cache_dir)
        feats = build_vertex_features(verts, faces, ops, self.use_wks,
                                      self.wks_n)
        # стек выражений: δ (E,N,3) + активации (E,M)
        en_list = [e for e in self.expr_names if e in deltas]
        delta_stack = np.stack([deltas[e] for e in en_list], axis=0)
        act_stack = np.stack([self.activations.get(e, np.zeros(self.n_muscles,
                                                               np.float32))
                              for e in en_list], axis=0)
        return {
            "name": name,
            "verts": torch.from_numpy(verts.astype(np.float32)),
            "faces": torch.from_numpy(faces.astype(np.int64)),
            "feats": torch.from_numpy(feats),
            "mass": torch.from_numpy(ops["mass"]),
            "evals": torch.from_numpy(ops["evals"]),
            "evecs": torch.from_numpy(ops["evecs"]),
            "gradX": _sp2torch(ops["gradX"]),
            "gradY": _sp2torch(ops["gradY"]),
            "delta": torch.from_numpy(delta_stack),     # (E, N, 3)
            "activations": torch.from_numpy(act_stack), # (E, M)
            "expr_names": en_list,
        }


def _sp2torch(m):
    """scipy.sparse → torch sparse_coo (float32)."""
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    val = torch.from_numpy(m.data.astype(np.float32))
    return torch.sparse_coo_tensor(idx, val, m.shape).coalesce()


def collate_single(batch):
    """Батч = по одной голове за раз (разная топология → не стекуем).
    Возвращаем список dict'ов; тренер итерирует по ним с grad accumulation."""
    return batch
