#!/usr/bin/env python3
"""
Создать reference-HDF5 (нейтраль + выражения) из FLAME expr-betas.

Каждое выражение = набор expr-betas (индексы 300..399). Пишем нейтраль и δ
каждого выражения в каноническую схему transfer_engine.

  python make_reference.py --out data/reference.h5 \
      --shape "" --expr "smile=308:8" --expr "brows=310:6,311:-4"
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_V6 = Path(__file__).resolve().parent.parent / "motion_groups_v6"
if str(_V6) not in sys.path:
    sys.path.insert(0, str(_V6))
import debug_head1_pipeline as pipe          # noqa: E402


def build_reference(out_h5, shape_str="", expr_specs=None, flame_path=None,
                    activations=None, muscle_names=None):
    """expr_specs: список "имя=300:8,302:-5". Пишет reference HDF5.

    activations: {имя_выраж: вектор (M,)} — активации мышц рига, которые дают
    это выражение (n-мерный вектор, общий для всех голов). muscle_names: имена
    мышц (порядок). Если заданы — пишутся в /expressions/<имя>/activations и
    /muscle_names (как ждёт transfer_engine.read_reference)."""
    import h5py
    flame_path = flame_path or pipe.FLAME_PKL
    v_t, sd, faces = pipe.load_flame(flame_path)
    shape = pipe.parse_betas_string(shape_str)
    v_rest_raw = pipe.apply_betas(v_t, sd, shape)
    neutral = pipe.normalize_bbox(v_rest_raw)
    m = v_rest_raw.mean(0)
    diag = np.linalg.norm((v_rest_raw - m).max(0) - (v_rest_raw - m).min(0))

    Path(out_h5).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as h:
        ng = h.create_group("neutral")
        ng.create_dataset("verts", data=neutral.astype(np.float32))
        ng.create_dataset("faces", data=faces.astype(np.int32))
        if muscle_names is not None:
            dt = h5py.string_dtype(encoding="utf-8")
            h.create_dataset("muscle_names",
                             data=np.array(muscle_names, dtype=object),
                             dtype=dt)
        eg = h.create_group("expressions")
        for spec in (expr_specs or []):
            if "=" in spec:
                name, betas = spec.split("=", 1)
            else:
                name, betas = spec, spec
            name = name.strip()
            expr = pipe.parse_betas_string(betas)
            v_e = pipe.apply_betas(v_t, sd, {**shape, **expr})
            head_expr = (v_e - m) / (diag + 1e-12)
            delta = head_expr - neutral
            g = eg.create_group(name)
            g.create_dataset("delta", data=delta.astype(np.float32))
            g.attrs["betas"] = betas
            if activations and name in activations:
                g.create_dataset("activations",
                                 data=np.asarray(activations[name],
                                                 dtype=np.float32))
        h.attrs["n_expr"] = len(expr_specs or [])
    print(f"reference → {out_h5}: {len(expr_specs or [])} выражений"
          + (" (+активации)" if activations else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/reference.h5")
    ap.add_argument("--shape", default="")
    ap.add_argument("--expr", action="append", default=[],
                    help="имя=betas, напр. smile=308:8")
    ap.add_argument("--flame", default=pipe.FLAME_PKL)
    args = ap.parse_args()
    build_reference(args.out, args.shape, args.expr, args.flame)


if __name__ == "__main__":
    main()
