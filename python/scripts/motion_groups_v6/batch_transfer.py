#!/usr/bin/env python3
"""
Батч-перенос эмоции на папку FBX-голов → один HDF5-датасет.

Для каждого FBX в папке:
  1. авто-анкоры (MediaPipe) на FLAME и FBX;
  2. FLAME с выбранной эмоцией → δ_native; FBX без δ (нейтраль);
  3. single-t диффузия → multi-t зоны → кластеры (на обеих головах);
  4. UV-перенос δ FLAME→FBX (та же логика, что в pipeline_viewer);
  5. запись в общий HDF5: нейтраль FBX (verts+faces) + перенесённый δ.

Каждая голова — своя группа в HDF5 (своя нейтраль и свой δ). Пока одна эмоция.

Headless (без GUI), вызывается из pipeline_viewer кнопкой или напрямую:
  python batch_transfer.py --fbx-dir <папка> --expr "300:8.0" [--out data/dataset.h5]
"""
import argparse
import os
from pathlib import Path

import numpy as np

import debug_head1_pipeline as pipe


def _static_diffusion(verts, faces, L, MM, srcs, total_time, steps):
    from scipy.sparse.linalg import factorized
    N = len(srcs)
    dt = total_time / max(steps, 1)
    solve = factorized((MM + dt * L).tocsc())
    A_diag = np.array(MM.diagonal())
    u = np.zeros((N, L.shape[0]))
    for ai in range(N):
        u[ai, srcs[ai]] = 1.0 / max(A_diag[srcs[ai]], 1e-12)
    for _ in range(steps):
        for ai in range(N):
            u[ai] = solve(MM @ u[ai])
    return u


def _build_head_zones(verts, faces, anchors, delta_native, p):
    """single-t диффузия → multi-t зоны (+маска) → кластеры. Возвращает dict
    с partition, n_anchors, vert_gcid (формат result_dict для переноса)."""
    faces64 = faces.astype(np.int64)
    L, MM = pipe.build_operators(verts, faces64)
    heat = _static_diffusion(verts, faces64, L, MM, anchors,
                             p['time'], p['steps'])
    enr, _ = pipe.enrich_heat_multi_t(
        verts, faces64, list(anchors),
        n_times=p['multi_t_n_times'], n_eigs=p['multi_t_n_eigs'],
        smooth_iters=5, smooth_alpha=0.5, mesh_label="batch")
    h1 = heat / heat.max(1, keepdims=True).clip(1e-12)
    active = h1.max(0) > p['heat_threshold']
    enr[:, ~active] = 0.0
    partition = pipe._argmax_partition(enr, threshold=p['heat_threshold'])
    n_anchors = len(anchors)
    # кластеры → per-vertex global cluster id
    vgid = -np.ones(len(verts), dtype=np.int64)
    vgw = np.zeros(len(verts)); gid = 0
    for a in range(n_anchors):
        masked = enr[a].copy(); masked[partition != a] = 0.0
        cls = pipe.cluster_zone(
            masked, delta_native, verts, anchor_idx=a,
            heat_threshold=p['heat_threshold'], n_clusters_max=p['n_clusters'],
            position_weight=p.get('position_weight', 0.0),
            clustering_method=p.get('clustering_method', 'kmeans'),
            similarity_threshold=0.3, print_quality=False)
        for cl in cls:
            for j, vi in enumerate(cl['indices']):
                if cl['heat_weights'][j] > vgw[vi]:
                    vgw[vi] = cl['heat_weights'][j]; vgid[vi] = gid
            gid += 1
    return {'verts': verts, 'faces': faces64, 'partition': partition,
            'n_anchors': n_anchors, 'delta_native': delta_native,
            'vert_gcid': vgid}


def _flame_with_expr(v_t, sd, shape, expr):
    """rest FLAME + δ_native эмоции (как normalized_expr в основном скрипте)."""
    v_rest_raw = pipe.apply_betas(v_t, sd, shape)
    v_rest = pipe.normalize_bbox(v_rest_raw)
    v_raw_e = pipe.apply_betas(v_t, sd, {**shape, **expr})
    m = v_rest_raw.mean(0)
    d = np.linalg.norm((v_rest_raw - m).max(0) - (v_rest_raw - m).min(0))
    head_expr = (v_raw_e - m) / (d + 1e-12)
    return v_rest, head_expr - v_rest


def batch_transfer(fbx_dir, expr_str, out_h5, flame_path=None,
                   shape_str="", landmarks=None, params=None,
                   progress=None):
    """Перенос эмоции expr на все FBX папки fbx_dir → HDF5 out_h5.
    Возвращает (n_ok, n_total)."""
    import h5py
    import auto_anchors

    flame_path = flame_path or pipe.FLAME_PKL
    p = dict(time=0.002, steps=60, heat_threshold=0.05, n_clusters=5,
             multi_t_n_times=8, multi_t_n_eigs=80, position_weight=0.0,
             clustering_method='kmeans', smooth_iters=3, smooth_alpha=0.5,
             uv_interp_delta=True, uv_world_orient=False, uv_flat=False)
    if params:
        p.update(params)
    shape = pipe.parse_betas_string(shape_str)
    expr = pipe.parse_betas_string(expr_str)

    # FLAME — общая для всех (анкоры + зоны считаем один раз)
    v_t, sd, faces_f = pipe.load_flame(flame_path)
    v_rest_f, delta_f = _flame_with_expr(v_t, sd, shape, expr)
    anch_f, dbg_f = auto_anchors.auto_anchors(
        v_rest_f, faces_f.astype(np.int64), landmark_indices=landmarks)
    if not dbg_f.get('ok') or len(anch_f) < 1:
        raise RuntimeError("MediaPipe не нашёл лицо на FLAME-голове")
    res_f = _build_head_zones(v_rest_f, faces_f, anch_f, delta_f, p)

    # перебор FBX
    exts = (".fbx", ".obj", ".ply", ".stl")
    files = sorted(f for f in Path(fbx_dir).iterdir()
                   if f.suffix.lower() in exts)
    n_total = len(files)
    Path(out_h5).parent.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with h5py.File(out_h5, "w") as h:
        h.attrs['expr'] = expr_str
        h.attrs['shape'] = shape_str
        h.attrs['n_heads'] = 0
        h.attrs['schema'] = ("per-head: verts(N,3) нейтраль, faces(K,3), "
                             "delta(N,3) перенесённая деформация эмоции")
        for fi, fpath in enumerate(files):
            name = fpath.stem
            try:
                v_raw, faces_x = pipe.load_custom_mesh(str(fpath))
                verts_x = pipe.normalize_bbox(v_raw)
                anch_x, dbg_x = auto_anchors.auto_anchors(
                    verts_x, faces_x.astype(np.int64),
                    landmark_indices=landmarks)
                if not dbg_x.get('ok') or len(anch_x) < 1:
                    raise RuntimeError("лицо не найдено MediaPipe")
                res_x = _build_head_zones(
                    verts_x, faces_x, anch_x,
                    np.zeros_like(verts_x), p)
                tr = pipe.transfer_deformations_uv(
                    res_f, res_x,
                    flat=("world" if p.get('uv_world_orient')
                          else bool(p.get('uv_flat', False))),
                    interp_delta=bool(p.get('uv_interp_delta', True)))
                if tr is None:
                    raise RuntimeError("перенос вернул None")
                delta = pipe.smooth_delta(
                    tr['delta'], faces_x.astype(np.int64),
                    n_iter=int(p.get('smooth_iters', 3)),
                    alpha=float(p.get('smooth_alpha', 0.5)))
                g = h.create_group(f"heads/{name}")
                g.create_dataset('verts', data=verts_x.astype(np.float32))
                g.create_dataset('faces', data=faces_x.astype(np.int32))
                g.create_dataset('delta', data=delta.astype(np.float32))
                g.attrs['n_verts'] = len(verts_x)
                n_ok += 1
                msg = f"[{fi+1}/{n_total}] {name}: OK ({len(verts_x)} верш.)"
            except Exception as e:
                msg = f"[{fi+1}/{n_total}] {name}: пропуск — {e}"
            print(" ", msg)
            if progress:
                progress(fi + 1, n_total, msg)
        h.attrs['n_heads'] = n_ok
    print(f"Готово: {n_ok}/{n_total} голов → {out_h5}")
    return n_ok, n_total


def main():
    ap = argparse.ArgumentParser(description="Батч-перенос эмоции → HDF5")
    ap.add_argument("--fbx-dir", required=True, help="папка с FBX-головами")
    ap.add_argument("--expr", required=True, help="betas эмоции, '300:8.0'")
    ap.add_argument("--shape", default="", help="shape betas FLAME")
    ap.add_argument("--out", default="data/dataset.h5", help="выходной HDF5")
    ap.add_argument("--flame", default=pipe.FLAME_PKL)
    ap.add_argument("--landmarks", default="",
                    help="индексы лендмарок MediaPipe через запятую")
    args = ap.parse_args()
    lm = ([int(s) for s in args.landmarks.replace(";", ",").split(",")
           if s.strip()] if args.landmarks.strip() else None)
    batch_transfer(args.fbx_dir, args.expr, args.out, flame_path=args.flame,
                   shape_str=args.shape, landmarks=lm)


if __name__ == "__main__":
    main()
