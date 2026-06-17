#!/usr/bin/env python3
"""
Чтение датасета батч-переноса (data/dataset.h5).

Печатает сводку: эмоция, число голов, по каждой голове — размеры и статистику δ.
Опционально открывает одну голову (нейтраль + деформированную) в Open3D.

  python read_dataset.py [data/dataset.h5]
  python read_dataset.py data/dataset.h5 --show head_0000   # показать в 3D
"""
import argparse

import numpy as np


def summarize(path):
    import h5py
    with h5py.File(path, "r") as h:
        print(f"Файл: {path}")
        print("Атрибуты:")
        for k, v in h.attrs.items():
            print(f"  {k}: {v}")
        heads = list(h["heads"].keys()) if "heads" in h else []
        print(f"\nГолов: {len(heads)}")
        for name in heads:
            g = h["heads"][name]
            verts = g["verts"]
            delta = g["delta"][:]
            mag = np.linalg.norm(delta, axis=1)
            print(f"  {name}: verts{verts.shape} faces{g['faces'].shape} "
                  f"δ: max={mag.max():.4f} mean={mag.mean():.4f} "
                  f"moved={int((mag > 1e-5).sum())} верш.")


def load_head(path, name):
    """Возвращает (rest verts (N,3), faces (K,3), delta (N,3))."""
    import h5py
    with h5py.File(path, "r") as h:
        g = h["heads"][name]
        return g["verts"][:], g["faces"][:], g["delta"][:]


def show_head(path, name):
    import open3d as o3d
    verts, faces, delta = load_head(path, name)
    deformed = verts + delta
    mag = np.linalg.norm(delta, axis=1)
    col = mag / max(mag.max(), 1e-9)
    colors = np.column_stack([col, np.zeros_like(col), 1 - col])  # синий→красный

    def mk(V, c):
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V.astype(np.float64)),
            o3d.utility.Vector3iVector(faces.astype(np.int32)))
        m.compute_vertex_normals()
        if c is None:
            m.paint_uniform_color([0.8, 0.78, 0.74])
        else:
            m.vertex_colors = o3d.utility.Vector3dVector(c)
        return m

    rest = mk(verts, None)
    defo = mk(deformed, colors)
    defo.translate([1.2 * (verts.max(0)[0] - verts.min(0)[0]), 0, 0])
    o3d.visualization.draw_geometries(
        [rest, defo], window_name=f"{name}: нейтраль | деформ. (δ цветом)",
        width=1100, height=800, mesh_show_back_face=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="data/dataset.h5")
    ap.add_argument("--show", default="", help="имя головы для 3D-просмотра")
    args = ap.parse_args()
    summarize(args.path)
    if args.show:
        show_head(args.path, args.show)


if __name__ == "__main__":
    main()
