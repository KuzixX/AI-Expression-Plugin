#!/usr/bin/env python3
"""
Mesh Viewer — простой просмотр одного меша (FBX/OBJ/PLY) в окне Open3D.

Запускается как подпроцесс (legacy draw_geometries нельзя смешивать с tkinter
в одном процессе). FBX грузится через assimp (obj-посредник).

  python mesh_viewer.py <путь к мешу> [<файл .npy с центрами сфер (N,3)>]
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import open3d as o3d


def load_mesh(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".obj", ".ply", ".stl", ".off"):
        return o3d.io.read_triangle_mesh(path)
    # FBX и прочее — через assimp в obj
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    tmp.close()
    r = subprocess.run(["assimp", "export", path, tmp.name],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"assimp: {r.stderr}")
    m = o3d.io.read_triangle_mesh(tmp.name)
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    return m


def main():
    if len(sys.argv) < 2:
        print("Использование: mesh_viewer.py <путь к мешу>")
        return
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f"Не найден: {path}")
        return
    m = load_mesh(path)
    if len(np.asarray(m.vertices)) == 0:
        print("Пустой меш / не удалось загрузить.")
        return
    m.compute_vertex_normals()
    if not m.has_vertex_colors():            # серым только если цветов нет
        m.paint_uniform_color([0.82, 0.74, 0.68])
    geoms = [m]
    # опц. второй аргумент — .npy с центрами WKS-групп → маленькие жёлтые сферы
    if len(sys.argv) >= 3 and os.path.exists(sys.argv[2]):
        try:
            centers = np.load(sys.argv[2])
            diag = float(np.linalg.norm(m.get_max_bound() - m.get_min_bound()))
            r = max(0.012 * diag, 1e-4)
            for c in np.asarray(centers, np.float64).reshape(-1, 3):
                s = o3d.geometry.TriangleMesh.create_sphere(radius=r)
                s.translate(c); s.paint_uniform_color([1.0, 0.85, 0.0])
                s.compute_vertex_normals()
                geoms.append(s)
            print(f"WKS-сфер: {len(geoms) - 1}")
        except Exception as e:
            print(f"sphere load error: {e}")
    o3d.visualization.draw_geometries(
        geoms, window_name=f"Mesh: {os.path.basename(path)}",
        width=1000, height=800, mesh_show_back_face=True)


if __name__ == "__main__":
    main()
