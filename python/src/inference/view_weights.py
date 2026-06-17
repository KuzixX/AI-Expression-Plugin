"""
Визуализатор предсказанных skinning weights на меше головы (Open3D GUI).

Слева 3D-голова, справа панель: слайдер по мышцам (вес выбранной мышцы как
тепловая карта) + кнопка argmax (каждая вершина окрашена цветом доминирующей
мышцы) + выбор головы.

  cd python
  python -m src.inference.view_weights --ckpt models/checkpoints/best.pt \
         --data data/transferred.h5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

from src.inference.predict import load_model, predict_weights   # noqa: E402


def _heat(v):
    """скаляр [0,1] → RGB (CMAP hot: black→red→yellow→white)."""
    v = np.clip(v, 0, 1)
    r = np.clip(v * 3, 0, 1)
    g = np.clip(v * 3 - 1, 0, 1)
    b = np.clip(v * 3 - 2, 0, 1)
    return np.stack([r, g, b], axis=1)


def _palette(n):
    import colorsys
    return np.array([colorsys.hsv_to_rgb(i / max(n, 1), 0.65, 0.95)
                     for i in range(n)])


class WeightViewer:
    def __init__(self, ckpt, data, k_eig, use_wks):
        self.model, self.M, self.device = load_model(ckpt, use_wks=use_wks)
        self.data = data; self.k_eig = k_eig; self.use_wks = use_wks
        import h5py
        with h5py.File(data, "r") as h:
            self.heads = list(h["heads"].keys())
        self.head_idx = 0
        self.muscle = 0
        self.mode = "muscle"                     # 'muscle' | 'argmax'
        self._load_head()
        self._build_gui()
        self._render()

    def _load_head(self):
        name = self.heads[self.head_idx]
        self.verts, self.faces, self.W = predict_weights(
            self.model, self.M, self.device, self.data, name,
            k_eig=self.k_eig, use_wks=self.use_wks)

    def _colors(self):
        if self.mode == "argmax":
            pal = _palette(self.M)
            dom = self.W.argmax(1)
            strength = self.W.max(1)
            col = pal[dom] * np.clip(strength, 0.15, 1.0)[:, None]
            return col
        w = self.W[:, self.muscle]
        w = w / max(w.max(), 1e-6)
        return _heat(w)

    def _build_gui(self):
        self.app = gui.Application.instance
        self.window = self.app.create_window("Skinning weights viewer", 1200, 800)
        w = self.window; em = w.theme.font_size
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.1, 0.1, 0.12, 1.0])
        w.add_child(self.scene)

        panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        panel.add_child(gui.Label("Weights view"))

        panel.add_child(gui.Label("head"))
        self.cb_head = gui.Combobox()
        for n in self.heads:
            self.cb_head.add_item(n)
        self.cb_head.set_on_selection_changed(self._on_head)
        panel.add_child(self.cb_head)

        self.btn_mode = gui.Button("mode: per-muscle")
        self.btn_mode.set_on_clicked(self._toggle_mode)
        panel.add_child(self.btn_mode)

        panel.add_child(gui.Label("muscle"))
        self.sl = gui.Slider(gui.Slider.INT)
        self.sl.set_limits(0, max(self.M - 1, 0))
        self.sl.set_on_value_changed(self._on_muscle)
        panel.add_child(self.sl)
        self.lbl = gui.Label(f"muscle 0 / {self.M-1}")
        panel.add_child(self.lbl)

        self.lbl_info = gui.Label("")
        panel.add_child(self.lbl_info)
        self.panel = panel
        w.add_child(panel)
        w.set_on_layout(self._layout)

    def _layout(self, ctx):
        r = self.window.content_rect
        pw = min(16 * self.window.theme.font_size, r.width * 0.25)
        self.scene.frame = gui.Rect(r.x, r.y, r.width - pw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def _render(self):
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(self.verts),
            o3d.utility.Vector3iVector(self.faces))
        mesh.compute_vertex_normals()
        mesh.vertex_colors = o3d.utility.Vector3dVector(self._colors())
        mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
        self.scene.scene.clear_geometry()
        self.scene.scene.add_geometry("head", mesh, mat)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            self.verts.min(0) - 0.05, self.verts.max(0) + 0.05)
        self.scene.setup_camera(50.0, bbox, bbox.get_center())
        # инфо: разброс весов
        w = self.W[:, self.muscle]
        self.lbl_info.text = (f"mode={self.mode}\n"
                              f"muscle {self.muscle}: "
                              f"max={w.max():.2f} mean={w.mean():.3f}\n"
                              f"active(>0.5): {(w>0.5).sum()} verts")

    def _on_head(self, text, idx):
        self.head_idx = idx; self._load_head(); self._render()

    def _toggle_mode(self):
        self.mode = "argmax" if self.mode == "muscle" else "muscle"
        self.btn_mode.text = ("mode: argmax (dominant)" if self.mode == "argmax"
                              else "mode: per-muscle")
        self._render()

    def _on_muscle(self, v):
        self.muscle = int(v)
        self.lbl.text = f"muscle {self.muscle} / {self.M-1}"
        if self.mode == "muscle":
            self._render()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="models/checkpoints/best.pt")
    ap.add_argument("--data", default="data/transferred.h5")
    ap.add_argument("--k-eig", type=int, default=128)
    ap.add_argument("--use-wks", action="store_true")
    args = ap.parse_args()
    gui.Application.instance.initialize()
    WeightViewer(args.ckpt, args.data, args.k_eig, args.use_wks)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
