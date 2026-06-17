#!/usr/bin/env python3
"""
FLAME Expression Browser — интерактивный просмотр выражений FLAME.

Слева 3D-вид головы, справа — слайдеры коэффициентов (betas). Крутишь слайдер
и сразу видишь, как меняется мимика/форма. Так можно вживую понять, за что
отвечает каждая PCA-компонента (имён у них нет — только индексы).

Раскладка betas в FLAME 2023 (shapedirs (V,3,400)):
  • индексы   0..299 → SHAPE      (форма головы),
  • индексы 300..399 → EXPRESSION (мимика).

Запуск:
  cd /Users/kuzix/Documents/GitHub/Muscle-autoskinner
  source .venv/bin/activate
  python python/scripts/motion_groups_v6/expression_browser.py
  # опц.:  --flame <путь.pkl>  --n-expr 20  --n-shape 10  --range 6.0
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


FLAME_PKL = str(Path(__file__).resolve().parents[3] / "Assets/Meshes/FLAME/"
                "FLAME2023 Open for commercial use/flame2023_Open.pkl")

SHAPE_START = 0
EXPR_START = 300            # в FLAME 2023 экспрессии начинаются с индекса 300


def load_flame(path):
    """FLAME .pkl → (v_template (V,3), shapedirs (V,3,K), faces (F,3))."""
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")

    def to_np(x):
        if hasattr(x, "r"):
            return np.array(x.r)
        if hasattr(x, "toarray"):
            return np.array(x.toarray())
        return np.array(x)

    return (to_np(d["v_template"]).astype(np.float64),
            to_np(d["shapedirs"]).astype(np.float64),
            to_np(d["f"]).astype(np.int64))


class ExpressionBrowser:
    def __init__(self, v_template, shapedirs, faces,
                 n_expr=20, n_shape=10, slider_range=6.0):
        self.v_t = v_template
        self.sd = shapedirs
        self.faces = faces.astype(np.int32)
        self.K = shapedirs.shape[2]
        self.slider_range = float(slider_range)

        # какие компоненты показываем слайдерами
        self.shape_ids = list(range(SHAPE_START,
                                    min(SHAPE_START + n_shape, EXPR_START)))
        self.expr_ids = list(range(EXPR_START,
                                   min(EXPR_START + n_expr, self.K)))
        self.all_ids = self.shape_ids + self.expr_ids
        self.betas = {i: 0.0 for i in self.all_ids}
        self.sliders = {}
        self.value_labels = {}

        self._build_gui()
        self._update_mesh()

    # ── geometry ──────────────────────────────────────────────────────────
    def _current_vertices(self):
        v = self.v_t.copy()
        active = [(i, b) for i, b in self.betas.items() if abs(b) > 1e-9]
        for i, b in active:
            v += b * self.sd[:, :, i]
        return v

    def _make_mesh(self, verts):
        m = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(verts),
            o3d.utility.Vector3iVector(self.faces))
        m.compute_vertex_normals()
        m.paint_uniform_color([0.82, 0.74, 0.68])
        return m

    def _update_mesh(self):
        verts = self._current_vertices()
        mesh = self._make_mesh(verts)
        mat = rendering.MaterialRecord()
        mat.shader = "defaultLit"
        self.scene.scene.remove_geometry("head")
        self.scene.scene.add_geometry("head", mesh, mat)

    # ── GUI ───────────────────────────────────────────────────────────────
    def _build_gui(self):
        self.window = gui.Application.instance.create_window(
            "FLAME Expression Browser", 1280, 860)
        w = self.window
        em = w.theme.font_size

        # 3D-сцена
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(w.renderer)
        self.scene.scene.set_background([0.1, 0.1, 0.12, 1.0])
        # камера на bbox головы
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            self.v_t.min(0) - 0.05, self.v_t.max(0) + 0.05)
        self.scene.setup_camera(50.0, bbox, bbox.get_center())

        # правая панель со слайдерами (скроллируемая)
        panel = gui.ScrollableVert(0.5 * em, gui.Margins(em, em, em, em))

        panel.add_child(gui.Label("FLAME betas — крути и смотри"))
        panel.add_child(gui.Label(f"диапазон ±{self.slider_range:.0f}"))

        def add_section(title, ids, tag):
            panel.add_fixed(0.5 * em)
            lbl = gui.Label(f"── {title} ──")
            panel.add_child(lbl)
            for i in ids:
                row = gui.Horiz(0.25 * em)
                name = gui.Label(f"{tag}{i - (EXPR_START if tag=='E' else 0):>3} "
                                 f"[{i}]")
                name.text = self._slider_caption(tag, i)
                s = gui.Slider(gui.Slider.DOUBLE)
                s.set_limits(-self.slider_range, self.slider_range)
                s.double_value = 0.0
                s.set_on_value_changed(self._make_cb(i))
                vlab = gui.Label("  0.0")
                self.sliders[i] = s
                self.value_labels[i] = vlab
                row.add_child(name)
                row.add_child(s)
                row.add_child(vlab)
                panel.add_child(row)

        if self.expr_ids:
            add_section("EXPRESSION (мимика)", self.expr_ids, "E")
        if self.shape_ids:
            add_section("SHAPE (форма)", self.shape_ids, "S")

        # кнопки
        panel.add_fixed(em)
        btn_reset = gui.Button("Сброс всех betas")
        btn_reset.set_on_clicked(self._on_reset)
        panel.add_child(btn_reset)

        btn_export = gui.Button("Экспорт текущей головы → OBJ")
        btn_export.set_on_clicked(self._on_export)
        panel.add_child(btn_export)

        btn_eye = gui.Button("Найти компоненты ЗАКРЫТИЯ ГЛАЗ")
        btn_eye.set_on_clicked(self._on_find_eye)
        panel.add_child(btn_eye)
        self.eye_label = gui.Label("")
        panel.add_child(self.eye_label)

        self.betas_label = gui.Label("активные betas: {}")
        panel.add_fixed(0.5 * em)
        panel.add_child(self.betas_label)

        w.add_child(self.scene)
        w.add_child(panel)
        self.panel = panel
        w.set_on_layout(self._on_layout)

    def _slider_caption(self, tag, i):
        if tag == "E":
            return f"E{i - EXPR_START:>2} [{i}]"
        return f"S{i:>2} [{i}]"

    def _on_layout(self, ctx):
        r = self.window.content_rect
        panel_w = min(22 * self.window.theme.font_size, r.width * 0.42)
        self.scene.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_w, r.y,
                                    panel_w, r.height)

    # ── callbacks ─────────────────────────────────────────────────────────
    def _make_cb(self, i):
        def cb(value):
            self.betas[i] = float(value)
            self.value_labels[i].text = f"{value:5.1f}"
            self._update_mesh()
            self._refresh_betas_label()
        return cb

    def _refresh_betas_label(self):
        active = {i: round(b, 1) for i, b in self.betas.items()
                  if abs(b) > 1e-9}
        self.betas_label.text = "активные: " + (str(active) if active else "{}")

    def _eye_region_mask(self):
        """Маска вершин зоны ВЕК (узкая полоса вокруг глаз): Y∈[0.58,0.74]
        высоты лица, |X| смещён от центра (глаза не на оси), Z — перёд."""
        v = self.v_t
        mn = v.min(0); mx = v.max(0); span = mx - mn
        yn = (v[:, 1] - mn[1]) / (span[1] + 1e-9)      # 0 низ → 1 верх
        cx = (mn[0] + mx[0]) / 2.0
        xw = abs(v[:, 0] - cx) / (span[0] / 2 + 1e-9)  # 0 центр → 1 край
        zn = (v[:, 2] - mn[2]) / (span[2] + 1e-9)      # 0 зад → 1 перёд
        # веки: на высоте глаз, СМЕЩЕНЫ от центра (0.12..0.5), спереди
        mask = ((yn > 0.58) & (yn < 0.74) & (xw > 0.10) & (xw < 0.50)
                & (zn > 0.60))
        return mask

    def _on_find_eye(self):
        """Ранжируем expr-компоненты по ОТНОСИТЕЛЬНОМУ движению зоны век:
        (движение век) / (движение всего лица). Так находим компоненты, которые
        бьют ИМЕННО по векам, а не двигают всё лицо (иначе в топе всегда первые
        PCA-оси). Закрытие глаз = веки сильно двигаются по Y."""
        mask = self._eye_region_mask()
        if int(mask.sum()) < 10:
            self.eye_label.text = "зона век не найдена"
            return
        # «лицо» = передняя половина (для нормировки)
        v = self.v_t
        zn = (v[:, 2] - v[:, 2].min()) / (np.ptp(v[:, 2]) + 1e-9)
        face = zn > 0.55
        scores = []
        for i in self.expr_ids:
            d = self.sd[:, :, i]
            eye_mag = float(np.linalg.norm(d[mask], axis=1).mean())
            face_mag = float(np.linalg.norm(d[face], axis=1).mean()) + 1e-9
            vert_frac = float(np.abs(d[mask, 1]).mean()
                              / (np.linalg.norm(d[mask], axis=1).mean() + 1e-9))
            # сконцентрированность на веках × вертикальность движения
            scores.append((i, (eye_mag / face_mag) * vert_frac, eye_mag))
        scores.sort(key=lambda s: -s[1])
        top = scores[:6]
        txt = "Компоненты ЗАКРЫТИЯ ГЛАЗ (по векам):\n" + "\n".join(
            f"  E{i - EXPR_START} [{i}]: score={sc:.3f}, mag={mg:.4f}"
            for i, sc, mg in top)
        print(txt)
        self.eye_label.text = txt
        # доп.: подсветим зону век на меше (покрасим вершины) для наглядности
        verts = self._current_vertices()
        mesh = self._make_mesh(verts)
        col = np.tile([0.82, 0.74, 0.68], (len(verts), 1))
        col[mask] = [0.1, 0.8, 0.2]
        mesh.vertex_colors = o3d.utility.Vector3dVector(col)
        mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
        self.scene.scene.remove_geometry("head")
        self.scene.scene.add_geometry("head", mesh, mat)

    def _on_reset(self):
        for i in self.all_ids:
            self.betas[i] = 0.0
            self.sliders[i].double_value = 0.0
            self.value_labels[i].text = "  0.0"
        self._update_mesh()
        self._refresh_betas_label()

    def _on_export(self):
        verts = self._current_vertices()
        mesh = self._make_mesh(verts)
        out = Path("expression_browser_export.obj").resolve()
        o3d.io.write_triangle_mesh(str(out), mesh)
        active = {i: round(b, 2) for i, b in self.betas.items()
                  if abs(b) > 1e-9}
        print(f"Сохранено: {out}")
        print(f"  betas: {active}")


def _setup_cyrillic_font():
    """Подключаем системный шрифт с кириллицей (иначе Open3D рисует '?')."""
    import os
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    path = next((p for p in candidates if os.path.exists(p)), None)
    if path is None:
        return
    fd = gui.FontDescription(path)
    try:
        fd.add_typeface_for_language(path, "ru")
    except Exception:
        pass
    gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, fd)


def main():
    ap = argparse.ArgumentParser(description="FLAME Expression Browser")
    ap.add_argument("--flame", default=FLAME_PKL, help="путь к flame*.pkl")
    ap.add_argument("--n-expr", type=int, default=20,
                    help="сколько expression-компонент показать (с 300)")
    ap.add_argument("--n-shape", type=int, default=10,
                    help="сколько shape-компонент показать (с 0)")
    ap.add_argument("--range", type=float, default=6.0,
                    help="диапазон слайдера ±range")
    args = ap.parse_args()

    if not Path(args.flame).exists():
        raise SystemExit(f"FLAME pkl не найден: {args.flame}\n"
                         f"Запускай из корня репо или укажи --flame.")

    print(f"Загружаю FLAME: {args.flame}")
    v_t, sd, faces = load_flame(args.flame)
    print(f"  вершин={len(v_t)}, компонент betas={sd.shape[2]} "
          f"(shape 0..299, expr 300..{sd.shape[2]-1})")
    print(f"  слайдеры: expr {args.n_expr} шт., shape {args.n_shape} шт.")
    print("  Крути слайдеры справа. Кнопки: сброс / экспорт в OBJ.")

    gui.Application.instance.initialize()
    _setup_cyrillic_font()
    ExpressionBrowser(v_t, sd, faces,
                      n_expr=args.n_expr, n_shape=args.n_shape,
                      slider_range=args.range)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
