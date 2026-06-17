#!/usr/bin/env python3
"""
Pipeline Viewer — интерактивный прогон всего пайплайна в одном окне.

Две головы рядом (FLAME + FBX). Справа — настройки и кнопка «Вперёд ▶», которая
по шагам проводит обе головы через стадии пайплайна, каждый раз перекрашивая:

  0. Выбор точек тепла (Shift+click): сначала на FLAME, потом на FBX
     (одинаковое число anchor'ов на каждой).
  1. Single-t диффузия (статично, без анимации) — суммарное тепло.
  2. Multi-t enrichment + argmax-зоны + маскировка по single-t reach.
  3. Кластеры (motion на FLAME, heat+позиция на FBX).
  4. Перенос выражения FLAME→FBX (UV-NN / барицентрика) — FBX деформируется.

Запуск (обычно кнопкой в основном GUI):
  cd /Users/kuzix/Documents/GitHub/Muscle-autoskinner
  source .venv/bin/activate
  python python/scripts/motion_groups_v6/pipeline_viewer.py \
      [--flame <pkl>] [--fbx <mesh>] [--expr "300:8.0,302:-5.0"]
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

import debug_head1_pipeline as pipe


STEPS = ["0 · выбор точек тепла",
         "1 · применить эмоцию",
         "2 · single-t диффузия",
         "3 · зоны (multi-t + маска)",
         "4 · кластеры",
         "5 · перенос выражения"]


def _setup_cyrillic_font():
    import os
    for p in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
              "/Library/Fonts/Arial Unicode.ttf",
              "/System/Library/Fonts/Supplemental/Arial.ttf",
              "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(p):
            fd = gui.FontDescription(p)
            try:
                fd.add_typeface_for_language(p, "ru")
            except Exception:
                pass
            gui.Application.instance.set_font(gui.Application.DEFAULT_FONT_ID, fd)
            return


def static_diffusion(verts, faces, L, MM, srcs, total_time, steps):
    """Single-t тепло БЕЗ анимации: тот же неявный шаг, что в animate_diffusion,
    но без отрисовки. Возвращает heat (K, N)."""
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


class Head:
    def __init__(self, name, verts, faces, delta_native=None, flame_model=None):
        self.name = name
        self.verts = np.ascontiguousarray(verts, np.float64)
        self.faces = np.ascontiguousarray(faces, np.int32)
        # оригиналы для восстановления после кропа (Сброс)
        self.verts0 = self.verts.copy()
        self.faces0 = self.faces.copy()
        self.delta_native = (np.zeros_like(self.verts) if delta_native is None
                             else np.asarray(delta_native, np.float64))
        self.has_deform = delta_native is not None
        # FLAME-модель (v_t, sd, shape) — для применения эмоции на лету
        self.flame_model = flame_model
        self.anchors = []
        self.heat = None
        self.partition = None
        self.n_anchors = 0
        self.vert_gcid = None
        self.colors = None          # текущая раскраска
        self.deformed = None        # перенесённая деформация (для FBX)
        self._L = self._MM = None
        self._tree = None

    def ops(self):
        if self._L is None:
            self._L, self._MM = pipe.build_operators(
                self.verts, self.faces.astype(np.int64))
        return self._L, self._MM

    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.verts)
        return self._tree

    def apply_expression(self, expr_dict):
        """Пересчитать delta_native для эмоции expr_dict (только если есть FLAME-
        модель). Деформация считается как в основном скрипте (normalized_expr).
        Возвращает True, если применено."""
        if self.flame_model is None:
            return False
        v_t, sd, shape = self.flame_model
        v_rest_raw = pipe.apply_betas(v_t, sd, shape)
        v_raw_e = pipe.apply_betas(v_t, sd, {**shape, **expr_dict})
        m = v_rest_raw.mean(0)
        d = np.linalg.norm((v_rest_raw - m).max(0) - (v_rest_raw - m).min(0))
        head_expr = (v_raw_e - m) / (d + 1e-12)
        self.delta_native = head_expr - self.verts
        self.has_deform = bool(expr_dict)
        return True

    def result_dict(self):
        """Совместимо с transfer_deformations_uv."""
        return {
            'label': self.name, 'verts': self.verts,
            'faces': self.faces.astype(np.int64),
            'partition': self.partition, 'n_anchors': self.n_anchors,
            'delta_native': self.delta_native, 'vert_gcid': self.vert_gcid,
        }


class PipelineViewer:
    def __init__(self, flame, fbxs, params):
        self.p = params
        # heads[0] = FLAME (референс), heads[1:] = источники FBX (1..3)
        self.heads = [flame] + list(fbxs)
        self.n_src = len(self.heads) - 1         # число источников FBX
        self.step = 0
        self._last_transfer = None
        self._raw_delta = {}                     # {src_index: сырой δ} для слайд.
        self._build_gui()
        for i in range(len(self.heads)):
            self._render(i)
        self._set_status()

    # ── GUI ──
    def _build_gui(self):
        self.app = gui.Application.instance
        self.window = self.app.create_window("Pipeline Viewer", 1500, 900)
        w = self.window
        em = w.theme.font_size

        # каркас-флаги: по сцене на голову + последний для UV
        self.wire = [False] * (len(self.heads) + 1)
        self.uv_wire_idx = len(self.heads)       # индекс UV в self.wire
        self.scenes = []
        for i in range(len(self.heads)):
            sc = gui.SceneWidget()
            sc.scene = rendering.Open3DScene(w.renderer)
            sc.scene.set_background([0.1, 0.1, 0.12, 1.0])
            sc.set_on_mouse(self._mouse_cb(i))
            sc.set_on_key(self._key_cb(i))       # W → каркас в этой сцене
            self.scenes.append(sc)
            w.add_child(sc)

        # нижняя сцена — UV-развёртка зон (как в основном скрипте)
        self.uv_scene = gui.SceneWidget()
        self.uv_scene.scene = rendering.Open3DScene(w.renderer)
        self.uv_scene.scene.set_background([0.12, 0.12, 0.14, 1.0])
        self.uv_scene.set_on_key(self._key_cb(self.uv_wire_idx))
        w.add_child(self.uv_scene)
        self._uv_packed = None                   # кэш UV-геометрии для перерис.

        panel = gui.Vert(0.4 * em, gui.Margins(em, em, em, em))
        panel.add_child(gui.Label("Конфигурация"))

        def num(label, val, integer=True):
            panel.add_child(gui.Label(label))
            e = gui.NumberEdit(gui.NumberEdit.INT if integer
                               else gui.NumberEdit.DOUBLE)
            if integer:
                e.int_value = int(val)
            else:
                e.double_value = float(val)
            panel.add_child(e)
            return e

        self.in_time = num("diffusion t", self.p['time'], integer=False)
        self.in_steps = num("steps", self.p['steps'])
        self.in_heat_th = num("heat threshold", self.p['heat_threshold'],
                              integer=False)
        self.in_nclu = num("max кластеров", self.p['n_clusters'])
        self.in_mt_times = num("multi-t n_times", self.p['multi_t_n_times'])
        self.in_mt_eigs = num("multi-t k_eigs", self.p['multi_t_n_eigs'])

        panel.add_child(gui.Label("expr betas (idx:val,..)"))
        self.in_expr = gui.TextEdit()
        self.in_expr.text_value = self.p.get('expr_str', '')
        panel.add_child(self.in_expr)

        self.cb_crop = gui.Checkbox(
            "Обрезать меш по активной зоне single-t → bbox")
        self.cb_crop.checked = False
        panel.add_child(self.cb_crop)

        panel.add_fixed(0.5 * em)
        panel.add_child(gui.Label("MediaPipe лендмарки (индексы, через запятую;\n"
                                  "пусто = набор по умолчанию)"))
        self.in_landmarks = gui.TextEdit()
        try:
            import auto_anchors as _aa
            self.in_landmarks.text_value = ",".join(
                str(i) for i in _aa.DEFAULT_LANDMARKS)
        except Exception:
            self.in_landmarks.text_value = ""
        panel.add_child(self.in_landmarks)
        btn_auto = gui.Button("Авто-анкоры (MediaPipe)")
        btn_auto.set_on_clicked(self._on_auto_anchors)
        panel.add_child(btn_auto)
        self.lbl_auto = gui.Label("ставит анкоры по лендмаркам лица")
        panel.add_child(self.lbl_auto)
        btn_autorun = gui.Button("⚡ Авто-перенос (весь пайплайн)")
        btn_autorun.set_on_clicked(self._on_auto_run)
        panel.add_child(btn_autorun)
        self.btn_next = gui.Button("Вперёд ▶")
        self.btn_next.set_on_clicked(self._on_next)
        panel.add_child(self.btn_next)
        btn_reset = gui.Button("Сброс")
        btn_reset.set_on_clicked(self._on_reset)
        panel.add_child(btn_reset)

        # ── пост-обработка перенесённой деформации (после шага 5) ──
        panel.add_fixed(em)
        panel.add_child(gui.Label("── Перенос: пост-обработка ──"))
        panel.add_child(gui.Label("усиление деформации"))
        self.sl_gain = gui.Slider(gui.Slider.DOUBLE)
        self.sl_gain.set_limits(0.0, 3.0)
        self.sl_gain.double_value = 1.0
        self.sl_gain.set_on_value_changed(lambda v: self._apply_transfer_delta())
        panel.add_child(self.sl_gain)
        panel.add_child(gui.Label("Laplacian smooth (iters)"))
        self.sl_smooth = gui.Slider(gui.Slider.INT)
        self.sl_smooth.set_limits(0, 50)
        self.sl_smooth.int_value = int(self.p.get('smooth_iters', 3))
        self.sl_smooth.set_on_value_changed(lambda v: self._apply_transfer_delta())
        panel.add_child(self.sl_smooth)
        self.lbl_gain = gui.Label("крути ползунки — обновляется вживую")
        panel.add_child(self.lbl_gain)
        btn_save = gui.Button("💾 Сохранить деформацию (FBX + HDF5)")
        btn_save.set_on_clicked(self._on_save_deformation)
        panel.add_child(btn_save)
        self.lbl_save = gui.Label("сохраняет деформ. меш + таблицу δ")
        panel.add_child(self.lbl_save)

        # ── релаксация UV-островов (после шага «зоны») ──
        panel.add_fixed(em)
        panel.add_child(gui.Label("── Relax UV-островов ──"))
        self.combo_relax = gui.Combobox()
        self.combo_relax.add_item("ARAP (минимум искажений)")
        self.combo_relax.add_item("Laplacian (граница фикс.)")
        self.combo_relax.add_item("Spring (изометрия рёбер)")
        panel.add_child(self.combo_relax)
        panel.add_child(gui.Label("итераций relax"))
        self.sl_relax_it = gui.Slider(gui.Slider.INT)
        self.sl_relax_it.set_limits(1, 50)
        self.sl_relax_it.int_value = 10
        panel.add_child(self.sl_relax_it)
        btn_relax = gui.Button("Relax")
        btn_relax.set_on_clicked(self._on_relax)
        panel.add_child(btn_relax)
        self.lbl_relax = gui.Label("(доступно после шага «зоны»)")
        panel.add_child(self.lbl_relax)

        panel.add_fixed(em)
        panel.add_child(gui.Label("── Статус ──"))
        self.lbl_status = gui.Label("")
        panel.add_child(self.lbl_status)
        panel.add_child(gui.Label("Shift+click:\nточки тепла (FLAME, потом FBX)"))
        panel.add_child(gui.Label("W: каркас в активном вьювере"))
        self.lbl_pick = gui.Label("")
        panel.add_child(self.lbl_pick)

        self.panel = panel
        w.add_child(panel)
        w.set_on_layout(self._layout)

    def _layout(self, ctx):
        r = self.window.content_rect
        pw = min(20 * self.window.theme.font_size, r.width * 0.30)
        uv_h = r.height * 0.28                    # нижняя полоса под UV
        top_h = r.height - uv_h
        area_w = r.width - pw                     # ширина под сцены (без панели)
        n = len(self.scenes)
        # сетка ячеек: до 2 голов — в ряд, 3-4 — 2×2
        cols = 1 if n == 1 else 2
        rows = int(np.ceil(n / cols))
        cw = area_w / cols
        ch = top_h / rows
        for i, sc in enumerate(self.scenes):
            cx = i % cols; cy = i // cols
            sc.frame = gui.Rect(int(r.x + cx * cw), int(r.y + cy * ch),
                                int(cw), int(ch))
        self.uv_scene.frame = gui.Rect(r.x, int(r.y + top_h),
                                       int(area_w), int(uv_h))
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def _remove_if_exists(self, scene, name):
        try:
            if scene.has_geometry(name):
                scene.remove_geometry(name)
        except Exception:
            pass

    def _update_head_mesh(self, i):
        """Лёгкое обновление ТОЛЬКО меша головы i (без clear_geometry и без
        пересоздания всей сцены/маркеров/камеры) — безопасно для частых вызовов
        на macOS/Metal."""
        hd = self.heads[i]
        sc = self.scenes[i]
        V = hd.deformed if hd.deformed is not None else hd.verts
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V),
            o3d.utility.Vector3iVector(hd.faces))
        mesh.compute_vertex_normals()
        if hd.colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(hd.colors)
        else:
            mesh.paint_uniform_color([0.8, 0.75, 0.7])
        mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
        self._remove_if_exists(sc.scene, f"h{i}")
        sc.scene.add_geometry(f"h{i}", mesh, mat)

    # ── рендер ──
    def _render(self, i, keep_camera=False):
        hd = self.heads[i]
        sc = self.scenes[i]
        V = hd.deformed if hd.deformed is not None else hd.verts
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(V),
            o3d.utility.Vector3iVector(hd.faces))
        mesh.compute_vertex_normals()
        if hd.colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(hd.colors)
        else:
            mesh.paint_uniform_color([0.8, 0.75, 0.7])
        mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
        sc.scene.clear_geometry()
        sc.scene.add_geometry(f"h{i}", mesh, mat)
        if self.wire[i]:                         # каркас поверх (клавиша W)
            wf = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            wf.paint_uniform_color([0.0, 0.0, 0.0])
            lm = rendering.MaterialRecord(); lm.shader = "unlitLine"
            lm.line_width = 1.0
            sc.scene.add_geometry(f"wire{i}", wf, lm)
        # маркеры anchor'ов
        for a in hd.anchors:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sph.translate(hd.verts[a]); sph.paint_uniform_color([1, 0, 0])
            sph.compute_vertex_normals()
            m2 = rendering.MaterialRecord(); m2.shader = "defaultLit"
            sc.scene.add_geometry(f"a{i}_{a}", sph, m2)
        if not keep_camera:                      # камеру не дёргаем при слайдере
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                hd.verts.min(0) - 0.05, hd.verts.max(0) + 0.05)
            sc.setup_camera(50.0, bbox, bbox.get_center())

    def _set_status(self):
        self.lbl_status.text = f"шаг: {STEPS[self.step]}"
        self.lbl_pick.text = "\n".join(
            f"{hd.name}: {len(hd.anchors)} анкор" for hd in self.heads)

    # ── UV-развёртка зон (нижняя сцена) ──
    def _render_uv(self, transfers=None):
        """Строим UV-острова зон FLAME + всех источников FBX (родная индексация
        зон, zmap отдельно по каждому источнику). transfers: список (по src) с
        результатом переноса или None. Требует partition на всех головах."""
        flame = self.heads[0]
        if flame.partition is None:
            return
        flat = pipe._uv_mode(self.p)
        zd_f = pipe.compute_zone_islands(
            flame.verts, flame.faces.astype(np.int64),
            flame.partition, flame.n_anchors, flat=flat)
        if not zd_f:
            return
        if transfers is None:
            transfers = [None] * self.n_src
        src_list = []
        for k in range(1, len(self.heads)):
            hd = self.heads[k]
            if hd.partition is None:
                src_list.append(None); continue
            zd_x = pipe.compute_zone_islands(
                hd.verts, hd.faces.astype(np.int64),
                hd.partition, hd.n_anchors, flat=flat)
            if not zd_x:
                src_list.append(None); continue
            zmap = pipe.match_zones_by_position(
                flame.result_dict(), hd.result_dict())
            src_list.append({'zd': zd_x, 'zmap': zmap,
                             'transfer': transfers[k - 1]})
        self._uv_zd = {'flame': zd_f, 'na': flame.n_anchors, 'src': src_list}
        self._draw_uv_panels(keep_camera=False)

    def _draw_uv_panels(self, keep_camera=True):
        """Рисуем UV-панели из кэша: FLAME, затем по каждому источнику —
        зоны FBX и (если есть transfer) перенесённые группы."""
        if not getattr(self, "_uv_zd", None):
            return
        zd_f = self._uv_zd['flame']
        na = self._uv_zd['na']
        self.uv_scene.scene.clear_geometry()
        mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"
        lmat = rendering.MaterialRecord(); lmat.shader = "unlitLine"
        lmat.line_width = 1.0
        pal = pipe.make_cluster_palette(max(na, 1))
        GREY = np.array([0.6, 0.6, 0.6])

        panels = [("FLAME", pipe._pack_islands_to_grid(zd_f, na, pal))]
        for k, src in enumerate(self._uv_zd['src']):
            if src is None:
                continue
            zmap = src['zmap']
            zd_x_m = {zmap[a]: isl for a, isl in src['zd'].items() if a in zmap}
            panels.append((f"FBX{k+1}", pipe._pack_islands_to_grid(
                zd_x_m, na, pal)))
            tr = src['transfer']
            if tr is not None and 'gcid' in tr:
                gcid = np.asarray(tr['gcid'])
                ng = int(gcid.max()) + 1 if gcid.max() >= 0 else 1
                palg = pipe.make_cluster_palette(max(ng, 1))
                cdict = {}
                for a in zd_x_m:
                    gidx = zd_x_m[a][2]; g = gcid[gidx]
                    cdict[a] = np.where((g >= 0)[:, None],
                                        palg[np.clip(g, 0, ng - 1)], GREY)
                panels.append((f"перенос{k+1}",
                               pipe._pack_islands_to_grid(zd_x_m, na, cdict)))

        gap = 1.25
        allV = []
        for pi, (title, packed) in enumerate(panels):
            if packed is None:
                continue
            V, F, C, _ = packed
            V = V.copy(); V[:, 0] += gap * pi
            allV.append(V)
            mesh = pipe.o3d_mesh(V, F, C)
            self.uv_scene.scene.add_geometry(f"uv{pi}", mesh, mat)
            if self.wire[self.uv_wire_idx]:      # каркас островов (клавиша W)
                wf = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
                wf.paint_uniform_color([0.0, 0.0, 0.0])
                self.uv_scene.scene.add_geometry(f"uvwire{pi}", wf, lmat)
            grid = pipe._uv_grid_lines(na)
            gp = np.asarray(grid.points).copy(); gp[:, 0] += gap * pi
            grid.points = o3d.utility.Vector3dVector(gp)
            self.uv_scene.scene.add_geometry(f"uvgrid{pi}", grid, lmat)
        if allV and not keep_camera:
            allV = np.vstack(allV)
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                allV.min(0) - 0.05, allV.max(0) + 0.05)
            self.uv_scene.setup_camera(50.0, bbox, bbox.get_center())

    def _on_relax(self):
        """Релаксируем UV-острова FLAME и всех источников выбранным методом."""
        if not getattr(self, "_uv_zd", None):
            self.lbl_relax.text = "сначала дойди до шага «зоны»"
            return
        method = ["arap", "laplacian", "spring"][self.combo_relax.selected_index]
        iters = int(self.sl_relax_it.int_value)

        def relax_zd(zd, head):
            new = {}
            for a, (uv, F, gi) in zd.items():
                ruv = pipe.relax_uv_island(uv, F, head.verts[gi],
                                           method=method, iters=iters)
                new[a] = (ruv, F, gi)
            return new

        self._uv_zd['flame'] = relax_zd(self._uv_zd['flame'], self.heads[0])
        for k, src in enumerate(self._uv_zd['src']):
            if src is not None:
                src['zd'] = relax_zd(src['zd'], self.heads[k + 1])
        self._draw_uv_panels(keep_camera=True)
        self.lbl_relax.text = f"relax: {method}, iters={iters} ✓"
        print(f"Relax UV: метод={method}, iters={iters}")

    # ── клавиши: W → каркас в той сцене, что в фокусе ──
    def _key_cb(self, i):
        def cb(event):
            if (event.type == gui.KeyEvent.Type.DOWN
                    and event.key == gui.KeyName.W):
                self.wire[i] = not self.wire[i]
                if i < len(self.heads):
                    self._render(i)
                else:
                    self._draw_uv_panels(keep_camera=True)   # сохраняем релакс
                return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED
        return cb

    # ── выбор точек ──
    def _mouse_cb(self, i):
        def cb(event):
            if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                    and event.is_modifier_down(gui.KeyModifier.SHIFT)
                    and self.step == 0):
                self._pick(i, event.x, event.y)
                return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED
        return cb

    def _pick(self, i, mx, my):
        sc = self.scenes[i]; frame = sc.frame
        x = mx - frame.x; y = my - frame.y

        def after(depth_image):
            depth = np.asarray(depth_image)
            yy = int(np.clip(y, 0, depth.shape[0] - 1))
            xx = int(np.clip(x, 0, depth.shape[1] - 1))
            d = depth[yy, xx]
            if d >= 1.0:
                return
            world = sc.scene.camera.unproject(x, y, d, frame.width, frame.height)
            p = np.array([world[0], world[1], world[2]])
            _, idx = self.heads[i].tree().query(p)
            self.heads[i].anchors.append(int(idx))
            self.app.post_to_main_thread(
                self.window, lambda: (self._render(i), self._set_status()))

        sc.scene.scene.render_to_depth_image(after)

    # ── шаги пайплайна ──
    def _on_next(self):
        try:
            if self.step == 0:
                self._step_apply_expr()
            elif self.step == 1:
                self._step_diffusion()
            elif self.step == 2:
                self._step_zones()
            elif self.step == 3:
                self._step_clusters()
            elif self.step == 4:
                self._step_transfer()
            else:
                return
        except Exception as e:
            import traceback; traceback.print_exc()
            self.lbl_status.text = f"ошибка: {e}"
            return
        self.step = min(self.step + 1, len(STEPS) - 1)
        self._set_status()

    def _on_auto_run(self):
        """Полный прогон за один клик: сброс → авто-анкоры (MediaPipe) →
        эмоция → диффузия (+кроп) → зоны (+relax) → кластеры → перенос."""
        try:
            self._on_reset()
            self._on_auto_anchors()
            if any(len(hd.anchors) < 1 for hd in self.heads):
                self.lbl_status.text = ("авто-перенос: MediaPipe не поставил "
                                        "анкоры на всех головах")
                return
            self._step_apply_expr()              # 0→1
            self._step_diffusion()               # 1→2
            self._step_zones()                   # 2→3 (внутри авто-relax)
            self._step_clusters()                # 3→4
            self._step_transfer()                # 4→5
            self.step = len(STEPS) - 1
            self._set_status()
            self.lbl_status.text = "авто-перенос: готово ✓"
            print("Авто-перенос: весь пайплайн выполнен за один шаг.")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.lbl_status.text = f"авто-перенос ошибка: {e}"

    def _read_cfg(self):
        self.p['time'] = float(self.in_time.double_value)
        self.p['steps'] = int(self.in_steps.int_value)
        self.p['heat_threshold'] = float(self.in_heat_th.double_value)
        self.p['n_clusters'] = int(self.in_nclu.int_value)
        self.p['multi_t_n_times'] = int(self.in_mt_times.int_value)
        self.p['multi_t_n_eigs'] = int(self.in_mt_eigs.int_value)

    def _step_apply_expr(self):
        """Применяем выбранную эмоцию к FLAME (после расстановки якорей).
        Показываем FLAME деформированной — дальше все шаги идут с этой δ."""
        if len(self.heads[0].anchors) < 1:
            raise RuntimeError("FLAME: сначала поставь точки тепла")
        expr = pipe.parse_betas_string(self.in_expr.text_value.strip())
        flame = self.heads[0]
        if not flame.apply_expression(expr):
            raise RuntimeError("нет FLAME-модели для применения эмоции")
        # показать деформированную FLAME (поверх rest)
        flame.deformed = flame.verts + flame.delta_native
        flame.colors = pipe.to_colors(
            np.linalg.norm(flame.delta_native, axis=1), pipe.CMAP_DISP)
        self._render(0)
        n = int(np.count_nonzero(list(expr.values()))) if expr else 0
        print(f"Шаг «эмоция»: применено {len(expr)} betas "
              f"(max ‖δ‖={np.linalg.norm(flame.delta_native,axis=1).max():.4f}).")

    def _step_diffusion(self):
        self._read_cfg()
        # диффузию считаем на rest-геометрии (deformed только для показа эмоции
        # на FLAME — на форму тепла он влиять не должен, тепло на rest-меше).
        for hd in self.heads:
            hd.deformed = None
        for i, hd in enumerate(self.heads):
            if len(hd.anchors) < 1:
                raise RuntimeError(f"{hd.name}: нет точек тепла")
            L, MM = hd.ops()
            hd.heat = static_diffusion(hd.verts, hd.faces.astype(np.int64),
                                       L, MM, hd.anchors,
                                       self.p['time'], self.p['steps'])
            hd.n_anchors = len(hd.anchors)
            hd.colors = pipe.to_colors(hd.heat.sum(0), pipe.CMAP_HEAT)
            self._render(i)
        # Нормализация по bbox АКТИВНОЙ зоны (полигоны влияния тепла всех
        # anchor'ов) — ВСЕГДА, независимо от чекбокса обрезки. Чекбокс лишь
        # дополнительно удаляет неактивную геометрию.
        for i, hd in enumerate(self.heads):
            if self.cb_crop.checked:
                self._crop_to_active(hd, i)      # удаление + норм. по активной
            else:
                self._normalize_to_active(hd, i)  # только норм. по активной
        print("Шаг 1: single-t диффузия посчитана.")

    def _active_mask(self, hd):
        """Маска вершин активной зоны: макс. нормированное влияние любого
        anchor'а > heat_threshold (объединение зон reach всех anchor'ов)."""
        hn = hd.heat / hd.heat.max(1, keepdims=True).clip(1e-12)
        keepv = hn.max(0) > float(self.p['heat_threshold'])
        keepv[hd.anchors] = True                 # anchors всегда активны
        return keepv

    def _normalize_to_active(self, hd, i):
        """Нормализуем меш так, чтобы АКТИВНАЯ зона вписывалась в bbox
        (центр + деление на диагональ bbox активных вершин). Применяется ко
        ВСЕМ вершинам — геометрия НЕ удаляется. δ масштабируется тем же
        фактором. Делается всегда (см. _step_diffusion)."""
        keepv = self._active_mask(hd)
        if int(keepv.sum()) < 4:
            return
        actV = hd.verts[keepv]
        center = actV.mean(0)
        diag = float(np.linalg.norm(actV.max(0) - actV.min(0))) + 1e-12
        hd.verts = np.ascontiguousarray((hd.verts - center) / diag, np.float64)
        if hd.has_deform:
            hd.delta_native = np.ascontiguousarray(
                hd.delta_native / diag, np.float64)
        hd._L = hd._MM = None; hd._tree = None
        self._render(i)
        print(f"  [{hd.name}] нормализация по bbox активной зоны "
              f"({int(keepv.sum())} верш. влияния), геометрия сохранена")

    def _crop_to_active(self, hd, i):
        """Обрезаем меш головы по ОБЪЕДИНЁННОЙ активной зоне single-t (вершины,
        где макс. нормированное влияние любого anchor'а > heat_threshold),
        переиндексируем и нормализуем обрезок в bbox. Heat/anchors переносим."""
        keepv = self._active_mask(hd)
        if keepv.sum() < 4 or keepv.all():
            return                                  # нечего/нечего резать
        F = hd.faces.astype(np.int64)
        fmask = keepv[F].all(1)                     # грани целиком в активной
        Fk = F[fmask]
        if len(Fk) == 0:
            return
        used = np.unique(Fk)                        # реально используемые верш.
        remap = -np.ones(len(hd.verts), dtype=np.int64)
        remap[used] = np.arange(len(used))
        newV_raw = hd.verts[used]
        newF = remap[Fk]
        newHeat = hd.heat[:, used]
        new_anchors = [int(remap[a]) for a in hd.anchors if remap[a] >= 0]
        # ре-нормализация обрезка в bbox-кадр; δ масштабируем тем же фактором
        # (normalize_bbox делит на диагональ → δ_new = δ/diag, сдвиг сокращается)
        diag = float(np.linalg.norm(newV_raw.max(0) - newV_raw.min(0))) + 1e-12
        newV = pipe.normalize_bbox(newV_raw)
        had = hd.has_deform
        new_delta = (hd.delta_native[used] / diag) if had else None
        hd.verts = np.ascontiguousarray(newV, np.float64)
        hd.faces = np.ascontiguousarray(newF, np.int32)
        hd.heat = newHeat
        hd.anchors = new_anchors
        hd.n_anchors = len(new_anchors)
        hd.delta_native = (np.ascontiguousarray(new_delta, np.float64)
                           if had else np.zeros_like(hd.verts))
        hd.has_deform = had
        hd._L = hd._MM = None; hd._tree = None      # пересчитать операторы
        hd.colors = pipe.to_colors(hd.heat.sum(0), pipe.CMAP_HEAT)
        self._render(i)
        print(f"  [{hd.name}] crop: {len(hd.verts)} верш. "
              f"(активная зона single-t), нормализовано в bbox")

    def _step_zones(self):
        for i, hd in enumerate(self.heads):
            heat_single = hd.heat.copy()
            enr, _ = pipe.enrich_heat_multi_t(
                hd.verts, hd.faces.astype(np.int64), list(hd.anchors),
                n_times=self.p['multi_t_n_times'],
                n_eigs=self.p['multi_t_n_eigs'],
                smooth_iters=5, smooth_alpha=0.5, mesh_label=hd.name)
            # маска по single-t reach
            h1 = heat_single / heat_single.max(1, keepdims=True).clip(1e-12)
            active = h1.max(0) > self.p['heat_threshold']
            enr[:, ~active] = 0.0
            hd.heat = enr
            hd.partition = pipe._argmax_partition(
                enr, threshold=self.p['heat_threshold'])
            pal = pipe.make_cluster_palette(max(hd.n_anchors, 1))
            col = np.tile([0.3, 0.3, 0.3], (len(hd.verts), 1))
            for a in range(hd.n_anchors):
                col[hd.partition == a] = pal[a]
            hd.colors = col
            self._render(i)
        self._render_uv()                        # UV-острова зон внизу
        # авто-relax сразу после первой развёртки (50 итераций выбр. методом)
        self.sl_relax_it.int_value = 50
        self._on_relax()
        print("Шаг 2: зоны (multi-t + маска) построены + авто-relax 50 итер.")

    def _step_clusters(self):
        for i, hd in enumerate(self.heads):
            clusters_pa = []
            for a in range(hd.n_anchors):
                masked = hd.heat[a].copy()
                masked[hd.partition != a] = 0.0
                cls = pipe.cluster_zone(
                    masked, hd.delta_native, hd.verts, anchor_idx=a,
                    heat_threshold=self.p['heat_threshold'],
                    n_clusters_max=self.p['n_clusters'],
                    position_weight=self.p.get('position_weight', 0.0),
                    clustering_method=self.p.get('clustering_method', 'kmeans'),
                    similarity_threshold=0.3, print_quality=False)
                clusters_pa.append(cls)
            # per-vertex global cluster id
            vgid = -np.ones(len(hd.verts), dtype=np.int64)
            vgw = np.zeros(len(hd.verts)); gid = 0
            n_total = sum(len(c) for c in clusters_pa)
            pal = pipe.make_cluster_palette(max(n_total, 1))
            col = np.tile([0.3, 0.3, 0.3], (len(hd.verts), 1))
            ci = 0
            for cls in clusters_pa:
                for cl in cls:
                    for j, vi in enumerate(cl['indices']):
                        wv = cl['heat_weights'][j]
                        col[vi] = pal[ci]
                        if wv > vgw[vi]:
                            vgw[vi] = wv; vgid[vi] = gid
                    gid += 1; ci += 1
            hd.vert_gcid = vgid
            hd.colors = col
            self._render(i)
        print(f"Шаг 3: кластеры построены.")

    def _step_transfer(self):
        src = self.heads[0].result_dict()        # FLAME
        zd_src = (self._uv_zd['flame']
                  if getattr(self, "_uv_zd", None) else None)
        self._raw_delta = {}
        transfers = []
        ok = 0
        for k in range(1, len(self.heads)):
            dst = self.heads[k].result_dict()
            zd_dst = None
            if (getattr(self, "_uv_zd", None) and self._uv_zd['src'][k - 1]):
                zd_dst = self._uv_zd['src'][k - 1]['zd']
            tr = pipe.transfer_deformations_uv(
                src, dst,
                flat=pipe._uv_mode(self.p),
                warp_heat=bool(self.p.get('uv_warp_heat', False)),
                warp_heat_t=float(self.p.get('uv_warp_heat_t', 0.05)),
                warp_min_dist=float(self.p.get('uv_warp_min_dist', 0.0)),
                interp_delta=bool(self.p.get('uv_interp_delta', True)),
                zd_src=zd_src, zd_dst=zd_dst)
            transfers.append(tr)
            if tr is not None:
                self._raw_delta[k] = tr['delta']
                ok += 1
        if ok == 0:
            raise RuntimeError("перенос вернул None (нужны зоны на головах)")
        self._apply_transfer_delta()             # применяем ко всем источникам
        # FLAME — со своей родной деформацией для сравнения
        flame = self.heads[0]
        flame.deformed = flame.verts + flame.delta_native
        flame.colors = pipe.to_colors(
            np.linalg.norm(flame.delta_native, axis=1), pipe.CMAP_DISP)
        self._render(0)
        # обновляем transfer в кэше каждого источника, перерисовываем (релакс
        # островов сохраняется — не пересобираем зоны).
        if getattr(self, "_uv_zd", None):
            for k, tr in enumerate(transfers):
                if self._uv_zd['src'][k] is not None:
                    self._uv_zd['src'][k]['transfer'] = tr
            self._draw_uv_panels(keep_camera=True)
        else:
            self._render_uv(transfers=transfers)
        print(f"Шаг 5: выражение перенесено на {ok} источник(ов).")

    def _apply_transfer_delta(self):
        """Пересборка деформированных FBX из кэш. сырого δ (по всем источникам):
        множитель усиления × Laplacian-смус. Вызывается слайдерами/кнопкой."""
        if not self._raw_delta:
            return
        try:
            gain = float(self.sl_gain.double_value)
            it = int(self.sl_smooth.int_value)
            for k, raw in self._raw_delta.items():
                fbx = self.heads[k]
                sm = pipe.smooth_delta(raw * gain, fbx.faces.astype(np.int64),
                                       n_iter=it,
                                       alpha=float(self.p.get('smooth_alpha', 0.5)))
                fbx.deformed = fbx.verts + sm
                fbx.colors = pipe.to_colors(np.linalg.norm(sm, axis=1),
                                            pipe.CMAP_DISP)
                self._update_head_mesh(k)
            self.lbl_gain.text = f"усиление ×{gain:.2f},  smooth iters={it}"
        except Exception as e:
            import traceback; traceback.print_exc()
            self.lbl_gain.text = f"ошибка: {e}"

    def _on_save_deformation(self):
        """Сохраняем ФИНАЛЬНУЮ деформацию каждого источника (с текущими
        усилением+smooth): деформ. меш в .fbx + per-vertex таблица δ в .h5."""
        if not self._raw_delta:
            self.lbl_save.text = "сначала выполни перенос (шаг 5)"
            return
        try:
            from pathlib import Path
            import datetime as _dt
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir = Path("python/scripts/debug_output") / f"deform_{ts}"
            gain = float(self.sl_gain.double_value)
            it = int(self.sl_smooth.int_value)
            saved = []
            for k, raw in self._raw_delta.items():
                hd = self.heads[k]
                sm = pipe.smooth_delta(raw * gain, hd.faces.astype(np.int64),
                                       n_iter=it,
                                       alpha=float(self.p.get('smooth_alpha', 0.5)))
                base = out_dir / hd.name
                w = pipe.export_deformation(str(base), hd.verts,
                                            hd.faces.astype(np.int64), sm,
                                            label=hd.name)
                saved.append(hd.name)
            self.lbl_save.text = (f"сохранено: {', '.join(saved)}\n→ {out_dir}/")
            print(f"Деформации сохранены в {out_dir}/")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.lbl_save.text = f"ошибка сохранения: {e}"

    def _on_auto_anchors(self):
        """MediaPipe-автопостановка анкоров на обеих головах (шаг 0).
        Рендер спереди → лендмарки → рейкаст → ближайшая вершина."""
        if self.step != 0:
            self.lbl_auto.text = "доступно только на шаге 0 (Сброс для возврата)"
            return
        try:
            import auto_anchors
        except Exception as e:
            self.lbl_auto.text = f"нет MediaPipe: {e}"
            return
        # индексы лендмарок из поля (пусто → набор по умолчанию)
        txt = self.in_landmarks.text_value.strip()
        lm_idx = None
        if txt:
            try:
                lm_idx = [int(s) for s in txt.replace(";", ",").split(",")
                          if s.strip()]
            except ValueError:
                self.lbl_auto.text = "ошибка: индексы — числа через запятую"
                return
        msg = []
        for i, hd in enumerate(self.heads):
            idx, dbg = auto_anchors.auto_anchors(
                hd.verts, hd.faces.astype(np.int64), landmark_indices=lm_idx)
            if not dbg.get('ok'):
                msg.append(f"{hd.name}: {dbg.get('reason','?')}")
                continue
            hd.anchors = list(idx)
            self._render(i)
            msg.append(f"{hd.name}: {len(idx)} анкор (вид {dbg['axis']}/"
                       f"{dbg['sign']:+.0f}, miss {dbg['miss']})")
        self.lbl_auto.text = "\n".join(msg)
        self._set_status()
        print("Авто-анкоры:", " | ".join(msg))

    def _on_reset(self):
        for hd in self.heads:
            # восстанавливаем исходный (необрезанный) меш
            hd.verts = hd.verts0.copy(); hd.faces = hd.faces0.copy()
            hd.anchors = []; hd.heat = None; hd.partition = None
            hd.vert_gcid = None; hd.colors = None; hd.deformed = None
            hd.delta_native = np.zeros_like(hd.verts); hd.has_deform = False
            hd._L = hd._MM = None; hd._tree = None
        self.step = 0
        self._last_transfer = None
        self._raw_delta = {}
        self._uv_zd = None
        self.uv_scene.scene.clear_geometry()
        for i in range(len(self.heads)):
            self._render(i)
        self._set_status()


def main():
    ap = argparse.ArgumentParser(description="Interactive pipeline viewer")
    ap.add_argument("--flame", default=pipe.FLAME_PKL)
    ap.add_argument("--fbx", action="append", default=[],
                    help="путь к FBX-источнику (можно до 3 раз)")
    ap.add_argument("--expr", default="")
    ap.add_argument("--shape", default="")
    args = ap.parse_args()

    v_t, sd, faces = pipe.load_flame(args.flame)
    shape = pipe.parse_betas_string(args.shape)
    v_rest = pipe.normalize_bbox(pipe.apply_betas(v_t, sd, shape))
    # стартуем с rest; эмоция применяется кнопкой ПОСЛЕ расстановки якорей.
    flame = Head("FLAME", v_rest, faces, delta_native=None,
                 flame_model=(v_t, sd, shape))

    fbx_paths = [p for p in (args.fbx or []) if p.strip()][:3]   # до 3 FBX
    fbxs = []
    for n, p in enumerate(fbx_paths, 1):
        if Path(p).exists():
            fv, ff = pipe.load_custom_mesh(p)
            fbxs.append(Head(f"FBX{n}", pipe.normalize_bbox(fv), ff))
        else:
            print(f"FBX не найден, пропуск: {p}")
    if not fbxs:                                  # нет FBX → одна копия FLAME
        print("FBX не задан → источник = копия FLAME (rest).")
        fbxs = [Head("FBX1", v_rest.copy(), faces.copy())]

    params = dict(time=0.002, steps=60, heat_threshold=0.05, n_clusters=5,
                  multi_t_n_times=8, multi_t_n_eigs=80, position_weight=0.0,
                  clustering_method='kmeans', smooth_iters=3, smooth_alpha=0.5,
                  uv_interp_delta=True, expr_str=args.expr)

    gui.Application.instance.initialize()
    _setup_cyrillic_font()
    PipelineViewer(flame, fbxs, params)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
