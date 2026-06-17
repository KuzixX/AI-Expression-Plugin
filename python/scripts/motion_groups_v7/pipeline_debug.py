#!/usr/bin/env python3
"""
v7 Pipeline step debug — пошаговый показ переноса в Open3D-GUI.

Две головы рядом: reference-нейтраль (FLAME) + первая голова из папки.
Кнопка "Next ▶" проводит обе головы по шагам пайплайна v7 (= v6) с перекраской:
  0. anchors  — авто-анкоры MediaPipe (красные точки)
  1. diffusion — single-t тепло
  2. zones     — multi-t + argmax зоны
  3. clusters  — кластеры
  4. transfer  — перенос δ первого выражения reference на голову

ВСЕ настройки берутся из JSON, переданного основным окном v7 (--params).

  python pipeline_debug.py --ref <reference.h5> --dir <heads_dir> --params <p.json>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent / "motion_groups_v6"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import debug_head1_pipeline as pipe          # noqa: E402
import auto_anchors                          # noqa: E402
import transfer_engine as eng                # noqa: E402


STEPS = ["0 · anchors", "1 · diffusion", "2 · zones",
         "3 · clusters", "4 · motion groups", "5 · WKS anchors",
         "6 · transfer"]


class Head:
    def __init__(self, name, verts, faces, delta_native=None):
        self.name = name
        self.verts = np.ascontiguousarray(verts, np.float64)
        self.faces = np.ascontiguousarray(faces, np.int32)
        self.delta_native = (np.zeros_like(self.verts) if delta_native is None
                             else np.asarray(delta_native, np.float64))
        self.anchors = []
        self.n_anchors = 0
        self.res = None          # result_dict после зон/кластеров
        self.zd = None           # UV-острова (relax)
        self.colors = None
        self.deformed = None
        self.wks = None          # WKS-сигнатурные лендмарки (индексы вершин)


class DebugViewer:
    def __init__(self, flame, head, exprs, p):
        self.p = p
        self.exprs = exprs                    # {имя: δ} из reference
        self.heads = [flame, head]
        self.step = 0
        self._build_gui()
        for i in range(2):
            self._render(i)
        self._set_status()

    def _build_gui(self):
        self.app = gui.Application.instance
        self.window = self.app.create_window("v7 Pipeline step debug", 1300, 800)
        w = self.window
        em = w.theme.font_size
        # каркас (W): по флагу на каждую 3D-сцену + UV
        self.wire = [False, False, False]          # head0, head1, uv
        self.scenes = []
        for i in range(2):
            sc = gui.SceneWidget()
            sc.scene = rendering.Open3DScene(w.renderer)
            sc.scene.set_background([0.1, 0.1, 0.12, 1.0])
            sc.set_on_key(self._key_cb(i))         # W → каркас в этой сцене
            self.scenes.append(sc)
            w.add_child(sc)
        # нижняя сцена — UV-развёртка зон (как в v6)
        self.uv_scene = gui.SceneWidget()
        self.uv_scene.scene = rendering.Open3DScene(w.renderer)
        self.uv_scene.scene.set_background([0.12, 0.12, 0.14, 1.0])
        self.uv_scene.set_on_key(self._key_cb(2))
        w.add_child(self.uv_scene)
        self._uv_zd = None
        self._last_transfer = None                 # для перерисовки UV по W

        panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        panel.add_child(gui.Label("Step debug"))
        self.btn = gui.Button("Next ▶")
        self.btn.set_on_clicked(self._on_next)
        panel.add_child(self.btn)
        btn_r = gui.Button("Reset")
        btn_r.set_on_clicked(self._on_reset)
        panel.add_child(btn_r)
        panel.add_fixed(em)
        panel.add_child(gui.Label("W: wireframe in hovered view"))
        self.lbl = gui.Label("")
        panel.add_child(self.lbl)
        self.panel = panel
        w.add_child(panel)
        w.set_on_layout(self._layout)

    def _layout(self, ctx):
        r = self.window.content_rect
        pw = min(16 * self.window.theme.font_size, r.width * 0.22)
        uv_h = r.height * 0.32                     # нижняя полоса под UV
        top_h = r.height - uv_h
        sw = (r.width - pw) / 2
        self.scenes[0].frame = gui.Rect(r.x, r.y, sw, top_h)
        self.scenes[1].frame = gui.Rect(r.x + sw, r.y, sw, top_h)
        self.uv_scene.frame = gui.Rect(r.x, int(r.y + top_h),
                                       int(r.width - pw), int(uv_h))
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    def _key_cb(self, i):
        def cb(event):
            if (event.type == gui.KeyEvent.Type.DOWN
                    and event.key == gui.KeyName.W):
                self.wire[i] = not self.wire[i]
                if i < 2:
                    self._render(i)
                else:
                    self._draw_uv(self._last_transfer)
                return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED
        return cb

    def _draw_uv(self, transfer=None):
        """UV-острова FLAME + головы (+ перенесённые группы) внизу, как в v6."""
        self._last_transfer = transfer
        flame, head = self.heads
        if flame.res is None or head.res is None:
            return
        flat = ("world" if self.p.get('uv_world_orient')
                else bool(self.p.get('uv_flat', False)))
        zd_f = flame.zd or pipe.compute_zone_islands(
            flame.verts, flame.faces.astype(np.int64),
            flame.res['partition'], flame.n_anchors, flat=flat)
        zd_x = head.zd or pipe.compute_zone_islands(
            head.verts, head.faces.astype(np.int64),
            head.res['partition'], head.n_anchors, flat=flat)
        if not zd_f or not zd_x:
            return
        zmap = pipe.match_zones_by_position(flame.res, head.res)
        zd_x_m = {zmap[a]: isl for a, isl in zd_x.items() if a in zmap}
        na = flame.n_anchors
        pal = pipe.make_cluster_palette(max(na, 1))
        GREY = np.array([0.6, 0.6, 0.6])
        panels = [pipe._pack_islands_to_grid(zd_f, na, pal),
                  pipe._pack_islands_to_grid(zd_x_m, na, pal)]
        if transfer is not None and 'gcid' in transfer:
            gcid = np.asarray(transfer['gcid'])
            ng = int(gcid.max()) + 1 if gcid.max() >= 0 else 1
            palg = pipe.make_cluster_palette(max(ng, 1))
            cdict = {}
            for a in zd_x_m:
                g = gcid[zd_x_m[a][2]]
                cdict[a] = np.where((g >= 0)[:, None],
                                    palg[np.clip(g, 0, ng - 1)], GREY)
            panels.append(pipe._pack_islands_to_grid(zd_x_m, na, cdict))

        self.uv_scene.scene.clear_geometry()
        mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"
        lmat = rendering.MaterialRecord(); lmat.shader = "unlitLine"
        gap = 1.25; allV = []
        for pi, packed in enumerate(panels):
            if packed is None:
                continue
            V, F, C, _ = packed
            V = V.copy(); V[:, 0] += gap * pi
            allV.append(V)
            m = pipe.o3d_mesh(V, F, C)
            self.uv_scene.scene.add_geometry(f"uv{pi}", m, mat)
            if self.wire[2]:                       # каркас островов (клавиша W)
                wf = o3d.geometry.LineSet.create_from_triangle_mesh(m)
                wf.paint_uniform_color([0.0, 0.0, 0.0])
                self.uv_scene.scene.add_geometry(f"uvwire{pi}", wf, lmat)
            grid = pipe._uv_grid_lines(na)
            gp = np.asarray(grid.points).copy(); gp[:, 0] += gap * pi
            grid.points = o3d.utility.Vector3dVector(gp)
            self.uv_scene.scene.add_geometry(f"uvgrid{pi}", grid, lmat)

        # WKS-лендмарки жёлтыми точками на панелях FLAME (0) и головы (1)
        placed_f = panels[0][3] if panels and panels[0] is not None else None
        placed_x = (panels[1][3] if len(panels) > 1 and panels[1] is not None
                    else None)
        self._add_uv_spheres(
            self._uv_wks_points(placed_f, zd_f, self.heads[0].wks),
            0.0, 0.02, [1.0, 0.85, 0.0], "wksf")
        self._add_uv_spheres(
            self._uv_wks_points(placed_x, zd_x_m, self.heads[1].wks),
            gap * 1, 0.02, [1.0, 0.85, 0.0], "wksx")

        # ── overlay-панель: обе развёртки наложены (FLAME синий заливкой,
        # голова красным каркасом поверх) — как в v6 ──
        pov = len(panels)
        OVA = [0.15, 0.45, 0.95]                   # FLAME — синий (заливка)
        OVB = [0.93, 0.25, 0.20]                   # голова — красный (каркас)
        A = pipe._pack_islands_to_grid(zd_f, na, OVA)
        B = pipe._pack_islands_to_grid(zd_x_m, na, OVB)
        if A is not None and B is not None:
            VA = A[0].copy(); VA[:, 0] += gap * pov; allV.append(VA)
            am = pipe.o3d_mesh(VA, A[1], A[2])
            if self.wire[2]:                           # по W — FLAME тоже каркасом
                aw = o3d.geometry.LineSet.create_from_triangle_mesh(am)
                aw.paint_uniform_color(OVA)
                self.uv_scene.scene.add_geometry("ov_a", aw, lmat)
            else:                                      # иначе — синяя заливка
                self.uv_scene.scene.add_geometry("ov_a", am, mat)
            VB = B[0].copy(); VB[:, 0] += gap * pov; allV.append(VB)
            bm = pipe.o3d_mesh(VB, B[1], B[2])
            wire = o3d.geometry.LineSet.create_from_triangle_mesh(bm)
            wire.paint_uniform_color(OVB)
            self.uv_scene.scene.add_geometry("ov_b", wire, lmat)
            grid = pipe._uv_grid_lines(na)
            gp = np.asarray(grid.points).copy(); gp[:, 0] += gap * pov
            grid.points = o3d.utility.Vector3dVector(gp)
            self.uv_scene.scene.add_geometry("ov_grid", grid, lmat)
            # WKS-лендмарки на overlay: FLAME синие, голова оранжевые + матч-линии
            ptsA = self._uv_wks_points(A[3], zd_f, self.heads[0].wks)
            ptsB = self._uv_wks_points(B[3], zd_x_m, self.heads[1].wks)
            self._add_uv_spheres(ptsA, gap * pov, 0.02, [0.2, 0.6, 1.0], "ovwf")
            self._add_uv_spheres(ptsB, gap * pov, 0.02, [1.0, 0.5, 0.1], "ovwx")
            self._add_uv_match_lines(ptsA, ptsB, gap * pov, 0.03, "ovwline")

        # ПОСТ-ВАРП панель: FLAME + заварпленная голова, лендмарки должны совпасть
        self._draw_wks_warp_panel(gap * (pov + 1), zd_f, zd_x_m, allV)

        if allV:
            allV = np.vstack(allV)
            bbox = o3d.geometry.AxisAlignedBoundingBox(
                allV.min(0) - 0.05, allV.max(0) + 0.05)
            self.uv_scene.setup_camera(50.0, bbox, bbox.get_center())

    def _uv_wks_points(self, placed, zd, wks):
        """[(зона, глоб_вершина, placed_xy)] для WKS-лендмарков в раскладке."""
        out = []
        if placed is None or wks is None:
            return out
        wset = set(int(v) for v in np.asarray(wks).tolist())
        for a in zd:
            if a not in placed:
                continue
            gidx = np.asarray(zd[a][2]); P = placed[a]
            for li, gv in enumerate(gidx):
                if int(gv) in wset:
                    out.append((a, int(gv), P[li]))
        return out

    def _add_uv_spheres(self, pts, xoff, z, color, tag, r=0.011):
        mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"
        for k, (_a, _gv, p) in enumerate(pts):
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=r)
            sph.translate([p[0] + xoff, p[1], z])
            sph.paint_uniform_color(color); sph.compute_vertex_normals()
            self.uv_scene.scene.add_geometry(f"{tag}{k}", sph, mat)

    def _add_uv_match_lines(self, pts_f, pts_x, xoff, z, tag):
        """Линии: лендмарк головы → ближайший лендмарк FLAME в ТОЙ ЖЕ зоне."""
        from collections import defaultdict
        fz = defaultdict(list)
        for (a, _gv, p) in pts_f:
            fz[a].append(p)
        pp = []; ln = []
        for (a, _gv, p) in pts_x:
            cand = fz.get(a)
            if not cand:
                continue
            C = np.asarray(cand); j = int(np.argmin(((C - p) ** 2).sum(1)))
            i0 = len(pp)
            pp.append([p[0] + xoff, p[1], z])
            pp.append([C[j][0] + xoff, C[j][1], z]); ln.append([i0, i0 + 1])
        if not ln:
            return
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.asarray(pp, np.float64))
        ls.lines = o3d.utility.Vector2iVector(np.asarray(ln, np.int32))
        ls.paint_uniform_color([1.0, 0.9, 0.0])
        lmat = rendering.MaterialRecord(); lmat.shader = "unlitLine"
        self.uv_scene.scene.add_geometry(tag, ls, lmat)

    def _draw_wks_warp_panel(self, panel_x, zd_f, zd_x_m, allV):
        """ПОСТ-ВАРП проверка матча: для каждой зоны нормализуем острова (как в
        переносе), матчим лендмарки и применяем _warp_island_heat. Рисуем FLAME
        (синий каркас) + ЗАВАРПЛЕННУЮ голову (красный каркас); лендмарки FLAME —
        синие сферы, головы — оранжевые. После варпа оранжевые должны сесть
        внутрь синих (совпасть)."""
        from scipy.spatial import cKDTree
        import math
        f_wks, h_wks = self.heads[0].wks, self.heads[1].wks
        if f_wks is None or h_wks is None:
            return
        f_arr = np.asarray(f_wks); h_arr = np.asarray(h_wks)
        t = float(self.p.get('uv_warp_heat_t', 0.05))
        md = float(self.p.get('uv_warp_min_dist', 0.0))
        align = bool(self.p.get('uv_align_pca_icp', False))
        zones = [a for a in sorted(zd_x_m) if a in zd_f]
        if not zones:
            return
        cols = max(1, int(math.ceil(math.sqrt(len(zones)))))
        S = 0.2; cell = 1.6                            # масштаб/шаг под др. панели
        mat = rendering.MaterialRecord(); mat.shader = "defaultUnlit"
        lmat = rendering.MaterialRecord(); lmat.shader = "unlitLine"
        for k, a in enumerate(zones):
            uv_s, Fk_s, gi_s = zd_f[a]
            uv_d, Fk_d, gi_d = zd_x_m[a]
            ns = pipe._normalize_island_uv(uv_s)
            nd = pipe._normalize_island_uv(uv_d)
            if align:
                nd = pipe._align_island_pca_icp(nd, ns)
            s_loc = np.where(np.isin(gi_s, f_arr))[0]
            d_loc = np.where(np.isin(gi_d, h_arr))[0]
            lm_local = lm_targets = None
            if len(s_loc) and len(d_loc):
                _, mi = cKDTree(ns[s_loc]).query(nd[d_loc])
                lm_local = d_loc; lm_targets = ns[s_loc[mi]]
            warped = pipe._warp_island_heat(
                nd, Fk_d, ns, Fk_s, t=t, min_dist=md,
                lm_local=lm_local, lm_targets=lm_targets)
            ox = panel_x + (k % cols) * cell
            oy = -(k // cols) * cell

            def _place(P):
                Q = np.zeros((len(P), 3))
                Q[:, 0] = P[:, 0] * S + ox; Q[:, 1] = P[:, 1] * S + oy
                return Q
            Vs = _place(ns); Vw = _place(warped)
            allV.append(Vs); allV.append(Vw)
            wf = o3d.geometry.LineSet.create_from_triangle_mesh(
                pipe.o3d_mesh(Vs, Fk_s))
            wf.paint_uniform_color([0.2, 0.5, 1.0])         # FLAME синий каркас
            self.uv_scene.scene.add_geometry(f"ww_f{k}", wf, lmat)
            wh = o3d.geometry.LineSet.create_from_triangle_mesh(
                pipe.o3d_mesh(Vw, Fk_d))
            wh.paint_uniform_color([1.0, 0.3, 0.2])         # голова красный каркас
            self.uv_scene.scene.add_geometry(f"ww_h{k}", wh, lmat)
            for j, li in enumerate(s_loc):                  # лендмарки FLAME (синие)
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
                sph.translate(Vs[li]); sph.paint_uniform_color([0.2, 0.6, 1.0])
                sph.compute_vertex_normals()
                self.uv_scene.scene.add_geometry(f"ww_sf{k}_{j}", sph, mat)
            for j, li in enumerate(d_loc):                  # головы (оранжевые)
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.016)
                sph.translate(Vw[li]); sph.paint_uniform_color([1.0, 0.5, 0.1])
                sph.compute_vertex_normals()
                self.uv_scene.scene.add_geometry(f"ww_sh{k}_{j}", sph, mat)

    def _render(self, i):
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
        if self.wire[i]:                           # каркас поверх (клавиша W)
            wf = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)
            wf.paint_uniform_color([0.0, 0.0, 0.0])
            lm = rendering.MaterialRecord(); lm.shader = "unlitLine"
            sc.scene.add_geometry(f"wire{i}", wf, lm)
        for a in hd.anchors:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sph.translate(hd.verts[a]); sph.paint_uniform_color([1, 0, 0])
            sph.compute_vertex_normals()
            m2 = rendering.MaterialRecord(); m2.shader = "defaultLit"
            sc.scene.add_geometry(f"a{i}_{a}", sph, m2)
        # WKS-лендмарки (анкоры подгонки) — жёлтые сферы, крупнее
        wks = hd.wks
        if wks is not None:
            m3 = rendering.MaterialRecord(); m3.shader = "defaultLit"
            for a in wks:
                sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.014)
                sph.translate(hd.verts[int(a)])
                sph.paint_uniform_color([1.0, 0.85, 0.0])
                sph.compute_vertex_normals()
                sc.scene.add_geometry(f"w{i}_{int(a)}", sph, m3)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            hd.verts.min(0) - 0.05, hd.verts.max(0) + 0.05)
        sc.setup_camera(50.0, bbox, bbox.get_center())

    def _set_status(self):
        self.lbl.text = f"step: {STEPS[self.step]}"

    # ── шаги ──
    def _on_next(self):
        try:
            if self.step == 0:
                self._step_anchors()
            elif self.step == 1:
                self._step_diffusion()
            elif self.step == 2:
                self._step_zones()
            elif self.step == 3:
                self._step_clusters()
            elif self.step == 4:
                self._step_motion_groups()
            elif self.step == 5:
                self._step_wks_anchors()
            elif self.step == 6:
                self._step_transfer()
            else:
                return
        except Exception as e:
            import traceback; traceback.print_exc()
            self.lbl.text = f"error: {e}"
            return
        self.step = min(self.step + 1, len(STEPS) - 1)
        self._set_status()

    def _step_anchors(self):
        lm = list(self.p.get('landmarks', (9, 4, 199)))
        for i, hd in enumerate(self.heads):
            idx, dbg = auto_anchors.auto_anchors(
                hd.verts, hd.faces.astype(np.int64), landmark_indices=lm)
            if not dbg.get('ok'):
                raise RuntimeError(f"{hd.name}: face not found")
            hd.anchors = list(idx); hd.n_anchors = len(idx)
            self._render(i)
        print("anchors:", [h.n_anchors for h in self.heads])

    def _step_diffusion(self):
        for i, hd in enumerate(self.heads):
            L, MM = pipe.build_operators(hd.verts, hd.faces.astype(np.int64))
            heat = eng._static_diffusion(
                hd.verts, hd.faces.astype(np.int64), L, MM, hd.anchors,
                self.p['time'], self.p['steps'])
            hd._heat_single = heat
            hd.colors = pipe.to_colors(heat.sum(0), pipe.CMAP_HEAT)
            self._render(i)

    def _step_zones(self):
        for i, hd in enumerate(self.heads):
            dn = hd.delta_native
            hd.res = eng._build_zones(hd.verts, hd.faces.astype(np.int64),
                                      hd.anchors, dn, self.p)
            part = hd.res['partition']
            pal = pipe.make_cluster_palette(max(hd.n_anchors, 1))
            col = np.tile([0.3, 0.3, 0.3], (len(hd.verts), 1))
            for a in range(hd.n_anchors):
                col[part == a] = pal[a]
            hd.colors = col
            self._render(i)
        self._draw_uv()                            # UV-острова зон внизу

    def _step_clusters(self):
        # UV-острова зон + relax (геометрия UV-развёртки) — без перекраски голов
        for i, hd in enumerate(self.heads):
            hd.zd = eng._relax_zones(hd.res, self.p)
        self._draw_uv()                            # перерисуем UV (relaxed)
        self.lbl.text = "step: 3 · clusters (UV islands + relax)"

    def _step_motion_groups(self):
        """Показываем МОУШН-ГРУППЫ (global cluster id) цветом на обеих головах +
        overlay двух развёрток внизу."""
        for i, hd in enumerate(self.heads):
            gid = hd.res['vert_gcid']
            ng = int(gid.max()) + 1 if gid.max() >= 0 else 1
            pal = pipe.make_cluster_palette(max(ng, 1))
            GREY = np.array([0.5, 0.5, 0.5])
            col = np.where((gid >= 0)[:, None],
                           pal[np.clip(gid, 0, ng - 1)], GREY)
            hd.colors = col
            self._render(i)
        self._draw_uv()                            # зоны + overlay (синий/красн.)

    def _step_wks_anchors(self):
        """После подгонки границ — ПОДГОНКА АНКОРОВ. Считаем WKS-сигнатурные
        лендмарки на FLAME и голове и рисуем их жёлтыми сферами: это якоря,
        которыми UV дополнительно тянется (нос↔нос и т.п.) поверх границы.
        Эти же лендмарки идут в перенос (след. шаг)."""
        def _f(k, d):
            return float(self.p.get(k, d))

        def _i(k, d):
            return int(self.p.get(k, d))
        kw = dict(
            sig_min=_f('wks_sig_min', 0.33), sig_max=_f('wks_sig_max', 0.50),
            sig_dist=_f('wks_sig_dist', 0.05),
            sig_smooth=_i('wks_sig_smooth', 0),
            n_eigs=_i('wks_desc_eigs', 80),
            n_channels=_i('wks_desc_channels', 60),
            channel=self.p.get('wks_desc_channel'),
            wks_sigma=_f('wks_desc_sigma', 7.0))
        for i, hd in enumerate(self.heads):
            hd.wks = eng.signature_landmark_verts(
                hd.verts, hd.faces.astype(np.int64), **kw)
            self._render(i)
        self._draw_uv(getattr(self, '_last_transfer', None))   # лендмарки в UV
        nf = len(self.heads[0].wks); nh = len(self.heads[1].wks)
        print(f"  WKS anchors: FLAME {nf} · head {nh} (жёлтые сферы + UV)")

    def _step_transfer(self):
        flame, head = self.heads
        ename = sorted(self.exprs)[0]
        res_src = dict(flame.res); res_src['delta_native'] = self.exprs[ename]
        # если посчитаны WKS-анкоры (шаг 5) — включаем их в warp (граница + якоря)
        wks_on = flame.wks is not None and head.wks is not None
        tr = pipe.transfer_deformations_uv(
            res_src, head.res,
            flat=("world" if self.p.get('uv_world_orient')
                  else bool(self.p.get('uv_flat', False))),
            align_pca_icp=bool(self.p.get('uv_align_pca_icp', False)),
            warp_heat=bool(self.p.get('uv_warp_heat', False)) or wks_on,
            warp_heat_t=float(self.p.get('uv_warp_heat_t', 0.05)),
            warp_min_dist=float(self.p.get('uv_warp_min_dist', 0.0)),
            interp_delta=bool(self.p.get('uv_interp_delta', True)),
            zd_src=flame.zd, zd_dst=head.zd,
            wks_src=flame.wks, wks_dst=head.wks)
        if tr is None:
            raise RuntimeError("transfer returned None")
        delta = pipe.smooth_delta(tr['delta'], head.faces.astype(np.int64),
                                  n_iter=int(self.p.get('smooth_iters', 3)),
                                  alpha=float(self.p.get('smooth_alpha', 0.5)))
        head.deformed = head.verts + delta
        head.colors = pipe.to_colors(np.linalg.norm(delta, axis=1),
                                     pipe.CMAP_DISP)
        self._render(1)
        # FLAME — со своим δ выражения
        flame.deformed = flame.verts + self.exprs[ename]
        flame.colors = pipe.to_colors(
            np.linalg.norm(self.exprs[ename], axis=1), pipe.CMAP_DISP)
        self._render(0)
        self._draw_uv(transfer=tr)                 # + панель перенесённых групп
        print(f"transfer '{ename}': max|δ|={np.linalg.norm(delta,axis=1).max():.4f}")

    def _on_reset(self):
        for hd in self.heads:
            hd.anchors = []; hd.n_anchors = 0; hd.res = None; hd.zd = None
            hd.colors = None; hd.deformed = None
        self.step = 0
        self._last_transfer = None
        self.uv_scene.scene.clear_geometry()
        for i in range(2):
            self._render(i)
        self._set_status()


def _setup_font():
    import os
    for fp in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
               "/System/Library/Fonts/Helvetica.ttc"):
        if os.path.exists(fp):
            fd = gui.FontDescription(fp)
            gui.Application.instance.set_font(
                gui.Application.DEFAULT_FONT_ID, fd)
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference HDF5")
    ap.add_argument("--dir", required=True, help="heads folder")
    ap.add_argument("--params", default="", help="JSON с параметрами")
    args = ap.parse_args()

    p = dict(eng.DEFAULT_PARAMS)
    if args.params and Path(args.params).exists():
        with open(args.params) as f:
            jp = json.load(f)
        if isinstance(jp.get('landmarks'), str):
            jp['landmarks'] = tuple(int(s) for s in
                                    jp['landmarks'].replace(";", ",").split(",")
                                    if s.strip())
        p.update(jp)

    neutral, faces, exprs, _acts, _mn = eng.read_reference(args.ref)
    neutral = pipe.normalize_bbox(neutral)
    files = eng.list_head_files(args.dir)
    if not files:
        raise SystemExit(f"no head files in {args.dir}")

    # ── стартовый диалог: выбор головы / выражения / силы ──
    sel = _start_dialog([f.stem for f in files], sorted(exprs))
    if sel is None:
        return
    head_idx, expr_name, strength = sel

    flame = Head("FLAME (ref)", neutral, faces)
    v_raw, faces_x = pipe.load_custom_mesh(str(files[head_idx]))
    head = Head(files[head_idx].stem, pipe.normalize_bbox(v_raw), faces_x)

    # сила выражения: масштабируем δ выбранного выражения; в debug показываем
    # только это одно выражение.
    chosen = {expr_name: exprs[expr_name] * float(strength)}

    gui.Application.instance.initialize()
    _setup_font()
    DebugViewer(flame, head, chosen, p)
    gui.Application.instance.run()


def _start_dialog(head_names, expr_names):
    """Маленькое tkinter-окно ДО Open3D: выбор головы, выражения, силы.
    Возвращает (head_index, expr_name, strength) или None при отмене."""
    import tkinter as tk
    from tkinter import ttk
    res = {}
    root = tk.Tk()
    root.title("Pipeline debug — setup")
    root.geometry("360x230")
    fr = ttk.Frame(root, padding=12); fr.pack(fill="both", expand=True)

    ttk.Label(fr, text="Head:").grid(row=0, column=0, sticky="w", pady=4)
    v_head = ttk.Combobox(fr, values=head_names, state="readonly", width=26)
    v_head.current(0); v_head.grid(row=0, column=1, pady=4)

    ttk.Label(fr, text="Expression:").grid(row=1, column=0, sticky="w", pady=4)
    v_expr = ttk.Combobox(fr, values=expr_names, state="readonly", width=26)
    v_expr.current(0); v_expr.grid(row=1, column=1, pady=4)

    ttk.Label(fr, text="Strength:").grid(row=2, column=0, sticky="w", pady=4)
    v_str = tk.DoubleVar(value=1.0)
    tk.Scale(fr, from_=0.0, to=3.0, resolution=0.1, orient="horizontal",
             variable=v_str, length=200).grid(row=2, column=1, pady=4)

    def ok():
        res['v'] = (v_head.current(), v_expr.get(), v_str.get())
        root.destroy()
    ttk.Button(fr, text="Start ▶", command=ok).grid(
        row=3, column=0, columnspan=2, pady=12, sticky="ew")
    root.protocol("WM_DELETE_WINDOW", root.destroy)
    root.mainloop()
    return res.get('v')


if __name__ == "__main__":
    main()
