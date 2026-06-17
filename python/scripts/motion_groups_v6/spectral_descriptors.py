#!/usr/bin/env python3
"""
Spectral Descriptors Viewer — HKS / WKS на двух головах рядом (FLAME + FBX).

Слева две головы (FLAME слева, FBX справа). Справа панель:
  • выбор дескриптора (HKS / WKS) и его настройки;
  • кнопка «Считать сигнатуры» — раскрашивает обе головы по выбранному каналу
    дескриптора;
  • выбор двух точек (Shift+click): первая — на FLAME, вторая — на FBX;
  • маленькое окошко выводит значения сигнатур в этих точках.

Дескрипторы строятся из собственных функций Лапласиана (L·v = λ·M·v), которые
уже умеет считать debug_head1_pipeline.compute_spectrum.

  HKS(x, t) = Σ_k exp(-t·λ_k) · φ_k(x)^2
  WKS(x, e) = Σ_k φ_k(x)^2 · exp(-(e - log λ_k)^2 / (2σ^2))   (нормировано)

Запуск (обычно через кнопку в основном GUI, но можно и напрямую):
  cd /Users/kuzix/Documents/GitHub/Muscle-autoskinner
  source .venv/bin/activate
  python python/scripts/motion_groups_v6/spectral_descriptors.py \
      [--flame <pkl>] [--fbx <mesh>] [--n-eigs 120]
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering

# переиспользуем утилиты основного пайплайна
import debug_head1_pipeline as pipe


# ── дескрипторы ────────────────────────────────────────────────────────────────

def hks(eigvals, eigvecs, n_t=100, t_min=None, t_max=None):
    """Heat Kernel Signature. Возвращает (S (N, n_t), times (n_t,)).
    HKS(x,t) = Σ_{k≥1} exp(-t·λ_k)·φ_k(x)^2."""
    lam = np.clip(np.asarray(eigvals, float), 0.0, None)
    phi = np.asarray(eigvecs, float)
    nz = lam > 1e-8                         # отбрасываем нулевую моду
    lam = lam[nz]; phi = phi[:, nz]
    if t_min is None:
        t_min = 4 * np.log(10) / lam.max()
    if t_max is None:
        t_max = 4 * np.log(10) / lam.min()
    times = np.logspace(np.log10(t_min), np.log10(t_max), n_t)
    phi2 = phi ** 2                          # (N, K)
    E = np.exp(-np.outer(times, lam))        # (n_t, K)
    S = phi2 @ E.T                           # (N, n_t)
    return S, times


def wks(eigvals, eigvecs, n_e=100, sigma_scale=7.0):
    """Wave Kernel Signature. Возвращает (S (N, n_e), energies (n_e,)).
    WKS(x,e) ∝ Σ_k φ_k(x)^2 · exp(-(e - log λ_k)^2 / (2σ^2))."""
    lam = np.clip(np.asarray(eigvals, float), 0.0, None)
    phi = np.asarray(eigvecs, float)
    nz = lam > 1e-8
    lam = lam[nz]; phi = phi[:, nz]
    log_lam = np.log(lam)
    e_min, e_max = log_lam.min(), log_lam.max()
    energies = np.linspace(e_min, e_max, n_e)
    delta = (e_max - e_min) / n_e
    sigma = sigma_scale * delta
    phi2 = phi ** 2                          # (N, K)
    # gauss (n_e, K)
    G = np.exp(-((energies[:, None] - log_lam[None, :]) ** 2) / (2 * sigma ** 2))
    norm = G.sum(1).clip(1e-12)             # нормировка по энергии
    S = (phi2 @ G.T) / norm[None, :]        # (N, n_e)
    return S, energies


def _plot_signatures(curves, width=320, height=180, channel=None):
    """Рисуем кривые сигнатур в RGB-картинку (без matplotlib).

    curves: список (label, values (C,), rgb (3,)). channel: индекс вертикальной
    линии-курсора. Возвращает uint8 (H, W, 3)."""
    img = np.full((height, width, 3), 255, np.uint8)
    # рамка
    img[0, :] = img[-1, :] = img[:, 0] = img[:, -1] = 200
    pad_l, pad_r, pad_t, pad_b = 4, 4, 4, 4
    pw = width - pad_l - pad_r
    ph = height - pad_t - pad_b
    valid = [c for c in curves if c[1] is not None and len(c[1]) > 1]
    if not valid:
        return img
    # общий вертикальный масштаб (по всем кривым)
    allv = np.concatenate([np.asarray(v, float) for _, v, _ in valid])
    vmin, vmax = float(allv.min()), float(allv.max())
    rng = vmax - vmin if vmax > vmin else 1.0
    C = max(len(v) for _, v, _ in valid)

    def px(i, C):
        return pad_l + int(round(i / max(C - 1, 1) * (pw - 1)))

    def py(val):
        f = (val - vmin) / rng
        return pad_t + int(round((1.0 - f) * (ph - 1)))

    # курсор канала
    if channel is not None and 0 <= channel < C:
        cx = px(channel, C)
        img[pad_t:pad_t + ph, cx] = np.array([180, 180, 180], np.uint8)

    def draw_line(x0, y0, x1, y1, col):
        n = max(abs(x1 - x0), abs(y1 - y0), 1)
        for s in range(n + 1):
            x = int(round(x0 + (x1 - x0) * s / n))
            y = int(round(y0 + (y1 - y0) * s / n))
            if 0 <= y < height and 0 <= x < width:
                img[y, x] = col

    for _label, vals, rgb in valid:
        vals = np.asarray(vals, float)
        col = np.array([int(c * 255) for c in rgb], np.uint8)
        Ci = len(vals)
        for i in range(Ci - 1):
            draw_line(px(i, Ci), py(vals[i]),
                      px(i + 1, Ci), py(vals[i + 1]), col)
    return img


# ── functional maps ────────────────────────────────────────────────────────────

def _normalize_desc(desc):
    """Нормируем КАЖДЫЙ канал дескриптора (столбец) к единичной L2-норме.
    Без этого HKS/WKS на двух головах имеют разный абсолютный масштаб (разные
    площади треугольников) → least-squares карты перекошен → слипание зон."""
    desc = np.asarray(desc, float)
    nrm = np.linalg.norm(desc, axis=0, keepdims=True)
    return desc / np.clip(nrm, 1e-12, None)


def _spectral_coeffs(desc, eigvecs, mass, k):
    """Коэффициенты дескрипторов в спектральном базисе: A = Φ^T · M · desc.
    desc (N, C), eigvecs (N, K) → (k, C). Дескрипторы поканально нормированы."""
    Phi = eigvecs[:, :k]
    return Phi.T @ (mass[:, None] * _normalize_desc(desc))


def compute_functional_map(src, dst, k=None):
    """Functional map C (k×k), переводящая функции SRC → DST в спектр. базисе.

    Классика (Ovsjanikov 2012): по парам дескрипторов решаем
    C = argmin ‖C·A_src − A_dst‖²  (least squares), где A = спектр. коэф-ты
    дескрипторов. Доп. член — диагональная согласованность по λ (commute с
    Лапласианом) для регуляризации. src/dst — HeadData с посчитанными .S."""
    if src.S is None or dst.S is None:
        return None
    k = int(k or min(src.eigvecs.shape[1], dst.eigvecs.shape[1]))
    k = min(k, src.eigvecs.shape[1], dst.eigvecs.shape[1])
    A = _spectral_coeffs(src.S, src.eigvecs, src.mass, k)      # (k, C)
    B = _spectral_coeffs(dst.S, dst.eigvecs, dst.mass, k)      # (k, C)
    # C·A ≈ B  →  C = B·A^+ (через решение по строкам с регуляризацией Тихонова)
    lam = 1e-3
    AAt = A @ A.T + lam * np.eye(k)
    C = (B @ A.T) @ np.linalg.inv(AAt)                         # (k, k)
    return C


def fmap_point_correspondence(C, src, dst, k):
    """Точечное соответствие dst→src из functional map C.
    Сопоставляем строки Φ_dst и (Φ_src·C^T) в спектре, NN. Возвращает idx (Ndst,)
    — для каждой вершины DST индекс соответствующей вершины SRC.

    Строки нормируются на единичную длину (точки на «спектральной сфере») — NN
    сравнивает направления, а не абсолют. величины собственных функций; это
    заметно снижает слипание (зоны на DST)."""
    from scipy.spatial import cKDTree
    Phi_s = src.eigvecs[:, :k]            # (Ns, k)
    Phi_d = dst.eigvecs[:, :k]            # (Nd, k)
    emb_s = Phi_s @ C.T                   # переносим src в базис dst
    emb_s = emb_s / np.clip(np.linalg.norm(emb_s, axis=1, keepdims=True),
                            1e-12, None)
    emb_d = Phi_d / np.clip(np.linalg.norm(Phi_d, axis=1, keepdims=True),
                            1e-12, None)
    tree = cKDTree(emb_s)
    _, idx = tree.query(emb_d)
    return idx


def _fmap_from_correspondence(idx, src, dst, k):
    """Обратный шаг ZoomOut: по точечному соответствию dst→src восстанавливаем
    functional map C (k×k): C = Φ_dstᵀ·M_dst · P · Φ_src, где P — соответствие.
    Реализуем как C = Φ_dst(:k)ᵀ · M_dst · Φ_src[idx, :k]."""
    Phi_d = dst.eigvecs[:, :k]                    # (Nd, k)
    Phi_s_mapped = src.eigvecs[idx, :k]           # (Nd, k) — src в порядке dst
    return Phi_d.T @ (dst.mass[:, None] * Phi_s_mapped)   # (k, k)


def zoomout_refine(C0, src, dst, k0, k_max, step=5):
    """ZoomOut (Melzi 2019): итеративно уточняем карту, наращивая базис k0→k_max.

    Каждый шаг: C(k) → точечное соответствие → C(k+step) из соответствия.
    Резко улучшает биективность (сырой LS-map «слипается» в зоны). Возвращает
    (C_final (k_max×k_max), idx_final (Ndst,))."""
    k = k0
    C = C0[:k, :k].copy()
    kmax = min(k_max, src.eigvecs.shape[1], dst.eigvecs.shape[1])
    while True:
        idx = fmap_point_correspondence(C, src, dst, k)
        k_next = min(k + step, kmax)
        C = _fmap_from_correspondence(idx, src, dst, k_next)
        k = k_next
        if k >= kmax:
            break
    idx = fmap_point_correspondence(C, src, dst, k)
    return C, idx


# ── одна голова: геометрия + спектр + рендер ───────────────────────────────────

class HeadData:
    def __init__(self, name, verts, faces, n_eigs):
        self.name = name
        self.verts = np.ascontiguousarray(verts, dtype=np.float64)
        self.faces = np.ascontiguousarray(faces, dtype=np.int32)
        print(f"[{name}] спектр: {len(verts)} верш., k={n_eigs} ...")
        self.eigvals, self.eigvecs = pipe.compute_spectrum(
            self.verts, faces.astype(np.int64), n_eigs=n_eigs)
        _L, MM = pipe.build_operators(self.verts, faces.astype(np.int64))
        self.mass = np.asarray(MM.diagonal(), dtype=np.float64)   # (N,) площади
        self.S = None            # текущая матрица сигнатур (N, C)
        self.axis = None         # times / energies оси канала
        self.picked = None       # индекс выбранной вершины
        self._tree = None
        self.fmap_color = None   # цвет, перенесённый functional-map (N,3) или None

    def kdtree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.verts)
        return self._tree

    def compute(self, kind, **kw):
        if kind == "HKS":
            self.S, self.axis = hks(self.eigvals, self.eigvecs, **kw)
        else:
            self.S, self.axis = wks(self.eigvals, self.eigvecs, **kw)

    def color_for_channel(self, ch):
        """Цвет вершин по каналу ch (CMAP_HEAT, лог-нормировка)."""
        v = self.S[:, int(ch)].copy()
        v = np.log1p(np.clip(v - v.min(), 0, None))
        v = v / (v.max() + 1e-12)
        return pipe.to_colors(v, pipe.CMAP_HEAT)


# ── viewer ─────────────────────────────────────────────────────────────────────

class SpectralViewer:
    def __init__(self, flame, fbx, n_eigs):
        self.n_eigs = n_eigs
        self.heads = [HeadData("FLAME", *flame, n_eigs=n_eigs),
                      HeadData("FBX", *fbx, n_eigs=n_eigs)]
        self.kind = "HKS"
        self.channel = 0
        self._build_gui()
        self._recompute()

    # --- GUI ---
    def _build_gui(self):
        self.app = gui.Application.instance
        self.window = self.app.create_window(
            "Spectral Descriptors — HKS / WKS", 1500, 900)
        w = self.window
        em = w.theme.font_size

        # две сцены
        self.scenes = []
        for hd in self.heads:
            sc = gui.SceneWidget()
            sc.scene = rendering.Open3DScene(w.renderer)
            sc.scene.set_background([0.1, 0.1, 0.12, 1.0])
            sc.set_on_mouse(self._make_mouse_cb(len(self.scenes)))
            self.scenes.append(sc)
            w.add_child(sc)

        # правая панель
        panel = gui.Vert(0.5 * em, gui.Margins(em, em, em, em))
        panel.add_child(gui.Label("Дескриптор"))
        self.combo = gui.Combobox()
        self.combo.add_item("HKS")
        self.combo.add_item("WKS")
        self.combo.set_on_selection_changed(self._on_kind)
        panel.add_child(self.combo)

        # настройки
        panel.add_fixed(0.5 * em)
        panel.add_child(gui.Label("k собственных функций"))
        self.in_eigs = gui.NumberEdit(gui.NumberEdit.INT)
        self.in_eigs.int_value = self.n_eigs
        panel.add_child(self.in_eigs)

        panel.add_child(gui.Label("число каналов (t / energy)"))
        self.in_nch = gui.NumberEdit(gui.NumberEdit.INT)
        self.in_nch.int_value = 100
        panel.add_child(self.in_nch)

        panel.add_child(gui.Label("WKS sigma scale"))
        self.in_sigma = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.in_sigma.double_value = 7.0
        panel.add_child(self.in_sigma)

        panel.add_fixed(0.3 * em)
        panel.add_child(gui.Label("Канал для раскраски"))
        self.sl_ch = gui.Slider(gui.Slider.INT)
        self.sl_ch.set_limits(0, 99)
        self.sl_ch.int_value = 0
        self.sl_ch.set_on_value_changed(self._on_channel)
        panel.add_child(self.sl_ch)
        self.lbl_ch = gui.Label("канал 0")
        panel.add_child(self.lbl_ch)

        btn = gui.Button("Считать сигнатуры")
        btn.set_on_clicked(self._on_compute)
        panel.add_fixed(0.5 * em)
        panel.add_child(btn)

        # ── Functional maps ──
        panel.add_fixed(em)
        panel.add_child(gui.Label("── Functional map ──"))
        panel.add_child(gui.Label("k базиса для карты"))
        self.in_fmk = gui.NumberEdit(gui.NumberEdit.INT)
        self.in_fmk.int_value = 40
        panel.add_child(self.in_fmk)
        self.cb_zoomout = gui.Checkbox("ZoomOut уточнение (биективность)")
        self.cb_zoomout.checked = True
        panel.add_child(self.cb_zoomout)
        panel.add_child(gui.Label("ZoomOut k_max"))
        self.in_zmax = gui.NumberEdit(gui.NumberEdit.INT)
        self.in_zmax.int_value = 80
        panel.add_child(self.in_zmax)
        btn_fm = gui.Button("Построить карту FLAME→FBX")
        btn_fm.set_on_clicked(self._on_fmap)
        panel.add_child(btn_fm)
        btn_fm_clear = gui.Button("Сброс карты (обычная раскраска)")
        btn_fm_clear.set_on_clicked(self._on_fmap_clear)
        panel.add_child(btn_fm_clear)
        self.lbl_fmap = gui.Label("карта не построена")
        panel.add_child(self.lbl_fmap)

        # окошко значений
        panel.add_fixed(em)
        panel.add_child(gui.Label("── Выбранные точки ──"))
        panel.add_child(gui.Label("Shift+click: 1) FLAME  2) FBX"))
        self.lbl_vals = gui.Label("точки не выбраны")
        self.lbl_vals.text = "точки не выбраны"
        panel.add_child(self.lbl_vals)

        self.lbl_dist = gui.Label("")
        panel.add_child(self.lbl_dist)

        # график сигнатур выбранных точек (FLAME оранж / FBX синий)
        panel.add_fixed(0.5 * em)
        panel.add_child(gui.Label("График сигнатур (оранж=FLAME, син=FBX)"))
        self.plot = gui.ImageWidget(
            o3d.geometry.Image(np.ascontiguousarray(_plot_signatures([]))))
        panel.add_child(self.plot)

        self.panel = panel
        w.add_child(panel)
        w.set_on_layout(self._on_layout)

    def _on_layout(self, ctx):
        r = self.window.content_rect
        pw = min(20 * self.window.theme.font_size, r.width * 0.30)
        sw = (r.width - pw) / 2
        self.scenes[0].frame = gui.Rect(r.x, r.y, sw, r.height)
        self.scenes[1].frame = gui.Rect(r.x + sw, r.y, sw, r.height)
        self.panel.frame = gui.Rect(r.get_right() - pw, r.y, pw, r.height)

    # --- mesh render ---
    def _render_head(self, i):
        hd = self.heads[i]
        sc = self.scenes[i]
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(hd.verts),
            o3d.utility.Vector3iVector(hd.faces))
        mesh.compute_vertex_normals()
        if hd.fmap_color is not None:                  # раскраска по карте
            mesh.vertex_colors = o3d.utility.Vector3dVector(hd.fmap_color)
        elif hd.S is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(
                hd.color_for_channel(self.channel))
        else:
            mesh.paint_uniform_color([0.8, 0.75, 0.7])
        mat = rendering.MaterialRecord(); mat.shader = "defaultLit"
        sc.scene.clear_geometry()
        sc.scene.add_geometry(f"head{i}", mesh, mat)
        # маркер выбранной точки
        if hd.picked is not None:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.012)
            sph.translate(hd.verts[hd.picked])
            sph.paint_uniform_color([0.1, 1.0, 0.1])
            sph.compute_vertex_normals()
            m2 = rendering.MaterialRecord(); m2.shader = "defaultLit"
            sc.scene.add_geometry(f"pick{i}", sph, m2)
        bbox = o3d.geometry.AxisAlignedBoundingBox(
            hd.verts.min(0) - 0.05, hd.verts.max(0) + 0.05)
        sc.setup_camera(50.0, bbox, bbox.get_center())

    def _recompute(self):
        for i in range(2):
            self._render_head(i)

    # --- callbacks ---
    def _on_kind(self, text, idx):
        self.kind = text

    def _on_channel(self, val):
        self.channel = int(val)
        self.lbl_ch.text = f"канал {self.channel}"
        for i in range(2):
            self._render_head(i)
        self._refresh_values()

    def _on_compute(self):
        n_eigs = int(self.in_eigs.int_value)
        n_ch = int(self.in_nch.int_value)
        sigma = float(self.in_sigma.double_value)
        # пересчёт спектра, если k изменился
        if n_eigs != self.n_eigs:
            self.n_eigs = n_eigs
            for hd in self.heads:
                hd.eigvals, hd.eigvecs = pipe.compute_spectrum(
                    hd.verts, hd.faces.astype(np.int64), n_eigs=n_eigs)
        kw = dict(n_t=n_ch) if self.kind == "HKS" else dict(
            n_e=n_ch, sigma_scale=sigma)
        for hd in self.heads:
            hd.compute(self.kind, **kw)
        self.sl_ch.set_limits(0, n_ch - 1)
        if self.channel >= n_ch:
            self.channel = n_ch - 1
            self.sl_ch.int_value = self.channel
        print(f"Сигнатуры {self.kind} посчитаны ({n_ch} каналов).")
        for i in range(2):
            self._render_head(i)
        self._refresh_values()

    def _on_fmap(self):
        a, b = self.heads          # a=FLAME (src), b=FBX (dst)
        if a.S is None or b.S is None:
            self.lbl_fmap.text = "сначала «Считать сигнатуры»"
            return
        k = int(self.in_fmk.int_value)
        k = min(k, a.eigvecs.shape[1], b.eigvecs.shape[1])
        C = compute_functional_map(a, b, k=k)
        if C is None:
            self.lbl_fmap.text = "не удалось построить карту"
            return
        # точечное соответствие dst(FBX)→src(FLAME) + опц. ZoomOut
        if self.cb_zoomout.checked:
            k_max = int(self.in_zmax.int_value)
            print(f"ZoomOut {k}→{k_max} ...")
            C, idx = zoomout_refine(C, a, b, k, k_max, step=5)
            k = C.shape[0]
        else:
            idx = fmap_point_correspondence(C, a, b, k)
        # цвет-«индикатор» по координатам FLAME → переносим на FBX по карте
        ref = a.verts - a.verts.min(0)
        ref = ref / (ref.max(0) + 1e-12)        # XYZ → RGB на FLAME
        a.fmap_color = ref
        b.fmap_color = ref[idx]                 # тот же цвет в соответств. точках
        # качество: покрытие (биективность) — % уникальных вершин FLAME
        cover = 100.0 * len(np.unique(idx)) / max(len(a.verts), 1)
        self.lbl_fmap.text = (f"карта k={k}, покрытие={cover:.0f}%\n"
                              f"одинаковый цвет = соответствие")
        print(f"Functional map: k={k}, покрытие={cover:.1f}% "
              f"({len(np.unique(idx))}/{len(a.verts)} верш. FLAME)")
        for i in range(2):
            self._render_head(i)

    def _on_fmap_clear(self):
        for hd in self.heads:
            hd.fmap_color = None
        self.lbl_fmap.text = "карта сброшена"
        for i in range(2):
            self._render_head(i)

    def _make_mouse_cb(self, i):
        def cb(event):
            if (event.type == gui.MouseEvent.Type.BUTTON_DOWN
                    and event.is_modifier_down(gui.KeyModifier.SHIFT)):
                self._pick(i, event.x, event.y)
                return gui.Widget.EventCallbackResult.HANDLED
            return gui.Widget.EventCallbackResult.IGNORED
        return cb

    def _pick(self, i, mx, my):
        sc = self.scenes[i]
        frame = sc.frame
        x = mx - frame.x
        y = my - frame.y

        def after_depth(depth_image):
            depth = np.asarray(depth_image)
            yy = int(np.clip(y, 0, depth.shape[0] - 1))
            xx = int(np.clip(x, 0, depth.shape[1] - 1))
            d = depth[yy, xx]
            if d >= 1.0:                       # клик мимо геометрии
                return
            world = sc.scene.camera.unproject(
                x, y, d, frame.width, frame.height)
            p = np.array([world[0], world[1], world[2]])
            _, idx = self.heads[i].kdtree().query(p)
            self.heads[i].picked = int(idx)
            self.app.post_to_main_thread(
                self.window, lambda: (self._render_head(i),
                                      self._refresh_values()))

        sc.scene.scene.render_to_depth_image(after_depth)

    def _refresh_values(self):
        parts = []
        for hd in self.heads:
            if hd.picked is None:
                parts.append(f"{hd.name}: —")
            elif hd.S is None:
                parts.append(f"{hd.name}: v{hd.picked} (нет сигнатур)")
            else:
                val = hd.S[hd.picked, self.channel]
                ax = hd.axis[self.channel]
                axname = "t" if self.kind == "HKS" else "e"
                parts.append(f"{hd.name}: v{hd.picked}  "
                             f"{self.kind}[{self.channel}]={val:.4e}  "
                             f"({axname}={ax:.3g})")
        self.lbl_vals.text = "\n".join(parts)

        # сравнение полных сигнатур двух точек
        a, b = self.heads
        if (a.picked is not None and b.picked is not None
                and a.S is not None and b.S is not None):
            sa = a.S[a.picked]; sb = b.S[b.picked]
            n = min(len(sa), len(sb))
            d = float(np.linalg.norm(sa[:n] - sb[:n]))
            ca = sa[:n] / (np.linalg.norm(sa[:n]) + 1e-12)
            cb = sb[:n] / (np.linalg.norm(sb[:n]) + 1e-12)
            cos = float(ca @ cb)
            self.lbl_dist.text = (f"‖Δ сигнатур‖ = {d:.4e}\n"
                                  f"cos-схожесть = {cos:.4f}")
        else:
            self.lbl_dist.text = ""

        self._refresh_plot()

    def _refresh_plot(self):
        a, b = self.heads
        curves = []
        if a.picked is not None and a.S is not None:
            curves.append(("FLAME", a.S[a.picked], (0.93, 0.45, 0.13)))
        if b.picked is not None and b.S is not None:
            curves.append(("FBX", b.S[b.picked], (0.13, 0.45, 0.95)))
        img = _plot_signatures(curves, channel=self.channel)
        self.plot.update_image(o3d.geometry.Image(np.ascontiguousarray(img)))


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


def _load_fbx_or_default(fbx_path):
    if fbx_path and Path(fbx_path).exists():
        v, f = pipe.load_custom_mesh(fbx_path)
        return pipe.normalize_bbox(v), f
    return None


def main():
    ap = argparse.ArgumentParser(description="Spectral descriptors viewer")
    ap.add_argument("--flame", default=pipe.FLAME_PKL)
    ap.add_argument("--fbx", default="")
    ap.add_argument("--n-eigs", type=int, default=120)
    args = ap.parse_args()

    if not Path(args.flame).exists():
        raise SystemExit(f"FLAME pkl не найден: {args.flame}")

    v_t, sd, faces = pipe.load_flame(args.flame)
    flame_v = pipe.normalize_bbox(pipe.apply_betas(v_t, sd, {}))
    flame = (flame_v, faces)

    fbx = _load_fbx_or_default(args.fbx)
    if fbx is None:
        print("FBX не задан/не найден → вторая голова = FLAME (для сравнения).")
        fbx = (flame_v.copy(), faces.copy())

    gui.Application.instance.initialize()
    _setup_cyrillic_font()
    SpectralViewer(flame, fbx, n_eigs=args.n_eigs)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
