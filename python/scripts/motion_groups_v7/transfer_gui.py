#!/usr/bin/env python3
r"""
motion_groups_v7 — GUI for batch transfer of an expression set onto a folder
of heads.

Single screen: pick reference HDF5 (neutral + expressions), heads folder and
output HDF5; tune parameters; hit "Transfer" → get one HDF5 with every head ×
every expression. No debug windows.

The code is cross-platform; only the launch differs per OS.

Run (macOS / Linux):
  cd <repo root>
  source .venv/bin/activate
  python python/scripts/motion_groups_v7/transfer_gui.py

Run (Windows):
  # easiest: double-click python\scripts\motion_groups_v7\run_transfer_gui.bat
  # (it builds .venv-win + installs deps on first run, then launches this GUI)
  # or manually from PowerShell at the repo root:
  py -3.9 -m venv .venv-win
  .\.venv-win\Scripts\Activate.ps1
  pip install -r python\scripts\motion_groups_v7\requirements-windows.txt
  python python\scripts\motion_groups_v7\transfer_gui.py
"""
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

import transfer_engine as eng


class _ToolTip:
    """Hover tooltip: shows a small help popup after a short delay."""

    def __init__(self, widget, text, delay=600):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._after = None
        self._tip = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)
        widget.bind("<ButtonPress>", self._hide)

    def _schedule(self, _e=None):
        self._cancel()
        self._after = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._after is not None:
            self.widget.after_cancel(self._after)
            self._after = None

    def _show(self):
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(tw, text=self.text, justify="left",
                       background="#ffffe0", relief="solid", borderwidth=1,
                       wraplength=320, font=("", 9), padx=6, pady=4)
        lbl.pack()

    def _hide(self, _e=None):
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class TransferGUI:
    def __init__(self, root):
        self.root = root
        root.title("v7 — Transfer expression set → HDF5")
        self._build()

    # ── widget builders (with optional hover tooltip) ──
    def _path_row(self, parent, label, default, is_dir=False, save=False,
                  ext="", tip=""):
        fr = ttk.Frame(parent)
        lab = ttk.Label(fr, text=label, width=22, anchor="w")
        lab.pack(side="left")
        var = tk.StringVar(value=default)
        ent = ttk.Entry(fr, textvariable=var)
        ent.pack(side="left", fill="x", expand=True, padx=4)

        def browse():
            if is_dir:
                p = filedialog.askdirectory(title=label)
            elif save:
                p = filedialog.asksaveasfilename(
                    title=label, defaultextension=ext,
                    filetypes=[("HDF5", "*.h5"), ("all", "*.*")])
            else:
                p = filedialog.askopenfilename(
                    title=label,
                    filetypes=[("HDF5", "*.h5"), ("all", "*.*")])
            if p:
                var.set(p)
        ttk.Button(fr, text="...", width=3, command=browse).pack(side="left")
        fr.pack(fill="x", pady=2)
        if tip:
            _ToolTip(lab, tip); _ToolTip(ent, tip)
        return var

    def _num(self, parent, label, default, tip=""):
        fr = ttk.Frame(parent)
        lab = ttk.Label(fr, text=label, width=30, anchor="w"); lab.pack(side="left")
        var = tk.StringVar(value=str(default))
        ent = ttk.Entry(fr, textvariable=var, width=12); ent.pack(side="left")
        fr.pack(fill="x", pady=1)
        if tip:
            _ToolTip(lab, tip); _ToolTip(ent, tip)
        return var

    def _combo(self, parent, label, values, default, tip=""):
        fr = ttk.Frame(parent)
        lab = ttk.Label(fr, text=label, width=30, anchor="w"); lab.pack(side="left")
        var = tk.StringVar(value=default)
        cb = ttk.Combobox(fr, textvariable=var, values=values, width=14,
                          state="readonly"); cb.pack(side="left")
        fr.pack(fill="x", pady=1)
        if tip:
            _ToolTip(lab, tip); _ToolTip(cb, tip)
        return var

    def _check(self, parent, label, default, tip=""):
        var = tk.BooleanVar(value=default)
        cb = ttk.Checkbutton(parent, text=label, variable=var)
        cb.pack(anchor="w", pady=1)
        if tip:
            _ToolTip(cb, tip)
        return var

    def _section(self, parent, title):
        ttk.Label(parent, text=title, font=("", 11, "bold")).pack(
            anchor="w", pady=(10, 2))

    def _build(self):
        d = eng.DEFAULT_PARAMS

        # ── скроллируемый контейнер (всё содержимое в нём) ──
        container = ttk.Frame(self.root)
        container.pack(fill="both", expand=True)
        canvas = tk.Canvas(container, highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical",
                            command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        outer = ttk.Frame(canvas, padding=12)
        win = canvas.create_window((0, 0), window=outer, anchor="nw")

        def _on_frame_cfg(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        outer.bind("<Configure>", _on_frame_cfg)

        def _on_canvas_cfg(e):                    # тянем ширину фрейма по канвасу
            canvas.itemconfigure(win, width=e.width)
        canvas.bind("<Configure>", _on_canvas_cfg)

        # колесо мыши (macOS/Win/Linux)
        def _on_wheel(e):
            delta = -1 * (e.delta) if abs(e.delta) >= 1 else 0
            if e.num == 5 or e.delta < 0:
                canvas.yview_scroll(1, "units")
            elif e.num == 4 or e.delta > 0:
                canvas.yview_scroll(-1, "units")
        canvas.bind_all("<MouseWheel>", _on_wheel)
        canvas.bind_all("<Button-4>", _on_wheel)
        canvas.bind_all("<Button-5>", _on_wheel)

        # ── presets ──
        self._section(outer, "Preset")
        pr = ttk.Frame(outer); pr.pack(fill="x")
        ttk.Label(pr, text="preset:", width=8, anchor="w").pack(side="left")
        self.preset_box = ttk.Combobox(pr, width=22, state="readonly")
        self.preset_box.pack(side="left", padx=4)
        self.preset_box.bind("<<ComboboxSelected>>",
                             lambda e: self._load_preset())
        ttk.Button(pr, text="Save as…", command=self._save_preset).pack(
            side="left", padx=2)
        ttk.Button(pr, text="Delete", command=self._delete_preset).pack(
            side="left", padx=2)
        self._refresh_preset_list()

        # ── paths ──
        self._section(outer, "Paths")
        self.v_ref = self._path_row(
            outer, "Reference HDF5 (expr):", "",
            tip="Reference HDF5 with the neutral head and the expression set "
                "(δ per expression, optional muscle activations). Source of the "
                "emotions to transfer.")
        self.v_dir = self._path_row(
            outer, "Heads folder (FBX/OBJ):", "", is_dir=True,
            tip="Folder with target heads (.fbx/.obj/.ply/.stl). Every file is "
                "treated as a separate head to receive all expressions.")
        self.v_out = self._path_row(
            outer, "Output HDF5:", "data/transferred.h5", save=True, ext=".h5",
            tip="Output dataset: every head × every expression. Per head: "
                "neutral, faces, expr/<name>/delta. Activations stored once.")

        # head index range (which heads from the folder to process)
        rr = ttk.Frame(outer); rr.pack(fill="x", pady=2)
        lab_r = ttk.Label(rr, text="Heads range (from..to, empty=all):",
                          width=30, anchor="w")
        lab_r.pack(side="left")
        self.v_hfrom = tk.StringVar(value="")
        self.v_hto = tk.StringVar(value="")
        ttk.Entry(rr, textvariable=self.v_hfrom, width=8).pack(side="left", padx=2)
        ttk.Label(rr, text="..").pack(side="left")
        ttk.Entry(rr, textvariable=self.v_hto, width=8).pack(side="left", padx=2)
        _ToolTip(lab_r, "Process only heads with index in [from..to] "
                        "(0-based, inclusive) from the sorted folder list. "
                        "Empty = from start / to end. Use to split a big "
                        "dataset into chunks.")
        import os as _os
        lab_w = ttk.Label(rr, text="workers:"); lab_w.pack(side="left",
                                                           padx=(12, 2))
        self.v_workers = tk.StringVar(value=str(max(_os.cpu_count() or 1, 1)))
        ttk.Entry(rr, textvariable=self.v_workers, width=5).pack(side="left")
        _ToolTip(lab_w, "Parallel processes (heads are independent). 1 = "
                        "sequential. Default = CPU cores. Big speedup on many "
                        "heads.")

        # two-column parameter area
        cols = ttk.Frame(outer); cols.pack(fill="x")
        left = ttk.Frame(cols); left.pack(side="left", fill="x", expand=True,
                                          padx=(0, 10))
        right = ttk.Frame(cols); right.pack(side="left", fill="x", expand=True)

        # 1) Diffusion
        self._section(left, "1. Diffusion")
        self.v_time = self._num(
            left, "time (diffusion t):", d['time'],
            tip="Total heat-diffusion time. Larger = heat spreads further from "
                "each anchor → bigger influence zones.")
        self.v_steps = self._num(
            left, "steps (solver):", d['steps'],
            tip="Implicit solver substeps for the diffusion. More steps = more "
                "accurate spread (no effect on animation, batch is static).")

        # 2) Clustering
        self._section(left, "2. Clustering")
        self.v_nclu = self._num(
            left, "max clusters:", d['n_clusters'],
            tip="Max motion clusters (sub-groups) per anchor zone.")
        self.v_hthr = self._num(
            left, "heat threshold:", d['heat_threshold'],
            tip="Vertices below this normalized heat are considered inactive "
                "(not part of any zone / not clustered).")
        self.v_pw = self._num(
            left, "position weight:", d['position_weight'],
            tip="Weight of vertex position vs. motion (δ) in clustering. 0 = "
                "cluster purely by motion direction.")
        self.v_cmeth = self._combo(
            left, "method:", ["kmeans", "agglomerative"], d['clustering_method'],
            tip="Clustering algorithm: kmeans (auto K by zone size) or "
                "agglomerative (by similarity threshold).")
        self.v_csim = self._num(
            left, "similarity threshold:", d['cluster_similarity_threshold'],
            tip="Agglomerative only: merge threshold in normalized feature "
                "space. Lower = more, smaller clusters.")

        # 3) Smoothing
        self._section(left, "3. Smoothing")
        self.v_smi = self._num(
            left, "δ smooth iters:", d['smooth_iters'],
            tip="Laplacian smoothing iterations on the transferred δ field "
                "(removes seams/noise between zones).")
        self.v_sma = self._num(
            left, "δ smooth alpha:", d['smooth_alpha'],
            tip="Smoothing strength per iteration (0..1). Higher = smoother but "
                "may wash out amplitude.")
        self.v_sgrp = self._check(
            left, "smooth transferred groups", d['smooth_transferred_groups'],
            tip="Majority-vote smoothing of discrete group labels across mesh "
                "faces (removes speckles on zone seams).")
        self.v_gsi = self._num(
            left, "group smooth iters:", d['group_smooth_iters'],
            tip="Iterations of group-label majority-vote smoothing.")

        # 4) MediaPipe anchors
        self._section(right, "4. MediaPipe anchors")
        self.v_lm = self._num(
            right, "landmark indices:",
            ",".join(str(x) for x in d['landmarks']),
            tip="MediaPipe Face Mesh landmark indices (0..477) used as heat "
                "anchors. Auto-placed via render+raycast. e.g. 9,4,199.")

        # 5) Multi-t
        self._section(right, "5. Multi-t")
        self.v_mtt = self._num(
            right, "n_times:", d['multi_t_n_times'],
            tip="Number of diffusion time scales for multi-t heat enrichment "
                "(richer per-vertex features).")
        self.v_mte = self._num(
            right, "k_eigs:", d['multi_t_n_eigs'],
            tip="Number of Laplacian eigenfunctions for the spectral multi-t "
                "expansion. More = finer detail, slower.")
        self.v_mtm = self._check(
            right, "mask by single-t reach", d['multi_t_mask_by_single_t'],
            tip="Limit multi-t zones to the area reached by single-t diffusion "
                "(keeps zones local to the face).")

        # 6) UV
        self._section(right, "6. UV")
        self.v_uflat = self._check(
            right, "flat projection", d['uv_flat'],
            tip="Unwrap each zone by planar PCA projection (fast, may overlap "
                "on curved zones) instead of Tutte+ARAP.")
        self.v_uworld = self._check(
            right, "world-flat (Y up)", d['uv_world_orient'],
            tip="Planar projection along zone normal with world Y mapped to +V "
                "(deterministic orientation, no canonical-orient instability). "
                "Overrides flat.")
        self.v_uicp = self._check(
            right, "PCA+ICP align", d['uv_align_pca_icp'],
            tip="Extra similarity (PCA+ICP) alignment of each FBX island to the "
                "FLAME island before NN matching.")
        self.v_uinterp = self._check(
            right, "barycentric δ interp", d['uv_interp_delta'],
            tip="Transfer δ via barycentric interpolation over FLAME triangles "
                "(smooth) instead of discrete nearest-neighbor (stepped).")
        self.v_urel = self._combo(
            right, "relax method:",
            ["arap", "laplacian", "spring", "none"], d['uv_relax_method'],
            tip="UV island relaxation before transfer: arap (min distortion), "
                "laplacian (smooth, fixed border), spring (edge isometry), none.")
        self.v_ureli = self._num(
            right, "relax iters:", d['uv_relax_iters'],
            tip="Iterations for UV relaxation (0 = off).")
        self.v_urelad = self._check(
            right, "adaptive relax (distortion)",
            d.get('uv_relax_adaptive', False),
            tip="Re-relax each island in rounds while its distortion keeps "
                "dropping (symmetric-Dirichlet, 4 = isometric). Stops at target "
                "or when it plateaus. Keeps the best UV — fixes badly stretched "
                "islands automatically.")
        self.v_ureltg = self._num(
            right, "relax target:", d.get('uv_relax_target', 4.5),
            tip="Target distortion to stop the adaptive relax (symmetric "
                "Dirichlet; 4 = perfect isometry, higher = looser). Try 4.3-5.0.")

        # 7) Boundary heat-warp (как в v6)
        self._section(right, "7. Boundary warp")
        self.v_warp = self._check(
            right, "heat-warp boundary", d['uv_warp_heat'],
            tip="Pull each FBX UV-island boundary onto the FLAME island edge; "
                "interior follows via a heat field. Exact edge match (v6).")
        self.v_warpt = self._num(
            right, "warp heat t:", d['uv_warp_heat_t'],
            tip="Diffusion time of the warp heat field. Larger = smoother, more "
                "global influence of the boundary pull.")
        self.v_warpmd = self._num(
            right, "warp min dist:", d['uv_warp_min_dist'],
            tip="Max allowed gap point→edge after warp (0 = land exactly on the "
                "edge; >0 = keep this much clearance).")
        self.v_uwksmatch = self._check(
            right, "WKS landmark match", d.get('uv_wks_match', False),
            tip="Before transfer, compute WKS signature landmarks on the FLAME "
                "source and on each head; add them as extra warp anchors so "
                "matching anatomical zones (nose↔nose, etc.) align in UV. "
                "Boundaries are still pulled; islands without a landmark use "
                "boundary only. Uses the sig min/max/dist/smooth + WKS params "
                "from the preview panel.")

        # ── run ──
        ttk.Button(outer, text="🛠 Make reference HDF5",
                   command=self._open_make_ref).pack(fill="x", pady=(12, 0))
        ttk.Button(outer, text="🔬 Pipeline step debug",
                   command=self._open_debug).pack(fill="x", pady=(4, 0))
        ttk.Button(outer, text="🗂 Inspect HDF5 structure",
                   command=self._open_inspect).pack(fill="x", pady=(4, 0))
        ttk.Button(outer, text="🧠 Train skinning network",
                   command=self._open_train).pack(fill="x", pady=(4, 0))
        self.btn = ttk.Button(outer, text="▶ Transfer expressions",
                              command=self._run)
        self.btn.pack(fill="x", pady=(6, 4))
        self.pbar = ttk.Progressbar(outer, mode="determinate")
        self.pbar.pack(fill="x")
        self.lbl = ttk.Label(outer, text="ready")
        self.lbl.pack(anchor="w", pady=4)

        # ── head browser from output HDF5 (carousel: 5 heads, big center) ──
        self._section(outer, "Browse heads (from HDF5)")
        prow = ttk.Frame(outer); prow.pack(fill="x")
        self.v_view_h5 = tk.StringVar(value="data/transferred.h5")
        ttk.Entry(prow, textvariable=self.v_view_h5).pack(
            side="left", fill="x", expand=True, padx=(0, 4))

        def browse_view():
            p = filedialog.askopenfilename(
                title="HDF5 to preview",
                filetypes=[("HDF5", "*.h5"), ("all", "*.*")])
            if p:
                self.v_view_h5.set(p)
        ttk.Button(prow, text="...", width=3, command=browse_view).pack(
            side="left")
        ttk.Label(prow, text="expr:").pack(side="left", padx=(6, 2))
        self.v_view_expr = ttk.Combobox(prow, width=14, state="readonly")
        self.v_view_expr.pack(side="left")
        self.v_view_expr.bind("<<ComboboxSelected>>",
                              lambda e: self._load_carousel())
        ttk.Button(prow, text="Load",
                   command=self._load_carousel).pack(side="left", padx=4)

        # вторая строка: усиление δ для наглядности + раскраска величины
        prow2 = ttk.Frame(outer); prow2.pack(fill="x")
        lab_g = ttk.Label(prow2, text="preview δ gain:")
        lab_g.pack(side="left")
        _ToolTip(lab_g, "Visual exaggeration of δ in previews only (does NOT "
                        "change saved data). Range -6..+6: negative = inverted "
                        "expression. Smooth slider; rerenders on release.")
        self.v_pgain = tk.DoubleVar(value=4.0)
        sc_g = tk.Scale(prow2, from_=-6.0, to=6.0, orient="horizontal",
                        variable=self.v_pgain, resolution=0.1, length=220)
        sc_g.pack(side="left")
        # перерисовываем при ОТПУСКАНИИ (плавный ход без рендера на каждый шаг)
        sc_g.bind("<ButtonRelease-1>", lambda e: self._refresh_carousel())
        self.v_pcolor = tk.BooleanVar(value=True)
        cb = ttk.Checkbutton(prow2, text="colorize |δ|", variable=self.v_pcolor,
                             command=self._refresh_carousel)
        cb.pack(side="left", padx=6)
        _ToolTip(cb, "Overlay |δ| magnitude as a 2-color gradient so the "
                     "expression is visible even with small displacements.")
        # два цвета градиента: low (нет изменений) → high (макс изменение)
        self._col_lo = [0.1, 0.2, 1.0]             # синий
        self._col_hi = [1.0, 0.1, 0.1]             # красный
        self.btn_lo = tk.Button(prow2, text="low", width=5,
                                bg=self._rgb_hex(self._col_lo),
                                command=lambda: self._pick_color("lo"))
        self.btn_lo.pack(side="left", padx=(8, 2))
        self.btn_hi = tk.Button(prow2, text="high", width=5,
                                bg=self._rgb_hex(self._col_hi),
                                command=lambda: self._pick_color("hi"))
        self.btn_hi.pack(side="left", padx=2)
        _ToolTip(self.btn_lo, "Color for NO change (|δ|≈0).")
        _ToolTip(self.btn_hi, "Color for MAX change zones.")

        # WKS — спектральный дескриптор формы на меше (раскраска голов)
        self.v_wks = tk.BooleanVar(value=False)
        cb_wks = ttk.Checkbutton(prow2, text="WKS", variable=self.v_wks,
                                 command=self._refresh_carousel)
        cb_wks.pack(side="left", padx=(10, 2))
        _ToolTip(cb_wks, "Color heads by Wave Kernel Signature (spectral, "
                         "pose-invariant shape descriptor). Overrides colorize.")

        # отрисовка всех MediaPipe-лендмарков поверх превью
        self.v_lmk = tk.BooleanVar(value=False)
        cb_lmk = ttk.Checkbutton(prow2, text="landmarks", variable=self.v_lmk,
                                 command=self._refresh_carousel)
        cb_lmk.pack(side="left", padx=(10, 2))
        _ToolTip(cb_lmk, "Overlay all MediaPipe Face Mesh landmarks (green dots) "
                         "on each preview. Useful to check face detection.")

        # сигнатурные лендмарки: центроиды групп вершин с WKS в [min,max]
        self.v_sig = tk.BooleanVar(value=False)
        cb_sig = ttk.Checkbutton(prow2, text="signature landmarks",
                                 variable=self.v_sig,
                                 command=self._refresh_carousel)
        cb_sig.pack(side="left", padx=(10, 2))
        _ToolTip(cb_sig, "Put a dot at the centroid of each cluster of vertices "
                         "whose WKS is in [sig min, sig max]. Clusters = vertices "
                         "linked by distance (sig dist). Set the range/dist below.")

        # сферы WKS-центроидов в отдельном 3D-вьювере (двойной клик по голове)
        self.v_sig_spheres = tk.BooleanVar(value=False)
        cb_sigsp = ttk.Checkbutton(prow2, text="WKS spheres in 3D",
                                   variable=self.v_sig_spheres)
        cb_sigsp.pack(side="left", padx=(10, 2))
        _ToolTip(cb_sigsp, "When you double-click a head to open it in the "
                           "separate 3D viewer, spawn a small yellow sphere at "
                           "each WKS group centroid (uses that head's sig "
                           "params). For inspecting landmark placement in 3D.")

        # параметры WKS (спектр + специфичные)
        prow3 = ttk.Frame(outer); prow3.pack(fill="x")

        def desc_num(label, var, tip, w=6, on_enter=None):
            l = ttk.Label(prow3, text=label); l.pack(side="left", padx=(8, 2))
            e = ttk.Entry(prow3, textvariable=var, width=w); e.pack(side="left")
            cb = on_enter or self._refresh_carousel
            e.bind("<Return>", lambda ev: cb())
            _ToolTip(l, tip); _ToolTip(e, tip)

        self.v_desc_eigs = tk.StringVar(value="80")
        self.v_desc_ch = tk.StringVar(value="60")
        self.v_desc_chan = tk.StringVar(value="")     # пусто = средний
        self.v_desc_sigma = tk.StringVar(value="7.0")
        desc_num("eigs:", self.v_desc_eigs,
                 "Number of Laplacian eigenfunctions used to build WKS "
                 "(spectral resolution). More = finer detail, slower.")
        desc_num("energy bins:", self.v_desc_ch,
                 "Number of WKS energy channels (the descriptor length).")
        desc_num("channel:", self.v_desc_chan,
                 "Which WKS energy channel to show (empty = middle; "
                 "low energy = coarse shape, high = fine detail).")
        desc_num("WKS σ:", self.v_desc_sigma,
                 "WKS gaussian window width over energy. Smaller = sharper "
                 "frequency selectivity (noisier), larger = smoother.")
        # диапазон сигнатуры [min,max] + радиус группировки
        self.v_sig_min = tk.StringVar(value="0.33")
        self.v_sig_max = tk.StringVar(value="0.50")
        self.v_sig_dist = tk.StringVar(value="0.05")
        self.v_sig_smooth = tk.StringVar(value="0")
        sig_apply = self._apply_sig_to_center        # Enter → центральной голове
        desc_num("sig min:", self.v_sig_min,
                 "Lower bound of normalized WKS [0..1] for signature landmarks "
                 "(in the hot colormap 'red' is ~0.33). Enter applies to the "
                 "CENTER head only.", on_enter=sig_apply)
        desc_num("sig max:", self.v_sig_max,
                 "Upper bound of normalized WKS [0..1] ('yellow' ~0.66, "
                 "'white' ~1.0). Keep below ~0.6 to target the red band. "
                 "Enter applies to the CENTER head only.", on_enter=sig_apply)
        desc_num("sig dist:", self.v_sig_dist,
                 "Cluster radius: vertices within this fraction of head size "
                 "(bbox diagonal) join one group. Larger = fewer groups. "
                 "Enter applies to the CENTER head only.", on_enter=sig_apply)
        desc_num("sig smooth:", self.v_sig_smooth,
                 "Signature smoothing (1-ring averaging) before selection. "
                 "Higher = kills noisy/garbage signatures. 0 = raw. Enter "
                 "applies to the CENTER head only.", on_enter=sig_apply)
        # авто-подбор расстояния под целевое число групп на каждой голове
        self.v_sig_target = tk.StringVar(value="8")
        desc_num("target:", self.v_sig_target,
                 "Target number of groups. 'solve dist' tunes sig dist per head "
                 "(binary search 0..1) so each head yields this many groups.",
                 w=4)
        btn_solve = ttk.Button(prow3, text="🎯 solve dist",
                               command=self._solve_sig_groups)
        btn_solve.pack(side="left", padx=4)
        _ToolTip(btn_solve, "For every head, search sig dist in [0,1] so the "
                            "number of groups equals 'target'. Per-head distance "
                            "is stored and used for the signature landmarks.")
        ttk.Label(prow3, text="(Enter to apply)", font=("", 8)).pack(
            side="left", padx=6)

        # карусель: 5 ячеек (2 слева, центр крупнее, 2 справа)
        self.carousel = ttk.Frame(outer)
        self.carousel.pack(fill="x", pady=6)
        self.cells = []                          # [(col, label_img, name, sz)]
        self._cell_head = [None] * 5             # имя головы в каждой ячейке
        small, big = 130, 200
        for i in range(5):
            col = ttk.Frame(self.carousel)
            col.pack(side="left", expand=True, padx=4)
            sz = big if i == 2 else small
            limg = tk.Label(col, background="#222")
            limg.pack()
            # двойной клик → открыть голову ячейки в 3D
            limg.bind("<Double-Button-1>",
                      lambda e, ci=i: self._open_head_3d(ci))
            lname = ttk.Label(col, text="", font=("", 8))
            lname.pack()
            self.cells.append((col, limg, lname, sz))

        # стрелки листания голов влево/вправо
        nav = ttk.Frame(outer); nav.pack(fill="x")
        ttk.Button(nav, text="◀ prev", width=8,
                   command=lambda: self._step_head(-1)).pack(side="left")
        ttk.Button(nav, text="next ▶", width=8,
                   command=lambda: self._step_head(1)).pack(side="right")

        # слайдер по головам
        self.head_slider = tk.Scale(outer, from_=0, to=0, orient="horizontal",
                                    command=self._on_head_slide,
                                    showvalue=True)
        self.head_slider.pack(fill="x")

        # кнопка пометки текущей (центральной) головы как плохой перенос
        mrow = ttk.Frame(outer); mrow.pack(fill="x", pady=2)
        self.btn_flag = ttk.Button(mrow, text="⚑ Mark center head: BAD transfer",
                                   command=self._toggle_bad)
        self.btn_flag.pack(side="left")
        self.lbl_flag = ttk.Label(mrow, text="")
        self.lbl_flag.pack(side="left", padx=8)

        self._preview_imgs = []                  # keep refs (else GC)
        self._head_names = []                    # имена голов текущего HDF5
        self._cache = {}                         # hkey → PIL.Image
        self._sig_count = {}                     # hkey → n групп
        self._sig_params = {}                    # head → {min,max,dist,smooth}

    # ── presets (JSON в motion_groups_v7/presets/) ──
    def _presets_dir(self):
        d = Path(__file__).resolve().parent / "presets"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _all_vars(self):
        """Все настраиваемые поля: имя → tk-переменная (для save/load)."""
        return {
            'ref': self.v_ref, 'dir': self.v_dir, 'out': self.v_out,
            'time': self.v_time, 'steps': self.v_steps,
            'n_clusters': self.v_nclu, 'heat_threshold': self.v_hthr,
            'position_weight': self.v_pw, 'clustering_method': self.v_cmeth,
            'cluster_similarity_threshold': self.v_csim,
            'smooth_iters': self.v_smi, 'smooth_alpha': self.v_sma,
            'smooth_transferred_groups': self.v_sgrp,
            'group_smooth_iters': self.v_gsi,
            'landmarks': self.v_lm,
            'multi_t_n_times': self.v_mtt, 'multi_t_n_eigs': self.v_mte,
            'multi_t_mask_by_single_t': self.v_mtm,
            'uv_flat': self.v_uflat, 'uv_world_orient': self.v_uworld,
            'uv_align_pca_icp': self.v_uicp, 'uv_interp_delta': self.v_uinterp,
            'uv_relax_method': self.v_urel, 'uv_relax_iters': self.v_ureli,
            'uv_warp_heat': self.v_warp, 'uv_warp_heat_t': self.v_warpt,
            'uv_warp_min_dist': self.v_warpmd,
            'head_from': self.v_hfrom, 'head_to': self.v_hto,
        }

    def _refresh_preset_list(self):
        names = sorted(p.stem for p in self._presets_dir().glob("*.json"))
        self.preset_box["values"] = names

    def _save_preset(self):
        from tkinter import simpledialog
        import json
        name = simpledialog.askstring("Save preset", "Preset name:",
                                      parent=self.root)
        if not name:
            return
        safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name)
        data = {k: v.get() for k, v in self._all_vars().items()}
        with open(self._presets_dir() / f"{safe}.json", "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self._refresh_preset_list()
        self.preset_box.set(safe)
        self.lbl.config(text=f"preset saved: {safe}")

    def _load_preset(self):
        import json
        name = self.preset_box.get()
        if not name:
            return
        path = self._presets_dir() / f"{name}.json"
        if not path.exists():
            return
        with open(path) as f:
            data = json.load(f)
        for k, var in self._all_vars().items():
            if k in data:
                try:
                    var.set(data[k])
                except Exception:
                    pass
        self.lbl.config(text=f"preset loaded: {name}")

    def _delete_preset(self):
        name = self.preset_box.get()
        if not name:
            return
        (self._presets_dir() / f"{name}.json").unlink(missing_ok=True)
        self.preset_box.set("")
        self._refresh_preset_list()
        self.lbl.config(text=f"preset deleted: {name}")

    def _collect(self):
        def fi(v): return int(float(v.get()))
        def ff(v): return float(v.get())
        lm = tuple(int(s) for s in self.v_lm.get().replace(";", ",").split(",")
                   if s.strip())
        return dict(
            time=ff(self.v_time), steps=fi(self.v_steps),
            n_clusters=fi(self.v_nclu), heat_threshold=ff(self.v_hthr),
            position_weight=ff(self.v_pw),
            clustering_method=self.v_cmeth.get(),
            cluster_similarity_threshold=ff(self.v_csim),
            smooth_iters=fi(self.v_smi), smooth_alpha=ff(self.v_sma),
            smooth_transferred_groups=self.v_sgrp.get(),
            group_smooth_iters=fi(self.v_gsi),
            multi_t_n_times=fi(self.v_mtt), multi_t_n_eigs=fi(self.v_mte),
            multi_t_mask_by_single_t=self.v_mtm.get(),
            uv_flat=self.v_uflat.get(), uv_world_orient=self.v_uworld.get(),
            uv_align_pca_icp=self.v_uicp.get(),
            uv_interp_delta=self.v_uinterp.get(),
            uv_relax_method=self.v_urel.get(),
            uv_relax_iters=fi(self.v_ureli),
            uv_relax_adaptive=self.v_urelad.get(),
            uv_relax_target=ff(self.v_ureltg),
            uv_warp_heat=self.v_warp.get(),
            uv_warp_heat_t=ff(self.v_warpt),
            uv_warp_min_dist=ff(self.v_warpmd),
            # WKS landmark match (использует параметры панели превью)
            uv_wks_match=self.v_uwksmatch.get(),
            wks_sig_min=ff(self.v_sig_min), wks_sig_max=ff(self.v_sig_max),
            wks_sig_dist=ff(self.v_sig_dist),
            wks_sig_smooth=fi(self.v_sig_smooth),
            wks_desc_eigs=fi(self.v_desc_eigs),
            wks_desc_channels=fi(self.v_desc_ch),
            wks_desc_channel=(int(float(self.v_desc_chan.get()))
                              if self.v_desc_chan.get().strip() else None),
            wks_desc_sigma=ff(self.v_desc_sigma),
            landmarks=lm,
        )

    def _run(self):
        ref = self.v_ref.get().strip()
        hd = self.v_dir.get().strip()
        out = self.v_out.get().strip()
        if not (Path(ref).exists() and Path(hd).is_dir() and out):
            self.lbl.config(text="error: check reference / folder / output")
            return
        try:
            params = self._collect()
        except ValueError:
            self.lbl.config(text="error: a numeric field is invalid")
            return
        # диапазон голов (from..to), пусто = None
        def _opt_int(v):
            s = v.get().strip()
            return int(s) if s else None
        try:
            h_from = _opt_int(self.v_hfrom); h_to = _opt_int(self.v_hto)
            nw = int(self.v_workers.get() or 1)
        except ValueError:
            self.lbl.config(text="error: heads range / workers must be integers")
            return

        self.btn.config(state="disabled")
        self.pbar["value"] = 0

        def prog(stage, cur, total, msg):
            def upd():
                if stage in ("head", "head_done", "head_err"):
                    self.pbar["maximum"] = max(total, 1)
                    self.pbar["value"] = cur
                self.lbl.config(text=msg)
            self.root.after(0, upd)

        def worker():
            try:
                n_ok, n_tot, n_expr = eng.run_transfer(
                    ref, hd, out, params=params, progress=prog,
                    head_from=h_from, head_to=h_to, n_workers=nw)
                self.root.after(0, lambda: self.lbl.config(
                    text=f"✓ done: {n_ok}/{n_tot} heads × {n_expr} expr "
                         f"→ {out}"))
                # авто-загрузка карусели по готовому датасету
                self.root.after(0, lambda: (self.v_view_h5.set(out),
                                            self._load_carousel()))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.lbl.config(
                    text=f"error: {e}"))
            finally:
                self.root.after(0, lambda: self.btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()

    def _open_train(self):
        """Запускаем GUI обучения skinning-сети (src/training/train_gui.py)
        отдельным процессом с cwd = python/ (нужно для -m src...)."""
        import subprocess
        import sys
        here = Path(__file__).resolve().parent
        python_dir = here.parents[1]               # .../python
        cmd = [sys.executable, "-m", "src.training.train_gui"]

        def worker():
            try:
                subprocess.run(cmd, cwd=str(python_dir))
            except Exception as e:
                print(f"train gui error: {e}")
        threading.Thread(target=worker, daemon=True).start()
        self.lbl.config(text="opened training GUI")

    def _open_inspect(self):
        """Окно-инспектор HDF5: дерево групп/датасетов (форма, тип) + атрибуты.
        По умолчанию открывает текущий output HDF5."""
        import h5py
        win = tk.Toplevel(self.root)
        win.title("HDF5 inspector")
        win.geometry("680x600")
        top = ttk.Frame(win, padding=8); top.pack(fill="both", expand=True)

        # путь
        pr = ttk.Frame(top); pr.pack(fill="x")
        path_var = tk.StringVar(value=self.v_out.get().strip()
                                or "data/transferred.h5")
        ttk.Entry(pr, textvariable=path_var).pack(
            side="left", fill="x", expand=True, padx=(0, 4))

        def browse():
            p = filedialog.askopenfilename(
                title="HDF5", filetypes=[("HDF5", "*.h5"), ("all", "*.*")])
            if p:
                path_var.set(p)
        ttk.Button(pr, text="...", width=3, command=browse).pack(side="left")

        # дерево (Name | Shape | Dtype)
        tree = ttk.Treeview(top, columns=("shape", "dtype"), show="tree headings")
        tree.heading("#0", text="Name")
        tree.heading("shape", text="Shape"); tree.heading("dtype", text="Dtype")
        tree.column("shape", width=160); tree.column("dtype", width=110)
        tree.pack(fill="both", expand=True, pady=6)
        vsb = ttk.Scrollbar(top, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.place(relx=1.0, rely=0.15, relheight=0.7, anchor="ne")

        info = tk.Text(top, height=7, wrap="word")
        info.pack(fill="x")

        def _attrs_str(obj):
            if not obj.attrs:
                return ""
            return "  attrs: " + ", ".join(
                f"{k}={obj.attrs[k]}" for k in obj.attrs)

        def fill():
            tree.delete(*tree.get_children())
            info.delete("1.0", "end")
            p = path_var.get().strip()
            if not Path(p).exists():
                info.insert("end", f"not found: {p}")
                return
            try:
                with h5py.File(p, "r") as h:
                    # корневые атрибуты
                    if h.attrs:
                        info.insert("end", "ROOT attrs:\n")
                        for k in h.attrs:
                            info.insert("end", f"  {k} = {h.attrs[k]}\n")

                    def add(node, parent_id):
                        for key in node:
                            obj = node[key]
                            if isinstance(obj, h5py.Group):
                                nid = tree.insert(parent_id, "end", text=key + "/",
                                                  values=("group",
                                                          _attrs_str(obj)))
                                add(obj, nid)
                            else:                       # dataset
                                tree.insert(parent_id, "end", text=key,
                                            values=(str(obj.shape),
                                                    str(obj.dtype)))
                    add(h, "")
            except Exception as e:
                info.insert("end", f"error: {e}")

        def on_select(_e):
            sel = tree.selection()
            if not sel:
                return
            # путь до выбранного узла
            parts = []
            iid = sel[0]
            while iid:
                parts.append(tree.item(iid, "text").rstrip("/"))
                iid = tree.parent(iid)
            hpath = "/".join(reversed(parts))
            p = path_var.get().strip()
            try:
                with h5py.File(p, "r") as h:
                    obj = h[hpath]
                    info.delete("1.0", "end")
                    info.insert("end", f"/{hpath}\n")
                    if obj.attrs:
                        for k in obj.attrs:
                            info.insert("end", f"  attr {k} = {obj.attrs[k]}\n")
                    if isinstance(obj, h5py.Dataset):
                        info.insert("end",
                                    f"  shape={obj.shape} dtype={obj.dtype}\n")
            except Exception as e:
                info.insert("end", f"\nerror: {e}")

        tree.bind("<<TreeviewSelect>>", on_select)
        ttk.Button(pr, text="Load", command=fill).pack(side="left", padx=2)
        fill()

    def _open_make_ref(self):
        """Окно-генератор reference-HDF5: список эмоций (имя + FLAME betas
        '308:10,305:8') → build_reference по указанному пути."""
        import make_reference

        import json
        win = tk.Toplevel(self.root)
        win.title("Make reference HDF5")
        win.geometry("540x520")
        fr = ttk.Frame(win, padding=10); fr.pack(fill="both", expand=True)

        preset_dir = Path(__file__).resolve().parent / "presets_ref"
        preset_dir.mkdir(parents=True, exist_ok=True)

        # ── presets ──
        prow = ttk.Frame(fr); prow.pack(fill="x")
        ttk.Label(prow, text="preset:").pack(side="left")
        preset_box = ttk.Combobox(prow, width=22, state="readonly")
        preset_box.pack(side="left", padx=4)

        def refresh_presets():
            preset_box["values"] = sorted(
                p.stem for p in preset_dir.glob("*.json"))

        ttk.Label(fr, text="Expressions (name + FLAME betas, e.g. 308:10,305:8):"
                  ).pack(anchor="w", pady=(8, 0))
        ttk.Label(fr, text="формат строки:  имя | бета-строка",
                  font=("", 8)).pack(anchor="w")

        rows_box = ttk.Frame(fr); rows_box.pack(fill="both", expand=True, pady=4)
        rows = []                                  # [(name_var, betas_var)]

        def add_row(name="", betas=""):
            r = ttk.Frame(rows_box); r.pack(fill="x", pady=1)
            nv = tk.StringVar(value=name); bv = tk.StringVar(value=betas)
            ttk.Entry(r, textvariable=nv, width=14).pack(side="left", padx=2)
            ttk.Entry(r, textvariable=bv).pack(
                side="left", fill="x", expand=True, padx=2)
            ttk.Button(r, text="✕", width=2,
                       command=lambda: (r.destroy(),
                                        rows.remove((nv, bv)))).pack(side="left")
            rows.append((nv, bv))

        def clear_rows():
            for r in rows_box.winfo_children():
                r.destroy()
            rows.clear()

        def load_preset():
            name = preset_box.get()
            if not name:
                return
            p = preset_dir / f"{name}.json"
            if not p.exists():
                return
            data = json.load(open(p))
            clear_rows()
            for item in data.get("expressions", []):
                add_row(item.get("name", ""), item.get("betas", ""))
            if "muscles" in data:
                nm_var.set(str(data["muscles"]))
            status.config(text=f"preset loaded: {name}")

        def save_preset():
            from tkinter import simpledialog
            nm = simpledialog.askstring("Save preset", "Preset name:",
                                        parent=win)
            if not nm:
                return
            safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in nm)
            data = {"expressions": [{"name": a.get().strip(),
                                     "betas": b.get().strip()}
                                    for a, b in rows
                                    if a.get().strip() and b.get().strip()],
                    "muscles": int(nm_var.get() or 8)}
            json.dump(data, open(preset_dir / f"{safe}.json", "w"),
                      indent=2, ensure_ascii=False)
            refresh_presets(); preset_box.set(safe)
            status.config(text=f"preset saved: {safe}")

        def delete_preset():
            name = preset_box.get()
            if name:
                (preset_dir / f"{name}.json").unlink(missing_ok=True)
                preset_box.set(""); refresh_presets()
                status.config(text=f"preset deleted: {name}")

        preset_box.bind("<<ComboboxSelected>>", lambda e: load_preset())
        ttk.Button(prow, text="Save as…", command=save_preset).pack(
            side="left", padx=2)
        ttk.Button(prow, text="Delete", command=delete_preset).pack(
            side="left", padx=2)

        add_row("smile", "308:10")
        add_row("brows", "310:7,311:-4")
        ttk.Button(fr, text="+ add expression",
                   command=lambda: add_row()).pack(anchor="w", pady=2)

        # путь сохранения
        pr = ttk.Frame(fr); pr.pack(fill="x", pady=(8, 2))
        ttk.Label(pr, text="Output:").pack(side="left")
        out_var = tk.StringVar(value="data/reference.h5")
        ttk.Entry(pr, textvariable=out_var).pack(
            side="left", fill="x", expand=True, padx=4)

        def browse_out():
            p = filedialog.asksaveasfilename(
                title="Reference HDF5", defaultextension=".h5",
                filetypes=[("HDF5", "*.h5")])
            if p:
                out_var.set(p)
        ttk.Button(pr, text="...", width=3, command=browse_out).pack(side="left")

        # размер вектора активаций (по нулям)
        ar = ttk.Frame(fr); ar.pack(fill="x", pady=2)
        ttk.Label(ar, text="muscles (zero activations):").pack(side="left")
        nm_var = tk.StringVar(value="8")
        ttk.Entry(ar, textvariable=nm_var, width=6).pack(side="left", padx=4)

        status = ttk.Label(fr, text=""); status.pack(anchor="w", pady=4)

        def generate():
            specs = []
            for nv, bv in rows:
                nm = nv.get().strip(); bt = bv.get().strip()
                if nm and bt:
                    specs.append(f"{nm}={bt}")
            if not specs:
                status.config(text="add at least one expression"); return
            out = out_var.get().strip()
            try:
                n_m = int(nm_var.get())
                muscles = [f"muscle_{i:02d}" for i in range(n_m)]
                acts = {s.split("=")[0]: [0.0] * n_m for s in specs}
                make_reference.build_reference(
                    out, shape_str="", expr_specs=specs,
                    activations=acts, muscle_names=muscles)
                status.config(text=f"✓ saved {len(specs)} expr → {out}")
                self.v_ref.set(out)               # сразу подставить в основное
            except Exception as e:
                import traceback; traceback.print_exc()
                status.config(text=f"error: {e}")

        ttk.Button(fr, text="▶ Generate reference",
                   command=generate).pack(fill="x", pady=6)

        refresh_presets()                          # заполнить список пресетов

    def _open_debug(self):
        """Запускаем пошаговый debug-вьювер с ТЕКУЩИМИ параметрами окна.
        Параметры пишем во временный JSON, передаём подпроцессу."""
        import subprocess
        import sys
        import json
        import tempfile
        ref = self.v_ref.get().strip()
        hd = self.v_dir.get().strip()
        if not (Path(ref).exists() and hd and Path(hd).is_dir()):
            self.lbl.config(text="debug: set reference HDF5 and heads folder")
            return
        try:
            params = self._collect()
        except ValueError:
            self.lbl.config(text="error: a numeric field is invalid")
            return
        # landmarks → строка (JSON не любит tuple, debug разберёт обратно)
        params = dict(params)
        params['landmarks'] = ",".join(str(x) for x in params['landmarks'])
        pj = tempfile.NamedTemporaryFile(suffix=".json", delete=False,
                                         mode="w")
        json.dump(params, pj); pj.close()
        here = Path(__file__).resolve().parent
        cmd = [sys.executable, str(here / "pipeline_debug.py"),
               "--ref", ref, "--dir", hd, "--params", pj.name]

        def worker():
            try:
                subprocess.run(cmd, cwd=str(here.parents[2]))
            except Exception as e:
                print(f"pipeline_debug error: {e}")
        threading.Thread(target=worker, daemon=True).start()
        self.lbl.config(text="opened pipeline step debug")

    # ── head browser (carousel) ──
    def _current_expr(self):
        v = self.v_view_expr.get()
        return None if v in ("", "neutral") else v

    def _load_carousel(self):
        """Считываем имена голов + список выражений из HDF5, настраиваем слайдер
        и рисуем карусель вокруг текущей позиции."""
        from pathlib import Path
        h5 = self.v_view_h5.get().strip()
        if not Path(h5).exists():
            self.lbl.config(text=f"HDF5 not found: {h5}")
            return
        try:
            heads = eng.list_output_heads(h5)
        except Exception as e:
            self.lbl.config(text=f"cannot read HDF5: {e}")
            return
        if not heads:
            self.lbl.config(text="no heads in HDF5")
            return
        import h5py
        with h5py.File(h5, "r") as f:
            g0 = f["heads"][heads[0]]
            exprs = list(g0["expr"].keys()) if "expr" in g0 else []
        if not self.v_view_expr["values"]:
            self.v_view_expr["values"] = ["neutral"] + exprs
        if self.v_view_expr.get() not in (["neutral"] + exprs):
            self.v_view_expr.set("neutral")
        self._head_names = heads
        self._cache = {}                          # сброс кэша при смене файла/выраж
        self._sig_count = {}
        self._sig_params = {}                     # пер-головные параметры — заново
        self.head_slider.config(from_=0, to=max(len(heads) - 1, 0))
        self.head_slider.set(0)
        self._load_sig_fields(heads[0])
        self._render_carousel(0)

    def _on_head_slide(self, val):
        if self._head_names:
            c = int(float(val))
            if 0 <= c < len(self._head_names):
                self._load_sig_fields(self._head_names[c])   # поля = центр. голова
            self._render_carousel(c)

    def _step_head(self, delta):
        """Листать головы стрелками влево/вправо (центр ±1)."""
        if not self._head_names:
            return
        c = max(0, min(len(self._head_names) - 1,
                       int(self.head_slider.get()) + delta))
        self.head_slider.set(c)                              # → _on_head_slide

    def _center_head(self):
        """Имя текущей центральной головы (позиция слайдера)."""
        if not self._head_names:
            return None
        c = int(self.head_slider.get())
        return self._head_names[c] if 0 <= c < len(self._head_names) else None

    # ── пер-головные параметры сигнатурных лендмарков ──
    def _sig_params_for(self, nm):
        """Параметры сигнатур для головы (или дефолты, если не заданы)."""
        p = self._sig_params.get(nm)
        return dict(p) if p else {'min': 0.33, 'max': 0.50,
                                  'dist': 0.05, 'smooth': 0}

    def _read_sig_fields(self):
        def _f(v, d):
            try:
                return float(v.get())
            except ValueError:
                return d

        def _i(v, d):
            try:
                return int(float(v.get()))
            except ValueError:
                return d
        return {'min': _f(self.v_sig_min, 0.33), 'max': _f(self.v_sig_max, 0.50),
                'dist': _f(self.v_sig_dist, 0.05),
                'smooth': _i(self.v_sig_smooth, 0)}

    def _load_sig_fields(self, nm):
        """Заполнить поля параметрами головы nm (для центральной)."""
        if not nm:
            return
        p = self._sig_params_for(nm)
        self.v_sig_min.set(f"{p['min']:.3g}")
        self.v_sig_max.set(f"{p['max']:.3g}")
        self.v_sig_dist.set(f"{p['dist']:.4g}")
        self.v_sig_smooth.set(str(int(p['smooth'])))

    def _apply_sig_to_center(self):
        """Enter в сигнатурных полях → назначить параметры ЦЕНТРАЛЬНОЙ голове."""
        nm = self._center_head()
        if not nm:
            return
        self._sig_params[nm] = self._read_sig_fields()
        self.v_sig.set(True)                                # показать точки
        self._render_carousel(int(self.head_slider.get()))

    def _update_flag_btn(self):
        """Обновляем подпись кнопки/метку по флагу центральной головы."""
        nm = self._center_head()
        h5 = self.v_view_h5.get().strip()
        if not nm or not Path(h5).exists():
            self.lbl_flag.config(text="")
            return
        try:
            bad = eng.get_head_flag(h5, nm)
        except Exception:
            bad = False
        self.lbl_flag.config(text=(f"⚑ {nm}: BAD" if bad else f"{nm}: ok"))

    def _toggle_bad(self):
        """Переключаем метку 'плохой перенос' для центральной головы (в HDF5)."""
        nm = self._center_head()
        h5 = self.v_view_h5.get().strip()
        if not nm or not Path(h5).exists():
            self.lbl_flag.config(text="no head / HDF5")
            return
        try:
            cur = eng.get_head_flag(h5, nm)
            eng.set_head_flag(h5, nm, not cur)
            self._update_flag_btn()
            # перерисовать карусель, чтобы обновить рамку/метку без рендера
            self._render_carousel(int(self.head_slider.get()))
        except Exception as e:
            self.lbl_flag.config(text=f"error: {e}")

    def _open_head_3d(self, cell_i):
        """Двойной клик по ячейке → открыть голову в Open3D (mesh_viewer v6).
        Экспортируем нейтраль+δ выбранного выражения во временный OBJ."""
        nm = self._cell_head[cell_i] if cell_i < len(self._cell_head) else None
        if not nm:
            return
        import subprocess
        import sys
        h5 = self.v_view_h5.get().strip()
        expr = self._current_expr()
        gain = float(self.v_pgain.get())
        # тот же дескриптор/раскраска, что в превью
        descriptor = "wks" if self.v_wks.get() else None
        colorize = bool(self.v_pcolor.get())

        def _i(v, d):
            try:
                return int(float(v.get()))
            except ValueError:
                return d
        d_eigs = _i(self.v_desc_eigs, 80); d_ch = _i(self.v_desc_ch, 60)
        d_chan = (int(float(self.v_desc_chan.get()))
                  if self.v_desc_chan.get().strip() else None)
        try:
            d_sig = float(self.v_desc_sigma.get())
        except ValueError:
            d_sig = 7.0
        clo = tuple(self._col_lo); chi = tuple(self._col_hi)
        here = Path(__file__).resolve().parent
        viewer = here.parent / "motion_groups_v6" / "mesh_viewer.py"
        sig_spheres = bool(self.v_sig_spheres.get())   # читаем в гл. потоке
        sp = self._sig_params_for(nm)                  # пер-головные параметры

        def worker():
            try:
                obj = eng.export_head_obj(
                    h5, nm, expr=expr, gain=gain,
                    descriptor=descriptor, colorize=colorize,
                    col_lo=clo, col_hi=chi,
                    desc_n_eigs=d_eigs, desc_channels=d_ch,
                    desc_channel=d_chan, desc_wks_sigma=d_sig)
                cmd = [sys.executable, str(viewer), obj]
                if sig_spheres:                        # сферы WKS-центроидов
                    centers = eng.signature_landmark_centroids_h5(
                        h5, nm, expr=expr, gain=gain,
                        sig_min=sp['min'], sig_max=sp['max'],
                        sig_dist=sp['dist'], sig_smooth=sp['smooth'],
                        n_eigs=d_eigs, n_channels=d_ch, channel=d_chan,
                        wks_sigma=d_sig)
                    if len(centers):
                        import numpy as _np
                        pf = obj + ".spheres.npy"
                        _np.save(pf, centers)
                        cmd.append(pf)
                subprocess.run(cmd)
            except Exception as e:
                print(f"open head 3d error: {e}")
        threading.Thread(target=worker, daemon=True).start()
        self.lbl.config(text=f"opening 3D: {nm} ({self.v_view_expr.get()})")

    @staticmethod
    def _rgb_hex(c):
        return "#%02x%02x%02x" % (int(c[0] * 255), int(c[1] * 255),
                                  int(c[2] * 255))

    def _pick_color(self, which):
        from tkinter import colorchooser
        cur = self._col_lo if which == "lo" else self._col_hi
        rgb, _hex = colorchooser.askcolor(
            color=self._rgb_hex(cur), title=f"{which} color")
        if rgb is None:
            return
        c = [v / 255.0 for v in rgb]
        if which == "lo":
            self._col_lo = c; self.btn_lo.config(bg=self._rgb_hex(c))
        else:
            self._col_hi = c; self.btn_hi.config(bg=self._rgb_hex(c))
        self._refresh_carousel()

    def _refresh_carousel(self):
        """Перерисовать карусель при смене глобальных опций (gain/colorize/WKS).
        Пер-головные параметры сигнатур (_sig_params) при этом сохраняются."""
        if self._head_names:
            self._cache = {}
            self._sig_count = {}
            self._render_carousel(int(self.head_slider.get()))

    def _solve_sig_groups(self):
        """Solve для ВСЕХ голов: берём полосу/сглаживание ЦЕНТРАЛЬНОЙ головы
        (из полей) и подбираем sig_dist под target на каждой голове. Результат
        пишем в пер-головные параметры _sig_params."""
        if not self._head_names:
            return
        h5 = self.v_view_h5.get().strip()
        try:
            target = int(float(self.v_sig_target.get()))
        except ValueError:
            self.lbl.config(text="target must be a number"); return

        def _f(v, d):
            try:
                return float(v.get())
            except ValueError:
                return d

        def _i(v, d):
            try:
                return int(float(v.get()))
            except ValueError:
                return d
        sig_min = _f(self.v_sig_min, 0.33)
        sig_max = _f(self.v_sig_max, 0.50)
        sig_smooth = _i(self.v_sig_smooth, 0)
        d_eigs = _i(self.v_desc_eigs, 80)
        d_ch = _i(self.v_desc_ch, 60)
        d_chan = (int(float(self.v_desc_chan.get()))
                  if self.v_desc_chan.get().strip() else None)
        d_sig = _f(self.v_desc_sigma, 7.0)
        nw = _i(self.v_workers, 1)
        names = list(self._head_names)

        def progress(done, total):                     # из фон. потока → в UI
            self.root.after(0, lambda d=done, t=total: self.lbl.config(
                text=f"solving sig dist… {d}/{t}  ({nw} proc)"))

        def worker():
            solved = eng.solve_sig_dist_batch(
                h5, names, target, sig_min=sig_min, sig_max=sig_max,
                desc_n_eigs=d_eigs, desc_channels=d_ch, desc_channel=d_chan,
                desc_wks_sigma=d_sig, sig_smooth=sig_smooth,
                n_workers=nw, progress=progress)
            # записать пер-головные параметры: полоса/сглаживание центра + свой dist
            params = dict(self._sig_params)
            for nm2, dist in solved.items():
                params[nm2] = {'min': sig_min, 'max': sig_max,
                               'dist': dist, 'smooth': sig_smooth}
            self._sig_params = params
            self._cache = {}
            self._sig_count = {}
            self.root.after(0, lambda: (
                self.v_sig.set(True),
                self._load_sig_fields(self._center_head()),
                self._render_carousel(int(self.head_slider.get())),
                self.lbl.config(text=f"solved sig dist for {len(solved)} heads "
                                     f"→ ~{target} groups each ({nw} proc)")))
        threading.Thread(target=worker, daemon=True).start()
        self.lbl.config(text=f"solving sig dist… ({nw} proc)")

    def _render_carousel(self, center):
        """Рисуем 5 голов: center в большой ячейке, по 2 слева/справа (если
        есть). Рендер кэшируется. Тяжёлый рендер — в фоне, показ — в главном."""
        h5 = self.v_view_h5.get().strip()
        expr = self._current_expr()
        gain = float(self.v_pgain.get())
        colorize = bool(self.v_pcolor.get())
        descriptor = "wks" if self.v_wks.get() else None
        clo = tuple(self._col_lo); chi = tuple(self._col_hi)
        # параметры дескриптора (с фолбэками при пустых/битых полях)
        def _i(v, d):
            try:
                return int(float(v.get()))
            except ValueError:
                return d
        d_eigs = _i(self.v_desc_eigs, 80)
        d_ch = _i(self.v_desc_ch, 60)
        d_chan = (int(float(self.v_desc_chan.get()))
                  if self.v_desc_chan.get().strip() else None)
        try:
            d_sig = float(self.v_desc_sigma.get())
        except ValueError:
            d_sig = 7.0
        show_lmk = bool(self.v_lmk.get())
        show_sig = bool(self.v_sig.get())
        # глобальная часть ключа; параметры сигнатур — ПЕР-ГОЛОВНЫЕ
        base = (gain, colorize, clo, chi, descriptor, d_eigs, d_ch, d_chan,
                d_sig, show_lmk, show_sig)

        def hkey(nm):
            if show_sig:
                p = self._sig_params_for(nm)
                st = (p['min'], p['max'], p['dist'], p['smooth'])
            else:
                st = ()
            return (nm, expr, base, st)

        names = self._head_names
        n = len(names)
        idxs = [center - 2 + k for k in range(5)]
        need = [(i, names[i]) for i in idxs if 0 <= i < n
                and hkey(names[i]) not in self._cache]
        self.lbl.config(text=f"head {center+1}/{n} ({self.v_view_expr.get()})")

        def worker():
            from PIL import Image
            for i, nm in need:
                try:
                    cnt = []
                    p = self._sig_params_for(nm)              # своя на голову
                    arr = eng.render_head_preview(
                        h5, nm, expr=expr, res=210,
                        gain=gain, colorize=colorize,
                        col_lo=clo, col_hi=chi, descriptor=descriptor,
                        desc_n_eigs=d_eigs, desc_channels=d_ch,
                        desc_channel=d_chan, desc_wks_sigma=d_sig,
                        show_landmarks=show_lmk, show_signature=show_sig,
                        sig_min=p['min'], sig_max=p['max'], sig_dist=p['dist'],
                        sig_smooth=p['smooth'], sig_counter=cnt)
                    self._cache[hkey(nm)] = Image.fromarray(arr)
                    self._sig_count[hkey(nm)] = cnt[0] if cnt else None
                except Exception:
                    self._cache[hkey(nm)] = None
            self.root.after(0, show)

        def show():
            from PIL import ImageTk
            self._preview_imgs = []
            self._cell_head = [None] * 5
            # сначала прячем все ячейки, затем упаковываем по порядку только
            # заполненные → пустые не занимают места, порядок слева-направо цел.
            for (col, _l, _n, _s) in self.cells:
                col.pack_forget()
            h5 = self.v_view_h5.get().strip()
            for cell_i, (col, limg, lname, sz) in enumerate(self.cells):
                hi = idxs[cell_i]
                im = (self._cache.get(hkey(names[hi]))
                      if 0 <= hi < n else None)
                if im is None:
                    continue
                nm = names[hi]
                self._cell_head[cell_i] = nm       # запомнить голову ячейки
                try:
                    bad = eng.get_head_flag(h5, nm)
                except Exception:
                    bad = False
                thumb = im.resize((sz, sz))
                tkimg = ImageTk.PhotoImage(thumb)
                self._preview_imgs.append(tkimg)
                # подсветка плохого переноса — красная рамка
                limg.config(image=tkimg, borderwidth=(3 if bad else 0),
                            relief=("solid" if bad else "flat"),
                            highlightthickness=(3 if bad else 0),
                            highlightbackground="#e53935",
                            highlightcolor="#e53935")
                mark = "⚑ " if bad else ""
                label = mark + ("▸ " + nm if cell_i == 2 else nm)
                if show_sig:                           # число групп под картинкой
                    ng = self._sig_count.get(hkey(nm))
                    if ng is not None:
                        label += f"\n{ng} groups"
                lname.config(
                    text=label, foreground=("#e53935" if bad else ""))
                col.pack(side="left", expand=True, padx=4)
            self._update_flag_btn()                # метка центральной головы

        if need:
            threading.Thread(target=worker, daemon=True).start()
        else:
            show()


def main():
    root = tk.Tk()
    root.geometry("960x980")
    TransferGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
