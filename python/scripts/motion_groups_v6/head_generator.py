#!/usr/bin/env python3
"""
Head Generator — генерация N случайных FLAME-голов для датасета (GUI, tkinter).

Берёт FLAME-модель, сэмплит случайные shape-беты (индексы 0..299) и сохраняет
каждую голову в .fbx (через assimp; фолбэк .obj). Сила различий регулируется:
  • «сила вариации» (std) — амплитуда случайных shape-бет (больше = головы
    сильнее отличаются);
  • «число shape-компонент» — сколько первых компонент варьируем (первые
    компоненты дают крупные изменения формы черепа/лица);
  • seed — воспроизводимость.

Запуск:
  cd /Users/kuzix/Documents/GitHub/Muscle-autoskinner
  source .venv/bin/activate
  python python/scripts/motion_groups_v6/head_generator.py
"""
import os
import subprocess
import tempfile
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

import numpy as np

import debug_head1_pipeline as pipe


SHAPE_START = 0
EXPR_START = 300


def _export_fbx(verts, faces, out_path):
    """verts+faces → FBX через assimp (obj-посредник). Фолбэк: .obj.
    Возвращает фактический путь."""
    tmp = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    tmp.close()
    pipe._write_obj(tmp.name, verts, faces)
    try:
        r = subprocess.run(["assimp", "export", tmp.name, out_path],
                           capture_output=True, text=True, timeout=120)
        ok = (r.returncode == 0 and os.path.exists(out_path))
    except Exception:
        ok = False
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    if ok:
        return out_path
    obj_path = os.path.splitext(out_path)[0] + ".obj"
    pipe._write_obj(obj_path, verts, faces)
    return obj_path


def generate_heads(v_t, sd, faces, out_dir, n_heads=10, std=1.5,
                   std_max=None, n_components=30, seed=0, normalize=True,
                   progress=None):
    """Генерируем n_heads голов случайными shape-бетами по первым
    n_components компонентам (0..n_components-1, в пределах shape 0..299).

    std_max is None       → у всех голов один разброс N(0, std).
    std_max задан и >std   → у КАЖДОЙ головы свой разброс s ~ uniform(std,
        std_max), затем betas ~ N(0, s). Даёт непрерывный спектр «силы
        различий»: и почти-нейтральные, и сильно отличающиеся головы.

    Сохраняем каждую в FBX. Возвращает список путей."""
    rng = np.random.default_rng(int(seed))
    n_comp = int(np.clip(n_components, 1, EXPR_START))
    use_range = (std_max is not None and float(std_max) > float(std))
    lo, hi = float(std), float(std_max) if use_range else (float(std), None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    manifest = []                                # (name, betas, std_use)
    for i in range(int(n_heads)):
        s = rng.uniform(lo, hi) if use_range else lo   # std для этой головы
        betas = rng.normal(0.0, s, size=n_comp)
        bdict = {int(c): float(betas[c]) for c in range(n_comp)
                 if abs(betas[c]) > 1e-6}
        verts = pipe.apply_betas(v_t, sd, bdict)
        if normalize:
            verts = pipe.normalize_bbox(verts)
        name = f"head_{i:04d}"
        path = _export_fbx(verts, faces, str(out_dir / f"{name}.fbx"))
        paths.append(path)
        manifest.append((name, bdict, s))
        if progress:
            progress(i + 1, int(n_heads), os.path.basename(path))
    # манифест: betas + использованный std каждой головы (воспроизводимость)
    with open(out_dir / "manifest.csv", "w") as f:
        f.write("head,std,betas_json\n")
        import json
        for name, bd, s in manifest:
            f.write(f"{name},{s:.4f},\"{json.dumps(bd)}\"\n")
    return paths


class HeadGeneratorGUI:
    def __init__(self, root):
        self.root = root
        root.title("FLAME Head Generator")
        self.flame = None
        self._build()

    def _build(self):
        f = ttk.Frame(self.root, padding=12)
        f.grid(sticky="nsew")
        r = 0

        ttk.Label(f, text="FLAME .pkl:").grid(row=r, column=0, sticky="w")
        self.v_flame = tk.StringVar(value=pipe.FLAME_PKL)
        ttk.Entry(f, textvariable=self.v_flame, width=44).grid(
            row=r, column=1, sticky="ew", padx=4)
        ttk.Button(f, text="...", width=3,
                   command=self._browse_flame).grid(row=r, column=2)
        r += 1

        ttk.Label(f, text="Папка вывода:").grid(row=r, column=0, sticky="w")
        self.v_out = tk.StringVar(
            value="python/scripts/debug_output/generated_heads")
        ttk.Entry(f, textvariable=self.v_out, width=44).grid(
            row=r, column=1, sticky="ew", padx=4)
        ttk.Button(f, text="...", width=3,
                   command=self._browse_out).grid(row=r, column=2)
        r += 1

        def num(label, default):
            nonlocal r
            ttk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=3)
            v = tk.StringVar(value=str(default))
            ttk.Entry(f, textvariable=v, width=12).grid(
                row=r, column=1, sticky="w", padx=4)
            r += 1
            return v

        self.v_n = num("Число голов:", 10)
        self.v_std = num("Сила различий — std (min):", 1.0)

        self.v_use_range = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Диапазон std (на каждую голову свой)",
                        variable=self.v_use_range).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=3)
        r += 1
        self.v_std_max = num("  std max (если диапазон):", 1.5)

        self.v_ncomp = num("Число shape-компонент (1..300):", 30)
        self.v_seed = num("Seed:", 0)

        self.v_norm = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Нормализовать в bbox",
                        variable=self.v_norm).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=3)
        r += 1

        self.btn = ttk.Button(f, text="▶ Сгенерировать",
                              command=self._on_generate)
        self.btn.grid(row=r, column=0, columnspan=3, pady=8, sticky="ew")
        r += 1

        ttk.Button(f, text="👁 Просмотр головы (выбрать FBX/OBJ)",
                   command=self._on_view).grid(
            row=r, column=0, columnspan=3, pady=(0, 6), sticky="ew")
        r += 1

        self.pbar = ttk.Progressbar(f, mode="determinate")
        self.pbar.grid(row=r, column=0, columnspan=3, sticky="ew")
        r += 1
        self.lbl = ttk.Label(f, text="готов")
        self.lbl.grid(row=r, column=0, columnspan=3, sticky="w", pady=4)

        f.columnconfigure(1, weight=1)

    def _browse_flame(self):
        p = filedialog.askopenfilename(
            title="FLAME .pkl", filetypes=[("pickle", "*.pkl"), ("all", "*.*")])
        if p:
            self.v_flame.set(p)

    def _browse_out(self):
        p = filedialog.askdirectory(title="Папка вывода")
        if p:
            self.v_out.set(p)

    def _on_view(self):
        """Выбрать меш и открыть в Open3D (подпроцесс — legacy draw_geometries
        нельзя в одном процессе с tkinter). Стартовая папка — папка вывода."""
        import sys
        start = self.v_out.get().strip() or "."
        p = filedialog.askopenfilename(
            title="Выбери голову для просмотра",
            initialdir=start if Path(start).exists() else ".",
            filetypes=[("Mesh", "*.fbx *.obj *.ply *.stl"), ("all", "*.*")])
        if not p:
            return
        here = Path(__file__).resolve().parent
        cmd = [sys.executable, str(here / "mesh_viewer.py"), p]

        def worker():
            try:
                subprocess.run(cmd)
            except Exception as e:
                print(f"Mesh viewer error: {e}")
        threading.Thread(target=worker, daemon=True).start()
        self.lbl.config(text=f"открываю: {Path(p).name}")

    def _on_generate(self):
        try:
            n = int(self.v_n.get())
            std = float(self.v_std.get())
            ncomp = int(self.v_ncomp.get())
            seed = int(self.v_seed.get())
            std_max = (float(self.v_std_max.get())
                       if self.v_use_range.get() else None)
        except ValueError:
            self.lbl.config(text="ошибка: числовые поля заполнены неверно")
            return
        flame_path = self.v_flame.get().strip()
        out_dir = self.v_out.get().strip()
        if not Path(flame_path).exists():
            self.lbl.config(text=f"FLAME не найден: {flame_path}")
            return
        self.btn.config(state="disabled")
        self.pbar["maximum"] = n
        self.pbar["value"] = 0

        def prog(i, total, name):
            self.root.after(0, lambda: (self.pbar.config(value=i),
                                        self.lbl.config(
                                            text=f"{i}/{total}: {name}")))

        def worker():
            try:
                if self.flame is None:
                    self.root.after(0, lambda: self.lbl.config(
                        text="загрузка FLAME..."))
                    self.flame = pipe.load_flame(flame_path)
                v_t, sd, faces = self.flame
                paths = generate_heads(
                    v_t, sd, faces, out_dir, n_heads=n, std=std,
                    std_max=std_max, n_components=ncomp, seed=seed,
                    normalize=self.v_norm.get(), progress=prog)
                self.root.after(0, lambda: self.lbl.config(
                    text=f"✓ готово: {len(paths)} голов → {out_dir}/"))
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.root.after(0, lambda: self.lbl.config(
                    text=f"ошибка: {e}"))
            finally:
                self.root.after(0, lambda: self.btn.config(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    HeadGeneratorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
