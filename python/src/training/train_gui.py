"""
GUI обучения skinning-сети (tkinter).

Загружаешь датасет (transferred.h5), настраиваешь гиперпараметры, жмёшь
«Обучить». Прогресс, лог и график train/val loss — в окне. Обучение в фоновом
потоке (окно не виснет), можно остановить.

  cd python
  python -m src.training.train_gui
"""
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from types import SimpleNamespace

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

from src.training.train import run as train_run            # noqa: E402


class TrainGUI:
    def __init__(self, root):
        self.root = root
        root.title("Skinning Net — Training")
        root.geometry("720x640")
        self._stop = False
        self._hist = {"epoch": [], "train": [], "val": []}
        self._build()

    def _row(self, parent, label, default, tip=""):
        fr = ttk.Frame(parent); fr.pack(fill="x", pady=2)
        ttk.Label(fr, text=label, width=22, anchor="w").pack(side="left")
        v = tk.StringVar(value=str(default))
        ttk.Entry(fr, textvariable=v, width=14).pack(side="left")
        if tip:
            ttk.Label(fr, text=tip, font=("", 8), foreground="#666").pack(
                side="left", padx=6)
        return v

    def _build(self):
        f = ttk.Frame(self.root, padding=12); f.pack(fill="both", expand=True)

        # dataset
        dr = ttk.Frame(f); dr.pack(fill="x", pady=4)
        ttk.Label(dr, text="Dataset (transferred.h5):", width=22,
                  anchor="w").pack(side="left")
        self.v_data = tk.StringVar(value="data/transferred.h5")
        ttk.Entry(dr, textvariable=self.v_data).pack(
            side="left", fill="x", expand=True, padx=4)

        def browse():
            p = filedialog.askopenfilename(
                title="Dataset HDF5", filetypes=[("HDF5", "*.h5")])
            if p:
                self.v_data.set(p)
        ttk.Button(dr, text="...", width=3, command=browse).pack(side="left")

        # ckpt dir
        cr = ttk.Frame(f); cr.pack(fill="x", pady=4)
        ttk.Label(cr, text="Checkpoint dir:", width=22, anchor="w").pack(
            side="left")
        self.v_ckpt = tk.StringVar(value="models/checkpoints")
        ttk.Entry(cr, textvariable=self.v_ckpt).pack(
            side="left", fill="x", expand=True, padx=4)

        # hyperparams (две колонки)
        cols = ttk.Frame(f); cols.pack(fill="x", pady=6)
        left = ttk.Frame(cols); left.pack(side="left", fill="x", expand=True)
        right = ttk.Frame(cols); right.pack(side="left", fill="x", expand=True)
        self.v_epochs = self._row(left, "epochs:", 50)
        self.v_batch = self._row(left, "batch (heads):", 4)
        self.v_lr = self._row(left, "learning rate:", 0.001)
        self.v_keig = self._row(left, "k eigenvectors:", 128, "spectral res")
        self.v_dim = self._row(right, "embed dim D:", 128)
        self.v_width = self._row(right, "diff width:", 128)
        self.v_blocks = self._row(right, "diff blocks:", 4)
        self.v_wks = tk.BooleanVar(value=False)
        ttk.Checkbutton(right, text="use WKS features",
                        variable=self.v_wks).pack(anchor="w", pady=2)

        # loss weights
        lw = ttk.Frame(f); lw.pack(fill="x")
        self.v_wdef = self._row(lw, "w_deform:", 10.0)
        self.v_wsm = self._row(lw, "w_smooth:", 0.0, "0=off (needs Lap op)")
        self.v_wsp = self._row(lw, "w_sparse:", 0.005)

        # buttons
        br = ttk.Frame(f); br.pack(fill="x", pady=8)
        self.btn = ttk.Button(br, text="▶ Train", command=self._on_train)
        self.btn.pack(side="left")
        self.btn_stop = ttk.Button(br, text="■ Stop", command=self._on_stop,
                                   state="disabled")
        self.btn_stop.pack(side="left", padx=6)

        self.pbar = ttk.Progressbar(f, mode="determinate")
        self.pbar.pack(fill="x", pady=2)
        self.lbl = ttk.Label(f, text="ready"); self.lbl.pack(anchor="w")

        # loss plot
        self.plot = tk.Label(f, background="#fff")
        self.plot.pack(fill="x", pady=4)

        # log
        self.log = tk.Text(f, height=8, wrap="word")
        self.log.pack(fill="both", expand=True)

    def _logmsg(self, s):
        self.root.after(0, lambda: (self.log.insert("end", s + "\n"),
                                    self.log.see("end")))

    def _on_stop(self):
        self._stop = True
        self.lbl.config(text="stopping...")

    def _draw_plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from PIL import Image, ImageTk
            import io
            fig, ax = plt.subplots(figsize=(6, 2.2), dpi=90)
            ax.plot(self._hist["epoch"], self._hist["train"], label="train")
            ax.plot(self._hist["epoch"], self._hist["val"], label="val")
            ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
            buf = io.BytesIO(); fig.tight_layout()
            fig.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
            img = ImageTk.PhotoImage(Image.open(buf))
            self._plot_img = img
            self.plot.config(image=img)
        except Exception as e:
            self._logmsg(f"plot error: {e}")

    def _on_train(self):
        data = self.v_data.get().strip()
        if not Path(data).exists():
            self.lbl.config(text=f"dataset not found: {data}"); return
        try:
            args = SimpleNamespace(
                data=data, ckpt=self.v_ckpt.get().strip(),
                epochs=int(self.v_epochs.get()), batch=int(self.v_batch.get()),
                lr=float(self.v_lr.get()), k_eig=int(self.v_keig.get()),
                dim=int(self.v_dim.get()), width=int(self.v_width.get()),
                blocks=int(self.v_blocks.get()), use_wks=self.v_wks.get(),
                val_frac=0.15, w_deform=float(self.v_wdef.get()),
                w_smooth=float(self.v_wsm.get()),
                w_sparse=float(self.v_wsp.get()))
        except ValueError:
            self.lbl.config(text="error: a numeric field is invalid"); return

        self._stop = False
        self._hist = {"epoch": [], "train": [], "val": []}
        self.log.delete("1.0", "end")
        self.btn.config(state="disabled"); self.btn_stop.config(state="normal")
        self.pbar["maximum"] = args.epochs; self.pbar["value"] = 0

        def prog(ep, n, tr, va):
            self._hist["epoch"].append(ep)
            self._hist["train"].append(tr); self._hist["val"].append(va)
            self.root.after(0, lambda: (
                self.pbar.config(value=ep),
                self.lbl.config(text=f"epoch {ep}/{n}  train {tr:.4f}  "
                                     f"val {va:.4f}"),
                self._draw_plot()))

        def worker():
            try:
                path, best = train_run(
                    args, progress_cb=prog,
                    should_stop=lambda: self._stop, log=self._logmsg)
                self.root.after(0, lambda: self.lbl.config(
                    text=f"✓ done. best val {best:.4f} → {path}"))
            except Exception as e:
                import traceback; traceback.print_exc()
                self.root.after(0, lambda: self.lbl.config(text=f"error: {e}"))
            finally:
                self.root.after(0, lambda: (
                    self.btn.config(state="normal"),
                    self.btn_stop.config(state="disabled")))

        threading.Thread(target=worker, daemon=True).start()


def main():
    root = tk.Tk()
    TrainGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
