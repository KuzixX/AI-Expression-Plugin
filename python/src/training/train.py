"""
Тренировочный цикл: DiffusionNet-skinning на датасете голов (transferred.h5).

Батч = головы (атомарная единица). Для каждой головы:
  фичи → модель → W_pred → skinning(W, act, dir) → δ_pred → лосс по всем δ_target.
Сплит ПО головам (без утечки). Grad accumulation по головам в батче.

  python -m src.training.train --data data/transferred.h5 --epochs 50
  (запускать из python/)
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_SRC.parent))

from src.data.dataset import HeadDataset, collate_single        # noqa: E402
from src.data.muscles import MuscleRig, make_dummy_rig          # noqa: E402
from src.models.skinning_net import MuscleSkinningNet           # noqa: E402
from src.deformation.skinning import skin_deformation           # noqa: E402
from src.training.losses import total_loss                     # noqa: E402


def split_heads(names, val_frac=0.15, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(names))
    n_val = max(1, int(len(names) * val_frac))
    val = set(names[i] for i in idx[:n_val])
    train = [n for n in names if n not in val]
    return train, [n for n in names if n in val]


def move(batch_item, device):
    for k in ("verts", "feats", "mass", "evals", "evecs", "gradX", "gradY",
              "delta", "activations"):
        batch_item[k] = batch_item[k].to(device)
    return batch_item


def build_rig(ds, neutral_for_dummy):
    """Риг из reference (если есть origin/direction) или заглушка для обкатки."""
    # пока reference хранит только activations → используем dummy-риг.
    # Когда добавишь origin/direction в HDF5 — здесь читать настоящий MuscleRig.
    return make_dummy_rig(ds.n_muscles, neutral_for_dummy)


def run(args, progress_cb=None, should_stop=None, log=print):
    """progress_cb(epoch, n_epochs, train_loss, val_loss) — для GUI.
    should_stop() → True прерывает обучение. log(str) — вывод сообщений."""
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"device: {device}")

    full = HeadDataset(args.data, k_eig=args.k_eig, use_wks=args.use_wks)
    log(f"heads: {len(full)}, muscles: {full.n_muscles}, "
          f"expr: {len(full.expr_names)}, C_in: {full.feature_dim()}")
    tr_names, va_names = split_heads(full.heads, args.val_frac)
    train_ds = HeadDataset(args.data, k_eig=args.k_eig, use_wks=args.use_wks,
                           head_names=tr_names)
    val_ds = HeadDataset(args.data, k_eig=args.k_eig, use_wks=args.use_wks,
                         head_names=va_names)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          collate_fn=collate_single, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_single, num_workers=0)

    # риг (dummy пока) — нужен один раз; origin/dir в общем reference-кадре
    sample0 = full[0]
    rig = build_rig(full, sample0["verts"].numpy())
    muscle_feats = torch.from_numpy(rig.muscle_features()).to(device)
    directions = torch.from_numpy(rig.directions).to(device)

    model = MuscleSkinningNet(
        C_in=full.feature_dim(), F_m=muscle_feats.shape[1],
        n_muscles=full.n_muscles, D=args.dim,
        diff_width=args.width, diff_blocks=args.blocks).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    log(f"params: {sum(p.numel() for p in model.parameters()):,}")

    ckpt_dir = Path(args.ckpt); ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(args.epochs):
        if should_stop is not None and should_stop():
            log("stopped by user"); break
        model.train(); t0 = time.time(); run_loss = 0.0; nb = 0
        for batch in train_ld:
            opt.zero_grad()
            bl = 0.0
            for item in batch:
                item = move(item, device)
                geo, align = rig.pair_priors(item["verts"].cpu().numpy())
                geo = torch.from_numpy(geo).to(device)
                align = torch.from_numpy(align).to(device)
                W = model(item["feats"], item["mass"], item["evals"],
                          item["evecs"], item["gradX"], item["gradY"],
                          muscle_feats, geo, align)
                d_pred = skin_deformation(W, item["activations"], directions)
                L, comps = total_loss(
                    d_pred, item["delta"], W,
                    w_deform=args.w_deform, w_smooth=args.w_smooth,
                    w_sparse=args.w_sparse)
                (L / len(batch)).backward()
                bl += comps["total"]
            opt.step()
            run_loss += bl / len(batch); nb += 1
        tr = run_loss / max(nb, 1)

        # валидация
        model.eval(); vl = 0.0; nv = 0
        with torch.no_grad():
            for batch in val_ld:
                for item in batch:
                    item = move(item, device)
                    geo, align = rig.pair_priors(item["verts"].cpu().numpy())
                    geo = torch.from_numpy(geo).to(device)
                    align = torch.from_numpy(align).to(device)
                    W = model(item["feats"], item["mass"], item["evals"],
                              item["evecs"], item["gradX"], item["gradY"],
                              muscle_feats, geo, align)
                    d_pred = skin_deformation(W, item["activations"], directions)
                    L, _ = total_loss(d_pred, item["delta"], W,
                                      w_deform=args.w_deform,
                                      w_smooth=args.w_smooth,
                                      w_sparse=args.w_sparse)
                    vl += L.item(); nv += 1
        va = vl / max(nv, 1)
        log(f"epoch {epoch+1}/{args.epochs}  train {tr:.4f}  val {va:.4f}  "
            f"({time.time()-t0:.1f}s)")
        if progress_cb is not None:
            progress_cb(epoch + 1, args.epochs, tr, va)
        if va < best_val:
            best_val = va
            torch.save({"model": model.state_dict(),
                        "n_muscles": full.n_muscles,
                        "C_in": full.feature_dim(),
                        "F_m": muscle_feats.shape[1],
                        "dim": args.dim, "width": args.width,
                        "blocks": args.blocks, "use_wks": args.use_wks},
                       ckpt_dir / "best.pt")
            log(f"  ✓ saved best (val {va:.4f}) → {ckpt_dir/'best.pt'}")
    return str(ckpt_dir / "best.pt"), best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/transferred.h5")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--k-eig", type=int, default=128)
    ap.add_argument("--use-wks", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--w-deform", type=float, default=10.0)
    ap.add_argument("--w-smooth", type=float, default=0.3)
    ap.add_argument("--w-sparse", type=float, default=0.005)
    ap.add_argument("--ckpt", default="models/checkpoints")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
