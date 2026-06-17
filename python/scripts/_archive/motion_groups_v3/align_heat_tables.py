"""
align_heat_tables.py — выравнивание двух heat-таблиц (FLAME и FBX) в общую
систему координат через два метода:

  Метод A (default): Nearest-neighbor в K-мерном heat-space.
    Для каждой FLAME-вершины ищем FBX-вершину с самым похожим heat-вектором.
    Output: aligned table (N_flame × K) где строка i означает "одна и та же
    анатомическая точка".

  Метод B: Канонические (anchor, percentile) координаты.
    Не зависит от топологии — обе меши пересэмплируются на регулярную сетку
    (K_anchors × P_percentiles). Result: две таблицы одинакового размера
    (K*P × K) независимо от N_flame и N_fbx.

Использование:
    python python/scripts/motion_groups_v3/align_heat_tables.py
        → последний run, оба метода

    python python/scripts/motion_groups_v3/align_heat_tables.py --run <path> --method A
    python python/scripts/motion_groups_v3/align_heat_tables.py --method B --n-percentiles 50

Output (в <run>/aligned/):
    method_A_correspondence.csv   — flame_vertex, fbx_vertex, match_distance
    method_A_heat_head1.csv       — оригинал HEAD1 heat (для удобства, N_flame × K)
    method_A_heat_fbx_aligned.csv — FBX heat пересортированный по FLAME (N_flame × K)
    method_A_residual.csv         — || H_flame - H_fbx_aligned || per FLAME vertex
    method_A_quality.png          — гистограмма match-distance + 3D скаттер качества

    method_B_canonical_head1.csv  — HEAD1 на регулярной (anchor, percentile) сетке
    method_B_canonical_fbx.csv    — FBX на той же сетке
    method_B_diff.csv             — попарная разность
    method_B_compare.png          — heatmap визуализация обеих таблиц + diff
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def latest_run(base="python/scripts/debug_output"):
    runs = sorted(Path(base).glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"Нет run_* в {base}/")
    return runs[-1]


def load_heat(run_dir: Path, sub: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Returns (H, df) where H is (N, K) and df is raw DataFrame."""
    p = run_dir / sub / "heat.csv"
    if not p.exists():
        raise FileNotFoundError(f"Нет {p}")
    df = pd.read_csv(p)
    return df.values.astype(np.float64), df


def normalize_per_anchor(H: np.ndarray) -> np.ndarray:
    """Per-anchor max-normalization → [0,1] per column.
    Делает таблицы comparable между мешами разной плотности."""
    return H / H.max(axis=0, keepdims=True).clip(min=1e-12)


# ── МЕТОД A: nearest-neighbor в heat-space ───────────────────────────────────

def method_A(H_flame: np.ndarray, H_fbx: np.ndarray,
              out_dir: Path, verts_flame: np.ndarray | None = None):
    """Для каждой FLAME-вершины находим ближайшую FBX-вершину в heat-space.

    Используем per-anchor max-нормализацию (heat absolute scales разные между
    мешами разной плотности — это критично).

    Возвращает: corr (N_flame,) — индексы FBX-вершин, и dists (N_flame,).
    """
    print("\n── Метод A: nearest-neighbor в K-мерном heat-space ──")
    N1, K = H_flame.shape
    N2, _ = H_fbx.shape

    H1n = normalize_per_anchor(H_flame)
    H2n = normalize_per_anchor(H_fbx)

    # L2 расстояние между каждой FLAME-вершиной и каждой FBX-вершиной
    # || h1 - h2 ||² = ||h1||² + ||h2||² - 2·h1·h2
    # для N1=5k × N2=8k это 40M операций — OK в numpy
    print(f"  Вычисляю pairwise distances {N1} × {N2}...")
    h1_sq = (H1n ** 2).sum(1, keepdims=True)       # (N1, 1)
    h2_sq = (H2n ** 2).sum(1, keepdims=True).T     # (1, N2)
    cross = H1n @ H2n.T                             # (N1, N2)
    D2 = h1_sq + h2_sq - 2 * cross
    D2 = np.maximum(D2, 0)

    # argmin по оси 1: для каждой FLAME найти ближайший FBX
    corr = np.argmin(D2, axis=1)                    # (N1,)
    dists = np.sqrt(D2[np.arange(N1), corr])        # (N1,)

    # Aligned table: H_fbx_aligned[i] = H_fbx[corr[i]]
    H_fbx_aligned = H_fbx[corr]                     # (N1, K)
    residual = np.linalg.norm(
        normalize_per_anchor(H_flame) - normalize_per_anchor(H_fbx_aligned),
        axis=1
    )

    # ── Save tables ───────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [f"anchor_{i}" for i in range(K)]

    pd.DataFrame({
        "flame_vertex": np.arange(N1),
        "fbx_vertex":   corr,
        "match_distance": dists,
    }).to_csv(out_dir / "method_A_correspondence.csv", index=False)

    pd.DataFrame(H_flame, columns=cols).to_csv(
        out_dir / "method_A_heat_head1.csv", index=False)

    pd.DataFrame(H_fbx_aligned, columns=cols).to_csv(
        out_dir / "method_A_heat_fbx_aligned.csv", index=False)

    pd.DataFrame({
        "flame_vertex": np.arange(N1),
        "residual_l2":  residual,
    }).to_csv(out_dir / "method_A_residual.csv", index=False)

    print(f"  ✓ saved 4 CSV в {out_dir}/")
    print(f"    Mean match distance:   {dists.mean():.4f}")
    print(f"    Median match distance: {np.median(dists):.4f}")
    print(f"    Max match distance:    {dists.max():.4f}")
    print(f"    Mean residual:         {residual.mean():.4f}")

    # ── Quality plot ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 5))

    # (1) Hist match distance
    ax1 = fig.add_subplot(131)
    ax1.hist(dists, bins=80, color='steelblue', alpha=0.8)
    ax1.axvline(dists.mean(), color='red', linestyle='--',
                 label=f"mean={dists.mean():.3f}")
    ax1.axvline(np.median(dists), color='orange', linestyle='--',
                 label=f"median={np.median(dists):.3f}")
    ax1.set_xlabel("match distance (L2 in normalized heat space)")
    ax1.set_ylabel("# FLAME vertices")
    ax1.set_title("Method A: match quality distribution")
    ax1.legend()

    # (2) Residual hist
    ax2 = fig.add_subplot(132)
    ax2.hist(residual, bins=80, color='darkorange', alpha=0.8)
    ax2.set_xlabel("|| h_flame - h_fbx_aligned ||")
    ax2.set_ylabel("# FLAME vertices")
    ax2.set_title("Method A: post-alignment residual")

    # (3) 3D scatter colored by match quality (if verts available)
    if verts_flame is not None:
        ax3 = fig.add_subplot(133, projection='3d')
        sc = ax3.scatter(verts_flame[:, 0], verts_flame[:, 1], verts_flame[:, 2],
                          c=dists, cmap='RdYlGn_r', s=3, alpha=0.7,
                          vmin=0, vmax=np.percentile(dists, 95))
        ax3.set_title("Match quality on FLAME mesh\n(green=good, red=bad)")
        plt.colorbar(sc, ax=ax3, shrink=0.6, label="distance")
    else:
        ax3 = fig.add_subplot(133)
        sorted_d = np.sort(dists)
        ax3.plot(sorted_d, color='steelblue')
        ax3.set_xlabel("FLAME vertex rank")
        ax3.set_ylabel("match distance")
        ax3.set_title("Sorted match distances")
        ax3.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_dir / "method_A_quality.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved method_A_quality.png")

    return corr, dists, H_fbx_aligned


# ── МЕТОД B: канонические (anchor, percentile) координаты ────────────────────

def method_B(H_flame: np.ndarray, H_fbx: np.ndarray, out_dir: Path,
              n_percentiles: int = 50, active_threshold: float = 0.05):
    """Регулярная сетка (anchor × percentile) — независимо от топологии.

    Для каждой меши:
      1. Для каждого anchor a:
         - Берём активные вершины (heat > threshold * max)
         - Сортируем по heat descending
         - Вычисляем процентили [0, 1/P, 2/P, ..., 1]
         - Для каждого процентиля p берём heat в этой точке распределения
         - А ТАКЖЕ heat от ДРУГИХ anchor'ов в этой же точке (K-1 величин)
      2. Получаем таблицу (P, K) — для anchor a, на процентиле p, K heat-значений

    Итого: 3D массив (K, P, K) для каждой меши.
    Сплющиваем в 2D таблицу (K*P, K) для CSV.
    """
    print(f"\n── Метод B: канонические координаты (P={n_percentiles}) ──")
    K = H_flame.shape[1]
    P = n_percentiles
    percentile_targets = np.linspace(0, 1, P)

    def resample(H):
        N = len(H)
        canon = np.zeros((K, P, K))                 # (anchor, percentile, all-K-heat)
        for a in range(K):
            ha = H[:, a]
            ha_max = max(ha.max(), 1e-12)
            active = ha > active_threshold * ha_max
            if active.sum() < 2:
                continue
            idx_active = np.where(active)[0]
            # сортируем по heat от anchor a (descending)
            order = np.argsort(-ha[idx_active])
            sorted_idx = idx_active[order]
            ranks = np.linspace(0, 1, len(sorted_idx))
            # для каждого percentile интерполируем по ranks
            for k in range(K):
                vals = H[sorted_idx, k]
                canon[a, :, k] = np.interp(percentile_targets, ranks, vals)
        return canon

    canon_flame = resample(H_flame)
    canon_fbx   = resample(H_fbx)

    # ── Сплющиваем в 2D таблицы для CSV ───────────────────────────────────────
    rows = []
    for a in range(K):
        for p_idx in range(P):
            rows.append((a, percentile_targets[p_idx]))
    coord_df = pd.DataFrame(rows, columns=["anchor_idx", "percentile"])

    cols = [f"heat_a{i}" for i in range(K)]
    flat_flame = canon_flame.reshape(-1, K)
    flat_fbx   = canon_fbx.reshape(-1, K)

    df_flame = pd.concat([coord_df, pd.DataFrame(flat_flame, columns=cols)], axis=1)
    df_fbx   = pd.concat([coord_df, pd.DataFrame(flat_fbx,   columns=cols)], axis=1)
    df_diff  = pd.concat([coord_df, pd.DataFrame(flat_flame - flat_fbx, columns=cols)], axis=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    df_flame.to_csv(out_dir / "method_B_canonical_head1.csv", index=False)
    df_fbx.to_csv(  out_dir / "method_B_canonical_fbx.csv",   index=False)
    df_diff.to_csv( out_dir / "method_B_diff.csv",            index=False)
    print(f"  ✓ saved 3 CSV в {out_dir}/")
    print(f"    Размер каждой таблицы: {K*P} строк × {K} heat-колонок")

    mean_abs_diff = np.abs(flat_flame - flat_fbx).mean()
    max_abs_diff  = np.abs(flat_flame - flat_fbx).max()
    print(f"    Mean |HEAD1 - FBX| (canonical): {mean_abs_diff:.4f}")
    print(f"    Max  |HEAD1 - FBX| (canonical): {max_abs_diff:.4f}")

    # ── Compare plot ──────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, K, figsize=(2.8 * K, 9), sharex=True)
    if K == 1: axes = axes.reshape(-1, 1)
    for a in range(K):
        # row 0: HEAD1
        im0 = axes[0, a].imshow(canon_flame[a], aspect='auto', cmap='magma',
                                  extent=[0, K-1, 1, 0])
        axes[0, a].set_title(f"HEAD1: anchor {a}")
        if a == 0: axes[0, a].set_ylabel("percentile\n(0=hot, 1=cold)")
        plt.colorbar(im0, ax=axes[0, a], fraction=0.04)

        # row 1: FBX
        im1 = axes[1, a].imshow(canon_fbx[a], aspect='auto', cmap='magma',
                                  extent=[0, K-1, 1, 0])
        axes[1, a].set_title(f"FBX: anchor {a}")
        if a == 0: axes[1, a].set_ylabel("percentile")
        plt.colorbar(im1, ax=axes[1, a], fraction=0.04)

        # row 2: diff
        diff = canon_flame[a] - canon_fbx[a]
        vmax = np.abs(diff).max()
        im2 = axes[2, a].imshow(diff, aspect='auto', cmap='RdBu_r',
                                  extent=[0, K-1, 1, 0], vmin=-vmax, vmax=vmax)
        axes[2, a].set_title(f"diff: anchor {a}")
        axes[2, a].set_xlabel("heat from anchor →")
        if a == 0: axes[2, a].set_ylabel("percentile")
        plt.colorbar(im2, ax=axes[2, a], fraction=0.04)

    fig.suptitle("Method B: HEAD1 vs FBX in canonical (anchor, percentile) coords",
                  y=1.005, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "method_B_compare.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ saved method_B_compare.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                     help="путь к run_*; default = самый свежий")
    ap.add_argument("--base", type=str, default="python/scripts/debug_output")
    ap.add_argument("--method", choices=["A", "B", "both"], default="both")
    ap.add_argument("--n-percentiles", type=int, default=50,
                     help="для метода B")
    args = ap.parse_args()

    run_dir = Path(args.run) if args.run else latest_run(args.base)
    run_dir = run_dir.resolve()
    print(f"● Run: {run_dir}")
    if not run_dir.exists():
        print("✗ Не существует"); sys.exit(1)

    H_flame, _ = load_heat(run_dir, "head1")
    H_fbx,   _ = load_heat(run_dir, "fbx")
    print(f"  HEAD1: {H_flame.shape[0]} verts × {H_flame.shape[1]} anchors")
    print(f"  FBX:   {H_fbx.shape[0]} verts × {H_fbx.shape[1]} anchors")
    assert H_flame.shape[1] == H_fbx.shape[1], "K_anchors должно совпадать"

    # FLAME vertex positions (для 3D scatter в методе A)
    v_path = run_dir / "head1" / "verts_rest.csv"
    verts_flame = pd.read_csv(v_path).values if v_path.exists() else None

    out_dir = run_dir / "aligned"
    out_dir.mkdir(exist_ok=True)
    print(f"● Output → {out_dir}/")

    if args.method in ("A", "both"):
        method_A(H_flame, H_fbx, out_dir, verts_flame=verts_flame)

    if args.method in ("B", "both"):
        method_B(H_flame, H_fbx, out_dir, n_percentiles=args.n_percentiles)

    print(f"\n✓ Готово. Всё в: {out_dir}/")


if __name__ == "__main__":
    main()
