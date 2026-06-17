"""
visualize_dumps.py — оффлайн-визуализация всех CSV/JSON дампов из run_*/.

Использование:
    python python/scripts/motion_groups_v3/visualize_dumps.py
        → берёт самый свежий run_* в python/scripts/debug_output/

    python python/scripts/motion_groups_v3/visualize_dumps.py --run <path>
        → конкретный run

Сохраняет PNG-файлы в <run>/plots/{head1,fbx,compare}/ — отдельная папка,
не мешает основному пайплайну. Использует matplotlib + seaborn чисто оффлайн
(никаких Open3D event loops, никаких GUI).

Графики:

  HEAT MATRIX (per mesh):
    - heat_matrix_log.png    — full heatmap (anchors × vertices), log color
    - heat_per_anchor_hist.png — гистограмма per-anchor
    - heat_decay_curves.png  — sorted decay для каждого anchor'а
    - heat_corr.png          — корреляция anchor'ов между собой
    - heat_svd_spectrum.png  — SVD-сингулярные значения heat-матрицы
    - heat_pca_2d.png        — PCA-проекция per-vertex heat-векторов
    - heat_3d_anchor_<i>.png — 3D scatter верт. с раскраской по heat от anchor i

  CLUSTERS (per mesh):
    - cluster_sizes.png      — bar plot размеров кластеров
    - cluster_heat_box.png   — boxplot heat_weight внутри каждого кластера
    - cluster_3d.png         — 3D scatter с раскраской по global cluster id
    - cluster_per_anchor.png — стек bar — сколько верт. в каждом anchor'е

  DELTA (per mesh):
    - delta_magnitude_hist.png  — гистограмма ||δ||
    - delta_xyz_components.png  — 3 hist'а по компонентам δx, δy, δz
    - delta_3d.png              — 3D scatter, цвет = ||δ||
    - delta_vs_heat.png         — scatter ||δ|| vs heat_total

  COMPARE (если есть оба head1 + fbx):
    - heat_compare_anchor_<i>.png — log-log scatter heat_head1[v_anchor] vs heat_fbx[v_anchor]
    - svd_compare.png             — спектры обеих heat-матриц
    - cluster_count_compare.png   — сравнение распределения кластеров
"""

from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

# matplotlib — НЕ интерактивный backend, только PNG
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted")
    HAS_SNS = True
except ImportError:
    HAS_SNS = False
    print("⚠ seaborn не установлен, графики будут проще: pip install seaborn")


# ── utility ──────────────────────────────────────────────────────────────────

def latest_run(base="python/scripts/debug_output"):
    runs = sorted(Path(base).glob("run_*"), key=lambda p: p.stat().st_mtime)
    if not runs:
        raise FileNotFoundError(f"Не нашёл ни одного run_* в {base}/")
    return runs[-1]


def safe_load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"  ⚠ Не удалось прочитать {path}: {e}")
        return None


def safe_load_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  ⚠ Не удалось прочитать {path}: {e}")
        return None


def save_fig(fig, out_dir: Path, name: str, dpi=120):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"    → {p.relative_to(p.parents[3]) if len(p.parents) >= 4 else p}")


# ── HEAT plots ───────────────────────────────────────────────────────────────

def plot_heat_all(heat_df: pd.DataFrame, verts_df: pd.DataFrame | None,
                   anchor_idx_df: pd.DataFrame | None,
                   out_dir: Path, mesh_label: str):
    """heat_df: rows=vertices, cols=anchor_0..anchor_K-1"""
    if heat_df is None:
        print(f"  [{mesh_label}] heat.csv отсутствует, пропускаю heat-графики")
        return

    H = heat_df.values.T           # (K, N) — anchors × verts
    K, N = H.shape
    print(f"  [{mesh_label}] heat: {K} anchors × {N} verts")

    # 1. Полная heatmap (log scale)
    fig, ax = plt.subplots(figsize=(min(14, 0.002 * N + 6), max(3, 0.4 * K + 2)))
    H_pos = np.clip(H, 1e-12, None)
    im = ax.imshow(H_pos, aspect='auto', cmap='magma',
                    norm=LogNorm(vmin=H_pos.min(), vmax=H_pos.max()))
    ax.set_xlabel(f"vertex idx (N={N})")
    ax.set_ylabel("anchor idx")
    ax.set_title(f"{mesh_label}: heat matrix (log scale)")
    plt.colorbar(im, ax=ax, label="heat")
    save_fig(fig, out_dir, "heat_matrix_log.png")

    # 2. Hist per-anchor
    fig, axes = plt.subplots(K, 1, figsize=(10, 1.6 * K), sharex=False)
    if K == 1: axes = [axes]
    for a in range(K):
        h = H[a]
        h_log = np.log10(np.clip(h, 1e-12, None))
        axes[a].hist(h_log, bins=80, color=f"C{a%10}", alpha=0.8)
        axes[a].set_ylabel(f"a{a}\ncount")
        axes[a].set_xlabel("log10(heat)" if a == K-1 else "")
    fig.suptitle(f"{mesh_label}: heat distribution per anchor (log)")
    fig.tight_layout()
    save_fig(fig, out_dir, "heat_per_anchor_hist.png")

    # 3. Decay curves (sorted descending)
    fig, ax = plt.subplots(figsize=(10, 5))
    for a in range(K):
        h = np.sort(H[a])[::-1]
        h_norm = h / max(h.max(), 1e-12)
        ax.plot(h_norm, label=f"anchor {a}", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("vertex rank (sorted desc)")
    ax.set_ylabel("heat / max(heat)  [log]")
    ax.set_title(f"{mesh_label}: heat decay curves (per-anchor max-normalized)")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir, "heat_decay_curves.png")

    # 4. Корреляция между anchor'ами (по vertex-heat-вектору)
    if K >= 2:
        corr = np.corrcoef(H)
        fig, ax = plt.subplots(figsize=(0.5 * K + 3, 0.5 * K + 2))
        if HAS_SNS:
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                         center=0, vmin=-1, vmax=1, ax=ax,
                         xticklabels=[f"a{i}" for i in range(K)],
                         yticklabels=[f"a{i}" for i in range(K)])
        else:
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
        ax.set_title(f"{mesh_label}: anchor-anchor heat correlation")
        save_fig(fig, out_dir, "heat_corr.png")

    # 5. SVD spectrum
    H_norm = H / H.max(axis=1, keepdims=True).clip(min=1e-12)
    try:
        U, S, Vt = np.linalg.svd(H_norm, full_matrices=False)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.semilogy(np.arange(1, len(S)+1), S, "o-", color="steelblue")
        ax1.set_xlabel("component")
        ax1.set_ylabel("singular value (log)")
        ax1.set_title(f"{mesh_label}: SVD spectrum")
        ax1.grid(True, which="both", alpha=0.3)
        cum = np.cumsum(S**2) / max((S**2).sum(), 1e-12)
        ax2.plot(np.arange(1, len(cum)+1), cum, "o-", color="darkorange")
        ax2.axhline(0.95, color="red", linestyle="--", alpha=0.5, label="95%")
        ax2.axhline(0.99, color="purple", linestyle="--", alpha=0.5, label="99%")
        ax2.set_xlabel("component")
        ax2.set_ylabel("cumulative energy")
        ax2.set_title(f"{mesh_label}: energy retention")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        save_fig(fig, out_dir, "heat_svd_spectrum.png")
    except Exception as e:
        print(f"  ⚠ SVD не удался: {e}")

    # 6. PCA-проекция per-vertex heat-векторов
    if K >= 2:
        H_v = H.T                  # (N, K)
        H_v_c = H_v - H_v.mean(0)
        try:
            U_pca, S_pca, Vt_pca = np.linalg.svd(H_v_c, full_matrices=False)
            proj = U_pca[:, :2] * S_pca[:2]
            fig, ax = plt.subplots(figsize=(7, 7))
            # цвет по argmax-anchor
            argmax_a = np.argmax(H_v, axis=1)
            sc = ax.scatter(proj[:, 0], proj[:, 1], c=argmax_a, cmap="tab10",
                             s=3, alpha=0.5)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"{mesh_label}: PCA of per-vertex heat-vectors\n(color = argmax anchor)")
            plt.colorbar(sc, ax=ax, label="dominant anchor")
            save_fig(fig, out_dir, "heat_pca_2d.png")
        except Exception as e:
            print(f"  ⚠ PCA не удался: {e}")

    # 7. 3D scatter верт. с раскраской по heat (если есть verts_rest)
    if verts_df is not None:
        V = verts_df.values
        for a in range(K):
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection='3d')
            h = H[a]
            h_log = np.log10(np.clip(h, 1e-8, None))
            sc = ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=h_log, cmap='hot',
                             s=3, alpha=0.7)
            # anchor highlight
            if anchor_idx_df is not None:
                a_idx_arr = anchor_idx_df.values.ravel()
                if a < len(a_idx_arr):
                    a_v = int(a_idx_arr[a])
                    ax.scatter([V[a_v, 0]], [V[a_v, 1]], [V[a_v, 2]],
                                s=200, c='cyan', marker='*',
                                edgecolors='black', linewidths=2,
                                label=f"anchor {a}")
                    ax.legend(loc="best")
            ax.set_title(f"{mesh_label}: heat from anchor {a} (log)")
            plt.colorbar(sc, ax=ax, label="log10(heat)", shrink=0.6)
            save_fig(fig, out_dir, f"heat_3d_anchor_{a}.png")


# ── CLUSTER plots ────────────────────────────────────────────────────────────

def plot_clusters(clusters_flat_df: pd.DataFrame | None,
                   clusters_json: dict | list | None,
                   verts_df: pd.DataFrame | None,
                   out_dir: Path, mesh_label: str):
    if clusters_flat_df is None and clusters_json is None:
        print(f"  [{mesh_label}] нет данных о кластерах")
        return

    # Если flat CSV нет — построим из JSON на лету
    if clusters_flat_df is None and clusters_json is not None and isinstance(clusters_json, list):
        rows = []
        # плоский JSON: каждый dict содержит anchor_idx, cluster_idx, indices, heat_weights
        if clusters_json and isinstance(clusters_json[0], dict):
            # global cluster id присваиваем по порядку появления
            for g, cl in enumerate(clusters_json):
                a = int(cl.get("anchor_idx", 0))
                lc = int(cl.get("cluster_idx", 0))
                idx = cl.get("indices", [])
                hw = cl.get("heat_weights", [0.0] * len(idx))
                for v, h in zip(idx, hw):
                    rows.append((int(v), a, lc, g, float(h)))
        if rows:
            clusters_flat_df = pd.DataFrame(rows, columns=[
                "vertex_idx", "anchor_idx", "local_cluster_id",
                "global_cluster_id", "heat_weight"])
            print(f"  [{mesh_label}] построил clusters_flat из JSON ({len(rows)} rows)")

    if clusters_flat_df is not None:
        df = clusters_flat_df
        n_clusters = int(df["global_cluster_id"].max()) + 1
        sizes = df.groupby("global_cluster_id").size()
        K = int(df["anchor_idx"].max()) + 1
        print(f"  [{mesh_label}] {n_clusters} кластеров, {len(df)} vertex-assignments")

        # 1. Размеры
        fig, ax = plt.subplots(figsize=(max(6, 0.3 * n_clusters), 4))
        colors = [plt.cm.tab20(int(df[df["global_cluster_id"] == g]["anchor_idx"].iloc[0]) % 20)
                   for g in sizes.index]
        ax.bar(sizes.index, sizes.values, color=colors)
        ax.set_xlabel("global cluster id")
        ax.set_ylabel("# vertices")
        ax.set_title(f"{mesh_label}: cluster sizes (color = anchor)")
        save_fig(fig, out_dir, "cluster_sizes.png")

        # 2. Boxplot heat_weight per cluster
        fig, ax = plt.subplots(figsize=(max(6, 0.3 * n_clusters), 5))
        data_per = [df[df["global_cluster_id"] == g]["heat_weight"].values
                     for g in range(n_clusters)]
        ax.boxplot(data_per, positions=range(n_clusters), showfliers=False,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red'))
        ax.set_yscale("log")
        ax.set_xlabel("global cluster id")
        ax.set_ylabel("heat_weight (log)")
        ax.set_title(f"{mesh_label}: heat_weight distribution per cluster")
        save_fig(fig, out_dir, "cluster_heat_box.png")

        # 3. Per-anchor: сколько глобальных кластеров и vert'ов
        per_anchor = df.groupby("anchor_idx").agg(
            n_verts=("vertex_idx", "count"),
            n_clusters=("local_cluster_id", lambda x: x.nunique()),
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax1.bar(per_anchor.index, per_anchor["n_verts"], color="steelblue")
        ax1.set_xlabel("anchor"); ax1.set_ylabel("# vertices")
        ax1.set_title(f"{mesh_label}: vertices per anchor")
        ax2.bar(per_anchor.index, per_anchor["n_clusters"], color="darkorange")
        ax2.set_xlabel("anchor"); ax2.set_ylabel("# clusters")
        ax2.set_title(f"{mesh_label}: clusters per anchor")
        save_fig(fig, out_dir, "cluster_per_anchor.png")

        # 4. 3D scatter раскрашенный по global cluster id
        if verts_df is not None:
            V = verts_df.values
            # default — серый, потом перекрашиваем кластеризованные
            colors_v = np.tile([0.7, 0.7, 0.7], (len(V), 1))
            palette = plt.cm.hsv(np.linspace(0, 1, n_clusters, endpoint=False))[:, :3]
            for _, row in df.iterrows():
                v = int(row["vertex_idx"])
                g = int(row["global_cluster_id"])
                if 0 <= v < len(V):
                    colors_v[v] = palette[g % n_clusters]
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=colors_v, s=3, alpha=0.7)
            ax.set_title(f"{mesh_label}: vertices colored by global cluster id\n({n_clusters} clusters)")
            save_fig(fig, out_dir, "cluster_3d.png")

    # JSON-based stats (rotation angle, |mu|, stretches) — если есть
    if clusters_json is not None and isinstance(clusters_json, list):
        # Поддерживаем оба формата: плоский [cl, cl, ...] и вложенный [[cl, cl], [cl]]
        if clusters_json and isinstance(clusters_json[0], dict):
            cluster_iter = clusters_json
        else:
            cluster_iter = [cl for grp in clusters_json for cl in grp]
        rot_angles, mu_mags, stretches_all = [], [], []
        for cl in cluster_iter:
            if not isinstance(cl, dict):
                continue
            R = cl.get("R")
            if R is not None:
                R_arr = np.array(R)
                if R_arr.shape == (3, 3):
                    ang = np.arccos(np.clip((np.trace(R_arr) - 1) / 2, -1, 1))
                    rot_angles.append(np.degrees(ang))
            mu = cl.get("mu")
            if mu is not None:
                mu_mags.append(np.linalg.norm(mu))
            s = cl.get("stretches")
            if s is not None:
                stretches_all.append(np.array(s))

        if rot_angles and mu_mags:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            axes[0].hist(rot_angles, bins=20, color="darkred", alpha=0.8)
            axes[0].set_xlabel("rotation angle (°)")
            axes[0].set_title("per-cluster rotation magnitude")
            axes[1].hist(mu_mags, bins=20, color="darkblue", alpha=0.8)
            axes[1].set_xlabel("|μ|")
            axes[1].set_title("per-cluster translation magnitude")
            if stretches_all:
                S = np.array(stretches_all)
                for i, label in enumerate(["σ1", "σ2", "σ3"]):
                    axes[2].hist(S[:, i], bins=20, alpha=0.5, label=label)
                axes[2].set_xlabel("stretch")
                axes[2].set_title("per-cluster stretches (svd)")
                axes[2].legend()
            fig.suptitle(f"{mesh_label}: polar-decomp stats")
            fig.tight_layout()
            save_fig(fig, out_dir, "cluster_polar_decomp.png")


# ── DELTA plots ──────────────────────────────────────────────────────────────

def plot_deltas(out_dir: Path, mesh_label: str, **deltas):
    """deltas: kwargs {'native': df, 'recon': df, 'smoothed': df, 'raw': df}
    каждый df — N×3 (dx,dy,dz)."""
    have = {k: v for k, v in deltas.items() if v is not None}
    if not have:
        print(f"  [{mesh_label}] нет delta-файлов")
        return

    # 1. Magnitude hist (overlay)
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, df in have.items():
        d = df.values
        mag = np.linalg.norm(d, axis=1)
        mag = mag[mag > 1e-9]
        if len(mag):
            ax.hist(mag, bins=80, alpha=0.5, label=f"{k} (max={mag.max():.4f})")
    ax.set_xlabel("||δ||")
    ax.set_ylabel("count")
    ax.set_yscale("log")
    ax.set_title(f"{mesh_label}: delta magnitude distribution")
    ax.legend()
    save_fig(fig, out_dir, "delta_magnitude_hist.png")

    # 2. XYZ-компоненты (берём первый доступный)
    name, df = next(iter(have.items()))
    d = df.values
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, comp in enumerate(["δx", "δy", "δz"]):
        axes[i].hist(d[:, i], bins=80, color=f"C{i}", alpha=0.8)
        axes[i].set_xlabel(comp)
        axes[i].axvline(0, color="red", linestyle="--", alpha=0.5)
    fig.suptitle(f"{mesh_label}: delta XYZ components ({name})")
    fig.tight_layout()
    save_fig(fig, out_dir, "delta_xyz_components.png")


# ── COMPARE head1 vs fbx ─────────────────────────────────────────────────────

def plot_compare(head1_dir: Path, fbx_dir: Path, out_dir: Path):
    h1_heat = safe_load_csv(head1_dir / "heat.csv")
    fbx_heat = safe_load_csv(fbx_dir / "heat.csv")
    if h1_heat is None or fbx_heat is None:
        print("  [compare] нет одной из heat-таблиц, пропускаю")
        return

    H1 = h1_heat.values.T   # (K, N1)
    H2 = fbx_heat.values.T  # (K, N2)
    K = min(H1.shape[0], H2.shape[0])
    print(f"  [compare] K={K} общих anchor'ов")

    # 1. SVD-спектры обоих
    U1, S1, _ = np.linalg.svd(H1 / H1.max(axis=1, keepdims=True).clip(min=1e-12),
                                full_matrices=False)
    U2, S2, _ = np.linalg.svd(H2 / H2.max(axis=1, keepdims=True).clip(min=1e-12),
                                full_matrices=False)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(np.arange(1, len(S1)+1), S1, "o-", label=f"HEAD1 ({H1.shape[1]} verts)")
    ax.semilogy(np.arange(1, len(S2)+1), S2, "s-", label=f"FBX ({H2.shape[1]} verts)")
    ax.set_xlabel("component")
    ax.set_ylabel("singular value (log)")
    ax.set_title("SVD spectra comparison")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save_fig(fig, out_dir, "svd_compare.png")

    # 2. Sorted-heat curves per anchor — overlay HEAD1 vs FBX
    fig, axes = plt.subplots(K, 1, figsize=(10, 2.5 * K), sharex=False)
    if K == 1: axes = [axes]
    for a in range(K):
        h1 = np.sort(H1[a])[::-1]; h1 = h1 / max(h1.max(), 1e-12)
        h2 = np.sort(H2[a])[::-1]; h2 = h2 / max(h2.max(), 1e-12)
        # Сожмём по оси X в относительный rank
        x1 = np.linspace(0, 1, len(h1))
        x2 = np.linspace(0, 1, len(h2))
        axes[a].plot(x1, h1, label="HEAD1", color="C0")
        axes[a].plot(x2, h2, label="FBX",   color="C1")
        axes[a].set_yscale("log")
        axes[a].set_ylabel(f"anchor {a}\nheat/max")
        axes[a].grid(True, which="both", alpha=0.3)
        axes[a].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("relative vertex rank [0..1]")
    fig.suptitle("Heat decay curves — HEAD1 vs FBX (per-anchor normalized)")
    fig.tight_layout()
    save_fig(fig, out_dir, "heat_decay_compare.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default=None,
                     help="путь к run_*/; default = самый свежий")
    ap.add_argument("--base", type=str, default="python/scripts/debug_output",
                     help="папка где искать run_*")
    args = ap.parse_args()

    run_dir = Path(args.run) if args.run else latest_run(args.base)
    run_dir = run_dir.resolve()
    print(f"\n● Run: {run_dir}")
    if not run_dir.exists():
        print(f"  ✗ Не существует"); sys.exit(1)

    plots_root = run_dir / "plots"
    plots_root.mkdir(exist_ok=True)
    print(f"● Plots → {plots_root}/\n")

    # ── HEAD 1 ────────────────────────────────────────────────────────────────
    h1_dir = run_dir / "head1"
    if h1_dir.exists():
        print(f"▼ HEAD 1 ({h1_dir})")
        plots_h1 = plots_root / "head1"
        plot_heat_all(
            heat_df=safe_load_csv(h1_dir / "heat.csv"),
            verts_df=safe_load_csv(h1_dir / "verts_rest.csv"),
            anchor_idx_df=safe_load_csv(h1_dir / "anchor_indices.csv"),
            out_dir=plots_h1, mesh_label="HEAD1",
        )
        plot_clusters(
            clusters_flat_df=safe_load_csv(h1_dir / "clusters_flat.csv"),
            clusters_json=safe_load_json(h1_dir / "clusters.json"),
            verts_df=safe_load_csv(h1_dir / "verts_rest.csv"),
            out_dir=plots_h1, mesh_label="HEAD1",
        )
        plot_deltas(
            out_dir=plots_h1, mesh_label="HEAD1",
            native=safe_load_csv(h1_dir / "delta_native.csv"),
            recon=safe_load_csv(h1_dir / "delta_recon.csv"),
            smoothed=safe_load_csv(h1_dir / "delta_smoothed.csv"),
        )

    # ── FBX ───────────────────────────────────────────────────────────────────
    fbx_dir = run_dir / "fbx"
    if fbx_dir.exists():
        print(f"\n▼ FBX ({fbx_dir})")
        plots_fbx = plots_root / "fbx"
        plot_heat_all(
            heat_df=safe_load_csv(fbx_dir / "heat.csv"),
            verts_df=safe_load_csv(fbx_dir / "verts_rest.csv"),
            anchor_idx_df=safe_load_csv(fbx_dir / "anchor_indices.csv"),
            out_dir=plots_fbx, mesh_label="FBX",
        )
        # FBX hасто только target_clusters.json
        plot_clusters(
            clusters_flat_df=safe_load_csv(fbx_dir / "clusters_flat.csv"),
            clusters_json=safe_load_json(fbx_dir / "target_clusters.json"),
            verts_df=safe_load_csv(fbx_dir / "verts_rest.csv"),
            out_dir=plots_fbx, mesh_label="FBX",
        )
        plot_deltas(
            out_dir=plots_fbx, mesh_label="FBX",
            raw=safe_load_csv(fbx_dir / "delta_raw.csv"),
            smoothed=safe_load_csv(fbx_dir / "delta_smoothed.csv"),
        )

    # ── COMPARE ───────────────────────────────────────────────────────────────
    if h1_dir.exists() and fbx_dir.exists():
        print(f"\n▼ COMPARE HEAD1 vs FBX")
        plot_compare(h1_dir, fbx_dir, plots_root / "compare")

    print(f"\n✓ Готово. Все графики: {plots_root}/")


if __name__ == "__main__":
    main()
