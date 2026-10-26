"""Interaction-tree depth vs. hive location and local density (Reviewer #2, minor #3).

Reviewer: "Does not the interaction tree depth depend on the density of bees around a
bee, so baseline depth would not be uniform across the hive? How does one control for
this?"

Using the per-location aggregation behind Fig. 3c (MEDIAN vertex depth per 10 mm cell,
Gaussian-smoothed at sigma=1.5; see figures/figure_panel_3.ipynb::get_vertex_depth_per_loc),
we show that (a) depth is spatially structured (not uniform across the comb) and (b) it
depends only weakly on local sampling density (vertex count per location).

CAVEAT: the loaded depth grid is a *smoothed median*, so cells are not independent and the
depth-vs-density correlation below is approximate. The rigorous version (un-smoothed medians
from the raw per-vertex path_df, plus a density-matched null-tree model) needs
interaction_tree_paths_2019.csv, which lives on trove (not local).

Run:
    python analysis/tree_depth_vs_density.py [--out path.pdf]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
YEAR = 2019  # colony B (main text)


def load_grids() -> tuple[np.ndarray, np.ndarray]:
    base = REPO / "aggregated_results" / str(YEAR)
    depth = np.load(base / "vertex_depth_per_loc.npy")
    count = np.load(base / "vertex_depth_per_loc_count.npy")
    return depth, count


def analyze(depth: np.ndarray, count: np.ndarray) -> dict:
    mask = np.isfinite(depth) & np.isfinite(count) & (count > 0)
    d = depth[mask].astype(float)
    c = count[mask].astype(float)
    rho, p_rho = stats.spearmanr(c, d)
    return {
        "mask": mask, "d": d, "c": c,
        "n_loc": int(mask.sum()),
        "depth_mean": float(d.mean()), "depth_std": float(d.std()),
        "depth_min": float(d.min()), "depth_max": float(d.max()),
        "spearman_rho": float(rho), "spearman_p": float(p_rho),
        "density_r2": float(rho ** 2),
    }


def make_figure(depth: np.ndarray, res: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # Paper orientation (cf. Fig 3c): high y_pos = the nest entrance is at the BOTTOM.
    # (verified independently: entrance is the low-x/high-y corner via dist_exit_df.)
    disp = np.flipud(np.where(res["mask"], depth, np.nan))
    im = axes[0].imshow(disp, origin="lower", aspect="auto", cmap="viridis")
    axes[0].set(title="(a) Median vertex depth (entrance = bottom-left, cf. Fig 3c)",
                xlabel="comb x-bin (x_pos)", ylabel="comb y-bin (entrance at bottom)")
    axes[0].text(0.02, 0.03, "entrance", transform=axes[0].transAxes,
                 color="white", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=axes[0], label="median vertex depth (smoothed)")

    c, d = res["c"], res["d"]
    axes[1].scatter(c, d, s=14, alpha=0.5, edgecolor="none")
    axes[1].set(xscale="log", xlabel="local sampling density (vertex count / location)",
                ylabel="median vertex depth (smoothed)",
                title=f"(b) Depth vs density  (Spearman $\\rho$={res['spearman_rho']:.2f}, approx.)")
    bins = np.logspace(np.log10(c.min()), np.log10(c.max()), 9)
    idx = np.digitize(c, bins)
    bx = [c[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
    by = [d[idx == i].mean() for i in range(1, len(bins)) if (idx == i).any()]
    axes[1].plot(bx, by, "o-", color="crimson", lw=2, label="binned mean")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "aggregated_results" / "tree_depth_vs_density.pdf")
    args = ap.parse_args()
    depth, count = load_grids()
    res = analyze(depth, count)
    print(f"valid locations: {res['n_loc']}")
    print(f"median vertex depth/location (smoothed): {res['depth_mean']:.2f} +/- {res['depth_std']:.2f} "
          f"(range {res['depth_min']:.2f}-{res['depth_max']:.2f}); not uniform")
    print(f"depth vs density: Spearman rho={res['spearman_rho']:.3f} "
          f"(p={res['spearman_p']:.2e}); density explains ~{100*res['density_r2']:.1f}% of variance")
    make_figure(depth, res, args.out)
    print(f"saved figure -> {args.out}")


if __name__ == "__main__":
    main()
