"""Parameter sweeps to find settings that produce larger phase shifts.

Sweep 1: density × speed_transfer_decay
Sweep 2: density × day_duration

Produces heatmaps of phase shift and amplitude for each sweep.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from simulation import (
    run_ablation_study, default_simulation_config, AblationConfig
)

N_SEEDS = 3  # 3 seeds for speed; enough to see trends
N_PERMS = 100
OUT_DIR = Path("visualizations")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def sweep_density_decay():
    """Sweep density × speed_transfer_decay."""
    densities = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    decays = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []

    total = len(densities) * len(decays) * N_SEEDS
    done = 0
    for density in densities:
        for decay in decays:
            for seed_i in range(N_SEEDS):
                seed = 42 + seed_i
                done += 1
                print(f"[{done}/{total}] density={density}, decay={decay}, seed={seed}")
                config = default_simulation_config()
                config.sim.sim_duration = 1200
                config.sim.day_duration = 400
                config.env.speed_transfer_decay = decay

                results = run_ablation_study(
                    density_factor=density,
                    ablations={"run": AblationConfig()},
                    config=config,
                    rhythmicity_permutations=N_PERMS,
                    rng=np.random.default_rng(seed),
                )
                s = results["run"]
                rows.append({
                    "density": density, "decay": decay, "seed": seed,
                    "amp_g1": s.amplitude_group1, "amp_g2": s.amplitude_group2,
                    "phase_shift": s.phase_shift_g2_minus_g1,
                    "p_g2": s.rhythmicity_p_value_group2,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sweep_density_decay_raw.csv", index=False)
    return df


def sweep_density_duration():
    """Sweep density × day_duration."""
    densities = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    durations = [100, 200, 300, 400, 600, 800]
    rows = []

    total = len(densities) * len(durations) * N_SEEDS
    done = 0
    for density in densities:
        for dur in durations:
            for seed_i in range(N_SEEDS):
                seed = 42 + seed_i
                done += 1
                print(f"[{done}/{total}] density={density}, day_dur={dur}, seed={seed}")
                config = default_simulation_config()
                config.sim.sim_duration = dur * 3  # Always 3 full cycles
                config.sim.day_duration = dur

                results = run_ablation_study(
                    density_factor=density,
                    ablations={"run": AblationConfig()},
                    config=config,
                    rhythmicity_permutations=N_PERMS,
                    rng=np.random.default_rng(seed),
                )
                s = results["run"]
                rows.append({
                    "density": density, "day_duration": dur, "seed": seed,
                    "amp_g1": s.amplitude_group1, "amp_g2": s.amplitude_group2,
                    "phase_shift": s.phase_shift_g2_minus_g1,
                    "p_g2": s.rhythmicity_p_value_group2,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "sweep_density_duration_raw.csv", index=False)
    return df


def plot_heatmap(df, x_col, y_col, val_col, title, filename, xlabel, ylabel,
                 cmap="RdYlBu_r", vmin=None, vmax=None, center=None):
    """Plot a heatmap from sweep results."""
    pivot = df.groupby([y_col, x_col])[val_col].mean().reset_index()
    matrix = pivot.pivot(index=y_col, columns=x_col, values=val_col)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(matrix.values, aspect="auto", cmap=cmap,
                   vmin=vmin, vmax=vmax, origin="lower")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels([f"{v:.2g}" for v in matrix.columns])
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels([f"{v:.2g}" for v in matrix.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Annotate cells
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = matrix.values[i, j]
            txt = f"{val:.1f}" if abs(val) >= 1 else f"{val:.2f}"
            color = "white" if abs(val) > (vmax or matrix.values.max()) * 0.7 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    fig.savefig(OUT_DIR / f"{filename}.png")
    print(f"Saved: {OUT_DIR}/{filename}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plt.switch_backend("Agg")

    # Sweep 1: density × decay
    print("=" * 60)
    print("SWEEP 1: density × speed_transfer_decay")
    print("=" * 60)
    df1 = sweep_density_decay()

    plot_heatmap(df1, "decay", "density", "phase_shift",
                 "Phase shift (°): density × decay", "sweep_density_decay_phase",
                 "Speed transfer decay (γ)", "Density factor",
                 cmap="RdYlBu_r", vmin=-10, vmax=60)
    plot_heatmap(df1, "decay", "density", "amp_g2",
                 "Group 2 amplitude: density × decay", "sweep_density_decay_amp",
                 "Speed transfer decay (γ)", "Density factor",
                 cmap="YlOrRd", vmin=0, vmax=0.4)

    # Sweep 2: density × day_duration
    print("\n" + "=" * 60)
    print("SWEEP 2: density × day_duration")
    print("=" * 60)
    df2 = sweep_density_duration()

    plot_heatmap(df2, "day_duration", "density", "phase_shift",
                 "Phase shift (°): density × day duration", "sweep_density_duration_phase",
                 "Day duration (steps)", "Density factor",
                 cmap="RdYlBu_r", vmin=-10, vmax=60)
    plot_heatmap(df2, "day_duration", "density", "amp_g2",
                 "Group 2 amplitude: density × day duration", "sweep_density_duration_amp",
                 "Day duration (steps)", "Density factor",
                 cmap="YlOrRd", vmin=0, vmax=0.4)

    print("\nDone.")
