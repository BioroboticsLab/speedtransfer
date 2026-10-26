"""Single-parameter sensitivity sweeps for supplementary material.

Three sweeps (one parameter varied, others at baseline):
  1. Density:      [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
  2. Day duration: [50, 75, 100, 150, 200, 300, 400, 600, 800]
  3. Decay rate:   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

Baseline: density=1.0, day_duration=100, decay=0.8
Produces line plots with error bars (amplitude + phase shift).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
from simulation import (
    run_ablation_study, default_simulation_config, AblationConfig
)

N_SEEDS = 5
N_PERMS = 200
OUT_DIR = Path("visualizations")

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def run_sweep(param_name, param_values, config_fn, seeds):
    """Run a sweep over one parameter."""
    rows = []
    total = len(param_values) * len(seeds)
    done = 0
    for val in param_values:
        for seed in seeds:
            done += 1
            print(f"[{done}/{total}] {param_name}={val}, seed={seed}")
            config = default_simulation_config()
            config_fn(config, val)  # apply the parameter

            results = run_ablation_study(
                density_factor=config._density_factor
                if hasattr(config, "_density_factor")
                else 1.0,
                ablations={"run": AblationConfig()},
                config=config,
                rhythmicity_permutations=N_PERMS,
                rng=np.random.default_rng(seed),
            )
            s = results["run"]
            rows.append({
                "param": param_name,
                "value": val,
                "seed": seed,
                "amp_g1": s.amplitude_group1,
                "amp_g2": s.amplitude_group2,
                "phase_shift": s.phase_shift_g2_minus_g1,
                "p_g2": s.rhythmicity_p_value_group2,
            })
    return pd.DataFrame(rows)


def sweep_density(seeds):
    """Sweep density factor."""
    values = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    rows = []
    total = len(values) * len(seeds)
    done = 0
    for density in values:
        for seed in seeds:
            done += 1
            print(f"[{done}/{total}] density={density}, seed={seed}")
            config = default_simulation_config()
            config.sim.sim_duration = 500
            config.sim.day_duration = 100

            results = run_ablation_study(
                density_factor=density,
                ablations={"run": AblationConfig()},
                config=config,
                rhythmicity_permutations=N_PERMS,
                rng=np.random.default_rng(seed),
            )
            s = results["run"]
            rows.append({
                "param": "density", "value": density, "seed": seed,
                "amp_g1": s.amplitude_group1, "amp_g2": s.amplitude_group2,
                "phase_shift": s.phase_shift_g2_minus_g1,
                "p_g2": s.rhythmicity_p_value_group2,
            })
    return pd.DataFrame(rows)


def sweep_day_duration(seeds):
    """Sweep day duration."""
    values = [50, 75, 100, 150, 200, 300, 400, 600, 800]
    rows = []
    total = len(values) * len(seeds)
    done = 0
    for dur in values:
        for seed in seeds:
            done += 1
            print(f"[{done}/{total}] day_duration={dur}, seed={seed}")
            config = default_simulation_config()
            config.sim.sim_duration = dur * 5  # 5 full days
            config.sim.day_duration = dur

            results = run_ablation_study(
                density_factor=1.0,
                ablations={"run": AblationConfig()},
                config=config,
                rhythmicity_permutations=N_PERMS,
                rng=np.random.default_rng(seed),
            )
            s = results["run"]
            rows.append({
                "param": "day_duration", "value": dur, "seed": seed,
                "amp_g1": s.amplitude_group1, "amp_g2": s.amplitude_group2,
                "phase_shift": s.phase_shift_g2_minus_g1,
                "p_g2": s.rhythmicity_p_value_group2,
            })
    return pd.DataFrame(rows)


def sweep_decay(seeds):
    """Sweep decay rate."""
    values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    rows = []
    total = len(values) * len(seeds)
    done = 0
    for decay in values:
        for seed in seeds:
            done += 1
            print(f"[{done}/{total}] decay={decay}, seed={seed}")
            config = default_simulation_config()
            config.sim.sim_duration = 500
            config.sim.day_duration = 100
            config.env.speed_transfer_decay = decay

            results = run_ablation_study(
                density_factor=1.0,
                ablations={"run": AblationConfig()},
                config=config,
                rhythmicity_permutations=N_PERMS,
                rng=np.random.default_rng(seed),
            )
            s = results["run"]
            rows.append({
                "param": "decay", "value": decay, "seed": seed,
                "amp_g1": s.amplitude_group1, "amp_g2": s.amplitude_group2,
                "phase_shift": s.phase_shift_g2_minus_g1,
                "p_g2": s.rhythmicity_p_value_group2,
            })
    return pd.DataFrame(rows)


def plot_sensitivity(df, param_col, xlabel, filename, baseline_val=None):
    """Two-panel line plot: amplitude (left) and phase shift (right)."""
    grouped = df.groupby("value").agg(
        amp_g2_mean=("amp_g2", "mean"),
        amp_g2_sem=("amp_g2", "sem"),
        phase_mean=("phase_shift", "mean"),
        phase_sem=("phase_shift", "sem"),
        p_mean=("p_g2", "mean"),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    # Left panel: amplitude
    ax1.errorbar(grouped["value"], grouped["amp_g2_mean"],
                 yerr=grouped["amp_g2_sem"], fmt="o-", color="#2196F3",
                 capsize=3, markersize=5, linewidth=1.5)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Group 2 amplitude ($A_2$)")
    ax1.set_ylim(bottom=0)
    if baseline_val is not None:
        ax1.axvline(baseline_val, color="gray", linestyle="--", alpha=0.5,
                    label=f"Baseline ({baseline_val})")
        ax1.legend(fontsize=8)

    # Right panel: phase shift
    # Only show phase shift where p < 0.05
    sig_mask = grouped["p_mean"] < 0.05
    nonsig_mask = ~sig_mask
    if sig_mask.any():
        ax2.errorbar(grouped.loc[sig_mask, "value"],
                     grouped.loc[sig_mask, "phase_mean"],
                     yerr=grouped.loc[sig_mask, "phase_sem"],
                     fmt="o-", color="#FF5722", capsize=3, markersize=5,
                     linewidth=1.5, label="Significant ($p<0.05$)")
    if nonsig_mask.any():
        ax2.scatter(grouped.loc[nonsig_mask, "value"],
                    grouped.loc[nonsig_mask, "phase_mean"],
                    marker="x", color="gray", s=40, zorder=5,
                    label="Not significant")
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Phase shift $\\Delta\\phi$ (°)")
    if baseline_val is not None:
        ax2.axvline(baseline_val, color="gray", linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{filename}.pdf")
    fig.savefig(OUT_DIR / f"{filename}.png")
    print(f"Saved: {OUT_DIR}/{filename}.pdf")
    plt.close(fig)


if __name__ == "__main__":
    plt.switch_backend("Agg")
    seeds = list(range(42, 42 + N_SEEDS))

    # Sweep 1: density
    print("=" * 60)
    print("SWEEP 1: Density")
    print("=" * 60)
    df_density = sweep_density(seeds)
    df_density.to_csv(OUT_DIR / "sensitivity_density.csv", index=False)
    plot_sensitivity(df_density, "value", "Density factor ($d$)",
                     "sensitivity_density", baseline_val=1.0)

    # Sweep 2: day duration
    print("\n" + "=" * 60)
    print("SWEEP 2: Day duration")
    print("=" * 60)
    df_duration = sweep_day_duration(seeds)
    df_duration.to_csv(OUT_DIR / "sensitivity_day_duration.csv", index=False)
    plot_sensitivity(df_duration, "value",
                     "Day duration $T_{\\mathrm{day}}$ (steps)",
                     "sensitivity_day_duration", baseline_val=100)

    # Sweep 3: decay
    print("\n" + "=" * 60)
    print("SWEEP 3: Decay rate")
    print("=" * 60)
    df_decay = sweep_decay(seeds)
    df_decay.to_csv(OUT_DIR / "sensitivity_decay.csv", index=False)
    plot_sensitivity(df_decay, "value", "Decay rate ($\\gamma$)",
                     "sensitivity_decay", baseline_val=0.8)

    print("\nAll sweeps done.")
