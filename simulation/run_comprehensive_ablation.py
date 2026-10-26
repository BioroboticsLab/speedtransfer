"""Run the ablation study with multi-seed averaging.

Produces:
  1. Full factorial over 4 structural components (16 combos × 5 seeds)
  2. Two additional single-factor ablations (external driver, initial positions)
  3. Combined CSV with mean ± SEM across seeds
"""
import numpy as np
import pandas as pd
from pathlib import Path
from simulation import (
    run_full_factorial_ablation,
    run_ablation_study,
    export_ablation_results,
    default_simulation_config,
    AblationConfig,
    SimulationSummary,
)

N_SEEDS = 5
N_PERMS = 200
DENSITY = 1.0
OUT_DIR = Path("visualizations")


def run_multi_seed_factorial(seeds: list[int]) -> pd.DataFrame:
    """Run 4-factor factorial ablation across multiple seeds."""
    factors = [
        "disable_walls",
        "disable_homing",
        "disable_speed_transfer",
        "homogenize_turning_noise",
    ]
    all_rows = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Factorial — seed {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*60}")
        config = default_simulation_config()
        config.sim.sim_duration = 500
        config.sim.day_duration = 100

        results = run_full_factorial_ablation(
            density_factor=DENSITY,
            factors=factors,
            config=config,
            rhythmicity_permutations=N_PERMS,
            rng=np.random.default_rng(seed),
        )

        for label, summary in results.items():
            all_rows.append({
                "label": label,
                "seed": seed,
                "source": "factorial",
                "amplitude_group1": summary.amplitude_group1,
                "amplitude_group2": summary.amplitude_group2,
                "phase_group1": summary.phase_group1,
                "phase_group2": summary.phase_group2,
                "phase_shift_g2_minus_g1": summary.phase_shift_g2_minus_g1,
                "rhythmicity_p_value_group1": summary.rhythmicity_p_value_group1,
                "rhythmicity_p_value_group2": summary.rhythmicity_p_value_group2,
            })

    return pd.DataFrame(all_rows)


def run_multi_seed_single_factor(seeds: list[int]) -> pd.DataFrame:
    """Run additional single-factor ablations (external driver, init positions)."""
    ablations = {
        "baseline": AblationConfig(),
        "no_external_driver": AblationConfig(disable_external_driver=True),
        "homogenize_init": AblationConfig(homogenize_initial_positions=True),
    }
    all_rows = []

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"Single-factor — seed {seed} ({i+1}/{len(seeds)})")
        print(f"{'='*60}")
        config = default_simulation_config()
        config.sim.sim_duration = 500
        config.sim.day_duration = 100

        results = run_ablation_study(
            density_factor=DENSITY,
            ablations=ablations,
            config=config,
            rhythmicity_permutations=N_PERMS,
            rng=np.random.default_rng(seed),
        )

        for label, summary in results.items():
            all_rows.append({
                "label": label,
                "seed": seed,
                "source": "single_factor",
                "amplitude_group1": summary.amplitude_group1,
                "amplitude_group2": summary.amplitude_group2,
                "phase_group1": summary.phase_group1,
                "phase_group2": summary.phase_group2,
                "phase_shift_g2_minus_g1": summary.phase_shift_g2_minus_g1,
                "rhythmicity_p_value_group1": summary.rhythmicity_p_value_group1,
                "rhythmicity_p_value_group2": summary.rhythmicity_p_value_group2,
            })

    return pd.DataFrame(all_rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean ± SEM across seeds for each condition."""
    metrics = [
        "amplitude_group1", "amplitude_group2",
        "phase_shift_g2_minus_g1",
        "rhythmicity_p_value_group1", "rhythmicity_p_value_group2",
    ]
    rows = []
    for label in df["label"].unique():
        sub = df[df["label"] == label]
        row = {"label": label, "n_seeds": len(sub), "source": sub["source"].iloc[0]}
        for m in metrics:
            row[f"{m}_mean"] = sub[m].mean()
            row[f"{m}_sem"] = sub[m].sem()
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    seeds = list(range(42, 42 + N_SEEDS))
    print(f"Running ablation study: {N_SEEDS} seeds, {N_PERMS} permutations, density={DENSITY}")

    # 1. Full factorial (16 combos × 5 seeds = 80 runs)
    df_factorial = run_multi_seed_factorial(seeds)
    df_factorial.to_csv(OUT_DIR / "ablation_factorial_raw.csv", index=False)
    print(f"\nSaved raw factorial: {len(df_factorial)} rows")

    # 2. Single-factor extras (3 conditions × 5 seeds = 15 runs)
    df_single = run_multi_seed_single_factor(seeds)
    df_single.to_csv(OUT_DIR / "ablation_single_factor_raw.csv", index=False)
    print(f"Saved raw single-factor: {len(df_single)} rows")

    # 3. Combine and summarize
    # Remove duplicate baselines from single-factor (factorial already has one)
    df_single_no_bl = df_single[df_single["label"] != "baseline"]
    df_all = pd.concat([df_factorial, df_single_no_bl], ignore_index=True)
    df_all.to_csv(OUT_DIR / "ablation_all_raw.csv", index=False)

    df_summary = summarize(df_all)
    df_summary.to_csv(OUT_DIR / "ablation_summary.csv", index=False)
    print(f"\nSummary ({len(df_summary)} conditions):")
    print(df_summary[["label", "amplitude_group2_mean", "amplitude_group2_sem",
                       "rhythmicity_p_value_group2_mean"]].to_string(index=False))
    print("\nDone.")
