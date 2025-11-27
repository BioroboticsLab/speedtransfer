"""
CLI helper to run the density × speed-transfer-decay sweep headlessly and cache results.

Matches the notebook logic but is server-friendly:
- Runs multiple seeds per parameter combo and aggregates mean/CI.
- Caches raw per-run results with metadata (day duration, sim duration, permutations, num_runs).
- Reuses cache when it fully covers the requested grid; otherwise recomputes and overwrites.
- Optionally saves plots (PNG) for quick inspection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

# Headless backend so this works on servers without a display.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from simulation import (  # noqa: E402
    AblationConfig,
    OutputOptions,
    SimulationParameters,
    default_simulation_config,
    run_ablation_study,
)


def _float_list(arg: str) -> list[float]:
    return [float(x) for x in arg.split(",") if x.strip()]


def _build_config(day_duration: int, sim_duration: int) -> object:
    cfg = default_simulation_config()
    cfg.sim = SimulationParameters(day_duration=day_duration, sim_duration=sim_duration)
    cfg.output = OutputOptions(output_dir=Path("visualizations/output"))
    return cfg


def _combo_grid(densities: Iterable[float], decays: Iterable[float]) -> set[tuple[float, float]]:
    return {(float(d), float(dec)) for d in densities for dec in decays}


def _load_cache(
    cache_file: Path,
    densities: list[float],
    decays: list[float],
    meta: dict,
) -> pd.DataFrame | None:
    if not cache_file.exists():
        return None

    required_cols = {
        "density_factor",
        "speed_transfer_decay",
        "day_duration",
        "sim_duration",
        "rhythmicity_permutations",
        "num_runs",
        "run_id",
    }
    cached = pd.read_csv(cache_file)
    if not required_cols.issubset(cached.columns):
        print(f"Cached sweep at {cache_file} missing metadata columns; recomputing and overwriting.")
        return None

    filtered = cached[
        cached["density_factor"].isin(densities)
        & cached["speed_transfer_decay"].isin(decays)
        & (cached["day_duration"] == meta["day_duration"])
        & (cached["sim_duration"] == meta["sim_duration"])
        & (cached["rhythmicity_permutations"] == meta["rhythmicity_permutations"])
        & (cached["num_runs"] == meta["num_runs"])
    ].copy()

    expected = _combo_grid(densities, decays)
    counts = filtered.groupby(["density_factor", "speed_transfer_decay"]).run_id.nunique()
    complete = counts.reindex(pd.MultiIndex.from_tuples(sorted(expected), names=["density_factor", "speed_transfer_decay"]), fill_value=0).ge(
        meta["num_runs"]
    )
    if complete.all():
        print(f"Loaded cached sweep results from {cache_file}. Delete this file to force recompute after model changes.")
        return filtered

    print(f"Cached sweep at {cache_file} is incomplete for this config; recomputing and overwriting.")
    return None


def _run_sweep(
    densities: list[float],
    decays: list[float],
    num_runs: int,
    rhythmicity_permutations: int,
    day_duration: int,
    sim_duration: int,
    seed: int,
    cache_file: Path,
    force_recompute: bool,
) -> pd.DataFrame:
    cfg = _build_config(day_duration, sim_duration)
    meta = {
        "day_duration": day_duration,
        "sim_duration": sim_duration,
        "rhythmicity_permutations": rhythmicity_permutations,
        "num_runs": num_runs,
    }

    cache_file.parent.mkdir(parents=True, exist_ok=True)

    cached = None if force_recompute else _load_cache(cache_file, densities, decays, meta)
    if cached is not None:
        return cached

    records = []
    seed_gen = np.random.default_rng(seed)
    for density in densities:
        for decay in decays:
            seeds = seed_gen.integers(0, 1_000_000_000, size=num_runs)
            for run_id, run_seed in enumerate(seeds):
                label = f"d{density:.2f}_decay{decay:.2f}_r{run_id}"
                ablations = {label: AblationConfig(speed_transfer_decay=decay)}
                rng = np.random.default_rng(int(run_seed))
                results = run_ablation_study(
                    density_factor=density,
                    ablations=ablations,
                    config=cfg,
                    rhythmicity_permutations=rhythmicity_permutations,
                    rng=rng,
                )
                summary = results[label]
                records.append(
                    {
                        "label": label,
                        "density_factor": density,
                        "agents_per_group": 500 * density,
                        "speed_transfer_decay": decay,
                        "amplitude_group1": summary.amplitude_group1,
                        "amplitude_group2": summary.amplitude_group2,
                        "phase_group1": summary.phase_group1,
                        "phase_group2": summary.phase_group2,
                        "phase_shift": summary.phase_shift_g2_minus_g1,
                        "p_value_group1": summary.rhythmicity_p_value_group1,
                        "p_value_group2": summary.rhythmicity_p_value_group2,
                        "day_duration": meta["day_duration"],
                        "sim_duration": meta["sim_duration"],
                        "rhythmicity_permutations": meta["rhythmicity_permutations"],
                        "num_runs": meta["num_runs"],
                        "run_id": run_id,
                        "seed": int(run_seed),
                    }
                )

    df = pd.DataFrame(records)
    df.sort_values(["density_factor", "speed_transfer_decay", "run_id"], inplace=True)
    df.to_csv(cache_file, index=False)
    print(f"Saved sweep results to {cache_file} for reuse. Delete this file to recompute.")
    return df


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    agg_df = (
        df.groupby(["density_factor", "speed_transfer_decay"])
        .agg(
            agents_per_group=("agents_per_group", "first"),
            amplitude_group2_mean=("amplitude_group2", "mean"),
            amplitude_group2_std=("amplitude_group2", "std"),
            amplitude_group2_count=("amplitude_group2", "count"),
            amplitude_group1_mean=("amplitude_group1", "mean"),
            amplitude_group1_std=("amplitude_group1", "std"),
            phase_shift_mean=("phase_shift", "mean"),
            phase_shift_std=("phase_shift", "std"),
        )
        .reset_index()
    )
    agg_df["amplitude_group2_sem"] = agg_df["amplitude_group2_std"] / np.sqrt(agg_df["amplitude_group2_count"].clip(lower=1))
    agg_df["amplitude_group2_ci95"] = 1.96 * agg_df["amplitude_group2_sem"]
    return agg_df


def _save_plots(agg_df: pd.DataFrame, density_factors: list[float], decay_factors: list[float], output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    pivot_amp = agg_df.pivot(index="density_factor", columns="speed_transfer_decay", values="amplitude_group2_mean").sort_index()
    pivot_phase = agg_df.pivot(index="density_factor", columns="speed_transfer_decay", values="phase_shift_mean").sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    im0 = axes[0].imshow(pivot_amp.values, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_xticks(range(len(pivot_amp.columns)))
    axes[0].set_xticklabels([f"{c:.2f}" for c in pivot_amp.columns])
    axes[0].set_yticks(range(len(pivot_amp.index)))
    axes[0].set_yticklabels([f"{r:.2f}" for r in pivot_amp.index])
    axes[0].set_xlabel("speed_transfer_decay")
    axes[0].set_ylabel("density_factor")
    axes[0].set_title("Amplitude (Group 2, mean across runs)")
    fig.colorbar(im0, ax=axes[0], label="speed amplitude")

    im1 = axes[1].imshow(pivot_phase.values, origin="lower", aspect="auto", cmap="magma")
    axes[1].set_xticks(range(len(pivot_phase.columns)))
    axes[1].set_xticklabels([f"{c:.2f}" for c in pivot_phase.columns])
    axes[1].set_yticks(range(len(pivot_phase.index)))
    axes[1].set_yticklabels([f"{r:.2f}" for r in pivot_phase.index])
    axes[1].set_xlabel("speed_transfer_decay")
    axes[1].set_title("Phase shift (Group2 - Group1, mean degrees)")
    fig.colorbar(im1, ax=axes[1], label="degrees")

    fig.savefig(output_prefix.with_name(output_prefix.name + "_heatmaps.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    for density in density_factors:
        subset = agg_df[agg_df["density_factor"] == density]
        subset = subset.sort_values("speed_transfer_decay")
        ax.errorbar(
            subset["speed_transfer_decay"],
            subset["amplitude_group2_mean"],
            yerr=subset["amplitude_group2_ci95"],
            marker="o",
            capsize=4,
            label=f"density {density:.2f} (agents≈{int(500*density)})",
        )
    ax.set_xlabel("speed_transfer_decay")
    ax.set_ylabel("Amplitude (Group 2)")
    ax.set_title("Non-rhythmic group entrainment vs decay (mean ±95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_prefix.with_name(output_prefix.name + "_lines.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run density × speed-transfer-decay sweep with caching.")
    parser.add_argument("--density-factors", type=_float_list, default="0.20,0.40,0.60,0.80,1.00,1.20")
    parser.add_argument("--decays", type=_float_list, default="0.0,0.4,0.8,0.95")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--rhythmicity-permutations", type=int, default=100)
    parser.add_argument("--day-duration", type=int, default=300)
    parser.add_argument("--sim-duration", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42, help="Seed for generating per-run seeds.")
    parser.add_argument("--cache-file", type=Path, default=Path("visualizations/sweep_cache/density_decay_sweep.csv"))
    parser.add_argument("--plot-prefix", type=Path, default=None, help="If set, save plots with this prefix (e.g., visualizations/output/density_decay_sweep)")
    parser.add_argument("--force-recompute", action="store_true", help="Ignore any existing cache and recompute.")
    args = parser.parse_args(argv)

    densities = args.density_factors if isinstance(args.density_factors, list) else _float_list(args.density_factors)
    decays = args.decays if isinstance(args.decays, list) else _float_list(args.decays)

    # Run or load sweep
    df = _run_sweep(
        densities=densities,
        decays=decays,
        num_runs=args.num_runs,
        rhythmicity_permutations=args.rhythmicity_permutations,
        day_duration=args.day_duration,
        sim_duration=args.sim_duration,
        seed=args.seed,
        cache_file=args.cache_file,
        force_recompute=args.force_recompute,
    )

    agg_df = _aggregate(df)

    # Always write aggregated CSV alongside cache for convenience
    agg_path = args.cache_file.with_name(args.cache_file.stem + "_agg.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Wrote aggregated sweep to {agg_path}")

    if args.plot_prefix:
        _save_plots(agg_df, densities, decays, args.plot_prefix)
        print(f"Saved plots to {args.plot_prefix}_heatmaps.png and {args.plot_prefix}_lines.png")
    else:
        print("Plots skipped (no --plot-prefix provided).")


if __name__ == "__main__":
    main()
