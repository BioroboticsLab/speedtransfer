"""
Generate a compact set of deterministic figures for reviewers.

All plots are written to visualizations/output without opening GUI windows. The
config trims simulation duration and grid sizes so the run finishes quickly
while still showing the effects of the external driver and ablations.
"""
import matplotlib

# Use a headless backend so plt.show() inside helpers does not block.
matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from simulation import (
    OutputOptions,
    SimulationParameters,
    default_simulation_config,
    export_ablation_results,
    plot_ablation_results,
    run_simulation,
    run_single_factor_ablation_suite,
)

OUTPUT_DIR = Path(__file__).parent / "output"


def _build_config() -> object:
    cfg = default_simulation_config()
    cfg.sim = SimulationParameters(day_duration=120, sim_duration=240)
    cfg.env.bins_x = 30
    cfg.env.bins_y = 20
    cfg.output = OutputOptions(output_dir=OUTPUT_DIR)
    return cfg


def generate_baseline_figures() -> None:
    cfg = _build_config()
    rng = np.random.default_rng(111)
    run_simulation(
        density_factors=(0.5,),
        config=cfg,
        ablation=None,
        show_point_motion=False,
        save_outputs=True,
        rhythmicity_permutations=50,
        rng=rng,
    )


def generate_ablation_grid() -> None:
    cfg = _build_config()
    rng = np.random.default_rng(222)
    results = run_single_factor_ablation_suite(
        density_factor=0.5,
        config=cfg,
        rhythmicity_permutations=30,
        rng=rng,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_ablation_results(results, OUTPUT_DIR / "ablation_results.csv")

    plt.ioff()
    plot_ablation_results(results)
    plt.savefig(OUTPUT_DIR / "ablation_grid.png", dpi=150, bbox_inches="tight")
    plt.close("all")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_baseline_figures()
    generate_ablation_grid()
    print(f"Saved figures to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
