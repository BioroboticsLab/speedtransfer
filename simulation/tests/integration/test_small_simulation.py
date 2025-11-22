import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation import (
    AblationConfig,
    SimulationParameters,
    OutputOptions,
    default_simulation_config,
    run_ablation_study,
)


def _make_small_config() -> tuple:
    """
    Lightweight configuration for fast integration checks.

    Keeps grid sizes small and durations short while retaining the external
    driver so amplitude differences are still measurable.
    """
    cfg = default_simulation_config()
    cfg.env.bins_x = 18
    cfg.env.bins_y = 12
    cfg.sim = SimulationParameters(day_duration=80, sim_duration=160)
    cfg.output = OutputOptions(save_mat=False, output_dir=Path("visualizations/output"))
    return cfg


def test_external_driver_increases_group1_amplitude():
    cfg = _make_small_config()
    rng = np.random.default_rng(2024)
    results = run_ablation_study(
        density_factor=0.1,
        ablations={
            "baseline": AblationConfig(),
            "no_driver": AblationConfig(disable_external_driver=True),
        },
        config=cfg,
        rhythmicity_permutations=30,
        rng=rng,
    )

    baseline = results["baseline"]
    no_driver = results["no_driver"]

    assert np.isfinite(baseline.amplitude_group1)
    assert baseline.amplitude_group1 > no_driver.amplitude_group1 + 0.1
    assert np.isfinite(baseline.phase_group1)
    assert np.isfinite(baseline.phase_group2)
