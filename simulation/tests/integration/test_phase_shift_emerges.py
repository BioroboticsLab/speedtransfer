"""Integration test: speed transfer creates a phase-shifted rhythm in Group 2.

This is the paper's central claim.  We verify it by running simulations at
production-like settings and asserting that:
1. Group 2 develops significant rhythmicity under baseline.
2. Group 2's rhythmicity vanishes without speed transfer.
3. Group 2's phase is shifted relative to Group 1 under baseline.
"""
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


def _config():
    cfg = default_simulation_config()
    # 3 full cycles, enough for reliable sine fitting
    cfg.sim = SimulationParameters(day_duration=200, sim_duration=600)
    cfg.output = OutputOptions(output_dir=Path("visualizations/output"))
    return cfg


def test_speed_transfer_creates_phase_shifted_rhythm():
    cfg = _config()
    rng = np.random.default_rng(4242)
    results = run_ablation_study(
        density_factor=0.5,
        ablations={
            "baseline": AblationConfig(),
            "no_transfer": AblationConfig(disable_speed_transfer=True),
        },
        config=cfg,
        rhythmicity_permutations=50,
        rng=rng,
    )

    baseline = results["baseline"]
    no_transfer = results["no_transfer"]

    # 1. Group 2 has measurable amplitude with speed transfer
    assert baseline.amplitude_group2 > 0.10, (
        f"Baseline Group 2 amplitude too low: {baseline.amplitude_group2:.4f}"
    )

    # 2. Group 2 amplitude should be negligible without speed transfer
    assert no_transfer.amplitude_group2 < 0.01, (
        f"No-transfer Group 2 amplitude unexpectedly high: {no_transfer.amplitude_group2:.4f}"
    )

    # 3. Group 1 maintains its rhythm in both conditions (external driver)
    assert baseline.amplitude_group1 > 0.5
    assert no_transfer.amplitude_group1 > 0.5

    # 4. Phase shift: Group 2 is shifted relative to Group 1 under baseline
    #    With proper settings, phase shifts are typically 5–15°
    phase_shift = abs(baseline.phase_shift_g2_minus_g1)
    assert phase_shift > 1.0, (
        f"Phase shift too small ({phase_shift:.2f}°) — expected a measurable lag"
    )
    assert phase_shift < 30.0, (
        f"Phase shift too large ({phase_shift:.2f}°) — may indicate a fitting issue"
    )
