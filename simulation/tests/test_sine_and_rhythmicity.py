import math
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation import fit_sine, _permutation_rhythmicity_test


def _generate_sinusoidal_speeds(
    period: float,
    amplitude: float,
    offset: float,
    phase_deg: float,
    noise_std: float,
    length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(1, length + 1, dtype=float)
    phase_steps = (phase_deg / 360.0) * period
    omega = 2.0 * math.pi / period
    clean = offset + amplitude * np.sin(omega * (x - phase_steps))
    noisy = clean + noise_std * rng.standard_normal(length)
    clipped = np.clip(noisy, 0.0, None)
    return x, clipped


def _phase_to_degrees(phase_steps: float, period: float) -> float:
    normalized = (phase_steps % period) / period
    return normalized * 360.0


def _shortest_phase_diff(a_deg: float, b_deg: float) -> float:
    return ((a_deg - b_deg + 180.0) % 360.0) - 180.0


def test_fit_sine_recovers_parameters_with_noise():
    period = 200
    amplitude = 20.0
    offset = 15.0
    phase_deg = 120.0
    length = 400
    rng = np.random.default_rng(1234)

    for noise_std in (0.0, 0.5, 1.0):
        x, y = _generate_sinusoidal_speeds(
            period=period,
            amplitude=amplitude,
            offset=offset,
            phase_deg=phase_deg,
            noise_std=noise_std,
            length=length,
            rng=rng,
        )
        fit = fit_sine(x, y, period)

        assert math.isclose(fit.y_offset, offset, rel_tol=0.15)
        assert math.isclose(fit.y_scale, amplitude, rel_tol=0.2)

        recovered_phase_deg = _phase_to_degrees(fit.x_offset, period)
        assert abs(_shortest_phase_diff(recovered_phase_deg, phase_deg)) < 20.0


def _write_debug_log(lines: list[str], filename: str = "rhythmicity_debug.log") -> None:
    log_path = Path(__file__).with_name(filename)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        fh.write("\n".join(lines) + "\n")


def test_rhythmicity_test_distinguishes_rhythmic_from_noise():
    period = 200
    amplitude = 10.0
    offset = 8.0
    length = 400
    rng_rhythmic = np.random.default_rng(5678)
    rng_non = np.random.default_rng(9012)

    x_rhythmic, y_rhythmic = _generate_sinusoidal_speeds(
        period=period,
        amplitude=amplitude,
        offset=offset,
        phase_deg=45.0,
        noise_std=0.8,
        length=length,
        rng=rng_rhythmic,
    )

    rhythmic_ps = []
    for seed in range(3):
        rng_perm = np.random.default_rng(2000 + seed)
        rhythmic_ps.append(
            _permutation_rhythmicity_test(
                x_rhythmic,
                y_rhythmic,
                period_length=period,
                permutations=200,
                rng=rng_perm,
            )
        )
    msg_rhythmic = f"Rhythmic p-values: {rhythmic_ps}"
    print(msg_rhythmic)
    _write_debug_log([msg_rhythmic])
    assert max(rhythmic_ps) < 0.05

    x_non = np.arange(1, length + 1, dtype=float)
    non_ps = []
    for idx in range(10):
        y_non = np.clip(offset + rng_non.standard_normal(length), 0.0, None)
        rng_perm = np.random.default_rng(3000 + idx)
        non_ps.append(
            _permutation_rhythmicity_test(
                x_non,
                y_non,
                period_length=period,
                permutations=200,
                rng=rng_perm,
            )
        )
    msg_non = f"Non-rhythmic p-values: {non_ps}"
    print(msg_non)
    _write_debug_log([msg_non])
    assert np.median(non_ps) > 0.2


def generate_sine_fit_diagnostic_figure(
    period: float = 200,
    amplitude: float = 15.0,
    offset: float = 10.0,
    phase_deg: float = 60.0,
    noise_std: float = 20.0,
    length: int = 400,
    output_path: Path | None = None,
) -> Path:
    rng = np.random.default_rng(4242)
    x, y = _generate_sinusoidal_speeds(
        period=period,
        amplitude=amplitude,
        offset=offset,
        phase_deg=phase_deg,
        noise_std=noise_std,
        length=length,
        rng=rng,
    )
    fit = fit_sine(x, y, period)
    p_value = _permutation_rhythmicity_test(
        x,
        y,
        period_length=period,
        permutations=200,
        rng=np.random.default_rng(999),
    )

    recovered_amp = fit.y_scale
    recovered_phase = _phase_to_degrees(fit.x_offset, period)
    print(
        "Ground truth -> amplitude={amp:.2f}, phase={phase:.1f}°, offset={offset:.2f}, noise_std={noise:.2f}".format(
            amp=amplitude,
            phase=phase_deg,
            offset=offset,
            noise=noise_std,
        )
    )
    print(
        "Recovered fit -> amplitude={amp:.2f}, phase={phase:.1f}°, offset={offset:.2f}, p-value={p:.4f}".format(
            amp=recovered_amp,
            phase=recovered_phase,
            offset=fit.y_offset,
            p=p_value,
        )
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, ".", label="generated speeds")
    ax.plot(x, fit(x), "r", label="sine fit")
    ax.set_title(
        f"Diagnostics: amp={fit.y_scale:.2f}, phase={_phase_to_degrees(fit.x_offset, period):.1f}°"
    )
    ax.set_xlabel("time step")
    ax.set_ylabel("speed")
    ax.legend()
    ax.grid(True, alpha=0.2)

    if output_path is None:
        output_path = Path(__file__).with_name("sine_fit_diagnostic.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    dest = generate_sine_fit_diagnostic_figure()
    print(f"Saved diagnostic plot to {dest}")
