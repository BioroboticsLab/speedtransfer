"""
Python translation of the accompanying MATLAB scripts for the SpeedTransfer project.

This module consolidates functionality from the following MATLAB sources:
    - draw_positions_from_gaussian.m
    - draw_velocities.m
    - fit_sine.m
    - mgd.m
    - run_simulation.m
    - sketch_agents_2D.m
    - sketch_agents_3D.m

It mirrors MATLAB behaviour while remaining idiomatic Python. All heavy numerical
work is delegated to NumPy/SciPy, and plotting is handled via Matplotlib just like
MATLAB figure windows. The entry point `run_simulation` reproduces the main
simulation loop, while the helper functions expose the standalone utilities.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
import csv
import sys
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.io import savemat as _savemat
except ImportError:  # pragma: no cover - SciPy may be unavailable
    _savemat = None

__all__ = [
    "AblationConfig",
    "EnvironmentParameters",
    "GroupParameters",
    "SimulationConfig",
    "SimulationParameters",
    "SimulationSummary",
    "export_ablation_results",
    "plot_ablation_results",
    "draw_positions_from_gaussian",
    "draw_velocities",
    "fit_sine",
    "mgd",
    "run_simulation",
    "run_ablation_study",
    "sketch_agents_2D",
    "sketch_agents_3D",
]


@dataclass
class GroupParameters:
    mu: np.ndarray
    cov: np.ndarray
    speed_fwd_mu: float
    speed_fwd_std: float
    min_fwd_speed: float
    speed_trn_mu: float
    speed_trn_std: float


@dataclass
class EnvironmentParameters:
    comb_size_x: float = 354.0
    comb_size_y: float = 205.0
    bins_x: int = 45
    bins_y: int = 30
    speed_scale: float = 50.0
    homing_pull: float = 0.1
    speed_transfer_scale: float = 1.0
    interaction_threshold_factor: float = 0.2

    @property
    def hive_bounds(self) -> np.ndarray:
        return np.array(
            [
                -self.comb_size_x / 2.0,
                self.comb_size_x / 2.0,
                -self.comb_size_y / 2.0,
                self.comb_size_y / 2.0,
            ],
            dtype=float,
        )


@dataclass
class SimulationParameters:
    day_duration: int = 400
    sim_duration: int = 800


@dataclass
class OutputOptions:
    save_mat: bool = True
    output_dir: Path = Path(".")


@dataclass
class AblationConfig:
    disable_walls: bool = False
    disable_homing: bool = False
    disable_speed_transfer: bool = False
    disable_external_driver: bool = False
    homogenize_turning_noise: bool = False
    homogenize_initial_positions: bool = False


@dataclass
class SimulationConfig:
    env: EnvironmentParameters
    sim: SimulationParameters
    group1: GroupParameters
    group2: GroupParameters
    output: OutputOptions


@dataclass
class SimulationSummary:
    density_factor: float
    amplitude_group1: float
    amplitude_group2: float
    phase_shift_g2_minus_g1: float
    phase_group1: float
    phase_group2: float
    rhythmicity_p_value_group1: float
    rhythmicity_p_value_group2: float
    sine_fit_group1: SineFit
    sine_fit_group2: SineFit


def default_simulation_config() -> SimulationConfig:
    env = EnvironmentParameters()
    sim = SimulationParameters()
    group1 = GroupParameters(
        mu=np.array([-175.0, -100.0]),
        cov=env.speed_scale * np.eye(2),
        speed_fwd_mu=0.05,
        speed_fwd_std=0.02,
        min_fwd_speed=-0.05,
        speed_trn_mu=0.0,
        speed_trn_std=0.5,
    )
    group2 = GroupParameters(
        mu=np.array([0.0, 0.0]),
        cov=env.speed_scale * np.eye(2),
        speed_fwd_mu=0.02,
        speed_fwd_std=0.01,
        min_fwd_speed=-0.05,
        speed_trn_mu=0.0,
        speed_trn_std=0.5,
    )
    output = OutputOptions()
    return SimulationConfig(env=env, sim=sim, group1=group1, group2=group2, output=output)


def _copy_group_params(group: GroupParameters) -> GroupParameters:
    return GroupParameters(
        mu=np.array(group.mu, dtype=float),
        cov=np.array(group.cov, dtype=float),
        speed_fwd_mu=group.speed_fwd_mu,
        speed_fwd_std=group.speed_fwd_std,
        min_fwd_speed=group.min_fwd_speed,
        speed_trn_mu=group.speed_trn_mu,
        speed_trn_std=group.speed_trn_std,
    )


def _ensure_rng(rng: Optional[np.random.Generator] = None) -> np.random.Generator:
    """Normalize RNG handling so every function can accept an optional generator."""
    return rng if rng is not None else np.random.default_rng()


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    """Normalize each row vector while guarding against zero-length rows."""
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _covariance_sqrt(covariance: np.ndarray) -> np.ndarray:
    """
    Return the principal square root of a symmetric covariance matrix.

    MATLAB relies on chol/sqrtm; here we do an eigen-decomposition to remain
    numerically stable even when the covariance is only semi-definite.
    """
    eigvals, eigvecs = np.linalg.eigh(covariance)
    if np.any(eigvals < -1e-12):
        raise ValueError("Covariance matrix must be positive semi-definite.")
    eigvals = np.clip(eigvals, 0.0, None)
    root = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    return root


def _bin_statistics(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    bins_x: int,
    bins_y: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and count per 2D bin, matching MATLAB's histcounts2/accumarray."""
    x_edges = np.linspace(x_range[0], x_range[1], bins_x + 1)
    y_edges = np.linspace(y_range[0], y_range[1], bins_y + 1)

    bin_x = np.digitize(x, x_edges) - 1
    bin_y = np.digitize(y, y_edges) - 1

    valid = (
        (bin_x >= 0)
        & (bin_x < bins_x)
        & (bin_y >= 0)
        & (bin_y < bins_y)
        & (~np.isnan(values))
    )

    mean_grid = np.full((bins_y, bins_x), np.nan, dtype=float)
    count_grid = np.zeros((bins_y, bins_x), dtype=float)
    sum_grid = np.zeros((bins_y, bins_x), dtype=float)

    np.add.at(sum_grid, (bin_y[valid], bin_x[valid]), values[valid])
    np.add.at(count_grid, (bin_y[valid], bin_x[valid]), 1.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean_grid = sum_grid / count_grid

    return mean_grid, count_grid


def _print_progress(current: int, total: int, prefix: str = "") -> None:
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    percent = 100 * current / total
    sys.stdout.write(f"\r{prefix} [{bar}] {percent:5.1f}%")
    sys.stdout.flush()
    if current >= total:
        sys.stdout.write("\n")


def _phase_to_degrees(phase_steps: float, period: float) -> float:
    if not np.isfinite(phase_steps):
        return float("nan")
    normalized = (phase_steps % period) / period
    return normalized * 360.0


def _phase_difference_degrees(phase1_deg: float, phase2_deg: float) -> float:
    if not (np.isfinite(phase1_deg) and np.isfinite(phase2_deg)):
        return float("nan")
    diff = (phase2_deg - phase1_deg + 180.0) % 360.0 - 180.0
    return diff


def _sample_gamma_from_stats(
    mean: float, std: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    if mean <= 0:
        return np.zeros(size, dtype=float)
    if std <= 0:
        return np.full(size, mean, dtype=float)
    shape = (mean / std) ** 2
    scale = (std**2) / mean
    return rng.gamma(shape, scale, size)


def _sample_vonmises_from_stats(
    mean: float, std: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    if std <= 0:
        return np.full(size, mean, dtype=float)
    kappa = 1.0 / max(std**2, 1e-8)
    return rng.vonmises(mean, kappa, size=size)


def _save_mat(filename: str | Path, data: dict) -> None:
    """Persist MATLAB-like outputs, falling back to NPZ if SciPy is missing."""
    if _savemat is not None:
        _savemat(str(filename), data)
    else:  # pragma: no cover - only triggered without SciPy
        alt_path = Path(filename).with_suffix(".npz")
        np.savez(alt_path, **data)
        warnings.warn(
            f"SciPy is unavailable; saved data to {alt_path.name} instead of {filename}.",
            RuntimeWarning,
        )


def draw_positions_from_gaussian(
    mean: Sequence[float],
    covariance: Sequence[Sequence[float]],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample a single 2D position from a specified multivariate Gaussian.

    This mirrors the small MATLAB helper and is primarily used during agent
    initialization. The function name is kept for compatibility, although the
    implementation simply forwards to NumPy's multivariate_normal.
    """
    rng = _ensure_rng(rng)
    mean_arr = np.asarray(mean, dtype=float)
    cov_arr = np.asarray(covariance, dtype=float)
    return rng.multivariate_normal(mean_arr, cov_arr)


def draw_velocities(
    N: int,
    v_fwd_mu: float,
    v_fwd_std: float,
    v_trn_mu: float,
    v_trn_std: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Placeholder translation of the empty MATLAB function.

    The MATLAB source never implemented this routine, so we intentionally raise
    to signal callers that a bespoke velocity sampler still needs definition.
    """
    raise NotImplementedError("draw_velocities was not implemented in MATLAB.")


@dataclass
class SineFit:
    x_offset: float
    y_offset: float
    y_scale: float
    period: float

    def __call__(self, x: Sequence[float] | np.ndarray) -> np.ndarray:
        """Evaluate the fitted sine curve at new coordinates."""
        x_arr = np.asarray(x, dtype=float)
        omega = 2.0 * np.pi / self.period
        return self.y_offset + np.sin(omega * (x_arr - self.x_offset)) * self.y_scale


def fit_sine(
    x: Sequence[float],
    y: Sequence[float],
    period_length: float,
) -> SineFit:
    """
    Fit a sine with fixed period to (x, y) samples by linear regression.

    MATLAB used `fittype` + `fit` which performs nonlinear optimisation. The
    equivalent in Python is to fit the linearized sin/cos basis and convert it
    back to amplitude/phase form, avoiding iterative solvers while keeping the
    same expressivity.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = ~np.isnan(y_arr)
    if mask.sum() < 3:
        return SineFit(x_offset=0.0, y_offset=float(np.nan), y_scale=0.0, period=period_length)

    x_arr = x_arr[mask]
    y_arr = y_arr[mask]

    omega = 2.0 * np.pi / period_length
    # Fit y = offset + A*sin(ωx) + B*cos(ωx) and convert to phase/amplitude form
    design = np.column_stack(
        [
            np.ones_like(x_arr),
            np.sin(omega * x_arr),
            np.cos(omega * x_arr),
        ]
    )
    coeffs, *_ = np.linalg.lstsq(design, y_arr, rcond=None)
    y_offset, a_coef, b_coef = coeffs
    y_scale = float(math.hypot(a_coef, b_coef))
    if y_scale == 0:
        x_offset = 0.0
    else:
        x_offset = float(np.arctan2(-b_coef, a_coef) / omega)

    return SineFit(
        x_offset=x_offset % period_length,
        y_offset=float(y_offset),
        y_scale=y_scale,
        period=period_length,
    )


def _permutation_rhythmicity_test(
    x: np.ndarray,
    y: np.ndarray,
    period_length: float,
    permutations: int,
    rng: np.random.Generator,
) -> float:
    """
    Approximate a p-value for rhythmicity via permutation of the measurements.

    Null hypothesis: the observed ordering of y-values has no relationship with
    the sinusoidal driver (i.e. amplitudes arise by chance). The statistic is the
    sine-fit amplitude; shuffling destroys rhythmic structure while preserving
    the marginal distribution. Returns NaN if insufficient data are available.
    """
    if permutations <= 0:
        return math.nan

    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    if y.size < 5:
        return math.nan

    observed_fit = fit_sine(x, y, period_length)
    observed_amp = observed_fit.y_scale
    if not np.isfinite(observed_amp):
        return math.nan

    exceedances = 0
    for _ in range(permutations):
        shuffled = rng.permutation(y)
        amp = fit_sine(x, shuffled, period_length).y_scale
        if amp >= observed_amp:
            exceedances += 1

    # Add-one smoothing avoids zero p-values with finite permutation counts.
    return (exceedances + 1) / (permutations + 1)


def mgd(
    N: int,
    d: int,
    rmean: Sequence[float],
    covariance: Sequence[Sequence[float]],
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate `N` samples from a multivariate Gaussian with mean/covariance.

    The MATLAB helper zero-centres the initial `randn` draws before imposing the
    desired covariance. We mimic that approach (it matters for small `N`) and
    use an eigen-based square root so positive semi-definite covariances behave.
    """
    rng = _ensure_rng(rng)
    mean_arr = np.asarray(rmean, dtype=float)
    cov_arr = np.asarray(covariance, dtype=float)

    if mean_arr.shape not in {(d,), (1, d)}:
        raise ValueError("Mean vector has incompatible shape.")
    mean_arr = mean_arr.reshape(-1)
    if cov_arr.shape != (d, d):
        raise ValueError("Covariance matrix must be square with shape (d, d).")

    N = max(int(N), 1)
    samples = rng.standard_normal((N, d))
    if N > 1:
        samples -= samples.mean(axis=0, keepdims=True)

    # Impose the desired covariance via the principal matrix square root
    transform = _covariance_sqrt(cov_arr)
    samples = samples @ transform.T + mean_arr
    return samples


def run_simulation(
    density_factors: Iterable[float] = (3.0, 7.0, 10.0),
    config: Optional[SimulationConfig] = None,
    ablation: Optional[AblationConfig] = None,
    show_point_motion: bool = True,
    save_outputs: bool = True,
    rhythmicity_permutations: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> list[SimulationSummary]:
    """
    Execute the full colony simulation for the requested density factors.

    Parameters
    ----------
    density_factors:
        Multipliers applied to the baseline group sizes (500 agents per group).
    config:
        Optional simulation configuration. When omitted the defaults that mimic
        the MATLAB values are used.
    ablation:
        Toggle individual components (walls, homing, drivers, etc.) so that
        ablation sweeps can be run without touching the core logic.
    show_point_motion:
        Keep matplotlib scatter plots updated. Disable for headless sweeps.
    save_outputs:
        Persist `.mat` files compatible with the MATLAB analysis pipeline.
    rhythmicity_permutations:
        Number of permutations used for the sine-amplitude significance test.
        Set to 0 to skip the statistical check.
    rng:
        Optional NumPy Generator for deterministic sampling.

    Returns
    -------
    List[SimulationSummary]
        One summary per density factor, capturing amplitudes and phase shifts to
        support rhythmicity comparisons during ablation studies.
    """
    rng = _ensure_rng(rng)
    cfg = config or default_simulation_config()
    abl = ablation or AblationConfig()

    summaries: list[SimulationSummary] = []

    for density_factor in density_factors:
        print(f"density factor: {density_factor:.1f}")

        group1_cfg = _copy_group_params(cfg.group1)
        group2_cfg = _copy_group_params(cfg.group2)

        if abl.homogenize_initial_positions:
            shared_mu = 0.5 * (group1_cfg.mu + group2_cfg.mu)
            shared_cov = 0.5 * (group1_cfg.cov + group2_cfg.cov)
            group1_cfg.mu = shared_mu.copy()
            group2_cfg.mu = shared_mu.copy()
            group1_cfg.cov = shared_cov.copy()
            group2_cfg.cov = shared_cov.copy()

        if abl.homogenize_turning_noise:
            mean_turn_mu = 0.5 * (group1_cfg.speed_trn_mu + group2_cfg.speed_trn_mu)
            mean_turn_std = 0.5 * (group1_cfg.speed_trn_std + group2_cfg.speed_trn_std)
            group1_cfg.speed_trn_mu = group2_cfg.speed_trn_mu = mean_turn_mu
            group1_cfg.speed_trn_std = group2_cfg.speed_trn_std = mean_turn_std

        group1_size = int(round(500 * density_factor))
        group2_size = int(round(500 * density_factor))
        total_agents = group1_size + group2_size

        hive_bounds = cfg.env.hive_bounds
        bins_x = cfg.env.bins_x
        bins_y = cfg.env.bins_y

        agents_group1 = np.hstack(
            [
                mgd(group1_size, 2, group1_cfg.mu, group1_cfg.cov, rng=rng),
                rng.uniform(0.0, 2.0 * np.pi, size=(group1_size, 1)),
            ]
        )
        agents_group2 = np.hstack(
            [
                mgd(group2_size, 2, group2_cfg.mu, group2_cfg.cov, rng=rng),
                rng.uniform(0.0, 2.0 * np.pi, size=(group2_size, 1)),
            ]
        )
        agents = np.vstack([agents_group1, agents_group2])

        speed_transfers = np.zeros(total_agents, dtype=float)

        sim_day = cfg.sim.day_duration
        sim_duration = cfg.sim.sim_duration

        speeds_history = np.full((total_agents, sim_duration), np.nan, dtype=float)
        cov_history = np.full((2, sim_duration), np.nan, dtype=float)
        speeds_bin = np.full((bins_y, bins_x, sim_duration), np.nan, dtype=float)
        speeds_bin_counts = np.zeros((bins_y, bins_x, sim_duration), dtype=float)
        positions_history = np.full((total_agents, sim_duration, 2), np.nan, dtype=float)

        idx_group1 = np.arange(group1_size)
        idx_group2 = np.arange(group1_size, total_agents)

        scatter_artists = None
        fig_motion = None
        if show_point_motion:
            fig_motion = plt.figure(1)
            ax = fig_motion.gca()
            ax.set_aspect("equal", adjustable="box")
            window_bounds = 1.1 * hive_bounds
            ax.set_xlim(window_bounds[0], window_bounds[1])
            ax.set_ylim(window_bounds[2], window_bounds[3])

        interaction_threshold = math.sqrt(cfg.env.speed_scale) * cfg.env.interaction_threshold_factor

        progress_prefix = f"Sim density {density_factor:.1f}"
        progress_step = max(1, sim_duration // 100)

        for t in range(sim_duration):
            time_step = t + 1

            if abl.disable_external_driver:
                speed_external_driver = 0.0
            else:
                speed_external_driver = (
                    cfg.env.speed_scale
                    * cfg.env.speed_transfer_scale
                    * group1_cfg.speed_fwd_mu
                    * 0.5
                    * (math.sin(time_step * 2.0 * math.pi / sim_day) + 1.0)
                )
            external_vector = np.concatenate(
                [
                    np.full(group1_size, speed_external_driver, dtype=float),
                    np.zeros(group2_size, dtype=float),
                ]
            )

            group1_gamma = _sample_gamma_from_stats(
                cfg.env.speed_scale * group1_cfg.speed_fwd_mu,
                cfg.env.speed_scale * group1_cfg.speed_fwd_std,
                group1_size,
                rng,
            )
            group2_gamma = _sample_gamma_from_stats(
                cfg.env.speed_scale * group2_cfg.speed_fwd_mu,
                cfg.env.speed_scale * group2_cfg.speed_fwd_std,
                group2_size,
                rng,
            )
            group1_speeds = np.maximum(group1_cfg.min_fwd_speed, group1_gamma)
            group2_speeds = np.maximum(group2_cfg.min_fwd_speed, group2_gamma)
            speeds = np.concatenate([group1_speeds, group2_speeds]) + external_vector
            if not abl.disable_speed_transfer:
                speeds += speed_transfers

            group1_turn = _sample_vonmises_from_stats(
                group1_cfg.speed_trn_mu,
                group1_cfg.speed_trn_std,
                group1_size,
                rng,
            )
            group2_turn = _sample_vonmises_from_stats(
                group2_cfg.speed_trn_mu,
                group2_cfg.speed_trn_std,
                group2_size,
                rng,
            )
            speeds_turn = np.concatenate([group1_turn, group2_turn])

            velocities = speeds[:, None] * np.column_stack(
                [np.cos(agents[:, 2]), np.sin(agents[:, 2])]
            )

            if not abl.disable_homing:
                home1 = np.tile(group1_cfg.mu, (group1_size, 1)) - agents[idx_group1, :2]
                home1 = _normalize_rows(home1)
                velocities[idx_group1] += (
                    (cfg.env.speed_scale * cfg.env.homing_pull * group1_cfg.speed_fwd_mu)
                    + (group1_cfg.speed_fwd_mu * cfg.env.speed_transfer_scale / 2.0)
                ) * home1

                home2 = np.tile(group2_cfg.mu, (group2_size, 1)) - agents[idx_group2, :2]
                home2 = _normalize_rows(home2)
                velocities[idx_group2] += (cfg.env.speed_scale * cfg.env.homing_pull * group2_cfg.speed_fwd_mu) * home2

            agents[:, :2] += velocities
            agents[:, 2] += speeds_turn

            if not abl.disable_walls:
                for axis, bound_idx in enumerate((0, 1)):
                    lower = hive_bounds[bound_idx * 2]
                    upper = hive_bounds[bound_idx * 2 + 1]

                    below = agents[:, axis] < lower
                    agents[below, axis] = lower
                    agents[below, 2] = np.arctan2(-agents[below, 1], -agents[below, 0])

                    above = agents[:, axis] > upper
                    agents[above, axis] = upper
                    agents[above, 2] = np.arctan2(-agents[above, 1], -agents[above, 0])

            diff = agents[:, None, :2] - agents[None, :, :2]
            distances = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(distances, 2.0 * interaction_threshold)
            interactions = distances < interaction_threshold
            interacting_rows, interacting_cols = np.where(interactions)
            interacting_indices = (
                np.unique(np.concatenate([interacting_rows, interacting_cols]))
                if interacting_rows.size
                else np.empty(0, dtype=int)
            )

            if not abl.disable_speed_transfer:
                speed_transfers *= 0.0
                if interacting_rows.size:
                    faster = speeds[interacting_cols] - speeds[interacting_rows]
                    faster = np.where(speeds[interacting_rows] < speeds[interacting_cols], faster, 0.0)
                    np.maximum.at(speed_transfers, interacting_rows, faster)
            else:
                speed_transfers *= 0.0

            positions_history[:, t, :] = agents[:, :2]
            speeds_history[:, t] = speeds

            means, counts = _bin_statistics(
                agents[:, 0],
                agents[:, 1],
                speeds,
                bins_x,
                bins_y,
                (hive_bounds[0], hive_bounds[1]),
                (hive_bounds[2], hive_bounds[3]),
            )
            speeds_bin[:, :, t] = means
            speeds_bin_counts[:, :, t] = counts

            if show_point_motion:
                ax = plt.figure(1).gca()
                interaction_points = agents[interacting_indices, :2] if interacting_indices.size else np.empty((0, 2))

                if scatter_artists is None:
                    scatter_group1 = ax.scatter(
                        agents[idx_group1, 0],
                        agents[idx_group1, 1],
                        c="blue",
                        s=10,
                    )
                    scatter_group2 = ax.scatter(
                        agents[idx_group2, 0],
                        agents[idx_group2, 1],
                        c="green",
                        s=10,
                    )
                    scatter_interactions = ax.scatter(
                        interaction_points[:, 0] if interaction_points.size else [],
                        interaction_points[:, 1] if interaction_points.size else [],
                        c="red",
                        s=20,
                    )
                    scatter_artists = (scatter_group1, scatter_group2, scatter_interactions)
                else:
                    scatter_artists[0].set_offsets(agents[idx_group1, :2])
                    scatter_artists[1].set_offsets(agents[idx_group2, :2])
                    scatter_artists[2].set_offsets(interaction_points)
                plt.pause(0.001)

            if time_step % progress_step == 0 or time_step == sim_duration:
                _print_progress(time_step, sim_duration, prefix=progress_prefix)

        group1_mean_time = speeds_history[idx_group1].mean(axis=0)
        group2_mean_time = speeds_history[idx_group2].mean(axis=0)
        group1_speed_mu = float(np.nanmean(group1_mean_time))
        group1_speed_sigma = float(np.nanmean(np.nanstd(speeds_history[idx_group1], axis=0)))
        group2_speed_mu = float(np.nanmean(group2_mean_time))
        group2_speed_sigma = float(np.nanmean(np.nanstd(speeds_history[idx_group2], axis=0)))

        cov_history[:, -1] = [
            float(np.trace(np.cov(agents[idx_group1, :2].T))),
            float(np.trace(np.cov(agents[idx_group2, :2].T))),
        ]

        print(f"group 1 (blue): speeds mu={group1_speed_mu:.4f}, std={group1_speed_sigma:.4f}")
        print(f"group 2 (green): speeds mu={group2_speed_mu:.4f}, std={group2_speed_sigma:.4f}")

        x_values = np.arange(1, sim_duration + 1, dtype=float)
        sine_fit1 = fit_sine(x_values, group1_mean_time, sim_day)
        sine_fit2 = fit_sine(x_values, group2_mean_time, sim_day)
        p_value_group1 = _permutation_rhythmicity_test(
            x_values, group1_mean_time, sim_day, rhythmicity_permutations, rng
        )
        p_value_group2 = _permutation_rhythmicity_test(
            x_values, group2_mean_time, sim_day, rhythmicity_permutations, rng
        )
        phase1_deg = _phase_to_degrees(float(sine_fit1.x_offset), sim_day)
        phase2_deg = _phase_to_degrees(float(sine_fit2.x_offset), sim_day)
        phase_shift = _phase_difference_degrees(phase1_deg, phase2_deg)
        summaries.append(
            SimulationSummary(
                density_factor=density_factor,
                amplitude_group1=sine_fit1.y_scale,
                amplitude_group2=sine_fit2.y_scale,
                phase_shift_g2_minus_g1=phase_shift,
                phase_group1=phase1_deg,
                phase_group2=phase2_deg,
                rhythmicity_p_value_group1=p_value_group1,
                rhythmicity_p_value_group2=p_value_group2,
                sine_fit_group1=sine_fit1,
                sine_fit_group2=sine_fit2,
            )
        )

        print(
            "Summary density {density:.1f}: amp1={amp1:.3f}, amp2={amp2:.3f}, "
            "phase1={phase1:.1f}°, phase2={phase2:.1f}°, phase_shift={phase:.1f}°, "
            "p1={p1:.3g}, p2={p2:.3g}".format(
                density=density_factor,
                amp1=sine_fit1.y_scale,
                amp2=sine_fit2.y_scale,
                phase1=phase1_deg,
                phase2=phase2_deg,
                phase=phase_shift,
                p1=p_value_group1 if np.isfinite(p_value_group1) else float("nan"),
                p2=p_value_group2 if np.isfinite(p_value_group2) else float("nan"),
            )
        )

        phase_vs_location = np.full((bins_y, bins_x), np.nan, dtype=float)
        number_of_samples_vs_location = np.nansum(speeds_bin, axis=2)

        for row in range(bins_y):
            for col in range(bins_x):
                series = speeds_bin[row, col, :]
                sine = fit_sine(x_values, series, sim_day)
                phase_vs_location[row, col] = sine.x_offset if not math.isnan(sine.y_offset) else np.nan

        if save_outputs and cfg.output.save_mat:
            cfg.output.output_dir.mkdir(parents=True, exist_ok=True)
            out_filename_phase = cfg.output.output_dir / f"phase_vs_location_density_{density_factor:.1f}.mat"
            out_filename_out = cfg.output.output_dir / f"OUT_density_{density_factor:.1f}.mat"
            _save_mat(
                out_filename_phase,
                {
                    "phase_vs_location": phase_vs_location,
                    "number_of_sample_vs_location": number_of_samples_vs_location,
                },
            )
            _save_mat(
                out_filename_out,
                {
                    "speeds": speeds_history,
                    "cov": cov_history,
                    "speeds_bin": speeds_bin,
                    "speeds_bin_n": speeds_bin_counts,
                    "positions": positions_history,
                },
            )

        fig_time = None
        fig_phase = None

        if show_point_motion or save_outputs:
            fig_time = plt.figure(2)
            ax2 = fig_time.gca()
            ax2.clear()
            ax2.plot(
                np.tile(x_values, (group1_size, 1)).T,
                speeds_history[idx_group1].T,
                ".b",
                alpha=0.3,
            )
            ax2.plot(
                np.tile(x_values, (group2_size, 1)).T,
                speeds_history[idx_group2].T,
                ".g",
                alpha=0.3,
            )
            ax2.plot(x_values, group2_mean_time, "r", label="Group 2 mean")
            ax2.plot(x_values, group1_mean_time, "y", label="Group 1 mean")
            ax2.plot(x_values, sine_fit1(x_values), label="Group 1 sine fit")
            ax2.plot(x_values, sine_fit2(x_values), label="Group 2 sine fit")
            ax2.set_title(f"Speed Evolution (density {density_factor:.1f})")
            ax2.set_xlabel("Time step")
            ax2.set_ylabel("Speed")
            ax2.legend()

            fig_phase = plt.figure(3)
            ax3 = fig_phase.gca()
            ax3.clear()
            im = ax3.imshow(phase_vs_location, origin="lower")
            plt.colorbar(im, ax=ax3)
            ax3.set_title(f"Phase vs Location (density {density_factor:.1f})")

        if save_outputs:
            cfg.output.output_dir.mkdir(parents=True, exist_ok=True)
            if fig_time is not None:
                fig_time.savefig(
                    cfg.output.output_dir / f"time_series_density_{density_factor:.1f}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
            if fig_phase is not None:
                fig_phase.savefig(
                    cfg.output.output_dir / f"phase_map_density_{density_factor:.1f}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
            if fig_motion is not None:
                fig_motion.savefig(
                    cfg.output.output_dir / f"motion_density_{density_factor:.1f}.png",
                    dpi=150,
                    bbox_inches="tight",
                )

    return summaries


def run_ablation_study(
    density_factor: float,
    ablations: dict[str, AblationConfig],
    config: Optional[SimulationConfig] = None,
    rhythmicity_permutations: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, SimulationSummary]:
    """
    Convenience helper to batch-run ablation settings and collect summaries.

    Parameters
    ----------
    density_factor:
        Single density factor shared across the ablation sweep.
    ablations:
        Mapping from human-readable label to `AblationConfig`.
    config:
        Optional shared simulation config.
    rhythmicity_permutations:
        Number of permutations for the rhythmicity test (forwarded to
        `run_simulation`).
    rng:
        Optional RNG; if provided it is advanced between runs so each ablation is
        driven by a reproducible, independent stream.
    """
    results: dict[str, SimulationSummary] = {}
    for label, abl in ablations.items():
        print(f"\n=== Ablation: {label} ===")
        summaries = run_simulation(
            density_factors=(density_factor,),
            config=config,
            ablation=abl,
            show_point_motion=False,
            save_outputs=False,
            rhythmicity_permutations=rhythmicity_permutations,
            rng=rng,
        )
        results[label] = summaries[0]
    return results


def run_single_factor_ablation_suite(
    density_factor: float,
    config: Optional[SimulationConfig] = None,
    rhythmicity_permutations: int = 200,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, SimulationSummary]:
    """
    Run baseline plus all single-factor ablations for the given density.

    Returns
    -------
    Dict[str, SimulationSummary]
        Mapping from ablation label to its simulation summary. Includes the
        "baseline" (no ablation) entry for reference.
    """
    configs: dict[str, AblationConfig] = {
        "baseline": AblationConfig(),
        "no_walls": AblationConfig(disable_walls=True),
        "no_homing": AblationConfig(disable_homing=True),
        "no_speed_transfer": AblationConfig(disable_speed_transfer=True),
        "no_external_driver": AblationConfig(disable_external_driver=True),
        "homogenize_turn": AblationConfig(homogenize_turning_noise=True),
        "homogenize_init": AblationConfig(homogenize_initial_positions=True),
    }
    return run_ablation_study(
        density_factor=density_factor,
        ablations=configs,
        config=config,
        rhythmicity_permutations=rhythmicity_permutations,
        rng=rng,
    )


def export_ablation_results(results: dict[str, SimulationSummary], path: str | Path) -> None:
    """Write ablation summaries to CSV for downstream analysis."""
    fieldnames = [
        "label",
        "density_factor",
        "amplitude_group1",
        "amplitude_group2",
        "phase_group1",
        "phase_group2",
        "phase_shift_g2_minus_g1",
        "rhythmicity_p_value_group1",
        "rhythmicity_p_value_group2",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for label, summary in results.items():
            writer.writerow(
                {
                    "label": label,
                    "density_factor": summary.density_factor,
                    "amplitude_group1": summary.amplitude_group1,
                    "amplitude_group2": summary.amplitude_group2,
                    "phase_group1": summary.phase_group1,
                    "phase_group2": summary.phase_group2,
                    "phase_shift_g2_minus_g1": summary.phase_shift_g2_minus_g1,
                    "rhythmicity_p_value_group1": summary.rhythmicity_p_value_group1,
                    "rhythmicity_p_value_group2": summary.rhythmicity_p_value_group2,
                }
            )


def plot_ablation_results(results: dict[str, SimulationSummary]) -> None:
    """Visualize amplitudes, phase shifts, and p-values across ablations."""
    labels = list(results.keys())
    amps1 = [results[label].amplitude_group1 for label in labels]
    amps2 = [results[label].amplitude_group2 for label in labels]
    phases = [results[label].phase_shift_g2_minus_g1 for label in labels]
    pvals1 = [results[label].rhythmicity_p_value_group1 for label in labels]
    pvals2 = [results[label].rhythmicity_p_value_group2 for label in labels]

    phases_g1 = [results[label].phase_group1 for label in labels]
    phases_g2 = [results[label].phase_group2 for label in labels]

    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=True)

    axes[0].plot(labels, amps1, marker="o", label="Group 1 amplitude")
    axes[0].plot(labels, amps2, marker="o", label="Group 2 amplitude")
    axes[0].set_ylabel("Speed amplitude")
    axes[0].legend()
    axes[0].set_title("Sine amplitudes across ablations")

    axes[1].plot(labels, phases_g1, marker="o", label="Group 1 phase")
    axes[1].plot(labels, phases_g2, marker="o", label="Group 2 phase")
    axes[1].set_ylabel("Phase (degrees)")
    axes[1].set_title("Absolute phase offsets (0°-360°)")
    axes[1].legend()

    axes[2].plot(labels, phases, marker="o", color="tab:orange")
    axes[2].set_ylabel("Phase shift (degrees)")
    axes[2].set_title("Phase shift (Group2 - Group1, degrees)")

    axes[3].semilogy(labels, pvals1, marker="o", label="Group 1 p-value")
    axes[3].semilogy(labels, pvals2, marker="o", label="Group 2 p-value")
    axes[3].axhline(0.05, color="red", linestyle="--", linewidth=1)
    axes[3].set_ylabel("Permutation p-value")
    axes[3].set_xlabel("Ablation")
    axes[3].legend()
    axes[3].set_title("Rhythmicity significance")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()


def sketch_agents_2D(
    N: int = 100,
    steps: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Minimal 2D random walk visualisation used during the MATLAB prototyping stage.
    Agents are initialised from a Gaussian cloud and move according to noisy
    forward/turn speeds. Intended for qualitative inspection only.
    """
    rng = _ensure_rng(rng)
    mu = [0.0, 0.0]
    cov = np.eye(2)

    positions = mgd(N, 2, mu, cov, rng=rng)
    orientations = rng.uniform(0.0, 2.0 * np.pi, size=(N, 1))
    agents = np.hstack([positions, orientations])

    speed_xy_avg = 1.0
    speed_xy_std = 1.0
    speed_trn_avg = 0.0
    speed_trn_std = 0.5

    plt.figure()
    for _ in range(steps):
        speeds = speed_xy_std * rng.standard_normal(N) + speed_xy_avg
        velocities = speeds[:, None] * np.column_stack(
            [np.sin(agents[:, 2]), np.cos(agents[:, 2])]
        )
        agents[:, :2] += velocities

        turns = speed_trn_std * rng.standard_normal(N) + speed_trn_avg
        agents[:, 2] += turns

        plt.cla()
        plt.scatter(agents[:, 0], agents[:, 1], s=10)
        plt.axis("equal")
        plt.pause(0.001)

    plt.scatter(agents[:, 0], agents[:, 1], s=10)
    plt.show()


def sketch_agents_3D(
    N1: int = 100,
    N2: int = 100,
    steps: int = 1000,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Two-group variant of the sketching tool that visualises interactions and
    homing behaviour. The implementation mirrors the MATLAB script closely so
    any observed patterns can be compared across languages.
    """
    rng = _ensure_rng(rng)
    mu1 = [-1.0, 0.0]
    mu2 = [1.0, 0.0]
    cov = np.eye(2)

    agents_group1 = np.hstack(
        [mgd(N1, 2, mu1, cov, rng=rng), rng.uniform(0.0, 2.0 * np.pi, size=(N1, 1))]
    )
    agents_group2 = np.hstack(
        [mgd(N2, 2, mu2, cov, rng=rng), rng.uniform(0.0, 2.0 * np.pi, size=(N2, 1))]
    )
    agents = np.vstack([agents_group1, agents_group2])

    homing_pull = 0.005
    speed_xy_avg = 1.0
    speed_xy_std = 1.0
    speed_trn_avg = 0.0
    speed_trn_std = 0.5
    threshold_dist = 0.15
    window_size = 4.0

    N = N1 + N2
    idx_blue = np.arange(N1)
    idx_green = np.arange(N1, N)
    speed_transfers = np.zeros(N, dtype=float)

    plt.figure()
    for _ in range(steps):
        speeds = np.maximum(
            0.0,
            speed_xy_std * rng.standard_normal(N) + speed_xy_avg,
        ) + speed_transfers

        velocities = speeds[:, None] * np.column_stack(
            [np.sin(agents[:, 2]), np.cos(agents[:, 2])]
        )

        home1 = _normalize_rows(np.tile(mu1, (N1, 1)) - agents[idx_blue, :2])
        home2 = _normalize_rows(np.tile(mu2, (N2, 1)) - agents[idx_green, :2])
        velocities[idx_blue] += homing_pull * (np.linalg.norm(home1) ** 3) * (home1 - velocities[idx_blue])
        velocities[idx_green] += homing_pull * (np.linalg.norm(home2) ** 3) * (home2 - velocities[idx_green])

        agents[:, :2] += velocities

        turns = speed_trn_std * rng.standard_normal(N) + speed_trn_avg
        agents[:, 2] += turns

        diff = agents[:, None, :2] - agents[None, :, :2]
        distances = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(distances, threshold_dist * 2.0)
        interactions = distances < threshold_dist
        rows, cols = np.where(interactions)

        speed_transfers *= 0.5
        if rows.size:
            gain = np.where(speeds[rows] < speeds[cols], speeds[cols] - speeds[rows], 0.0)
            np.add.at(speed_transfers, rows, gain)

        plt.cla()
        plt.scatter(agents[idx_blue, 0], agents[idx_blue, 1], s=10, c="blue")
        plt.scatter(agents[idx_green, 0], agents[idx_green, 1], s=10, c="green")
        plt.scatter(agents[rows, 0], agents[rows, 1], s=10, c="red")
        plt.xlim(-window_size, window_size)
        plt.ylim(-window_size, window_size)
        plt.pause(0.001)

    plt.show()


if __name__ == "__main__":
    run_simulation(density_factors=[1.0], show_point_motion=True, save_outputs=True)
    
    #results = run_single_factor_ablation_suite(density_factor=3.0)
    #export_ablation_results(results, "ablation_results.csv") 
    #plot_ablation_results(results) 
