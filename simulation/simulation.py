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
from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.io import savemat as _savemat
except ImportError:  # pragma: no cover - SciPy may be unavailable
    _savemat = None

__all__ = [
    "draw_positions_from_gaussian",
    "draw_velocities",
    "fit_sine",
    "mgd",
    "run_simulation",
    "sketch_agents_2D",
    "sketch_agents_3D",
]


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
    show_point_motion: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """
    Execute the full colony simulation for the requested density factors.

    The routine intentionally sticks close to the MATLAB structure: the nested
    loops, the per-step statistics collection, and even the plotting order are
    maintained. Only minor Python niceties (vectorization, helpers, dataclasses)
    have been introduced to keep the code approachable to MATLAB users.
    """
    rng = _ensure_rng(rng)

    for density_factor in density_factors:
        print(f"density factor: {density_factor:.1f}")

        speed_scale = 50.0
        homing_pull = 0.1
        speed_transfer_scale = 1.0

        group1_cfg = {
            "mu": np.array([-175.0, -100.0]),
            "cov": speed_scale * np.eye(2),
            "speed_fwd_mu": 0.05,
            "speed_fwd_std": 0.02,
            "min_fwd_speed": -0.05,
            "speed_trn_mu": 0.0,
            "speed_trn_std": 0.5,
        }
        group2_cfg = {
            "mu": np.array([0.0, 0.0]),
            "cov": speed_scale * np.eye(2),
            "speed_fwd_mu": 0.02,
            "speed_fwd_std": 0.01,
            "min_fwd_speed": -0.05,
            "speed_trn_mu": 0.0,
            "speed_trn_std": 0.5,
        }

        group1_size = int(round(500 * density_factor))
        group2_size = int(round(500 * density_factor))
        total_agents = group1_size + group2_size

        hive_bounds = np.array([-354 / 2, 354 / 2, -205 / 2, 205 / 2], dtype=float)
        bins_x = 45
        bins_y = 30

        agents_group1 = np.hstack(
            [
                mgd(group1_size, 2, group1_cfg["mu"], group1_cfg["cov"], rng=rng),
                rng.uniform(0.0, 2.0 * np.pi, size=(group1_size, 1)),
            ]
        )
        agents_group2 = np.hstack(
            [
                mgd(group2_size, 2, group2_cfg["mu"], group2_cfg["cov"], rng=rng),
                rng.uniform(0.0, 2.0 * np.pi, size=(group2_size, 1)),
            ]
        )
        agents = np.vstack([agents_group1, agents_group2])

        # Holds per-agent speed bonuses gained via interactions
        speed_transfers = np.zeros(total_agents, dtype=float)

        sim_day = 400
        sim_duration = 800

        # Preallocate arrays matching the MATLAB OUT struct.
        speeds_history = np.full((total_agents, sim_duration), np.nan, dtype=float)
        cov_history = np.full((2, sim_duration), np.nan, dtype=float)
        speeds_bin = np.full((bins_y, bins_x, sim_duration), np.nan, dtype=float)
        speeds_bin_counts = np.zeros((bins_y, bins_x, sim_duration), dtype=float)
        positions_history = np.full((total_agents, sim_duration, 2), np.nan, dtype=float)

        idx_group1 = np.arange(group1_size)
        idx_group2 = np.arange(group1_size, total_agents)

        scatter_artists = None
        if show_point_motion:
            fig = plt.figure(1)
            ax = fig.gca()
            ax.set_aspect("equal", adjustable="box")
            window_bounds = 1.1 * hive_bounds
            ax.set_xlim(window_bounds[0], window_bounds[1])
            ax.set_ylim(window_bounds[2], window_bounds[3])

        for t in range(sim_duration):
            time_step = t + 1

            speed_external_driver = (
                speed_scale
                * speed_transfer_scale
                * group1_cfg["speed_fwd_mu"]
                * 0.5
                * (math.sin(time_step * 2.0 * math.pi / sim_day) + 1.0)
            )
            external_vector = np.concatenate(
                [
                    np.full(group1_size, speed_external_driver, dtype=float),
                    np.zeros(group2_size, dtype=float),
                ]
            )

            # Draw intrinsic forward speeds for each group and clip below the
            # minimum specified drift.
            group1_speeds = np.maximum(
                group1_cfg["min_fwd_speed"],
                speed_scale * group1_cfg["speed_fwd_std"] * rng.standard_normal(group1_size)
                + speed_scale * group1_cfg["speed_fwd_mu"],
            )
            group2_speeds = np.maximum(
                group2_cfg["min_fwd_speed"],
                speed_scale * group2_cfg["speed_fwd_std"] * rng.standard_normal(group2_size)
                + speed_scale * group2_cfg["speed_fwd_mu"],
            )
            speeds = np.concatenate([group1_speeds, group2_speeds]) + speed_transfers + external_vector

            group1_turn = group1_cfg["speed_trn_std"] * rng.standard_normal(group1_size) + group1_cfg["speed_trn_mu"]
            group2_turn = group2_cfg["speed_trn_std"] * rng.standard_normal(group2_size) + group2_cfg["speed_trn_mu"]
            speeds_turn = np.concatenate([group1_turn, group2_turn])

            # Base translation is along the current heading by the chosen speed.
            velocities = speeds[:, None] * np.column_stack(
                [np.cos(agents[:, 2]), np.sin(agents[:, 2])]
            )

            home1 = np.tile(group1_cfg["mu"], (group1_size, 1)) - agents[idx_group1, :2]
            home1 = _normalize_rows(home1)
            velocities[idx_group1] += (
                (speed_scale * homing_pull * group1_cfg["speed_fwd_mu"])
                + (group1_cfg["speed_fwd_mu"] * speed_transfer_scale / 2.0)
            ) * home1

            home2 = np.tile(group2_cfg["mu"], (group2_size, 1)) - agents[idx_group2, :2]
            home2 = _normalize_rows(home2)
            velocities[idx_group2] += (speed_scale * homing_pull * group2_cfg["speed_fwd_mu"]) * home2

            agents[:, :2] += velocities
            agents[:, 2] += speeds_turn

            for axis, bound_idx in enumerate((0, 1)):
                lower = hive_bounds[bound_idx * 2]
                upper = hive_bounds[bound_idx * 2 + 1]

                # Enforce reflective boundaries: clamp positions and point agents inward.
                below = agents[:, axis] < lower
                agents[below, axis] = lower
                agents[below, 2] = np.arctan2(-agents[below, 1], -agents[below, 0])

                above = agents[:, axis] > upper
                agents[above, axis] = upper
                agents[above, 2] = np.arctan2(-agents[above, 1], -agents[above, 0])

            # Pairwise distance matrix (N x N) used to decide interactions.
            diff = agents[:, None, :2] - agents[None, :, :2]
            distances = np.linalg.norm(diff, axis=2)
            interaction_threshold = math.sqrt(speed_scale) * 0.2
            np.fill_diagonal(distances, 2.0 * interaction_threshold)
            interactions = distances < interaction_threshold
            interacting_rows, interacting_cols = np.where(interactions)

            speed_transfers *= 0.0
            if interacting_rows.size:
                # Pull speed from faster neighbors while forbidding self-loops
                faster = speeds[interacting_cols] - speeds[interacting_rows]
                faster = np.where(speeds[interacting_rows] < speeds[interacting_cols], faster, 0.0)
                np.maximum.at(speed_transfers, interacting_rows, faster)

            positions_history[:, t, :] = agents[:, :2]
            speeds_history[:, t] = speeds

            # Bin current speeds per spatial cell to reconstruct the MATLAB heatmaps.
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
                    scatter_artists = (scatter_group1, scatter_group2)
                else:
                    scatter_artists[0].set_offsets(agents[idx_group1, :2])
                    scatter_artists[1].set_offsets(agents[idx_group2, :2])
                plt.pause(0.001)

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

        phase_vs_location = np.full((bins_y, bins_x), np.nan, dtype=float)
        number_of_samples_vs_location = np.nansum(speeds_bin, axis=2)

        for row in range(bins_y):
            for col in range(bins_x):
                series = speeds_bin[row, col, :]
                sine = fit_sine(x_values, series, sim_day)
                phase_vs_location[row, col] = sine.x_offset if not math.isnan(sine.y_offset) else np.nan

        plt.figure(3)
        ax3 = plt.gca()
        im = ax3.imshow(phase_vs_location, origin="lower")
        plt.colorbar(im, ax=ax3)
        ax3.set_title(f"Phase vs Location (density {density_factor:.1f})")

        out_filename_phase = f"phase_vs_location_density_{density_factor:.1f}.mat"
        out_filename_out = f"OUT_density_{density_factor:.1f}.mat"
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

        plt.figure(2)
        ax2 = plt.gca()
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
    run_simulation()
