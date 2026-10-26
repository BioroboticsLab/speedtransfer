"""
Reproduce Figure 3b: spatial heatmap of peak-activity phase from the
baseline simulation.

This script:
  1. Runs the baseline simulation (density=1.0) and captures per-agent
     speed and position histories via the history_callback.
  2. Fits a cosinor (sine) model per agent in sliding 3-day windows to
     extract the phase of peak activity.
  3. Accumulates the fitted phase at every position each agent visited
     during each day, then takes the circular mean per spatial bin.
  4. Converts phase to "hour of peak activity" and plots a heatmap
     matching the style of the original Figure 3b.

No external dependencies beyond the project's own simulation.py are needed.
"""
import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Ensure the simulation module is importable
sys.path.insert(0, os.path.dirname(__file__))
from simulation import (
    run_simulation,
    fit_sine,
    default_simulation_config,
    AblationConfig,
)


# ---------------------------------------------------------------------------
# 1.  Run the baseline simulation and capture history
# ---------------------------------------------------------------------------
def run_baseline_and_capture(density: float = 1.0, seed: int = 42):
    """Run baseline simulation and return the history dict.
    
    Uses the same parameters as the ablation study:
      day_duration = 100, sim_duration = 500, density = 1.0
    """
    captured = {}

    def _callback(df, history):
        captured[df] = history

    cfg = default_simulation_config()
    # Match ablation study parameters
    cfg.sim.day_duration = 100
    cfg.sim.sim_duration = 500
    rng = np.random.default_rng(seed)

    run_simulation(
        density_factors=[density],
        config=cfg,
        ablation=AblationConfig(),          # baseline – nothing disabled
        show_point_motion=False,
        save_outputs=False,
        rhythmicity_permutations=0,         # skip permutation test for speed
        rng=rng,
        history_callback=_callback,
    )

    return captured[density]


# ---------------------------------------------------------------------------
# 2.  Sliding-window cosinor fit per agent
# ---------------------------------------------------------------------------
def get_daily_phases(speeds: np.ndarray, period: int) -> dict:
    """
    For each agent, fit a sine in sliding 3-period windows and return the
    fitted phase (in radians, matching the original script's convention).

    Returns
    -------
    dict  mapping (agent_id, day_window_index) -> phase_radians
    """
    n_agents, n_timesteps = speeds.shape
    n_windows = int(n_timesteps / period) - 2       # 3-day sliding window

    phases = {}

    for agent_id in range(n_agents):
        ts = speeds[agent_id]
        for w in range(n_windows):
            start = period * w
            end   = period * (w + 3)
            y = ts[start:end]

            # Skip constant series
            if np.ptp(y) == 0:
                continue

            x = np.arange(start, end, dtype=float)
            sf = fit_sine(x, y, period)

            if not math.isnan(sf.y_offset):
                # Convert x_offset to a phase angle in [0, 2π)
                phase_rad = (sf.x_offset / period) * 2 * math.pi
                phases[(agent_id, w)] = phase_rad

    return phases


# ---------------------------------------------------------------------------
# 3.  Accumulate phase per spatial location
# ---------------------------------------------------------------------------
def accumulate_phase_per_location(
    positions: np.ndarray,
    phases: dict,
    period: int,
    grid_size: tuple[int, int],
    hive_bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    For every (agent, day-window) pair with a fitted phase, record that
    phase at every spatial bin the agent visited during that day.

    Uses circular mean (atan2 of mean sin/cos) to aggregate phases per bin.

    Returns (phase_grid, count_grid) where phase_grid values are in radians.
    """
    n_agents, n_timesteps, _ = positions.shape
    bins_y, bins_x = grid_size
    x_min, x_max, y_min, y_max = hive_bounds

    sin_sum = np.zeros((bins_y, bins_x))
    cos_sum = np.zeros((bins_y, bins_x))
    count   = np.zeros((bins_y, bins_x))

    x_edges = np.linspace(x_min, x_max, bins_x + 1)
    y_edges = np.linspace(y_min, y_max, bins_y + 1)

    for (agent_id, w), phase in phases.items():
        day_start = w * period
        day_end   = (w + 1) * period
        day_end   = min(day_end, n_timesteps)

        pos = positions[agent_id, day_start:day_end, :]
        bx = np.digitize(pos[:, 0], x_edges) - 1
        by = np.digitize(pos[:, 1], y_edges) - 1
        valid = (bx >= 0) & (bx < bins_x) & (by >= 0) & (by < bins_y)

        # Unique bins visited this day (avoid over-counting)
        visited = set(zip(by[valid], bx[valid]))
        for (r, c) in visited:
            sin_sum[r, c] += math.sin(phase)
            cos_sum[r, c] += math.cos(phase)
            count[r, c]   += 1

    with np.errstate(invalid="ignore"):
        phase_grid = np.arctan2(sin_sum, cos_sum)

    # Mask bins with too few samples
    phase_grid[count < 3] = np.nan

    return phase_grid, count


# ---------------------------------------------------------------------------
# 4.  Convert phase to "hour of peak activity" and plot
# ---------------------------------------------------------------------------
def phase_to_hour(phase_rad: np.ndarray, period: int) -> np.ndarray:
    """
    Convert phase (radians) to simulated hour-of-day.

    The simulation's day starts at timestep 0 with a sinusoidal driver
    sin(2π·t/T).  Peak driver is at t = T/4.
    We map phase → hour so that the driver peak corresponds to ~12:00 noon.
    """
    # Fraction of the day
    frac = phase_rad / (2 * math.pi)
    # Shift so phase=0 ≈ midnight → phase_peak ≈ noon
    hours = (frac * 24 + 6) % 24        # +6 to centre driver peak at 12
    return hours


def plot_figure_3b(
    phase_grid: np.ndarray,
    count_grid: np.ndarray,
    period: int,
    save_path: str | Path | None = None,
):
    """Produce a publication-quality heatmap in the style of Figure 3b."""

    hour_grid = phase_to_hour(phase_grid, period)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(
        hour_grid,
        ax=ax,
        cmap="viridis_r",
        cbar_kws={"label": "Hour of peak activity", "shrink": 0.75},
        xticklabels=10,
        yticklabels=10,
        square=True,
        robust=True,
        linewidths=0,
        rasterized=True,
    )
    ax.set_xlabel("x position (bins)")
    ax.set_ylabel("y position (bins)")
    ax.set_title("Simulated phase of peak activity (baseline)")
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path is None:
        save_path = Path(__file__).parent / "visualizations" / "figure_3b_simulated.pdf"

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    print(f"Saved figure to {save_path}")

    # Also save a PNG for quick preview
    png_path = str(save_path).replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Saved PNG to {png_path}")

    plt.close(fig)
    return save_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Reproducing Figure 3b (simulated phase heatmap)")
    print("=" * 60)

    density = 1.0
    print(f"\n[1/4] Running baseline simulation (density={density}) ...")
    history = run_baseline_and_capture(density=density, seed=42)

    speeds    = history["speeds"]           # (n_agents, sim_duration)
    positions = history["positions"]        # (n_agents, sim_duration, 2)
    period    = history["day_duration"]      # e.g. 400

    cfg = default_simulation_config()
    hive_bounds = cfg.env.hive_bounds       # [x_min, x_max, y_min, y_max]
    bins_y = cfg.env.bins_y
    bins_x = cfg.env.bins_x

    print(f"    Agents: {speeds.shape[0]}, Timesteps: {speeds.shape[1]}, Period: {period}")
    print(f"    Grid: {bins_y} × {bins_x}")

    print("\n[2/4] Fitting cosinor per agent in sliding windows ...")
    phases = get_daily_phases(speeds, period)
    print(f"    Fitted {len(phases)} (agent, day) phase values")

    print("\n[3/4] Accumulating phase per spatial bin ...")
    phase_grid, count_grid = accumulate_phase_per_location(
        positions, phases, period, (bins_y, bins_x), hive_bounds
    )
    n_covered = np.sum(~np.isnan(phase_grid))
    print(f"    Bins with valid phase: {n_covered} / {bins_y * bins_x}")

    print("\n[4/4] Plotting ...")
    save_path = plot_figure_3b(phase_grid, count_grid, period)

    print("\nDone!")


if __name__ == "__main__":
    main()
