"""
Generate MP4 animations that highlight speed transfer effects.

The first video tracks two agents (one rhythmic driver, one non-rhythmic) and
annotates speed jumps caused by interactions. Additional videos increase the
group sizes and plot mean speeds with confidence intervals to show the effect
at scale.
"""
import argparse
import math
from pathlib import Path
import os
import sys

TRANSFER_EPS = 1e-3

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / "visualizations" / ".matplotlib_cache")))

import matplotlib

# Headless backend so animations render without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, colors
import numpy as np

from simulation import (
    AblationConfig,
    compute_sine_summary,
    fit_sine,
    OutputOptions,
    SimulationParameters,
    default_simulation_config,
    run_simulation,
)

OUTPUT_DIR = Path(__file__).parent / "output"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render speed transfer videos with optional new seeds/configs.")
    parser.add_argument(
        "--randomize-seeds",
        action="store_true",
        help="Use fresh random seeds for all videos instead of the defaults.",
    )
    parser.add_argument("--pair-seed", type=int, help="Seed for the two-agent videos.")
    parser.add_argument("--pair-duration", type=int, default=240, help="Simulation duration for the two-agent videos.")
    parser.add_argument(
        "--require-pair-transfer",
        action="store_true",
        help="Retry the two-agent simulation with new seeds until at least one speed transfer occurs.",
    )
    parser.add_argument("--max-attempts", type=int, default=15, help="Max attempts when retrying for transfer.")
    parser.add_argument(
        "--mean-duration",
        type=int,
        default=240,
        help="Simulation duration for the larger group videos.",
    )
    parser.add_argument(
        "--mean-seeds",
        type=int,
        nargs="+",
        metavar="SEED",
        help="Seeds for the >1-agent-per-group videos (must match number of group sizes >1, unless omitted).",
    )
    parser.add_argument(
        "--group-sizes",
        type=int,
        nargs="+",
        default=(1, 25, 50),
        help="Group sizes (per group) to simulate; defaults to 1, 25, 50.",
    )
    return parser.parse_args(argv)


def _video_config(sim_duration: int) -> object:
    cfg = default_simulation_config()
    cfg.sim = SimulationParameters(day_duration=200, sim_duration=sim_duration)
    cfg.output = OutputOptions(save_mat=False, output_dir=OUTPUT_DIR)
    return cfg


def _simulate_with_history(density_factor: float, sim_duration: int, seed: int) -> dict:
    history: dict[float, dict] = {}

    def _capture(density: float, payload: dict) -> None:
        history[density] = payload

    cfg = _video_config(sim_duration)
    rng = np.random.default_rng(seed)
    run_simulation(
        density_factors=(density_factor,),
        config=cfg,
        ablation=AblationConfig(),
        show_point_motion=False,
        save_outputs=False,
        rhythmicity_permutations=0,
        rng=rng,
        history_callback=_capture,
    )
    hist = history[density_factor]
    hist["hive_bounds"] = cfg.env.hive_bounds
    t = np.arange(1, sim_duration + 1, dtype=float)
    driver_amp = cfg.env.speed_scale * cfg.env.speed_transfer_scale * cfg.group1.speed_fwd_mu * 0.5
    hist["driver_wave"] = driver_amp * (np.sin(2.0 * np.pi * t / cfg.sim.day_duration) + 1.0)
    hist["driver_period"] = cfg.sim.day_duration
    # Recompute sine summary excluding the early transient (first quarter day) so visuals and summary align.
    time_axis = np.arange(1, sim_duration + 1, dtype=float)
    valid_mask = time_axis >= (0.25 * cfg.sim.day_duration)
    g1_mean = hist["group_mean_speeds"]["group1"]
    g2_mean = hist["group_mean_speeds"]["group2"]
    hist["sine_summary"] = compute_sine_summary(
        time_axis,
        g1_mean,
        g2_mean,
        cfg.sim.day_duration,
        rhythmicity_permutations=0,
        rng=np.random.default_rng(seed),
        valid_mask=valid_mask,
    )
    hist["sine_valid_mask"] = valid_mask
    return hist


def _has_speed_transfer(history: dict) -> bool:
    transfers = history.get("speed_transfers")
    if transfers is None:
        return False
    idx_g2 = history["group2_indices"]
    return bool(np.any(transfers[idx_g2] > 0.0))


def _save_animation(anim: animation.FuncAnimation, path: Path, fps: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
    anim.save(path, writer=writer)
    plt.close(anim._fig)


def _draw_transfer_background(ax: plt.Axes, time_axis: np.ndarray, counts: np.ndarray):
    if counts is None or not np.any(counts):
        return None
    vmax = float(np.max(counts))
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("Oranges")
    for step, count in enumerate(counts):
        if count <= 0:
            continue
        color = cmap(norm(count))
        ax.axvspan(time_axis[step] - 0.5, time_axis[step] + 0.5, color=color, alpha=0.25, linewidth=0)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


def _first_max_time(period: float, phase_offset: float, start: float, end: float) -> float | None:
    """First sine peak within [start, end] given period and phase (offset in same units as x)."""
    if not np.isfinite(period) or period == 0 or not np.isfinite(phase_offset):
        return None
    t = phase_offset + period / 4.0
    while t < start:
        t += period
    while t - period >= start:
        t -= period
    if start <= t <= end:
        return t
    return None


def _arena_limits(positions: np.ndarray, hive_bounds: np.ndarray | None) -> tuple[tuple[float, float], tuple[float, float]]:
    if hive_bounds is not None:
        return (float(hive_bounds[0]), float(hive_bounds[1])), (float(hive_bounds[2]), float(hive_bounds[3]))

    pos_clean = np.nan_to_num(positions, nan=0.0)
    xy_min = pos_clean.min(axis=(0, 1))
    xy_max = pos_clean.max(axis=(0, 1))
    span = xy_max - xy_min
    margin = 0.1 * (span + 1e-6)
    return (float(xy_min[0] - margin[0]), float(xy_max[0] + margin[0])), (float(xy_min[1] - margin[1]), float(xy_max[1] + margin[1]))


def _init_position_speed_axes(
    xlim: tuple[float, float], ylim: tuple[float, float], figsize: tuple[int, int], position_title: str
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    fig, (ax_pos, ax_speed) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [2, 1]})
    ax_pos.set_xlim(xlim)
    ax_pos.set_ylim(ylim)
    ax_pos.set_aspect("equal", adjustable="box")
    ax_pos.grid(True, alpha=0.2)
    ax_pos.set_title(position_title)
    return fig, ax_pos, ax_speed


def render_two_agent_video(history: dict, output_path: Path) -> None:
    # Deprecated duplicate of render_two_agent_motion_and_speed; kept for backward compatibility.
    render_two_agent_motion_and_speed(history, output_path)


def render_two_agent_motion_and_speed(history: dict, output_path: Path) -> None:
    positions = history["positions"]
    speeds = history["speeds"]
    transfers = history.get("speed_transfers")
    idx_g1 = history["group1_indices"]
    idx_g2 = history["group2_indices"]
    hive_bounds = history.get("hive_bounds")

    time_axis = np.arange(1, speeds.shape[1] + 1, dtype=float)
    g1_speed = speeds[idx_g1[0]]
    g2_speed = speeds[idx_g2[0]]

    transfer_counts = (
        np.count_nonzero(transfers[idx_g2] > TRANSFER_EPS, axis=0) if transfers is not None else np.zeros_like(time_axis)
    )
    transfer_mask = transfer_counts > 0

    xlim, ylim = _arena_limits(positions, hive_bounds)

    fig, ax_pos, ax_speed = _init_position_speed_axes(
        xlim, ylim, figsize=(9, 6), position_title="Arena (two agents + trails)"
    )

    scat_g1 = ax_pos.scatter([], [], c="tab:blue", s=40, label="Group 1 (rhythmic)")
    scat_g2 = ax_pos.scatter([], [], c="tab:green", s=40, label="Group 2 (non-rhythmic)")
    trail_g1, = ax_pos.plot([], [], c="tab:blue", alpha=0.6, linewidth=1)
    trail_g2, = ax_pos.plot([], [], c="tab:green", alpha=0.6, linewidth=1)

    line_g1, = ax_speed.plot([], [], color="tab:blue", label="Group 1 speed")
    line_g2, = ax_speed.plot([], [], color="tab:green", label="Group 2 speed")
    bg = _draw_transfer_background(ax_speed, time_axis, transfer_counts)
    if bg is not None:
        cbar = plt.colorbar(bg, ax=ax_speed, pad=0.01)
        cbar.set_label("Speed transfers (agents)")

    ax_pos.set_xlim(xlim)
    ax_pos.set_ylim(ylim)
    ax_pos.legend(loc="upper right")

    ax_speed.set_xlim(time_axis[0], time_axis[-1])
    y_min = float(min(g1_speed.min(), g2_speed.min())) - 0.5
    y_max = float(max(g1_speed.max(), g2_speed.max())) + 0.5
    ax_speed.set_ylim(y_min, y_max)
    ax_speed.set_xlabel("Time step")
    ax_speed.set_ylabel("Speed")
    ax_speed.grid(True, alpha=0.2)
    ax_speed.legend(handles=[line_g1, line_g2], loc="upper right")

    trail_len = 15

    def _update(frame: int) -> None:
        upto = frame + 1
        g1_xy = positions[idx_g1[0], frame]
        g2_xy = positions[idx_g2[0], frame]
        transfer_now = transfer_mask[frame]

        scat_g1.set_offsets([g1_xy])
        scat_g2.set_offsets([g2_xy])
        highlight_color = colors.to_rgba("orange") if transfer_now else colors.to_rgba("tab:blue")
        scat_g1.set_facecolor(highlight_color)
        scat_g2.set_facecolor(colors.to_rgba("orange") if transfer_now else colors.to_rgba("tab:green"))

        start = max(0, frame - trail_len)
        trail_g1.set_data(positions[idx_g1[0], start:upto, 0], positions[idx_g1[0], start:upto, 1])
        trail_g2.set_data(positions[idx_g2[0], start:upto, 0], positions[idx_g2[0], start:upto, 1])

        line_g1.set_data(time_axis[:upto], g1_speed[:upto])
        line_g2.set_data(time_axis[:upto], g2_speed[:upto])

        ax_speed.set_title(f"Speeds (highlighted spans = speed transfer boosts), t={frame+1}")

    anim = animation.FuncAnimation(fig, _update, frames=speeds.shape[1], interval=40, repeat=False)
    _save_animation(anim, output_path)


def render_motion_and_mean_speed(history: dict, output_path: Path, title: str) -> None:
    speeds = history["speeds"]
    transfers = history.get("speed_transfers")
    idx_g1 = history["group1_indices"]
    idx_g2 = history["group2_indices"]
    positions = history["positions"]
    hive_bounds = history.get("hive_bounds")

    time_axis = np.arange(1, speeds.shape[1] + 1, dtype=float)
    g1 = speeds[idx_g1]
    g2 = speeds[idx_g2]

    def _stats(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / math.sqrt(max(1, arr.shape[0]))
        ci = 1.96 * sem
        return mean, mean - ci, mean + ci

    g1_mean, g1_low, g1_high = _stats(g1)
    g2_mean, g2_low, g2_high = _stats(g2)
    transfer_counts = (
        np.count_nonzero(transfers[idx_g2] > TRANSFER_EPS, axis=0) if transfers is not None else np.zeros_like(time_axis)
    )
    transfer_mask = transfer_counts > 0
    day_duration = float(history.get("day_duration", speeds.shape[1]))

    sine_summary = history.get("sine_summary")
    if sine_summary is not None:
        sine_fit_g1 = sine_summary["sine_fit_group1"]
        sine_fit_g2 = sine_summary["sine_fit_group2"]
    else:
        sine_fit_g1 = fit_sine(time_axis, g1_mean, day_duration)
        sine_fit_g2 = fit_sine(time_axis, g2_mean, day_duration)
    driver_wave = history.get("driver_wave")
    driver_period = float(history.get("driver_period", day_duration))
    if driver_wave is None:
        driver_wave = np.full_like(time_axis, np.nan, dtype=float)

    y_min = float(np.nanmin(np.vstack([g1_low, g2_low]))) - 0.5
    y_max = float(np.nanmax(np.vstack([g1_high, g2_high]))) + 0.5
    xlim, ylim = _arena_limits(positions, hive_bounds)

    fig, ax_pos, ax_speed = _init_position_speed_axes(
        xlim, ylim, figsize=(9, 6), position_title="Arena (mean-speed cohorts)"
    )

    scat_g1 = ax_pos.scatter([], [], c="tab:blue", s=12, alpha=0.7, label="Group 1 (rhythmic)")
    scat_g2 = ax_pos.scatter([], [], c="tab:green", s=12, alpha=0.7, label="Group 2 (non-rhythmic)")
    ax_pos.legend(loc="upper right")

    line_g1, = ax_speed.plot([], [], color="tab:blue", label="Group 1 mean")
    line_g2, = ax_speed.plot([], [], color="tab:green", label="Group 2 mean")
    line_fit_g1, = ax_speed.plot([], [], "--", color="tab:blue", alpha=0.8, label="Group 1 sine fit")
    line_fit_g2, = ax_speed.plot([], [], "--", color="tab:green", alpha=0.8, label="Group 2 sine fit")
    line_driver, = ax_speed.plot([], [], ":", color="0.3", linewidth=1.6, label="Ground truth driver")
    vline_g1 = ax_speed.axvline(color="tab:blue", linestyle="--", alpha=0.6, linewidth=1.5)
    vline_g2 = ax_speed.axvline(color="tab:green", linestyle="--", alpha=0.6, linewidth=1.5)
    vline_driver = ax_speed.axvline(color="0.3", linestyle=":", alpha=0.7, linewidth=1.2)
    for v in (vline_g1, vline_g2, vline_driver):
        v.set_visible(False)
    bg = _draw_transfer_background(ax_speed, time_axis, transfer_counts)
    if bg is not None:
        cbar = plt.colorbar(bg, ax=ax_speed, pad=0.01)
        cbar.set_label("Speed transfers (agents)")
    bands: list[matplotlib.collections.PolyCollection] = []
    ax_speed.set_xlim(time_axis[0], time_axis[-1])
    ax_speed.set_ylim(y_min, y_max)
    ax_speed.set_xlabel("Time step")
    ax_speed.set_ylabel("Speed")
    ax_speed.grid(True, alpha=0.2)
    ax_speed.legend(handles=[line_g1, line_g2, line_fit_g1, line_fit_g2, line_driver], loc="upper right")
    ax_speed.set_title(title)

    def _update(frame: int) -> None:
        upto = frame + 1
        scat_g1.set_offsets(positions[idx_g1, frame, :2].reshape(-1, 2))
        scat_g2.set_offsets(positions[idx_g2, frame, :2].reshape(-1, 2))
        if transfer_mask.any():
            active = transfers[idx_g2, frame] > TRANSFER_EPS
            base_color = colors.to_rgba("tab:green")
            highlight = colors.to_rgba("orange")
            facecolors = np.tile(base_color, (len(idx_g2), 1))
            facecolors[active] = highlight
            scat_g2.set_facecolors(facecolors)

        line_g1.set_data(time_axis[:upto], g1_mean[:upto])
        line_g2.set_data(time_axis[:upto], g2_mean[:upto])
        if frame + 1 == speeds.shape[1]:
            line_fit_g1.set_data(time_axis, sine_fit_g1(time_axis))
            line_fit_g2.set_data(time_axis, sine_fit_g2(time_axis))
            line_driver.set_data(time_axis, driver_wave)

            max_g1 = _first_max_time(sine_fit_g1.period, float(sine_fit_g1.x_offset), time_axis[0], time_axis[-1])
            max_g2 = _first_max_time(sine_fit_g2.period, float(sine_fit_g2.x_offset), time_axis[0], time_axis[-1])
            max_driver = _first_max_time(driver_period, 0.0, time_axis[0], time_axis[-1])
            for vline, t in ((vline_g1, max_g1), (vline_g2, max_g2), (vline_driver, max_driver)):
                if t is not None:
                    vline.set_xdata([t, t])
                    vline.set_visible(True)
                else:
                    vline.set_visible(False)
        else:
            line_fit_g1.set_data([], [])
            line_fit_g2.set_data([], [])
            line_driver.set_data([], [])
            for vline in (vline_g1, vline_g2, vline_driver):
                vline.set_visible(False)

        for band in bands:
            band.remove()
        bands.clear()
        bands.append(
            ax_speed.fill_between(
                time_axis[:upto], g1_low[:upto], g1_high[:upto], color="tab:blue", alpha=0.15, linewidth=0
            )
        )
        bands.append(
            ax_speed.fill_between(
                time_axis[:upto], g2_low[:upto], g2_high[:upto], color="tab:green", alpha=0.15, linewidth=0
            )
        )

    anim = animation.FuncAnimation(fig, _update, frames=speeds.shape[1], interval=35, repeat=False)
    _save_animation(anim, output_path)


def main() -> None:
    args = _parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(os.environ["MPLCONFIGDIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    seed_rng = np.random.default_rng() if args.randomize_seeds else None

    def _rand_seed(fallback: int) -> int:
        if seed_rng is not None:
            return int(seed_rng.integers(low=0, high=1_000_000_000))
        return fallback

    group_sizes = tuple(args.group_sizes)
    non_pair_sizes = [g for g in group_sizes if g != 1]

    pair_seed = args.pair_seed if args.pair_seed is not None else _rand_seed(2025)
    if non_pair_sizes:
        if args.mean_seeds is not None:
            mean_seeds = list(args.mean_seeds)
            if len(mean_seeds) != len(non_pair_sizes):
                raise ValueError("Number of mean_seeds must match number of group sizes > 1")
        else:
            mean_seeds = [_rand_seed(3030 + i * 1010) for i in range(len(non_pair_sizes))]
    else:
        mean_seeds = []

    density_by_group = {size: size / 500.0 for size in group_sizes}
    seed_by_group = {1: pair_seed}
    duration_by_group = {1: args.pair_duration}
    for size, seed in zip(non_pair_sizes, mean_seeds):
        seed_by_group[size] = seed
        duration_by_group[size] = args.mean_duration

    used_seeds: dict[int, int] = {}
    attempts = 0
    for g1_n in group_sizes:
        density = density_by_group[g1_n]
        seed = seed_by_group[g1_n]
        history: dict

        if g1_n == 1:
            while True:
                history = _simulate_with_history(density_factor=density, sim_duration=duration_by_group[g1_n], seed=seed)
                attempts += 1
                if not args.require_pair_transfer or _has_speed_transfer(history) or attempts >= args.max_attempts:
                    break
                seed = _rand_seed(_rand_seed(2025))
                seed_by_group[g1_n] = seed
            render_two_agent_motion_and_speed(history, OUTPUT_DIR / "speed_transfer_mean_1_per_group.mp4")
        else:
            history = _simulate_with_history(density_factor=density, sim_duration=duration_by_group[g1_n], seed=seed)
            title = f"Speed transfer (n={g1_n} per group, density={density:.3f})"
            render_motion_and_mean_speed(history, OUTPUT_DIR / f"speed_transfer_mean_{g1_n}_per_group.mp4", title)

        used_seeds[g1_n] = seed_by_group[g1_n]

    print(
        f"Wrote speed-transfer videos to {OUTPUT_DIR}\n"
        f"Seeds by group size: {used_seeds}, single-agent attempts={attempts}"
    )


if __name__ == "__main__":
    main()
