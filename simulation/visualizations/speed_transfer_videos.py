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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", str((PROJECT_ROOT / "visualizations" / ".matplotlib_cache")))

import matplotlib

# Headless backend so animations render without a GUI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation, colors
from matplotlib.patches import Patch
import numpy as np

from simulation import (
    AblationConfig,
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
    cfg.sim = SimulationParameters(day_duration=120, sim_duration=sim_duration)
    cfg.env.bins_x = 24
    cfg.env.bins_y = 16
    cfg.env.comb_size_x = 80.0
    cfg.env.comb_size_y = 50.0
    cfg.env.interaction_threshold_factor = 0.35
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


def _draw_transfer_spans(ax: plt.Axes, time_axis: np.ndarray, transfer_mask: np.ndarray, upto: int) -> None:
    active_steps = np.nonzero(transfer_mask[:upto])[0]
    for step in active_steps:
        ax.axvspan(time_axis[step] - 0.5, time_axis[step] + 0.5, color="orange", alpha=0.2, linewidth=0)


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

    transfer_mask = np.any(transfers[idx_g2] > 0.0, axis=0) if transfers is not None else np.zeros_like(time_axis, dtype=bool)
    transfer_intervals = []
    if transfer_mask.any():
        active = np.concatenate([[False], transfer_mask, [False]])
        starts = np.where(~active[:-1] & active[1:])[0]
        ends = np.where(active[:-1] & ~active[1:])[0]
        transfer_intervals = list(zip(starts, ends))

    xlim, ylim = _arena_limits(positions, hive_bounds)

    fig, ax_pos, ax_speed = _init_position_speed_axes(
        xlim, ylim, figsize=(9, 6), position_title="Arena (two agents + trails)"
    )

    scat_g1 = ax_pos.scatter([], [], c="tab:blue", s=40, label="Group 1 (rhythmic)")
    scat_g2 = ax_pos.scatter([], [], c="tab:green", s=40, label="Group 2 (non-rhythmic)")
    trail_g1, = ax_pos.plot([], [], c="tab:blue", alpha=0.6, linewidth=1)
    trail_g2, = ax_pos.plot([], [], c="tab:green", alpha=0.6, linewidth=1)

    for start, end in transfer_intervals:
        ax_speed.axvspan(time_axis[start] - 0.5, time_axis[end - 1] + 0.5, color="orange", alpha=0.2, linewidth=0)

    line_g1, = ax_speed.plot([], [], color="tab:blue", label="Group 1 speed")
    line_g2, = ax_speed.plot([], [], color="tab:green", label="Group 2 speed")
    transfer_patch = Patch(color="orange", alpha=0.2, label="Speed transfer")

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
    ax_speed.legend(handles=[line_g1, line_g2, transfer_patch], loc="upper right")

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
    transfer_mask = np.any(transfers[idx_g2] > 0.0, axis=0) if transfers is not None else np.zeros_like(time_axis, dtype=bool)

    y_min = float(np.nanmin(np.vstack([g1_low, g2_low]))) - 0.5
    y_max = float(np.nanmax(np.vstack([g1_high, g2_high]))) + 0.5
    xlim, ylim = _arena_limits(positions, hive_bounds)

    fig, ax_pos, ax_speed = _init_position_speed_axes(
        xlim, ylim, figsize=(9, 6), position_title="Arena (mean-speed cohorts)"
    )

    scat_g1 = ax_pos.scatter([], [], c="tab:blue", s=12, alpha=0.7, label="Group 1 (rhythmic)")
    scat_g2 = ax_pos.scatter([], [], c="tab:green", s=12, alpha=0.7, label="Group 2 (non-rhythmic)")
    ax_pos.legend(loc="upper right")

    for step in np.nonzero(transfer_mask)[0]:
        ax_speed.axvspan(time_axis[step] - 0.5, time_axis[step] + 0.5, color="orange", alpha=0.15, linewidth=0)

    line_g1, = ax_speed.plot([], [], color="tab:blue", label="Group 1 mean")
    line_g2, = ax_speed.plot([], [], color="tab:green", label="Group 2 mean")
    transfer_patch = Patch(color="orange", alpha=0.15, label="Speed transfer")
    bands: list[matplotlib.collections.PolyCollection] = []
    ax_speed.set_xlim(time_axis[0], time_axis[-1])
    ax_speed.set_ylim(y_min, y_max)
    ax_speed.set_xlabel("Time step")
    ax_speed.set_ylabel("Speed")
    ax_speed.grid(True, alpha=0.2)
    ax_speed.legend(handles=[line_g1, line_g2, transfer_patch], loc="upper right")
    ax_speed.set_title(title)

    def _update(frame: int) -> None:
        upto = frame + 1
        scat_g1.set_offsets(positions[idx_g1, frame, :2].reshape(-1, 2))
        scat_g2.set_offsets(positions[idx_g2, frame, :2].reshape(-1, 2))
        if transfer_mask.any():
            active = transfers[idx_g2, frame] > 0.0
            base_color = colors.to_rgba("tab:green")
            highlight = colors.to_rgba("orange")
            facecolors = np.tile(base_color, (len(idx_g2), 1))
            facecolors[active] = highlight
            scat_g2.set_facecolors(facecolors)

        line_g1.set_data(time_axis[:upto], g1_mean[:upto])
        line_g2.set_data(time_axis[:upto], g2_mean[:upto])

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
