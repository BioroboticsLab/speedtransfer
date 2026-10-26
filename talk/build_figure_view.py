#!/usr/bin/env python3
"""Build a single self-contained HTML page showing every talk figure.

Reads ``template.html``, converts each source figure to a compact web format,
and inlines it as a data URI so the output is one portable file that runs
offline with no external requests.

Usage:
    python3 talk/build_figure_view.py

Re-run after any figure is re-plotted.
"""

import base64
import mimetypes
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parent.parent
TALK = REPO / "talk"
TEMPLATE = TALK / "template.html"
OUTPUT = TALK / "speed_transfer_talk_figures.html"

MAX_WIDTH = 1700
WEBP_QUALITY = 88
VIDEO_FPS = 3  # 24 frames = one simulated day -> ~8 s loop

# key -> path relative to the repository root
STILLS = {
    "overview": "figures/imgs/overview_figure.png",
    "fig1": "speed_transfer_paper/figure_1_colony_B.png",
    "fig_s1a": "speed_transfer_paper/figure_S1a.png",
    "fig_s1c": "speed_transfer_paper/figure_S1c.png",
    "fig_s1d": "speed_transfer_paper/figure_S1d.png",
    "fig_s1e": "speed_transfer_paper/figure_S1e.png",
    "fig2": "speed_transfer_paper/figure_2_colony_B.png",
    "fig_s2": "speed_transfer_paper/figure_s2.png",
    "fig_s4": "speed_transfer_paper/figure_s4.png",
    "fig_s5": "speed_transfer_paper/figure_s5.png",
    "fig3": "speed_transfer_paper/figure_3_colony_B.png",
    "fig3b_sim": "simulation/visualizations/figure_3b_simulated.png",
    "abl_single": "simulation/visualizations/ablation_single_factor.png",
    "abl_factorial": "simulation/visualizations/ablation_factorial_heatmap.png",
    "sens_density": "simulation/visualizations/sensitivity_density.png",
    "sens_decay": "simulation/visualizations/sensitivity_decay.png",
    "sens_day": "simulation/visualizations/sensitivity_day_duration.png",
    "sweep_dd_amp": "simulation/visualizations/sweep_density_decay_amp.png",
    "sweep_dd_phase": "simulation/visualizations/sweep_density_decay_phase.png",
    "sweep_du_amp": "simulation/visualizations/sweep_density_duration_amp.png",
    "sweep_du_phase": "simulation/visualizations/sweep_density_duration_phase.png",
}

ANIMATIONS = {
    "vid_fit": "figures/animations/flow_fit.gif",
    "vid_interactions": "figures/animations/flow_n_interactions.gif",
    "vid_velchange": "figures/animations/flow_vel_change.gif",
}


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def convert_still(src: Path, dst: Path) -> None:
    image = Image.open(src).convert("RGB")
    width, height = image.size
    if width > MAX_WIDTH:
        image = image.resize((MAX_WIDTH, round(height * MAX_WIDTH / width)), Image.LANCZOS)
    image.save(dst, "WEBP", quality=WEBP_QUALITY, method=6)


def convert_animation(src: Path, dst: Path) -> None:
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", str(src),
            "-vf", f"fps={VIDEO_FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(dst),
        ],
        check=True,
    )


def main() -> int:
    if not TEMPLATE.exists():
        print(f"error: missing template at {TEMPLATE}", file=sys.stderr)
        return 1
    if not shutil.which("ffmpeg"):
        print("error: ffmpeg is required to convert the animations", file=sys.stderr)
        return 1

    html = TEMPLATE.read_text(encoding="utf-8")
    missing = []

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        for key, rel in {**STILLS, **ANIMATIONS}.items():
            src = REPO / rel
            token = f"__ASSET_{key}__"
            if token not in html:
                print(f"  skip   {key:16s} (no placeholder in template)")
                continue
            if not src.exists():
                missing.append(rel)
                continue

            if key in STILLS:
                dst = work / f"{key}.webp"
                convert_still(src, dst)
            else:
                dst = work / f"{key}.mp4"
                convert_animation(src, dst)

            html = html.replace(token, data_uri(dst))
            print(f"  embed  {key:16s} {dst.stat().st_size / 1024:8.0f} KB  <- {rel}")

    leftover = [line for line in html.splitlines() if "__ASSET_" in line]
    if missing:
        print("\nerror: source figures not found:", file=sys.stderr)
        for rel in missing:
            print(f"  {rel}", file=sys.stderr)
        return 1
    if leftover:
        print(f"\nerror: {len(leftover)} placeholder(s) left unfilled", file=sys.stderr)
        return 1

    OUTPUT.write_text(html, encoding="utf-8")
    print(f"\nwrote {OUTPUT.relative_to(REPO)}  ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
