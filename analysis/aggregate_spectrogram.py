"""Aggregate Lomb-Scargle spectral analysis -- estimate the *true* rhythm period
(Reviewer #2, major #2).

The reviewer notes the paper assumes a 24 h period, whereas a hallmark of circadian
clocks is a period close to but not exactly 24 h -- and socially entrained nurse bees
might deviate. They ask us to use spectral methods to estimate the true period.

For each bee we compute a Lomb-Scargle periodogram of movement speed vs. time (uneven
sampling -> Lomb-Scargle is appropriate) on a FIXED period grid, so periodograms can be
averaged across individuals. We report:
  * the aggregate (mean) periodogram -- combining bees sharpens the population estimate;
  * the distribution of per-bee circadian peak periods (with the caveat that individual
    estimates are resolution-limited to ~P^2/T for a record of length T);
  * a per-bee CSV (peak period, power, false-alarm probability) for later age splitting.

Run (ephemeral env, matching pandas so the pickles read):
    uv run --no-project --with "pandas==2.2.3" --with "numpy==1.26.4" --with astropy \
        --with matplotlib python analysis/aggregate_spectrogram.py --max-bees 400
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

REPO = Path(__file__).resolve().parents[1]
PERIOD_GRID = np.linspace(4.0, 48.0, 800)          # hours
FREQ_GRID = 1.0 / PERIOD_GRID                        # cycles / hour
CIRCADIAN = (18.0, 30.0)                             # band for the "circadian peak"
MIN_POINTS = 200                                     # 10-min bins: ~1.4 d of coverage
MIN_SPAN_H = 72.0                                    # >= 3 days
FAP_THRESHOLD = 0.01                                 # significant rhythm


def data_dir() -> Path:
    base = os.environ.get("SPEEDTRANSFER_DATA_PATH", "/Users/tim/DEV/circadian_data")
    return Path(base) / "velocities_20_8-14_9_2019"


def bee_periodogram(t: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict] | None:
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    if y.size < MIN_POINTS or np.ptp(t) < MIN_SPAN_H or np.ptp(y) == 0:
        return None
    ls = LombScargle(t, y)
    power = ls.power(FREQ_GRID)                       # 'standard' normalization, in [0, 1]
    band = (PERIOD_GRID >= CIRCADIAN[0]) & (PERIOD_GRID <= CIRCADIAN[1])
    i_peak = np.where(band)[0][np.argmax(power[band])]
    peak_power = float(power[i_peak])
    info = dict(
        peak_period_h=float(PERIOD_GRID[i_peak]),
        peak_power=peak_power,
        fap=float(ls.false_alarm_probability(peak_power)),
        n=int(y.size),
        span_days=float(np.ptp(t) / 24.0),
        period_resolution_h=float(PERIOD_GRID[i_peak] ** 2 / np.ptp(t)),  # ~P^2/T
    )
    return power, info


def _process_bee(path_str: str):
    f = Path(path_str)
    try:
        df = pd.read_pickle(f)
        s = df.set_index("datetime")["velocity"].resample("10min").mean().dropna()
        t = (s.index - s.index[0]).total_seconds().to_numpy() / 3600.0
        res = bee_periodogram(t, s.to_numpy(dtype=float))
    except Exception:
        return None
    if res is None:
        return None
    power, info = res
    info["bee"] = int(f.stem)
    return power.tolist(), info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bees", type=int, default=0, help="0 = all")
    ap.add_argument("--jobs", type=int, default=0, help="0 = cpu_count")
    ap.add_argument("--out", type=Path, default=REPO / "aggregated_results" / "aggregate_spectrogram.pdf")
    ap.add_argument("--csv", type=Path, default=REPO / "aggregated_results" / "bee_peak_periods.csv")
    args = ap.parse_args()

    files = sorted(data_dir().glob("*.pickle"), key=lambda p: int(p.stem))
    if args.max_bees:
        files = files[: args.max_bees]
    jobs = args.jobs or mp.cpu_count()
    print(f"analysing {len(files)} bees on {jobs} workers ...", flush=True)
    results = []
    with mp.Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_bee, [str(f) for f in files], chunksize=4), 1):
            if r:
                results.append(r)
            if i % 500 == 0:
                print(f"  ...{i}/{len(files)} processed ({len(results)} usable)", flush=True)
    rows: list[dict] = [info for _p, info in results]
    powers: list[np.ndarray] = [np.array(p) for p, info in results if info["fap"] < FAP_THRESHOLD]

    meta = pd.DataFrame(rows)
    meta.to_csv(args.csv, index=False)
    if not powers:
        print("no bees with a significant rhythm yet")
        return

    sig = meta[meta.fap < FAP_THRESHOLD]
    mean_power = np.mean(powers, axis=0)
    agg_peak = PERIOD_GRID[(PERIOD_GRID >= CIRCADIAN[0]) & (PERIOD_GRID <= CIRCADIAN[1])][
        np.argmax(mean_power[(PERIOD_GRID >= CIRCADIAN[0]) & (PERIOD_GRID <= CIRCADIAN[1])])
    ]
    pk = sig.peak_period_h.to_numpy()
    print(f"bees analysed: {len(meta)}   with significant rhythm (FAP<{FAP_THRESHOLD}): {len(sig)}")
    print(f"AGGREGATE periodogram circadian peak: {agg_peak:.2f} h")
    print(f"per-bee circadian peak period: median {np.median(pk):.2f} h  "
          f"IQR [{np.percentile(pk,25):.2f}, {np.percentile(pk,75):.2f}]  "
          f"within 23-25h: {np.mean((pk>=23)&(pk<=25)):.0%}")
    print(f"typical per-bee period resolution ~{np.median(sig.period_resolution_h):.2f} h "
          f"(median span {np.median(sig.span_days):.1f} d)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(PERIOD_GRID, mean_power, color="steelblue")
    axes[0].axvline(24, color="crimson", ls=":", label="24 h")
    axes[0].axvline(agg_peak, color="black", ls="--", lw=1, label=f"peak {agg_peak:.1f} h")
    axes[0].set(xlabel="period [h]", ylabel="mean Lomb-Scargle power",
                title=f"(a) Aggregate periodogram (n={len(sig)} rhythmic bees)")
    axes[0].legend()
    axes[1].hist(pk, bins=np.arange(18, 30.5, 0.5), color="steelblue", edgecolor="white")
    axes[1].axvline(24, color="crimson", ls=":", label="24 h")
    axes[1].axvline(np.median(pk), color="black", ls="--", lw=1, label=f"median {np.median(pk):.1f} h")
    axes[1].set(xlabel="per-bee circadian peak period [h]", ylabel="# bees",
                title="(b) Distribution of individual peak periods")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"saved figure -> {args.out}")
    print(f"saved per-bee peaks -> {args.csv}")


if __name__ == "__main__":
    main()
