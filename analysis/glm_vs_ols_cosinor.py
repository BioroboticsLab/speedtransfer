"""GLM (Gamma, log link) vs. OLS cosinor -- SI robustness check (Reviewer #2, major #3).

The reviewer notes the OLS cosinor assumes normal errors, violated for strictly-positive,
right-skewed speed data, and recommends a GLM (Doyle et al. 2022, PMID 35306404). We refit
each bee's rhythm with a Gamma-family GLM (log link) and compare against the OLS cosinor used
in the main text, to show the rhythmicity conclusions are unchanged: bees called rhythmic by
OLS are also called rhythmic by the GLM, and their acrophase (peak-time) estimates agree.

Each bee's velocity series is read from <DATA>/velocities_20_8-14_9_2019/<bee_id>.pickle
(datetime, velocity, time_passed). Fitting is on the full available series per bee, period 24 h.
Parallelised across cores for the full ~3400-bee run.

Run (on cirrus, in place):
    SPEEDTRANSFER_DATA_PATH=/mnt/local_storage/circadian/data \
    ~/circadian_venv/bin/python ~/circadian_analysis/glm_vs_ols_cosinor.py \
        --jobs 12 --csv ~/circadian_out/glm_vs_ols_2019.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f as f_dist

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cosinor_glm import fit_cosinor_glm  # noqa: E402  (Gamma GLM, log link)

PERIOD = 24.0
OMEGA = 2 * np.pi / PERIOD
MIN_POINTS = 50
ALPHA = 0.05
BIN_RULE = "10min"  # bin raw ~6 Hz velocity to 10-min means before fitting (matches the paper)


def data_dir() -> Path:
    base = os.environ.get("SPEEDTRANSFER_DATA_PATH", "/Users/tim/DEV/circadian_data")
    return Path(base) / "velocities_20_8-14_9_2019"


def load_bee(path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_pickle(path)
    s = df.set_index("datetime")["velocity"].resample(BIN_RULE).mean().dropna()
    t = (s.index - s.index[0]).total_seconds().to_numpy() / 3600.0
    return t, s.to_numpy(dtype=float)


def fit_ols_cosinor(t: np.ndarray, y: np.ndarray) -> dict:
    """OLS cosinor: y ~ 1 + cos(wt) + sin(wt); F-test for the joint rhythmic terms."""
    m = np.isfinite(t) & np.isfinite(y)
    t, y = t[m], y[m]
    n = y.size
    if n < MIN_POINTS or np.ptp(y) == 0:
        return dict(p_value=np.nan, acrophase_hours=np.nan, amplitude=np.nan, r_squared=np.nan, n=n)
    X = np.column_stack([np.ones(n), np.cos(OMEGA * t), np.sin(OMEGA * t)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    df1, df2 = 2, n - 3
    if ss_res <= 0 or df2 <= 0:
        p = 0.0 if ss_tot > 0 else np.nan
    else:
        fstat = ((ss_tot - ss_res) / df1) / (ss_res / df2)
        p = float(f_dist.sf(fstat, df1, df2))
    acro = np.arctan2(beta[2], beta[1]) % (2 * np.pi)
    return dict(p_value=p, acrophase_hours=acro / OMEGA, amplitude=float(np.hypot(beta[1], beta[2])),
                r_squared=r2, n=n)


def circ_diff_hours(a: float, b: float) -> float:
    d = (a - b) % PERIOD
    return min(d, PERIOD - d)


def _process_bee(path_str: str) -> dict | None:
    f = Path(path_str)
    try:
        t, v = load_bee(f)
    except Exception:
        return None
    ols = fit_ols_cosinor(t, v)
    glm = fit_cosinor_glm(t, v, period=PERIOD)
    return dict(bee=int(f.stem), n=ols["n"], p_ols=ols["p_value"], p_glm=glm["p_value"],
                acro_ols=ols["acrophase_hours"], acro_glm=glm.get("acrophase_hours", np.nan),
                r2_ols=ols["r_squared"])


def summarise(df: pd.DataFrame) -> None:
    df = df.dropna(subset=["p_ols", "p_glm"])
    n = len(df)
    if n == 0:
        print("no fittable bees")
        return
    sig_ols = df.p_ols < ALPHA
    sig_glm = df.p_glm < ALPHA
    both = sig_ols & sig_glm
    phase_diffs = np.array([circ_diff_hours(a, b) for a, b in zip(df.acro_ols[both], df.acro_glm[both])])
    print(f"bees compared: {n}")
    print(f"rhythmic @ {ALPHA}:  OLS {sig_ols.sum()}/{n} ({sig_ols.mean():.1%})   "
          f"GLM {sig_glm.sum()}/{n} ({sig_glm.mean():.1%})")
    print(f"significance agreement: {(sig_ols == sig_glm).mean():.1%}  "
          f"(both sig {both.sum()}, both non {(~sig_ols & ~sig_glm).sum()}, disagree {(sig_ols ^ sig_glm).sum()})")
    if phase_diffs.size:
        print(f"acrophase |diff| among both-significant: median {np.median(phase_diffs):.2f} h, "
              f"90th pct {np.percentile(phase_diffs, 90):.2f} h  (n={phase_diffs.size})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-bees", type=int, default=0, help="0 = all")
    ap.add_argument("--jobs", type=int, default=0, help="0 = cpu_count")
    ap.add_argument("--csv", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(data_dir().glob("*.pickle"), key=lambda p: int(p.stem))
    if args.max_bees:
        files = files[: args.max_bees]
    jobs = args.jobs or mp.cpu_count()
    print(f"fitting {len(files)} bees on {jobs} workers ...", flush=True)
    rows = []
    with mp.Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_process_bee, [str(f) for f in files], chunksize=4), 1):
            if r:
                rows.append(r)
            if i % 500 == 0:
                print(f"  ...{i}/{len(files)} processed ({len(rows)} fittable)", flush=True)
    df = pd.DataFrame(rows)
    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"saved per-bee comparison -> {args.csv}")
    summarise(df)


if __name__ == "__main__":
    main()
