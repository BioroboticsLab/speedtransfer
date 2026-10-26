"""Day/night comparison of the age-vs-distance-from-entrance distribution
(Reviewer #2, Minor comment 2) -- STREAMING version.

The 2019 interaction pickle is 27 GB (won't fit in RAM), so we stream the CSV
(interactions_side{side}_{year}.csv, 41 GB) reading only the 7 needed columns in
chunks and accumulate streaming statistics:
  * dist x age count histograms (day / night) -> exact binned median age vs distance
  * OLS sufficient statistics (X'X, X'y, y'y, n) for age ~ dist * is_night (exact)
  * per-cell mean-age grids (day / night) -> 2-D map of the night-day difference

Each interaction contributes BOTH partners' start position + age + time. Interaction
time is converted to Berlin local time and split into day (09:00-18:00) and night
(21:00-06:00), exactly as the manuscript Methods define day/night.

Run on cirrus:
  MPLCONFIGDIR=... TMPDIR=... /home/beesbook/circadian_venv/bin/python -u daynight_stream.py --year 2019
"""
from __future__ import annotations
import argparse, math, os, sys, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ZEN = os.environ.get("SPEEDTRANSFER_INTERACTIONS", "/mnt/trove/storage/beesbook/speedtransfer/data/zenodo")  # see README
EXIT_POS = {2016: (0, 250), 2019: (5, 264)}
COLONY = {2016: "A", 2019: "B"}
DAY = (9, 18)     # [09:00, 18:00)
NIGHT = (21, 6)   # [21:00, 24:00) U [00:00, 06:00)
USECOLS = ["interaction_start", "x_pos_start_bee0", "y_pos_start_bee0", "age_bee0",
           "x_pos_start_bee1", "y_pos_start_bee1", "age_bee1"]

# fixed histogram supports (clip outliers; report clip fraction)
DMAX, DBINS = 800.0, 800            # dist axis: 1-unit bins over [0, 800]
AMAX = 90                            # age axis: integer days 0..90
NMAPX = NMAPY = 45                   # 2-D map grid


def berlin_hour(s: pd.Series) -> np.ndarray:
    dt = pd.to_datetime(s, utc=True, errors="coerce").dt.tz_convert("Europe/Berlin")
    return dt.dt.hour.to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2019)
    ap.add_argument("--side", type=int, default=0)
    ap.add_argument("--exit-x", type=float, default=None, help="override entrance x")
    ap.add_argument("--exit-y", type=float, default=None, help="override entrance y")
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--maxrows", type=int, default=0, help="0 = all (debug cap)")
    ap.add_argument("--out", default=os.path.join(os.environ.get("REVISION_OUT", "."), "daynight_{year}.pdf"))
    ap.add_argument("--csv", default=os.path.join(os.environ.get("REVISION_OUT", "."), "daynight_{year}.csv"))
    args = ap.parse_args()
    year, side = args.year, args.side
    ex, ey = EXIT_POS[year]
    if args.exit_x is not None and args.exit_y is not None:
        ex, ey = args.exit_x, args.exit_y
        print(f"OVERRIDE exit_pos -> ({ex}, {ey})", flush=True)
    path = f"{ZEN}/interactions_side{side}_{year}.csv"
    print(f"streaming {path} ({round(os.path.getsize(path)/1e9,1)} GB) chunk={args.chunk:,}", flush=True)

    # map grid support from exit + generous comb size (positions ~0..~500)
    xb = np.linspace(-50, 550, NMAPX + 1)
    yb = np.linspace(-50, 550, NMAPY + 1)

    # accumulators
    hist = {"day": np.zeros((DBINS, AMAX + 1)), "night": np.zeros((DBINS, AMAX + 1))}
    mapsum = {"day": np.zeros((NMAPX, NMAPY)), "night": np.zeros((NMAPX, NMAPY))}
    mapcnt = {"day": np.zeros((NMAPX, NMAPY)), "night": np.zeros((NMAPX, NMAPY))}
    XtX = np.zeros((4, 4)); Xty = np.zeros(4); yty = 0.0; n_ols = 0
    n_total = n_day = n_night = n_twi = n_clip = 0
    dmin, dmax = np.inf, -np.inf
    t0 = time.time(); seen = 0

    reader = pd.read_csv(path, usecols=USECOLS, chunksize=args.chunk)
    for ci, chunk in enumerate(reader):
        # stack both partners into long arrays
        h = berlin_hour(chunk["interaction_start"])
        x = np.concatenate([chunk["x_pos_start_bee0"].to_numpy(float), chunk["x_pos_start_bee1"].to_numpy(float)])
        y = np.concatenate([chunk["y_pos_start_bee0"].to_numpy(float), chunk["y_pos_start_bee1"].to_numpy(float)])
        age = np.concatenate([chunk["age_bee0"].to_numpy(float), chunk["age_bee1"].to_numpy(float)])
        hh = np.concatenate([h, h])
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(age) & np.isfinite(hh)
        x, y, age, hh = x[m], y[m], age[m], hh[m]
        dist = np.hypot(x - ex, y - ey)
        if dist.size:
            dmin = min(dmin, float(dist.min())); dmax = max(dmax, float(dist.max()))
        day = (hh >= DAY[0]) & (hh < DAY[1])
        night = (hh >= NIGHT[0]) | (hh < NIGHT[1])
        n_total += x.size; n_day += int(day.sum()); n_night += int(night.sum())
        n_twi += int((~day & ~night).sum())

        for lab, sel in (("day", day), ("night", night)):
            if not sel.any():
                continue
            dd = dist[sel]; aa = age[sel]; xx = x[sel]; yy = y[sel]
            n_clip += int((dd >= DMAX).sum())
            di = np.clip((dd / DMAX * DBINS).astype(int), 0, DBINS - 1)
            ai = np.clip(np.round(aa).astype(int), 0, AMAX)
            np.add.at(hist[lab], (di, ai), 1)
            ix = np.clip(np.digitize(xx, xb) - 1, 0, NMAPX - 1)
            iy = np.clip(np.digitize(yy, yb) - 1, 0, NMAPY - 1)
            np.add.at(mapsum[lab], (ix, iy), aa)
            np.add.at(mapcnt[lab], (ix, iy), 1)
            # OLS sufficient stats
            isn = 0.0 if lab == "day" else 1.0
            Xr = np.column_stack([np.ones_like(dd), dd, np.full_like(dd, isn), dd * isn])
            XtX += Xr.T @ Xr; Xty += Xr.T @ aa; yty += float(aa @ aa); n_ols += dd.size

        seen += len(chunk)
        if (ci + 1) % 5 == 0:
            print(f"  chunk {ci+1}: rows~{seen:,}  day={n_day:,} night={n_night:,}  {time.time()-t0:.0f}s", flush=True)
        if args.maxrows and seen >= args.maxrows:
            print("  (hit --maxrows cap)", flush=True); break

    print(f"\nDONE reading: pairs day={n_day:,} night={n_night:,} twilight(excluded)={n_twi:,}", flush=True)
    print(f"dist range: [{dmin:.1f}, {dmax:.1f}]  clipped(>= {DMAX})={n_clip:,}", flush=True)

    # ---- binned median age vs distance from the histograms ----
    NDB = 12
    edges = np.linspace(0, min(dmax, DMAX), NDB + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    ages = np.arange(AMAX + 1)
    def quantiles_from_hist(hmat, qs=(0.25, 0.5, 0.75)):
        # rebin fine dist bins (width DMAX/DBINS) into NDB coarse bins, then per-bin age quantiles
        fine_ctr = (np.arange(DBINS) + 0.5) * (DMAX / DBINS)
        coarse = np.clip(np.digitize(fine_ctr, edges) - 1, 0, NDB - 1)
        out = {q: np.full(NDB, np.nan) for q in qs}
        cnt = np.zeros(NDB)
        for cb in range(NDB):
            rows = hmat[coarse == cb].sum(axis=0)   # age distribution in this dist bin
            tot = rows.sum(); cnt[cb] = tot
            if tot == 0:
                continue
            cdf = np.cumsum(rows) / tot
            for q in qs:
                out[q][cb] = ages[np.searchsorted(cdf, q)]
        return out, cnt
    dq, dcnt = quantiles_from_hist(hist["day"])
    nq, ncnt = quantiles_from_hist(hist["night"])

    # ---- OLS solve from sufficient stats ----
    beta = np.linalg.solve(XtX, Xty)
    rss = yty - beta @ Xty
    sigma2 = rss / (n_ols - 4)
    cov = sigma2 * np.linalg.inv(XtX)
    se = np.sqrt(np.diag(cov))
    tvals = beta / se
    pvals = [math.erfc(abs(t) / math.sqrt(2.0)) for t in tvals]
    names = ["Intercept", "dist", "is_night", "dist:is_night"]
    print("\n=== OLS  age ~ dist * is_night  (exact, streamed; normal-approx p) ===", flush=True)
    print(f"{'term':>14} {'coef':>12} {'se':>12} {'t':>10} {'p':>12}", flush=True)
    for nm, b, s, t, p in zip(names, beta, se, tvals, pvals):
        print(f"{nm:>14} {b:12.5f} {s:12.5f} {t:10.2f} {p:12.2e}", flush=True)
    print(f"\nday slope (age-days per unit distance): {beta[1]:.4f}", flush=True)
    print(f"night vs day slope difference (dist:is_night): {beta[3]:.4f}  p={pvals[3]:.2e}  n={n_ols:,}", flush=True)

    # near/deep median age day vs night (from histograms)
    def med_age_over(hmat, dlo, dhi):
        fine_ctr = (np.arange(DBINS) + 0.5) * (DMAX / DBINS)
        sel = (fine_ctr >= dlo) & (fine_ctr < dhi)
        rows = hmat[sel].sum(axis=0); tot = rows.sum()
        if tot == 0: return np.nan
        return ages[np.searchsorted(np.cumsum(rows) / tot, 0.5)]
    q1d, q3d = edges[3], edges[-1]  # rough near (<=~Q of bins) vs deep
    print(f"\nmedian age near-entrance (<{edges[3]:.0f}):  day={med_age_over(hist['day'],0,edges[3]):.1f}  night={med_age_over(hist['night'],0,edges[3]):.1f}", flush=True)
    print(f"median age deep (>{edges[-4]:.0f}):        day={med_age_over(hist['day'],edges[-4],DMAX):.1f}  night={med_age_over(hist['night'],edges[-4],DMAX):.1f}", flush=True)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    axes[0].plot(ctr, dq[0.5], "-o", color="darkorange", label=f"day 09–18 (n={n_day:,})")
    axes[0].fill_between(ctr, dq[0.25], dq[0.75], color="darkorange", alpha=0.15)
    axes[0].plot(ctr, nq[0.5], "-o", color="navy", label=f"night 21–06 (n={n_night:,})")
    axes[0].fill_between(ctr, nq[0.25], nq[0.75], color="navy", alpha=0.15)
    axes[0].set(xlabel="distance from entrance", ylabel="median bee age [days]",
                title=f"(a) Age vs distance, day vs night — colony {COLONY[year]}")
    axes[0].legend(frameon=False, fontsize=8)

    with np.errstate(invalid="ignore"):
        md = mapsum["day"] / np.where(mapcnt["day"] > 20, mapcnt["day"], np.nan)
        mn = mapsum["night"] / np.where(mapcnt["night"] > 20, mapcnt["night"], np.nan)
    diff = mn - md
    im = axes[1].imshow(diff.T, origin="lower", cmap="coolwarm",
                        extent=[xb[0], xb[-1], yb[0], yb[-1]], aspect="auto", vmin=-6, vmax=6)
    axes[1].plot([ex], [ey], "k*", ms=13)
    axes[1].set(title="(b) Mean age: night − day (per comb cell)", xlabel="x", ylabel="y")
    fig.colorbar(im, ax=axes[1], label="Δ age [days]")
    fig.tight_layout()
    out = args.out.format(year=year)
    fig.savefig(out, bbox_inches="tight")
    print("saved figure ->", out, flush=True)

    pd.DataFrame({"dist_center": ctr, "n_day": dcnt, "age_med_day": dq[0.5], "age_q1_day": dq[0.25],
                  "age_q3_day": dq[0.75], "n_night": ncnt, "age_med_night": nq[0.5],
                  "age_q1_night": nq[0.25], "age_q3_night": nq[0.75]}).to_csv(args.csv.format(year=year), index=False)
    print("saved csv ->", args.csv.format(year=year), flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
