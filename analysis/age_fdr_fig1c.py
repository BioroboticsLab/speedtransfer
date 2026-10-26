"""Fig 1c per-age-group rhythmic proportions, before/after BH-FDR (Reviewer #2, major #3).

Merges per-bee-day cosinor fits with the bee birth (tagging/introduction) dates and reports the
fraction of significantly rhythmic bee-days per 5-day age cohort, with and without Benjamini-
Hochberg correction across all fits -- i.e. whether the Fig 1c proportions survive multiple-testing
correction.

Age source: the *bayesian alive-bees* table (`annotated_tagged_date`), i.e. age is measured from
when the bee was introduced to the hive -- NOT the bb_utils hatch date. This reproduces the paper's
Fig 1c baseline (colony A 2016 youngest cohort 31.2% vs paper's 29%; the hatch date gives 62.6%).
2016 = colony A (Zenodo 4438013); 2019 (colony B) needs the equivalent alive_bees_2019_bayesian.

Run:
    ~/circadian_venv/bin/python age_fdr_fig1c.py \
        --cosinor ~/circadian_data/cosinor_2016.csv \
        --alive   ~/circadian_data/meta/alive_bees_bayesian.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALPHA = 0.05
COHORT = 5  # day age bins


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cosinor", type=Path, required=True)
    ap.add_argument("--alive", type=Path, required=True,
                    help="alive_bees_bayesian CSV with bee_id, annotated_tagged_date")
    args = ap.parse_args()

    d = pd.read_csv(args.cosinor, usecols=["bee_id", "date", "p_value"]).dropna(subset=["p_value"])
    d["date"] = pd.to_datetime(d["date"], utc=True, errors="coerce").dt.tz_localize(None)

    a = pd.read_csv(args.alive)
    a["tag"] = pd.to_datetime(a["annotated_tagged_date"], errors="coerce")
    d = d.merge(a[["bee_id", "tag"]], on="bee_id", how="left")
    d["age"] = (d["date"] - d["tag"]).dt.total_seconds() / 86400.0
    d = d.dropna(subset=["age"])
    d = d[d["age"] >= 0]

    d["p_bh"] = bh_adjust(d["p_value"].to_numpy())
    d["cohort"] = (d["age"] // COHORT * COHORT).astype(int)

    print(f"bee-day fits with age: {len(d)}  (unique bees {d.bee_id.nunique()})")
    print(f"overall rhythmic @ {ALPHA}: raw {(d.p_value < ALPHA).mean():.1%}  ->  BH {(d.p_bh < ALPHA).mean():.1%}")
    print(f"\n{'age cohort':>12}  {'n':>7}  {'% rhythmic raw':>15}  {'% rhythmic BH':>14}")
    for coh, g in d.groupby("cohort"):
        if coh > 40:
            continue
        print(f"{coh:>3}-{coh + COHORT - 1:<3} days  {len(g):>7}  "
              f"{(g.p_value < ALPHA).mean():>14.1%}  {(g.p_bh < ALPHA).mean():>13.1%}")
    tail = d[d.cohort > 40]
    if len(tail):
        print(f"   40+ days  {len(tail):>7}  {(tail.p_value < ALPHA).mean():>14.1%}  {(tail.p_bh < ALPHA).mean():>13.1%}")


if __name__ == "__main__":
    main()
