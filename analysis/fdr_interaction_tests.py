"""Benjamini-Hochberg (FDR) companion for the 36-cell interaction tests.

Rebuttal decision: report uncorrected p-values in the main text and BH-corrected
p-values in the SI as a robustness check (Reviewer #1, l.111-112 / multiple testing).

The interaction analysis compares real vs. null-model speed changes across a 6x6
(=36) quantile matrix, for four grouping metrics (start speed, age, phase, R^2),
two colonies (2016 = A, 2019 = B), with a Welch test per cell (the test reported in
the main text). This script re-derives the number of significant cells before and
after BH correction. With per-cell sample sizes in the millions the p-values are
astronomically small, so the correction leaves every conclusion unchanged.

Run:
    python analysis/fdr_interaction_tests.py            # print summary
    python analysis/fdr_interaction_tests.py --latex    # also emit an SI table
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
ALPHAS: tuple[float, ...] = (0.05, 0.01)
MAIN_TEST = "Welch test"  # the test reported in the manuscript
N_CELLS = 36


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (identical to statsmodels fdr_bh)."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def _metric_of(path: str) -> str:
    return path.split("side0_")[1].split("_", 1)[1].replace(".csv", "")


def collect() -> pd.DataFrame:
    """Return one row per (colony, metric, alpha) with raw vs. BH significant counts."""
    sources = {
        ("B", 2019): sorted(glob.glob(str(REPO / "aggregated_results/2019/test_vel_change_side0_2019_*.csv"))),
        ("A", 2016): sorted(glob.glob(str(REPO / "data/2016/test_vel_change_side0_2016_*.csv"))),
    }
    rows: list[dict] = []
    for (colony, year), files in sources.items():
        for f in files:
            sub = pd.read_csv(f)
            sub = sub[sub["test_name"] == MAIN_TEST]
            if sub.empty:
                continue
            p = sub["p_value"].to_numpy(dtype=float)
            p_bh = bh_adjust(p)
            for alpha in ALPHAS:
                rows.append(
                    dict(
                        colony=colony,
                        year=year,
                        metric=_metric_of(f),
                        alpha=alpha,
                        n=int(p.size),
                        sig_raw=int((p < alpha).sum()),
                        sig_bh=int((p_bh < alpha).sum()),
                    )
                )
    return pd.DataFrame(rows).sort_values(["alpha", "colony", "metric"]).reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    print(f"Welch test, {N_CELLS} cells per matrix — significant cells raw -> BH:")
    for alpha in ALPHAS:
        print(f"\n  alpha = {alpha}")
        for _, r in df[df.alpha == alpha].iterrows():
            changed = "" if r.sig_raw == r.sig_bh else "   <-- CHANGED"
            print(f"    Colony {r.colony} ({r.year}) {r.metric:10s}: "
                  f"{r.sig_raw:2d}/{r.n} -> {r.sig_bh:2d}/{r.n}{changed}")
    if (df.sig_raw == df.sig_bh).all():
        print("\n  => BH correction leaves every conclusion unchanged.")


def latex_table(df: pd.DataFrame) -> str:
    a = df[df.alpha == 0.05]
    lines = [
        r"\begin{table}[h]\centering",
        r"\caption{\added{Interaction tests remain significant after Benjamini--Hochberg "
        r"(FDR) correction. Number of significant cells (of 36; Welch test, $\alpha=0.05$) "
        r"before and after correction, per grouping metric and colony.}}",
        r"\label{tab:fdr_interaction}",
        r"\begin{tabular}{llcc}",
        r"\toprule",
        r"Colony & Metric & Uncorrected & BH-corrected \\",
        r"\midrule",
    ]
    for _, r in a.iterrows():
        lines.append(f"{r.colony} ({r.year}) & {r.metric.replace('_', ' ')} & "
                     f"{r.sig_raw}/36 & {r.sig_bh}/36 \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true", help="emit an SI LaTeX table")
    args = ap.parse_args()
    df = collect()
    print_summary(df)
    if args.latex:
        print("\n" + latex_table(df))


if __name__ == "__main__":
    main()
