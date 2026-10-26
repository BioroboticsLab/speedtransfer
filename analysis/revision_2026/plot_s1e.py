"""Regenerate Fig S1e (colony A) from the corrected dist_exit_df, matching the
original: round distance, per-bin median, drop sparsest 5%, min-max normalize
(phase & phase-std reversed), plot the four rhythm parameters, with the four
right-hand scales giving the original values of each normalized trace."""
import os
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dist_exit_df_side0_2016_CORRECTED.csv"))
df["dist_round"] = df["entrance_dist_focal_median"].round(-1)
g = df.groupby("dist_round").median(numeric_only=True).reset_index()
g = g[g["phase_focal_count"] > g["phase_focal_count"].quantile(0.05)].sort_values("dist_round")

def mm(x, rev=False):
    n = (x - x.min()) / (x.max() - x.min())
    return 1 - n if rev else n

specs = [  # column, reverse, color, style, label, header
    ("phase_focal_median",     True,  "steelblue", "-",  "Phase Median", r"$T(\phi)$"),
    ("phase_focal_std",        True,  "goldenrod", ":",  "Phase Std",     r"$Sd_\phi$"),
    ("age_focal_median",       False, "seagreen",  "--", "Age Median",    "Age"),
    ("r_squared_focal_median", False, "chocolate", "-",  "R² Median", r"$R^2$"),
]
fig, ax = plt.subplots(figsize=(10, 4.4))
for col, rev, color, ls, lab, _ in specs:
    ax.plot(g.dist_round, mm(g[col], rev), color=color, ls=ls, lw=2, label=lab)
ax.set_xlabel("Distance to entrance [comb units]")
ax.set_ylabel("Phase / Synchronicity / Age / $R^2$\n(Min–Max Normalized)")
ax.set_title("Rhythm parameters at distances to entrance (colony A, corrected)")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(-0.03, 1.03)

# right-hand original-value scales (one column per trace), matching the original
yt = np.linspace(0, 1, 6)
x0 = 1.06
for k, (col, rev, color, ls, lab, hdr) in enumerate(specs):
    xc = x0 + k * 0.075
    lo, hi = g[col].min(), g[col].max()
    ax.text(xc, 1.07, hdr, transform=ax.transAxes, ha="center", va="bottom", color=color, fontsize=10)
    for yv in yt:
        orig = (hi - yv * (hi - lo)) if rev else (lo + yv * (hi - lo))
        val = f"{orig:.2f}" if col == "r_squared_focal_median" else f"{orig:.1f}"
        ax.text(xc, yv, val, transform=ax.transAxes, ha="center", va="center", color=color, fontsize=8)
fig.subplots_adjust(right=0.72)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "speed_transfer_paper", "figure_S1e_corrected.pdf")
fig.savefig(out, bbox_inches="tight"); print("saved", out)
