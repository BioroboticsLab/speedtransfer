"""Colony-B-only day/night figure (single verified-entrance colony) for the
reworked SI Note 5."""
import os
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "daynight_2019.csv"))
OUT = Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "speed_transfer_paper", "daynight_age_distance.pdf"))
fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(df.dist_center, df.age_med_day, "-o", color="darkorange", ms=4,
        label=f"day 09–18 (n={int(df.n_day.sum()):,})")
ax.fill_between(df.dist_center, df.age_q1_day, df.age_q3_day, color="darkorange", alpha=0.13)
ax.plot(df.dist_center, df.age_med_night, "-o", color="navy", ms=4,
        label=f"night 21–06 (n={int(df.n_night.sum()):,})")
ax.fill_between(df.dist_center, df.age_q1_night, df.age_q3_night, color="navy", alpha=0.13)
ax.set_xlabel("distance from entrance [comb units]")
ax.set_ylabel("median bee age [days]  (shaded: IQR)")
ax.set_title("Age vs. distance from entrance, day vs. night (colony B)")
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.tight_layout(); fig.savefig(OUT, bbox_inches="tight")
print("saved ->", OUT)
