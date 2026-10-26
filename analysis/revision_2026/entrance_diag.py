"""Clean entrance diagnostic on the ACTUAL filter_overlap dist_exit_df positions.
Plain count-weighted Pearson correlation of each per-position circadian metric vs
distance from a candidate entrance. Compares (0,250) [published], (346,270)
[proposed], and the grid-search best. No WLS re-weighting (which distorts R2)."""
import os
import numpy as np, pandas as pd

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dist_exit_df_side0_2016_CORRECTED.csv"))
need = ['x_pos_start_focal','y_pos_start_focal','phase_focal_median','age_focal_median','r_squared_focal_median']
df = df.dropna(subset=need)
x = df['x_pos_start_focal'].to_numpy(float); y = df['y_pos_start_focal'].to_numpy(float)
w = df['phase_focal_count'].to_numpy(float) if 'phase_focal_count' in df.columns else np.ones(len(df))
print(f"n_positions={len(df):,}  x[{x.min():.0f},{x.max():.0f}] y[{y.min():.0f},{y.max():.0f}]  totalcount={w.sum():,.0f}")

def wcorr(a, b, w):
    W = w / w.sum(); ma = (a*W).sum(); mb = (b*W).sum(); ca = a-ma; cb = b-mb
    va = (W*ca*ca).sum(); vb = (W*cb*cb).sum()
    return (W*ca*cb).sum()/np.sqrt(va*vb) if va > 0 and vb > 0 else np.nan

# for phase, entrance = point where phase INCREASES most with distance (max r).
for var, want in [('phase_focal_median','max'), ('age_focal_median','min'), ('r_squared_focal_median','min')]:
    v = df[var].to_numpy(float)
    print(f"\n=== {var}  (weighted r; R2 = r^2) ===")
    for label, (ex, ey) in [("published (0,250)", (0,250)), ("proposed  (346,270)", (346,270))]:
        d = np.hypot(x-ex, y-ey); r = wcorr(d, v, w)
        print(f"  {label:22s}: r={r:+.3f}  R2={r*r:.3f}")
    best = None
    for ex in np.linspace(-60, 430, 50):
        for ey in np.linspace(-60, 300, 37):
            r = wcorr(np.hypot(x-ex, y-ey), v, w)
            if not np.isfinite(r):
                continue
            score = r if want == 'max' else -r
            if best is None or score > best[0]:
                best = (score, r, (ex, ey))
    print(f"  BEST ({'phase↑dist' if want=='max' else 'metric↓dist'}): ({best[2][0]:.0f},{best[2][1]:.0f})  r={best[1]:+.3f}  R2={best[1]**2:.3f}")
