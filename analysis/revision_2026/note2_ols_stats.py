import os
import pandas as pd, statsmodels.formula.api as smf, argparse
ap=argparse.ArgumentParser(); ap.add_argument("--year",type=int,required=True); a=ap.parse_args()
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dist_exit_df_side0_2016_CORRECTED.csv" if a.year==2016 else "dist_exit_df_side0_2019_recomputed.csv")
df = pd.read_csv(path).dropna()
print(f"===== {a.year}  (OLS, position-level, corrected entrance) =====")
def show(dv,iv,name):
    m=smf.ols(f"{dv} ~ {iv}",data=df).fit()
    print(f"[{name}] n={int(m.nobs)} R2={m.rsquared:.3f}")
    for t in ['Intercept',iv]:
        print(f"   {t:24s} coef={m.params[t]:+13.4f} se={m.bse[t]:.4g} t={m.tvalues[t]:+.1f} p={m.pvalues[t]:.1e}")
show("entrance_dist_focal_median","age_focal_median","dist~age")
show("entrance_dist_focal_median","phase_focal_median","dist~phase")
show("entrance_dist_focal_median","r_squared_focal_median","dist~R2")
show("phase_focal_median","age_focal_median","phase~age(pos-level)")
