import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import sys
from pathlib import Path

def prepare_df(path):
    df = pd.read_csv(path)
    df["is_circadian"] = (df.p_value < 0.05).astype(int)
    df["phase"] = ((-24 * df["phase"] / (2 * np.pi)) + 12) % 24
    return df

def run_regressions(df, year, side, variable):
    print(f"\n--- Results for {year} side {side}---")
    print(f"Weighted Linear Regression (entrance_dist_focal_median ~ {variable}):")

    # WLS: entrance_dist_focal_median ~ age_focal_median
    group_vars = df.groupby(variable)['entrance_dist_focal_median'].var()
    df['variance'] = df[variable].map(group_vars)
    # avoid dividing through 0
    df['variance'].replace(0, 1e-8, inplace=True)
    df['variance'].fillna(df['variance'].median(), inplace=True)
    df['inv_var_weight'] = 1 / df['variance']

    group_counts = df[variable].value_counts(normalize=True)
    df['sampling_weight'] = df[variable].map(lambda g: 1 / group_counts[g])
    df['sampling_weight'] /= df['sampling_weight'].mean()

    df['combined_weight'] = df['inv_var_weight'] * df['sampling_weight']

    wls_model = smf.wls(f"entrance_dist_focal_median ~ {variable}", data=df, weights=df["combined_weight"]).fit()
    print(f"\nWLS regression (entrance_dist ~ {variable}):")
    print(wls_model.summary())
    print("P-values:", wls_model.pvalues)

    bp_test = sms.het_breuschpagan(wls_model.resid, wls_model.model.exog)
    labels = ['LM-Statistics', 'LM-p-Value', 'F-Statistics', 'F-p-Value']
    print(dict(zip(labels, bp_test)))

if __name__ == "__main__":
    sys.path.append(str(Path("correlation_of_dist_entrance_tests.py").resolve().parents[0]))
    import path_settings

    dist_exit_df_2016 = pd.read_csv(path_settings.DIST_EXIT_SIDE_0_DF_PATH_2016)
    dist_exit_df_2016 = dist_exit_df_2016.dropna()
    dist_exit_df_2019 = pd.read_csv(path_settings.DIST_EXIT_SIDE_0_DF_PATH_2019)
    dist_exit_df_2019 = dist_exit_df_2019.dropna()

    run_regressions(dist_exit_df_2016, 2016, 0, 'age_focal_median')
    run_regressions(dist_exit_df_2019, 2019, 0, 'age_focal_median')
    run_regressions(dist_exit_df_2016, 2016, 0, 'r_squared_focal_median')
    run_regressions(dist_exit_df_2019, 2019, 0, 'r_squared_focal_median')
    run_regressions(dist_exit_df_2016, 2016, 0, 'phase_focal_median')
    run_regressions(dist_exit_df_2019, 2019, 0, 'phase_focal_median')
    run_regressions(dist_exit_df_2016, 2016, 0, 'phase_focal_std')
    run_regressions(dist_exit_df_2019, 2019, 0, 'phase_focal_std')