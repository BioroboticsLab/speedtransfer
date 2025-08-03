import pandas as pd
import numpy as np
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import path_settings

def prepare_df(path):
    df = pd.read_csv(path)
    df["is_circadian"] = (df.p_value < 0.05).astype(int)
    df["phase"] = ((-24 * df["phase"] / (2 * np.pi)) + 12) % 24
    return df

def run_regressions(df, year, side):
    print(f"\n--- Results for {year} side {side}---")
    # Logistische Regression: is_circadian ~ age
    print(df.columns)
    #logit_model = smf.logit("age_focal_median ~ entrance_dist_focal_median", data=df).fit(disp=0)
    print("Logistische Regression (age_focal_median ~ entrance_dist_focal_median):")
    #print(logit_model.summary())
    #print("P-Values:", logit_model.pvalues)

    # WLS: phase ~ age
    group_vars = df.groupby('age_focal_median')['entrance_dist_focal_median'].var()
    df['variance'] = df['age_focal_median'].map(group_vars)
    df['inv_var_weight'] = 1 / df['variance']

    group_counts = df['age_focal_median'].value_counts(normalize=True)
    df['sampling_weight'] = df['age_focal_median'].map(lambda g: 1 / group_counts[g])
    df['sampling_weight'] /= df['sampling_weight'].mean()

    df['combined_weight'] = df['inv_var_weight'] * df['sampling_weight']

    wls_model = smf.wls("entrance_dist_focal_median ~ age_focal_median", data=df, weights=df["combined_weight"]).fit()
    print("\nWLS regression (phase ~ age):")
    print(wls_model.summary())
    print("P-values:", wls_model.pvalues)

    bp_test = sms.het_breuschpagan(wls_model.resid, wls_model.model.exog)
    labels = ['LM-Statistics', 'LM-p-Value', 'F-Statistics', 'F-p-Value']
    print(dict(zip(labels, bp_test)))

if __name__ == "__main__":
    dist_exit_df_2016 = pd.read_csv(path_settings.DIST_EXIT_SIDE_0_DF_PATH_2016)
    dist_exit_df_2016 = dist_exit_df_2016.dropna()
    dist_exit_df_2019 = pd.read_csv(path_settings.DIST_EXIT_SIDE_0_DF_PATH_2019)
    dist_exit_df_2019 = dist_exit_df_2019.dropna()
    #interaction_df_2016_0 = prepare_df(path_settings.INTERACTION_SIDE_0_DF_PATH_2016)
    #interaction_df_2019_0 = prepare_df(path_settings.INTERACTION_SIDE_0_DF_PATH_2016)
    cosinor_df_2019 = prepare_df(path_settings.COSINOR_DF_PATH_2019)

    run_regressions(dist_exit_df_2016, 2016, 0)
    run_regressions(dist_exit_df_2019, 2019, 0)
    #run_regressions(interaction_df_2016_0, 2019, 0)