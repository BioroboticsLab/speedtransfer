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

def run_regressions(df, year):
    print(f"\n--- Results for {year} ---")
    # Logistische Regression: is_circadian ~ age
    logit_model = smf.logit("is_circadian ~ age", data=df).fit(disp=0)
    print("Logistische Regression (is_circadian ~ age):")
    print(logit_model.summary())
    print("P-Values:", logit_model.pvalues)

    # WLS: phase ~ age
    group_vars = df.groupby('age')['phase'].var()
    df['variance'] = df['age'].map(group_vars)
    df['inv_var_weight'] = 1 / df['variance']

    group_counts = df['age'].value_counts(normalize=True)
    df['sampling_weight'] = df['age'].map(lambda g: 1 / group_counts[g])
    df['sampling_weight'] /= df['sampling_weight'].mean()

    df['combined_weight'] = df['inv_var_weight'] * df['sampling_weight']

    wls_model = smf.wls("phase ~ age", data=df, weights=df["combined_weight"]).fit()
    print("\nWLS regression (phase ~ age):")
    print(wls_model.summary())
    print("P-values:", wls_model.pvalues)

    bp_test = sms.het_breuschpagan(wls_model.resid, wls_model.model.exog)
    labels = ['LM-Statistics', 'LM-p-Value', 'F-Statistics', 'F-p-Value']
    print(dict(zip(labels, bp_test)))

if __name__ == "__main__":
    cosinor_df_2016 = prepare_df(path_settings.COSINOR_DF_PATH_2016)
    cosinor_df_2019 = prepare_df(path_settings.COSINOR_DF_PATH_2019)

    run_regressions(cosinor_df_2016, 2016)
    run_regressions(cosinor_df_2019, 2019)



