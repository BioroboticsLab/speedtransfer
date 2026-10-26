import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import Gamma
from statsmodels.genmod.families.links import log
import statsmodels.stats.multitest as multitest

def fit_cosinor_glm(
    times: np.ndarray,
    values: np.ndarray,
    period: float = 24.0
) -> dict:
    """
    Fits a cosinor model using GLM with Gamma family and Log link function.
    
    Args:
        times: Array of time points (in hours).
        values: Array of response values (e.g., speed). Must be positive.
        period: Period of the rhythm (default 24h).
        
    Returns:
        dict containing:
            - amplitude: estimated amplitude
            - acrophase: estimated peak time (in radians)
            - p_value: p-value for the rhythmic component (f-test/likelihood ratio test equivalent)
            - model: the fitted statsmodels GLMResults object
            - r_squared: pseudo R-squared
    """
    # Remove NaNs
    mask = ~np.isnan(values) & (values > 0) # Gamma requires positive values
    t = times[mask]
    y = values[mask]
    
    if len(y) < 5:
        return {
            "amplitude": np.nan, "acrophase": np.nan, "p_value": np.nan, 
            "mesor": np.nan, "r_squared": np.nan
        }

    # Design matrix: Intercept, Cos, Sin
    omega = 2 * np.pi / period
    X = np.column_stack([
        np.ones(len(t)),
        np.cos(omega * t),
        np.sin(omega * t)
    ])
    
    # Fit GLM with Gamma family and Log link
    # The model is log(E[y]) = beta0 + beta1*cos(wt) + beta2*sin(wt)
    # y ~ Gamma(mu, scale)
    try:
        model = sm.GLM(y, X, family=Gamma(link=log())).fit()
    except Exception as e:
        print(f"GLM Fit failed: {e}")
        return {
            "amplitude": np.nan, "acrophase": np.nan, "p_value": np.nan, 
            "mesor": np.nan, "r_squared": np.nan
        }

    # Extract coefficients
    beta0, beta1, beta2 = model.params
    
    # Amplitude and Acrophase logic for log-link
    # A * cos(wt - phi) = A * (cos(wt)cos(phi) + sin(wt)sin(phi))
    # linear predictor eta = b0 + b1*cos + b2*sin
    # Amplitude of the oscillation in the linear predictor is sqrt(b1^2 + b2^2)
    # The actual variation in y is exp(eta), so amplitude isn't a single number in y-space 
    # but we represent the 'strength' of rhythm by the linear predictor amplitude.
    amp_lin = np.sqrt(beta1**2 + beta2**2)
    # Acrophase phi such that cos(phi) ~ b1, sin(phi) ~ b2
    phi = np.arctan2(beta2, beta1) 
    
    # P-value for rhythmicity: compare to null model (intercept only)
    # Wald test for joint significance of beta1 and beta2
    # restricted model is just column 0 (intercept)
    # we can use model.wald_test checking params 1 and 2
    wald = model.wald_test(np.eye(3)[1:3]) # test beta1=0 and beta2=0
    p_val = wald.pvalue

    return {
        "amplitude": amp_lin, # On log scale
        "acrophase": phi % (2 * np.pi),
        "acrophase_hours": (phi % (2 * np.pi)) / omega,
        "p_value": float(p_val),
        "mesor": np.exp(beta0), # Baseline level
        "aic": model.aic,
        "bic": model.bic_llf,
        "n_obs": len(y)
    }

def adjust_pvalues_bh(p_values: list[float]) -> list[float]:
    """
    Apply Benjamini-Hochberg correction for multiple testing.
    Handles NaNs by ignoring them in correction and returning NaN.
    """
    p_arr = np.array(p_values)
    mask = ~np.isnan(p_arr)
    
    if not np.any(mask):
        return p_arr.tolist()
        
    p_valid = p_arr[mask]
    _rejected, p_corrected, _alphacSidak, _bonferroni = multitest.multipletests(
        p_valid, alpha=0.05, method='fdr_bh'
    )
    
    result = np.full_like(p_arr, np.nan)
    result[mask] = p_corrected
    
    return result.tolist()

if __name__ == "__main__":
    # verification
    t = np.linspace(0, 48, 100)
    # Generate synthetic gamma data with rhythm
    mu = np.exp(1.0 + 0.5 * np.cos(2*np.pi*t/24 - 2.0))
    y = np.random.gamma(shape=10.0, scale=mu/10.0)
    
    res = fit_cosinor_glm(t, y)
    print("Fitted GLM:", res)
    
    # Check null case
    y_null = np.random.gamma(shape=2.0, scale=1.0, size=100)
    res_null = fit_cosinor_glm(t, y_null)
    print("Null GLM:", res_null)
    
    # Check MH correction
    p_vals = [0.001, 0.04, 0.1, 0.8, np.nan, 0.0001]
    adj = adjust_pvalues_bh(p_vals)
    print("P-values:", p_vals)
    print("Adjusted:", adj)
