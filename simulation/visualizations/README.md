# Visualization Harness

Deterministic scripts for generating reviewer-facing figures without opening GUI windows.

## How to run
- From repo root: `python -m visualizations.run_all`
- Outputs are written to `visualizations/output/` (PNGs plus the optional `.mat` files produced by `run_simulation`).

## What gets produced
- Baseline run at density 0.5 with trimmed duration (120 day / 240 sim steps): motion snapshot, time-series with sine fits, phase map vs. location.
- Ablation grid across the single-factor suite with matching settings, saved to `visualizations/output/ablation_grid.png` and `ablation_results.csv`.

## Tips
- All RNGs are fixed for reproducibility; adjust seeds inside `visualizations/run_all.py` if needed.
- The backend is forced to Agg so the script is safe to run in CI or headless environments.
