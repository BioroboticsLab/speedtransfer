import sys
from pathlib import Path

import numpy as np

# Ensure local simulation module is importable when running tests directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation import _bin_statistics, mgd


def test_mgd_matches_requested_moments():
    rng = np.random.default_rng(123)
    mean = np.array([1.0, -2.0])
    cov = np.array([[2.0, 0.5], [0.5, 1.0]])

    samples = mgd(5000, 2, mean, cov, rng=rng)
    sample_mean = samples.mean(axis=0)
    centered = samples - sample_mean
    sample_cov = centered.T @ centered / (samples.shape[0] - 1)

    # Sampling noise tolerated with loose absolute error bounds
    assert np.allclose(sample_mean, mean, atol=0.05)
    assert np.allclose(sample_cov, cov, atol=0.05)


def test_bin_statistics_reports_means_and_counts():
    x = np.array([0.1, 0.9, 0.4, 0.6])
    y = np.array([0.2, 0.8, 0.5, 0.7])
    values = np.array([1.0, 2.0, 3.0, np.nan])

    means, counts = _bin_statistics(
        x=x,
        y=y,
        values=values,
        bins_x=2,
        bins_y=2,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
    )

    # Bin layout (row major, y-bin 0 is the lower half):
    # [ (0,1) | (1,1) ]  <- y bin 1
    # [ (0,0) | (1,0) ]  <- y bin 0
    assert counts.tolist() == [[1.0, 0.0], [1.0, 1.0]]
    assert np.isclose(means[0, 0], 1.0)  # lower-left bin mean
    assert np.isnan(means[0, 1])  # lower-right bin is empty
    assert np.isclose(means[1, 0], 3.0)  # upper-left bin mean
