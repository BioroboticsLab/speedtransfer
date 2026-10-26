"""Unit tests for Gamma and Von Mises sampling functions."""
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulation import _sample_gamma_from_stats, _sample_vonmises_from_stats


class TestSampleGammaFromStats:
    def test_moments_match(self):
        rng = np.random.default_rng(100)
        mean, std = 5.0, 1.5
        samples = _sample_gamma_from_stats(mean, std, 10_000, rng)

        assert np.abs(samples.mean() - mean) < 0.1, (
            f"Mean mismatch: expected ~{mean}, got {samples.mean():.3f}"
        )
        assert np.abs(samples.std() - std) < 0.1, (
            f"Std mismatch: expected ~{std}, got {samples.std():.3f}"
        )

    def test_all_positive(self):
        rng = np.random.default_rng(200)
        samples = _sample_gamma_from_stats(2.0, 1.0, 5_000, rng)
        assert np.all(samples > 0), "Gamma samples should be strictly positive"

    def test_zero_std_returns_constant(self):
        rng = np.random.default_rng(300)
        samples = _sample_gamma_from_stats(3.0, 0.0, 100, rng)
        assert np.allclose(samples, 3.0, atol=0.01), (
            f"Zero-std samples should be ~3.0, got range [{samples.min():.4f}, {samples.max():.4f}]"
        )

    def test_zero_mean_returns_zeros(self):
        rng = np.random.default_rng(350)
        samples = _sample_gamma_from_stats(0.0, 1.0, 100, rng)
        assert np.all(samples == 0.0), "Zero-mean should return all zeros"

    def test_output_shape(self):
        rng = np.random.default_rng(400)
        samples = _sample_gamma_from_stats(4.0, 1.0, 50, rng)
        assert samples.shape == (50,)


class TestSampleVonMisesFromStats:
    def test_circular_mean_matches(self):
        rng = np.random.default_rng(500)
        mean, std = 1.0, 0.3
        samples = _sample_vonmises_from_stats(mean, std, 10_000, rng)

        # Circular mean should be close to the requested mean
        circ_mean = np.arctan2(np.sin(samples).mean(), np.cos(samples).mean())
        assert np.abs(circ_mean - mean) < 0.1, (
            f"Circular mean mismatch: expected ~{mean:.3f}, got {circ_mean:.3f}"
        )

    def test_zero_std_high_concentration(self):
        rng = np.random.default_rng(600)
        samples = _sample_vonmises_from_stats(0.0, 0.0, 1_000, rng)
        # Zero std → returns constant
        assert np.allclose(samples, 0.0), "Zero-std should return all zeros (the mean)"

    def test_output_shape(self):
        rng = np.random.default_rng(700)
        samples = _sample_vonmises_from_stats(0.0, 0.5, 75, rng)
        assert samples.shape == (75,)
