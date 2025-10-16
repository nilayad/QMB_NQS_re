"""Unit tests for MCMC sampling."""

import pytest
import numpy as np
import tensorflow as tf
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nqs.sampling import (
    MetropolisSampler,
    compute_autocorrelation,
    estimate_correlation_time
)
from nqs.models import RBM


class TestMetropolisSampler:
    """Test Metropolis-Hastings sampler."""
    
    def test_initialization(self):
        """Test sampler initialization."""
        sampler = MetropolisSampler(
            num_sites=4,
            num_samples=100,
            num_chains=2,
            warmup_steps=10
        )
        
        assert sampler.num_sites == 4
        assert sampler.num_samples == 100
        assert sampler.num_chains == 2
        assert sampler.warmup_steps == 10
    
    def test_initialize_chains(self):
        """Test chain initialization."""
        sampler = MetropolisSampler(
            num_sites=4,
            num_samples=100,
            num_chains=3
        )
        
        configs = sampler._initialize_chains()
        
        # Check shape
        assert configs.shape == (3, 4)
        
        # Check values are ±1
        assert np.all((configs == 1) | (configs == -1))
    
    def test_propose_flip(self):
        """Test single-spin flip proposal."""
        sampler = MetropolisSampler(
            num_sites=4,
            num_samples=100,
            num_chains=2
        )
        
        configs = np.array([[1, 1, -1, -1], [1, -1, 1, -1]])
        proposed, sites = sampler._propose_flip(configs)
        
        # Check that exactly one spin was flipped per chain
        for i in range(2):
            diff = configs[i] != proposed[i]
            assert np.sum(diff) == 1
    
    def test_sampling(self):
        """Test full sampling procedure."""
        # Create simple model
        model = RBM(num_sites=4, num_hidden=8)
        
        sampler = MetropolisSampler(
            num_sites=4,
            num_samples=50,
            num_chains=2,
            warmup_steps=10
        )
        
        # Sample
        samples, stats = sampler.sample(
            model=lambda s: model(s, training=False),
            show_progress=False
        )
        
        # Check output shape
        expected_samples = 50 * 2  # num_samples * num_chains
        assert samples.shape == (expected_samples, 4)
        
        # Check statistics
        assert 'acceptance_rate' in stats
        assert 0 <= stats['acceptance_rate'] <= 1


class TestAutocorrelation:
    """Test autocorrelation analysis."""
    
    def test_compute_autocorrelation(self):
        """Test autocorrelation computation."""
        # Create synthetic data with known autocorrelation
        n = 1000
        t = np.arange(n)
        
        # White noise (no autocorrelation)
        data = np.random.randn(n)
        autocorr = compute_autocorrelation(data, max_lag=50)
        
        # First value should be 1 (self-correlation)
        assert np.abs(autocorr[0] - 1.0) < 1e-10
        
        # Others should be small for white noise
        assert np.abs(np.mean(autocorr[1:])) < 0.1
    
    def test_estimate_correlation_time(self):
        """Test correlation time estimation."""
        # White noise
        data = np.random.randn(1000)
        tau = estimate_correlation_time(data, max_lag=50)
        
        # Should be close to 1 for white noise
        assert 0.5 <= tau <= 2.0
        
        # Highly correlated data
        data_corr = np.repeat(np.random.randn(100), 10)
        tau_corr = estimate_correlation_time(data_corr, max_lag=50)
        
        # Should be larger
        assert tau_corr > tau


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
