"""Tests for training module."""

import pytest
import numpy as np
import tensorflow as tf
from nqs import MLP, TFIMHamiltonian, VMCTrainer, MetropolisSampler


class TestModelCreation:
    """Test model creation and building."""
    
    def test_mlp_creation(self):
        """Test MLP model creation."""
        model = MLP(num_sites=4, hidden_sizes=[32])
        # Build model by passing dummy input
        dummy_input = tf.ones((1, 4))
        _ = model(dummy_input)
        assert model.built
        assert model.count_params() > 0
    
    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        model = MLP(num_sites=4, hidden_sizes=[16])
        dummy_input = tf.ones((5, 4))
        output = model(dummy_input)
        assert output is not None
        assert output.shape[0] == 5


class TestHamiltonian:
    """Test Hamiltonian creation."""
    
    def test_tfim_creation(self):
        """Test TFIM Hamiltonian creation."""
        hamiltonian = TFIMHamiltonian(num_sites=4, J=1.0, h=0.5)
        assert hamiltonian is not None


class TestSampler:
    """Test MCMC sampler."""
    
    def test_sampler_creation(self):
        """Test MetropolisSampler creation."""
        model = MLP(num_sites=4, hidden_sizes=[16])
        # Build model
        _ = model(tf.ones((1, 4)))
        
        sampler = MetropolisSampler(model, num_sites=4)
        assert sampler is not None


class TestVMCTrainer:
    """Test VMC training."""
    
    def test_trainer_creation(self):
        """Test creating VMC trainer with correct signature."""
        # Create and build model
        model = MLP(num_sites=4, hidden_sizes=[32])
        _ = model(tf.ones((1, 4)))
        
        # Create components
        hamiltonian = TFIMHamiltonian(num_sites=4, J=1.0, h=0.5)
        sampler = MetropolisSampler(model, num_sites=4)
        
        # Create trainer with correct signature: (model, hamiltonian, sampler)
        trainer = VMCTrainer(model, hamiltonian, sampler)
        assert trainer is not None
    
    def test_trainer_with_learning_rate(self):
        """Test VMC trainer with custom learning rate."""
        model = MLP(num_sites=4, hidden_sizes=[16])
        _ = model(tf.ones((1, 4)))
        
        hamiltonian = TFIMHamiltonian(num_sites=4, J=1.0, h=0.5)
        sampler = MetropolisSampler(model, num_sites=4)
        
        # Create with custom learning rate
        trainer = VMCTrainer(model, hamiltonian, sampler, learning_rate=0.01)
        assert trainer is not None


class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_complete_setup(self):
        """Test complete VMC setup."""
        L = 4
        
        # Create model
        model = MLP(num_sites=L, hidden_sizes=[32])
        _ = model(tf.ones((1, L)))
        
        # Create Hamiltonian
        hamiltonian = TFIMHamiltonian(num_sites=L, J=1.0, h=0.5)
        
        # Create sampler
        sampler = MetropolisSampler(model, num_sites=L)
        
        # Create trainer
        trainer = VMCTrainer(model, hamiltonian, sampler, learning_rate=0.001)
        
        # Verify all components
        assert model.count_params() > 0
        assert hamiltonian is not None
        assert sampler is not None
        assert trainer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
