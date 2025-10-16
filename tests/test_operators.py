"""Unit tests for quantum operators."""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nqs.operators import (
    SpinOperators,
    TransverseFieldIsing,
    Heisenberg,
    config_to_spins,
    spins_to_config
)


class TestSpinOperators:
    """Test spin operator functionality."""
    
    def test_sigma_z(self):
        """Test σ^z operator."""
        num_sites = 2
        
        # Test site 0
        sz = SpinOperators.sigma_z(0, num_sites)
        
        # Check values for all configurations
        # |00⟩ -> +1, |01⟩ -> +1, |10⟩ -> -1, |11⟩ -> -1
        expected = np.array([1, 1, -1, -1])
        np.testing.assert_array_equal(sz, expected)
    
    def test_sigma_x(self):
        """Test σ^x operator."""
        num_sites = 2
        
        # Test site 0
        connected, mel = SpinOperators.sigma_x(0, num_sites)
        
        # σ^x flips the bit at site 0
        # |00⟩ -> |10⟩, |01⟩ -> |11⟩, |10⟩ -> |00⟩, |11⟩ -> |01⟩
        expected_connected = np.array([2, 3, 0, 1])
        np.testing.assert_array_equal(connected, expected_connected)
        
        # All matrix elements should be 1
        np.testing.assert_array_equal(mel, np.ones(4))


class TestTransverseFieldIsing:
    """Test TFIM Hamiltonian."""
    
    def test_initialization(self):
        """Test Hamiltonian initialization."""
        hamiltonian = TransverseFieldIsing(
            num_sites=4,
            J=1.0,
            h=1.0,
            pbc=True
        )
        
        assert hamiltonian.num_sites == 4
        assert hamiltonian.J == 1.0
        assert hamiltonian.h == 1.0
        assert hamiltonian.pbc == True
    
    def test_diagonal_energy_two_sites(self):
        """Test diagonal energy for 2-site system."""
        hamiltonian = TransverseFieldIsing(
            num_sites=2,
            J=1.0,
            h=0.0,  # No transverse field
            pbc=True
        )
        
        # Check diagonal energies
        # |00⟩: σ^z_0 = +1, σ^z_1 = +1 -> E = -J*(+1)*(+1) = -1
        # |11⟩: σ^z_0 = -1, σ^z_1 = -1 -> E = -J*(-1)*(-1) = -1
        # |01⟩, |10⟩: E = -J*(+1)*(-1) = +1
        
        assert hamiltonian.diagonal_energy[0] == -1.0  # |00⟩
        assert hamiltonian.diagonal_energy[1] == 1.0   # |01⟩
        assert hamiltonian.diagonal_energy[2] == 1.0   # |10⟩
        assert hamiltonian.diagonal_energy[3] == -1.0  # |11⟩


class TestHelperFunctions:
    """Test utility functions."""
    
    def test_config_to_spins(self):
        """Test configuration to spins conversion."""
        num_sites = 3
        
        # Config 0 = |000⟩ = [+1, +1, +1]
        spins = config_to_spins(0, num_sites)
        np.testing.assert_array_equal(spins, [1, 1, 1])
        
        # Config 7 = |111⟩ = [-1, -1, -1]
        spins = config_to_spins(7, num_sites)
        np.testing.assert_array_equal(spins, [-1, -1, -1])
        
        # Config 5 = |101⟩ = [-1, +1, -1]
        spins = config_to_spins(5, num_sites)
        np.testing.assert_array_equal(spins, [-1, 1, -1])
    
    def test_spins_to_config(self):
        """Test spins to configuration conversion."""
        # [+1, +1, +1] -> 0
        config = spins_to_config(np.array([1, 1, 1]))
        assert config == 0
        
        # [-1, -1, -1] -> 7
        config = spins_to_config(np.array([-1, -1, -1]))
        assert config == 7
        
        # [-1, +1, -1] -> 5
        config = spins_to_config(np.array([-1, 1, -1]))
        assert config == 5
    
    def test_roundtrip_conversion(self):
        """Test that conversions are inverses."""
        num_sites = 4
        
        for config in range(2**num_sites):
            spins = config_to_spins(config, num_sites)
            config_back = spins_to_config(spins)
            assert config == config_back


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
