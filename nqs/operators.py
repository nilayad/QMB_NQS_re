"""Quantum operators for spin systems.

This module implements quantum operators for spin-1/2 systems including
Pauli matrices and Hamiltonian construction for various models.
"""

import numpy as np
from typing import List, Tuple, Optional
import tensorflow as tf


class SpinOperators:
    """Collection of spin-1/2 operators."""
    
    @staticmethod
    def sigma_z(site: int, num_sites: int) -> np.ndarray:
        """Compute σ^z operator at a specific site.
        
        Args:
            site: Site index (0-indexed)
            num_sites: Total number of sites
            
        Returns:
            Diagonal σ^z operator as 1D array (diagonal elements only)
        """
        configs = np.arange(2**num_sites)
        return 1.0 - 2.0 * ((configs >> site) & 1)
    
    @staticmethod
    def sigma_x(site: int, num_sites: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute σ^x operator at a specific site.
        
        Args:
            site: Site index (0-indexed)
            num_sites: Total number of sites
            
        Returns:
            Tuple of (connected_configs, matrix_elements)
            connected_configs: States connected by σ^x
            matrix_elements: All ones (σ^x flips spin)
        """
        configs = np.arange(2**num_sites)
        connected_configs = configs ^ (1 << site)  # Flip bit at site
        matrix_elements = np.ones_like(configs, dtype=np.float64)
        return connected_configs, matrix_elements
    
    @staticmethod
    def sigma_y(site: int, num_sites: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute σ^y operator at a specific site.
        
        Args:
            site: Site index (0-indexed)
            num_sites: Total number of sites
            
        Returns:
            Tuple of (connected_configs, matrix_elements)
        """
        configs = np.arange(2**num_sites)
        connected_configs = configs ^ (1 << site)
        # σ^y = i(|↓⟩⟨↑| - |↑⟩⟨↓|), imaginary part handled separately
        spins = (configs >> site) & 1
        matrix_elements = 1.0 - 2.0 * spins  # +1 for up->down, -1 for down->up
        return connected_configs, matrix_elements


class TransverseFieldIsing:
    """Transverse-field Ising model Hamiltonian.
    
    H = -J Σᵢ σᵢᶻ σᵢ₊₁ᶻ - h Σᵢ σᵢˣ
    """
    
    def __init__(self, num_sites: int, J: float = 1.0, h: float = 1.0, 
                 pbc: bool = True):
        """Initialize Hamiltonian.
        
        Args:
            num_sites: Number of spin sites
            J: Ising coupling strength
            h: Transverse field strength
            pbc: Use periodic boundary conditions
        """
        self.num_sites = num_sites
        self.J = J
        self.h = h
        self.pbc = pbc
        self._precompute_local_energies()
    
    def _precompute_local_energies(self):
        """Precompute diagonal (σ^z σ^z) terms."""
        configs = np.arange(2**self.num_sites)
        
        # Compute Ising term: -J Σᵢ σᵢᶻ σᵢ₊₁ᶻ
        energy = np.zeros(2**self.num_sites)
        num_bonds = self.num_sites if self.pbc else self.num_sites - 1
        
        for i in range(num_bonds):
            j = (i + 1) % self.num_sites
            sz_i = SpinOperators.sigma_z(i, self.num_sites)
            sz_j = SpinOperators.sigma_z(j, self.num_sites)
            energy -= self.J * sz_i * sz_j
        
        self.diagonal_energy = energy
        
        # Precompute off-diagonal connections (σ^x terms)
        self.connections = []
        for i in range(self.num_sites):
            connected, _ = SpinOperators.sigma_x(i, self.num_sites)
            self.connections.append(connected)
    
    def local_energy(self, configs: np.ndarray, 
                     log_psi: np.ndarray) -> np.ndarray:
        """Compute local energy for given configurations.
        
        E_loc(s) = Σₛ' H(s,s') ψ(s')/ψ(s)
        
        Args:
            configs: Configuration indices (integers)
            log_psi: Log amplitude log|ψ(s)| or callable that takes config integers
            
        Returns:
            Local energy for each configuration
        """
        batch_size = len(configs)
        e_loc = self.diagonal_energy[configs]
        
        # Get log_psi for current configs
        if callable(log_psi):
            # Convert configs to spins
            current_spins = np.array([config_to_spins(c, self.num_sites) for c in configs])
            log_psi_current = log_psi(current_spins)
        else:
            log_psi_current = log_psi[configs]
        
        # Add transverse field contributions
        for i in range(self.num_sites):
            connected = self.connections[i][configs]
            
            # Get log_psi for connected configs
            if callable(log_psi):
                connected_spins = np.array([config_to_spins(c, self.num_sites) for c in connected])
                log_psi_connected = log_psi(connected_spins)
            else:
                log_psi_connected = log_psi[connected]
            
            # e_loc += -h * exp(log|ψ(s')|/|ψ(s)|)
            e_loc += -self.h * np.exp(log_psi_connected - log_psi_current)
        
        return e_loc
    
    def find_connections(self, config: int) -> Tuple[np.ndarray, np.ndarray]:
        """Find all states connected to config and their matrix elements.
        
        Args:
            config: Configuration index
            
        Returns:
            Tuple of (connected_states, matrix_elements)
        """
        connected = [config]
        mel = [self.diagonal_energy[config]]
        
        # Add σ^x connections
        for i in range(self.num_sites):
            connected.append(self.connections[i][config])
            mel.append(-self.h)
        
        return np.array(connected), np.array(mel)


class Heisenberg:
    """Heisenberg model Hamiltonian.
    
    H = J Σᵢ (σᵢˣσᵢ₊₁ˣ + σᵢʸσᵢ₊₁ʸ + Δσᵢᶻσᵢ₊₁ᶻ)
    """
    
    def __init__(self, num_sites: int, J: float = 1.0, Delta: float = 1.0,
                 pbc: bool = True):
        """Initialize Heisenberg Hamiltonian.
        
        Args:
            num_sites: Number of spin sites
            J: Exchange coupling
            Delta: Anisotropy parameter
            pbc: Use periodic boundary conditions
        """
        self.num_sites = num_sites
        self.J = J
        self.Delta = Delta
        self.pbc = pbc
        self._precompute_connections()
    
    def _precompute_connections(self):
        """Precompute all connections."""
        num_bonds = self.num_sites if self.pbc else self.num_sites - 1
        
        # Precompute diagonal (σ^z σ^z) terms
        configs = np.arange(2**self.num_sites)
        energy = np.zeros(2**self.num_sites)
        
        for i in range(num_bonds):
            j = (i + 1) % self.num_sites
            sz_i = SpinOperators.sigma_z(i, self.num_sites)
            sz_j = SpinOperators.sigma_z(j, self.num_sites)
            energy += self.J * self.Delta * sz_i * sz_j
        
        self.diagonal_energy = energy
        
        # Precompute off-diagonal connections
        self.sx_connections = []
        self.sy_connections = []
        self.sy_phases = []
        
        for i in range(num_bonds):
            j = (i + 1) % self.num_sites
            
            # σ^x_i σ^x_j connections (flip both spins)
            configs_flip_both = configs ^ (1 << i) ^ (1 << j)
            self.sx_connections.append(configs_flip_both)
            
            # σ^y_i σ^y_j connections (also flip both, but with phase)
            self.sy_connections.append(configs_flip_both)
            
            # Compute phases for σ^y σ^y
            spin_i = (configs >> i) & 1
            spin_j = (configs >> j) & 1
            # Phase is (-1)^(s_i + s_j)
            phases = 1.0 - 2.0 * ((spin_i + spin_j) & 1)
            self.sy_phases.append(phases)
    
    def local_energy(self, configs: np.ndarray,
                     log_psi: np.ndarray) -> np.ndarray:
        """Compute local energy for given configurations.
        
        Args:
            configs: Configuration indices
            log_psi: Log amplitude log|ψ(s)| or callable that takes config integers
            
        Returns:
            Local energy for each configuration
        """
        e_loc = self.diagonal_energy[configs]
        
        # Get log_psi for current configs
        if callable(log_psi):
            current_spins = np.array([config_to_spins(c, self.num_sites) for c in configs])
            log_psi_current = log_psi(current_spins)
        else:
            log_psi_current = log_psi[configs]
        
        num_bonds = self.num_sites if self.pbc else self.num_sites - 1
        
        for bond_idx in range(num_bonds):
            # σ^x σ^x term
            connected_sx = self.sx_connections[bond_idx][configs]
            if callable(log_psi):
                connected_sx_spins = np.array([config_to_spins(c, self.num_sites) for c in connected_sx])
                log_psi_sx = log_psi(connected_sx_spins)
            else:
                log_psi_sx = log_psi[connected_sx]
            e_loc += self.J * np.exp(log_psi_sx - log_psi_current)
            
            # σ^y σ^y term
            connected_sy = self.sy_connections[bond_idx][configs]
            phases = self.sy_phases[bond_idx][configs]
            if callable(log_psi):
                connected_sy_spins = np.array([config_to_spins(c, self.num_sites) for c in connected_sy])
                log_psi_sy = log_psi(connected_sy_spins)
            else:
                log_psi_sy = log_psi[connected_sy]
            e_loc += self.J * phases * np.exp(log_psi_sy - log_psi_current)
        
        return e_loc


def config_to_spins(config: int, num_sites: int) -> np.ndarray:
    """Convert configuration integer to spin array.
    
    Args:
        config: Configuration as integer
        num_sites: Number of sites
        
    Returns:
        Spin configuration as array of ±1
    """
    spins = np.array([(config >> i) & 1 for i in range(num_sites)])
    return 1 - 2 * spins  # Convert 0,1 to +1,-1


def spins_to_config(spins: np.ndarray) -> int:
    """Convert spin array to configuration integer.
    
    Args:
        spins: Spin configuration (±1)
        
    Returns:
        Configuration as integer
    """
    bits = (1 - spins) // 2  # Convert +1,-1 to 0,1
    config = 0
    for i, bit in enumerate(bits):
        config |= (int(bit) << i)
    return config
