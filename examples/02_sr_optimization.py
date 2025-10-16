#!/usr/bin/env python3
"""Stochastic Reconfiguration optimization example.

This example demonstrates:
- Using SR for faster convergence
- State preparation with imaginary time evolution
- Comparing SR vs VMC
- Advanced diagnostics
"""

import numpy as np
import tensorflow as tf
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nqs import (
    RBM,
    TransverseFieldIsing,
    MetropolisSampler,
    StochasticReconfiguration,
    VMCTrainer
)
from utils import (
    plot_training_history,
    plot_convergence_comparison
)


def main():
    """Run SR optimization and compare with VMC."""
    
    # Set seeds
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("=" * 60)
    print("Stochastic Reconfiguration Example")
    print("=" * 60)
    
    # System parameters
    num_sites = 4
    J = 1.0
    h = 1.0
    
    print(f"\nSystem: TFIM with {num_sites} sites")
    print(f"  J = {J}, h = {h}")
    
    # Create Hamiltonian
    hamiltonian = TransverseFieldIsing(
        num_sites=num_sites,
        J=J,
        h=h,
        pbc=True
    )
    
    # Sampler for both methods
    sampler_sr = MetropolisSampler(
        num_sites=num_sites,
        num_samples=500,
        num_chains=4,
        warmup_steps=50
    )
    
    sampler_vmc = MetropolisSampler(
        num_sites=num_sites,
        num_samples=500,
        num_chains=4,
        warmup_steps=50
    )
    
    print("\n" + "-" * 60)
    print("Training with Stochastic Reconfiguration")
    print("-" * 60)
    
    # Create RBM for SR
    model_sr = RBM(
        num_sites=num_sites,
        num_hidden=2 * num_sites,
        use_bias=True
    )
    
    # SR optimizer
    sr_trainer = StochasticReconfiguration(
        model=model_sr,
        hamiltonian=hamiltonian,
        sampler=sampler_sr,
        learning_rate=0.01,
        diagonal_shift=0.01
    )
    
    # Train with SR
    history_sr = sr_trainer.train(
        num_steps=100,
        log_interval=10,
        show_progress=True
    )
    
    print("\n" + "-" * 60)
    print("Training with VMC for comparison")
    print("-" * 60)
    
    # Create RBM for VMC
    model_vmc = RBM(
        num_sites=num_sites,
        num_hidden=2 * num_sites,
        use_bias=True
    )
    
    # VMC trainer
    vmc_trainer = VMCTrainer(
        model=model_vmc,
        hamiltonian=hamiltonian,
        sampler=sampler_vmc,
        learning_rate=0.01
    )
    
    # Train with VMC
    history_vmc = vmc_trainer.train(
        num_steps=100,
        log_interval=10,
        show_progress=True
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)
    
    sr_final = history_sr['energy'][-1]
    vmc_final = history_vmc['energy'][-1]
    
    print(f"\nFinal Energies:")
    print(f"  SR:  {sr_final:.6f} (error: {history_sr['energy_error'][-1]:.6f})")
    print(f"  VMC: {vmc_final:.6f} (error: {history_vmc['energy_error'][-1]:.6f})")
    
    # Create results directory
    os.makedirs("results/sr", exist_ok=True)
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    
    histories = {
        'SR': history_sr,
        'VMC': history_vmc
    }
    
    plot_convergence_comparison(
        histories,
        save_path="results/sr/sr_vs_vmc.png",
        show=False
    )
    print("  Saved: results/sr/sr_vs_vmc.png")
    
    # Individual training histories
    plot_training_history(
        history_sr,
        save_path="results/sr/sr_training.png",
        show=False
    )
    print("  Saved: results/sr/sr_training.png")
    
    plot_training_history(
        history_vmc,
        save_path="results/sr/vmc_training.png",
        show=False
    )
    print("  Saved: results/sr/vmc_training.png")
    
    print("\n" + "=" * 60)
    print("SR optimization completed!")
    print("=" * 60)
    
    # Observations
    print("\nKey Observations:")
    print("  - SR typically converges faster than VMC")
    print("  - SR uses natural gradient (quantum geometric tensor)")
    print("  - SR requires solving linear system at each step")
    print("  - VMC is simpler but may need more iterations")


if __name__ == "__main__":
    main()
