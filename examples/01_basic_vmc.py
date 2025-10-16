#!/usr/bin/env python3
"""Basic VMC training example for transverse-field Ising model.

This example demonstrates:
- Creating an NQS model (MLP)
- Setting up a Hamiltonian (TFIM)
- Configuring MCMC sampling
- Training with VMC
- Visualizing results
"""

import numpy as np
import tensorflow as tf
import yaml
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nqs import (
    create_nqs_model,
    TransverseFieldIsing,
    MetropolisSampler,
    VMCTrainer
)
from utils import (
    plot_training_history,
    plot_energy_histogram,
    blocking_analysis
)


def main():
    """Run basic VMC training."""
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    print("=" * 60)
    print("Basic VMC Training Example")
    print("=" * 60)
    
    # System parameters
    num_sites = 4
    J = 1.0
    h = 1.0
    
    print(f"\nSystem: Transverse-Field Ising Model")
    print(f"  Sites: {num_sites}")
    print(f"  J (coupling): {J}")
    print(f"  h (field): {h}")
    
    # Create Hamiltonian
    hamiltonian = TransverseFieldIsing(
        num_sites=num_sites,
        J=J,
        h=h,
        pbc=True
    )
    
    # Create NQS model
    print("\nCreating MLP model...")
    model = create_nqs_model(
        model_type='mlp',
        num_sites=num_sites,
        hidden_sizes=[64, 64],
        activation='tanh'
    )
    
    # Setup MCMC sampler
    print("Setting up MCMC sampler...")
    sampler = MetropolisSampler(
        num_sites=num_sites,
        num_samples=1000,
        num_chains=4,
        warmup_steps=100,
        sweep_factor=1
    )
    
    # Create VMC trainer
    print("Initializing VMC trainer...")
    trainer = VMCTrainer(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        learning_rate=0.001
    )
    
    # Train
    print("\nStarting VMC training...")
    print("-" * 60)
    history = trainer.train(
        num_steps=100,
        log_interval=10,
        show_progress=True
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)
    eval_results = trainer.evaluate(num_samples=10000)
    
    print(f"\nFinal Results:")
    print(f"  Energy: {eval_results['energy']:.6f} ± {eval_results['energy_error']:.6f}")
    print(f"  Variance: {eval_results['variance']:.6f}")
    print(f"  τ_int: {eval_results['tau_int']:.2f}")
    print(f"  Effective samples: {eval_results['eff_samples']:.0f}")
    
    # Visualize results
    print("\nGenerating plots...")
    
    # Create results directory
    os.makedirs("results/vmc", exist_ok=True)
    
    # Training history
    plot_training_history(
        history,
        save_path="results/vmc/training_history.png",
        show=False
    )
    print("  Saved: results/vmc/training_history.png")
    
    # Sample for energy histogram
    samples, _ = sampler.sample(
        lambda s: model(s, training=False),
        show_progress=False
    )
    
    # Compute local energies
    from nqs.operators import spins_to_config
    configs = np.array([spins_to_config(s) for s in samples])
    log_psi_func = lambda c: model(np.array([
        [1 - 2*((int(cfg) >> i) & 1) for i in range(num_sites)]
        for cfg in c
    ]), training=False)
    e_loc = hamiltonian.local_energy(configs, log_psi_func)
    
    plot_energy_histogram(
        e_loc,
        save_path="results/vmc/energy_histogram.png",
        show=False
    )
    print("  Saved: results/vmc/energy_histogram.png")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
