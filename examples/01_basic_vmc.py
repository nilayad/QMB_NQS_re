"""Basic VMC training example for TFIM ground state."""

import sys
from pathlib import Path
import yaml
import tensorflow as tf
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from nqs import MLP, TFIMHamiltonian, VMCTrainer, MetropolisSampler


def main():
    """Run basic VMC training."""
    
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config' / 'vmc_config.yaml'
    
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        L = config['model']['L']
        J = config['hamiltonian']['J']
        h = config['hamiltonian']['h']
        hidden_sizes = config['model'].get('hidden_sizes', [64, 64])
        n_epochs = config['training']['n_epochs']
        n_samples = config['training']['n_samples']
        n_steps = config['training']['n_mcmc_steps']
        learning_rate = config['training'].get('learning_rate', 0.001)
    except Exception as e:
        print(f"⚠ Could not load config, using defaults: {e}")
        L = 8
        J = 1.0
        h = 0.5
        hidden_sizes = [64, 128, 64]
        n_epochs = 500
        n_samples = 2000
        n_steps = 100
        learning_rate = 0.001
    
    print("=" * 60)
    print("Basic VMC Training for TFIM Ground State")
    print("=" * 60)
    print(f"System size: L = {L}")
    print(f"Coupling: J = {J}, Transverse field: h = {h}")
    print(f"Architecture: MLP {hidden_sizes}")
    print(f"Training: {n_epochs} epochs, {n_samples} samples, {n_steps} MCMC steps")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)
    print()
    
    # Step 1: Create MLP model
    print("Creating MLP model...")
    model = MLP(
        num_sites=L,
        hidden_sizes=hidden_sizes,
        activation='tanh',
        use_complex=False
    )
    
    # Build model by calling it once
    dummy_input = tf.ones((1, L))
    _ = model(dummy_input)
    
    print(f"✓ Model created with {model.count_params()} parameters")
    print()
    
    # Step 2: Create Hamiltonian
    print("Creating TFIM Hamiltonian...")
    hamiltonian = TFIMHamiltonian(
        num_sites=L,
        J=J,
        h=h,
        pbc=True
    )
    print("✓ Hamiltonian created")
    print()
    
    # Step 3: Create MCMC Sampler
    print("Creating Metropolis sampler...")
    sampler = MetropolisSampler(model, num_sites=L)
    print("✓ Sampler created")
    print()
    
    # Step 4: Create VMC Trainer
    print("Creating VMC trainer...")
    trainer = VMCTrainer(
        model=model,
        hamiltonian=hamiltonian,
        sampler=sampler,
        learning_rate=learning_rate
    )
    print("✓ VMC trainer created")
    print()
    
    # Step 5: Run training
    print("=" * 60)
    print("Starting VMC Training")
    print("=" * 60)
    print()
    
    try:
        history = trainer.train(
            num_epochs=n_epochs,
            num_samples=n_samples,
            mcmc_steps=n_steps,
            verbose=True
        )
        
        # Print results
        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        if 'energy' in history and len(history['energy']) > 0:
            final_energy = history['energy'][-1]
            print(f"Final Energy: {final_energy:.6f}")
            
            if 'variance' in history and len(history['variance']) > 0:
                final_variance = history['variance'][-1]
                print(f"Energy Variance: {final_variance:.6f}")
            
            # Simple error estimate from last 100 samples
            if len(history['energy']) > 100:
                recent_energies = history['energy'][-100:]
                error = np.std(recent_energies) / np.sqrt(len(recent_energies))
                print(f"Energy Error (last 100): ±{error:.6f}")
        
        print("=" * 60)
        
        # Save model
        checkpoint_dir = Path('checkpoints')
        checkpoint_dir.mkdir(exist_ok=True)
        model_path = checkpoint_dir / 'vmc_ground_state.keras'
        model.save(str(model_path))
        print(f"\n✓ Model saved to {model_path}")
        
        # Save training history
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        history_path = results_dir / 'vmc_history.npz'
        np.savez(str(history_path), **history)
        print(f"✓ Training history saved to {history_path}")
        
        return 0
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"❌ Training Error: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
