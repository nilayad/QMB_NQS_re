"""Variational Monte Carlo training for Neural Quantum States.

Implements VMC algorithm with gradient descent optimization.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Dict, List, Callable, Tuple
from tqdm import tqdm
import time

from .sampling import MetropolisSampler, estimate_correlation_time
from .operators import TransverseFieldIsing, Heisenberg


class VMCTrainer:
    """Variational Monte Carlo trainer for NQS.
    
    Minimizes energy expectation value E = ⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ using gradient descent.
    """
    
    def __init__(self,
                 model: tf.keras.Model,
                 hamiltonian,
                 sampler: MetropolisSampler,
                 optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
                 learning_rate: float = 1e-3):
        """Initialize VMC trainer.
        
        Args:
            model: NQS model
            hamiltonian: Hamiltonian operator
            sampler: MCMC sampler
            optimizer: TensorFlow optimizer (defaults to Adam)
            learning_rate: Learning rate
        """
        self.model = model
        self.hamiltonian = hamiltonian
        self.sampler = sampler
        
        if optimizer is None:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        else:
            self.optimizer = optimizer
        
        # Training history
        self.history = {
            'energy': [],
            'energy_error': [],
            'variance': [],
            'acceptance_rate': [],
            'gradient_norm': [],
            'time_per_step': []
        }
    
    def _compute_local_energies(self, samples: np.ndarray) -> np.ndarray:
        """Compute local energies for sampled configurations.
        
        E_loc(s) = ⟨s|H|ψ⟩/⟨s|ψ⟩
        
        Args:
            samples: Spin configurations (num_samples, num_sites)
            
        Returns:
            Local energies
        """
        # Get log amplitudes
        log_psi = self.model(samples, training=False)
        
        # Create callable for Hamiltonian
        def log_psi_func(configs):
            return self.model(configs, training=False)
        
        # Compute local energies using Hamiltonian
        if hasattr(self.hamiltonian, 'local_energy'):
            # Convert samples to config integers if needed
            if isinstance(self.hamiltonian, (TransverseFieldIsing, Heisenberg)):
                from .operators import spins_to_config
                configs = np.array([spins_to_config(s) for s in samples])
                e_loc = self.hamiltonian.local_energy(configs, log_psi_func)
            else:
                e_loc = self.hamiltonian.local_energy(samples, log_psi)
        else:
            raise ValueError("Hamiltonian must implement local_energy method")
        
        return e_loc
    
    @tf.function
    def _compute_energy_and_gradient(self, samples: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor]]:
        """Compute energy and its gradient using automatic differentiation.
        
        Args:
            samples: Spin configurations
            
        Returns:
            Tuple of (energy, variance, gradients)
        """
        with tf.GradientTape() as tape:
            # Forward pass
            log_psi = self.model(samples, training=True)
            
            # Compute local energies (approximate via finite differences for efficiency)
            # In practice, we use the precomputed local energies from sampling
            # Here we return a placeholder for the gradient computation
            energy = tf.constant(0.0, dtype=tf.float32)
        
        # Get gradients
        gradients = tape.gradient(energy, self.model.trainable_variables)
        
        return energy, tf.constant(0.0), gradients
    
    def _compute_vmc_gradient(self, samples: np.ndarray, e_loc: np.ndarray) -> List[tf.Tensor]:
        """Compute VMC gradient: ∇E = 2⟨E_loc ∇log|ψ|⟩ - 2⟨E_loc⟩⟨∇log|ψ|⟩.
        
        Args:
            samples: Spin configurations
            e_loc: Local energies
            
        Returns:
            List of gradients for each trainable variable
        """
        samples_tf = tf.constant(samples, dtype=tf.float32)
        e_loc_tf = tf.constant(e_loc, dtype=tf.float32)
        
        # Center local energies for numerical stability
        e_mean = tf.reduce_mean(e_loc_tf)
        e_loc_centered = e_loc_tf - e_mean
        
        with tf.GradientTape() as tape:
            log_psi = self.model(samples_tf, training=True)
            # Weight by centered local energy
            loss = tf.reduce_mean(e_loc_centered * log_psi)
        
        # Compute gradient
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        # Multiply by 2 (from derivative of |ψ|²)
        gradients = [2.0 * g if g is not None else None for g in gradients]
        
        return gradients
    
    def train_step(self, show_progress: bool = False) -> Dict[str, float]:
        """Perform one VMC training step.
        
        Args:
            show_progress: Show sampling progress bar
            
        Returns:
            Dictionary with step metrics
        """
        start_time = time.time()
        
        # Sample configurations
        samples, sample_stats = self.sampler.sample(
            lambda s: self.model(s, training=False),
            show_progress=show_progress
        )
        
        # Compute local energies
        e_loc = self._compute_local_energies(samples)
        
        # Compute statistics
        energy = np.mean(e_loc)
        energy_var = np.var(e_loc)
        energy_std = np.sqrt(energy_var / len(e_loc))
        
        # Compute and apply gradients
        gradients = self._compute_vmc_gradient(samples, e_loc)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Compute gradient norm
        grad_norm = np.sqrt(sum(
            tf.reduce_sum(g**2).numpy() for g in gradients if g is not None
        ))
        
        step_time = time.time() - start_time
        
        metrics = {
            'energy': float(energy),
            'energy_error': float(energy_std),
            'variance': float(energy_var),
            'acceptance_rate': sample_stats['acceptance_rate'],
            'gradient_norm': float(grad_norm),
            'time_per_step': step_time
        }
        
        return metrics
    
    def train(self, 
              num_steps: int,
              log_interval: int = 10,
              show_progress: bool = True) -> Dict[str, List[float]]:
        """Train NQS using VMC.
        
        Args:
            num_steps: Number of training steps
            log_interval: Steps between logging
            show_progress: Show progress bar
            
        Returns:
            Training history
        """
        pbar = tqdm(range(num_steps), disable=not show_progress, desc="VMC training")
        
        for step in pbar:
            # Training step
            metrics = self.train_step(show_progress=False)
            
            # Log metrics
            for key, value in metrics.items():
                self.history[key].append(value)
            
            # Update progress bar
            if step % log_interval == 0:
                pbar.set_postfix({
                    'E': f"{metrics['energy']:.6f}",
                    'σ': f"{metrics['energy_error']:.6f}",
                    'acc': f"{metrics['acceptance_rate']:.3f}"
                })
        
        pbar.close()
        
        return self.history
    
    def evaluate(self, num_samples: int = 10000) -> Dict[str, float]:
        """Evaluate current model energy.
        
        Args:
            num_samples: Number of samples for evaluation
            
        Returns:
            Evaluation metrics
        """
        # Create temporary sampler with more samples
        eval_sampler = MetropolisSampler(
            self.sampler.num_sites,
            num_samples=num_samples // self.sampler.num_chains,
            num_chains=self.sampler.num_chains,
            warmup_steps=self.sampler.warmup_steps
        )
        
        # Sample
        samples, _ = eval_sampler.sample(
            lambda s: self.model(s, training=False),
            show_progress=False
        )
        
        # Compute energies
        e_loc = self._compute_local_energies(samples)
        
        # Error estimation using blocking method
        error = self._blocking_error(e_loc)
        
        # Autocorrelation analysis
        tau_int = estimate_correlation_time(e_loc)
        
        return {
            'energy': float(np.mean(e_loc)),
            'energy_std': float(np.std(e_loc)),
            'energy_error': float(error),
            'variance': float(np.var(e_loc)),
            'tau_int': float(tau_int),
            'eff_samples': len(e_loc) / tau_int
        }
    
    def _blocking_error(self, data: np.ndarray, max_block_size: int = 512) -> float:
        """Estimate error using blocking method.
        
        Args:
            data: Observable data
            max_block_size: Maximum block size
            
        Returns:
            Error estimate
        """
        n = len(data)
        block_size = 1
        errors = []
        
        while block_size <= min(max_block_size, n // 4):
            # Reshape into blocks
            n_blocks = n // block_size
            blocked = data[:n_blocks * block_size].reshape(n_blocks, block_size)
            block_means = np.mean(blocked, axis=1)
            
            # Error from block means
            error = np.std(block_means) / np.sqrt(n_blocks - 1)
            errors.append(error)
            
            block_size *= 2
        
        # Return maximum (most conservative estimate)
        return max(errors) if errors else np.std(data) / np.sqrt(n)


class AdaptiveLearningRate:
    """Adaptive learning rate scheduler for VMC.
    
    Adjusts learning rate based on energy variance.
    """
    
    def __init__(self,
                 initial_lr: float = 1e-3,
                 patience: int = 10,
                 factor: float = 0.5,
                 min_lr: float = 1e-6):
        """Initialize adaptive scheduler.
        
        Args:
            initial_lr: Initial learning rate
            patience: Steps to wait before reducing
            factor: Reduction factor
            min_lr: Minimum learning rate
        """
        self.initial_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        
        self.current_lr = initial_lr
        self.best_energy = float('inf')
        self.wait = 0
    
    def step(self, energy: float) -> float:
        """Update learning rate based on energy.
        
        Args:
            energy: Current energy
            
        Returns:
            Updated learning rate
        """
        if energy < self.best_energy:
            self.best_energy = energy
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.wait = 0
        
        return self.current_lr


def compute_fidelity(model1: tf.keras.Model,
                    model2: tf.keras.Model,
                    num_samples: int = 1000,
                    num_sites: int = 4) -> float:
    """Compute fidelity between two quantum states.
    
    F = |⟨ψ₁|ψ₂⟩|² / (⟨ψ₁|ψ₁⟩⟨ψ₂|ψ₂⟩)
    
    Args:
        model1: First NQS model
        model2: Second NQS model
        num_samples: Number of samples for estimation
        num_sites: Number of sites
        
    Returns:
        Fidelity estimate
    """
    # Generate random configurations
    samples = 2 * np.random.randint(0, 2, size=(num_samples, num_sites)) - 1
    
    # Get amplitudes
    log_psi1 = model1(samples, training=False).numpy()
    log_psi2 = model2(samples, training=False).numpy()
    
    # Compute overlap
    log_overlap = log_psi1 + log_psi2
    overlap_sq = np.mean(np.exp(2 * log_overlap))
    
    # Normalizations
    norm1_sq = np.mean(np.exp(2 * log_psi1))
    norm2_sq = np.mean(np.exp(2 * log_psi2))
    
    fidelity = overlap_sq / (norm1_sq * norm2_sq)
    
    return float(np.abs(fidelity))
