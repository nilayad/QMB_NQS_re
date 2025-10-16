"""Stochastic Reconfiguration optimizer for Neural Quantum States.

Implements natural gradient descent using the quantum geometric tensor (Fubini-Study metric).
"""

import tensorflow as tf
import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy import linalg
import time

from .sampling import MetropolisSampler


class StochasticReconfiguration:
    """Stochastic Reconfiguration (SR) optimizer.
    
    Uses natural gradient: θ_{t+1} = θ_t - η S^{-1} F
    where S is the quantum geometric tensor and F is the force (energy gradient).
    """
    
    def __init__(self,
                 model: tf.keras.Model,
                 hamiltonian,
                 sampler: MetropolisSampler,
                 learning_rate: float = 0.01,
                 diagonal_shift: float = 0.01,
                 use_iterative: bool = False,
                 max_iter: int = 1000):
        """Initialize SR optimizer.
        
        Args:
            model: NQS model
            hamiltonian: Hamiltonian operator
            sampler: MCMC sampler
            learning_rate: Learning rate (preconditioner scale)
            diagonal_shift: Regularization for S matrix (typically 0.001-0.1)
            use_iterative: Use iterative solver (conjugate gradient) instead of direct
            max_iter: Maximum iterations for iterative solver
        """
        self.model = model
        self.hamiltonian = hamiltonian
        self.sampler = sampler
        self.learning_rate = learning_rate
        self.diagonal_shift = diagonal_shift
        self.use_iterative = use_iterative
        self.max_iter = max_iter
        
        # Training history
        self.history = {
            'energy': [],
            'energy_error': [],
            'variance': [],
            'gradient_norm': [],
            'sr_condition_number': [],
            'time_per_step': []
        }
    
    def _compute_local_energies(self, samples: np.ndarray) -> np.ndarray:
        """Compute local energies.
        
        Args:
            samples: Spin configurations
            
        Returns:
            Local energies
        """
        log_psi = self.model(samples, training=False)
        
        def log_psi_func(configs):
            return self.model(configs, training=False)
        
        from .operators import spins_to_config
        configs = np.array([spins_to_config(s) for s in samples])
        e_loc = self.hamiltonian.local_energy(configs, log_psi_func)
        
        return e_loc
    
    def _compute_log_derivatives(self, samples: tf.Tensor) -> List[tf.Tensor]:
        """Compute derivatives of log wavefunction.
        
        O_k = ∂log(ψ)/∂θ_k
        
        Args:
            samples: Spin configurations
            
        Returns:
            List of gradients (one per parameter)
        """
        with tf.GradientTape() as tape:
            log_psi = self.model(samples, training=True)
            # Sum to get total log amplitude
            loss = tf.reduce_sum(log_psi)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        
        return gradients
    
    def _compute_quantum_geometric_tensor(self,
                                         samples: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Compute quantum geometric tensor (Fubini-Study metric).
        
        S_kl = ⟨O_k^* O_l⟩ - ⟨O_k^*⟩⟨O_l⟩
        
        Args:
            samples: Spin configurations
            
        Returns:
            Tuple of (S_matrix, mean_derivatives)
        """
        samples_tf = tf.constant(samples, dtype=tf.float32)
        
        # Compute log derivatives for all samples
        batch_size = len(samples)
        all_derivatives = []
        
        # Process in batches to avoid memory issues
        batch_sample_size = min(100, batch_size)
        
        for i in range(0, batch_size, batch_sample_size):
            batch = samples_tf[i:i+batch_sample_size]
            derivs = self._compute_log_derivatives(batch)
            
            # Convert to numpy and flatten
            batch_derivs = []
            for d in derivs:
                if d is not None:
                    batch_derivs.append(d.numpy())
            
            all_derivatives.append(batch_derivs)
        
        # Concatenate batches
        num_params = len(all_derivatives[0])
        concatenated_derivs = []
        
        for param_idx in range(num_params):
            param_derivs = []
            for batch_derivs in all_derivatives:
                # Flatten parameter derivatives
                flat = batch_derivs[param_idx].reshape(
                    batch_derivs[param_idx].shape[0], -1)
                param_derivs.append(flat)
            concatenated_derivs.append(np.concatenate(param_derivs, axis=0))
        
        # Stack all parameters
        O = np.concatenate([d for d in concatenated_derivs], axis=1)  # (batch, total_params)
        
        # Compute mean
        O_mean = np.mean(O, axis=0, keepdims=True)
        
        # Center
        O_centered = O - O_mean
        
        # Compute S = ⟨(O - ⟨O⟩)(O - ⟨O⟩)^T⟩
        S = np.dot(O_centered.T, O_centered) / batch_size
        
        # Add diagonal regularization
        S += self.diagonal_shift * np.eye(S.shape[0])
        
        return S, concatenated_derivs
    
    def _compute_force(self,
                       samples: np.ndarray,
                       e_loc: np.ndarray,
                       derivatives: List[np.ndarray]) -> np.ndarray:
        """Compute force vector.
        
        F_k = ⟨E_loc O_k⟩ - ⟨E_loc⟩⟨O_k⟩
        
        Args:
            samples: Spin configurations
            e_loc: Local energies
            derivatives: Log derivatives for each parameter
            
        Returns:
            Force vector (flattened)
        """
        samples_tf = tf.constant(samples, dtype=tf.float32)
        e_loc_centered = e_loc - np.mean(e_loc)
        
        # Compute force for each parameter
        forces = []
        
        for derivs in derivatives:
            # derivs has shape (batch_size, *param_shape)
            # Flatten parameter dimensions
            batch_size = derivs.shape[0]
            derivs_flat = derivs.reshape(batch_size, -1)
            
            # F = ⟨E_loc * O⟩ - ⟨E_loc⟩⟨O⟩ = ⟨(E_loc - ⟨E_loc⟩) * O⟩
            force = np.dot(e_loc_centered, derivs_flat) / batch_size
            forces.append(force)
        
        # Concatenate all forces
        F = np.concatenate(forces)
        
        return F
    
    def _solve_sr_equation(self, S: np.ndarray, F: np.ndarray) -> np.ndarray:
        """Solve S * x = F for natural gradient.
        
        Args:
            S: Quantum geometric tensor
            F: Force vector
            
        Returns:
            Natural gradient x
        """
        if self.use_iterative:
            # Use conjugate gradient
            from scipy.sparse.linalg import cg
            x, info = cg(S, F, maxiter=self.max_iter, atol=1e-5)
            if info != 0:
                print(f"Warning: CG did not converge (info={info})")
        else:
            # Direct solve
            try:
                x = linalg.solve(S, F, assume_a='pos')
            except linalg.LinAlgError:
                # Fallback to least squares if singular
                x = linalg.lstsq(S, F)[0]
        
        return x
    
    def _unflatten_update(self, flat_update: np.ndarray) -> List[np.ndarray]:
        """Convert flat update back to parameter shapes.
        
        Args:
            flat_update: Flattened parameter update
            
        Returns:
            List of updates matching model parameters
        """
        updates = []
        idx = 0
        
        for var in self.model.trainable_variables:
            param_size = np.prod(var.shape)
            update = flat_update[idx:idx+param_size]
            updates.append(update.reshape(var.shape))
            idx += param_size
        
        return updates
    
    def train_step(self, show_progress: bool = False) -> Dict[str, float]:
        """Perform one SR training step.
        
        Args:
            show_progress: Show progress bar
            
        Returns:
            Step metrics
        """
        start_time = time.time()
        
        # Sample configurations
        samples, _ = self.sampler.sample(
            lambda s: self.model(s, training=False),
            show_progress=show_progress
        )
        
        # Compute local energies
        e_loc = self._compute_local_energies(samples)
        
        # Compute quantum geometric tensor and derivatives
        S, derivatives = self._compute_quantum_geometric_tensor(samples)
        
        # Compute force
        F = self._compute_force(samples, e_loc, derivatives)
        
        # Solve SR equation: S * update = F
        flat_update = self._solve_sr_equation(S, F)
        
        # Unflatten and apply update
        updates = self._unflatten_update(flat_update)
        
        for var, update in zip(self.model.trainable_variables, updates):
            var.assign_sub(self.learning_rate * update)
        
        # Compute metrics
        energy = np.mean(e_loc)
        energy_std = np.std(e_loc) / np.sqrt(len(e_loc))
        variance = np.var(e_loc)
        grad_norm = np.linalg.norm(flat_update)
        
        # Condition number of S
        try:
            cond = np.linalg.cond(S)
        except:
            cond = np.inf
        
        step_time = time.time() - start_time
        
        metrics = {
            'energy': float(energy),
            'energy_error': float(energy_std),
            'variance': float(variance),
            'gradient_norm': float(grad_norm),
            'sr_condition_number': float(cond),
            'time_per_step': step_time
        }
        
        return metrics
    
    def train(self,
              num_steps: int,
              log_interval: int = 10,
              show_progress: bool = True) -> Dict[str, List[float]]:
        """Train using Stochastic Reconfiguration.
        
        Args:
            num_steps: Number of training steps
            log_interval: Logging interval
            show_progress: Show progress bar
            
        Returns:
            Training history
        """
        from tqdm import tqdm
        
        pbar = tqdm(range(num_steps), disable=not show_progress, desc="SR training")
        
        for step in pbar:
            # Training step
            metrics = self.train_step(show_progress=False)
            
            # Log
            for key, value in metrics.items():
                self.history[key].append(value)
            
            # Update progress
            if step % log_interval == 0:
                pbar.set_postfix({
                    'E': f"{metrics['energy']:.6f}",
                    'σ': f"{metrics['energy_error']:.6f}",
                    'cond': f"{metrics['sr_condition_number']:.2e}"
                })
        
        pbar.close()
        
        return self.history


class StatePreparation:
    """State preparation using imaginary time evolution with SR.
    
    Prepares ground state by evolving in imaginary time: |ψ(τ)⟩ = e^(-τH)|ψ(0)⟩
    """
    
    def __init__(self,
                 model: tf.keras.Model,
                 hamiltonian,
                 sampler: MetropolisSampler,
                 target_state: Optional[np.ndarray] = None,
                 time_step: float = 0.01):
        """Initialize state preparation.
        
        Args:
            model: NQS model
            hamiltonian: Hamiltonian
            sampler: Sampler
            target_state: Target state for fidelity tracking (optional)
            time_step: Imaginary time step
        """
        self.sr_optimizer = StochasticReconfiguration(
            model, hamiltonian, sampler,
            learning_rate=time_step,
            diagonal_shift=0.01
        )
        self.target_state = target_state
        
        self.history = {
            'energy': [],
            'fidelity': [],
            'time': []
        }
    
    def prepare(self,
                num_steps: int,
                fidelity_interval: int = 10,
                show_progress: bool = True) -> Dict[str, List[float]]:
        """Prepare ground state.
        
        Args:
            num_steps: Number of imaginary time steps
            fidelity_interval: Steps between fidelity computation
            show_progress: Show progress
            
        Returns:
            Preparation history
        """
        from tqdm import tqdm
        
        pbar = tqdm(range(num_steps), disable=not show_progress, 
                   desc="State preparation")
        
        for step in pbar:
            # SR step (imaginary time evolution)
            metrics = self.sr_optimizer.train_step(show_progress=False)
            
            self.history['energy'].append(metrics['energy'])
            self.history['time'].append(step * self.sr_optimizer.learning_rate)
            
            # Compute fidelity if target state provided
            if self.target_state is not None and step % fidelity_interval == 0:
                from .training import compute_fidelity
                # Placeholder - would need proper implementation
                fidelity = 0.0  # compute_fidelity(self.sr_optimizer.model, target)
                self.history['fidelity'].append(fidelity)
            
            pbar.set_postfix({'E': f"{metrics['energy']:.6f}"})
        
        pbar.close()
        
        return self.history
