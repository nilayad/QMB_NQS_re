"""MCMC sampling for Neural Quantum States.

Implements Metropolis-Hastings algorithm for sampling from |ψ(s)|².
"""

import numpy as np
import tensorflow as tf
from typing import Callable, Tuple, Optional, Dict
from tqdm import tqdm


class MetropolisSampler:
    """Metropolis-Hastings sampler for NQS.
    
    Samples configurations from the probability distribution P(s) = |ψ(s)|²/Z.
    """
    
    def __init__(self, 
                 num_sites: int,
                 num_samples: int = 1000,
                 num_chains: int = 1,
                 warmup_steps: int = 100,
                 sweep_factor: int = 1):
        """Initialize Metropolis sampler.
        
        Args:
            num_sites: Number of lattice sites
            num_samples: Number of samples to generate per chain
            num_chains: Number of parallel Markov chains
            warmup_steps: Thermalization steps before sampling
            sweep_factor: Number of single-spin flips per sweep (multiplier of num_sites)
        """
        self.num_sites = num_sites
        self.num_samples = num_samples
        self.num_chains = num_chains
        self.warmup_steps = warmup_steps
        self.sweep_factor = sweep_factor
        self.steps_per_sweep = num_sites * sweep_factor
        
        # Statistics
        self.acceptance_rate = 0.0
    
    def _initialize_chains(self) -> np.ndarray:
        """Initialize random spin configurations.
        
        Returns:
            Initial configurations shape (num_chains, num_sites) with ±1 spins
        """
        return 2 * np.random.randint(0, 2, size=(self.num_chains, self.num_sites)) - 1
    
    def _propose_flip(self, configs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Propose single-spin flips.
        
        Args:
            configs: Current configurations (num_chains, num_sites)
            
        Returns:
            Tuple of (proposed_configs, flipped_sites)
        """
        proposed = configs.copy()
        sites = np.random.randint(0, self.num_sites, size=self.num_chains)
        
        for chain_idx, site in enumerate(sites):
            proposed[chain_idx, site] *= -1
        
        return proposed, sites
    
    def _acceptance_probability(self,
                                current_log_psi: tf.Tensor,
                                proposed_log_psi: tf.Tensor) -> np.ndarray:
        """Compute acceptance probability.
        
        P_accept = min(1, |ψ(s')|² / |ψ(s)|²) = min(1, exp(2[log|ψ(s')| - log|ψ(s)|]))
        
        Args:
            current_log_psi: log|ψ(s)| for current configs
            proposed_log_psi: log|ψ(s')| for proposed configs
            
        Returns:
            Acceptance probabilities for each chain
        """
        log_prob_ratio = 2.0 * (proposed_log_psi - current_log_psi)
        return np.minimum(1.0, np.exp(log_prob_ratio.numpy()))
    
    def sample(self,
               model: Callable[[np.ndarray], tf.Tensor],
               initial_configs: Optional[np.ndarray] = None,
               show_progress: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """Generate samples using Metropolis-Hastings.
        
        Args:
            model: Function that computes log|ψ(s)| from spin configurations
            initial_configs: Initial configurations (num_chains, num_sites)
            show_progress: Show progress bar
            
        Returns:
            Tuple of (samples, statistics)
            samples: Array of shape (num_samples * num_chains, num_sites)
            statistics: Dictionary with sampling statistics
        """
        # Initialize
        if initial_configs is None:
            configs = self._initialize_chains()
        else:
            configs = initial_configs.copy()
        
        current_log_psi = model(configs)
        
        samples = []
        total_steps = self.warmup_steps + self.num_samples * self.steps_per_sweep
        accepted = 0
        total_proposals = 0
        
        # Create progress bar
        pbar = tqdm(total=total_steps, disable=not show_progress,
                   desc="MCMC sampling")
        
        for step in range(total_steps):
            # Propose flip
            proposed_configs, _ = self._propose_flip(configs)
            proposed_log_psi = model(proposed_configs)
            
            # Accept/reject
            accept_prob = self._acceptance_probability(current_log_psi, proposed_log_psi)
            accept_mask = np.random.random(self.num_chains) < accept_prob
            
            # Update accepted chains
            configs[accept_mask] = proposed_configs[accept_mask]
            current_log_psi = tf.where(accept_mask, proposed_log_psi, current_log_psi)
            
            accepted += np.sum(accept_mask)
            total_proposals += self.num_chains
            
            # Collect samples after warmup
            if step >= self.warmup_steps and (step - self.warmup_steps) % self.steps_per_sweep == 0:
                samples.append(configs.copy())
            
            pbar.update(1)
        
        pbar.close()
        
        # Combine samples from all chains
        samples = np.concatenate(samples, axis=0)
        
        # Compute statistics
        self.acceptance_rate = accepted / total_proposals
        statistics = {
            'acceptance_rate': self.acceptance_rate,
            'num_samples': len(samples),
            'num_chains': self.num_chains,
            'warmup_steps': self.warmup_steps
        }
        
        return samples, statistics


class ParallelTempering:
    """Parallel tempering (replica exchange) sampler.
    
    Runs multiple chains at different temperatures to improve mixing.
    """
    
    def __init__(self,
                 num_sites: int,
                 num_samples: int = 1000,
                 num_replicas: int = 4,
                 temp_range: Tuple[float, float] = (0.1, 2.0),
                 warmup_steps: int = 100,
                 exchange_interval: int = 10):
        """Initialize parallel tempering sampler.
        
        Args:
            num_sites: Number of lattice sites
            num_samples: Number of samples from lowest temperature
            num_replicas: Number of temperature replicas
            temp_range: (min_temp, max_temp) temperature range
            warmup_steps: Thermalization steps
            exchange_interval: Steps between replica exchange attempts
        """
        self.num_sites = num_sites
        self.num_samples = num_samples
        self.num_replicas = num_replicas
        self.warmup_steps = warmup_steps
        self.exchange_interval = exchange_interval
        
        # Set up temperature ladder (geometric spacing)
        self.temperatures = np.geomspace(temp_range[0], temp_range[1], num_replicas)
        self.beta = 1.0 / self.temperatures
        
        # Create sampler for each replica
        self.samplers = [
            MetropolisSampler(num_sites, num_samples, num_chains=1, 
                            warmup_steps=0, sweep_factor=1)
            for _ in range(num_replicas)
        ]
        
        self.exchange_rate = 0.0
    
    def _exchange_replicas(self,
                          configs: np.ndarray,
                          log_psi: tf.Tensor) -> Tuple[np.ndarray, tf.Tensor, int]:
        """Attempt replica exchange.
        
        Args:
            configs: Current configurations for all replicas
            log_psi: log|ψ(s)| for all replicas
            
        Returns:
            Tuple of (updated_configs, updated_log_psi, num_exchanges)
        """
        num_exchanges = 0
        
        # Try exchanging adjacent temperature pairs
        for i in range(self.num_replicas - 1):
            # Exchange probability
            delta_beta = self.beta[i] - self.beta[i + 1]
            delta_log_psi = log_psi[i] - log_psi[i + 1]
            log_prob = 2.0 * delta_beta * delta_log_psi
            
            if np.log(np.random.random()) < log_prob.numpy():
                # Exchange configurations
                configs[[i, i + 1]] = configs[[i + 1, i]]
                log_psi = tf.tensor_scatter_nd_update(
                    log_psi, [[i], [i + 1]], [log_psi[i + 1], log_psi[i]])
                num_exchanges += 1
        
        return configs, log_psi, num_exchanges
    
    def sample(self,
               model: Callable[[np.ndarray], tf.Tensor],
               show_progress: bool = True) -> Tuple[np.ndarray, Dict[str, float]]:
        """Sample using parallel tempering.
        
        Args:
            model: Function computing log|ψ(s)|
            show_progress: Show progress bar
            
        Returns:
            Samples from the lowest temperature (target distribution)
        """
        # Initialize all replicas
        configs = 2 * np.random.randint(0, 2, 
                                       size=(self.num_replicas, self.num_sites)) - 1
        log_psi = model(configs)
        
        samples = []
        total_exchanges = 0
        exchange_attempts = 0
        
        total_steps = self.warmup_steps + self.num_samples
        pbar = tqdm(total=total_steps, disable=not show_progress,
                   desc="Parallel tempering")
        
        for step in range(total_steps):
            # Update each replica
            for replica_idx in range(self.num_replicas):
                # Single Metropolis step at this temperature
                proposed_config, _ = self.samplers[replica_idx]._propose_flip(
                    configs[replica_idx:replica_idx+1])
                proposed_log_psi = model(proposed_config)
                
                # Temperature-scaled acceptance
                log_prob_ratio = 2.0 * self.beta[replica_idx] * (
                    proposed_log_psi - log_psi[replica_idx])
                
                if np.log(np.random.random()) < log_prob_ratio.numpy()[0]:
                    configs[replica_idx] = proposed_config[0]
                    log_psi = tf.tensor_scatter_nd_update(
                        log_psi, [[replica_idx]], [proposed_log_psi[0]])
            
            # Attempt replica exchange
            if step % self.exchange_interval == 0:
                configs, log_psi, num_ex = self._exchange_replicas(configs, log_psi)
                total_exchanges += num_ex
                exchange_attempts += self.num_replicas - 1
            
            # Collect samples from lowest temperature after warmup
            if step >= self.warmup_steps:
                samples.append(configs[0].copy())
            
            pbar.update(1)
        
        pbar.close()
        
        samples = np.array(samples)
        self.exchange_rate = total_exchanges / exchange_attempts if exchange_attempts > 0 else 0.0
        
        statistics = {
            'exchange_rate': self.exchange_rate,
            'temperatures': self.temperatures.tolist(),
            'num_samples': len(samples),
        }
        
        return samples, statistics


def compute_autocorrelation(samples: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """Compute autocorrelation function of samples.
    
    Args:
        samples: Time series data (num_samples, ...)
        max_lag: Maximum lag to compute
        
    Returns:
        Autocorrelation function
    """
    n = len(samples)
    max_lag = min(max_lag, n // 2)
    
    # Flatten samples for autocorrelation
    if samples.ndim > 1:
        samples_flat = samples.reshape(n, -1).mean(axis=1)
    else:
        samples_flat = samples
    
    mean = np.mean(samples_flat)
    var = np.var(samples_flat)
    
    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        autocorr[lag] = np.mean(
            (samples_flat[:n-lag] - mean) * (samples_flat[lag:] - mean)) / var
    
    return autocorr


def estimate_correlation_time(samples: np.ndarray, max_lag: int = 100) -> float:
    """Estimate integrated autocorrelation time.
    
    τ_int = 1 + 2 Σ_{t=1}^{t_max} ρ(t)
    
    Args:
        samples: Time series data
        max_lag: Maximum lag for autocorrelation
        
    Returns:
        Integrated autocorrelation time
    """
    autocorr = compute_autocorrelation(samples, max_lag)
    
    # Find where autocorrelation becomes small (< 1/e)
    cutoff = np.where(autocorr < np.exp(-1))[0]
    if len(cutoff) > 0:
        t_max = cutoff[0]
    else:
        t_max = len(autocorr)
    
    tau_int = 1.0 + 2.0 * np.sum(autocorr[1:t_max])
    return max(1.0, tau_int)  # At least 1
