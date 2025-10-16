"""Statistical diagnostics for VMC simulations.

Tools for error analysis, autocorrelation, and blocking methods.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy import stats


def blocking_analysis(data: np.ndarray, 
                     max_block_size: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Perform blocking analysis for error estimation.
    
    The blocking method estimates the error by grouping correlated samples
    into blocks and computing statistics on block averages.
    
    Args:
        data: Time series data (num_samples,)
        max_block_size: Maximum block size (default: len(data)//4)
        
    Returns:
        Dictionary containing:
            - block_sizes: Array of block sizes
            - errors: Error estimates for each block size
            - plateau_error: Best error estimate (from plateau)
    """
    n = len(data)
    
    if max_block_size is None:
        max_block_size = n // 4
    
    block_sizes = []
    errors = []
    
    block_size = 1
    while block_size <= max_block_size:
        n_blocks = n // block_size
        if n_blocks < 2:
            break
        
        # Reshape into blocks
        blocked = data[:n_blocks * block_size].reshape(n_blocks, block_size)
        block_means = np.mean(blocked, axis=1)
        
        # Standard error of block means
        error = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
        
        block_sizes.append(block_size)
        errors.append(error)
        
        block_size *= 2
    
    block_sizes = np.array(block_sizes)
    errors = np.array(errors)
    
    # Find plateau (where error stabilizes)
    if len(errors) > 3:
        # Use maximum as conservative estimate
        plateau_error = np.max(errors)
    else:
        plateau_error = errors[-1] if len(errors) > 0 else np.std(data) / np.sqrt(n)
    
    return {
        'block_sizes': block_sizes,
        'errors': errors,
        'plateau_error': plateau_error
    }


def jackknife_error(data: np.ndarray, 
                   func: Optional[callable] = None) -> Tuple[float, float]:
    """Compute jackknife error estimate.
    
    Args:
        data: Sample data
        func: Function to compute statistic (default: mean)
        
    Returns:
        Tuple of (estimate, error)
    """
    if func is None:
        func = np.mean
    
    n = len(data)
    
    # Full sample estimate
    theta = func(data)
    
    # Leave-one-out estimates
    theta_i = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        theta_i[i] = func(data[mask])
    
    # Jackknife estimate
    theta_jack = n * theta - (n - 1) * np.mean(theta_i)
    
    # Jackknife error
    error = np.sqrt((n - 1) * np.mean((theta_i - np.mean(theta_i))**2))
    
    return theta_jack, error


def bootstrap_error(data: np.ndarray,
                   func: Optional[callable] = None,
                   num_bootstrap: int = 1000,
                   confidence: float = 0.95) -> Dict[str, float]:
    """Compute bootstrap error estimate and confidence intervals.
    
    Args:
        data: Sample data
        func: Function to compute statistic (default: mean)
        num_bootstrap: Number of bootstrap samples
        confidence: Confidence level for intervals
        
    Returns:
        Dictionary with estimate, error, and confidence intervals
    """
    if func is None:
        func = np.mean
    
    n = len(data)
    
    # Original estimate
    theta = func(data)
    
    # Bootstrap samples
    bootstrap_estimates = np.zeros(num_bootstrap)
    for i in range(num_bootstrap):
        indices = np.random.randint(0, n, size=n)
        bootstrap_estimates[i] = func(data[indices])
    
    # Error estimate
    error = np.std(bootstrap_estimates, ddof=1)
    
    # Confidence intervals
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_estimates, 100 * alpha / 2)
    upper = np.percentile(bootstrap_estimates, 100 * (1 - alpha / 2))
    
    return {
        'estimate': theta,
        'error': error,
        'lower_ci': lower,
        'upper_ci': upper,
        'bootstrap_samples': bootstrap_estimates
    }


def integrated_autocorrelation_time(data: np.ndarray,
                                   max_lag: Optional[int] = None,
                                   window: str = 'auto') -> Tuple[float, np.ndarray]:
    """Compute integrated autocorrelation time.
    
    τ_int = 1/2 + Σ_{t=1}^{W} ρ(t)
    
    Args:
        data: Time series data
        max_lag: Maximum lag to compute (default: len(data)//2)
        window: Window size ('auto' or integer)
        
    Returns:
        Tuple of (tau_int, autocorrelation_function)
    """
    n = len(data)
    
    if max_lag is None:
        max_lag = n // 2
    
    # Compute autocorrelation
    mean = np.mean(data)
    var = np.var(data)
    
    if var == 0:
        return 1.0, np.zeros(1)
    
    autocorr = np.zeros(max_lag)
    for lag in range(max_lag):
        autocorr[lag] = np.mean((data[:n-lag] - mean) * (data[lag:] - mean)) / var
    
    # Determine window
    if window == 'auto':
        # Find where autocorr becomes small and oscillatory
        # Use criterion: autocorr[t] < exp(-t/tau_int)
        tau_int = 0.5
        for t in range(1, max_lag):
            tau_int += autocorr[t]
            
            # Check for convergence
            if t > 5 * tau_int:
                break
            
            # Check if autocorr becomes too small
            if autocorr[t] < 0.01:
                break
        
        tau_int = max(0.5, tau_int)
    else:
        W = int(window)
        tau_int = 0.5 + np.sum(autocorr[1:W+1])
    
    return tau_int, autocorr


def effective_sample_size(data: np.ndarray) -> float:
    """Compute effective sample size accounting for autocorrelation.
    
    n_eff = n / (2 * τ_int)
    
    Args:
        data: Time series data
        
    Returns:
        Effective number of independent samples
    """
    n = len(data)
    tau_int, _ = integrated_autocorrelation_time(data)
    
    n_eff = n / (2.0 * tau_int)
    
    return max(1.0, n_eff)


def geweke_test(data: np.ndarray,
               first_frac: float = 0.1,
               last_frac: float = 0.5,
               num_intervals: int = 20) -> Dict[str, np.ndarray]:
    """Geweke convergence diagnostic.
    
    Compares means from early and late parts of the chain.
    
    Args:
        data: Time series data
        first_frac: Fraction of chain for first window
        last_frac: Fraction of chain for last window
        num_intervals: Number of test intervals
        
    Returns:
        Dictionary with z-scores and p-values
    """
    n = len(data)
    
    # First window
    first_size = int(first_frac * n)
    first_data = data[:first_size]
    first_mean = np.mean(first_data)
    first_var = np.var(first_data, ddof=1) / first_size
    
    # Test multiple intervals in last part
    last_size = int(last_frac * n)
    interval_size = last_size // num_intervals
    
    z_scores = []
    p_values = []
    
    for i in range(num_intervals):
        start = n - last_size + i * interval_size
        end = start + interval_size
        
        if end > n:
            break
        
        last_data = data[start:end]
        last_mean = np.mean(last_data)
        last_var = np.var(last_data, ddof=1) / len(last_data)
        
        # Z-score
        z = (first_mean - last_mean) / np.sqrt(first_var + last_var)
        z_scores.append(z)
        
        # P-value (two-tailed)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        p_values.append(p)
    
    return {
        'z_scores': np.array(z_scores),
        'p_values': np.array(p_values),
        'converged': np.all(np.array(p_values) > 0.05)  # 95% confidence
    }


def gelman_rubin_diagnostic(chains: np.ndarray) -> float:
    """Gelman-Rubin convergence diagnostic (R-hat).
    
    Compares within-chain and between-chain variance.
    R-hat should be close to 1 for convergence.
    
    Args:
        chains: Multiple chains, shape (num_chains, num_samples)
        
    Returns:
        R-hat statistic
    """
    num_chains, n = chains.shape
    
    # Chain means
    chain_means = np.mean(chains, axis=1)
    overall_mean = np.mean(chain_means)
    
    # Between-chain variance
    B = n / (num_chains - 1) * np.sum((chain_means - overall_mean)**2)
    
    # Within-chain variance
    chain_vars = np.var(chains, axis=1, ddof=1)
    W = np.mean(chain_vars)
    
    # Pooled variance estimate
    var_plus = ((n - 1) / n) * W + (1 / n) * B
    
    # R-hat
    R_hat = np.sqrt(var_plus / W)
    
    return R_hat


def compute_running_average(data: np.ndarray, 
                           window: int = 100) -> np.ndarray:
    """Compute running average.
    
    Args:
        data: Time series
        window: Window size for averaging
        
    Returns:
        Running average
    """
    if window == 1:
        return data
    
    cumsum = np.cumsum(np.insert(data, 0, 0))
    running_avg = (cumsum[window:] - cumsum[:-window]) / window
    
    # Pad beginning
    padded = np.concatenate([
        np.cumsum(data[:window]) / np.arange(1, window + 1),
        running_avg
    ])
    
    return padded


def detect_outliers(data: np.ndarray, 
                   threshold: float = 3.0) -> np.ndarray:
    """Detect outliers using modified Z-score.
    
    Args:
        data: Data array
        threshold: Number of standard deviations for outlier
        
    Returns:
        Boolean mask of outliers
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    
    if mad == 0:
        # Use standard deviation instead
        std = np.std(data)
        if std == 0:
            return np.zeros(len(data), dtype=bool)
        modified_z_scores = np.abs(data - np.mean(data)) / std
    else:
        modified_z_scores = 0.6745 * np.abs(data - median) / mad
    
    return modified_z_scores > threshold
