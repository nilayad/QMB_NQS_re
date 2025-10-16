"""Visualization utilities for NQS training and analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import matplotlib.gridspec as gridspec


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12


def plot_training_history(history: Dict[str, List[float]],
                         save_path: Optional[str] = None,
                         show: bool = True) -> plt.Figure:
    """Plot VMC/SR training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save figure (optional)
        show: Whether to display the figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.3, wspace=0.3)
    
    # Energy
    ax1 = fig.add_subplot(gs[0, :])
    steps = np.arange(len(history['energy']))
    
    ax1.plot(steps, history['energy'], 'b-', alpha=0.7, label='Energy')
    if 'energy_error' in history:
        energy = np.array(history['energy'])
        error = np.array(history['energy_error'])
        ax1.fill_between(steps, energy - error, energy + error, 
                        alpha=0.2, color='b', label='±1σ')
    
    ax1.set_xlabel('Training Step')
    ax1.set_ylabel('Energy')
    ax1.set_title('Energy Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Variance
    if 'variance' in history:
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(steps, history['variance'], 'r-', alpha=0.7)
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Energy Variance')
        ax2.set_title('Energy Variance')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    # Acceptance rate
    if 'acceptance_rate' in history:
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(steps, history['acceptance_rate'], 'g-', alpha=0.7)
        ax3.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Target')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Acceptance Rate')
        ax3.set_title('MCMC Acceptance Rate')
        ax3.set_ylim([0, 1])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Gradient norm
    if 'gradient_norm' in history:
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(steps, history['gradient_norm'], 'm-', alpha=0.7)
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Gradient Norm')
        ax4.set_title('Gradient Magnitude')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
    
    # Condition number (for SR)
    if 'sr_condition_number' in history:
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.plot(steps, history['sr_condition_number'], 'c-', alpha=0.7)
        ax5.set_xlabel('Training Step')
        ax5.set_ylabel('Condition Number')
        ax5.set_title('S-Matrix Condition Number')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_autocorrelation(autocorr: np.ndarray,
                        max_lag: Optional[int] = None,
                        save_path: Optional[str] = None,
                        show: bool = True) -> plt.Figure:
    """Plot autocorrelation function.
    
    Args:
        autocorr: Autocorrelation values
        max_lag: Maximum lag to plot
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    if max_lag is None:
        max_lag = len(autocorr)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    lags = np.arange(min(len(autocorr), max_lag))
    ax.plot(lags, autocorr[:max_lag], 'b-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax.axhline(y=np.exp(-1), color='r', linestyle='--', alpha=0.5, label='e⁻¹')
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Function')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_blocking_analysis(block_sizes: np.ndarray,
                          errors: np.ndarray,
                          save_path: Optional[str] = None,
                          show: bool = True) -> plt.Figure:
    """Plot blocking analysis results.
    
    Args:
        block_sizes: Block sizes
        errors: Error estimates
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(block_sizes, errors, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=errors[-1], color='r', linestyle='--', alpha=0.5, 
              label=f'Plateau: {errors[-1]:.6f}')
    
    ax.set_xlabel('Block Size')
    ax.set_ylabel('Standard Error')
    ax.set_title('Blocking Analysis')
    ax.set_xscale('log', base=2)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_energy_histogram(energies: np.ndarray,
                         true_energy: Optional[float] = None,
                         save_path: Optional[str] = None,
                         show: bool = True) -> plt.Figure:
    """Plot histogram of local energies.
    
    Args:
        energies: Local energy values
        true_energy: True ground state energy (if known)
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(energies, bins=50, density=True, alpha=0.7, color='b', edgecolor='black')
    
    # Add mean line
    mean_energy = np.mean(energies)
    ax.axvline(mean_energy, color='r', linestyle='--', linewidth=2,
              label=f'Mean: {mean_energy:.6f}')
    
    # Add true energy if provided
    if true_energy is not None:
        ax.axvline(true_energy, color='g', linestyle='--', linewidth=2,
                  label=f'True: {true_energy:.6f}')
    
    ax.set_xlabel('Local Energy')
    ax.set_ylabel('Probability Density')
    ax.set_title('Local Energy Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_spin_correlation(samples: np.ndarray,
                         save_path: Optional[str] = None,
                         show: bool = True) -> plt.Figure:
    """Plot spin-spin correlation function.
    
    C(r) = ⟨σᵢ σᵢ₊ᵣ⟩
    
    Args:
        samples: Spin configurations (num_samples, num_sites)
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    num_samples, num_sites = samples.shape
    
    # Compute correlations
    correlations = np.zeros(num_sites)
    for r in range(num_sites):
        corr = 0.0
        for i in range(num_sites):
            j = (i + r) % num_sites
            corr += np.mean(samples[:, i] * samples[:, j])
        correlations[r] = corr / num_sites
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(num_sites), correlations, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Distance r')
    ax.set_ylabel('Correlation ⟨σᵢσᵢ₊ᵣ⟩')
    ax.set_title('Spin-Spin Correlation Function')
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_wavefunction_amplitudes(model,
                                num_sites: int,
                                save_path: Optional[str] = None,
                                show: bool = True) -> plt.Figure:
    """Plot wavefunction amplitudes for all basis states.
    
    Only practical for small systems (num_sites <= 10).
    
    Args:
        model: NQS model
        num_sites: Number of sites
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    if num_sites > 10:
        print("Warning: Too many states to plot, using num_sites=10")
        num_sites = 10
    
    # Generate all configurations
    from itertools import product
    configs = np.array(list(product([-1, 1], repeat=num_sites)))
    
    # Get amplitudes
    log_psi = model(configs, training=False).numpy()
    amplitudes = np.exp(log_psi)
    
    # Normalize
    amplitudes = amplitudes / np.sqrt(np.sum(amplitudes**2))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bar plot
    x = np.arange(len(amplitudes))
    ax1.bar(x, amplitudes, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Basis State Index')
    ax1.set_ylabel('Amplitude |ψ|')
    ax1.set_title('Wavefunction Amplitudes')
    ax1.grid(True, alpha=0.3)
    
    # Heatmap of configuration patterns
    ax2.imshow(configs.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    ax2.set_xlabel('Basis State Index')
    ax2.set_ylabel('Site')
    ax2.set_title('Spin Configurations')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_convergence_comparison(histories: Dict[str, Dict],
                               save_path: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
    """Compare convergence of different methods.
    
    Args:
        histories: Dictionary of {method_name: history_dict}
        save_path: Path to save figure
        show: Whether to display
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for method_name, history in histories.items():
        steps = np.arange(len(history['energy']))
        ax.plot(steps, history['energy'], label=method_name, linewidth=2)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Energy')
    ax.set_title('Method Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig
