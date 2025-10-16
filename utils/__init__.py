"""Utility modules for NQS analysis and visualization."""

from .diagnostics import (
    blocking_analysis,
    jackknife_error,
    bootstrap_error,
    integrated_autocorrelation_time,
    effective_sample_size,
    geweke_test,
    gelman_rubin_diagnostic,
    compute_running_average,
    detect_outliers
)

from .plotting import (
    plot_training_history,
    plot_autocorrelation,
    plot_blocking_analysis,
    plot_energy_histogram,
    plot_spin_correlation,
    plot_wavefunction_amplitudes,
    plot_convergence_comparison
)

__all__ = [
    # Diagnostics
    'blocking_analysis',
    'jackknife_error',
    'bootstrap_error',
    'integrated_autocorrelation_time',
    'effective_sample_size',
    'geweke_test',
    'gelman_rubin_diagnostic',
    'compute_running_average',
    'detect_outliers',
    
    # Plotting
    'plot_training_history',
    'plot_autocorrelation',
    'plot_blocking_analysis',
    'plot_energy_histogram',
    'plot_spin_correlation',
    'plot_wavefunction_amplitudes',
    'plot_convergence_comparison',
]
