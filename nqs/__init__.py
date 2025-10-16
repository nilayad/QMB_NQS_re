"""Neural Quantum States package.

This package implements Neural Quantum States with Variational Monte Carlo
and Stochastic Reconfiguration for quantum many-body systems.
"""

from .models import (
    NQSBase,
    RBM,
    MLP,
    ConvolutionalNQS,
    RecurrentNQS,
    create_nqs_model
)

from .operators import (
    SpinOperators,
    TransverseFieldIsing,
    Heisenberg,
    config_to_spins,
    spins_to_config
)

from .sampling import (
    MetropolisSampler,
    ParallelTempering,
    compute_autocorrelation,
    estimate_correlation_time
)

from .training import (
    VMCTrainer,
    AdaptiveLearningRate,
    compute_fidelity
)

from .stochastic_reconfiguration import (
    StochasticReconfiguration,
    StatePreparation
)

__all__ = [
    # Models
    'NQSBase',
    'RBM',
    'MLP',
    'ConvolutionalNQS',
    'RecurrentNQS',
    'create_nqs_model',
    
    # Operators
    'SpinOperators',
    'TransverseFieldIsing',
    'Heisenberg',
    'config_to_spins',
    'spins_to_config',
    
    # Sampling
    'MetropolisSampler',
    'ParallelTempering',
    'compute_autocorrelation',
    'estimate_correlation_time',
    
    # Training
    'VMCTrainer',
    'AdaptiveLearningRate',
    'compute_fidelity',
    
    # Stochastic Reconfiguration
    'StochasticReconfiguration',
    'StatePreparation',
]

__version__ = '0.1.0'
