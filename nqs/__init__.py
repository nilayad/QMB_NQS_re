"""
Neural Quantum States Package

A modern implementation of Neural Quantum States (NQS) for quantum many-body physics.
Includes VMC training, Stochastic Reconfiguration, and multiple architectures.
"""

__version__ = "0.1.0"
__author__ = "Nilay Deshpande"

# Import submodules
from . import operators
from . import sampling
from . import models
from . import training

# Import from operators (using YOUR actual class names)
from .operators import (
    SpinOperators,
    TransverseFieldIsing,
    Heisenberg,
    config_to_spins,
    spins_to_config
)

# Import from sampling (using YOUR actual class names)
from .sampling import (
    MetropolisSampler,
    ParallelTempering,
    compute_autocorrelation,
    estimate_correlation_time
)

# Import from models (using YOUR actual class names)
from .models import (
    NQSBase,
    RBM,
    MLP,
    ConvolutionalNQS,
    RecurrentNQS,
    create_nqs_model
)

# Import from training (using YOUR actual class names)
from .training import (
    VMCTrainer,
    AdaptiveLearningRate,
    compute_fidelity
)

# Import from stochastic_reconfiguration (using YOUR actual class names)
from .stochastic_reconfiguration import (
    StochasticReconfiguration,
    StatePreparation
)

# Create convenient aliases
SROptimizer = StochasticReconfiguration
FidelityOptimizer = StatePreparation
TFIMHamiltonian = TransverseFieldIsing  # Common alias
build_nqs = create_nqs_model  # Common alias
SimpleMLP = MLP  # Common alias

# Define public API
__all__ = [
    # Submodules
    'operators',
    'sampling',
    'models',
    'training',
    
    # Operators
    'SpinOperators',
    'TransverseFieldIsing',
    'TFIMHamiltonian',  # Alias
    'Heisenberg',
    'config_to_spins',
    'spins_to_config',
    
    # Sampling
    'MetropolisSampler',
    'ParallelTempering',
    'compute_autocorrelation',
    'estimate_correlation_time',
    
    # Models
    'NQSBase',
    'RBM',
    'MLP',
    'SimpleMLP',  # Alias
    'ConvolutionalNQS',
    'RecurrentNQS',
    'create_nqs_model',
    'build_nqs',  # Alias
    
    # Training
    'VMCTrainer',
    'AdaptiveLearningRate',
    'compute_fidelity',
    
    # Stochastic Reconfiguration
    'StochasticReconfiguration',
    'StatePreparation',
    'SROptimizer',  # Alias
    'FidelityOptimizer',  # Alias
]
