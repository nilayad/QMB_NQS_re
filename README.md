# Neural Quantum States with VMC and Stochastic Reconfiguration

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.8+](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready implementation of Neural Quantum States (NQS) for quantum many-body systems using Variational Monte Carlo (VMC) and Stochastic Reconfiguration (SR) optimization.

## Overview

This package provides a complete framework for representing and optimizing quantum wavefunctions using neural networks. It implements state-of-the-art methods from quantum machine learning and computational physics.

### Key Features

- **Multiple NQS Architectures**
  - Restricted Boltzmann Machines (RBM)
  - Multi-layer Perceptrons (MLP)
  - Convolutional Neural Networks
  - Recurrent Neural Networks (LSTM/GRU)

- **Optimization Methods**
  - Variational Monte Carlo (VMC) with gradient descent
  - Stochastic Reconfiguration (natural gradient)
  - Imaginary time evolution for state preparation

- **Advanced Sampling**
  - Metropolis-Hastings MCMC
  - Parallel tempering for improved mixing
  - Autocorrelation analysis

- **Statistical Analysis**
  - Blocking method for error estimation
  - Jackknife and bootstrap resampling
  - Convergence diagnostics (Geweke, Gelman-Rubin)

- **Hamiltonians**
  - Transverse-field Ising model
  - Heisenberg model (XXZ)
  - Extensible operator framework

## Installation

### Prerequisites

- Python 3.8 or higher
- TensorFlow 2.8+
- NumPy, SciPy
- Matplotlib (for visualization)

### Install from source

```bash
git clone https://github.com/nilayad/QMB_NQS_re.git
cd QMB_NQS_re
pip install -r requirements.txt
pip install -e .
```

### Quick install

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic VMC Training

```python
import numpy as np
from nqs import (
    create_nqs_model,
    TransverseFieldIsing,
    MetropolisSampler,
    VMCTrainer
)

# Setup system
num_sites = 4
hamiltonian = TransverseFieldIsing(num_sites=num_sites, J=1.0, h=1.0)

# Create NQS model
model = create_nqs_model('mlp', num_sites=num_sites, hidden_sizes=[64, 64])

# Configure sampler
sampler = MetropolisSampler(
    num_sites=num_sites,
    num_samples=1000,
    num_chains=4,
    warmup_steps=100
)

# Train with VMC
trainer = VMCTrainer(model, hamiltonian, sampler, learning_rate=0.001)
history = trainer.train(num_steps=500)

# Evaluate
results = trainer.evaluate(num_samples=10000)
print(f"Energy: {results['energy']:.6f} ± {results['energy_error']:.6f}")
```

### Stochastic Reconfiguration

```python
from nqs import RBM, StochasticReconfiguration

# Create RBM model
model = RBM(num_sites=num_sites, num_hidden=2*num_sites)

# SR optimizer
sr_trainer = StochasticReconfiguration(
    model=model,
    hamiltonian=hamiltonian,
    sampler=sampler,
    learning_rate=0.01,
    diagonal_shift=0.01
)

# Train with natural gradient
history = sr_trainer.train(num_steps=300)
```

## Examples

See the `examples/` directory for complete tutorials:

- `01_basic_vmc.py` - Basic VMC training workflow
- `02_sr_optimization.py` - Stochastic Reconfiguration and comparison with VMC

Run examples:

```bash
python examples/01_basic_vmc.py
python examples/02_sr_optimization.py
```

## Configuration

Training parameters can be specified via YAML configuration files:

```yaml
# config/vmc_config.yaml
system:
  num_sites: 4
  hamiltonian: "tfim"
  J: 1.0
  h: 1.0

model:
  type: "mlp"
  hidden_sizes: [64, 64]

training:
  num_steps: 500
  learning_rate: 0.001
```

## Project Structure

```
QMB_NQS_re/
├── nqs/                          # Core NQS package
│   ├── __init__.py              # Package exports
│   ├── models.py                # Neural network architectures
│   ├── operators.py             # Quantum operators and Hamiltonians
│   ├── sampling.py              # MCMC sampling algorithms
│   ├── training.py              # VMC training
│   └── stochastic_reconfiguration.py  # SR optimizer
├── utils/                        # Utilities
│   ├── diagnostics.py           # Statistical analysis
│   └── plotting.py              # Visualization
├── config/                       # Configuration files
│   ├── vmc_config.yaml
│   └── sr_config.yaml
├── examples/                     # Example scripts
│   ├── 01_basic_vmc.py
│   └── 02_sr_optimization.py
├── tests/                        # Unit tests
│   ├── test_operators.py
│   └── test_sampling.py
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
└── README.md                     # This file
```

## Neural Quantum States

### What are NQS?

Neural Quantum States use neural networks to represent quantum wavefunctions:

ψ(s) = exp(NN(s))

where `s` represents a quantum state configuration and `NN` is a neural network.

### Variational Monte Carlo

VMC optimizes the wavefunction by minimizing the energy expectation:

E[θ] = ⟨ψ(θ)|H|ψ(θ)⟩ / ⟨ψ(θ)|ψ(θ)⟩

The gradient is computed via:

∇E = 2⟨E_loc(s) ∇log|ψ(s)|⟩ - 2⟨E_loc⟩⟨∇log|ψ(s)|⟩

### Stochastic Reconfiguration

SR uses the natural gradient (Quantum Fisher Information):

θ → θ - η S⁻¹F

where:
- S is the quantum geometric tensor (Fubini-Study metric)
- F is the energy gradient
- η is the learning rate

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Run specific tests:

```bash
pytest tests/test_operators.py -v
pytest tests/test_sampling.py -v
```

## Performance Tips

1. **Batch Size**: Use multiple chains for better statistics
2. **Warmup**: Ensure adequate thermalization (100+ steps)
3. **Learning Rate**: Start with 0.001 for VMC, 0.01 for SR
4. **Regularization**: SR diagonal shift typically 0.01-0.1
5. **Architecture**: RBM for small systems, MLP for larger

## Advanced Features

### Custom Hamiltonians

```python
class CustomHamiltonian:
    def __init__(self, num_sites):
        self.num_sites = num_sites
    
    def local_energy(self, configs, log_psi):
        # Implement E_loc(s) = ⟨s|H|ψ⟩/⟨s|ψ⟩
        return e_loc
```

### Custom Observables

```python
def compute_magnetization(samples):
    return np.mean(np.sum(samples, axis=1))
```

## References

1. Carleo, G., & Troyer, M. (2017). Solving the quantum many-body problem with artificial neural networks. *Science*, 355(6325), 602-606.

2. Stokes, J., et al. (2020). Quantum Natural Gradient. *Quantum*, 4, 269.

3. Choo, K., et al. (2020). Fermionic neural-network states for ab-initio electronic structure. *Nature Communications*, 11(1), 1-7.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{nqs_vmc_sr,
  title = {Neural Quantum States with VMC and Stochastic Reconfiguration},
  author = {QMB NQS Research},
  year = {2024},
  url = {https://github.com/nilayad/QMB_NQS_re}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Support

For questions and issues:
- Open an issue on GitHub
- Check the examples directory
- Review the documentation in docstrings

## Acknowledgments

This implementation is based on research in quantum machine learning and many-body physics. Special thanks to the quantum computing and neural network communities for their foundational work.
