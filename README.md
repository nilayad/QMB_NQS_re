# QMB_NQS_re — Neural Quantum States (NQS)

> Work in progress. Code is provided for reference. Not fully functional.  
> Parts were generated with GitHub Copilot. Use at your own risk.

---

## What is this?

Implementation of Neural Quantum States (NQS) for quantum many-body systems using:
- Variational Monte Carlo (VMC)
- Stochastic Reconfiguration (SR)

Supports RBM, MLP, CNN, and RNN wavefunction architectures.

---

## Requirements

You need Python 3.8+ and the following packages installed manually:

- numpy
- scipy
- tensorflow (2.8 or newer)
- pyyaml
- matplotlib
- seaborn
- tqdm

Install them however you prefer (pip, conda, etc). No install script is provided.

Example with pip:
```
pip install numpy scipy tensorflow pyyaml matplotlib seaborn tqdm
```

---

## How to get the code

Clone the repository:

```
git clone https://github.com/nilayad/QMB_NQS_re.git
cd QMB_NQS_re
```

No installation step is needed. Just run scripts directly from the repo folder.

---

## Project layout

```
QMB_NQS_re/
├── nqs/                    Core NQS package
│   ├── __init__.py
│   ├── models.py           Neural network architectures (RBM, MLP, CNN, RNN)
│   ├── operators.py        Quantum operators and Hamiltonians
│   ├── sampling.py         Metropolis-Hastings MCMC sampler
│   ├── training.py         VMC trainer
│   └── stochastic_reconfiguration.py   SR optimizer
├── utils/                  Statistical analysis and plotting
│   ├── diagnostics.py
│   └── plotting.py
├── config/                 YAML config files
│   ├── vmc_config.yaml
│   └── sr_config.yaml
├── examples/               Example scripts
│   ├── 01_basic_vmc.py
│   └── 02_sr_optimization.py
└── tests/                  Unit tests
    ├── test_operators.py
    ├── test_sampling.py
    └── test_training.py
```

---

## Run an example

After cloning, from inside the repo folder:

```
python examples/01_basic_vmc.py
python examples/02_sr_optimization.py
```

---

## Run tests

```
pytest tests/
```

---

## License

MIT — see LICENSE file.
