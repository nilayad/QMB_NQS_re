# utils_selector.py
from registry import HAMILTONIANS, SAMPLERS, ED_SOLVERS
from registry import HAMILTONIAN_ARGS, SAMPLER_ARGS

def get_hamiltonian(name):
    if name not in HAMILTONIANS:
        raise ValueError(f"Unknown Hamiltonian '{name}'")
    return HAMILTONIANS[name], HAMILTONIAN_ARGS.get(name, {})

def get_sampler(name):
    if name not in SAMPLERS:
        raise ValueError(f"Unknown sampler '{name}'")
    return SAMPLERS[name], SAMPLER_ARGS.get(name, {})

def get_ed_solver(name):
    if name not in ED_SOLVERS:
        raise ValueError(f"Unknown ED solver '{name}'")
    return ED_SOLVERS[name]
