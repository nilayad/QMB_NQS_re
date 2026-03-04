import sys
sys.path.append(".")

import numpy as np
import tensorflow as tf

dtype_float = tf.float64
dtype_comp = tf.complex128
tf.keras.backend.set_floatx('float64')

import utils_selector as sel
import ED_utill as edut


def compute_operator_final(model, operator_name, sampler_name, n_samples, L,
                            J=1., g=1., pbc=True, n_therm=100, n_sweeps=1):
    """
    Evaluate the expectation value of a Hamiltonian operator for a trained NQS model.

    Args:
        model          : tf.keras.Model, log|ψ(s)|
        operator_name  : str, key in HAMILTONIANS registry
        sampler_name   : str, key in SAMPLERS registry
        n_samples      : number of Monte Carlo samples
        L              : number of lattice sites
        J              : coupling constant
        g              : transverse field
        pbc            : periodic boundary conditions
        n_therm        : thermalization sweeps
        n_sweeps       : MC sweeps between samples

    Returns:
        e_mean         : float, ⟨H⟩ estimate
        e_err          : float, statistical error estimate
    """
    op_fn, _ = sel.get_hamiltonian(operator_name)
    samp_fn, _ = sel.get_sampler(sampler_name)

    samples = samp_fn(model, n_samples, L, n_sweeps=n_sweeps, n_therm=n_therm)
    e_loc = op_fn(model, samples, J, g, pbc=pbc)

    e_mean = tf.reduce_mean(e_loc).numpy()
    e_err = tf.math.reduce_std(e_loc).numpy() / np.sqrt(n_samples)

    return e_mean, e_err


def compute_all_operators(model, sampler_name, n_samples, L,
                          J=1., g=1., pbc=True, n_therm=100, n_sweeps=1):
    """
    Evaluate all registered Hamiltonian operators for a trained NQS model.

    Args:
        model         : tf.keras.Model, log|ψ(s)|
        sampler_name  : str, key in SAMPLERS registry
        n_samples     : number of Monte Carlo samples
        L             : number of lattice sites
        J             : coupling constant
        g             : transverse field
        pbc           : periodic boundary conditions
        n_therm       : thermalization sweeps
        n_sweeps      : MC sweeps between samples

    Returns:
        results       : dict mapping operator_name -> (e_mean, e_err)
    """
    from registry import HAMILTONIANS

    samp_fn, _ = sel.get_sampler(sampler_name)
    samples = samp_fn(model, n_samples, L, n_sweeps=n_sweeps, n_therm=n_therm)

    results = {}
    for op_name, op_fn in HAMILTONIANS.items():
        e_loc = op_fn(model, samples, J, g, pbc=pbc)
        e_mean = tf.reduce_mean(e_loc).numpy()
        e_err = tf.math.reduce_std(e_loc).numpy() / np.sqrt(n_samples)
        results[op_name] = (e_mean, e_err)
        print(f"  {op_name}: E = {e_mean:.6f} ± {e_err:.6f}")

    return results
