import sys
sys.path.append(".")

import numpy as np
import tensorflow as tf

dtype_float = tf.float64
dtype_comp = tf.complex128
tf.keras.backend.set_floatx('float64')

import utils_selector as sel
import ED_utill as edut


def train_gs(model, hamiltonian_name, sampler_name, n_samples, L,
             n_epochs, lr=0.01, J=1., g=1., pbc=True, n_therm=100, n_sweeps=1):
    """
    Train NQS model to approximate ground state using VMC + gradient descent.

    Args:
        model           : tf.keras.Model, log|ψ(s)|
        hamiltonian_name: str, key in HAMILTONIANS registry
        sampler_name    : str, key in SAMPLERS registry
        n_samples       : number of Monte Carlo samples per epoch
        L               : number of lattice sites
        n_epochs        : number of training epochs
        lr              : learning rate
        J               : Ising/exchange coupling
        g               : transverse field
        pbc             : periodic boundary conditions
        n_therm         : thermalization steps for sampler
        n_sweeps        : MC sweeps between samples

    Returns:
        energies        : list of mean energy per epoch
    """
    ham_fn, _ = sel.get_hamiltonian(hamiltonian_name)
    samp_fn, _ = sel.get_sampler(sampler_name)

    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
    energies = []

    samples = None
    for epoch in range(n_epochs):
        samples = samp_fn(model, n_samples, L, n_sweeps=n_sweeps, n_therm=n_therm,
                          init_samples=samples)

        with tf.GradientTape() as tape:
            log_psi = model(samples)
            e_loc = ham_fn(model, samples, J, g, pbc=pbc)
            e_mean = tf.reduce_mean(e_loc)
            # VMC gradient: ∂E/∂θ = 2 Re[⟨(E_loc - E) ∂log ψ/∂θ⟩]
            loss = 2.0 * tf.reduce_mean((e_loc - e_mean) * log_psi)

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        energies.append(e_mean.numpy())

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  E = {e_mean.numpy():.6f}")

    return energies
