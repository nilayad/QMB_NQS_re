import numpy as np
import tensorflow as tf

dtype_float = tf.float64
dtype_comp = tf.complex128
tf.keras.backend.set_floatx('float64')

import utils_selector as sel


def sr_gs_optimize(model, hamiltonian_name, sampler_name, n_samples, L,
                   n_epochs, lr=0.01, diag_shift=0.01, J=1., g=1., pbc=True,
                   n_therm=100, n_sweeps=1):
    """
    Optimize NQS ground state using Stochastic Reconfiguration (SR).

    SR uses the natural gradient: Δθ = S^{-1} F, where S is the quantum
    geometric tensor and F is the energy gradient.

    Args:
        model           : tf.keras.Model, log|ψ(s)|
        hamiltonian_name: str, key in HAMILTONIANS registry
        sampler_name    : str, key in SAMPLERS registry
        n_samples       : number of Monte Carlo samples per epoch
        L               : number of lattice sites
        n_epochs        : number of training epochs
        lr              : learning rate (η)
        diag_shift      : diagonal regularization for S matrix (ε)
        J               : coupling constant
        g               : transverse field
        pbc             : periodic boundary conditions
        n_therm         : thermalization steps
        n_sweeps        : MC sweeps between samples

    Returns:
        energies        : list of mean energy per epoch
    """
    ham_fn, _ = sel.get_hamiltonian(hamiltonian_name)
    samp_fn, _ = sel.get_sampler(sampler_name)

    energies = []
    samples = None

    for epoch in range(n_epochs):
        samples = samp_fn(model, n_samples, L, n_sweeps=n_sweeps, n_therm=n_therm,
                          init_samples=samples)

        with tf.GradientTape(persistent=True) as tape:
            log_psi = model(samples)
            e_loc = ham_fn(model, samples, J, g, pbc=pbc)
            e_mean = tf.reduce_mean(e_loc)

        # Compute O_k = ∂log ψ / ∂θ_k for each sample
        # Jacobian: shape (n_samples, n_params)
        grads_per_sample = []
        for s in range(n_samples):
            with tf.GradientTape() as t2:
                lp = model(samples[s:s+1])
            g_s = t2.gradient(lp, model.trainable_variables)
            flat = tf.concat([tf.reshape(g_k, [-1]) for g_k in g_s], axis=0)
            grads_per_sample.append(flat)

        O = tf.stack(grads_per_sample, axis=0)  # (n_samples, n_params)
        O_mean = tf.reduce_mean(O, axis=0, keepdims=True)  # (1, n_params)
        dO = O - O_mean  # centered

        # S matrix: S_{kl} = ⟨dO_k dO_l⟩
        S = tf.matmul(tf.transpose(dO), dO) / tf.cast(n_samples, dtype_float)
        n_params = S.shape[0]
        S = S + diag_shift * tf.eye(n_params, dtype=dtype_float)

        # Force vector: F_k = 2 Re ⟨(E_loc - E) O_k⟩
        e_centered = tf.cast(e_loc - e_mean, dtype_float)
        F = 2.0 * tf.reduce_mean(e_centered[:, None] * dO, axis=0)

        # Natural gradient: Δθ = S^{-1} F
        delta = tf.linalg.solve(S, F[:, None])[:, 0]

        # Apply update
        idx = 0
        for var in model.trainable_variables:
            size = tf.size(var)
            update = tf.reshape(delta[idx:idx+size], var.shape)
            var.assign_sub(lr * update)
            idx += size

        energies.append(e_mean.numpy())

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  E = {e_mean.numpy():.6f}")

    return energies
