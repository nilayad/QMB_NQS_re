import numpy as np
import tensorflow as tf

dtype_float = tf.float64
dtype_comp = tf.complex128
tf.keras.backend.set_floatx('float64')

import utils_selector as sel


def sr_op_optimize(model_gs, model_op, hamiltonian_name, sampler_name, n_samples, L,
                   n_epochs, lr=0.01, diag_shift=0.01, J=1., g=1., pbc=True,
                   n_therm=100, n_sweeps=1):
    """
    Optimize NQS representation of an excited state / operator application
    using Stochastic Reconfiguration.

    Trains model_op to represent H|ψ_gs⟩ / ||H|ψ_gs⟩||  or a related state.

    Args:
        model_gs        : tf.keras.Model, frozen ground state log|ψ_gs(s)|
        model_op        : tf.keras.Model, trainable log|φ(s)|
        hamiltonian_name: str, key in HAMILTONIANS registry
        sampler_name    : str, key in SAMPLERS registry
        n_samples       : number of Monte Carlo samples per epoch
        L               : number of lattice sites
        n_epochs        : number of training epochs
        lr              : learning rate
        diag_shift      : S matrix regularization
        J               : coupling constant
        g               : transverse field
        pbc             : periodic boundary conditions
        n_therm         : thermalization sweeps
        n_sweeps        : MC sweeps between samples

    Returns:
        overlaps        : list of overlap ⟨φ|H|ψ_gs⟩ per epoch
    """
    ham_fn, _ = sel.get_hamiltonian(hamiltonian_name)
    samp_fn, _ = sel.get_sampler(sampler_name)

    overlaps = []
    samples = None

    for epoch in range(n_epochs):
        # Sample from ground state distribution |ψ_gs|²
        samples = samp_fn(model_gs, n_samples, L, n_sweeps=n_sweeps, n_therm=n_therm,
                          init_samples=samples)

        log_psi_gs = model_gs(samples)
        log_phi = model_op(samples)

        # Compute H|ψ_gs⟩ weights: e_loc(s) = ⟨s|H|ψ_gs⟩ / ψ_gs(s)
        e_loc = ham_fn(model_gs, samples, J, g, pbc=pbc)

        # Overlap: ⟨φ|H|ψ_gs⟩ ≈ ⟨e_loc · φ/ψ_gs⟩_{|ψ_gs|²}
        ratio = tf.exp(log_phi - log_psi_gs)
        overlap = tf.reduce_mean(e_loc * ratio)

        # Compute O_k for model_op
        grads_per_sample = []
        for s in range(n_samples):
            with tf.GradientTape() as t2:
                lp = model_op(samples[s:s+1])
            g_s = t2.gradient(lp, model_op.trainable_variables)
            flat = tf.concat([tf.reshape(g_k, [-1]) for g_k in g_s], axis=0)
            grads_per_sample.append(flat)

        O = tf.stack(grads_per_sample, axis=0)
        O_mean = tf.reduce_mean(O, axis=0, keepdims=True)
        dO = O - O_mean

        S = tf.matmul(tf.transpose(dO), dO) / tf.cast(n_samples, dtype_float)
        n_params = S.shape[0]
        S = S + diag_shift * tf.eye(n_params, dtype=dtype_float)

        # Force: gradient of -|⟨φ|H|ψ_gs⟩|² w.r.t. φ params
        w = e_loc * ratio - overlap
        F = 2.0 * tf.reduce_mean(w[:, None] * dO, axis=0)

        delta = tf.linalg.solve(S, F[:, None])[:, 0]

        idx = 0
        for var in model_op.trainable_variables:
            size = tf.size(var)
            update = tf.reshape(delta[idx:idx+size], var.shape)
            var.assign_sub(lr * update)
            idx += size

        overlaps.append(overlap.numpy())

        if epoch % 100 == 0:
            print(f"Epoch {epoch:4d}  overlap = {overlap.numpy():.6f}")

    return overlaps
