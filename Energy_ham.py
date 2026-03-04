import tensorflow as tf
import numpy as np

dtype_float = tf.float64
dtype_comp = tf.complex128
dtype_int = tf.int32
tf.keras.backend.set_floatx('float64')


def local_energy_tf(model, samples, J, g, pbc=True):
    """
    Compute local energy for transverse-field Ising model using TensorFlow.

    H = -J Σ_{<ij>} σ^z_i σ^z_j - g Σ_i σ^x_i

    Args:
        model  : callable, log|ψ(s)|, takes (batch, L) tensor, returns (batch,)
        samples: (batch, L) tf.Tensor with spin values in {-1, +1}
        J      : Ising coupling
        g      : transverse field strength
        pbc    : periodic boundary conditions

    Returns:
        e_loc  : (batch,) tf.Tensor of local energies (real)
    """
    samples = tf.cast(samples, dtype_float)
    batch_size = tf.shape(samples)[0]
    L = tf.shape(samples)[1]

    log_psi_s = model(samples)

    # --- diagonal (σ^z σ^z) term ---
    n_bonds = L if pbc else L - 1
    e_diag = tf.zeros(batch_size, dtype=dtype_float)
    for j in tf.range(n_bonds):
        jp1 = (j + 1) % L
        e_diag = e_diag - J * samples[:, j] * samples[:, jp1]

    # --- off-diagonal (σ^x) term ---
    e_offdiag = tf.zeros(batch_size, dtype=dtype_float)
    for i in tf.range(L):
        # flip spin i
        flip_mask = tf.one_hot(i, L, on_value=-2.0, off_value=0.0, dtype=dtype_float)
        s_flip = samples + flip_mask * samples  # s_flip[:,i] = -s[:,i], rest unchanged
        log_psi_flip = model(s_flip)
        ratio = tf.exp(log_psi_flip - log_psi_s)
        e_offdiag = e_offdiag - g * ratio

    return e_diag + e_offdiag


def diag_field_energy(model, samples, g):
    """
    Compute diagonal transverse-field contribution only.

    E_field = -g Σ_i σ^x_i  (off-diagonal part)

    This evaluates the ratio ψ(s')/ψ(s) for each single-spin flip s'.

    Args:
        model  : callable log|ψ(s)|
        samples: (batch, L) tf.Tensor, spins in {-1, +1}
        g      : field strength

    Returns:
        e_field: (batch,) tf.Tensor
    """
    samples = tf.cast(samples, dtype_float)
    batch_size = tf.shape(samples)[0]
    L = tf.shape(samples)[1]

    log_psi_s = model(samples)
    e_field = tf.zeros(batch_size, dtype=dtype_float)

    for i in tf.range(L):
        flip_mask = tf.one_hot(i, L, on_value=-2.0, off_value=0.0, dtype=dtype_float)
        s_flip = samples + flip_mask * samples
        log_psi_flip = model(s_flip)
        ratio = tf.exp(log_psi_flip - log_psi_s)
        e_field = e_field - g * ratio

    return e_field


def hamiltonian_heis(model, samples, J, pbc=True):
    """
    Compute local energy for isotropic Heisenberg model using TensorFlow.

    H = -J Σ_{<ij>} (σ^x_i σ^x_j + σ^y_i σ^y_j + σ^z_i σ^z_j)
      = -J Σ_{<ij>} (2 S^+_i S^-_j + 2 S^-_i S^+_j + σ^z_i σ^z_j)

    Args:
        model  : callable log|ψ(s)|
        samples: (batch, L) tf.Tensor, spins in {-1, +1}
        J      : exchange coupling
        pbc    : periodic boundary conditions

    Returns:
        e_loc  : (batch,) tf.Tensor of local energies (real)
    """
    samples = tf.cast(samples, dtype_float)
    batch_size = tf.shape(samples)[0]
    L = tf.shape(samples)[1]

    log_psi_s = model(samples)

    n_bonds = L if pbc else L - 1
    e_diag = tf.zeros(batch_size, dtype=dtype_float)
    e_offdiag = tf.zeros(batch_size, dtype=dtype_float)

    for j in tf.range(n_bonds):
        jp1 = (j + 1) % L

        # σ^z_j σ^z_{j+1} diagonal term
        e_diag = e_diag - J * samples[:, j] * samples[:, jp1]

        # S^+_j S^-_{j+1}: requires s_j = -1 and s_{j+1} = +1
        # After flip: s_j -> +1, s_{j+1} -> -1
        can_up = tf.logical_and(
            tf.equal(samples[:, j], -1.0),
            tf.equal(samples[:, jp1], 1.0)
        )
        # S^-_j S^+_{j+1}: requires s_j = +1 and s_{j+1} = -1
        can_down = tf.logical_and(
            tf.equal(samples[:, j], 1.0),
            tf.equal(samples[:, jp1], -1.0)
        )
        can_flip = tf.logical_or(can_up, can_down)

        # build flipped configuration (flip sites j and jp1)
        flip_j = tf.one_hot(j, L, on_value=-2.0, off_value=0.0, dtype=dtype_float)
        flip_jp1 = tf.one_hot(jp1, L, on_value=-2.0, off_value=0.0, dtype=dtype_float)
        s_flip = samples + flip_j * samples + flip_jp1 * samples

        log_psi_flip = model(s_flip)
        ratio = tf.exp(log_psi_flip - log_psi_s)

        # coefficient: S^+S^- + S^-S^+ gives factor of 2 * (1/2)(1/2) = 1/2 in spin-1/2
        # For Pauli matrices: σ^+σ^- + σ^-σ^+ contributes 1 when spins differ
        contrib = tf.where(can_flip, -J * ratio, tf.zeros_like(ratio))
        e_offdiag = e_offdiag + contrib

    return e_diag + e_offdiag
