import tensorflow as tf
import numpy as np

dtype_float = tf.float64
dtype_int = tf.int32
tf.keras.backend.set_floatx('float64')


def metropolis_sample_tf(model, n_samples, L, n_sweeps=1, n_therm=100, init_samples=None):
    """
    Metropolis-Hastings sampler for spin-1/2 systems (single spin flip).

    Args:
        model      : callable log|ψ(s)|, takes (batch, L) -> (batch,)
        n_samples  : number of samples to return
        L          : number of lattice sites
        n_sweeps   : number of full sweeps between recorded samples
        n_therm    : thermalization sweeps
        init_samples: (n_samples, L) initial configurations; random if None

    Returns:
        samples    : (n_samples, L) tf.Tensor with spin values in {-1, +1}
    """
    if init_samples is None:
        samples = tf.cast(
            2 * tf.random.uniform((n_samples, L), minval=0, maxval=2, dtype=tf.int32) - 1,
            dtype_float
        )
    else:
        samples = tf.cast(init_samples, dtype_float)

    log_psi = model(samples)

    total_sweeps = n_therm + n_sweeps
    for _ in range(total_sweeps):
        for _ in range(L):
            # propose a random single spin flip per chain
            flip_sites = tf.random.uniform((n_samples,), minval=0, maxval=L, dtype=tf.int32)
            flip_mask = tf.one_hot(flip_sites, L, on_value=-2.0, off_value=0.0, dtype=dtype_float)
            proposed = samples + flip_mask * samples

            log_psi_prop = model(proposed)
            log_ratio = 2.0 * (log_psi_prop - log_psi)
            log_u = tf.math.log(tf.random.uniform((n_samples,), dtype=dtype_float))
            accept = log_u < log_ratio

            accept_f = tf.cast(accept[:, None], dtype_float)
            samples = accept_f * proposed + (1.0 - accept_f) * samples
            log_psi = tf.where(accept, log_psi_prop, log_psi)

    return samples


def metropolis_two_spin_swap_tf(model, n_samples, L, n_sweeps=1, n_therm=100, init_samples=None):
    """
    Metropolis sampler using two-spin swap moves (conserves magnetization).

    Args:
        model      : callable log|ψ(s)|
        n_samples  : number of samples
        L          : number of sites
        n_sweeps   : sweeps between recorded samples
        n_therm    : thermalization sweeps
        init_samples: initial configurations; random (balanced) if None

    Returns:
        samples    : (n_samples, L) tf.Tensor
    """
    if init_samples is None:
        half = L // 2
        base = np.array([1.0] * half + [-1.0] * (L - half))
        inits = np.array([np.random.permutation(base) for _ in range(n_samples)])
        samples = tf.constant(inits, dtype=dtype_float)
    else:
        samples = tf.cast(init_samples, dtype_float)

    log_psi = model(samples)

    total_sweeps = n_therm + n_sweeps
    for _ in range(total_sweeps):
        for _ in range(L):
            # pick two random sites to swap
            i = tf.random.uniform((n_samples,), minval=0, maxval=L, dtype=tf.int32)
            j = tf.random.uniform((n_samples,), minval=0, maxval=L, dtype=tf.int32)

            mask_i = tf.one_hot(i, L, dtype=dtype_float)
            mask_j = tf.one_hot(j, L, dtype=dtype_float)

            si = tf.reduce_sum(samples * mask_i, axis=1, keepdims=True)
            sj = tf.reduce_sum(samples * mask_j, axis=1, keepdims=True)

            # swap: proposed[i] = s[j], proposed[j] = s[i]
            proposed = samples - mask_i * samples + mask_i * sj - mask_j * samples + mask_j * si

            log_psi_prop = model(proposed)
            log_ratio = 2.0 * (log_psi_prop - log_psi)
            log_u = tf.math.log(tf.random.uniform((n_samples,), dtype=dtype_float))
            accept = log_u < log_ratio

            accept_f = tf.cast(accept[:, None], dtype_float)
            samples = accept_f * proposed + (1.0 - accept_f) * samples
            log_psi = tf.where(accept, log_psi_prop, log_psi)

    return samples


def metropolis_two_spin_swap_scatter_tf(model, n_samples, L, n_sweeps=1, n_therm=100, init_samples=None):
    """
    Metropolis sampler using two-spin swap with scatter updates.

    Args:
        model      : callable log|ψ(s)|
        n_samples  : number of samples
        L          : number of sites
        n_sweeps   : sweeps between recorded samples
        n_therm    : thermalization sweeps
        init_samples: initial configurations; None for random

    Returns:
        samples    : (n_samples, L) tf.Tensor
    """
    return metropolis_two_spin_swap_tf(model, n_samples, L, n_sweeps, n_therm, init_samples)


def metropolis_sample_h_target(model, n_samples, L, h_samples, n_sweeps=1, n_therm=100):
    """
    Metropolis sampler biased toward a target distribution h_samples.

    Mixes single spin-flip proposals with samples drawn from h_samples.

    Args:
        model    : callable log|ψ(s)|
        n_samples: number of output samples
        L        : number of sites
        h_samples: (M, L) reference configurations used as proposals
        n_sweeps : sweeps between recorded samples
        n_therm  : thermalization sweeps

    Returns:
        samples  : (n_samples, L) tf.Tensor
    """
    h_samples = tf.cast(h_samples, dtype_float)
    M = tf.shape(h_samples)[0]

    idx = tf.random.uniform((n_samples,), minval=0, maxval=M, dtype=tf.int32)
    samples = tf.gather(h_samples, idx)
    log_psi = model(samples)

    total_sweeps = n_therm + n_sweeps
    for _ in range(total_sweeps):
        for _ in range(L):
            flip_sites = tf.random.uniform((n_samples,), minval=0, maxval=L, dtype=tf.int32)
            flip_mask = tf.one_hot(flip_sites, L, on_value=-2.0, off_value=0.0, dtype=dtype_float)
            proposed = samples + flip_mask * samples

            log_psi_prop = model(proposed)
            log_ratio = 2.0 * (log_psi_prop - log_psi)
            log_u = tf.math.log(tf.random.uniform((n_samples,), dtype=dtype_float))
            accept = log_u < log_ratio

            accept_f = tf.cast(accept[:, None], dtype_float)
            samples = accept_f * proposed + (1.0 - accept_f) * samples
            log_psi = tf.where(accept, log_psi_prop, log_psi)

    return samples


def metropolis_sample_tj_tf(model, n_samples, L, n_particles, n_sweeps=1, n_therm=100, init_samples=None):
    """
    Metropolis sampler for t-J model (hopping + spin swaps).

    Site encoding: 0 = hole, +1 = spin-up, -1 = spin-down.

    Args:
        model      : callable log|ψ(s)|
        n_samples  : number of samples
        L          : number of sites
        n_particles: number of electrons (holes = L - n_particles)
        n_sweeps   : sweeps between recorded samples
        n_therm    : thermalization sweeps
        init_samples: initial configurations; None for random

    Returns:
        samples    : (n_samples, L) tf.Tensor (int-valued: 0, +1, -1)
    """
    if init_samples is None:
        vals = [1, -1, 0]
        inits = []
        for _ in range(n_samples):
            cfg = np.zeros(L, dtype=np.float64)
            sites = np.random.choice(L, n_particles, replace=False)
            for k, s in enumerate(sites):
                cfg[s] = 1.0 if k % 2 == 0 else -1.0
            inits.append(cfg)
        samples = tf.constant(np.array(inits), dtype=dtype_float)
    else:
        samples = tf.cast(init_samples, dtype_float)

    log_psi = model(samples)

    total_sweeps = n_therm + n_sweeps
    for _ in range(total_sweeps):
        for _ in range(L):
            i = tf.random.uniform((n_samples,), minval=0, maxval=L, dtype=tf.int32)
            j = tf.random.uniform((n_samples,), minval=0, maxval=L, dtype=tf.int32)

            mask_i = tf.one_hot(i, L, dtype=dtype_float)
            mask_j = tf.one_hot(j, L, dtype=dtype_float)

            si = tf.reduce_sum(samples * mask_i, axis=1, keepdims=True)
            sj = tf.reduce_sum(samples * mask_j, axis=1, keepdims=True)

            proposed = samples - mask_i * samples + mask_i * sj - mask_j * samples + mask_j * si

            log_psi_prop = model(proposed)
            log_ratio = 2.0 * (log_psi_prop - log_psi)
            log_u = tf.math.log(tf.random.uniform((n_samples,), dtype=dtype_float))
            accept = log_u < log_ratio

            accept_f = tf.cast(accept[:, None], dtype_float)
            samples = accept_f * proposed + (1.0 - accept_f) * samples
            log_psi = tf.where(accept, log_psi_prop, log_psi)

    return samples


def metropolis__target(model, n_samples, L, target_samples, n_sweeps=1, n_therm=50):
    """
    Metropolis sampler initialized from target_samples.

    Args:
        model         : callable log|ψ(s)|
        n_samples     : number of output samples
        L             : number of sites
        target_samples: (M, L) reference configurations for initialization
        n_sweeps      : sweeps between recorded samples
        n_therm       : thermalization sweeps

    Returns:
        samples       : (n_samples, L) tf.Tensor
    """
    target_samples = tf.cast(target_samples, dtype_float)
    M = tf.shape(target_samples)[0]

    idx = tf.random.uniform((n_samples,), minval=0, maxval=M, dtype=tf.int32)
    samples = tf.gather(target_samples, idx)
    log_psi = model(samples)

    total_sweeps = n_therm + n_sweeps
    for _ in range(total_sweeps):
        for _ in range(L):
            flip_sites = tf.random.uniform((n_samples,), minval=0, maxval=L, dtype=tf.int32)
            flip_mask = tf.one_hot(flip_sites, L, on_value=-2.0, off_value=0.0, dtype=dtype_float)
            proposed = samples + flip_mask * samples

            log_psi_prop = model(proposed)
            log_ratio = 2.0 * (log_psi_prop - log_psi)
            log_u = tf.math.log(tf.random.uniform((n_samples,), dtype=dtype_float))
            accept = log_u < log_ratio

            accept_f = tf.cast(accept[:, None], dtype_float)
            samples = accept_f * proposed + (1.0 - accept_f) * samples
            log_psi = tf.where(accept, log_psi_prop, log_psi)

    return samples


def metropolis_sample_multiflip(model, n_samples, L, n_flips=2, n_sweeps=1, n_therm=100, init_samples=None):
    """
    Metropolis sampler with multi-spin flip proposals.

    Args:
        model      : callable log|ψ(s)|
        n_samples  : number of samples
        L          : number of sites
        n_flips    : number of spins to flip simultaneously
        n_sweeps   : sweeps between recorded samples
        n_therm    : thermalization sweeps
        init_samples: initial configurations; None for random

    Returns:
        samples    : (n_samples, L) tf.Tensor
    """
    if init_samples is None:
        samples = tf.cast(
            2 * tf.random.uniform((n_samples, L), minval=0, maxval=2, dtype=tf.int32) - 1,
            dtype_float
        )
    else:
        samples = tf.cast(init_samples, dtype_float)

    log_psi = model(samples)

    total_sweeps = n_therm + n_sweeps
    for _ in range(total_sweeps):
        # flip n_flips random sites per chain
        flip_mask = tf.zeros((n_samples, L), dtype=dtype_float)
        for _ in range(n_flips):
            flip_sites = tf.random.uniform((n_samples,), minval=0, maxval=L, dtype=tf.int32)
            flip_mask = flip_mask + tf.one_hot(flip_sites, L, on_value=1.0, off_value=0.0, dtype=dtype_float)
        # compute effective sign flip: even number of flips on same site = no flip
        flip_mask = tf.math.mod(tf.cast(flip_mask, tf.int32), 2)
        flip_mask = tf.cast(flip_mask, dtype_float)
        # apply: s' = s * (1 - 2*mask)
        proposed = samples * (1.0 - 2.0 * flip_mask)

        log_psi_prop = model(proposed)
        log_ratio = 2.0 * (log_psi_prop - log_psi)
        log_u = tf.math.log(tf.random.uniform((n_samples,), dtype=dtype_float))
        accept = log_u < log_ratio

        accept_f = tf.cast(accept[:, None], dtype_float)
        samples = accept_f * proposed + (1.0 - accept_f) * samples
        log_psi = tf.where(accept, log_psi_prop, log_psi)

    return samples
