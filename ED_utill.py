import tensorflow as tf
import numpy as np
import scipy
from scipy import sparse

dtype_float =tf.float64
dtype_comp = tf.complex128
dtype_int = tf.int32
tf.keras.backend.set_floatx('float64')  
import scipy.sparse.linalg
import matplotlib.pyplot as plt

Id = sparse.csr_matrix(np.eye(2))
Sx = sparse.csr_matrix([[0., 1.], [1., 0.]])
Sz = sparse.csr_matrix([[1., 0.], [0., -1.]])
Sy = sparse.csr_matrix([[0., -1j], [1j, 0.]])
Splus = sparse.csr_matrix([[0., 1.], [0., 0.]])
Sminus = sparse.csr_matrix([[0., 0.], [1., 0.]])


def singlesite_to_full(op, i, L):
    op_list = [Id]*L  # = [Id, Id, Id ...] with L entries
    op_list[i] = op
    full = op_list[0]
    for op_i in op_list[1:]:
        full = sparse.kron(full, op_i, format="csr")
    return full


def gen_sx_list(L):
    return [singlesite_to_full(Sx, i, L) for i in range(L)]


def gen_sz_list(L):
    return [singlesite_to_full(Sz, i, L) for i in range(L)]


def gen_sy_list(L):
    return [singlesite_to_full(Sy, i, L) for i in range(L)]


def gen_hamiltonian(sx_list, sz_list, g, J=1., pbc=True):
    L = len(sx_list)
    H = sparse.csr_matrix((2**L, 2**L))
    for j in range(L-1):
        H = H - J *( sz_list[j] * sz_list[(j+1)%L])
        H = H - g * sx_list[j]
    j = L-1
    H = H - g * sx_list[j]
    if pbc: # X_(L-1)X_(0) only for pbc
        H = H - J *( sz_list[j] * sz_list[(j+1)%L])
    return H

def gen_hamiltonian_heisenberg(sx_list, sz_list,sy_list, J=1.,g=None, pbc=True):
    L = len(sx_list)
    H = sparse.csr_matrix((2**L, 2**L))
    for j in range(L-1):
        H = H - J *( sx_list[j] * sx_list[(j+1)%L]+sy_list[j] * sy_list[(j+1)%L]+sz_list[j] * sz_list[(j+1)%L])
        
    j = L-1
    
    if pbc: # X_(L-1)X_(0) only for pbc
        H = H - J *( sx_list[j] * sx_list[(j+1)%L]+ sy_list[j] * sy_list[(j+1)%L]+sz_list[j] * sz_list[(j+1)%L])
    return H

def gen_hamiltonian_2D(sx_list, sz_list, bonds, g, J=1.0):
    """
    2D transverse-field Ising model Hamiltonian:
        H = -J Σ_{<ij>} σ^z_i σ^z_j  - g Σ_i σ^x_i
    Identical structure to 1D gham().
    """
    L = len(sx_list)
    H = sparse.csr_matrix((2**L, 2**L))

    # bond terms
    for (i, j) in bonds:
        H = H - J * (sz_list[i] * sz_list[j])

    # field terms
    for i in range(L):
        H = H - g * sx_list[i]

    return H



def build_tj_hamiltonian(configs, bonds, t=1.0, J=1.0):
    """
    Build sparse t-J Hamiltonian in configuration basis.
    """
    configs_np = configs.numpy()
    nconf, L = configs_np.shape

    index = {tuple(configs_np[i]): i for i in range(nconf)}
    H = sparse.dok_matrix((nconf, nconf), dtype=np.float64)

    for a in range(nconf):
        cfg = tf.constant(configs_np[a], dtype=dtype_int)

        # exchange (diagonal)
        e = 0.0
        for (i, j) in bonds:
            e += exchange_energy(cfg, i, j, J).numpy()
        H[a, a] += e

        # hopping (off-diagonal)
        for (i, j) in bonds:
            for (p, q) in [(i, j), (j, i)]:
                can, new_cfg = hop(cfg, p, q)
                if can.numpy():
                    b = index[tuple(new_cfg.numpy())]
                    H[a, b] += -t

    return H.tocsr()

# ============================================================
# utill functions
# ============================================================


def project_q_operator(L, q,op_sing):
    #projects operators onto q-momentum sector
    print("Generating projected operator ... ", end="", flush=True)
    # fix: use numpy complex dtype
    op = sparse.csr_matrix((2**L, 2**L), dtype=np.complex128)
    for j in range(L):
        op_j = singlesite_to_full(op_sing, j, L)
        phase = np.exp(1j * q * j)
        op = op + phase * op_j
    print("done", flush=True)
    return op / np.sqrt(L)


def get_momentum_values_symmetric(L):
    return [2*np.pi*n/L for n in range(-L//2 + 1, L//2 + 1)]

def get_momentum_values(L):
    return [2*np.pi*n/L for n in range(L)]


def all_H_configs(N, dtype=dtype_float):
    """
    Return tensor of shape (2^N, N) with spin values in {-1, +1}.
    Works entirely in TensorFlow.
    """
    nstates = 1 << N  # 2^N
    
    idx = tf.range(nstates, dtype=tf.int32)[:, None]   # shape (S,1)
    shifts = tf.range(N-1,-1,-1, dtype=tf.int32)[None, :]      # shape (1,N)
    
    bits = tf.bitwise.bitwise_and(tf.bitwise.right_shift(idx, shifts), 1)
    # map {0,1} -> {-1,+1}
    spins = tf.cast(2 * bits - 1, dtype)
    return spins[::-1]


def all_H_configs_2D(Lx, Ly, dtype=dtype_float):
    """
    Return tensor of shape (2^(Lx*Ly), Lx, Ly) with spin values in {-1, +1}
    for a 2D lattice.
    """
    N = Lx * Ly
    nstates = 1 << N  # 2^N

    # integer values 0 ... 2^N - 1
    idx = tf.range(nstates, dtype=dtype_int)[:, None]   # shape (S,1)
    shifts = tf.range(N, dtype=dtype_int)[None, :]      # shape (1,N)

    # extract bits
    bits = tf.bitwise.bitwise_and(tf.bitwise.right_shift(idx, shifts), 1)
    spins_1d = tf.cast(2 * bits - 1, dtype)  # shape (S, N)

    # reshape into 2D lattice
    spins_2d = tf.reshape(spins_1d, (nstates, Lx, Ly))
    return spins_2d


def all_tj_configs(L, n_particles):
    """
    Generate all t-J configurations with fixed particle number.
    Encoding per site:
        0  : hole
        +1 : spin up
        -1 : spin down

    Returns:
        configs: (Nconf, L) tf.Tensor
    """
    vals = tf.constant([0, 1, -1], dtype=dtype_int)
    grids = tf.meshgrid(*([vals] * L), indexing="ij")
    configs = tf.reshape(tf.stack(grids, axis=-1), (-1, L))

    n_elec = tf.reduce_sum(tf.cast(configs != 0, dtype_int), axis=1)
    mask = tf.equal(n_elec, n_particles)

    return tf.boolean_mask(configs, mask)



def exchange_energy(cfg, i, j, J):
    """
    J (S_i · S_j - 1/4 n_i n_j)
    """
    si = cfg[i]
    sj = cfg[j]

    both = tf.logical_and(si != 0, sj != 0)
    same = tf.equal(si, sj)

    return tf.where(
        both,
        tf.where(same, tf.constant(0.0, dtype_float),
                       tf.constant(-J, dtype_float)),
        tf.constant(0.0, dtype_float)
    )


def hop(cfg, i, j):
    """
    Hopping c†_i c_j
    """
    ci = cfg[i]
    cj = cfg[j]

    can = tf.logical_and(ci == 0, cj != 0)

    new_cfg = tf.tensor_scatter_nd_update(
        cfg,
        indices=[[i], [j]],
        updates=[cj, 0]
    )
    return can, new_cfg


# ============================================================
# 2D EXTENSIONS
# ============================================================

def gen_bonds_2D(Lx, Ly, pbc=True):
    """
    Generate nearest-neighbor bonds for a 2D Lx × Ly square lattice
    with site indexing i = x*Ly + y.
    Returns list of tuples (i, j).
    """
    bonds = []
    for x in range(Lx):
        for y in range(Ly):
            i = x*Ly + y
            
            # Right neighbor
            if y + 1 < Ly:
                j = x*Ly + (y + 1)
                bonds.append((i, j))
            elif pbc:
                j = x*Ly + 0
                bonds.append((i, j))

            # Down neighbor
            if x + 1 < Lx:
                j = (x + 1)*Ly + y
                bonds.append((i, j))
            elif pbc:
                j = 0*Ly + y
                bonds.append((i, j))
    return bonds

def gen_bonds_2D_J2(Lx, Ly, pbc=True):
    """
    Generate next-nearest-neighbor (diagonal) bonds for a 2D Lx × Ly
    square lattice with site indexing i = x*Ly + y.

    Returns list of tuples (i, j).
    """
    bonds = []

    for x in range(Lx):
        for y in range(Ly):
            i = x * Ly + y

            # (x+1, y+1)
            x2 = x + 1
            y2 = y + 1
            if x2 < Lx and y2 < Ly:
                j = x2 * Ly + y2
                bonds.append((i, j))
            elif pbc:
                j = (x2 % Lx) * Ly + (y2 % Ly)
                bonds.append((i, j))

            # (x+1, y-1)
            x2 = x + 1
            y2 = y - 1
            if x2 < Lx and y2 >= 0:
                j = x2 * Ly + y2
                bonds.append((i, j))
            elif pbc:
                j = (x2 % Lx) * Ly + (y2 % Ly)
                bonds.append((i, j))

            # (x-1, y+1)
            x2 = x - 1
            y2 = y + 1
            if x2 >= 0 and y2 < Ly:
                j = x2 * Ly + y2
                bonds.append((i, j))
            elif pbc:
                j = (x2 % Lx) * Ly + (y2 % Ly)
                bonds.append((i, j))

            # (x-1, y-1)
            x2 = x - 1
            y2 = y - 1
            if x2 >= 0 and y2 >= 0:
                j = x2 * Ly + y2
                bonds.append((i, j))
            elif pbc:
                j = (x2 % Lx) * Ly + (y2 % Ly)
                bonds.append((i, j))

    return bonds



def gen_bonds_1D(L, pbc=True):
    bonds = []
    for i in range(L - 1):
        bonds.append((i, i + 1))
    if pbc:
        bonds.append((L - 1, 0))
    return bonds
