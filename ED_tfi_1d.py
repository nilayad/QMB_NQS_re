import numpy as np
import scipy.sparse.linalg
import ED_utill as edut


def calc_gs(L, g, J=1., pbc=True):
    """
    Compute ground state of 1D transverse-field Ising model.
    H = -J Σ σ^z_i σ^z_{i+1} - g Σ σ^x_i
    Returns (E0, psi0).
    """
    sx_list = edut.gen_sx_list(L)
    sz_list = edut.gen_sz_list(L)
    H = edut.gen_hamiltonian(sx_list, sz_list, g, J=J, pbc=pbc)
    E, psi = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    return E[0], psi[:, 0]


def calc_gs_2D(Lx, Ly, g, J=1., pbc=True):
    """
    Compute ground state of 2D transverse-field Ising model on Lx×Ly lattice.
    Returns (E0, psi0).
    """
    L = Lx * Ly
    sx_list = edut.gen_sx_list(L)
    sz_list = edut.gen_sz_list(L)
    bonds = edut.gen_bonds_2D(Lx, Ly, pbc=pbc)
    H = edut.gen_hamiltonian_2D(sx_list, sz_list, bonds, g, J=J)
    E, psi = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    return E[0], psi[:, 0]


def calc_gs_heis_1D(L, J=1., pbc=True):
    """
    Compute ground state of 1D Heisenberg model.
    H = -J Σ (σ^x_i σ^x_{i+1} + σ^y_i σ^y_{i+1} + σ^z_i σ^z_{i+1})
    Returns (E0, psi0).
    """
    sx_list = edut.gen_sx_list(L)
    sz_list = edut.gen_sz_list(L)
    sy_list = edut.gen_sy_list(L)
    H = edut.gen_hamiltonian_heisenberg(sx_list, sz_list, sy_list, J=J, pbc=pbc)
    E, psi = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    return E[0], psi[:, 0]


def calc_gs_heis_2D(Lx, Ly, J=1., pbc=True):
    """
    Compute ground state of 2D Heisenberg model on Lx×Ly lattice.
    Returns (E0, psi0).
    """
    L = Lx * Ly
    sx_list = edut.gen_sx_list(L)
    sz_list = edut.gen_sz_list(L)
    sy_list = edut.gen_sy_list(L)
    bonds = edut.gen_bonds_2D(Lx, Ly, pbc=pbc)
    from scipy import sparse
    H = sparse.csr_matrix((2**L, 2**L), dtype=np.complex128)
    for (i, j) in bonds:
        H = H - J * (sx_list[i] * sx_list[j] + sy_list[i] * sy_list[j] + sz_list[i] * sz_list[j])
    E, psi = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    return E[0], psi[:, 0]


def calc_gs_zf(L, J=1., pbc=True):
    """
    Compute ground state of 1D Ising model with zero transverse field (g=0).
    H = -J Σ σ^z_i σ^z_{i+1}
    Returns (E0, psi0).
    """
    return calc_gs(L, g=0.0, J=J, pbc=pbc)


def calc_zfield_2D(Lx, Ly, g, J=1., pbc=True):
    """
    Compute ground state of 2D Ising model with transverse field.
    Returns (E0, psi0).
    """
    return calc_gs_2D(Lx, Ly, g=g, J=J, pbc=pbc)


def calc_gs_tj_1D(L, n_particles, t=1., J=1., pbc=True):
    """
    Compute ground state of 1D t-J model with fixed particle number.
    Returns (E0, psi0).
    """
    bonds = edut.gen_bonds_1D(L, pbc=pbc)
    configs = edut.all_tj_configs(L, n_particles)
    H = edut.build_tj_hamiltonian(configs, bonds, t=t, J=J)
    E, psi = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    return E[0], psi[:, 0]


def calc_gs_tj_2D(Lx, Ly, n_particles, t=1., J=1., pbc=True):
    """
    Compute ground state of 2D t-J model with fixed particle number.
    Returns (E0, psi0).
    """
    bonds = edut.gen_bonds_2D(Lx, Ly, pbc=pbc)
    configs = edut.all_tj_configs(Lx * Ly, n_particles)
    H = edut.build_tj_hamiltonian(configs, bonds, t=t, J=J)
    E, psi = scipy.sparse.linalg.eigsh(H, k=1, which='SA')
    return E[0], psi[:, 0]
