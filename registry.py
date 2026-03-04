# registry.py
import Energy_ham as enham
import model_sampler as samp
import ED_tfi_1d as ed   # replace with your actual ED module

# -------------------------------
# Hamiltonian registry
# -------------------------------
HAMILTONIANS = {
    "local_energy_tf": enham.local_energy_tf,
    "diag_field_energy": enham.diag_field_energy,
    "hamiltonian_heis": enham.hamiltonian_heis,
}

HAMILTONIAN_ARGS = {
    "local_energy_tf": {},
    "diag_field_energy": {},
    "hamiltonian_heis": {},
}

# -------------------------------
# Sampler registry
# -------------------------------
SAMPLERS = {
    "metropolis_sample_tf": samp.metropolis_sample_tf,
    "metropolis_two_spin_swap_scatter_tf": samp.metropolis_two_spin_swap_scatter_tf,
    "metropolis_two_spin_swap_tf": samp.metropolis_two_spin_swap_tf,
    "metropolis_sample_h_target": samp.metropolis_sample_h_target,
    "metropolis_sample_tj_tf": samp.metropolis_sample_tj_tf,
    "metropolis__target" : samp.metropolis__target,
    "metropolis_sample_multiflip": samp.metropolis_sample_multiflip
}

SAMPLER_ARGS = {
    "metropolis_sample_tf": {},
    "metropolis_two_spin_swap_scatter_tf": {},
    "metropolis_two_spin_swap_tf": {},
    "metropolis_sample_h_target": {},
    "metropolis_sample_tj_tf": {},
    "metropolis__target":{},
    "metropolis_sample_multiflip":{}
    
}

# -------------------------------
# Exact diagonalization registry
# -------------------------------
ED_SOLVERS = {
    "calc_gs": ed.calc_gs,
    "calc_gs_2D": ed.calc_gs_2D,
    "calc_gs_heis_1D": ed.calc_gs_heis_1D,
    "calc_gs_heis_2D": ed.calc_gs_heis_2D,
    "calc_gs_zf": ed.calc_gs_zf,
    "calc_gs_zf_2D":ed.calc_zfield_2D,
    "calc_gs_tj_1D": ed.calc_gs_tj_1D,
    "calc_gs_tj_2D": ed.calc_gs_tj_2D
    
}
