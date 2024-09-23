"""Call of (finite) DMRG."""

from tenpy.networks.mps import MPS
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg

N = 16  # number of sites

H_MPO=1
#IMPORT HMPO
model_params={"L": N, "J": 1., "g": 1., "bc_MPS": "finite"}

model=QHChain(model_params,H_MPO)
sites = model.lat.mps_sites()
psi = MPS.from_product_state(sites, ['up'] * N, "finite")
dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True}
info = dmrg.run(psi, model, dmrg_params)
print("E =", info['E'])
# E = -20.01638790048513
print("max. bond dimension =", max(psi.chi))
# max. bond dimension = 27
