"""Call of (finite) DMRG."""

import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models import multilayer_qh as mod
from tenpy.models.multilayer_qh import interlace_zero
#from tenpy.algorithms import simulation
import itertools
#from tenpy.mps.mps import iMPS
np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

NLL = 2; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)

#Potential data for (single/multilayered) model Laughlin
V = { 'eps':Veps, 'xiK':2., 'rV(r)': { ('L','L'): {'rV': rvr} }, 'coulomb': { ('L','L'):  {'v':-1., 'xi': xi}} }

root_config = [0]*NLL

model_par = {

	#ahhhh ok ok so it constructs the periodic one, ggwp with 24 sites for some reason
	# - for reason of there being 2 layers!
	'boundary_conditions': ('periodic', 8),
	'verbose': 2,
	#'layers': [ ('L', l) for l in range(NLL) ],
	'layers':[ ('L', 1)],
	'Lx': 12.,
	'Vs': V,
	'cons_C': 'total',
	'cons_K': True,
	'root_config': root_config,
	'exp_approx': 'slycot',
}

print ("-"*10 + "Comparing analytic V(q) Yukawa and V(r) Yukawa" +"-"*10)
print("START MODEL")
M = mod.QH_model(model_par)
print('OLD CODE FINISHED')


from tenpy.networks.mps import MPS
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPO

N=8

H_MPO=M.H_mpo
#print(M.H_mpo[0][0])
#print(len(M.H_mpo[0][0][0][0][0]))
#quit()
#IMPORT HMPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import QH_MultilayerFermionSite

from tenpy.networks.mpo import MPO
model_params={"L": 8, "bc_MPS": "finite", 'site':None, 'bc':'periodic'}
#lattice=Lattice(model_params)

hilber_space_single_site=QH_MultilayerFermionSite(N=1)
define_chain=Chain(N,hilber_space_single_site)

print(len(H_MPO))

#NEED TO MATCH THESE PROPERLY???
#NEED TO READ OUT Ws!
H = MPO.from_grids([hilber_space_single_site] * N, M.H_mpo, bc='periodic', IdL=0, IdR=-1)
#H = MPO.from_grids([hilber_space_single_site] * N, M.H_mpo, bc='finite', IdL=0, IdR=-1)
#MANAGED TO DEFINE A CHAIN WITH (HOPEFULLY) CORRECT MATRICES

quit()
model=QHChain(model_params,H_MPO)

quit()


sites = model.lat.mps_sites()
psi = MPS.from_product_state(sites, ['up'] * N, "finite")
dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True}
info = dmrg.run(psi, model, dmrg_params)
print("E =", info['E'])
# E = -20.01638790048513
print("max. bond dimension =", max(psi.chi))
# max. bond dimension = 27
