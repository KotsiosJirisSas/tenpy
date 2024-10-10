"""Call of (finite) DMRG."""
import sys
import os
sys.path.append('/home/v/vasiliou/tenpynew2/tenpy') 
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models import multilayer_qh_DP_final as mod
import itertools


from tenpy.networks.mps import MPS
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import QH_MultilayerFermionSite



np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

NLL = 1; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)

#Potential data for (single/multilayered) model Laughlin
V = { 'eps':Veps, 'xiK':2., 'rV(r)': { ('L','L'): {'rV': rvr} }, 'coulomb': { ('L','L'):  {'v':-1., 'xi': xi}} }

root_config = [0]*NLL


N=2
model_par = {

	#ahhhh ok ok so it constructs the periodic one, ggwp with 24 sites for some reason
	# - for reason of there being 2 layers!
	'boundary_conditions': ('infinite', N), #for finite replace by periodic here
	'verbose': 2,
	#'layers': [ ('L', l) for l in range(NLL) ],
	'layers':[ ('L', 1)],
	'Lx': 12.,
	'Vs': V,
	'cons_C': 'total',
	'cons_K': False,
	'root_config': root_config,
	'exp_approx': 'slycot',
}

print ("-"*10 + "Comparing analytic V(q) Yukawa and V(r) Yukawa" +"-"*10)
print("START MODEL")
M = mod.QH_model(model_par)
print('OLD CODE FINISHED')
quit()

'''
H_MPO=M.H_mpo


model_params={"L": N, "bc_MPS": "infinite", 'site':None, 'bc':'periodic'}
#NEW CODE STARTS HERE

hilber_space_single_site=QH_MultilayerFermionSite(N=1)
#DO WE NEED H_bonds in chain?
define_chain=Chain(N,hilber_space_single_site)
H = MPO.from_grids([hilber_space_single_site] * N, M.H_mpo, bc='infinite', IdL=0, IdR=-1)
#MANAGED TO DEFINE A CHAIN WITH (HOPEFULLY) CORRECT MATRICES
'''