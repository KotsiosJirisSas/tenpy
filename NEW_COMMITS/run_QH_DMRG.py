
"""Code for running DMRG """
import sys
import os
sys.path.append('/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy') 
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models import multilayer_qh_DP_final as mod
import itertools
from tenpy.networks.mps import MPS
from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import QH_MultilayerFermionSite_2
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
import QH_G2MPO
import QH_Graph_final


np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

NLL = 1; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)

#Potential data for (single/multilayered) model Laughlin
V = { 'eps':Veps, 'xiK':2., 'rV(r)': { ('L','L'): {'rV': rvr} }, 'coulomb': { ('L','L'):  {'v':-1., 'xi': xi}} }

root_config = [0]*NLL


N=3
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

print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)


G=M.MPOgraph
G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)


root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')

L = len(G_new) #System size for finite case, or unit cell for infinite
sites = [spin] * L 
M = MPOGraph(sites=sites,bc='infinite',max_range=None) #: Initialize MPOGRAPH instance

'''
M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
'''

States=QH_Graph_final.obtain_states_from_graphs(G_new,L)
print("Ordering states",".."*10)

M.states = States #: Initialize aux. states in model
M._ordered_states = QH_G2MPO.set_ordered_states(States) #: sort these states(assign an index to each one)
print("Finished",".."*10 )



print("Test sanity"+".."*10)
M.test_sanity()
M.graph = G_new #: INppuut the graph in the model 
print("Test passed!"+".."*10)
grids =M._build_grids()#:Build the grids from the graph
print("Building MPO"+".."*10)
H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
print("Built"+".."*10)





#initialize wavefunction as MPS
pstate=["empty", "full","empty"]
psi = MPS.from_product_state(sites, pstate, bc="infinite")


#initialize MPOModel
lattice=Chain(N,spin, bc="periodic",  bc_MPS="infinite")
model=MPOModel(lattice, H)

dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True}

print("Run DMRG:")
engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)  
E0, psi = engine.run()


print("E =", E0)
print("Finished running DMRG")




Length=psi.correlation_length()
print('correlation length:',Length)



filling=psi.expectation_value("nOp")

print('Filling:',filling)


E_spec=psi.entanglement_spectrum()
print('entanglement spectrum:',E_spec)



EE=psi.entanglement_entropy()
print('entanglement entropy:',EE)


#SAVE THE DATA
import h5py
from tenpy.tools import hdf5_io

data = {"psi": psi,  # e.g. an MPS
        "dmrg_params":dmrg_params, "model_par":model_par, "model": model }

name="nu=1_3_charge_conservation_no_K_conservation"
with h5py.File(name+".h5", 'w') as f:
    hdf5_io.save_to_hdf5(f, data)