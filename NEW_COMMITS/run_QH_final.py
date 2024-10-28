
"""Code for running DMRG """
import sys
import os
sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models import multilayer_qh_DP_final as mod
import itertools
from tenpy.networks.mps import MPS
from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS

from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPO

from tenpy.networks.site import QH_MultilayerFermionSite_2
from tenpy.networks.site import QH_MultilayerFermionSite_3
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
import QH_G2MPO
import QH_Graph_final



np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)
#########################
Lx = 14;			# circumference
LL = 0;			# which Landau level to put in
mixing_chi = 450; #Bond dimension in initial sweeps
chi = 450;		#Bond dimension of MPS
xi = 6;			# The Gaussian falloff for the Coulomb potential
Veps = 1e-4		# how accurate to approximate the MPO


NLL = 1; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)
V = { 'eps':Veps, 'xiK':xi, 'GaussianCoulomb': {('L','L'):{'v':1, 'xi':xi}} }

root_config = np.array([0, 1, 0])		# this is how the initial wavefunction looks
broj=20
N=3*broj
model_par = {
	'verbose': 3,
	'layers': [ ('L', LL) ],
	'Lx': Lx,
	'Vs': V,
	'boundary_conditions': ('periodic', N),
	'cons_C': 'total', #Conserve number for each species (only one here!)
	'cons_K': False, #Conserve K
	'root_config': root_config, #Uses this to figure out charge assignments
	'exp_approx': '1in', #For multiple orbitals, 'slycot' is more efficient; but for 1 orbital, Roger's handmade code '1in' is slightly more efficient
}
"""
dmrg_par = {
	'N_STEPS': 2,
	#'STARTING_ENV_FROM_PSI': 21,
	'MAX_STEPS': 36,
	'MIN_STEPS': 16,
	'MAX_ERROR_E' : 1e-6,
	'MAX_ERROR_S' : 1e-4,
	'CHI_LIST': {0:mixing_chi, 12:chi},
	'TRUNC_CUT': 1e-9,
	'LANCZOS_PAR' : {'N_min': 2, 'N_max': 20, 'p_tol': 5e-6, 'p_tol_to_trunc': 1/25., 'cache_v':np.inf},
	'mixer': (0.000001, 2., 10, 'id'),
}
"""










print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)


G=M.MPOgraph

#print(G[0][('Mk', 'AL-6-aL.11', 0)])
#quit()
G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
#print("start")
#print(G_new[0]["('Mk', 'AL-6-aL.11', 27)"])
#quit()
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
#spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
L = len(G_new)

sites=[]
for i in range(L):
	spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
	#spin=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=('N','K'),site_loc=i)
	#spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
	sites.append(spin)


#sites = [spin] * L 
print(sites)
#quit()
#print(len(sites))

M = MPOGraph(sites=sites,bc='finite',max_range=None) #: Initialize MPOGRAPH instance

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
pstate=["empty", "full","empty"]*broj
psi = MPS.from_product_state(sites, pstate, bc="finite")

#simple_lattice()

#initialize MPOModel

from tenpy.models.lattice import Lattice
from tenpy.models.lattice import IrregularLattice
#lattice=Chain(L=N,site=sites, bc="periodic",  bc_MPS="finite")
#lattice=Chain(N,spin, bc="periodic",  bc_MPS="finite")
pos= [[i] for i in range(L)]

#quit()
lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="finite")
x=lattice.mps_sites()
#print(lattice.N_cells)
#print(lattice.N_sites)
#print(lattice.N_sites_per_ring)
#print(lattice.unit_cell_positions)
#quit()
#print(len(x))
from tenpy.models.lattice import CustomLattice
#lattice=CustomLattice(sites,bc="periodic",  bc_MPS="finite")
#irr_lat = IrregularLattice(lattice, remove=[[L - 1, 1],[L-2,1]])
#for i in range(N):
#	print(i)
#x=irr_lat.mps_sites()
#print(len(x))
#print('irregular lattice')
#quit()
model=MPOModel(lattice, H)

#print(lattice.Ls)
#print(psi.L)
#quit()
dmrg_params = {"trunc_params": {"chi_max": 450, "svd_min": 1.e-9}, "mixer": (0.000001, 2., 10, 'id')}
#'LANCZOS_PAR' : {'N_min': 2, 'N_max': 20, 'p_tol': 5e-6, 'p_tol_to_trunc': 1/25., 'cache_v':np.inf}


"""
'N_STEPS': 2,
	#'STARTING_ENV_FROM_PSI': 21,
	'MAX_STEPS': 36,
	'MIN_STEPS': 16,
	'MAX_ERROR_E' : 1e-6,
	'MAX_ERROR_S' : 1e-4,
	'CHI_LIST': {0:mixing_chi, 12:chi},
	'TRUNC_CUT': 1e-9,
	'LANCZOS_PAR' : {'N_min': 2, 'N_max': 20, 'p_tol': 5e-6, 'p_tol_to_trunc': 1/25., 'cache_v':np.inf},
	'mixer': (0.000001, 2., 10, 'id'),
#mixer_params={"amplitude":,"decay":, "disable_after":}
"""

dmrg_params={"max_E_err": 1e-6,"max_S_err": 1e-4,"N_sweeps_check":2,"trunc_params": {"chi_max": 450, "trunc_cut": 1.e-9}, "mixer":True,
             "lanczos_params": {'chi_list':{0:mixing_chi, 12:chi},'N_min': 2, 'N_max': 20, 'P_tol': 5e-6, 'P_tol_to_trunc': 1/25., 'cache_v':np.inf}, "max_sweeps":36}

print("Run DMRG:")
engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)  
E0, psi = engine.run()


print("E =", E0)
print("Finished running DMRG")


quit()

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