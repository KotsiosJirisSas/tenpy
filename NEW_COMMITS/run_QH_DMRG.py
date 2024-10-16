#!/usr/bin/env python
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
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import QH_MultilayerFermionSite_2
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
import QH_G2MPO
import QH_Graph_final
print(sys.executable)


np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)


###	Layer naming scheme	###
# See line 22 of multilayer_qh for details
# A layer is a component - like a spin species, or bilayer index. It can have any key as name.
# Each layer can have multiple Landau levels.
# The Hilbert space is specified by
# 'layers': [  ( layer_key, landau level), (layer_key, landau level) , . . . ]
# For example,  [ ('up', 0), ('up', 1), ('down', 1) ]
# will have two species, 'up' and 'down'; 'up' contains LLs of n = 0 and 1, and 'down' a single n = 1 LL.
# For nu = 1/3,   we will use  [ ('L', LL)], with LL specified below.


##	These are parameters to run
Lx = 16;			# circumference
LL = 0;			# which Landau level to put in
mixing_chi = 850; #Bond dimension in initial sweeps
chi = 1250;		#Bond dimension of MPS
xi = 6;			# The Gaussian falloff for the Coulomb potential
Veps = 1e-5		# how accurate to approximate the MPO

### Specifying potentials ###
# A variety of potentials have been coded - haldane, TK, GaussianCoulomb, or an arbitrary real space V(r)
# The general syntax is
# V[potential_type] =  {  (layer1, layer2): info_dict, (layer1, layer2): info_dict, ... }
# where the tuple (layer1, layer2) specifies which two species are interacting, and info_dict details the potential, with data that depends on the type of potential

# V also has some other possible keys; for instance:
# eps: MPO approximation error
# xiK: estimate of real space extent of potential. This gives heuristic for how the integrals are performed for the matrix elements of the interaction (can be set to whatever 'xi' is without problem)


#Potential data for (single/multilayered) model Laughlin
def rV(r):		# this is r * V(r)
	return np.exp(-(r/xi)**2/2.)
#V = { 'eps':Veps, 'xiK':xi, 'rV(r)': {('L','L'):{'rV':rV}} } # allow you to specify an arbitrary potential

V = { 'eps':Veps, 'xiK':xi, 'GaussianCoulomb': {('L','L'):{'v':1, 'xi':xi}} } # use the built-in function

root_config = np.array([0, 1, 0])		# this is how the initial wavefunction looks

model_par = {
	'verbose': 3,
	'layers': [ ('L', LL) ],
	'Lx': Lx,
	'Vs': V,
#	'boundary_conditions': ('periodic', 1),
	'cons_C': 'total', #Conserve number for each species (only one here!)
	'cons_K': False, #Conserve K
	'root_config': root_config, #Uses this to figure out charge assignments
	'exp_approx': '1in', #For multiple orbitals, 'slycot' is more efficient; but for 1 orbital, Roger's handmade code '1in' is slightly more efficient
}

#DMRG run parameters. See DMRG.run for list of such parameters.

dmrg_par = {
	'N_STEPS': 2,
	'STARTING_ENV_FROM_PSI': 21,
	'MAX_STEPS': 36,
	'MIN_STEPS': 16,
	'MAX_ERROR_E' : 1e-6,
	'MAX_ERROR_S' : 1e-4,
	'CHI_LIST': {0:mixing_chi, 12:chi},
	'TRUNC_CUT': 1e-9,
	'LANCZOS_PAR' : {'N_min': 2, 'N_max': 20, 'p_tol': 5e-6, 'p_tol_to_trunc': 1/25., 'cache_v':np.inf},
	'mixer': (0.000001, 2., 10, 'id'),
}





print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)
#quit()

G=M.MPOgraph

G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
#print(len(G_new))
#quit()
G_new=[G_new[0],G_new[0],G_new[0]]
print(len(G_new))
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
lattice=Chain(L,spin, bc="periodic",  bc_MPS="infinite")
model=MPOModel(lattice, H)

dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True, "max_sweeps":100}

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