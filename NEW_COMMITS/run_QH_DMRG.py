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
from tenpy.networks.site import QH_MultilayerFermionSite
print(sys.executable)




###	Layer naming scheme	###
# See line 22 of multilayer_qh for details
# A layer is a component - like a spin species, or bilayer index. It can have any key as name.
# Each layer can have multiple Landau levels.
# The Hilbert space is specified by
# 'layers': [  ( layer_key, landau level), (layer_key, landau level) , . . . ]
# For example,  [ ('up', 0), ('up', 1), ('down', 1) ]
# will have two species, 'up' and 'down'; 'up' contains LLs of n = 0 and 1, and 'down' a single n = 1 LL.
# For nu = 1/3,   we will use  [ ('L', LL)], with LL specified below.

np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)
#########################
Lx = 14;			# circumference
LL = 0;			# which Landau level to put in
mixing_chi = 450; #Bond dimension in initial sweeps
chi = 450;		#Bond dimension of MPS
xi = 6;			# The Gaussian falloff for the Coulomb potential
Veps = 1e-4		# how accurate to approximate the MPO


V = { 'eps':Veps, 'xiK':xi, 'GaussianCoulomb': {('L','L'):{'v':1, 'xi':xi}} }

root_config = np.array([0, 1, 0])		# this is how the initial wavefunction looks

model_par = {
	'verbose': 3,
	'layers': [ ('L', LL) ],
	'Lx': Lx,
	'Vs': V,
	'boundary_conditions': ('infinite', 1),
	'cons_C': 'total', #Conserve number for each species (only one here!)
	'cons_K': False, #Conserve K
	'root_config': root_config, #Uses this to figure out charge assignments
	'exp_approx': '1in', #For multiple orbitals, 'slycot' is more efficient; but for 1 orbital, Roger's handmade code '1in' is slightly more efficient
}

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



G_og=[{('a_', 0, 3): {('a_', 0, 4): [('StrOp', 1.0)], ('a', 0, 3, 'A', 0, 0): [('AOp', 1.0)], ('a', 0, 3, 'A', 0, 1): [('AOp', 1.0)]}, ('_a', 0, 3): {('_a', 0, 2): [('StrOp', 1.0)]}, ('A_', 0, 6): {}, ('a', 0, 3, 'A', 0, 1): {('a', 0, 3, 'A', 0, 1): [('Id', 0.34475573503532636)], ('_a', 0, 2): [('AOp', -0.019830660130506394)]}, ('_a', 0, 2): {('_a', 0, 1): [('StrOp', 1.0)]}, ('A', 0, 1, 'a', 0, 0): {('_A', 0, 0): [('aOp', 0.13832145119393363)], ('A', 0, 1, 'a', 0, 0): [('Id', 0.8005881679881267)]}, ('A', 0, 2, 'a', 0, 1): {('_A', 0, 1): [('aOp', -0.01359559860484505)], ('A', 0, 2, 'a', 0, 2): [('Id', 0.21324533869805767)], ('A', 0, 2, 'a', 0, 1): [('Id', 0.3425333055817517)]}, ('_A', 0, 1): {('_A', 0, 0): [('StrOp', 1.0)]}, ('a', 0, 2, 'A', 0, 2): {('_a', 0, 1): [('AOp', 0.013062430119161455)], ('a', 0, 2, 'A', 0, 2): [('Id', 0.3425333055817541)], ('a', 0, 2, 'A', 0, 1): [('Id', -0.21324533869805679)]}, ('_a', 0, 4): {('_a', 0, 3): [('StrOp', 1.0)]}, ('_A', 0, 0): {'F': [('AOp', 1.0)]}, ('a', 0, 3, 'A', 0, 0): {('a', 0, 3, 'A', 0, 0): [('Id', 0.4619497352374192)], ('_a', 0, 2): [('AOp', 0.025248692719033152)]}, ('a', 0, 4, 'A', 0, 1): {('a', 0, 4, 'A', 0, 1): [('Id', 0.33871110444159785)], ('_a', 0, 3): [('AOp', 0.006402357895473034)], ('a', 0, 4, 'A', 0, 0): [('Id', -0.040305948895300314)]}, ('a', 0, 5, 'A', 0, 0): {('a', 0, 5, 'A', 0, 0): [('Id', 0.5138512188414278)], ('_a', 0, 4): [('AOp', 0.00011023908392563828)]}, 'F': {'F': [('Id', 1.0)]}, ('A', 0, 0, 'a', 0, 0): {('A', 0, 0, 'a', 0, 0): [('Id', 0.8337251361291302)], ('A', 0, 0, 'a', 0, 1): [('Id', 0.09379746220264774)], 'F': [('nOp', -0.2660018610388971)]}, ('A', 0, 1, 'a', 0, 3): {('A', 0, 1, 'a', 0, 4): [('Id', 0.23708339098378298)], ('_A', 0, 0): [('aOp', -0.04380586818225882)], ('A', 0, 1, 'a', 0, 3): [('Id', 0.38172630104765676)]}, ('_A', 0, 2): {('_A', 0, 1): [('StrOp', 1.0)]}, ('a_', 0, 2): {('a_', 0, 3): [('StrOp', 1.0)], ('a', 0, 2, 'A', 0, 0): [('AOp', 1.0)], ('a', 0, 2, 'A', 0, 1): [('AOp', 1.4142135623730951)]}, ('A_', 0, 5): {('A', 0, 5, 'a', 0, 0): [('aOp', 1.0)], ('A_', 0, 6): [('StrOp', 1.0)]}, ('_A', 0, 5): {('_A', 0, 4): [('StrOp', 1.0)]}, ('a', 0, 1, 'A', 0, 4): {('a', 0, 1, 'A', 0, 4): [('Id', 0.38172630104763583)], ('_a', 0, 0): [('AOp', 0.030390807643815607)], ('a', 0, 1, 'A', 0, 3): [('Id', -0.23708339098377437)]}, ('a', 0, 1, 'A', 0, 1): {('_a', 0, 0): [('AOp', 0.0061305932314431565)], ('a', 0, 1, 'A', 0, 1): [('Id', 0.7630762915951965)], ('a', 0, 1, 'A', 0, 2): [('Id', 0.14797043763385084)]}, ('a', 0, 4, 'A', 0, 0): {('a', 0, 4, 'A', 0, 1): [('Id', 0.040305948895300314)], ('_a', 0, 3): [('AOp', 0.0006454127172432714)], ('a', 0, 4, 'A', 0, 0): [('Id', 0.33871110444159785)]}, ('_A', 0, 4): {('_A', 0, 3): [('StrOp', 1.0)]}, ('A_', 0, 4): {('A_', 0, 5): [('StrOp', 1.0)], ('A', 0, 4, 'a', 0, 0): [('aOp', 1.4142135623730951)]}, ('A', 0, 0, 'a', 0, 1): {('A', 0, 0, 'a', 0, 0): [('Id', -0.09379746220264774)], ('A', 0, 0, 'a', 0, 1): [('Id', 0.8337251361291302)], 'F': [('nOp', 0.05253891639028042)]}, ('A', 0, 1, 'a', 0, 2): {('_A', 0, 0): [('aOp', -0.01009450230953929)], ('A', 0, 1, 'a', 0, 1): [('Id', -0.14797043763386378)], ('A', 0, 1, 'a', 0, 2): [('Id', 0.7630762915952091)]}, ('a', 0, 2, 'A', 0, 0): {('_a', 0, 1): [('AOp', 0.042544852049328626)], ('a', 0, 2, 'A', 0, 0): [('Id', 0.6410408600755128)]}, ('A', 0, 0, 'a', 0, 5): {('A', 0, 0, 'a', 0, 5): [('Id', 0.3495781422729576)], 'F': [('nOp', -0.23103170492184383)]}, ('A', 0, 2, 'a', 0, 2): {('_A', 0, 1): [('aOp', 0.013062430119161288)], ('A', 0, 2, 'a', 0, 2): [('Id', 0.3425333055817517)], ('A', 0, 2, 'a', 0, 1): [('Id', -0.21324533869805767)]}, ('A', 0, 3, 'a', 0, 1): {('A', 0, 3, 'a', 0, 1): [('Id', 0.34475573503532886)], ('_A', 0, 2): [('aOp', -0.019830660130507678)]}, ('A', 0, 4, 'a', 0, 0): {('A', 0, 4, 'a', 0, 1): [('Id', 0.040305948895300446)], ('_A', 0, 3): [('aOp', 0.0006454127172432719)], ('A', 0, 4, 'a', 0, 0): [('Id', 0.3387111044415981)]}, ('A', 0, 0, 'a', 0, 4): {('A', 0, 0, 'a', 0, 4): [('Id', 0.2943158206367369)], ('A', 0, 0, 'a', 0, 3): [('Id', -0.36314516089054333)], 'F': [('nOp', -0.026162348887412326)]}, ('a', 0, 1, 'A', 0, 0): {('_a', 0, 0): [('AOp', 0.1383214511939045)], ('a', 0, 1, 'A', 0, 0): [('Id', 0.8005881679881443)]}, ('a', 0, 2, 'A', 0, 1): {('_a', 0, 1): [('AOp', -0.013595598604845185)], ('a', 0, 2, 'A', 0, 2): [('Id', 0.21324533869805679)], ('a', 0, 2, 'A', 0, 1): [('Id', 0.3425333055817541)]}, ('a_', 0, 1): {('a', 0, 1, 'A', 0, 1): [('AOp', 1.4142135623730951)], ('a', 0, 1, 'A', 0, 0): [('AOp', 1.0)], ('a', 0, 1, 'A', 0, 3): [('AOp', 1.4142135623730951)], ('a_', 0, 2): [('StrOp', 1.0)]}, ('A_', 0, 1): {('A', 0, 1, 'a', 0, 1): [('aOp', 1.4142135623730951)], ('A', 0, 1, 'a', 0, 0): [('aOp', 1.0)], ('A', 0, 1, 'a', 0, 3): [('aOp', 1.4142135623730951)], ('A_', 0, 2): [('StrOp', 1.0)]}, ('a_', 0, 6): {}, ('_a', 0, 5): {('_a', 0, 4): [('StrOp', 1.0)]}, ('_a', 0, 0): {'F': [('aOp', 1.0)]}, ('A', 0, 1, 'a', 0, 1): {('_A', 0, 0): [('aOp', 0.006130593231424408)], ('A', 0, 1, 'a', 0, 1): [('Id', 0.7630762915952091)], ('A', 0, 1, 'a', 0, 2): [('Id', 0.14797043763386378)]}, ('A', 0, 3, 'a', 0, 0): {('A', 0, 3, 'a', 0, 0): [('Id', 0.4619497352374154)], ('_A', 0, 2): [('aOp', 0.025248692719034436)]}, ('A', 0, 4, 'a', 0, 1): {('A', 0, 4, 'a', 0, 1): [('Id', 0.3387111044415981)], ('_A', 0, 3): [('aOp', 0.006402357895473009)], ('A', 0, 4, 'a', 0, 0): [('Id', -0.040305948895300446)]}, ('a_', 0, 5): {('a', 0, 5, 'A', 0, 0): [('AOp', 1.0)], ('a_', 0, 6): [('StrOp', 1.0)]}, ('A', 0, 5, 'a', 0, 0): {('A', 0, 5, 'a', 0, 0): [('Id', 0.5138512188414278)], ('_A', 0, 4): [('aOp', 0.00011023908392563828)]}, ('a', 0, 1, 'A', 0, 3): {('a', 0, 1, 'A', 0, 4): [('Id', 0.23708339098377437)], ('_a', 0, 0): [('AOp', -0.043805868182256937)], ('a', 0, 1, 'A', 0, 3): [('Id', 0.38172630104763583)]}, ('_A', 0, 3): {('_A', 0, 2): [('StrOp', 1.0)]}, ('A_', 0, 2): {('A_', 0, 3): [('StrOp', 1.0)], ('A', 0, 2, 'a', 0, 0): [('aOp', 1.0)], ('A', 0, 2, 'a', 0, 1): [('aOp', 1.4142135623730951)]}, ('A', 0, 0, 'a', 0, 3): {('A', 0, 0, 'a', 0, 4): [('Id', 0.36314516089054333)], ('A', 0, 0, 'a', 0, 3): [('Id', 0.2943158206367369)], 'F': [('nOp', -0.015888996698317146)]}, ('A', 0, 1, 'a', 0, 4): {('A', 0, 1, 'a', 0, 4): [('Id', 0.38172630104765676)], ('_A', 0, 0): [('aOp', 0.03039080764381981)], ('A', 0, 1, 'a', 0, 3): [('Id', -0.23708339098378298)]}, ('_a', 0, 1): {('_a', 0, 0): [('StrOp', 1.0)]}, 'R': {('A', 0, 0, 'a', 0, 5): [('nOp', 1.0)], ('A', 0, 0, 'a', 0, 0): [('nOp', 1.4142135623730951)], ('A_', 0, 1): [('AOp', 1.0)], ('A', 0, 0, 'a', 0, 3): [('nOp', 1.4142135623730951)], ('a_', 0, 1): [('aOp', 1.0)], 'R': [('Id', 1.0)], ('A', 0, 0, 'a', 0, 2): [('nOp', 1.0)]}, ('A', 0, 2, 'a', 0, 0): {('_A', 0, 1): [('aOp', 0.042544852049328445)], ('A', 0, 2, 'a', 0, 0): [('Id', 0.6410408600755139)]}, ('A_', 0, 3): {('A_', 0, 4): [('StrOp', 1.0)], ('A', 0, 3, 'a', 0, 0): [('aOp', 1.0)], ('A', 0, 3, 'a', 0, 1): [('aOp', 1.0)]}, ('a_', 0, 4): {('a_', 0, 5): [('StrOp', 1.0)], ('a', 0, 4, 'A', 0, 0): [('AOp', 1.4142135623730951)]}, ('a', 0, 1, 'A', 0, 2): {('_a', 0, 0): [('AOp', -0.010094502309538074)], ('a', 0, 1, 'A', 0, 1): [('Id', -0.14797043763385084)], ('a', 0, 1, 'A', 0, 2): [('Id', 0.7630762915951965)]}, ('A', 0, 0, 'a', 0, 2): {('A', 0, 0, 'a', 0, 2): [('Id', 0.8330402460772576)], 'F': [('nOp', 0.8454122673973181)]}}]


print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)
#quit()
G=M.MPOgraph
#dict1=G[0]
#dict2=G_og[0]
#for a in G[0].keys():
#	only_in_dict1 =  dict2[a].keys()#-dict1[a].keys()
#	for b in dict2[a].keys():
#		print(dict1[a][b]==dict2[a][b])
#		print(dict1[a][b])
#		print(dict2[a][b])
#		#print(only_in_dict1)
#quit()

#print(G_og[0]['R']['R'])
#print('BREEAAAAAAAK')
#print(G[0]==G_og[0])
#quit()

#print(G[0].keys())
#x=G[0][('_a', 0, 2)]

#print(x)
#quit()
G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)

#print(G_new[0]["('_a', 0, 2)"])
#quit()
#print(G_new)
#quit()
#print(len(G_new))
#quit()
#G_new=[G_new[0],G_new[0],G_new[0]]
#print(len(G_new))
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
#spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
spin=QH_MultilayerFermionSite(N=1)
L = len(G_new) #System size for finite case, or unit cell for infinite
sites = [spin] * L 
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
M.graph = G_new #: Inpput the graph in the model 
print("Test passed!"+".."*10)
grids =M._build_grids()#:Build the grids from the graph

#quit()
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