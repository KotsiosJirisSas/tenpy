#!/usr/bin/env python
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150

import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.xxz_chain import XXZChain
from tenpy.models.tf_ising import TFIChain

tenpy.tools.misc.setup_logging(to_stdout="INFO")


from tenpy.models.lattice import TrivialLattice
from tenpy.models.model import MPOModel
from tenpy.networks.mpo import MPOEnvironment
from tenpy.networks.mps import MPSEnvironment


import pickle
from tenpy.linalg.charges import LegCharge
from tenpy.linalg.charges import ChargeInfo
from tenpy.networks.site import QH_MultilayerFermionSite_2
from tenpy.linalg.np_conserved import Array

from tenpy.networks.site import QH_MultilayerFermionSite

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
#sort the problem and match values to konstantinos values
Lx = 22;			# circumference
LL = 0;			# which Landau level to put in
xi = 6;			# The Gaussian falloff for the Coulomb potential
Veps = 1e-4		# how accurate to approximate the MPO


N=6


def rvr(r):
    return np.exp(-r/xi)
V = { 'eps':Veps, 'xiK':xi, 'GaussianCoulomb': {('L','L'):{'v':1, 'xi':xi}} }

root_config = np.array([0, 1, 0])		# this is how the initial wavefunction looks

model_par = {
    'verbose': 3,
    'layers': [ ('L', LL),('L',LL) ],
    'Lx': Lx,
    'Vs': V,
    'boundary_conditions': ('infinite', N),
    'cons_C': 'total', #Conserve number for each species (only one here!)
    'cons_K': False, #Conserve K
    'root_config': root_config, #Uses this to figure out charge assignments
    'exp_approx': '1in', #For multiple orbitals, 'slycot' is more efficient; but for 1 orbital, Roger's handmade code '1in' is slightly more efficient
}










print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)


G=M.MPOgraph
print(len(G))
quit()
#print(G[0].keys())
#print(G[0][('_a', 0, 9)])
#quit()
G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
#print(G_new[0][('_a', 0, 9)])
#print(G_new[0]["('_a', 0, 9)"])
#quit()
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
spin=QH_MultilayerFermionSite(N=1,root_config=root_config_,conserve='N')
L = len(G_new)


sites = [spin] * L 
#print(sites)

#SORT THE STUPID ASSERTION PROBLEM WHICH ARISES AT CERTAIN VALUES
M = MPOGraph(sites=sites,bc='infinite',max_range=None) #: Initialize MPOGRAPH instance

'''
M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
'''

States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)
print("Ordering states",".."*10)

M.states = States #: Initialize aux. states in model
M._ordered_states = QH_G2MPO.set_ordered_states(States) #: sort these states(assign an index to each one)



#remove rows
#yet to implement remove columns
for i in range(L):
    
    for element in not_included_couplings:
        
        G_new[i].pop(element[0],None)
        print(element[0])


print("Finished",".."*10 )


print("Test sanity"+".."*10)
M.test_sanity()
M.graph = G_new #: INppuut the graph in the model 
print("Test passed!"+".."*10)
#grids =M._build_grids()#:Build the grids from the graph
print("Building MPO"+".."*10)
#quit()

H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
print("Built"+".."*10)



#initialize wavefunction as MPS


lattice=Chain(N,spin, bc="periodic",  bc_MPS="infinite")

model=MPOModel(lattice, H)

print("created model",".."*30)
    
