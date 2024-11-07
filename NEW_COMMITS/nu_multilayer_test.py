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

#tenpy.tools.misc.setup_logging(to_stdout="INFO")


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

Lx=14
LL=0
chi=600
mixing_chi=600
chi2=600
chi3=600
xi=6

Nlayer = 2; Veps = 1e-4
#sort the problem and match values to konstantinos values
V = {'eps':Veps, 'coulomb': dict([ ((s,s),{'v':1, 'xi':1}) for s in range(Nlayer) ]) }  # V1 coulomb

root_config = np.array([0, 1, 0])         # the DMRG is seeded with this config, the model can work
root_config = np.tile(root_config, Nlayer).reshape((Nlayer,-1)).transpose().reshape((-1))
N=3
model_par = {
    'verbose': 2,
    'Nlayer': Nlayer,
    'Lx': 12,                # circumference
    'Vs': V,
    'cons_C': 'total',
    'cons_K': True,
    'root_config': root_config,
    'ignore_herm_conj': True,    # this option only works with slycot
    'exp_approx': 'slycot',      # use slycot xponential approximation when constructing
    'boundary_conditions': ('infinite', N)
}










print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)


G=M.MPOgraph
#print(len(G))
#quit()


#OK FIRST AND SECOND ARE NOT THE SAME, BUT FIRST AND THIRD ARE
"""
for i in range(len(G[0].keys())):
    print('one')
    print(list(G[0].keys())[i])
    print(list(G[2].keys())[i])
"""
#print(G[1]==G[0])
#print(G[0][('_a', 0, 9)])
#quit()
G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)

#print(G_new[0]['IdL'])
#quit()
#print(G_new[0][('_a', 0, 9)])
#print(G_new[0]["('_a', 0, 9)"])
#quit()
root_config_ = np.array(root_config)
root_config_ = root_config_.reshape(6,1)
#spin=QH_MultilayerFermionSite(N=1)#,root_config=root_config_,conserve='N')

spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
L = len(G_new)


sites = [spin] * L 
print("NUMBER OF SITES")
#print(sites)
print(len(sites))
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
print(L)
print(len(G_new))
for i in range(L):
    
    for element in not_included_couplings[i]:
        
        G_new[i].pop(element[0],None)
        print(i)
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


lattice=Chain(L,spin, bc="periodic",  bc_MPS="infinite")

model=MPOModel(lattice, H)

print("created model",".."*30)
    

pstate=["empty", "empty","full","full","empty","empty"]
psi = MPS.from_product_state(sites, pstate, bc="infinite")

dmrg_params={"max_E_err": 1e-6,"max_S_err": 1e-7,"N_sweeps_check":2,"trunc_params": {"chi_max": 450, "trunc_cut": 1.e-7}, "mixer":True,
             "lanczos_params": {'chi_list':{0:mixing_chi, 12:chi},'N_min': 2, 'N_max': 20, 'P_tol': 2e-6, 'P_tol_to_trunc': 1/25., 'cache_v':np.inf}, "max_sweeps":36}

print("Run DMRG:")
engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)  
E0, psi = engine.run()

print("entanglement entropy")
x=psi.entanglement_entropy()
print(x)