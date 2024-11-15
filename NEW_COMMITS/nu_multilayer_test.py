#!/usr/bin/env python
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys
print('INNNNN')
sys.path.append('/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy') 
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

Lx = 14;            # circumference
LL = 0;         # which Landau level to put in
mixing_chi = 300; #Bond dimension in initial sweeps
chi = 340;      #Bond dimension of MPS
chi2 = 400
chi3 = 400
#chi4 = 4500
xi = 6; #
#xi = 1;            # The Gaussian falloff for the Coulomb potential
Veps = 1e-4 # how accurate to approximate the MPO
Nlayer = 1; Veps = 1e-4
V = {'eps':Veps, 'GaussianCoulomb': dict([ ((s,s),{'v':1, 'xi':xi}) for s in range(Nlayer) ]) }

V={'eps': 0.0001, 'GaussianCoulomb': {(0, 0): {'v': 1, 'xi': 6}}}
#print(V)
#quit()

a=set(['e','a','c','d','2'])
#print(list(a))
#quit()


root_config = np.array([0, 1, 0])         # the DMRG is seeded with this config, the model can work
root_config = np.tile(root_config, Nlayer).reshape((Nlayer,-1)).transpose().reshape((-1))

root_config = np.array([0, 1, 0])   
N=3

model_par = {
    'verbose': 3,
    'Nlayer':Nlayer,
    'Lx': Lx,
    'Vs': V,
    'boundary_conditions': ('infinite', N),#????????????????????????
#   'boundary_conditions': ('periodic', 1),
    'cons_C': 'total', #Conserve number for each species (only one here!)
    'cons_K': True, #Conserve K
    'root_config': root_config, #Uses this to figure out charge assignments
    #'ignore_herm_conj': True,      # this option only works with slycot
    'exp_approx': '1in',
}










print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)


G=M.MPOgraph#[0]
#print(len(G))
#quit()

name="MPO_QH_1_3"#_single"

with open(name+'.pkl', 'rb') as f:
    loaded_xxxx = pickle.load(f, encoding='latin1')
print(loaded_xxxx.keys())
#quit()
G_old=loaded_xxxx['graph'][0]
#print(G_old)
#G=create_infinite_DMRG_model(N)[0]
print(len(G))
print(len(G_old))
#quit()
#x=M_i.H_MPO
diff=0
suma=0

#print(G)
#quit()
"""
for key in G_old.keys():
    #print(key)
    print('**'*100)
    print(key)
    print(G[key].keys())
    if list(G[key].keys())==[]:
        print('what what')
        print(key)
        quit()
    for key2 in G[key].keys():
        diff+=(G[key][key2][0][1]-G_old[key][key2][0][1])**2
        suma+=(G[key][key2][0][1]+G_old[key][key2][0][1])**2
        
        print(G[key][key2][0][1])
        print(G_old[key][key2][0][1])
    #print(G[key])
    #print(G_old[key])
    #print(G[key]==G_old[key])

#print(diff/suma)
"""
#quit()
#TODO: FIX JORDAN-WIGNER FOR NEW TENPY
#print(diff/suma)
#quit()

#G=G_old.copy()
#G=[G[0]]

for n in range(len(G)):
    extra_keys=[]
    for key in G[n].keys():
        #print(key)
        print('**'*100)
        print(G[n][key].keys())
        if list(G[n][key].keys())==[]:
            print('what what')
            print(key)
            extra_keys.append(key)
    for k in extra_keys:
        G[n].pop(k, None)
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
root_config_ = root_config_.reshape(3,1)
#spin=QH_MultilayerFermionSite(N=1)#,root_config=root_config_,conserve='N')

spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
L = len(G_new)


sites = [spin] * L 




#N=2
#AOp = [ np.diag([1.], -1) for s in range(N) ]	
#print(AOp[1])
#quit()
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

print(M._ordered_states)
quit()
#remove rows
#yet to implement remove columns
print(L)
print(len(G_new))
#print(not_included_couplings)
for i in range(L):
    
    for element in not_included_couplings[i]:
        if element[1]=='row':
            G_new[i].pop(element[0],None)
            #print(element[0])
        else:
            for key in G_new[i].keys():
                try:
                    G_new[i][key].pop(element[0],None)
                except:
                    pass


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

#print(H._W[0].to_ndarray())
val=H._W[0].to_ndarray()
print(np.sum(val**2))


#val=2*val/np.sqrt(np.sum(val**2))

B_val=np.array(loaded_xxxx['B'])

B_val=np.transpose(B_val,(0,1,2,3))
print(B_val.shape)
x=228

def find_indices_in_other_array(single_m,B_val):
    for z in range(x):
        count=0
        for n in range(x):
            if np.sum((B_val[z,n]-single_m)**2)<0.000001:# or np.sum(B_val[k,i]**2)!=0:
            
                #print('x'*50)
                print(B_val[z,n])
                return True
                count+=1;
                break;
        if count>0:
            break;
    
    return False


for k in range(x):
    count=0    
    for i in range(x):
        if np.sum(val[k,i]**2)!=0:# or np.sum(B_val[k,i]**2)!=0:
         
            
           
            if np.sum((val[k,i]-B_val[k,i])**2)>0.0001:
                print('x'*50)
                print(val[k,i])
                boolara=find_indices_in_other_array(val[k,i],B_val)
                print(boolara)
                if not boolara:
                    print('WRONG'*100)
                    quit()
                    count+=1
                    break;
    if count>0:
        break;
print('one is permutation of the other')
quit()

for i in range(x):
    count=0
    for k in range(x):
        
        if np.sum(val[i,k]**2)!=0 or np.sum(B_val[i,k]**2)!=0:
            print('x'*50)
            print(i,k)
            indi,indk=i,k
            print(val[i,k])
            #print(val[0,i])
            print(B_val[i,k])
            print(indi,indk)
            if i!=k:
                count+=1
                break;
    if count>0:
       break;
#quit()
matrix=np.zeros((2*x,2*x))
matrix2=np.zeros((2*x,2*x))
print('now')
for i in range(x):
    for j in range(x):
        for z in range(2):
            for m in range(2):
                #print(2*i)
                matrix[2*i+z,2*j+m]=val[i,j,z,m]
                matrix2[2*i+z,2*j+m]=B_val[i,j,z,m]
                if (i,j) ==(0,1):
                    print(val[i,j,z,m])
                    #print()
print(val[indi,indk])
print(indi,indk)
print(matrix[2*indi:2*indi+2,2*indk:2*indk+2])
print(matrix2[2*indi:2*indi+2,2*indk:2*indk+2])
quit()

lambd,s=np.linalg.eigh(matrix)
lambd2,s=np.linalg.eigh(matrix2)
#print(lambd2)
#print(lambd2-lambd)
print(np.sum((lambd)**2))
print(np.sum((lambd-lambd2)**2))
quit()

MPO_old=np.transpose(MPO_old,(0,1,2,3))
for i in range(228):
    print('x'*50)
    print(MPO_old[0,i])
    print(val[0,i])
#print(MPO_old==MPO_old2)
print(np.sum((MPO_old)**2))
print(MPO_old.shape)
print(val.shape)
diff=np.sum((MPO_old-val)**2)
scale=np.sum((MPO_old+val)**2)
print(diff/scale)
quit()
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