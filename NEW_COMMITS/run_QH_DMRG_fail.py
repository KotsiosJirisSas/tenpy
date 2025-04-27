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

def eliminate_elements_from_the_graph(G,not_included_couplings):
    """
    Trims down the MPO from the keys that are not connected to any legs, i.e. are connected to zero block.
    
    input: G which is graph, not_included_couplings - list of expelled keys
    output: G without unnecessary keys
    """

    L=len(G)
    for i in range(L):
       
        for element in not_included_couplings:
        
            try:
                G[i].pop(element[0],None)
                
            except:
                pass
            
            for key in G[i].keys():
                try:
                    G[i][key].pop(element[0],None)
                except:
                    pass
    return G


###	Layer naming scheme	###
# See line 22 of multilayer_qh for details
# A layer is a component - like a spin species, or bilayer index. It can have any key as name.
# Each layer can have multiple Landau levels.
# The Hilbert space is specified by
# 'layers': [  ( layer_key, landau level), (layer_key, landau level) , . . . ]
# For example,  [ ('up', 0), ('up', 1), ('down', 1) ]
# will have two species, 'up' and 'down'; 'up' contains LLs of n = 0 and 1, and 'down' a single n = 1 LL.
# For nu = 1/3,   we will use  [ ('L', LL)], with LL specified below.

def trim_down_the_MPO(G):
    extra_keys=[]
    for n in range(len(G)):
        extra_keys_row=[]
        for key in G[n].keys():
            #print(key)
            
            #print(G[n][key].keys())
            if list(G[n][key].keys())==[]:
                #print('**'*100)
                #print('what what')
                #print(key)
                extra_keys_row.append(key)
        extra_keys.append(extra_keys_row)
        for k in extra_keys_row:
            #pass
            G[n].pop(k, None)
    #print(extra_keys)
    
    return G, extra_keys


np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

NLL = 1; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)

#Potential data for (single/multilayered) model Laughlin
V = { 'eps':Veps, 'xiK':2., 'rV(r)': { ('L','L'): {'rV': rvr} }, 'coulomb': { ('L','L'):  {'v':-1., 'xi': xi}} }


#FOR COMPARISON WITH MPOGRAPH
Veps = 1e-3
V = { 'eps':Veps, 'xiK':2., 'GaussianCoulomb': { ('L','L'):  {'v':1., 'xi': xi}} }

#FOR COMPARISON WITH VMK
#COMPARISON WITH VMK SUCCESS
#Veps = 1e-3
#V = {'eps':Veps,'haldane':{('L', 'L'): [0,1,0] }}
#V=

#COMPARISON AGAIN
#V={'eps': 0.001, 'GaussianCoulomb': {('L', 'L'): {'xi': 2, 'v': 1}}}

root_config=[0,1,0]

N=3
model_par = {

	
	'boundary_conditions': ('infinite', N), #for finite replace by periodic here
	'verbose': 2,
	#'layers': [ ('L', l) for l in range(NLL) ],
	'layers':[ ('L', 0)],
	'Lx': 12.,
	'Vs': V,
	'cons_C': 'total',
	'cons_K': False,
	'root_config': root_config,
	'exp_approx': 'slycot',
}
import pickle
with open('Graph2.pkl','rb') as f:
    G_og=pickle.load(f,encoding='latin1')

#Graph:


#G_og, extra_keys=trim_down_the_MPO(G_og)
#trim down the MPO

G_imported=QH_Graph_final.obtain_new_tenpy_MPO_graph(G_og)
#print(G_new)


print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)
#quit()
G=M.MPOgraph
#print(len(G_og))
#print(len(G))
#print(G_og==G)
"""
first_one=G_og[0]
our_one=G[0]
counter=0
for key in first_one.keys():
    for key2 in first_one[key].keys():
        #print(first_one[key][key2]==our_one[key][key2])
        #converted = {k: float(v) if isinstance(v, np.floating) else v for k, v in our_one[key][key2][0].items()}
        #print(converted)
        for i,element in enumerate(first_one[key][key2]):
           

            if np.abs(1-our_one[key][key2][i][1]/element[1])>10**(-3):
                print("£"*100)
                counter+=1
                print(key)
                print(key2)
                print(first_one[key][key2])
                print(our_one[key][key2])
                print("£"*100)
       
        if first_one[key][key2]!=our_one[key][key2]:
            print("£"*100)
            counter+=1
            print(key)
            print(key2)
            print(first_one[key][key2])
            print(our_one[key][key2])
            print("£"*100)
        
	#print(first_one[key]==our_one[key])
print(counter)   
"""
#quit()

G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)

G_new, extra_keys=trim_down_the_MPO(G_new)
#print(G_new[0][])
#{"('Mk', 'AL-6-aL.11', 0)": [('Id', -0.0022936103664083296)], "('Mk', 'AL-6-aL.11', 1)": [('Id', -0.012011554777185388)], "('Mk', 'AL-6-aL.11', 2)": [('Id', 0.006983104212694983)], "('Mk', 'AL-6-aL.11', 3)": [('Id', 0.03226845207125104)], "('Mk', 'AL-6-aL.11', 4)": [('Id', 0.0052054391215796345)], "('Mk', 'AL-6-aL.11', 5)": [('Id', -0.052969495630489076)], "('Mk', 'AL-6-aL.11', 6)": [('Id', 0.004782068458101691)], "('Mk', 'AL-6-aL.11', 7)": [('Id', -0.009226238884621787)], "('Mk', 'AL-6-aL.11', 8)": [('Id', 0.12181012215361176)], "('Mk', 'AL-6-aL.11', 9)": [('Id', 0.02680274923762687)], "('Mk', 'AL-6-aL.11', 10)": [('Id', 0.02613075320426292)], "('Mk', 'AL-6-aL.11', 11)": [('Id', -0.03887269255818529)], "('Mk', 'AL-6-aL.11', 12)": [('Id', -0.011587382233824599)], "('Mk', 'AL-6-aL.11', 13)": [('Id', -0.051890294683347715)], "('Mk', 'AL-6-aL.11', 14)": [('Id', -0.045196435317422325)], "('Mk', 'AL-6-aL.11', 15)": [('Id', 0.00770847110335915)], "('Mk', 'AL-6-aL.11', 16)": [('Id', -0.08796050017973375)], "('Mk', 'AL-6-aL.11', 17)": [('Id', 0.0524504387817054)], "('Mk', 'AL-6-aL.11', 18)": [('Id', -0.22693959239263164)], "('Mk', 'AL-6-aL.11', 19)": [('Id', -0.07287589757499625)], "('Mk', 'AL-6-aL.11', 20)": [('Id', -0.017177801279661543)], "('Mk', 'AL-6-aL.11', 21)": [('Id', 0.11145661506377952)], "('Mk', 'AL-6-aL.11', 22)": [('Id', -0.21779847616078557)], "('Mk', 'AL-6-aL.11', 23)": [('Id', 0.11180765580960449)], "('Mk', 'AL-6-aL.11', 24)": [('Id', 0.0018771109848217421)], "('Mk', 'AL-6-aL.11', 25)": [('Id', 0.23475438624686132)], "('Mk', 'AL-6-aL.11', 26)": [('Id', -0.15502741255614333)], "('Mk', 'AL-6-aL.11', 27)": [('Id', 0.544024526982654)], "('_A', 0, 5)": [('aOp', 4.342318515562255e-07)]}


#print(G_new[0]["('_a', 0, 2)"])
#quit()
#print(G_new)
#quit()
#print(len(G_new))
#quit()
#G_new=[G_new[0],G_new[0],G_new[0]]
#print(len(G_new))

print('LLALALALALLALAL')
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
#spin=QH_MultilayerFermionSite(N=1)
L = len(G_new) #System size for finite case, or unit cell for infinite
sites = [spin] * L 
M = MPOGraph(sites=sites,bc='infinite',max_range=None) #: Initialize MPOGRAPH instance

'''
M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
'''
print(len(G_new))

States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)
print("Ordering states",".."*10)
print(States)
previous_length=len(States[0])+1
Nlayer=1
while len(States[0])!=previous_length:

    previous_length=len(States[0])
    
    not_included_couplings_copy=[]
    for i in range(Nlayer):
        
        for m in not_included_couplings[i]:
        
            not_included_couplings_copy.append(m)
    

    
    #obtain new basis of keys for reduced graph
    G_new=eliminate_elements_from_the_graph(G_new,not_included_couplings_copy)
    G_new, extra_keys=trim_down_the_MPO(G_new)
    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)



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

dmrg_params = {"trunc_params": {"chi_max": 200, "svd_min": 1.e-10}, "mixer": True, "max_sweeps":100}

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