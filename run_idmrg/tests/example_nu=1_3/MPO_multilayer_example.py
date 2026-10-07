#!/mnt/users/dperkovic/myenv/bin/python
#source /mnt/users/dperkovic/myenv/bin/activate
"""Code for running DMRG """
import sys
import os
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 

#sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 

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
from tenpy.models.lattice import Lattice
from tenpy.networks.site import QH_MultilayerFermionSite_final

#from tenpy.networks.site import QH_MultilayerFermionSite_final_2
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy/NEW_COMMITS') 
import QH_G2MPO
import  QH_Graph_final
from tenpy.networks.site import QH_MultilayerFermionSite
print(sys.executable)
print(sys.version)
#quit()
print()

from make_iMPS import load_param
from make_iMPS import load_data
import subprocess

subprocess.run(["pip", "freeze"])
#quit()
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
V = { 'eps':Veps, 'xiK':2., 'rV(r)': { ('L','L'): {'rV': rvr} }, 'coulomb': { ('L','L'):  {'v':1., 'xi': xi}} }


#FOR COMPARISON WITH MPOGRAPH
Veps = 1e-5
V = { 'eps':Veps, 'xiK':2., 'GaussianCoulomb': { ('L','L'):  {'v':1., 'xi': xi}} }



#FOR COMPARISON WITH VMK
#COMPARISON WITH VMK SUCCESS
#Veps = 1e-3
#V = {'eps':Veps,'haldane':{('L', 'L'): [0,1,0] }}
#V=

#COMPARISON AGAIN
#V={'eps': 0.001, 'GaussianCoulomb': {('L', 'L'): {'xi': 2, 'v': 1}}}
Veps = 1e-6
V = {'eps':Veps,'haldane':{('L', 'L'): [0,1.0],('L', 'L'): [0,1.0] }}
root_config=[0,1,0,0,1,0]


import pickle




def make_iMPO(model_par,Na,graph=False):
    print("Start model in Old Tenpy",".."*10)
    M = mod.QH_model(model_par)
    print("Old code finished producing MPO graph",".."*10)
 
    if not graph:
        G=[M.MPOgraph[0],M.MPOgraph[1]]*Na*3
        #G=M.MPOgraph*Na
    else:
        G=[graph[0],graph[1]]*(3*Na)
    Nlayer=2
    print('nlayer:', Nlayer)
    #quit()
    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
    G_new, extra_keys=trim_down_the_MPO(G_new)



    #
    root_config_ = np.array([0,0,1,1,0,0])
    root_config_ = root_config_.reshape(3,2)

   
    L = len(G_new) #System size for finite case, or unit cell for infinite
    print(L)
    
    sites = []
    for i in range(3*Na*Nlayer):
        spin=QH_MultilayerFermionSite_final(N=Nlayer,root_config=root_config_,conserve=('each','K'),site_loc=i)
        #spin=QH_MultilayerFermionSite_final(N=1,root_config=root_config_,conserve='N',site_loc=i)
        sites.append(spin)
  
    
   
   
    M = MPOGraph(sites=sites,bc='infinite',max_range=None) #: Initialize MPOGRAPH instance

    '''
    M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

    Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
    The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
    '''
    #print(len(G_new))

    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)
    print("Ordering states",".."*10)
    #print(States)
    previous_length=len(States[0])+1
   
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
    
   
    print("Building MPO"+".."*10)
    ##############
    
    H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
    print("Built"+".."*10)
  


    """
    for i in range(len(H._W)):
        _,H._W[i]=H._W[i].sort_legcharge(bunch=False)
    """

    #quit()
    #quit()
    #initialize MPOModel

    pos= [[i] for i in range(L)]

    lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="infinite")
    #lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="infinite")
    x=lattice.mps_sites()
    model=MPOModel(lattice, H)
  
    #produces MPO with (N,K) conservation
    return model,sites


Na=1


name_load='DMRG_18.0_Haldane_barrier_0.1_mu_1_3_'

name_load='Data'
L=3
model_par_new,conserve,root_config_,loaded_xxxx=load_param(name_load)
print(model_par_new['Vs'])

#LABELS IN HALDANE DICTATE THE INTERACTION STRENGTHS BETWEEN PARTICLES CALLED UP DOWN
model_par_new['Vs']={'haldane': {('up', 'down'): [0, 0.2, 0], ('down', 'down'): [0, 1.0, 0], ('up', 'up'): [0, 1.0, 0]}, 'eps': 0.0001}


V = { 'eps':Veps, 'xiK':2., 'GaussianCoulomb': { ('up','up'):  {'v':1., 'xi': xi}, ('down','down'):  {'v':1., 'xi': xi}, ('up','down'):  {'v':1., 'xi': xi}} }
model_par_new['Vs']=V
model_par_new['mu']=[1,0]
#MEANS PARTICLE OF TYPE 0 down IN LL 0
#PARTICLE OF TYPE up IS IN LL0
# note this is how we write these things
# layers = [ (mu, a), (mu', a'), . . . (species, band = LL#) ]
model_par_new['layers']=[ ('down', 0),('up', 0)]
print(model_par_new)


#graph=loaded_xxxx['graph'][0]


graph=False
model,sites=make_iMPO(model_par_new,Na,graph)




"""

TODO:
TEST MPO BY COMPARING IT TO KONSTANTINOS THING HE SENDS.
THEN MAKE A TEST OF DMRG WITH K CONSERVATION USING KONSTANTINOS MPO TO SEE IF IT YIELDS
A BETTER GROUND STATE GUESS.



###### ENERGY SEEMS CONSISTENT WITH ONLY 'N' conservation, at least for state=010
#### energy we get makes NO SENSE AT ALL FOR LOADED STATE, AS ENERGY SHOULD BE ZERO. BUT IT
IS NOT. it 0.02


IF WE LOAD STATE WITH MORE UNIT CELLS, WE GET SMALLER ENERGY.
THERE MUST BE SOMETHING GOING WRONG WITH THE ENVIRONMENTS
pRETTY SURE MPO IS GOOD BUT SHOULD CHECK
similar thing happens when i increase the number of sites in DMRG algorithm - energy becomes smaller, and
in cetner sites do have 1/3 filling

SO THE PROBLEM SEEMS TO BE THAT THE CONSERVATION OF MOMENTUM WORKS WITHIN THE CELL ONLY,
AND NOT WITHIN THE WHOLE THING. THAT IS THE SYSTEM DOES NOT ALLOW JUMPS PAST THE UNIT CELL "walls"

"""
#quit()

#pstate=["empty", "full","empty"]*Na




pstate=["empty", "full","empty","empty", "full","empty"]*Na

#pstate=["empty", "empty","full"]*Na
psi = MPS.from_product_state(sites, pstate, bc="infinite")

#psi= load_data(loaded_xxxx,sites,Na)


print(psi._B[0])

print(model.H_MPO._W[0])

#E=model.H_MPO.expectation_value(psi)
#print(E)
#quit()

#quit()
#print(psi.expectation_value('nOp'))



#ENERGY SEEMS TO BE COMPLETELY FINE
#SO PROBABLY ERROR IN UPDATING ENV?
#print(E)
#0.204101  0.2930513 0.0060751 0.6333648 0.7268156 0.6333648 0.0060751 0.2930513 0.204101 ]


mixer={'amplitude': 0.05,
           'decay': 2,
           'disable_after':30}
dmrg_params = {"trunc_params": {"chi_max": 40, "svd_min": 1.e-10}, "mixer": True, "max_sweeps":100, 'mixer_params':mixer}#'chi_list': {0: 5, 10: 10, 20: 40}}
print(dmrg_params)
print("Run DMRG:")

engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)  

E0, psi = engine.run()

quit()
#print(psi.expectation_value("nOp"))



#####
## OK i doubt MPO is construced wrong. ON increasing the unit cell we seem to get 1/3 state
#and energy drops compared to smaller unit cell.
#most probably it is environment construction...
Na=1
model,sites=make_iMPO(model_par,Na,graph)
Bs=[]
Ss=[]
for i in range(3):
    Bs.append(np.transpose(psi._B[i].to_ndarray(),(1,0,2)))
    Ss.append(psi._S[i])
Bs=Bs*Na
Ss=Ss*Na
qflat=psi._B[0].legs[0].to_qflat()
from tenpy.linalg.charges import LegCharge
chargeinfo=sites[0].leg.chinfo
print(chargeinfo)
left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
print('finito')
mps=MPS.from_Bflat(sites,Bs,SVs=Ss, bc='infinite',legL=left_leg)
print(mps.expectation_value('nOp'))

E=model.H_MPO.expectation_value(mps)

print(E)
quit()




#initialize wavefunction as MPS


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
