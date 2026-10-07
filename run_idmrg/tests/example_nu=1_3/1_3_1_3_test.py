#!/mnt/users/dperkovic/myenv
#source /mnt/users/dperkovic/myenv/bin/activate
"""Code for running DMRG """
import sys
import os

import pickle
import sys


sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 



import h5py
from tenpy.tools import hdf5_io

import numpy as np
from tenpy.linalg import np_conserved as npc
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
import QH_G2MPO
import  QH_Graph_final
from tenpy.networks.site import QH_MultilayerFermionSite
print(sys.executable)
print(sys.version)

import subprocess
import tenpy

#logger is important here, prints INFO for DMRG
tenpy.tools.misc.setup_logging(to_stdout="INFO")



subprocess.run(["pip", "freeze"])
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

def make_iMPO(load_graph,model_par,num_sites):

    ###############
    ######## produces the MPO graph and MPOModel for iDMRG given model_par
    #############
    print("Import graph",".."*10)
    with open(load_graph,'rb') as f:
        G=pickle.load(f)
    print("MPO graph imported from a file",".."*10)
    
    Nlayer=len(model_par['layers'])
    G=G*(num_sites//Nlayer)
    print('nlayer:', Nlayer)

    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
    G_new, extra_keys=trim_down_the_MPO(G_new)



    #
    root_config_ = model_par['root_config']
    root_config_ = root_config_.reshape(num_sites//Nlayer,Nlayer)
    print('root configuration:')
    print(root_config_)
  
    L = len(G_new) #System size for finite case, or unit cell for infinite
    print('total length of the unit cell')
    print(L)
    
    sites = []
    for i in range(num_sites):
        if model_par['cons_K']:
            if model_par['cons_C']=='total':
                conserve=('N','K')
            else:
                conserve=(model_par['cons_C'],'K')
        else:
            conserve=model_par['cons_C']
       
        spin=QH_MultilayerFermionSite_final(N=Nlayer,root_config=root_config_,conserve=conserve,site_loc=i)
        #print(spin.leg)
       
        sites.append(spin)
   
   
   
   
    M = MPOGraph(sites=sites,bc='infinite',max_range=None) #: Initialize MPOGRAPH instance

    '''
    M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

    Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
    The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
    '''


    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)
    print("Ordering states",".."*10)
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
  


    
    #initialize MPOModel

    pos= [[i] for i in range(L)]
    lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="infinite")
    x=lattice.mps_sites()
    model=MPOModel(lattice, H)
  
    #produces MPO with (N,K) conservation
    return model,sites




#name_load='DMRG_18.0_Haldane_barrier_0.1_mu_1_3_'

#name_load='Data'
#L=3
#model_par_new,conserve,root_config_,loaded_xxxx=load_param(name_load)
#print('PARAMETRI DRAGI')
#print(model_par_new)


#MEANS PARTICLE OF TYPE 0 down IN LL 0
#PARTICLE OF TYPE up IS IN LL0
# note this is how we write these things
# layers = [ (mu, a), (mu', a'), . . . (species, band = LL#) ]

#chemical potential
Lx=float(sys.argv[1])
num_sites=int(sys.argv[2])



print('1/3 1/3 BILAYER TESTARA HALDANE')



layers=[ ('down', 0),('up',0)]
xi=5


root_config_=np.array([0,0,1,1,0,0])
Veps = 1e-5
#isotropic Nlayer interaction
V = { 'eps':Veps, 'xiK':2}
"""
V['GaussianCoulomb']={}
for layer in layers:
    for layer2 in layers:
        key=(layer[0],layer2[0])
        V['GaussianCoulomb'][key]={'v':1., 'xi': xi}
"""

#NEED TO FIX THIS DOUBLE COUNTING OF INTERACTIONS
V['haldane']={}
for layer in layers:
   
    key=(layer[0],layer[0])
    print(key)
    V['haldane'][key]=[0,1.]
V['haldane'][('up','down')]=[0,1.]
model_par={ #for finite replace by periodic here
	'verbose': 2,
    'root_config': root_config_,
            'exp_approx': 'slycot', 
            'Vs': V, 
            'cons_K': True, 
            'cons_C': 'each', #or total
            'Lx': Lx}

model_par['layers']=layers




name='haldane_1_3_1_3_graph_Lx='+str(Lx)

path='/mnt/users/dperkovic/DMRG_modified/data_load/haldane_1_3_1_3/'
load_graph=path+name+'.pkl'
#num_sites=len(model_par['root_config'])
model,sites=make_iMPO(load_graph,model_par,num_sites)






pstate=["empty","empty","full","full", "empty","empty"]*(num_sites//(len(root_config_)))
psi = MPS.from_product_state(sites, pstate, bc="infinite")




mixer={'amplitude': 0.05,'decay': 2, 'disable_after':30}
chi_max=2000
dmrg_params = {
    'max_E_err': 1.e-9,
        'max_S_err': 1.e-7,
    "trunc_params": {"chi_max": chi_max, "svd_min": 1.e-8}, 
    "mixer": True, "max_sweeps":120, 
      'chi_list': {0: 100, 5: 500, 10: 1000, 20: chi_max}}

print(dmrg_params)
print("Run DMRG:")

engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)  

E0, psi = engine.run()





print("E =", E0)
print("Finished running DMRG")




Length=psi.correlation_length()
print('correlation length:',Length)



filling=psi.expectation_value("nOp")

print('Filling:',filling)


E_spec=psi.entanglement_spectrum(by_charge=True)
print('entanglement spectrum:',E_spec)



EE=psi.entanglement_entropy()
print('entanglement entropy:',EE)



data = {"Lx":Lx,"energy":E0,"correlation_length": Length, "entanglement_entropy": EE, "density":filling, "entanglement_spectrum": E_spec,"model_par":model_par}


path="/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/1_3_1_3_decouple/"

######################
##### Store data from the MPS
############
name="Data_Lx="+str(Lx)+'_num_sites='+str(num_sites)
with h5py.File(path+name+".h5", 'w') as f:
    hdf5_io.save_to_hdf5(f, data)
##########
########
#######


######################
##### Store MPS separately
############
B=[]
for i in range(len(psi._B)):
    B.append(psi._B[i].to_ndarray())
leg=psi._B[0].legs[0].to_qflat()
data = {"B_flat": B,"qflat_left":leg , # e.g. an MPS
        "dmrg_params":dmrg_params, "model_par":model_par}#, "model": model }

name="MPS_Data_Lx="+str(Lx)+'_num_sites='+str(num_sites)
with h5py.File(path+name+".h5", 'w') as f:
    hdf5_io.save_to_hdf5(f, data)

