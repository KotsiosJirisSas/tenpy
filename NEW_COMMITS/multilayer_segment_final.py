#!/usr/bin/env python
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys

print(sys.version)
sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 
sys.path.append('/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy') 
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.xxz_chain import XXZChain
from tenpy.models.tf_ising import TFIChain
import QH_G2MPO
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


from tenpy.models.lattice import Lattice
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

from tenpy.networks.site import QH_MultilayerFermionSite_final
from tenpy.networks.site import QH_MultilayerFermionSite_3
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
import QH_G2MPO
import QH_Graph_final

import h5py
#import hdfdict
from tenpy.tools import hdf5_io
#SAVE THE DATA
#import h5py
#from tenpy.tools import hdf5_io


from multilayer_creating_MPS_generic_interface_minimal import QH_system
import pickle
np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

def trim_down_the_MPO(G):
    extra_keys=[]
    #print(G[0]==G[2])
   
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
    #print(G[2][('a_', np.int64(0), 20)])

    return G, extra_keys

def eliminate_elements_from_the_graph(G,not_included_couplings):
    L=len(G)
    for i in range(L):
       
        for element in not_included_couplings:
            if 1==1:
                try:
                    G[i].pop(element[0],None)
                    #print(element[0])
                except:
                    pass
            
            for key in G[i].keys():
                try:
                    G[i][key].pop(element[0],None)
                except:
                    pass
    return G

def get_old_basis(G_old,basis_old,permutation_old):
    """
    Creates TeNpy2 loaded basis

    G_old: gives the MPOgraph, used to eliminate basis
    basis_old: gives the old basis which is untrimmed and unpermuted before 
    permutation_old: gives the permutation of old basis before and after sorting the charges
    returns:
    ordered_old_basis

    GENERALIZED FOR N LAYERS, BECAUSE EACH BOND HAS DIFFERENT PERMUTATIONS AND BASIS
    """
    #print(G_old[0][1])
    
   
    G_old, extra_keys=trim_down_the_MPO(G_old)


    #print(extra_keys)
    
    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_old,len(G_old))
 
    #for i in range(len(list(States[0]))):
    #    print(list(States[0])[i])
    #print(len(States[0]))
    #quit()
    previous_length=len(States[0])+1
    while len(States[0])!=previous_length:

        previous_length=len(States[0])
        Nlayer=2
        not_included_couplings_copy=[]
        for i in range(Nlayer):
            print(i)
            #print(not_included_couplings[i])
            for m in not_included_couplings[i]:
                print(m)
                not_included_couplings_copy.append(m)
        #quit()
        #print(not_included_couplings_copy)
        #quit()
        G_old=eliminate_elements_from_the_graph(G_old,not_included_couplings_copy)
        G_old, extra_keys=trim_down_the_MPO(G_old)
        States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_old,len(G_old))

  
    #print(not_included_couplings)
    #states should give the correct values on G_old
    States2=[]
    for i in range(len(States)):
        States2.append(QH_G2MPO.basis_map(States[i]))
    States=States2
    print(len(States))
    ordered_old_basis=[[] for i in range(len(basis_old))]
    for i in range(Nlayer):
        removed_total= set(basis_old[i])- set(States[i])
        print(len(States[i]))
        print(len(removed_total))
        #quit()
        #print(States[0])
        print(len(basis_old[i]))

        
        for m in removed_total:
            basis_old[i].remove(m)
        #print(permutation_old)
        #quit()
        #this gives us appropriate basis
        #now calculate:
        #print(basis_old)
        #print(permutation_old)
        #permutation_old=np.arange(62)
        print(len(basis_old[i]))
        #quit()
        print(len(permutation_old[i]))
        a=np.array(basis_old[i])[permutation_old[i]]
        ordered_old_basis[i]=list(np.array(basis_old[i])[permutation_old[i]])
        ordered_old_basis[i]=[str(x) for x in ordered_old_basis[i]]
    
    print(len(ordered_old_basis[0]))
    print(len(ordered_old_basis[1]))
  
    #quit()
    #print( np.array(ordered_old_basis)==np.array(basis_old))
    #quit()
    #print(ordered_old_basis)
    #quit()
    #print(len(ordered_old_basis))
    #quit()
    return ordered_old_basis


def create_segment_DMRG_model(model_par,L,root_config_,conserve,loaded_xxxx,add=False):

    """
    Creates MPOModel given model parameters with L sites on a segment.

    model_par: parameters for creation of MPOModel
    L: int, number of sites on chain
    root_config_: root configuration
    conserve: gives tuple of conserved quantities
   
    returns:
    MPOModel
    Sites - list of sites 
    """


    #SET PARAMETERS
    #TODO: MAKE MODEL PARAMETERS LOAD FROM OLD TENPY2
        




    # construct MPO Model from the Graph using the tenpy2 code
    print("Start model in Old Tenpy",".."*10)
    #M = mod.QH_model(model_par)
    print("Old code finished producing MPO graph",".."*10)

    #quit()
    #load tenpy2 graph into tenpy3 graph
    #G=M.MPOgraph
    G=loaded_xxxx
    #print()
    #G=[loaded_xxxx['graph'][0]]*L
    #print(lenG)
    #quit()

    #quit()
    G, extra_keys=trim_down_the_MPO(G)
    #trim down the MPO
    #print(extra_keys)
   

    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
    #print(len(G_new[0].keys()))
    #quit()
    
    cell=len(root_config_)
    
    print('asserting that the size is compatible with enivornment.....')
    #assert L%cell==0


    #root_config_ = np.array([0,1,0])
  

    #define Hilbert spaces for each site with appropriate conservation laws
    if not add:
        sites=[]
        for i in range(L):
        
            spin=QH_MultilayerFermionSite_final(N=2,root_config=root_config_,conserve=('each','K'),site_loc=i)
            
            sites.append(spin)
    else:
        
        
        sites=[]
        for i in range(-add,L-add):
            #print(i)
            spin=QH_MultilayerFermionSite_final(N=2,root_config=root_config_,conserve=('each','K'),site_loc=i)
           
            sites.append(spin)
           
        
   
    

    M = MPOGraph(sites=sites,bc='segment',max_range=None) #: Initialize MPOGRAPH instance

    '''
    M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

    Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
    The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
    '''
    print(L)
    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)
    
    print("Ordering states",".."*10)
    print(len(States[0]))
    bond=0
  
   
  

    #TODO: IMPLEMENT THIS IN SINGLE LAYER TOO
    previous_length=len(States[0])+1
    while len(States[0])!=previous_length:

        previous_length=len(States[0])
        Nlayer=2
        not_included_couplings_copy=[]
        for i in range(Nlayer):
            print(i)
            #print(not_included_couplings[i])
            for m in not_included_couplings[i]:
                print(m)
                not_included_couplings_copy.append(m)
        #quit()
        #print(not_included_couplings_copy)
        #quit()
        G_new=eliminate_elements_from_the_graph(G_new,not_included_couplings_copy)
        G_new, extra_keys=trim_down_the_MPO(G_new)
        States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)
    
   
    #print(G_new[1]["('a_', 1, 20)"])
    #quit()
    #print(len(States[0]))
    #quit()
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


    #sort leg charges to make DMRG algortihm quicker
    perms2=H.sort_legcharges()
    #print(len(States[0]))
    #quit()
    #print(perms2)
    #print(M.states[0] )
    #quit()
    #orderes the state according to the charges
    #MAKES SOME REDUNDANT COPIES
    ordered_states=[]
    for k in range(len(M._ordered_states)):
        ordered_states_=[]
        for i in range(len(M._ordered_states[k])):
            b=[key for key, value in  M._ordered_states[k].items() if value == i]
            ordered_states_.append(b[0])
        ordered_states.append(ordered_states_)
    #ordered_states=M._ordered_states
    #print(perms2[0])
    #print(len(ordered_states[k]))
    #quit()
    #perms2=np.arange(len())
    
    for k in range(len(M._ordered_states)):
        ordered_states[k]=np.array(ordered_states[k])[perms2[k]]
        ordered_states[k]=list(ordered_states[k])
    
    #quit()
    #Define lattice on which MPO is defined
    pos= [[i] for i in range(L)]
    lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="segment")
    x=lattice.mps_sites()
    
    #create model from lattice and MPO
    model=MPOModel(lattice, H)
    print("created model",".."*30)

    #assert model.H_MPO.is_equal(model.H_MPO.dagger())
    print('asserted')
   
    
    return model,sites, ordered_states

def load_data(loaded_xxxx,sites,shift=0,side='right',charge_shift=[0,0,0]):
    """
    loads MPS as segment mps of length len(sites)
    name: Str, name of the .pkl file from which we import
    sites: list of class:Sites, list of hilbert spaces corresponding to each site
    """
    L=len(sites)
    #finds length of an infinite unit cell
    print(loaded_xxxx.keys())

    if side=='right':
        Bflat0=loaded_xxxx['MPS_Bs'].copy()
        #load singular values
        Ss=loaded_xxxx['MPS_Ss'].copy()

        #load charge infromation
        
        qflat2=loaded_xxxx['MPS_qflat'][0]
    else:
        Bflat0=loaded_xxxx['MPS_Bs'].copy()
        #load singular values
        Ss=loaded_xxxx['MPS_Ss'].copy()
       
        #just need the leftmost leg, not all charges bro
        qflat2=loaded_xxxx['MPS_qflat'][0]
       
    #shift=0
    #change qflat into representation consistent with Tenpy3
    #this is just charges of the leftmost leg
    qflat=[]
    #print(len(qflat2))
    #print(len(qflat2[0]))
    #print(qflat2)
    for i in range(len(qflat2)):

        kopy=[]
        for m in range(len(qflat2[i])):
            if m==0:
                shifted=qflat2[i][0]+shift*qflat2[i][1]+shift*qflat2[i][2]
                kopy.append(shifted)
                #HOW DO YOU SHIFT QFLAT2 ACTUALLY,
                #talk to konstantinos in the morning
            else:
                kopy.append(qflat2[i][m])
        qflat.append(kopy)


    print("shift charges")
    print(charge_shift)
    for i in range(len(qflat)):
        qflat[i][0]+=charge_shift[0]
        qflat[i][1]+=charge_shift[1]
        qflat[i][2]+=charge_shift[2]
    qflat=np.array(qflat)
   
   
    
   
    

        

    infinite_unit_cell=len(Bflat0)

    number=len(sites)//infinite_unit_cell+1
    
    #enlarge the unit cell accordingly,
    #TODO: MAKE THIS PROCESS QUICKER
    Bflat=Bflat0*number


  
  

    #PERMUTE Ss values
    last_ss=Ss[-1]
    Ss.pop(-1)
    Ss.insert(0,last_ss)
    Ss=Ss*number
    #SINCE CONVERTING IT DIRECTLY TO SEGMENT MPS WE NEED TO ADD SINGULAR VALUES FOR THE FINAL RIGHTMOST LEG AS WELL!
    Ss.append(Ss[0])
    

    #cut the mps into mps of wanted size
    Bflat=Bflat[:L]
    Ss=Ss[:L+1]
    
    #define left most leg of charges
    chargeinfo=sites[0].leg.chinfo
    
    left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
    if side=='left':
        print('*'*100)
        print("changed leg")
        print(left_leg)
        print('*'*100)
    #quit()
    #create MPS,
    #charges are calculated from the left leg
    
    print('loading the MPS')
    mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    print('loaded mps from data',".."*30)
   
   
    return mps

def load_environment(loaded_xxxx,location,root_config_,conserve,permute, side='right',old=False,charge_shift=[0,0,0]):

    #TODO: GENERALIZE TO ALL CONSERVATION LAWS SO THAT IT IS LOADED MORE SMOOTHLY
    """
    loads environment on from old tenpy2 code
    name:   Str, the file name in which the environment is saved
    location: Int, sets location of the site at which environment is located (-1 for left and len(sites) for right usually) 
    side: Str, if right loads right side, otherwise loads left side


    TODO: NEED TO ENSURE WE HAVE THE CORRECT NUMBER OF SITE
    """
    print("loading "+side+ " environment",'..'*20)
 
    if side =='right':

        print(loaded_xxxx.keys())
        if old:
            Bflat=loaded_xxxx['RP']
            qflat_list_c=loaded_xxxx['RP_q']
        else:
            Bflat=loaded_xxxx['RP_B_new']
            qflat_list_c=loaded_xxxx['RP_q_new']
       
        #change the shape of Bflat and qflat so that it is consistent with new tenpy
        Bflat=np.transpose(Bflat, (1, 0, 2))

        #permute to be consistent with MPO
        Bflat=Bflat[:,permute,:]
        qflat_list_c[0]=qflat_list_c[0][permute]
        qflat_list=[qflat_list_c[1],qflat_list_c[0],qflat_list_c[2]]
        labels=['vL', 'wL', 'vL*']
        conj_q=[1,1,-1]
        shift=0

        
    else:
        print(loaded_xxxx.keys())
        #quit()
        Bflat=loaded_xxxx['LP']
        qflat_list_c=loaded_xxxx['LP_q']
        #transpose bflat and qflat to make legs consistent with TeNpy3
        Bflat=np.transpose(Bflat, (2, 0, 1))
        #Bflat=np.transpose(Bflat, (1, 0, 2))
        #permute to be consistent with MPO
        Bflat=Bflat[:,permute,:]
        qflat_list_c[0]=qflat_list_c[0][permute]
        #qflat_list=[qflat_list_c[1],qflat_list_c[0],qflat_list_c[2]]
        qflat_list=[qflat_list_c[2],qflat_list_c[0],qflat_list_c[1]]

        labels=['vR*', 'wR', 'vR']
        conj_q=[1,-1,-1]
        shift=2
        
       
   
   
    


    #create site at the end of the chain
    site=QH_MultilayerFermionSite_final(N=2,root_config=root_config_,conserve=('each','K'),site_loc=location)
    chargeinfo=site.leg.chinfo

    #shifts K by num_site-2 to get the correct charge matching in K sector
    #rule is simple. K= \sum_i N_i i, so shift of each K value is just N_i*(num_sites-2)
    #first column in qflat has information on K charges, and second on N charges

    #suma_1=0
    for i in range(len(qflat_list[0])):
        #TELLS US BY HOW MANY MPO UNIT CELLS WE SHIFT
        
        #each charge does the shifting
        #ADD CONSTANT SHIFT ON AUXILIARY LEGS
      

        
        qflat_list[0][i][0]+=qflat_list[0][i][1]*(location-shift)
        qflat_list[0][i][0]+=qflat_list[0][i][2]*(location-shift)
        qflat_list[0][i][0]+= charge_shift[0]
        qflat_list[0][i][1]+= charge_shift[1]
        qflat_list[0][i][2]+= charge_shift[2]
        #suma_1+=qflat_list[0][i][1]

    #suma_2=0
    for i in range(len(qflat_list[1])):
        qflat_list[1][i][0]+=qflat_list[1][i][1]*(location-shift)
        qflat_list[1][i][0]+=qflat_list[1][i][2]*(location-shift)
        #suma_2+=qflat_list[1][i][1]
    
    #suma_3=0
    for i in range(len(qflat_list[2])):

        #ADD CONSTANT SHIFT ON AUXILIARY LEGS
      

        qflat_list[2][i][0]+=qflat_list[2][i][1]*(location-shift)
        qflat_list[2][i][0]+=qflat_list[2][i][2]*(location-shift)
        
        #ADD CHARGE SHIFT
        qflat_list[2][i][0]+= charge_shift[0]
        qflat_list[2][i][1]+= charge_shift[1]
        qflat_list[2][i][2]+= charge_shift[2]
        #suma_3+=qflat_list[2][i][1]
    
    legcharges=[]
    #creates all three legs of MPO
    for i in range(len(qflat_list)):
        legcharge=LegCharge.from_qflat(chargeinfo,qflat_list[i],qconj=conj_q[i]).bunch()[1]
        legcharges.append(legcharge)


    
    #creates MPO
    environment=Array.from_ndarray( Bflat,
                        legcharges,
                        dtype=np.float64,
                        qtotal=None,
                        cutoff=None,
                        labels=labels,
                        raise_wrong_sector=True,
                        warn_wrong_sector=True)
    print(environment)
    print("environment is loaded",'..'*20)
    x=environment.qtotal
    print(x)
    
    return environment



def load_param(name):

    with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/multilayer_with_interlayer_interaction_Haldane/"+name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    #IF LOADED FROM H5 FILE UNCOMMENT

    #loaded_xxxx=hdfdict.load("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/"+name+'.h5')
    
    print(loaded_xxxx.keys())
    #print(loaded_xxxx['graph'][0])
    #quit()
    #keys=[ 'exp_approx',  'cons_K',  'MPS_Ss', 'cons_C', 'Lx', 'root_config', 'LL', 'Vs']
    #model_par={}
    #for k in keys:
    #    model_par[k]=loaded_xxxx[k]
    #model_par = loaded_xxxx['Model']
    model_par = loaded_xxxx['Model']
    #print(model_par)
    #
    #model_par['root_config']=np.array([1,0,0,1])
    root_config_ = model_par['root_config'].reshape(3,2)

    conserve=[]
    """
    if  model_par['cons_C']=='total':
        conserve.append('N')

    if model_par['cons_K']:
        conserve.append('K')
    """
    conserve=['N','K']
    #load root_configuration and conservation laws
    if len(conserve)==1:
        conserve=conserve[0]
    else:
        conserve=tuple(conserve)
    return model_par,conserve,root_config_,loaded_xxxx



def load_environments_from_file(name,name_load,side='right'):
    file_path="/mnt/users/dperkovic/quantum_hall_dmrg/data_load/multilayer_with_interlayer_interaction_Haldane/"+name+'.npz'
    data =np.load(file_path,allow_pickle=True)
    
    file_path="/mnt/users/dperkovic/quantum_hall_dmrg/data_load/multilayer_with_interlayer_interaction_Haldane/"+name_load+'.pkl'
    with open(file_path, 'rb') as f:
        loaded_qs= pickle.load(f, encoding='latin1')
    print(loaded_qs.keys())
    if side=='right':
        dictionary={}
        dictionary['RP']=data['RP']
        #print(data['RP_q'])

        
        dictionary['RP_q']=loaded_qs['RP_q']
        
        
    else:
        dictionary={}
        dictionary['LP']=data['LP']
        dictionary['LP_q']=loaded_qs['LP_q']
   
    return dictionary

def find_permutation(source, target):
    return [source.index(x) for x in target]

def load_permutation_of_basis(loaded_xxxx,ordered_states,new_MPO):
    #TODO: COMMENT OUT
    #GENERALIZED FOR N LAYERS, BECAUSE EACH BOND HAS DIFFERENT PERMUTATIONS AND BASIS
    print(loaded_xxxx.keys())
    #quit()
    #print(len(loaded_xxxx['indices']))
   
    #quit()
    G_old,basis_old,permutation_old=loaded_xxxx['graph'],loaded_xxxx['indices'][::-1],loaded_xxxx['permutations'][:2][::-1]
    #print(len(basis_old))
    #print(basis_old)
    #quit()
    #print(G_old[0].keys())
    #print(basis_old)
    #quit()
    
    #print(basis_old)
    #quit()
    #quit()
    
    old_basis=get_old_basis(G_old,basis_old,permutation_old)
    for i in range(len(old_basis)):
        assert len(old_basis[i])==len(ordered_states[i])
    #print(len(basis_old))
    #print(len(old_basis))
    #total_permutation=[]
    #print()
    #quit()
    #print(old_basis)
    #print(old_basis)
    #print(ordered_states[0])
    #quit()
    #print(set(old_basis)-set(ordered_states[0]))
    #print(set(ordered_states[0])-set(old_basis))
    #quit()
    permutation=[]
    for i in range(len(old_basis)):
        permutation.append(find_permutation(old_basis[i],ordered_states[i]))
    
    for i in range(len(old_basis)):
        #asserts two bases are the same
        assert np.all(ordered_states[i]==np.array(old_basis[i])[permutation[i]])
    #quit()
    #print(ordered_states[0]==np.array(old_basis)[permutation])
    #quit()
    #print(loaded_xxxx.keys())
    #quit()
    sanity_check_permutation(loaded_xxxx,new_MPO, permutation[0],permutation[1])
    #GIVES THE CORRECT PERMUTATION of old basis to get a new basis!!
    return permutation
def sanity_check_permutation(loaded_xxxx,new_MPO, permute,permute2):

    print('checking sanity check of permutation...')
    #permute old Bflat and check its the same as new Bflat for hamiltonian
    print(loaded_xxxx.keys())
   
    Bflat_old=loaded_xxxx['MPO_B'][0]
    #print(Bflat_old.shape)

    #print(Bflat_old.shape)
    #DIFFERENT PERMUTATION SINCE DIFFERENT BASIS ON DIFFERENT LEGS FOR NON TRIVIAL LENGTH OF MPO
    Bflat_old=Bflat_old[permute,:,:,:]
    Bflat_old=Bflat_old[:,permute2,:,:]
    
    """
    permute=np.arange(len(Bflat_old))
    #permute[0]=0
    permute[1]=2
    permute[2]=1
    Bflat_old=Bflat_old[permute,:,:,:]
    Bflat_old=Bflat_old[:,permute,:,:]

    permute=np.arange(len(Bflat_old))
    #permute[0]=0
    permute[1]=3
    permute[3]=1
    Bflat_old=Bflat_old[permute,:,:,:]
    Bflat_old=Bflat_old[:,permute,:,:]
    """
    B_new=new_MPO.to_ndarray()
    print(B_new.shape)
    #quit()
    er=np.sum(( Bflat_old-B_new)**2)
    er2=np.sum(( Bflat_old+B_new)**2)
    print(er)
    print(er2)
    thresh=10**(-8)
    
    #consistent permutation but need to check differently
    #TODO: better check of chemical potential term!
    counter=0
    for i in range(len(Bflat_old)):
        #one element can differ, ie chemical potential
        
        for j in range(len(Bflat_old)):
            crit=np.sum((Bflat_old[j,i]-B_new[j,i])**2)
            if crit>0.00001:
                counter+=1
                #print('x'*50)
                #print(j,i)
                #print(Bflat_old[j,i])
            
                #print(B_new[j,i])
                #remove chemical potential difference
                if counter==1:
                    er-=crit
    print(er/er2)
    if er/er2<thresh:
        print('permutation is consistent')
    else:
        raise ValueError("inconsistent permutation") 

    return None

def load_graph(name):

    with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/multilayer_with_interlayer_interaction_Haldane/"+name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    return loaded_xxxx
    


def bulk_bulk_boundary(name_load,name_save,shift):
    model_par,conserve,root_config_,loaded_xxxx=load_param(name_load)
    
    
    #TODO: COMMENT OUT STUFF
   
    #graph is fine
    
    print(loaded_xxxx['MPO_B'][0].shape)
  
    graph=loaded_xxxx['graph']
    
    #quit()
    print(len(graph))
   
    L=199

    assert (2*L)%6==2
    half=(L//6)*6

    graph=[graph[0],graph[1]]*L
    #graph= load_graph(name_graph)
    L=len(graph)

   

    #length of the system is L,
    #half is L//2
 

   

    model_par['boundary_conditions']= ('infinite', L)
    print('__'*100)
    print(model_par)

   

    LL=0
    model_par['layers']=[ ('L', LL) ]
    
    
    #INSERT UNIT CELL IN BETWEEN!
    M,sites,ordered_states=create_segment_DMRG_model(model_par,L,root_config_,conserve,graph,add=False)


    perm=load_permutation_of_basis(loaded_xxxx,ordered_states,M.H_MPO._W[0])
    
  
     
    ###############################################################
    #construct the wavefunction with merged interface
    params = {}
    params['verbose'] = 1
    params['sys length'] = L
    params['N_layer']=2
    params['half']=half
    params['avg_filling']=2/3
    params['unit cell'] = 3#in the case of pf-apf
    params['sites added'] = added_sites#sides added in the middle
  
    params['model data file'] = '/mnt/users/dperkovic/quantum_hall_dmrg/data_load/multilayer_with_interlayer_interaction_Haldane/'+name_load+'.pkl'
    QHsys = QH_system(params=params)
    QHsys.shift_import=shift
    QHsys.patch_WFs()
    psi_total=QHsys.psi_merged
    ###############################################################
    print(psi_total)
    #quit()
    filling= psi_total.expectation_value("nOp")
    K=np.sum(np.arange(len(filling))//2*(filling-1/3))
    diff=np.sum((filling-1/3))

    print('#'*100)
    print('Initial guess, number of particles and momentum sector')
    print('N='+str(diff))
    print('K='+str(K))
    print('#'*100)
    
    print("loading right env")
    name='Environment_R(2_3,0)'
    name_load_q2='Envs_qs(2_3,0)'
    data=load_environments_from_file(name,name_load_q2,side='right')

    #put a charge shift by hand here??
    #MAKE SURE YOU USE RIGHT PERMUTATION
    #need to shift charges due to differing conventions for charges on 2/3,0 and 1/3,1/3 
    #this convention assumes that 2/3,0 is on LHS and 1/3,1/3 on RHS
    

    
    #if it is the other way around then shifara=0
    right_env=load_environment(data,(len(sites))//2-1,root_config_,conserve, perm[0],side='right',old=True,charge_shift=[0,0,0])
    print('right_env'*100)
    #print(right_env)

    leg_MPS=psi_total._B[-1].get_leg('vR').to_qflat()[0]
    #leg_MPO
    leg_env=right_env.get_leg('vL').to_qflat()[0]
    shift_final=leg_MPS-leg_env
    print('print out the shift')
    print(shift_final)
    print('right_env'*100)

    name='Environment_R(2_3,0)'
    name_load_q2='Envs_qs(2_3,0)'
    data=load_environments_from_file(name,name_load_q2,side='right')

    #
    right_env=load_environment(data,(len(sites))//2-1,root_config_,conserve, perm[0],side='right',old=True,charge_shift=shift_final)
    #quit()

    #leg_MPO=M.H_MPO._W[-1].get_leg('wL')
    leg_MPO=M.H_MPO._W[-1]#.get_leg('wR')
    #leg_MPO=M.H_MPO._W[-1]#.get_leg('wL')
    print('MPO match')
    print(leg_MPO)
    print(right_env)
    print(psi_total._B[-1])
    print(psi_total._B[-2])
    print(psi_total._B[-3])
    print(psi_total._B[-4])
    #quit()
   
   



    #leg_MPS
    #leg_MPS=psi_total._B[0].get_leg('vL')
    #leg_MPO
    #
 
    
    name='Environment_L(1_3,1_3)'
    name_load_q2='Envs_qs(1_3,1_3)'
    data=load_environments_from_file(name,name_load_q2,side='left')
    left_environment=load_environment(data,-1,root_config_,conserve, perm[0],side='left',old=True,charge_shift=[0,0,0])
    #print(M.H_MPO._W[0])
    print('psi'*100)
    
    #print(psi_halfinf_left._B[1])
    #print(psi_halfinf_left._B[2])
    #print(psi_halfinf_left._B[3])
    #print(psi_halfinf_left._B[4])
    #print(psi_halfinf_left._B[5])
    #print(psi_total._B[0])
    print('leftenv'*100)
    print(left_environment)
   
    print(psi_total._B[0])
    #quit()

    init_env_data_halfinf={}


    init_env_data_halfinf['init_LP'] = left_environment    #DEFINE LEFT ENVIROMENT
    init_env_data_halfinf['age_LP'] =0

    init_env_data_halfinf['init_RP'] = right_env   #DEFINE RIGHT ENVIROMENT
    init_env_data_halfinf['age_RP'] =0



    
    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-7,
        'max_S_err': 1.e-5,
        'trunc_params': {
            'chi_max': 3500,
            'svd_min': 1.e-7,
        },
        'max_sweeps': 50}

    eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_total, M, dmrg_params,
                                        resume_data={'init_env_data': init_env_data_halfinf})
    #print(eng_halfinf.chi_max)
    #
    print("enviroment works")
    print("running DMRG")
    #
    #print("MPS qtotal:", M.qtotal)
    #print("MPO qtotal:", psi_halfinf.qtotal)
    E0, psi=eng_halfinf.run()



    #calculate and store data


    filling=psi.expectation_value("nOp")

    print('Filling:',filling)


    E_spec=psi.entanglement_spectrum(by_charge=True)
    #print('entanglement spectrum:',E_spec)



    EE=psi.entanglement_entropy()
    print('entanglement entropy:',EE)


    

    data = { "dmrg_params":dmrg_params,"energy":E0, "model_par":model_par,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

  

   
    with h5py.File("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/multilayer_with_interlayer_interaction/"+name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)




"""
name_load2='Data(1_3,1_3)'
name_load='Data(2_3,0)'
model_par,conserve,root_config_,loaded_xxxx=load_param(name_load2)
model_par2,conserve,root_config_,loaded_xxxx2=load_param(name_load)

loaded={}
print(loaded_xxxx.keys())
loaded['Model']=loaded_xxxx['Model']
loaded['root_config']=model_par['root_config']
loaded['graph']=loaded_xxxx['graph']
loaded['MPO_B']=loaded_xxxx['MPO_B']
loaded['indices'] = loaded_xxxx['indices']
loaded['permutations'] = loaded_xxxx['permutations']

loaded['MPS_Ss']=loaded_xxxx['MPS_Ss']
loaded['MPS_qflat']=loaded_xxxx['MPS_qflat']
loaded['MPS_Bs']=loaded_xxxx['MPS_Bs']

loaded['MPS_2_Ss']=loaded_xxxx2['MPS_Ss']
loaded['MPS_2_qflat']=loaded_xxxx2['MPS_qflat']
loaded['MPS_2_Bs']=loaded_xxxx2['MPS_Bs']
print('done')

with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/multilayer_with_interlayer_interaction_Haldane/Data.pkl", "wb") as f:
    pickle.dump(loaded, f)

quit()
"""

added_sites=int(sys.argv[1])
shift1=int(sys.argv[2])
shift2=int(sys.argv[3])
shift3=int(sys.argv[4])
shift=np.array([shift1,shift2,shift3])

#name_load='Data(1_3,1_3)'
#name_load='Data(2_3,0)'
name_save="Final_method_Haldane_Ly=18_L=398_shift="+str(shift)+"_num_sites="+str(added_sites)
name_load='Data'




print("#"*100)
print("Start the calculation")
print(name_save)
print("#"*100)
bulk_bulk_boundary(name_load,name_save,shift)

