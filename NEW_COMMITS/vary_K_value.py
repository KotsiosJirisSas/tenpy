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

from tenpy.networks.site import QH_MultilayerFermionSite_2
from tenpy.networks.site import QH_MultilayerFermionSite_3
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
import QH_G2MPO
import QH_Graph_final

import h5py
from tenpy.tools import hdf5_io
#SAVE THE DATA
#import h5py
#from tenpy.tools import hdf5_io
import pickle
np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

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

def eliminate_elements_from_the_graph(G,not_included_couplings):
    L=len(G)
    for i in range(L):
       
        for element in not_included_couplings[i]:
            if element[1]=='row':
                G[i].pop(element[0],None)
                #print(element[0])
            else:
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

    """
    #print(G_old[0][1])
   
    G_old, extra_keys=trim_down_the_MPO(G_old)
    #print(extra_keys)
    
    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_old,len(G_old))
    #for i in range(len(list(States[0]))):
    #    print(list(States[0])[i])
    #print(len(States[0]))
    
   
    #print(not_included_couplings)
    #print(len(States[0]))
    #states should give the correct values on G_old
    States=QH_G2MPO.basis_map(States[0])
    #print(States)
    #print(basis_old)
    removed_total= set(basis_old)- set(States)
    #print(removed_total)
    #£()
    #print(States)
    for m in removed_total:
        basis_old.remove(m)
    
    #print(permutation_old)
    
    #this gives us appropriate basis
    #now calculate:
    #print(basis_old)
    #print(permutation_old)
    #permutation_old=np.arange(62)
    #print(basis_old)
    
    ordered_old_basis=list(np.array(basis_old)[permutation_old])
    ordered_old_basis=[str(x) for x in ordered_old_basis]

    #print(ordered_old_basis[0])
    #print(ordered_old_basis[1])
    #print(ordered_old_basis[2])
    #print(ordered_old_basis[3])
    #print(ordered_old_basis[4])
    
    #print( np.array(ordered_old_basis)==np.array(basis_old))
    
    #print(ordered_old_basis)
    
    #print(len(ordered_old_basis))
    
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

    
    #load tenpy2 graph into tenpy3 graph
    #G=M.MPOgraph
    G=loaded_xxxx
    #print()
    #G=[loaded_xxxx['graph'][0]]*L
    #print(lenG)
    
    G, extra_keys=trim_down_the_MPO(G)
    #trim down the MPO
  

    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
    #print(len(G_new[0].keys()))
    
    
    cell=len(root_config_)

    print('asserting that the size is compatible with enivornment.....')
    #assert L%cell==cell-1
    #root_config_ = np.array([0,1,0])
  
    conserve=('N','K')
    #define Hilbert spaces for each site with appropriate conservation laws
    if not add:
        sites=[]
        for i in range(L):
        
            spin=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=conserve,site_loc=i)
            #print(spin.Id)
            
            sites.append(spin)
    else:
        #print('in')
        
        sites=[]
        for i in range(-add,L-add):
            print(i)
            spin=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=conserve,site_loc=i)
           
            sites.append(spin)
      


    M = MPOGraph(sites=sites,bc='segment',max_range=None) #: Initialize MPOGRAPH instance

    '''
    M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

    Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
    The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
    '''

    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)
    
    print("Ordering states",".."*10)

    M.states = States #: Initialize aux. states in model
    M._ordered_states = QH_G2MPO.set_ordered_states(States) #: sort these states(assign an index to each one)
    print("Finished",".."*10 )
    #print(M.states[0] )
  
    #print(not_included_couplings)
    G_new=eliminate_elements_from_the_graph(G_new,not_included_couplings)
    
    


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
    
    #print(perms2)
    #print(M.states[0] )
    
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
    
    #perms2=np.arange(len())
    
    for k in range(len(M._ordered_states)):
        ordered_states[k]=np.array(ordered_states[k])[perms2[k]]
        ordered_states[k]=list(ordered_states[k])
    
    
    #Define lattice on which MPO is defined
    pos= [[i] for i in range(L)]
    lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="segment")
    x=lattice.mps_sites()
    
    #create model from lattice and MPO
    model=MPOModel(lattice, H)
    print("created model",".."*30)

    assert model.H_MPO.is_equal(model.H_MPO.dagger())
    print('asserted')
   
    
    return model,sites, ordered_states

def load_data(loaded_xxxx,sites,shift=0,charge_shift=[0,0]):
    """
    loads MPS as segment mps of length len(sites)
    name: Str, name of the .pkl file from which we import
    sites: list of class:Sites, list of hilbert spaces corresponding to each site
    """
    L2=len(sites)
    print(L2)
    #finds length of an infinite unit cell
    Bflat0=loaded_xxxx['MPS_Bs']
    #load singular values
    Ss=loaded_xxxx['MPS_Ss']
    #print(loaded_xxxx.keys())
    
    #print(Bflat0[3].shape)
    #load charge infromation
    try:
        qflat2=loaded_xxxx['MPS_qflat']
        print(qflat2.shape)
    except:
        qflat2=loaded_xxxx['MPS_qflat'][0]
    #print(qflat2[0].shape)
    #print(len(qflat2[0][0][0]))
    #quit()
    
    #print(qflat2)
    

        

    infinite_unit_cell=len(Bflat0)

    number=len(sites)//infinite_unit_cell+1
    
    #enlarge the unit cell accordingly,
    #TODO: MAKE THIS PROCESS QUICKER
    Bflat=Bflat0*number


    #change qflat into representation consistent with Tenpy3
    #this is just charges of the leftmost leg
    qflat=[]
    for i in range(len(qflat2)):
        kopy=[]
        for m in range(len(qflat2[i])):
            if m==0:
                kopy.append(qflat2[i][m]+shift*qflat2[i][1])
            else:
                kopy.append(qflat2[i][m])
        qflat.append(kopy)
    
    qflat=np.array(qflat)
    print("smece JEDNO")
    for i in range(len(qflat)):
        qflat[i][0]+=charge_shift[0]
        qflat[i][1]+=charge_shift[1]
    #PERMUTE Ss values
    last_ss=Ss[-1]
    Ss.pop(-1)
    Ss.insert(0,last_ss)
    Ss=Ss*number
    #SINCE CONVERTING IT DIRECTLY TO SEGMENT MPS WE NEED TO ADD SINGULAR VALUES FOR THE FINAL RIGHTMOST LEG AS WELL!
    Ss.append(Ss[0])
    

    #cut the mps into mps of wanted size
    Bflat=Bflat[:L2]
    Ss=Ss[:L2+1]
    print(len(Ss))
    print(len(Bflat))
    #define left most leg of charges
    chargeinfo=sites[0].leg.chinfo
    
    left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
    print(kopy)
    #create MPS,
    #charges are calculated from the left leg
    mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    print('loaded mps from data',".."*30)
    return mps


def project_left_side_of_mps( psi_halfinf):

    #project the left edge of MPS to a single schmidt value
    #makes convergence on boundary with vacuum quicker
    print('projecting MPS',".."*30)
    print(psi_halfinf.segment_boundaries)
    #delete environments so that they get reset after the projection
    psi_halfinf.segment_boundaries=(None,None)
    S =  psi_halfinf.get_SL(0)
    proj = np.zeros(len(S), bool)
    proj[np.argmax(S)] = True


    leg=psi_halfinf._B[0].get_leg('vL')
    c=leg.get_qindex(np.argmax(S))
    charge=leg.charges[c[0]]

    B = psi_halfinf.get_B(0, form='B')
    B.iproject(proj, 'vL')
    psi_halfinf.set_B(0, B, form='B')
    psi_halfinf.set_SL(0, np.ones(1, float))
    psi_halfinf.canonical_form_finite(cutoff=0.0)
    psi_halfinf.test_sanity()
    

    print('projected MPS',".."*30)
 
    return psi_halfinf,charge


def load_environment(loaded_xxxx,location,root_config_,conserve,permute, side='right',old=False):

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
            Bflat=loaded_xxxx['RP_B']
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
        shift=2
    else:
        print(loaded_xxxx.keys())
        
        Bflat=loaded_xxxx['LP1_B2']
        qflat_list_c=loaded_xxxx['LP2_q']
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
    site=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=conserve,site_loc=location)
    chargeinfo=site.leg.chinfo

    #shifts K by num_site-2 to get the correct charge matching in K sector
    #rule is simple. K= \sum_i N_i i, so shift of each K value is just N_i*(num_sites-2)
    #first column in qflat has information on K charges, and second on N charges

    #suma_1=0
    for i in range(len(qflat_list[0])):
        qflat_list[0][i][0]+=qflat_list[0][i][1]*(location-shift)
        #suma_1+=qflat_list[0][i][1]

    #suma_2=0
    for i in range(len(qflat_list[1])):
        qflat_list[1][i][0]+=qflat_list[1][i][1]*(location-shift)
        #suma_2+=qflat_list[1][i][1]
    
    #suma_3=0
    for i in range(len(qflat_list[2])):
        qflat_list[2][i][0]+=qflat_list[2][i][1]*(location-shift)
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
    #print(environment)
    print("environment is loaded",'..'*20)
    x=environment.qtotal
    #print(x)
    
    return environment




def set_left_environment_to_vacuum(leg_HMPO,leg_MPS):
  
    """
    produces left environment which corresponds to vacuum
    so far works only it total charge is zero
    leg_HMPO,leg_MPS: give the legs of MPO and MPS
    """
    duljina_MPS=len(leg_MPS.to_qflat())
    duljina=len(leg_HMPO.to_qflat())


    labels=['vR*', 'wR', 'vR']


    
    #gives qflat for non-trivial leg    
    legcharges=[]
    legcharges.append(leg_MPS)
    legcharges.append(leg_HMPO.conj())
    legcharges.append(leg_MPS.conj())
  

    
    #set data flat
    Bflat=np.zeros(duljina*duljina_MPS*duljina_MPS)
    
    Bflat=np.reshape(Bflat,(duljina_MPS,duljina,duljina_MPS))
    
    #find the first index at which charges are zero?
    qflat=leg_HMPO.conj().to_qflat()

    for i,m in enumerate(qflat):
        find_zero=np.all(m==0)
        if find_zero:
            index=i
            break;
            
    
    a,b,c=Bflat.shape
    Bflat=0*Bflat
    for i in range(a):
        Bflat[i,index,i]=1
  
  
    
    #define left environment
    environment=Array.from_ndarray( Bflat,
                        legcharges,
                        dtype=np.float64,
                        qtotal=None,
                        cutoff=None,
                        labels=labels,
                        raise_wrong_sector=True,
                        warn_wrong_sector=True)
    #print(environment)
    print("vacuum left environment is loaded",'..'*20)
    #print(environment.qtotal)
    
    return environment

def load_param(name):
    print(name)
    with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/2_3_single_layer/"+name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')

    print(loaded_xxxx.keys())
    #print(loaded_xxxx['graph'][0])
    
    #keys=[ 'exp_approx',  'cons_K',  'MPS_Ss', 'cons_C', 'Lx', 'root_config', 'LL', 'Vs']
    #model_par={}
    #for k in keys:
    #    model_par[k]=loaded_xxxx[k]
    model_par = loaded_xxxx['Model']
    #print(model_par)
    print(model_par['root_config'])
    #quit()
    root_config_ = model_par['root_config'].reshape(len(model_par['root_config']),1)

    conserve=[]
    if  model_par['cons_C']=='total':
        conserve.append('N')

    if model_par['cons_K']:
        conserve.append('K')
    #load root_configuration and conservation laws
    if len(conserve)==1:
        conserve=conserve[0]
    else:
        conserve=tuple(conserve)
    return model_par,conserve,root_config_,loaded_xxxx

def find_permutation(source, target):
    return [source.index(x) for x in target]

def load_permutation_of_basis(loaded_xxxx,ordered_states,new_MPO):
    print(loaded_xxxx.keys())
    
    G_old,basis_old,permutation_old=loaded_xxxx['graph'],loaded_xxxx['indices'][0],loaded_xxxx['permutations'] 
    #print(G_old[0].keys())
    #print(basis_old)
    
    
    #print(basis_old)
    
    
    
    old_basis=get_old_basis(G_old,basis_old,permutation_old)
    assert len(old_basis)==len(ordered_states[0])
  
    
    permutation=find_permutation(old_basis,ordered_states[0])
   
    #asserts two bases are the same
    assert np.all(ordered_states[0]==np.array(old_basis)[permutation])
    
 
    
    sanity_check_permutation(loaded_xxxx,new_MPO, permutation)
    #GIVES THE CORRECT PERMUTATION of old basis to get a new basis!!
    return permutation
def sanity_check_permutation(loaded_xxxx,new_MPO, permute):

    print('checking sanity check of permutation...')
    #permute old Bflat and check its the same as new Bflat for hamiltonian
    print(loaded_xxxx.keys())
    try:
        Bflat_old=loaded_xxxx['MPO_B']
        print(Bflat_old.shape)
    except:
        Bflat_old=loaded_xxxx['MPO_B'][0]
    #print(len(Bflat_old))
    #print(len(Bflat_old[0]))
    #print(len(Bflat_old[0][0]))
    #print(len(Bflat_old[0][0][0]))
    #print(len(Bflat_old[0][0][0][0]))
    #print(Bflat_old.shape)

    print(Bflat_old.shape)
   
    Bflat_old=Bflat_old[permute,:,:,:]
    Bflat_old=Bflat_old[:,permute,:,:]
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

    with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/2_3_single_layer/"+name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    return loaded_xxxx
    

def add_cells_to_projected_wf(psi_halfinf,pstate,sites,charge):
    """
    adds pstate left to the projected WF
    psi_halfinf:
    """
    
    #ADD THREE SITES WITH DIFFERENT K
    #adds extra sites again
    a=MPS.from_product_state(sites[:len(pstate)],pstate,'finite')
    print(a._B[0])
    
    #print(a._B[1])
    #print(a._B[2])
    #print(psi_halfinf._B[0])
    #print(len(psi_halfinf._B))
    Bflat=[]
    Ss=psi_halfinf._S

    for i in range(len(psi_halfinf._B)):
        #print('bla'*20)
        Bflat.append(np.transpose(psi_halfinf._B[i].to_ndarray(),(1,0,2)))
    
        #print(psi_halfinf._B[i].to_ndarray().shape)

    for i in range(len(pstate)):
        Bflat.insert(0,np.transpose(a._B[len(pstate)-i-1].to_ndarray(),(1,0,2)))
        Ss.insert(0,[1.])
   
    
    
   
   
    #print(len(Ss))
    #TODO: FIGURE OUT CHARGES ON THE LEFT LEG:
    add=0
    for k,i in enumerate(pstate):
        if i=='full':
            add+=k%3+1
    add=3
    #print(add)
    #print(pstate)
    
    """
    FOR HALDANE POTENTIAL
    if pstate[0]=='empty':
        qflat=[[-6-len(pstate),0]] #if empty,empty,full
    else:
        qflat=[[-6+len(pstate),0]]
    """

    if pstate[0]=='empty':
        qflat=[charge]
        #FOR 1/3 AND 2/3
        qflat[0][0]= qflat[0][0]-len(pstate)
        
    else:
        qflat=[charge]
        #FOR 1/3
        qflat[0][0]= qflat[0][0]+len(pstate)
    print(qflat)
    #qflat=[[-6+len(pstate),0]] #if full,empty,empty
    #qflat=[[-6,0]] #if 010
    #qflat=[[-6-len(pstate),0]]
    chargeinfo=sites[0].leg.chinfo
    left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
    
    #left_leg=a._B[0].get_leg('vL')
    #print(psi_halfinf._B[0])


    #left_leg=psi_halfinf._B[0].get_leg('vL')
    #print(len(sites))
    psi=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    #print(psi._B[0])
    
    return psi


def load_environments_from_file(name,name_load,side='right'):
    file_path="/mnt/users/dperkovic/quantum_hall_dmrg/data_load/2_3_single_layer/"+name+'.npz'
    data =np.load(file_path,allow_pickle=True)
    
    file_path="/mnt/users/dperkovic/quantum_hall_dmrg/data_load/2_3_single_layer/"+name_load+'.pkl'
    with open(file_path, 'rb') as f:
        loaded_qs= pickle.load(f, encoding='latin1')
    
    if side=='right':
        dictionary={}
        dictionary['RP_B']=data['RP']
        #print(data['RP_q'])

        
        dictionary['RP_q']=loaded_qs['RP_q']
        
        
    else:
        dictionary={}
        dictionary['LP']=data['LP']
        dictionary['LP_q']=loaded_qs['LP_q']
   
    return dictionary

def run_vacuum_boundary_modified_K(name_load,name_save,pstate=[]):
    model_par,conserve,root_config_,loaded_xxxx=load_param(name_load)
   
    #load_graph(name_graph)
    #print(len(loaded_xxxx['graph']))
    
    #print(model_par)
    
    
    L=3*23-1
    graph=[loaded_xxxx['graph'][0]]*L
    L=len(graph)+len(pstate)
   
    #adds extra sites to the graph
    for i in range(len(pstate)):
        graph.insert(0,graph[i])
   
    #L=14

    
    #model_par.pop('mpo_boundary_conditions')


    model_par['boundary_conditions']= ('infinite', L)
    print('__'*100)
    print(model_par)

    #NEED TO IMPORT LAYERS TOO


    LL=0
    model_par['layers']=[ ('L', LL) ]
    #print(model_par)
    

    #INSERT UNIT CELL IN BETWEEN!
   
    M,sites,ordered_states=create_segment_DMRG_model(model_par,L,root_config_,conserve,graph,add=False)#len(pstate))
    

    perm=load_permutation_of_basis(loaded_xxxx,ordered_states,M.H_MPO._W[0])
    print('AAAAA')
    print(perm)
    
    psi_halfinf=load_data(loaded_xxxx,sites[len(pstate):],shift=len(pstate))
  
    #THIS ONE HAS BOTH N,K conservation
    psi_halfinf,charge=project_left_side_of_mps(psi_halfinf)
    print(psi_halfinf._B[0])
    print(charge)
    
    #print('origig'*100)
    #print(psi_halfinf._B[-1])
    #psi_halfinf.canonical_form_finite()
    #print(psi_halfinf._B[0])


    

    if len(pstate)>0:
  
        psi_halfinf=add_cells_to_projected_wf(psi_halfinf,pstate,sites,charge)
    #psi_halfinf.add_B(-1,a._B[2])
    #psi_halfinf.add_B(-2,a._B[1])
    #psi_halfinf.add_B(-3,a._B[0])
    psi_halfinf.canonical_form_finite(cutoff=0.0)
    #print(len(psi_halfinf._B))
    

    print('psi'*100)
    #print(psi_halfinf._B[-1])
    #print(len(sites))
    
    right_env=load_environment(loaded_xxxx,len(sites),root_config_,conserve, perm,side='right',old=True)
    print('env'*100)
    print(right_env)
    



    #leg_MPS
    leg_MPS=psi_halfinf._B[0].get_leg('vL')
    #leg_MPO
    leg_MPO=M.H_MPO._W[0].get_leg('wL')
    left_environment=set_left_environment_to_vacuum(leg_MPO,leg_MPS)


    init_env_data_halfinf={}


    init_env_data_halfinf['init_LP'] = left_environment    #DEFINE LEFT ENVIROMENT
    init_env_data_halfinf['age_LP'] =0

    init_env_data_halfinf['init_RP'] = right_env   #DEFINE RIGHT ENVIROMENT
    init_env_data_halfinf['age_RP'] =0




    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-9,
        'max_S_err': 1.e-5,
        'trunc_params': {
            'chi_max': 2000,
            'svd_min': 1.e-9,
        },
        'max_sweeps': 50
    }

    eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_halfinf, M, dmrg_params,
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


    

    #data = {"psi": psi,  # e.g. an MPS
    #        "dmrg_params":dmrg_params, "model_par":model_par, "model": M,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

    data = { "dmrg_params":dmrg_params,"energy":E0, "model_par":model_par,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

   
  

   
    with h5py.File("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/one_third_coloumb_xi=1/"+name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)

def patch_WF_together(psi1,psi2,sites):

    """
    patches psi1 and psi2 together
    """
    #ADD THREE SITES WITH DIFFERENT K
    #adds extra sites again
   
    
    
    Bflat=[]
    Ss=[]
    Ss1=psi1._S
    Ss2=psi2._S
    print(psi1._B[0])
    #print(psi1._B[0])

    #constructs total Bflat by appending psis
    #same for Ss

    for i in range(len(psi1._B)):
    
        Ss.append(Ss1[i])
        Bflat.append(np.transpose(psi1._B[i].to_ndarray(),(1,0,2)))
   
    duljina_l=psi1._B[-1].shape[-1]

    legL=psi1._B[-1].get_leg('vR')
    print(legL)
 
    legR=psi2._B[0].get_leg('vL')
    print(legR)

  
    leg_physical=sites[len(psi1._B)].leg
    print(leg_physical)
    #quit()
    #print(duljina_l,duljina_r)
    #duljina_r=psi2._B[0].shape[0]
    #print(psi1._B[-1].shape)
    #print(np.transpose(psi1._B[-1].to_ndarray(),(1,0,2)).shape)
    #print(psi2._B[0].shape)
    B_new=set_MPS_boundary(legL,leg_physical,legR)

    #print(B_new.shape)
    #quit()

    #Ss.append(Ss1[-1])
    Ss.append(np.random.random(Ss1[-1].shape))
    Bflat.append(np.transpose(B_new,(1,0,2)))
    #Ss.append(Ss2[0])

    for i in range(len(psi2._B)):
      
        Ss.append(Ss2[i])
        Bflat.append(np.transpose(psi2._B[i].to_ndarray(),(1,0,2)))
    Ss.append(Ss2[-1])
   

    #reads charges off the left leg of MPS
    left_leg=psi1._B[0].get_leg('vL')
    print("start shit")

    print('TRY TRY SHORTER')
    print(len(Bflat))
    print(len(Ss))
    psi=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    psi.canonical_form_finite()
    print(len(Ss))
    print(len(sites))
    netq=[]
    for i in range(len(sites)-1):
        #print(i)
        B_i = psi.get_B(i)  # Get MPS tensor at site i
        S_i = np.array(psi.get_SL(i))
        #print(B_i)
        legL=B_i.get_leg('vL')
        #qflat_L=legL.to_qflat()
       
        #qflat_R=legR.to_qflat()
       
        leg_physical=sites[i].leg
        q_net=np.array([0.0,0.0])
        for m in range(len(S_i)):
            c=legL.get_qindex(m)
            chargeL=legL.charges[c[0]]
            #print(c)
            #print(chargeL)
           
            #print(c)
            #print(chargeR)
            q_net+=(np.abs(S_i[m])**2)*(chargeL)
        q_net1=q_net#/np.sum(np.abs(S_i)**2)

        B_i = psi.get_B(i+1)  # Get MPS tensor at site i
        S_i = np.array(psi.get_SL(i+1))

        legL=B_i.get_leg('vL')
        #qflat_L=legL.to_qflat()
       
        #qflat_R=legR.to_qflat()
       
     
        q_net=np.array([0.0,0.0])
        for m in range(len(S_i)):
            c=legL.get_qindex(m)
            #print(c)
            chargeL=legL.charges[c[0]]
            
            q_net+=(np.abs(S_i[m])**2)*(chargeL)
        q_net2=q_net#/np.sum(np.abs(S_i)**2)
    
        
        #qphys=q_net/np.abs(np.sum(S_i**2))
        qphys=q_net1-q_net2
        netq.append((-qphys[1]+2)/3)
        #print('NET CHARGE')
        #print((qphys[1]+2)/3)
        #print(np.sum(np.abs(S_i)**2))
    #psi=MPS.from_Bflat(sites[:len(Bflat)],Bflat, bc='segment',legL=left_leg)
    filling= psi.expectation_value("nOp")
    netq=np.array(netq)
    print(filling[:60]-netq[:60])
    #print(netq[:60])
    print(filling[len(psi1._B)])
    print(filling[len(psi1._B)])
    #print(filling[1:]-netq)
    print(len(psi._B)*2/3)
    print(np.sum(filling))
    print(np.sum(netq))
    print('Patched two wavefunctions together sucessfully')
    quit()
    return psi

def set_MPS_boundary(legL,leg_physical, legR):
  
    """
    produces left/right environment which corresponds to vacuum
    so far works only it total charge is zero
    leg_HMPO,leg_MPS: give the legs of MPO and MPS

    """
    duljina_LMPS=len(legL.to_qflat())
    duljina_RMPS=len(legR.to_qflat())
    duljina_physical=len(leg_physical.to_qflat())
    #gives corresponding labels to the environment
   
   
    L=legL.to_qflat()
    R=legR.to_qflat()
    ch_physical=leg_physical.to_qflat()[0]
 
   
    #WILL NEED TO CREATE THIS POOPOO AGAIN
    #CAN FILL ONLY SOME CHARGE SECTORS
    #set data flat
    Bflat=np.zeros(duljina_LMPS*duljina_physical*duljina_RMPS)
    
    Bflat=np.reshape(Bflat,(duljina_LMPS,duljina_physical,duljina_RMPS))
  
    for i,charge in enumerate(L):
        charge_2=charge+ch_physical
        ind=np.where((R==charge_2).all(axis=1))[0]
        Bflat[i,0,ind]+=np.random.random(len(ind))
        #print(np.random.random(len(ind)))
   
    ch_physical=leg_physical.to_qflat()[1]
    for i,charge in enumerate(L):
        charge_2=charge+ch_physical
        ind=np.where((R==charge_2).all(axis=1))[0]
        Bflat[i,1,ind]+=np.random.random(len(ind))
    
    #Bflat+=np.random.random((duljina_LMPS,duljina_physical,duljina_RMPS))
    print('average weight')
    print(np.sum(Bflat**2))
    print("OK filled")
 
  
    
    return Bflat





def run_vacuum_boundary_modified_K_2(name_load,name_save,name_graph,pstate=[]):
    model_par,conserve,root_config_,loaded_xxxx=load_param(name_load)
   

    #print(loaded_xxxx.keys())
    #print(len(loaded_xxxx['graph']))
    
    L=200
    L=101
    #L=150
    #L=199
    #L=35
    graph=[loaded_xxxx['graph'][0]]*L
    #graph=load_graph(name_graph)
    print(len(loaded_xxxx['graph']))
    
    #print(model_par)
    
    
    #L=3*23-1
    #graph=[loaded_xxxx['graph'][0]]*L
    L=len(graph)#+len(pstate)
    print('Length of chain:',L)
    #adds extra sites to the graph
    #for i in range(len(pstate)):
    #    graph.insert(0,graph[i])
   
    #L=14
    name_load2='DMRG_19.0_Haldane_barrier_0.0_mu_1_3_x2_'
    model_par,conserve,root_config_,loaded_xxxx2=load_param(name_load2)
    
    #model_par.pop('mpo_boundary_conditions')


    model_par['boundary_conditions']= ('infinite', L)
    print('__'*100)
    print(model_par)

    #NEED TO IMPORT LAYERS TOO


    LL=0
    model_par['layers']=[ ('L', LL) ]
    #print(model_par)
    

    #INSERT UNIT CELL IN BETWEEN!
    
    M,sites,ordered_states=create_segment_DMRG_model(model_par,L,root_config_,conserve,graph,add=False)
    

    perm=load_permutation_of_basis(loaded_xxxx,ordered_states,M.H_MPO._W[0])
    
    """
    sites=[]
    add=0
    conserve=('N','K')
    for i in range(-add,L-add):
        spin=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=conserve,site_loc=i)
        
        sites.append(spin)
    psi2=load_data(loaded_xxxx,sites[51:],shift=51)
   

 
    psi1=load_data(loaded_xxxx2,sites[:50],shift=len(pstate),charge_shift=[3,0])
    """
    #psi2=load_data(loaded_xxxx,sites[:50],shift=len(pstate))
    #filling= psi1.expectation_value("nOp")
    #print(filling)
    #quit()
    
    #print(psi1._B[0])
    #print(psi2)
   
    #patch_WF_together(psi1,psi2,sites)
    #quit()
   
    #deviation=np.sum(filling[:A]-1/3)
    #print(deviation)
    #quit()
    


    psi_halfinf=load_data(loaded_xxxx,sites[len(pstate):],shift=len(pstate))
  
    #THIS ONE HAS BOTH N,K conservation
    psi_halfinf,charge=project_left_side_of_mps(psi_halfinf)
    print('origig'*100)
    #charge=[0,0]
    #print(charge)
    filling= psi_halfinf.expectation_value("nOp")
    p_bfr=np.sum((filling-2/3)*np.arange(len(filling)))
    
    N_bfr=np.sum(filling-2/3)

    print(p_bfr)
    print(N_bfr)
    #A=3
    #print()
    print(filling)
    #print(psi_halfinf._B[-1])
    print(psi_halfinf._B[0])
    
    #psi_halfinf.canonical_form_finite()
    #print(psi_halfinf._B[0])


    
    
    if len(pstate)>0:
        print('ADDED CELLS')
        psi_halfinf=add_cells_to_projected_wf(psi_halfinf,pstate,sites,charge)
    #psi_halfinf.add_B(-1,a._B[2])
    #psi_halfinf.add_B(-2,a._B[1])
    #psi_halfinf.add_B(-3,a._B[0])
    psi_halfinf.canonical_form_finite(cutoff=0.0)
    #print(len(psi_halfinf._B))
    

    filling= psi_halfinf.expectation_value("nOp")
    p_after=np.sum((filling-2/3)*np.arange(len(filling)))
    N_aft=np.sum(filling)
    print('number of electrons')
    print(N_bfr,N_aft)
    print('dipole')
    print(p_bfr,p_after)
    print()
    quit()
    
    #quit()
    print('psi'*100)
    #print(psi_halfinf._B[-1])
    #print(len(sites))
    name='Environment_R_'+num
    data=load_environments_from_file(name,name_load,side='right')
    conserve=('N','K')
    right_env=load_environment(data,len(sites),root_config_,conserve, perm,side='right',old=True)
    print('env'*100)
    print(right_env)
    

    leg_MPS=psi_halfinf._B[-1].get_leg('vR').to_qflat()[0]
    #leg_MPO
    leg_env=right_env.get_leg('vL').to_qflat()[0]
    shift_final=leg_MPS-leg_env
    print(shift_final)
    quit()
    #leg_MPS
    leg_MPS=psi_halfinf._B[0].get_leg('vL')
    #leg_MPO
    leg_MPO=M.H_MPO._W[0].get_leg('wL')
    left_environment=set_left_environment_to_vacuum(leg_MPO,leg_MPS)


    init_env_data_halfinf={}


    init_env_data_halfinf['init_LP'] = left_environment    #DEFINE LEFT ENVIROMENT
    init_env_data_halfinf['age_LP'] =0

    init_env_data_halfinf['init_RP'] = right_env   #DEFINE RIGHT ENVIROMENT
    init_env_data_halfinf['age_RP'] =0




    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-9,
        'max_S_err': 1.e-6,
        'trunc_params': {
            'chi_max': 2500,
            'svd_min': 1.e-7,
        },
        'max_sweeps': 50
    }

    eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_halfinf, M, dmrg_params,
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
    print('entanglement spectrum:',E_spec)



    EE=psi.entanglement_entropy()
    print('entanglement entropy:',EE)


    

    #data = {"psi": psi,  # e.g. an MPS
    #        "dmrg_params":dmrg_params, "model_par":model_par, "model": M,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

    data = { "dmrg_params":dmrg_params,"energy":E0, "model_par":model_par,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

  
  

   
    with h5py.File("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/single_layer_laughlin_2_3/"+name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)


name_graph=str(sys.argv[1])
num=str(sys.argv[1])
name_load=str(sys.argv[2])
#print(name_graph)
pstate=str(sys.argv[3])
if pstate=='[]':
    pstate=[]
else:
    pstate=pstate.split(',')
#print(pstate)

#name_graph='Gs_18.0_Haldane_barrier_0.035_mu_1_3.data'
#name_graph='Gs18.0_Haldane_barrier_0.1_mu_1_3_'
#name_load='DMRG_18.0_Haldane_barrier_0.1_mu_1_3_'
#name_load='Lx_16_QH_nu_1_3'
#name_save='no_mu_added_momentum_segment_'+name_load+'_'+str(pstate)
#run_vacuum_boundary_modified_K(name_load,name_save,pstate=pstate)
print(name_load)
print(name_graph)
#name_load='DMRG_18_Coulomb_barrier_0.075_mu_1_3.data'
name_save='L_y='+str(num)+'_infinite_well_L=125_nu=2_3_single_layer_'+name_graph+'_'+str(pstate[:3])+'_num='+str(len(pstate))

name_load='DMRG_'+num+'_Haldane_barrier_0.0_mu_1_3_x2_'
name_load='DMRG_18.0_Haldane_barrier_single_layer_mu_2_3'
#name_load='DMRG_18.0_Coulomb_barrier_-0.2_mu_1_3_'
#name_save='xi=1_L=200_'+'Coulumb_infinite_well'+'_'+str(pstate[:3])+'_num='+str(len(pstate))
print('***'*30)
print("file name:")
print(name_save)
print('***'*30)
run_vacuum_boundary_modified_K_2(name_load,name_save,name_graph,pstate)
#run_vacuum_boundary_modified_K_2(name_load,name_save,pstate)
