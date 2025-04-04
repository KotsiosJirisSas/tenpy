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
import pickle
np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

def trim_down_the_MPO(G):
    """
    Trims down the MPO from the keys that are not connected to any legs, i.e. are connected to zero block.
    This bit of code eliminates those keys that are multiplied by zero within the MPO itself.
    input: G which is graph
    output: G without unnecessary keys, extra keys - list of expelled keys
    """
    extra_keys=[]

   
    for n in range(len(G)):
        extra_keys_row=[]
        for key in G[n].keys():
           
            if list(G[n][key].keys())==[]:
                
                extra_keys_row.append(key)
            
        extra_keys.append(extra_keys_row)
        for k in extra_keys_row:
            #pass
            G[n].pop(k, None)
   
    return G, extra_keys

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

def get_old_basis(G_old,basis_old,permutation_old,Nlayer=1):
    """
    Creates TeNpy2 loaded basis

    G_old: gives the MPOgraph, used to eliminate basis
    basis_old: gives the old basis which is untrimmed and unpermuted before 
    permutation_old: gives the permutation of old basis before and after sorting the charges
    returns:
    ordered_old_basis

    GENERALIZED FOR N LAYERS, BECAUSE EACH BOND HAS DIFFERENT PERMUTATIONS AND BASIS
    """
    
    
    #trim the MPO
    G_old, extra_keys=trim_down_the_MPO(G_old)
    
    #determine the keys and keys that should be eliminated
    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_old,len(G_old))
 
 
    
    previous_length=len(States[0])+1

    #iteratively eliminates all the codewords that are not connected to anything
    while len(States[0])!=previous_length:

        previous_length=len(States[0])
        #Nlayer is number of layers

        
        not_included_couplings_copy=[]
        for i in range(Nlayer):
            for m in not_included_couplings[i]:
                not_included_couplings_copy.append(m)
       
        
        G_old=eliminate_elements_from_the_graph(G_old,not_included_couplings_copy)
        G_old, extra_keys=trim_down_the_MPO(G_old)

        #obtain new basis of keys for reduced graph
        States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_old,len(G_old))

  
    #states should give the correct values on G_old

    #Produce the keywords in TENPY3 notation
    States2=[]
    for i in range(len(States)):
        States2.append(QH_G2MPO.basis_map(States[i]))
    States=States2

    #we want to order the basis corresponding to permutations
    ordered_old_basis=[[] for i in range(len(basis_old))]
    for i in range(Nlayer):
        #gives all codewords that are unnecessary and removes them from basis old,
        #we have to do it this way, so that order is preserved
        removed_total= set(basis_old[i]) - set(States[i])
        

        
        for m in removed_total:
            basis_old[i].remove(m)
        
        #permute the basis in a given way
        ordered_old_basis[i]=list(np.array(basis_old[i])[permutation_old[i]])
        ordered_old_basis[i]=[str(x) for x in ordered_old_basis[i]]

    return ordered_old_basis


def create_segment_DMRG_model(model_par,L,root_config_,conserve,loaded_xxxx,add=False):

    """
    Creates MPOModel given graph from tenpy2.

    model_par: parameters for creation of MPOModel
    L: int, number of sites on chain
    root_config_: root configuration
    conserve: gives tuple of conserved quantities
    loaded_xxxx: graph in old tenpy
   
    returns:
    MPOModel
    Sites - list of sites 
    """


    




    # construct MPO Model from the Graph using the tenpy2 code
    print("Start model in Old Tenpy",".."*10)
    print("Old code finished producing MPO graph",".."*10)

    
    #load tenpy2 graph into tenpy3 graph
 
    G=loaded_xxxx
    
    

    #trim down the MPO
    G, extra_keys=trim_down_the_MPO(G)

    #gives the graph in tenpy3
    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
   
    
    
    
    print('asserting that the size is compatible with enivornment.....')
    #assert L%cell==0


   

    #define Hilbert spaces for each site with appropriate conservation laws
    if not add:
        #if we did not add any additional cells to the system
        sites=[]
        for i in range(L):
            #produce a single site
            spin=QH_MultilayerFermionSite_final(N=1,root_config=root_config_,conserve=('each','K'),site_loc=i)
            
            sites.append(spin)
    else:
        
        
        sites=[]
        #produce sites
        for i in range(-add,L-add):
       
            spin=QH_MultilayerFermionSite_final(N=1,root_config=root_config_,conserve=('each','K'),site_loc=i)
           
            sites.append(spin)
    """
    leg_physical=sites[0].leg
    print(leg_physical)
    leg_physical=sites[1].leg
    print(leg_physical)
    leg_physical=sites[2].leg
    print(leg_physical)
    leg_physical=sites[3].leg
    print(leg_physical)
    quit()
    """
    M = MPOGraph(sites=sites,bc='segment',max_range=None) #: Initialize MPOGRAPH instance

    '''
    M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

    Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
    The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
    '''
   
    
    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)
    
    print("Ordering states",".."*10)
  
  
   
  
    #iteratively eliminates all the codewords that are not connected to anything
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
    M.graph = G_new #: INppuut the graph in the model 
    print("Test passed!"+".."*10)
    grids =M._build_grids()#:Build the grids from the graph
    print("Building MPO"+".."*10)


    H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
    print("Built"+".."*10)


    #sort leg charges to make DMRG algortihm quicker
    perms2=H.sort_legcharges()
    
    
    #orderes the state according to the charges
    ordered_states=[]
    for k in range(len(M._ordered_states)):
        ordered_states_=[]
        for i in range(len(M._ordered_states[k])):
            b=[key for key, value in  M._ordered_states[k].items() if value == i]
            ordered_states_.append(b[0])
        ordered_states.append(ordered_states_)
   
    
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

    #assert model.H_MPO.is_equal(model.H_MPO.dagger())
    print('asserted')
   
    
    return model,sites, ordered_states

def load_data(loaded_xxxx,sites,shift=0,side='right',charge_shift=[0,0]):
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
        
        qflat2=loaded_xxxx['MPS_qflat']#[0]
    else:
        Bflat0=loaded_xxxx['MPS_Bs'].copy()
        #load singular values
        Ss=loaded_xxxx['MPS_Ss'].copy()
       
        #just need the leftmost leg, not all charges bro
        qflat2=loaded_xxxx['MPS_qflat']#[0]
       
    print(Bflat0[0].shape)
    print(Bflat0[1].shape)
    print(Bflat0[2].shape)
    print(Bflat0[3].shape)
    #change qflat into representation consistent with Tenpy3
    #this is just charges of the leftmost leg
    qflat=[]

    
    for i in range(len(qflat2)):

        kopy=[]
        for m in range(len(qflat2[i])):
            if m==0:
                shifted=qflat2[i][0]+shift*qflat2[i][1]
                kopy.append(shifted)
            
            else:
                kopy.append(qflat2[i][m])
        qflat.append(kopy)


    print("shift charges")
    print(charge_shift)
    for i in range(len(qflat)):
        qflat[i][0]+=charge_shift[0]
        qflat[i][1]+=charge_shift[1]
        
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
    
    
    #create MPS,
    #charges are calculated from the left leg
    print('loading the MPS')
    #mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)

    
    mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    print('loaded mps from data',".."*30)
    if side=='right':
        leg=mps._B[1].get_leg('vL')
        

        mps=MPS.from_Bflat(sites[1:],Bflat[1:],SVs=Ss[1:], bc='segment',legL=leg)
    
   
    return mps


def project_side_of_mps( psi_halfinf,side='left',charge=[0,0]):
    """
    projects MPS to a single schmidt value:
    psi_half_inf: MPS
    side: 'left' or 'right' depending which side we want to project.
    If left we project to largest SV, if right we project to largest schmidt value
    with correponsing charge list
    
    """
    #project the left edge of MPS to a single schmidt value
    #makes convergence on boundary with vacuum quicker
    print('projecting MPS',".."*30)
    
    if side=='left':
        #delete environments so that they get reset after the projection
        psi_halfinf.segment_boundaries=(None,None)


        
       

        S =  psi_halfinf.get_SL(0)
        proj = np.zeros(len(S), bool)
        #projects onto a largest schmidt value
        ind=np.argsort(S)[-1]
        proj[ind] = True
       
        leg=psi_halfinf._B[0].get_leg('vL')
        c=leg.get_qindex(np.argmax(S))
       
        charge=leg.charges[c[0]]
       
        B = psi_halfinf.get_B(0, form='B')
        B.iproject(proj, 'vL')
    
       
        psi_halfinf.set_B(0, B, form='B')
        psi_halfinf.set_SL(0, np.ones(1, float))
        psi_halfinf.canonical_form_finite(cutoff=0.0)
        psi_halfinf.test_sanity()
      
        
    else:
        #delete environments so that they get reset after the projection
     
        psi_halfinf.segment_boundaries=(None,None)
        S =  psi_halfinf.get_SL(-1)
   

        #instead need to find one corresponding to the above charge hmhmhmhm.
        print('Charge of leg to which we project')
        print(charge)
        

        leg=psi_halfinf._B[-1].get_leg('vR')
        flats=leg.to_qflat()
        

        #need to change this...
        ind=[]
    
        #finds the schmidt values with the corresponding charge
        for i in range(len(flats)):
            if np.all(flats[i]==charge):
                ind.append(i)
        
    
        ind=np.array(ind)
        #finds indices of such charges
        maxi=ind[np.argmax(S[ind])]
       
        
       
        
        
        #project to the corresponding sector

        proj = np.zeros(len(S), bool)
        proj[maxi] = True
       
        B = psi_halfinf.get_B(-1, form='B')
       
        B.iproject(proj, 'vR')
        psi_halfinf.set_B(-1, B, form='B')
        psi_halfinf.set_SL(-1, np.ones(1, float))
        psi_halfinf.canonical_form_finite(cutoff=0.0)
        psi_halfinf.test_sanity()
        
      

    print('projected MPS',".."*30)
 
    return psi_halfinf,charge


def project_side_of_mps_given_charges( psi_halfinf,charge=[[0,0]]):
    """
    projects MPS to a single schmidt value:
    psi_half_inf: MPS
    side: 'left' or 'right' depending which side we want to project.
    If left we project to largest SV, if right we project to largest schmidt value
    with correponsing charge list
    
    """
    #project the left edge of MPS to a single schmidt value
    #makes convergence on boundary with vacuum quicker
    print('projecting MPS',".."*30)

    #delete environments so that they get reset after the projection
    
    psi_halfinf.segment_boundaries=(None,None)
    S =  psi_halfinf.get_SL(0)


    #instead need to find one corresponding to the above charge hmhmhmhm.
    print('Charge of leg to which we project')
    print(charge)
    

    leg=psi_halfinf._B[0].get_leg('vL')
    flats=leg.to_qflat()
    print(flats)
    

    #need to change this...
    ind=[]

    #finds the schmidt values with the corresponding charge
    for j in range(len(charge)):
        for i in range(len(flats)):
            if np.all(flats[i]==charge[j]):
                ind.append(i)
    

    ind=np.array(ind)

    print(ind)
    
    
    
    
    #project to the corresponding sector

    proj = np.zeros(len(S), bool)
    proj[ind] = True
    Ss=S[ind]
    B = psi_halfinf.get_B(0, form='B')
    
    B.iproject(proj, 'vL')
    psi_halfinf.set_B(0, B, form='B')
   
    psi_halfinf.set_SL(0, Ss)
    #psi_halfinf.set_SL(0, np.ones(1, float))
    psi_halfinf.canonical_form_finite(cutoff=0.0)
    psi_halfinf.test_sanity()
    
    

    print('projected MPS',".."*30)
 
    return psi_halfinf,charge

def project_RHS_of_mps_vac( psi_halfinf,charge=[0,0]):


    #delete environments so that they get reset after the projection
    #TRY WITH -1, if this dont work, then len-1
    psi_halfinf.segment_boundaries=(None,None)
    S =  psi_halfinf.get_SL(-1)
   

    #instead need to find one corresponding to the above charge hmhmhmhm.
    print('CHARGE NEEDS TO BE PRINTED')
    print(charge)
    

    leg=psi_halfinf._B[-1].get_leg('vR')
    flats=leg.to_qflat()
    #print(flats)



    

    maxi=np.argmax(S)
   
 

    proj = np.zeros(len(S), bool)
    proj[ maxi] = True
    
    B = psi_halfinf.get_B(-1, form='B')
    print(B)
    B.iproject(proj, 'vR')
    psi_halfinf.set_B(-1, B, form='B')
    psi_halfinf.set_SL(-1, np.ones(1, float))
    psi_halfinf.canonical_form_finite(cutoff=0.0)
    psi_halfinf.test_sanity()
    print(psi_halfinf._B[-1])
    

    print('projected MPS',".."*30)
 
    return psi_halfinf,charge



def project_and_match_charges( psi_halfinf,side='right'):
    """
    projects MPS to a single schmidt value:
    psi_half_inf: MPS
    side:  'right' 
    we project to largest SV
    """
    print('projecting MPS',".."*30)
    
    if side=='right':
        #delete environments so that they get reset after the projection
        psi_halfinf.segment_boundaries=(None,None)


        
     

        S =  psi_halfinf.get_SL(-1)
        proj = np.zeros(len(S), bool)
        ind=np.argsort(S)[-4]
        proj[ind] = True
       
        leg=psi_halfinf._B[-1].get_leg('vR')
        c=leg.get_qindex(ind)
   
        charge=leg.charges[c[0]]
  
        B = psi_halfinf.get_B(-1, form='B')
   
        B.iproject(proj, 'vR')
  
       
        psi_halfinf.set_B(-1, B, form='B')
        psi_halfinf.set_SL(-1, np.ones(1, float))
        psi_halfinf.canonical_form_finite(cutoff=0.0)
        psi_halfinf.test_sanity()

        
 
        

    print('projected MPS',".."*30)
 
    return psi_halfinf,charge


def patch_WF_together(psi1,psi2,sites,pstate):

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
    leg_physical=[]
    for i in range(len(pstate)+1):
        leg_physical.append(sites[len(psi1._B)+i].leg)
    print(leg_physical)

    #print(duljina_l,duljina_r)
    #duljina_r=psi2._B[0].shape[0]
    print(psi1._B[-1].shape)
    print(np.transpose(psi1._B[-1].to_ndarray(),(1,0,2)).shape)
    print(psi2._B[0].shape)
    B_new=set_MPS_boundary(legL,leg_physical,legR,pstate)
    

    Ss.append(Ss1[-1])
    Ss.append(Ss1[-1])
    Bflat.append(np.transpose(B_new[0],(1,0,2)))
    print(np.sum(np.abs(B_new[0])**2))


    #last one, this one degines connection betwen left adn right
    #Ss.append(Ss1[-1])
    #print(B_new.shape)
    #quit()
    
    #Ss.append(np.random.random(Ss1[-1].shape))
    
    #Ss.append(Ss2[0])

    for i in range(len(psi2._B)):
      
        Ss.append(Ss2[i+1])
        Bflat.append(np.transpose(psi2._B[i].to_ndarray(),(1,0,2)))
    #Ss.append(Ss2[-1])
   

    #reads charges off the left leg of MPS
    left_leg=psi1._B[0].get_leg('vL')
    print("start shit")

    print('TRY TRY SHORTER')
    print(len(Bflat))
    print(len(Ss))
    #MAIN POINT IS THAT ERROR WAS REMOVED, AND ISNTEAD JUST WARNING HAPPENS WITH SETTING ALL OTHER ELEMENTS TO ZERO
    psi=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    filling= psi.expectation_value("nOp")
    print(psi)
    print(filling)
    #quit()
    #print(len(psi._B)*1/2)
    #print(np.sum(filling))
    print(psi._B[len(psi1._B)+len(pstate)+1])
    print(psi2._B[0])
    #psi=MPS.from_Bflat(sites[:len(Bflat)],Bflat, bc='segment',legL=left_leg)
    leg=psi._B[len(psi1._B)+len(pstate)+1].get_leg('vL')
    
    charges=leg.charges
    print('length of leg',print(len(charges)))
    psi2,ch=project_side_of_mps_given_charges(psi2,charges)
    
    
    print(psi2)
    #print(psi2._B)
    print('Patched two wavefunctions together sucessfully')


     
    Bflat=[]
    Ss=[]
    Ss1=psi._S
    Ss2=psi2._S
    #print(psi1._B[0])
    #print(psi1._B[0])

    #constructs total Bflat by appending psis
    #same for Ss

    for i in range(len(psi1._B)):
    
        Ss.append(Ss1[i])
        Bflat.append(np.transpose(psi._B[i].to_ndarray(),(1,0,2)))
   
  
 
    #Ss.append(Ss1[len(psi1._B)])
    for i in range(len(pstate)+1):
        #try this and see what happens
        #Ss.append(Ss1[-1])
        #print(Ss1[-1].shape)
        Ss.append(Ss1[len(psi1._B)+i])
        print(psi._B[len(psi1._B)+i].to_ndarray().shape)

        Bflat.append(np.transpose(psi._B[len(psi1._B)+i].to_ndarray(),(1,0,2)))
   


    for i in range(len(psi2._B)):
      
        Ss.append(Ss2[i])
        Bflat.append(np.transpose(psi2._B[i].to_ndarray(),(1,0,2)))
    Ss.append(Ss2[-1])
   

    #reads charges off the left leg of MPS
    left_leg=psi1._B[0].get_leg('vL')
    print("start shit")

    print('TRY TRY SHORTER')
   
    #MAIN POINT IS THAT ERROR WAS REMOVED, AND ISNTEAD JUST WARNING HAPPENS WITH SETTING ALL OTHER ELEMENTS TO ZERO
    psi=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    filling= psi.expectation_value("nOp")
    print(psi)
    print(filling)
    a=np.sum(filling)
    print(np.sum(filling))
    print(np.sum((filling-0.5)*np.arange(len(filling))))
 
    
    return psi


def load_environment(loaded_xxxx,location,root_config_,conserve,permute, side='right',old=False,charge_shift=[0,0]):

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

        
       
        Bflat=loaded_xxxx['RP']
        qflat_list_c=loaded_xxxx['RP_q']
    
       
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
        shift=3
        
       
   
   
    


    #create site at the end of the chain
    site=QH_MultilayerFermionSite_final(N=1,root_config=root_config_,conserve=('each','K'),site_loc=location)
    chargeinfo=site.leg.chinfo

    #shifts K by num_site-2 to get the correct charge matching in K sector
    #rule is simple. K= \sum_i N_i i, so shift of each K value is just N_i*(num_sites-2)
    #first column in qflat has information on K charges, and second on N charges


    for i in range(len(qflat_list[0])):
      
        
        #shifts momentum
        qflat_list[0][i][0]+=qflat_list[0][i][1]*(location-shift)

        #ADD CONSTANT SHIFT ON AUXILIARY LEGS
        qflat_list[0][i][0]+= charge_shift[0]
        qflat_list[0][i][1]+= charge_shift[1]
  

    
    for i in range(len(qflat_list[1])):
        #shifts momentum
        qflat_list[1][i][0]+=qflat_list[1][i][1]*(location-shift)
    

    for i in range(len(qflat_list[2])):

        
      
        #shifts momentum
        qflat_list[2][i][0]+=qflat_list[2][i][1]*(location-shift)
        
       
        #ADD CONSTANT SHIFT ON AUXILIARY LEGS
        qflat_list[2][i][0]+= charge_shift[0]
        qflat_list[2][i][1]+= charge_shift[1]
    
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

    
    return environment




def set_environment_to_vacuum(leg_HMPO,leg_MPS,side='left'):
  
    """
    produces left/right environment which corresponds to vacuum
    so far works only it total charge is zero
    leg_HMPO,leg_MPS: give the legs of MPO and MPS

    """
    duljina_MPS=len(leg_MPS.to_qflat())
    duljina=len(leg_HMPO.to_qflat())

    #gives corresponding labels to the environment
    if side=='left':
        labels=['vR*', 'wR', 'vR']
    else:
        labels=['vL*', 'wL', 'vL']

    
    #gives qflat for non-trivial leg    
    legcharges=[]
    legcharges.append(leg_MPS)
    legcharges.append(leg_HMPO.conj())
    legcharges.append(leg_MPS.conj())
  

    
    #set data flat
    Bflat=np.zeros(duljina*duljina_MPS*duljina_MPS)
    
    Bflat=np.reshape(Bflat,(duljina_MPS,duljina,duljina_MPS))
    
    #find all indices of MPO at which charge is zero
    qflat=leg_HMPO.conj().to_qflat()
    
    
    index=[]
    for i,m in enumerate(qflat):
        find_zero=np.all(m==0)
        if find_zero:
            index.append(i)
            
            
    #set only 0 charges of MPO to have value 1 because this conserves N,K
    a,b,c=Bflat.shape
    Bflat=0*Bflat
    for m in index:
        for i in range(a):
            Bflat[i,m,i]=1
  
    
    #define left environment
    environment=Array.from_ndarray( Bflat,
                        legcharges,
                        dtype=np.float64,
                        qtotal=None,
                        cutoff=None,
                        labels=labels,
                        raise_wrong_sector=True,
                        warn_wrong_sector=True)
    print(environment)
    print("vacuum left environment is loaded",'..'*20)
    print(environment.qtotal)
    
    return environment

def load_param(name):

    with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/pf_apf_final/"+name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    #IF LOADED FROM H5 FILE UNCOMMENT
    print(loaded_xxxx.keys())
    model_par = loaded_xxxx['Parameters']
    #print(model_par)
    #quit()
    root_config_ = model_par['root_config'].reshape(4,1)
    #print(root_config_)
    
    conserve=[]
   
    conserve=['N','K']
    #load root_configuration and conservation laws
    if len(conserve)==1:
        conserve=conserve[0]
    else:
        conserve=tuple(conserve)
    return model_par,conserve,root_config_,loaded_xxxx

def set_MPS_boundary(legL,leg_physical, legR,pstate=[]):

  
    """
    produces left/right environment which corresponds to vacuum
    so far works only it total charge is zero
    leg_HMPO,leg_MPS: give the legs of MPO and MPS

    """
    duljina_LMPS=len(legL.to_qflat())
    duljina_RMPS=len(legR.to_qflat())
    duljina_physical=len(leg_physical[0].to_qflat())
    #gives corresponding labels to the environment
   
   
 
    Bs=[]
       
    #create first leg part
    duljina_left=duljina_LMPS
    L=legL.to_qflat()
    leg_physical_site=leg_physical[0]
    print("number of sites in this thing")
    print(len(leg_physical))
    #need to fill both so that charge on this side
 
    #remove duplicates
    #qflat = list(map(list, set(map(tuple, qflat))))

    #qflat=np.array(qflat)
    #sort charges
    #sorted_charges = np.lexsort(np.array(qflat).T) 
    #qflat= qflat[sorted_charges]
    #set left leg to be equal to qflat
    #L=np.array(qflat)
    R=legR.to_qflat()


    #NOW AT THE LAST SITE WE HAVE TO MATCH TO SITE to the right
    #HERE WE WANT TO HAVE BOTH CHARGES
    duljina_RMPS=len(R)
    Bflat=np.zeros(duljina_left*duljina_physical*duljina_RMPS)  
    Bflat=np.reshape(Bflat,(duljina_left,duljina_physical,duljina_RMPS))
    leg_physical_site=leg_physical[-1]
    print("number of sites in this thing")
    print(len(leg_physical))
    print(duljina_RMPS)
    #need to fill both so that charge on this side
    #turns out to be 0.5
    ch_physical=leg_physical_site.to_qflat()[0]
    for i,charge in enumerate(L):
        charge_2=charge+ch_physical
       
      
        ind=np.where((R==charge_2).all(axis=1))[0]
        print(ind)
        Bflat[i,0,ind]+=np.random.random(len(ind))
        #print(np.random.random(len(ind)))

    #print('left-right charges')
    ch_physical=leg_physical_site.to_qflat()[1]
    for i,charge in enumerate(L):
        charge_2=charge+ch_physical
      
        #produce sites only when these match!!!
        ind=np.where((R==charge_2).all(axis=1))[0]
        print(ind)
        Bflat[i,1,ind]+=np.random.random(len(ind))
    Bflat[:,0,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
    Bflat[:,1,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
    #print(R)
    Bs.append(Bflat)
    
    
    return Bs







def load_environments_from_file(name,name_load,side='right'):
    file_path="/mnt/users/dperkovic/quantum_hall_dmrg/data_load/pf_apf_final/"+name+'.npz'
    data =np.load(file_path,allow_pickle=True)
    
    file_path="/mnt/users/dperkovic/quantum_hall_dmrg/data_load/pf_apf_final/"+name_load+'.pkl'
    with open(file_path, 'rb') as f:
        loaded_qs= pickle.load(f, encoding='latin1')
    print(loaded_qs.keys())
    if side=='right':
        dictionary={}

        
        dictionary['RP']=data['RP']
        #print(data['RP_q'])

        #if reconstructed RP_reconstructed_q'
        #REMOVE
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
    
    #print(len(loaded_xxxx['indices']))
   
    
    G_old,basis_old,permutation_old=loaded_xxxx['graph'],loaded_xxxx['indices'][::-1],[loaded_xxxx['permutations']]
    
    #print(len(basis_old))
    #print(basis_old)
    
    #print(G_old[0].keys())
    #print(basis_old)
    
    
    #print(basis_old)
    
    
    
    old_basis=get_old_basis(G_old,basis_old,permutation_old)
    for i in range(len(old_basis)):
        assert len(old_basis[i])==len(ordered_states[i])
    #print(len(basis_old))
    #print(len(old_basis))
    #total_permutation=[]
    #print()
    
    #print(old_basis)
    #print(old_basis)
    #print(ordered_states[0])
    
    #print(set(old_basis)-set(ordered_states[0]))
    #print(set(ordered_states[0])-set(old_basis))
    
    permutation=[]
    for i in range(len(old_basis)):
        permutation.append(find_permutation(old_basis[i],ordered_states[i]))
    
    for i in range(len(old_basis)):
        #asserts two bases are the same
        assert np.all(ordered_states[i]==np.array(old_basis[i])[permutation[i]])
    
    #print(ordered_states[0]==np.array(old_basis)[permutation])
    
    #print(loaded_xxxx.keys())
    
    sanity_check_permutation(loaded_xxxx,new_MPO, permutation[0],permutation[0])
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

    with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/pf_apf_final/"+name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    return loaded_xxxx
    

def add_cells_to_projected_wf(psi_halfinf,pstate,sites,charge=[0,0]):
    """
    adds pstate left to the projected WF
    psi_halfinf:
    """
    print('adding cells...................')
    #ADD THREE SITES WITH DIFFERENT K
    #adds extra sites again
    a=MPS.from_product_state(sites[:len(pstate)],pstate,'finite')
    #print(a._B[0])
    
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
 
   
    qflat=[[0,0]]
    #qflat=[[-16,0]]
    #qflat=[[-24,0]]
    chargeinfo=sites[0].leg.chinfo
    left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
    
   
    #print(psi_halfinf._B[0])
    psi=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)

    
    leg=psi._B[len(pstate)].get_leg('vL')
    charge_new=leg.charges[0]
    charge_diff=[]
    
    for i in range(len(charge)):
        charge_diff.append(-charge_new[i]+charge[i])
    

    left_leg=LegCharge.from_qflat(chargeinfo,[charge_diff],qconj=1).bunch()[1]
    psi=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    
    print(charge)
    #print(psi._B[len(pstate)])
    
    return psi,charge_diff



def bulk_vacuum_boundary(name_load,name_save,name_graph,pstate=[]):
    model_par,conserve,root_config_,loaded_xxxx=load_param(name_load)
    #model_par,conserve,root_config_,loaded_xxxx2=load_param(name_load2)
    loaded_xxxx2={'MPS_Ss':loaded_xxxx['MPS_2_Ss'],'MPS_qflat':loaded_xxxx['MPS_2_qflat'],'MPS_Bs':loaded_xxxx['MPS_2_Bs']}
    #TODO: COMMENT OUT STUFF
   
    #graph is fine
    
    print(loaded_xxxx['MPO_B'][0].shape)
  
    graph=loaded_xxxx['graph']
    
    
    print(len(graph))
    L=1+25*6
    L=218
    L=402
 
    #assert (L)%4==1
    half=0

    graph=[graph[0]]*L
    #graph= load_graph(name_graph)
    L=len(graph)
    #print(L)
    #quit()

   

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
    #print('AAAAA')
    #print(perm)

    print(len(pstate))
    

    #print(len(sites[half+len(pstate):]))
    psi_halfinf_right=load_data(loaded_xxxx2,sites[half+len(pstate):],shift=len(pstate),side='right')
    #quit()
    #print(len(psi_halfinf_right._B))
    print('psi_OLD_OLD'*100)
    print(psi_halfinf_right._B[0])
    print(psi_halfinf_right._B[1])
    print(psi_halfinf_right._B[2])
    print(psi_halfinf_right._B[3])
    print(psi_halfinf_right._B[4])
    print('#'*100)
    print(len(psi_halfinf_right._B))
    #quit()
    psi_halfinf_right,charge2=project_side_of_mps(psi_halfinf_right,side='left')

   
    #gives charge2 which corresponds to the largest Schmidt value of RHS state
    print("Charge of the largest Schmidt value of LHS WF:")
    print(charge2)
 
   
   
    #THIS ONE HAS BOTH N,K conservation
    if len(pstate)>0:
        psi_halfinf_right,charge2=add_cells_to_projected_wf(psi_halfinf_right,pstate,sites[half:],charge=charge2)

    print("Charge of the largest Schmidt value after adding cells:")
    print(charge2)

    print("Charge difference in two Schmidt values:")
 
    

    filling=psi_halfinf_right.expectation_value("nOp")

    print(filling)
    #quit()
    N_projected=np.sum(filling)
    print('N:',N_projected)
    print('N_expected:',L/2)
    #psi_halfinf.add_B(-1,a._B[2])
    #psi_halfinf.add_B(-2,a._B[1])
    #psi_halfinf.add_B(-3,a._B[0])
    #psi_total.canonical_form_finite(cutoff=0.0)
    #print(len(psi_halfinf._B))
    

    print('psi'*100)
    #print(psi_total._B[-1])
    print(psi_halfinf_right._B[-1])
    print(psi_halfinf_right._B[-2])
    print(psi_halfinf_right._B[-3])
    print(psi_halfinf_right._B[-4])
  
    print(len(sites))
    #environment LOAD
  
   
    name='Environment_R_APf'
    print("loading right env")
    name_env_q='Envs_qs'
    data=load_environments_from_file(name,name_env_q,side='right')

    
    #print(M.H_MPO._W[-1])
    #MAKE SURE YOU USE RIGHT PERMUTATION
    right_env=load_environment(data,(len(sites)-half)-1-1,root_config_,conserve, perm[0],side='right',old=True)
    print('right_env'*100)
    print(right_env)
  
   



   

    #leg_MPS
    leg_MPS=psi_halfinf_right._B[0].get_leg('vL')
    #leg_MPO
    leg_MPO=M.H_MPO._W[0].get_leg('wL')
    left_environment=set_environment_to_vacuum(leg_MPO,leg_MPS,side='left')

    #print(psi_halfinf_left._B[2])
    #print(psi_halfinf_left._B[3])
    #print(psi_halfinf_left._B[4])
    #print(psi_halfinf_left._B[5])
    #print(psi_total._B[0])
   
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
            'chi_max': 4500,
            'svd_min': 1.e-7,
        },
        'max_sweeps': 50}

    eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_halfinf_right, M, dmrg_params,
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

  

   
    with h5py.File("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/pf_apf/"+name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)

def bulk_vacuum_boundary_left_side(name_load,name_save,name_graph,pstate=[]):
    model_par,conserve,root_config_,loaded_xxxx=load_param(name_load)
    #model_par,conserve,root_config_,loaded_xxxx2=load_param(name_load2)
    loaded_xxxx2={'MPS_Ss':loaded_xxxx['MPS_2_Ss'],'MPS_qflat':loaded_xxxx['MPS_2_qflat'],'MPS_Bs':loaded_xxxx['MPS_2_Bs']}
    #TODO: COMMENT OUT STUFF
   
    #graph is fine
    
    print(loaded_xxxx['MPO_B'][0].shape)
  
    graph=loaded_xxxx['graph']
    
    
    print(len(graph))
    L=1+25*6
    L=300
   
    #assert (L)%4==1
    half=0

    graph=[graph[0]]*L
    #graph= load_graph(name_graph)
    L=len(graph)
    #print(L)
    #quit()

   

    #length of the system is L,
    #half is L//2
 

   

    model_par['boundary_conditions']= ('infinite', L)
    print('__'*100)
    print(model_par)

   

    LL=0
    model_par['layers']=[ ('L', LL) ]
    
    
    #INSERT UNIT CELL IN BETWEEN!
    M,sites,ordered_states=create_segment_DMRG_model(model_par,L,root_config_,conserve,graph,add=half)
 
 
    perm=load_permutation_of_basis(loaded_xxxx,ordered_states,M.H_MPO._W[0])
    #print('AAAAA')
    #print(perm)

    print(len(pstate))
    

    #print(len(sites[half+len(pstate):]))
    psi_halfinf_right=load_data(loaded_xxxx,sites[half+len(pstate):],shift=len(pstate),side='left')
    #quit()
    #print(len(psi_halfinf_right._B))
    
    print(len(psi_halfinf_right._B))
    #quit()
    psi_halfinf_right,charge2=project_RHS_of_mps_vac(  psi_halfinf_right)
    print('psi_proj_proj'*100)
    print(psi_halfinf_right._B[0])
    print(psi_halfinf_right._B[1])
    print(psi_halfinf_right._B[2])
    print(psi_halfinf_right._B[3])
    print(psi_halfinf_right._B[4])
    print('#'*100)
    #gives charge2 which corresponds to the largest Schmidt value of RHS state
    print("Charge of the largest Schmidt value of LHS WF:")
    #print(charge2)
 
   
   
    #THIS ONE HAS BOTH N,K conservation
    if len(pstate)>0:
        psi_halfinf_right,charge2=add_cells_to_projected_wf(psi_halfinf_right,pstate,sites[half:],charge=charge2)

    print("Charge of the largest Schmidt value after adding cells:")
    #print(charge2)

    print("Charge difference in two Schmidt values:")
 
    

    filling=psi_halfinf_right.expectation_value("nOp")

    print(filling)
    #quit()
    N_projected=np.sum(filling)
    print('N:',N_projected)
    print('N_expected:',L/2)
    #psi_halfinf.add_B(-1,a._B[2])
    #psi_halfinf.add_B(-2,a._B[1])
    #psi_halfinf.add_B(-3,a._B[0])
    #psi_total.canonical_form_finite(cutoff=0.0)
    #print(len(psi_halfinf._B))
    

    print('psi'*100)
    #print(psi_total._B[-1])
    print(psi_halfinf_right._B[-1])
    print(psi_halfinf_right._B[-2])
    print(psi_halfinf_right._B[-3])
    print(psi_halfinf_right._B[-4])
  
    print(len(sites))
    #environment LOAD
  
   
    name='Environment_L_Pf'
    name_env_q='Envs_qs'
    data=load_environments_from_file(name,name_env_q,side='left')
    left_environment=load_environment(data,-half-1,root_config_,conserve, perm[0],side='left',old=True)
    
 


   

    #leg_MPS
    leg_MPS=psi_halfinf_right._B[-1].get_leg('vR')
    #leg_MPO
    leg_MPO=M.H_MPO._W[-1].get_leg('wR')

    right_env=set_environment_to_vacuum(leg_MPO,leg_MPS,side='right')
    print(right_env)
    #print(psi_halfinf_left._B[2])
    #print(psi_halfinf_left._B[3])
    #print(psi_halfinf_left._B[4])
    #print(psi_halfinf_left._B[5])
    #print(psi_total._B[0])
   
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
            'chi_max': 4500,
            'svd_min': 1.e-7,
        },
        'max_sweeps': 50}
    
    eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_halfinf_right, M, dmrg_params,
                                        resume_data={'init_env_data': init_env_data_halfinf})
    #print(eng_halfinf.chi_max)
    #quit()
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

  

   
    with h5py.File("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/pf_apf/"+name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)

def bulk_bulk_boundary(name_load,name_save,name_graph,pstate=[]):
    model_par,conserve,root_config_,loaded_xxxx=load_param(name_load)
    #model_par,conserve,root_config_,loaded_xxxx2=load_param(name_load2)
    loaded_xxxx2={'MPS_Ss':loaded_xxxx['MPS_2_Ss'],'MPS_qflat':loaded_xxxx['MPS_2_qflat'],'MPS_Bs':loaded_xxxx['MPS_2_Bs']}
    #TODO: COMMENT OUT STUFF
   
    #graph is fine
    
    print(loaded_xxxx['MPO_B'][0].shape)
  
    graph=loaded_xxxx['graph']
    
    
    print(len(graph))
    L=1+25*6
    L=218+4*8
    L=402+1
    L=101
    
    #L=218
    #assert (L)%4==1
    #assert (L)%4==3
    #half=(L//8)*4
    
    
  
   
    L=150+12*4+16+16+8+32*2-2
    #add one more site
    #half=51+3+3*12+8+4+32
    #add one more site
    half=51+3+3*12+8+4+32+16
  
    graph=[graph[0]]*L
    #graph= load_graph(name_graph)
    L=len(graph)
    #print(L)
    #quit()

   

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
    #print('AAAAA')
    #print(perm)

    #print(len(pstate))
    

       
    #Load left hand side
    psi_halfinf_right=load_data(loaded_xxxx2,sites[len(pstate)+1+half-1:],shift=len(pstate)+half+1-1,side='right',charge_shift=[0,0])
    
    psi_halfinf_left=load_data(loaded_xxxx,sites[:half],shift=0,side='left')
    #print(len(sites[half+len(pstate):]))
    print('DO IT DO IT DO IT')
    print(len(sites[1+half+len(pstate):]))
    print(len(sites[:half]))
  
    
    #quit()
    #print(len(psi_halfinf_right._B))
    #print('psi_OLD_OLD'*100)
    #print(psi_halfinf_right._B[0])
    #print(psi_halfinf_right._B[1])
    #print(psi_halfinf_right._B[2])
    #print(psi_halfinf_right._B[3])
    #print(psi_halfinf_right._B[4])
    #print('#'*100)
    #print(len(psi_halfinf_right._B))
    #quit()
    #filling=psi_halfinf_right.expectation_value("nOp")
    #print('filling before projection')
    #print(filling)
    
   
   
    #N_projected=np.sum(filling)

    #print('N:',N_projected)
    #print('N_expected:',(1/2)*len(filling))
  

 
   
   
    #THIS ONE HAS BOTH N,K conservation
    #if len(pstate)>0:
    #    psi_halfinf_right,charge2=add_cells_to_projected_wf(psi_halfinf_right,pstate,sites[half:],charge=charge2)



   
    #N_projected=np.sum(filling)
    #print('N:',N_projected)
    #print('N_expected:',(1/2)*len(filling))

    print("Charge of the largest Schmidt of RHS wavefunction:")
   
  
    #filling=psi_halfinf_left.expectation_value("nOp")
    #print(filling)
    #print('LEFT: filling before the projection')


    psi_total=patch_WF_together(psi_halfinf_left,psi_halfinf_right,sites,pstate)
    #quit()
    #print(psi_halfinf_left._B[-1])
    #print(charge2)
    print('WF'*100)
    print(psi_total)
 


    
    #print('LEFT: filling after the projection')
    #print(filling)
    #quit()
  

    #print('TOTAL: filling after the projection')
    #print(filling)
    #quit()
    #N_projected=np.sum(filling)
    #print('N:',N_projected)
    #print('N_expected:',L/2)
    #psi_halfinf.add_B(-1,a._B[2])
    #psi_halfinf.add_B(-2,a._B[1])
    #psi_halfinf.add_B(-3,a._B[0])
    #psi_total.canonical_form_finite(cutoff=0.0)
    #print(len(psi_halfinf._B))
    

    #print('psi'*100)
    #print(psi_total._B[-1])
    #print(psi_halfinf_right._B[-1])
  
    #print(len(sites))
    #environment LOAD
  
    name='Environment_R_APf'
    print("loading right env")
    name_env_q='Envs_qs'
    data=load_environments_from_file(name,name_env_q,side='right')

    #print(M.H_MPO._W[-1])
    #MAKE SURE YOU USE RIGHT PERMUTATION
    right_env=load_environment(data,len(sites)-2,root_config_,conserve, perm[0],side='right',old=True)

    #as a test set RHS env to vacuum
    #may need to do it like this...
    #leg_MPS=psi_halfinf_right._B[-1].get_leg('vR')
    #leg_MPO
    #leg_HMPO=M.H_MPO._W[-1].get_leg('wR')
    #right_env=set_environment_to_vacuum(leg_HMPO,leg_MPS,side='right')
    print('right_env'*100)
    print(right_env)
    
    print(psi_total._B[-1])
    print(psi_total._B[-2])
    print(psi_total._B[-3])
    print(psi_total._B[-4])
    print(psi_total._B[-5])
    
    #quit()



    name='Environment_L_Pf'
    name_env_q='Envs_qs'
    data=load_environments_from_file(name,name_env_q,side='left')
    left_environment=load_environment(data,-1,root_config_,conserve, perm[0],side='left',old=True,charge_shift=[0,0])
    #print(M.H_MPO._W[0])
    print('psi'*100)
    
    print(psi_halfinf_left._B[0])
    print(psi_halfinf_left._B[1])
    print(psi_halfinf_left._B[2])
    #print(psi_halfinf_left._B[4])
    #print(psi_halfinf_left._B[5])
    #print(psi_total._B[0])
    print('leftenv'*100)
    print(left_environment)
    #print(psi_halfinf_left._B[0])
    #print(psi_total._B[0])
  
   
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
            'chi_max': 4500,
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
    #quit()
    E0, psi=eng_halfinf.run()



    #calculate and store data


    filling=psi.expectation_value("nOp")

    print('Filling:',filling)


    E_spec=psi.entanglement_spectrum(by_charge=True)
    #print('entanglement spectrum:',E_spec)



    EE=psi.entanglement_entropy()
    print('entanglement entropy:',EE)


    

    data = { "dmrg_params":dmrg_params,"energy":E0, "model_par":model_par,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

  

   
    with h5py.File("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/pf_apf/"+name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)

"""
R=[[0,0],[1,1]]
R=np.array(R)
charge_2=np.array([0,1])
ind=np.where((R==charge_2).all(axis=1))[0]
print('INDEX,INDEX')
print( ind)
"""
 
#name='Environment_R_reconstructed_Pf'
#print("loading right env")
#name_env_q='Envs_qs_pf'
#data=load_environments_from_file(name,name_env_q,side='right')
name_graph=str(sys.argv[1])
name_load2=str(sys.argv[2])
#print(name_graph)
pstate=str(sys.argv[3])
if pstate=='[]':
    pstate=[]
else:
    pstate=pstate.split(',')


name_load='Data'
#name_load='Envs_qs'
#name_save=name_graph+'_'+str(pstate[:4])+'_num='+str(len(pstate))

#name_save='Ly=18_L=221_large_no_external_potential_pf_apf_'+str(pstate[:4])+'_num='+str(len(pstate))
#name_save='not_reconst_pf_vac_boundary'+str(pstate[:4])+'_num='+str(len(pstate))

name_save='fourth_eigen_Ly=18_L=402_large_no_external_potential_pf_apf_'+str(pstate[:4])+'_num='+str(len(pstate))
#name_save='PROJECTED_LEFT_SIDE_pf_vac_boundary_L=402_'+str(pstate[:4])+'_num='+str(len(pstate))


name_save='Topo_dipole_technique_new_technique_Ly=18_L=302_pf_apf_'+str(pstate[:4])+'_num='+str(len(pstate))
print("#"*100)
print("Start the calculation")
print(name_save)
print("#"*100)
#bulk_vacuum_boundary_left_side(name_load,name_save,name_graph)
#bulk_vacuum_boundary(name_load,name_save,name_graph,pstate=pstate)
bulk_bulk_boundary(name_load,name_save,name_graph,pstate=pstate)

