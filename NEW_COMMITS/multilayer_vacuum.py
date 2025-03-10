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
        
        #print(not_included_couplings_copy)
        
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
        
        #print(States[0])
        print(len(basis_old[i]))

        
        for m in removed_total:
            basis_old[i].remove(m)
        #print(permutation_old)
        
        #this gives us appropriate basis
        #now calculate:
        #print(basis_old)
        #print(permutation_old)
        #permutation_old=np.arange(62)
        print(len(basis_old[i]))
        
        print(len(permutation_old[i]))
        a=np.array(basis_old[i])[permutation_old[i]]
        ordered_old_basis[i]=list(np.array(basis_old[i])[permutation_old[i]])
        ordered_old_basis[i]=[str(x) for x in ordered_old_basis[i]]
    
    print(len(ordered_old_basis[0]))
    print(len(ordered_old_basis[1]))
  
    
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
    #print(extra_keys)
   

    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
    #print(len(G_new[0].keys()))
    
    
    cell=len(root_config_)
    
    print('asserting that the size is compatible with enivornment.....')
    #assert L%cell==0


    #root_config_ = np.array([0,1,0])
  

    #define Hilbert spaces for each site with appropriate conservation laws
    if not add:
        sites=[]
        for i in range(L):
        
            spin=QH_MultilayerFermionSite_final(N=2,root_config=root_config_,conserve=('each','K'),site_loc=i)
            #print(spin.Id)
            
            sites.append(spin)
    else:
        print('in')
        
        sites=[]
        for i in range(-add,L-add):
            #print(i)
            spin=QH_MultilayerFermionSite_final(N=2,root_config=root_config_,conserve=('each','K'),site_loc=i)
           
            sites.append(spin)
            print(i)
        
   
    

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
        
        #print(not_included_couplings_copy)
        
        G_new=eliminate_elements_from_the_graph(G_new,not_included_couplings_copy)
        G_new, extra_keys=trim_down_the_MPO(G_new)
        States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)
    
   
    #print(G_new[1]["('a_', 1, 20)"])
    
    #print(len(States[0]))
    
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
        #print(loaded_xxxx.keys())
        
        #print(Bflat0[3].shape)
        #load charge infromation
        
        qflat2=loaded_xxxx['MPS_qflat'][0]

        print()
        #print('qflat2'*30)
        #print(qflat2)
        
        
    else:
        Bflat0=loaded_xxxx['MPS_Bs'].copy()
        #load singular values
        Ss=loaded_xxxx['MPS_Ss'].copy()
        print(Ss)
        
        #just need the leftmost leg, not all charges bro
        qflat2=loaded_xxxx['MPS_qflat'][0]
        print(len(qflat2))
       
      
        #    print(qflat2[i][:30])
        #    print('ENENENDNEDN')
        
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
    print(charge_shift)
    for i in range(len(qflat)):
        qflat[i][0]+=charge_shift[0]
        qflat[i][1]+=charge_shift[1]
        qflat[i][2]+=charge_shift[2]
    qflat=np.array(qflat)
    #print(qflat2)
    #print(qflat2[0].shape)
    
    #print(len(qflat2[0][0][0]))
   
    
   
    

        

    infinite_unit_cell=len(Bflat0)
    print(infinite_unit_cell)

    number=len(sites)//infinite_unit_cell+1
    print(number)
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
        print("changed leg")
        print(left_leg)
    
    #create MPS,
    #charges are calculated from the left leg
    


    print('loading the MPS')
    mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    print('loaded mps from data',".."*30)
    filling=mps.expectation_value("nOp")
    N=np.sum(filling)
    print(N)
    #quit()
    #TODO: FIX THIS STUFF IN THE MORNING
    
    return mps


def project_side_of_mps( psi_halfinf,side='left',charge=[0,0]):

    #project the left edge of MPS to a single schmidt value
    #makes convergence on boundary with vacuum quicker
    print('projecting MPS',".."*30)
    print(psi_halfinf.segment_boundaries)
    if side=='left':
        #delete environments so that they get reset after the projection
        psi_halfinf.segment_boundaries=(None,None)


        
        #ind=np.where(leg.charges==charge)[0][0]
        
        #a=leg.get_charge(c)
        #print(a)
        #

        S =  psi_halfinf.get_SL(0)
        print('LENGTH S')
        print(len(S))
        proj = np.zeros(len(S), bool)
        ind=np.argsort(S)[-1]
        proj[ind] = True
        print(ind)
        leg=psi_halfinf._B[0].get_leg('vL')

        print(leg)
        c=leg.get_qindex(ind)
        #print(leg.charges[c[0]])
        charge=leg.charges[c[0]]
        print(charge)
        B = psi_halfinf.get_B(0, form='B')
        #print(B)
        B.iproject(proj, 'vL')
        #print(B)
        #quit()
        psi_halfinf.set_B(0, B, form='B')
        psi_halfinf.set_SL(0, np.ones(1, float))
        psi_halfinf.canonical_form_finite(cutoff=0.0)
        psi_halfinf.test_sanity()
        #print(psi_halfinf._B[0])
        
    else:
        #delete environments so that they get reset after the projection
        #TRY WITH -1, if this dont work, then len-1
        psi_halfinf.segment_boundaries=(None,None)
        S =  psi_halfinf.get_SL(-1)
        print(len(S))

        #instead need to find one corresponding to the above charge hmhmhmhm.
        print(charge)
        

        leg=psi_halfinf._B[-1].get_leg('vR')
        flats=leg.to_qflat()
        #print(flats)

        #need to change this...

        #CHOOSE MODE WHETHER TO PROJECT TO HIGHEST LAMBDA OR GIVEN CHARGE
        fit_to_charg=False
        if fit_to_charg:
            ind=[]
            print(charge)
            print(flats[75:80])
            for i in range(len(flats)):
                if np.all(flats[i]==charge):
                    ind.append(i)
            
            #ind=np.where(flats==charge)[0]
            ind=np.array(ind)
            print(ind)
            print(len(ind))
        
            print(S[ind])
            maxi=ind[np.argmax(S[ind])]
            print(maxi)
            print(len(ind))
            proj = np.zeros(len(S), bool)
            proj[maxi] = True
        else:
            proj = np.zeros(len(S), bool)
            proj[np.argmax(S)] = True
        
        #print(ind)
        #print(charge)
        
        #print(len(flats[ind]))
        #print(flats[ind])
        
        
        
        #project to the corresponding sector

       
        #c=leg.get_qindex(np.argmax(S))
        #print(c)
        #a=leg.get_charge(c)
        #print(a)
        #print(leg.charges[c[0]])
        #charge=leg.charges[c[0]]
        #print(charge)
        B = psi_halfinf.get_B(-1, form='B')
        print(B)
        B.iproject(proj, 'vR')
        #print(B)
        psi_halfinf.set_B(-1, B, form='B')
        psi_halfinf.set_SL(-1, np.ones(1, float))
        psi_halfinf.canonical_form_finite(cutoff=0.0)
        psi_halfinf.test_sanity()
        print(psi_halfinf._B[-1])
        

    print('projected MPS',".."*30)
 
    return psi_halfinf,charge

def project_and_match_charges( psi_halfinf,side='right'):

    #project the left edge of MPS to a single schmidt value
    #makes convergence on boundary with vacuum quicker
    print('projecting MPS',".."*30)
    
    if side=='right':
        #delete environments so that they get reset after the projection
        psi_halfinf.segment_boundaries=(None,None)


        
        #ind=np.where(leg.charges==charge)[0][0]
        
        #a=leg.get_charge(c)
        #print(a)
        #

        S =  psi_halfinf.get_SL(-1)
        proj = np.zeros(len(S), bool)
        proj[np.argmax(S)] = True
       
        leg=psi_halfinf._B[-1].get_leg('vR')
        c=leg.get_qindex(np.argmax(S))
        #print(leg.charges[c[0]])
        charge=leg.charges[c[0]]
        print(charge)
        B = psi_halfinf.get_B(-1, form='B')
        print(B)
        
        #quit()
        B.iproject(proj, 'vR')
        #print(B)
       
        psi_halfinf.set_B(-1, B, form='B')
        psi_halfinf.set_SL(-1, np.ones(1, float))
        psi_halfinf.canonical_form_finite(cutoff=0.0)
        psi_halfinf.test_sanity()
        #print(psi_halfinf._B[0])
        
 
        

    print('projected MPS',".."*30)
 
    return psi_halfinf,charge



def patch_WF_together(psi1,psi2,sites):

    """
    patches psi1 and psi2 together
    """
    #ADD THREE SITES WITH DIFFERENT K
    #adds extra sites again
   
    
    #print(a._B[1])
    #print(a._B[2])
    #print(psi_halfinf._B[0])
    #print(len(psi_halfinf._B))
    Bflat=[]
    Ss=[]
    Ss1=psi1._S
    Ss2=psi2._S

    for i in range(len(psi1._B)):
        #print('bla'*20)
        Ss.append(Ss1[i])
        Bflat.append(np.transpose(psi1._B[i].to_ndarray(),(1,0,2)))
    #print(Bflat[3].shape)
    print(psi1._B[-1])
    print(psi2._B[0])
    
    #print(len(Ss[3]))
    #Ss.append(Ss1[-1])
    for i in range(len(psi2._B)):
        #print('bla'*20)
        Ss.append(Ss2[i])
        Bflat.append(np.transpose(psi2._B[i].to_ndarray(),(1,0,2)))
    Ss.append(Ss2[-1])

    print(len(Ss))
    print(len(Bflat))
   

    #TODO: READ qflat from somewhere
    #qflat=[[-6-len(pstate),0]] #if empty,empty,full
    #qflat=[[-6+len(pstate),0]] #if full,empty,empty
    #qflat=[[-6,0]] #if 010
    #chargeinfo=sites[0].leg.chinfo
    #left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
    
    

    left_leg=psi1._B[0].get_leg('vL')
    #print(len(sites))
    psi=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    print(psi._B[0])

    print('Patched two wavefunctions together sucessfully')
    return psi


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
        #print('wflat'*100)
        #print(qflat_list_c[0])
        

        
    else:
        print(loaded_xxxx.keys())
        
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

    with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/laughlin_3_again/"+name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    #IF LOADED FROM H5 FILE UNCOMMENT

    #loaded_xxxx=hdfdict.load("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/"+name+'.h5')
    
    print(loaded_xxxx.keys())
    #print(loaded_xxxx['MPS_RHS_qflat'])
    
    #print(loaded_xxxx['graph'][0])
    
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
    print(environment)
    print("vacuum left environment is loaded",'..'*20)
    print(environment.qtotal)
    
    return environment



def load_right_environment_again(loaded_xxxx,permute,leg_HMPO,leg_MPS):
  
    """
    produces left environment which corresponds to vacuum
    so far works only it total charge is zero
    leg_HMPO,leg_MPS: give the legs of MPO and MPS
    """

    Bflat=loaded_xxxx['RP']
    
    
    #change the shape of Bflat and qflat so that it is consistent with new tenpy
    Bflat=np.transpose(Bflat, (1, 0, 2))

    #permute to be consistent with MPO
    Bflat=Bflat[:,permute,:]
    duljina_MPS=len(leg_MPS.to_qflat())
    duljina=len(leg_HMPO.to_qflat())


    labels=['vL', 'wL', 'vL*']


    
    #gives qflat for non-trivial leg    
    legcharges=[]
    legcharges.append(leg_MPS.conj())
    legcharges.append(leg_HMPO.conj())
    legcharges.append(leg_MPS)
  

    
    #set data flat
 
    
    #find the first index at which charges are zero?
    #qflat=leg_HMPO.conj().to_qflat()
  
  
    
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
    print("right environment is loaded",'..'*20)
    print(environment.qtotal)
    #quit()
    return environment

def set_right_environment_to_vacuum(leg_HMPO,leg_MPS):
  
    """
    produces left environment which corresponds to vacuum
    so far works only it total charge is zero
    leg_HMPO,leg_MPS: give the legs of MPO and MPS
    """
    duljina_MPS=len(leg_MPS.to_qflat())
    duljina=len(leg_HMPO.to_qflat())


    labels=['vL*', 'wL', 'vL']


    
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
    print(environment)
    print("vacuum right environment is loaded",'..'*20)
    print(environment.qtotal)
    
    return environment


def load_environments_from_file(name,name_load,side='right'):
    file_path="/mnt/users/dperkovic/quantum_hall_dmrg/data_load/laughlin_2_again/"+name+'.npz'
    data =np.load(file_path,allow_pickle=True)
    
    file_path="/mnt/users/dperkovic/quantum_hall_dmrg/data_load/laughlin_3_again/"+name_load+'.pkl'
    with open(file_path, 'rb') as f:
        loaded_qs= pickle.load(f, encoding='latin1')
    
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
    
    print(len(loaded_xxxx['indices']))
    #print(loaded_xxxx['permutations_R'])
    
    G_old,basis_old,permutation_old=loaded_xxxx['graph'],loaded_xxxx['indices'][::-1],loaded_xxxx['permutations'][:2][::-1]
    #print(loaded_xxxx['permutations_R'])
    
    print(len(basis_old))
    #print(G_old[0].keys())
    #print(basis_old)
    
    
    #print(basis_old)
    
    
    
    old_basis=get_old_basis(G_old,basis_old,permutation_old)
    for i in range(len(old_basis)):
        assert len(old_basis[i])==len(ordered_states[i])
    print(len(basis_old))
    print(len(old_basis))
    #total_permutation=[]
    #print()
    
    #print(old_basis)
    print(old_basis)
    print(ordered_states[0])
    
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
    
    sanity_check_permutation(loaded_xxxx,new_MPO, permutation[0],permutation[1])
    #GIVES THE CORRECT PERMUTATION of old basis to get a new basis!!
    return permutation
def sanity_check_permutation(loaded_xxxx,new_MPO, permute,permute2):

    print('checking sanity check of permutation...')
    #permute old Bflat and check its the same as new Bflat for hamiltonian
    print(loaded_xxxx.keys())
   
    Bflat_old=loaded_xxxx['MPO_B'][0]
    #print(Bflat_old.shape)

    print(Bflat_old.shape)
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
                print('x'*50)
                print(j,i)
                print(Bflat_old[j,i])
            
                print(B_new[j,i])
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

    with open("/mnt/users/dperkovic/quantum_hall_dmrg/data_load/laughlin_2_again/"+name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    return loaded_xxxx
    

def add_cells_to_projected_wf(psi_halfinf,pstate,sites,charge=[0,0,0]):
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
 
   
    qflat=[[0,0,0]]
    #qflat=[[-16,0]]
    #qflat=[[-24,0]]
    chargeinfo=sites[0].leg.chinfo
    left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
    
   
    print(psi_halfinf._B[0])
    psi=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)

    
    leg=psi._B[len(pstate)].get_leg('vL')
    charge_new=leg.charges[0]
    charge_diff=[]
    
    for i in range(len(charge)):
        charge_diff.append(-charge_new[i]+charge[i])
    

    left_leg=LegCharge.from_qflat(chargeinfo,[charge_diff],qconj=1).bunch()[1]
    psi=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    
    print(charge)
    print(psi._B[len(pstate)])
    
    return psi,charge_diff




def bulk_vacuum_boundary(name_load,name_save,name_graph,pstate=[]):
    model_par,conserve,root_config_,loaded_xxxx=load_param(name_load)
    #TODO: COMMENT OUT STUFF
   
    #graph is fine
    print(loaded_xxxx['MPO_B'][0].shape)
    
    graph=loaded_xxxx['graph']#load_graph(name_graph)
    
    
    print(len(graph))
    
    
    L=3*6+1
    L=value
    #assert (2*L)%6==2
   
    graph=[graph[0],graph[1]]*L
    print(len(graph))
    L=len(graph)#doubles L
    

   

    #length of the system is L,
    #half is L//2
 

    #model_par.pop('mpo_boundary_conditions')


    model_par['boundary_conditions']= ('infinite', L)
    print('__'*100)
    print(model_par)

   

    LL=0
    model_par['layers']=[ ('L', LL) ]
    
    
    #INSERT UNIT CELL IN BETWEEN!
    M,sites,ordered_states=create_segment_DMRG_model(model_par,L,root_config_,conserve,graph)


    perm=load_permutation_of_basis(loaded_xxxx,ordered_states,M.H_MPO._W[0])
    print('AAAAA')
    print(perm)
    
    print(len(pstate))
    #psi_halfinf=load_data(loaded_xxxx,sites[len(pstate):])
   
    psi_halfinf_right=load_data(loaded_xxxx,sites[len(pstate):],shift=len(pstate)//2,side='right')
    print(len(psi_halfinf_right._B))
    print('psi_OLD_OLD'*100)
    
    print(psi_halfinf_right._B[-1])
    
    print("MPMPMPMPMPMPMPPMP")
    print(psi_halfinf_right)


    filling=psi_halfinf_right.expectation_value("nOp")
    N_orig=np.sum(filling)
    psi_halfinf_right,charge2=project_side_of_mps(psi_halfinf_right,side='left')
    #print(charge2)
    #quit()
    
    #print(psi_halfinf_right._B[0])
    #print(psi_halfinf_right._B[-3])
    #print(psi_halfinf_right._B[-4])
    #THIS ONE HAS BOTH N,K conservation
   
   
    #THIS ONE HAS BOTH N,K conservation
    if len(pstate)>0:
        print('ENTER AND IT SHOULD')
        psi_halfinf_right,charge2=add_cells_to_projected_wf(psi_halfinf_right,pstate,sites,charge=charge2)
    
    
    print('left_edge_print')
  
    print(psi_halfinf_right._B[-1])
    

   
    filling=psi_halfinf_right.expectation_value("nOp")
    N=np.sum(filling)
    print('N,K')
    print('projected',N)
    print('unprojected', N_orig)
    print(np.sum(filling*0+1/3))
    #quit()
 
    #psi_halfinf.add_B(-2,a._B[1])
    #psi_halfinf.add_B(-3,a._B[0])
    #psi_total.canonical_form_finite(cutoff=0.0)
    #print(len(psi_halfinf._B))
    

    #print('psi'*100)
    #print(psi_total._B[-1])
    #print(psi_halfinf_right._B[-1])
  
    print(len(sites))
    #environment LOAD
    name='Environment_R_101_root_conf'
    #print("loading right env")
    data=load_environments_from_file(name,name_load,side='right')

    #print(M.H_MPO._W[-1])
    #MAKE SURE YOU USE RIGHT PERMUTATION
    print('psi_new'*100)
    print(psi_halfinf_right._B[-1])
    leg_MPS=psi_halfinf_right._B[-1].get_leg('vR')
    #leg_MPO
    leg_MPO=M.H_MPO._W[-1].get_leg('wR')
    #right_env=load_environment(data,(len(sites))//2-1,root_config_,conserve, perm[0],side='right',old=True)
    #print(right_env)
    #quit()
    right_env=load_right_environment_again(data,perm[0],leg_MPO,leg_MPS)
    #quit()
    print('right_env'*100)
    #print(right_env)
    
    #
   



    #leg_MPS
    #leg_MPS=psi_total._B[0].get_leg('vL')
    #leg_MPO
    #leg_MPO=M.H_MPO._W[0].get_leg('wL')
    #name='Environment_L'
    #data=load_environments_from_file(name,name_load,side='left')
    #leg_MPS
    leg_MPS=psi_halfinf_right._B[0].get_leg('vL')
    #leg_MPO
    leg_MPO=M.H_MPO._W[0].get_leg('wL')
    left_environment=set_left_environment_to_vacuum(leg_MPO,leg_MPS)
    #print(M.H_MPO._W[0])
    #print('psi'*100)
    
    #print(psi_halfinf_left._B[1])
    #print(psi_halfinf_left._B[2])
    #print(psi_halfinf_left._B[3])
    #print(psi_halfinf_left._B[4])
    #print(psi_halfinf_left._B[5])
    #print(psi_total._B[0])
    #print('leftenv'*100)
    #print(left_environment)
    
    

    init_env_data_halfinf={}
    #print('*'*100)
    #print(M.H_MPO._W[0])
    #print(right_env)
    #print(psi_halfinf_right._B[-1])
    init_env_data_halfinf['init_LP'] = left_environment    #DEFINE LEFT ENVIROMENT
    init_env_data_halfinf['age_LP'] =0

    init_env_data_halfinf['init_RP'] = right_env   #DEFINE RIGHT ENVIROMENT
    init_env_data_halfinf['age_RP'] =0

    print('N,K')
    print(N)
    print(np.sum(filling*0+1/3))

    print("Dipole")

    print(np.sum((filling-1/3)*(np.arange(len(filling))//2)))

    dmrg_params = {
        'mixer': True,
         'mixer_params': {
        'amplitude':amp,
        'decay': decay,
        'disable_after': 15,
    },
        'max_E_err': 1.e-9,
        'max_S_err': 1.e-6,
        'trunc_params': {
            'chi_max': 2500,
            'svd_min': 1.e-8,
        },
        'max_sweeps': max_sweeps
    }

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
    print("Dipole")
    print(np.sum((filling-1/3)*(np.arange(len(filling))//2))*8*np.pi**2/18**2)

    print('N,K')
    print(np.sum(filling))
    print(np.sum(filling*0+1/3))
    E_spec=psi.entanglement_spectrum()
    #print('entanglement spectrum:',E_spec)



    EE=psi.entanglement_entropy()
    print('entanglement entropy:',EE)


    

    #data = {"psi": psi,  # e.g. an MPS
    #        "dmrg_params":dmrg_params, "model_par":model_par, "model": M,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

    data = { "dmrg_params":dmrg_params,"energy":E0, "model_par":model_par,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

    #with open("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/"+name_save+".pickle", 'wb') as f:
    #    pickle.dump( data,f)
  

   
    with h5py.File("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/multilayer_laughlin_2_3/"+name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)


def bulk_vacuum_boundary_opposite_side(name_load,name_save,name_graph,pstate=[]):
    model_par,conserve,root_config_,loaded_xxxx=load_param(name_load)
    #TODO: COMMENT OUT STUFF
   
    #graph is fine
    print(loaded_xxxx['MPO_B'][0].shape)
    
    graph=loaded_xxxx['graph']#load_graph(name_graph)
    
    
    print(len(graph))
    
    L=4*20-3
    
    L=6*5
    #assert (2*L)%6==2
   
    graph=[graph[0],graph[1]]*L
    print(len(graph))
    L=len(graph)#doubles L
    

   

    #length of the system is L,
    #half is L//2
 

    #model_par.pop('mpo_boundary_conditions')


    model_par['boundary_conditions']= ('infinite', L)
    print('__'*100)
    print(model_par)

   

    LL=0
    model_par['layers']=[ ('L', LL) ]
    
    
    #INSERT UNIT CELL IN BETWEEN!
    M,sites,ordered_states=create_segment_DMRG_model(model_par,L,root_config_,conserve,graph)


    perm=load_permutation_of_basis(loaded_xxxx,ordered_states,M.H_MPO._W[0])
    print('AAAAA')
    print(perm)
    
    print(len(pstate))
    #psi_halfinf=load_data(loaded_xxxx,sites[len(pstate):])
   
    psi_halfinf_right=load_data(loaded_xxxx,sites[len(pstate):],shift=len(pstate),side='left')
    
    print(len(psi_halfinf_right._B))
    print('psi_OLD_OLD'*100)
    
    print(psi_halfinf_right._B[0])
    
    #psi_halfinf_right,charge2=project_side_of_mps(psi_halfinf_right,side='right')
    #print(charge2)
    
    #print(psi_halfinf_right._B[0])
    #print(psi_halfinf_right._B[-3])
    #print(psi_halfinf_right._B[-4])
    #THIS ONE HAS BOTH N,K conservation
   
   
    #THIS ONE HAS BOTH N,K conservation
    if len(pstate)>0:
        psi_halfinf_right,charge2=add_cells_to_projected_wf(psi_halfinf_right,pstate,sites,charge=charge2)

    
    print('left_edge_print')
  
    print(psi_halfinf_right._B[0])
    

   

    #psi_halfinf.add_B(-2,a._B[1])
    #psi_halfinf.add_B(-3,a._B[0])
    #psi_total.canonical_form_finite(cutoff=0.0)
    #print(len(psi_halfinf._B))
    

    print('psi'*100)
    #print(psi_total._B[-1])
    print(psi_halfinf_right._B[0])
  
    print(len(sites))
    #environment LOAD
    name='Environment_R1'
    print("loading right env")
    #data=load_environments_from_file(name,name_load,side='right')

    #print(M.H_MPO._W[-1])
    #MAKE SURE YOU USE RIGHT PERMUTATION
    #right_env=load_environment(data,(len(sites))//2-1,root_config_,conserve, perm[0],side='right',old=True)
    print('right_env'*100)
    #print(right_env)
    
    #
   
    # WHY DOES IT NOT WORKKKKKK
    #for i in range(100):
    #    print(psi_halfinf_right.get_SR(i)>10**(-8))
    

    #leg_MPS
    #leg_MPS=psi_total._B[0].get_leg('vL')
    #leg_MPO
    #leg_MPO=M.H_MPO._W[0].get_leg('wL')
    name='Environment_L'
    data=load_environments_from_file(name,name_load,side='left')
    #leg_MPS
    left_environment=load_environment(data,-1,root_config_,conserve, perm[0],side='left',old=True)
    leg_MPS=psi_halfinf_right._B[-1].get_leg('vR')
    #leg_MPO
    leg_MPO=M.H_MPO._W[-1].get_leg('wR')
    right_env=set_right_environment_to_vacuum(leg_MPO,leg_MPS)
    print(M.H_MPO._W[0])
    print('psi'*100)

    
    #print(psi_halfinf_left._B[1])
    #print(psi_halfinf_left._B[2])
    #print(psi_halfinf_left._B[3])
    #print(psi_halfinf_left._B[4])
    #print(psi_halfinf_left._B[5])
    #print(psi_total._B[0])
    print('leftenv'*100)
    print(left_environment)
    
    

    init_env_data_halfinf={}


    init_env_data_halfinf['init_LP'] = left_environment    #DEFINE LEFT ENVIROMENT
    init_env_data_halfinf['age_LP'] =0

    init_env_data_halfinf['init_RP'] = right_env   #DEFINE RIGHT ENVIROMENT
    init_env_data_halfinf['age_RP'] =0




    dmrg_params = {
        'mixer': True,
         'mixer_params': {
        'amplitude': 1.e-5,
        'decay': 1.,
        'disable_after': 5,
    },
        'max_E_err': 1.e-8,
        'max_S_err': 1.e-5,
        'trunc_params': {
            'chi_max': 2000,
            'svd_min': 1.e-7,
        },
        'max_sweeps': 50
    }

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


    E_spec=psi.entanglement_spectrum()
    #print('entanglement spectrum:',E_spec)



    EE=psi.entanglement_entropy()
    print('entanglement entropy:',EE)


    

    #data = {"psi": psi,  # e.g. an MPS
    #        "dmrg_params":dmrg_params, "model_par":model_par, "model": M,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

    data = { "dmrg_params":dmrg_params,"energy":E0, "model_par":model_par,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec }

    #with open("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/"+name_save+".pickle", 'wb') as f:
    #    pickle.dump( data,f)
  

   
    with h5py.File("/mnt/users/dperkovic/quantum_hall_dmrg/segment_data/multilayer_1_3x2/"+name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)



#name_graph=str(sys.argv[1])
#name_load=str(sys.argv[2])
#print(name_graph)
#L
value=int(sys.argv[1])
index_of_ss=int(sys.argv[2])
pstate=str(sys.argv[3])
amp=float(sys.argv[4])
decay=float(sys.argv[5])
max_sweeps=50
if pstate=='[]':
    pstate=[]
else:
    pstate=pstate.split(',')
#print(pstate)

print(amp)
#name_load='DMRG_18.0_Haldane_barrier_0.01_mu_2_3_'
name_load='DMRG_18.0_Haldane_barrier_0.01_mu_2_3_'
#name_load='Lx_16_QH_nu_1_3'
name_graph='[]'
name_load='DMRG_18.0_Haldane_barrier_0.0_mu_1_3_x2_101_root_conf'
#name_load='DMRG_14.0_Haldane_barrier_0.0_mu_1_3_x2_'
#name_save='BETTER_AS_SMALLER_1_3x2_L='+str(value)+'_1_3_1_3_vacuum'+name_load+'_'+str(pstate[:6])+'_num='+str(len(pstate))
name_save='101_root_conf_with_index='+str(index_of_ss)+'_MIXER_LONG_CONVERGENCE_AMP='+str(amp)+'_decay_'+str(decay)+'_L='+str(value)+'_sweeps_'+str(max_sweeps)+'_18_projected_TRY_2_3_DENSITY_CHECK_ADDED_'+str(pstate[:6])+'_num='+str(len(pstate))

#bulk_bulk_boundary(name_load,name_save,name_graph,pstate=pstate)
#bulk_vacuum_boundary(name_load,name_save,name_graph,pstate=pstate)
bulk_vacuum_boundary_opposite_side(name_load,name_save,name_graph,pstate=pstate)
quit()
