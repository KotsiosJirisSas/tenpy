#!/usr/bin/env python
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy') 
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


def create_infinite_DMRG_model(N):
    np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)
    #########################
    #sort the problem and match values to konstantinos values
    Lx = 14;            # circumference
    LL = 0;         # which Landau level to put in
    mixing_chi = 400; #Bond dimension in initial sweeps
    chi = 400;      #Bond dimension of MPS
    chi2 = 400
    chi3 = 400
    #chi4 = 4500
    xi = 1; #
    #xi = 1;            # The Gaussian falloff for the Coulomb potential
    Veps = 1e-4 # how accurate to approximate the MPO
    V = { 'eps':Veps, 'xiK':xi, 'GaussianCoulomb': {('L','L'):{'v':1, 'xi':xi}} }
    root_config = np.array([0, 1, 0])       # this is how the initial wavefunction looks
    N = 3
    model_par = {
        'verbose': 3,
        'layers': [ ('L', LL) ],
        'Lx': Lx,
        'Vs': V,
        'boundary_conditions': ('infinite', N),#????????????????????????
    #   'boundary_conditions': ('periodic', 1),
        'cons_C': 'total', #Conserve number for each species (only one here!)
        'cons_K': False, #Conserve K
        'root_config': root_config, #Uses this to figure out charge assignments
        'exp_approx': '1in', #For multiple orbitals, 'slycot' is more efficient; but for 1 orbital, Roger's handmade code '1in' is slightly more efficient
    }
        









    print("Start model in Old Tenpy",".."*10)
    M = mod.QH_model(model_par)
    print("Old code finished producing MPO graph",".."*10)


    G=M.MPOgraph
    #import pickle
    #print(G)
    # Assuming 'data' is your object (like your transposed tensor) that you want to save
    #with open('data_MPO_GRAPH.pkl', 'wb') as f:
    #    pickle.dump(G[0], f)

    #quit()
    #print(G[0].keys())
    #print(G[0][('_a', 0, 9)])
    #quit()
    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
    #print(G_new[0][('_a', 0, 9)])
    #print(G_new[0]["('_a', 0, 9)"])
    #quit()
    root_config_ = np.array([0,1,0])
    root_config_ = root_config_.reshape(3,1)
    spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
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
    print(not_included_couplings)
    for i in range(L):
       
        for element in not_included_couplings[i]:
            
            #G_new[i].pop(element[0],None)
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
    
    #bunch_legs=1
    H.sort_legcharges()
    #print(H._W[0])
    #quit()
    #initialize wavefunction as MPS
   
    print(N)
    lattice=Chain(N,spin, bc="periodic",  bc_MPS="infinite")
    
    model=MPOModel(lattice, H)

    print("created model",".."*30)
    #quit()
    return model

def load_data(name):
    #LOADS MPS
    with open(name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    #print(loaded_xxxx.keys())
    Bflat=loaded_xxxx['Bs']

    Ss=loaded_xxxx['Ss']
    qflat2=loaded_xxxx['qflat']
    #print(qflat2.shape)
    #quit()
    qflat=[]
    for i in range(len(qflat2)):
        qflat.append(qflat2[i][0])
    qflat=np.array(qflat)
    #print(qflat)
    #quit()
    root_config_ = np.array([0,1,0])
    root_config_ = root_config_.reshape(3,1)
    spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
    
    L=3
    sites = [spin] * L
    #broj=1
    #pstate=["empty", "full","empty"]*broj

    Ss=[Ss[2],Ss[0],Ss[1]]
    
    #chargeinfo=ChargeInfo([1],['N'])
    chargeinfo=sites[0].leg.chinfo
    #print(chargeinfo)
    #quit()
    left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
    #a=left_leg.bunch()
    #print(a)
    #print(left_leg)
    #quit()
    left_leg.sort()
    #LegCharge(chargeinfo, slices, charges, qconj=1)
    #psi=MPS.from_product_state(sites, pstate,bc='infinite')
    #print(psi._B)
    mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='infinite',legL=left_leg)
    print(mps)
    print('loaded mps from data',".."*30)
    return mps

def load_environment(name):
    #LOAD ENVIRONMENT
    print("loading environment",'..'*20)
    with open(name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    Bflat=loaded_xxxx['RP_B']
    qflat_list_c=loaded_xxxx['RP_q']
   
    print(Bflat.shape)
    shape=Bflat.shape
   
    #change the shape of Bflat and qflat
    Bflat=np.transpose(Bflat, (1, 0, 2))
    print(len(qflat_list_c))
    qflat_list=[qflat_list_c[1],qflat_list_c[0],qflat_list_c[2]]
    


    #define site that defines charge information for environment
    root_config_ = np.array([0,1,0])
    root_config_ = root_config_.reshape(3,1)
    length=2
    #site=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=('N','K'),site_loc=length)
    site=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
    chargeinfo=site.leg.chinfo
    legcharges=[]
    
    #loads right environment
    labels=['vL', 'wL', 'vL*']
    conj_q=[1,1,-1]

    for i in range(len(qflat_list)):
        #print(i)
        legcharge=LegCharge.from_qflat(chargeinfo,qflat_list[i],qconj=conj_q[i]).bunch()[1]
        legcharges.append(legcharge)


    
    
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

def project_and_find_segment_mps(mps,N):

    print('projecting MPS',".."*30)
    mps2=mps.extract_segment(0, N)
    psi_halfinf = mps2.copy()  # the oringinal MPS

    #EXTRACT A SEGMENT
    S = mps2.get_SL(0)
    proj = np.zeros(len(S), bool)
    proj[np.argmax(S)] = True
    B = psi_halfinf.get_B(0, form='B')
    B.iproject(proj, 'vL')
    psi_halfinf.set_B(0, B, form='B')
    psi_halfinf.set_SL(0, np.ones(1, float))
    psi_halfinf.canonical_form_finite()
    psi_halfinf.test_sanity()

    print('projected MPS',".."*30)
    #B=psi_halfinf.get_B(0)
    #print(B)
    #quit()
    return psi_halfinf





def set_left_environment_projected(psi0_i,init_env_data,H_MPO,leg):

    #THIS ASSUMES ENVIRONMENT HAS TOTAL CHARGE OF 0!!


    print('setting left environment',".."*30)
    #starting from infinite MPS and its enviroment and MPO define enviroment for segment DMRG
    init_env_data_halfinf = init_env_data.copy()
   

    #unprojected vacuum environment
    init_env_data_halfinf['init_LP'] = MPOEnvironment(psi0_i, H_MPO, psi0_i).init_LP(0, 0)#[4,:,4]
    Bflat_big=init_env_data_halfinf['init_LP'].to_ndarray()
    #CALCULATES NON-ZERO INDICES
    indexi_non_zero=np.array(np.where( Bflat_big==1))
    #print(indexi_non_zero)
    #since it is projected onto single state, we take 0,0 legs
    #keep track of legs with charges 1
    #keep only 0,0 on leftmost, right most leg, and all indices at MPO leg
    #it turns out that typically in Bflat only one index is occupied,
    #and that is first index with zero charge
    keep_only_one_index_left_leg=np.where(indexi_non_zero[0]==0)[0]
    keep_only_one_index_right_one=np.where(indexi_non_zero[2]==0)[0]
    intersection = np.intersect1d( keep_only_one_index_left_leg, keep_only_one_index_right_one)
  
    indexi_non_zero_new=[]
    for i in range(3):
        legara=[]
        for element in intersection:
            legara.append(indexi_non_zero[i,element])
        indexi_non_zero_new.append(legara)
    indexi_non_zero_new=tuple(indexi_non_zero_new)
    
    print(indexi_non_zero_new)
    print("left environment")
    print(init_env_data_halfinf['init_LP'])

    middle_leg=init_env_data_halfinf['init_LP'].get_leg('wR')
    duljina_MPS=len(leg.to_qflat())
    duljina=len(middle_leg.to_qflat())


    labels=['vR*', 'wR', 'vR']


    
    #gives qflat for non-trivial leg    
    legcharges=[]
    legcharges.append(leg)
    legcharges.append(middle_leg)
    legcharges.append(leg.conj())
  

    #set data flat
    data_flat=np.zeros(duljina*duljina_MPS*duljina_MPS)
    data_flat=np.reshape(data_flat,(duljina_MPS,duljina,duljina_MPS))

    #set relevant indices to non zero values
    print(indexi_non_zero_new)
    data_flat[indexi_non_zero_new]=1
   
        
    array_defined=Array.from_ndarray( data_flat,
                        legcharges,
                        dtype=np.float64,
                        qtotal=None,
                        cutoff=None,
                        labels=labels,
                        raise_wrong_sector=True,
                        warn_wrong_sector=True)
    
    
    
    init_env_data_halfinf['init_LP'] =array_defined
    init_env_data_halfinf['age_LP'] = 0


    print('setleft environment',".."*30)
   
    print(array_defined)
    print(array_defined.qtotal)
  
    #TODO: SET THAT PROJECTED STUFF IS INDEED VACUUM
    return init_env_data_halfinf

def set_infinite_like_segment(mps,M_i,last):
    mps2=mps.extract_segment(0,last)
    psi_halfinf = mps2.copy() 

    psi_halfinf.canonical_form_finite(cutoff=0.0)
    init_env_data_halfinf={}
    #initialize right enviroment
    #INSTEAD OF 44 SHOULD JUST BE A LAST ELEMENT?
    #what is the second parameter???

    init_env_data_halfinf['init_RP'] = MPOEnvironment(mps, M_i.H_MPO, mps).init_RP(last, 0)    #DEFINE RIGHT ENVIROMENT
    init_env_data_halfinf['age_RP'] =0
    
    init_env_data_halfinf['init_LP'] = MPOEnvironment(mps,  M_i.H_MPO, mps).init_LP(0, 0)#[4,:,4]

    init_env_data_halfinf['age_LP'] = 0
    return psi_halfinf,init_env_data_halfinf

#name="qflat_QH_1_3-2"
#load_data(name)
#quit()
N=3
M_i=create_infinite_DMRG_model(N)

name='env_QH_1_3_no_K'
right_env=load_environment(name)
#print(right_env)
#quit()

name="qflat_QH_1_3_no_K"

#returns infinite DMRG MPS
mps=load_data(name)
#print(mps)
#a=mps.expectation_value('nOp')
#print(a)
#quit()
#quit()
#print(mps._B[2])
#quit()


root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
#spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')

spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
sites=[spin]*3
pstate=["empty", "full","empty"]
#mps = MPS.from_product_state(sites, pstate, bc="infinite")


N=15


#DEFINE MPO MODEL M_i for infinite DMRG AND TURN IT INTO A SEGMENT
first=0
last=N*3-2
M_s = M_i.extract_segment(first, last )#enlarge=N)

first, last = M_s.lat.segment_first_last
#print(last)
#quit()

#MULTIPLY BY CELL SIZE


mps2=mps.extract_segment(0,last)
psi_halfinf = mps2.copy() 

print(psi_halfinf)

psi_halfinf=project_and_find_segment_mps(mps,last)
psi_halfinf.canonical_form_finite(cutoff=0.0)
init_env_data_halfinf={}
#initialize right enviroment
#INSTEAD OF 44 SHOULD JUST BE A LAST ELEMENT?
#what is the second parameter???
print('projected charge')
print(psi_halfinf._B[0].qtotal)
print(M_s.H_MPO._W[0].qtotal)
#quit()
age=50

#init_env_data_halfinf['init_LP']= MPOEnvironment(mps, M_i.H_MPO, mps).init_LP(0, 0)#set_left_environment(mps,init_env_data_halfinf,M_i.H_MPO)


#print(init_env_data_halfinf['init_LP'])
#quit()
init_env_data_halfinf['init_RP'] = right_env   #DEFINE RIGHT ENVIROMENT

#print(right_env)
#quit()
leg=psi_halfinf._B[0].get_leg('vL')
init_env_data_halfinf=set_left_environment_projected(mps,init_env_data_halfinf,M_i.H_MPO,leg)
init_env_data_halfinf['init_RP'] = MPOEnvironment(mps, M_i.H_MPO, mps).init_RP(last, age)    #DEFINE RIGHT ENVIROMENT
print("bababa")
print(right_env)

#quit()

print(init_env_data_halfinf['init_RP'])
init_env_data_halfinf['age_RP'] =0
init_env_data_halfinf['age_LP'] =0
#quit()
#print(init_env_data_halfinf['init_RP'])
#quit()
#initialize left enviroment




dmrg_params = {
    'mixer': True,
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 550,
        'svd_min': 1.e-10,
    },
}
#print()
#print(len(M_s))
#print(M_s.lat.site)
#quit()
#print(M_s.H_MPO.L)
#print(psi_halfinf)
#quit()
#OK DOESNT WORK FOR PROJECTED
#WORKS FOR UNPROJECTED WF

for i in range(1):
    x=psi_halfinf._B[i].qtotal
    print(x)
    b=M_i.H_MPO._W[i].qtotal
    print(b)
    a=right_env.qtotal
    print(a)
    c=init_env_data_halfinf['init_LP'].qtotal
    print(c)

#quit()

eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_halfinf, M_s, dmrg_params,
                                     resume_data={'init_env_data': init_env_data_halfinf})


print("enviroment works")
print("running DMRG")
#quit()
E,psi=eng_halfinf.run()
filling=psi.expectation_value("nOp")
print(filling)
