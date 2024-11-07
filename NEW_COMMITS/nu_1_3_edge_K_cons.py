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


def create_segment_DMRG_model(L):
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

    model_par = {
        'verbose': 3,
        'layers': [ ('L', LL) ],
        'Lx': Lx,
        'Vs': V,
        'boundary_conditions': ('infinite', L),#????????????????????????
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
    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
    
    root_config_ = np.array([0,1,0])
    root_config_ = root_config_.reshape(3,1)


    root_config_ = np.array([0,1,0])
    root_config_ = root_config_.reshape(3,1)


    sites=[]
    for i in range(L):
      
        spin=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=('N','K'),site_loc=i)
        #print(spin.Id)
        #quit()
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


    print(not_included_couplings)
    for i in range(L):
       
        for element in not_included_couplings[i]:
            
            G_new[i].pop(element[0],None)
            print(element[0])


    print("Test sanity"+".."*10)
    M.test_sanity()
    M.graph = G_new #: INppuut the graph in the model 
    print("Test passed!"+".."*10)
    grids =M._build_grids()#:Build the grids from the graph
    print("Building MPO"+".."*10)


    H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
    print("Built"+".."*10)


    H.sort_legcharges()
    #initialize wavefunction as MPS

    
    pos= [[i] for i in range(L)]

    #quit()
    lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="segment")
    x=lattice.mps_sites()
        
    model=MPOModel(lattice, H)
   

    print("created model",".."*30)
    return model,sites

def load_data(name,sites):
    #name='qflat_QH_1_3-2'
 
    with open(name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    #print(loaded_xxxx['qflat'])
    #quit()
    number=len(sites)//3+1
    #print(number)
    Bflat=loaded_xxxx['Bs']*number
    #print(len(Bflat))
    Ss=loaded_xxxx['Ss']
    qflat2=loaded_xxxx['qflat']
    #quit()
    print(len(Ss))
    print(len(Bflat))
    #DEFINE K-charge on the left boundary
    #ADD K AND N conservatrions
    qflat=[]
    for i in range(len(qflat2)):
        kopy=[]
        for m in range(len(qflat2[i])):
            kopy.append(qflat2[i][m])
        qflat.append(kopy)
    qflat=np.array(qflat)
    #print(qflat)
    #quit()
    #ALSO NEED CHARGE VALUES?
    #print(qflat)
    #quit()

    #SINCE CONVERTING IT DIRECTLY TO SEGMENT MPS WE NEED TO ADD SINGULAR VALUES FOR THE FINAL RIGHTMOST LEG AS WELL!

    Ss=[Ss[2],Ss[0],Ss[1]]*number
    Ss.append(Ss[0])
    
    Bflat=Bflat[:L]
    Ss=Ss[:L+1]
    
    chargeinfo=sites[0].leg.chinfo
    #print(chargeinfo)
    #quit()
    left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
    
    mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
    print('loaded mps from data',".."*30)
    return mps


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
    B=psi_halfinf.get_B(0)
    #print(B)
    #quit()
    return psi_halfinf


def set_left_environment(psi0_i,init_env_data,H_MPO):

    print('setting left environment',".."*30)
    #starting from infinite MPS and its enviroment and MPO define enviroment for segment DMRG
    init_env_data_halfinf = init_env_data.copy()
   
    init_env_data_halfinf['init_LP'] = MPOEnvironment(psi0_i, H_MPO, psi0_i).init_LP(0, 0)#[4,:,4]
    print("IN"*100)
    labels=['vR*', 'wR', 'vR']


    #obtain qflat of nontrivial stuff
    label='wR'
    print(init_env_data_halfinf['init_LP'])
    #quit()
    #get qflat from here
    a=np.array(init_env_data_halfinf['init_LP'].get_leg( label))
    a=str(a)
    words =a.split('\n')
    #STRIP TO GET QFLAT
    words.pop(0)
    kopy=[]
    for i in words:
        kopy.append(i[:-2])
    #print(kopy)
    kopy2=[]
    for i in kopy:
        kopy2.append(i.split(' '))
    #print(kopy2)
    kopy2.pop(-1)
    
    kopy3=[]
    for i in kopy2:
        kopy3.append(int(i[-1].strip('[')))
    
    #gives qflat for non-trivial leg    
    #print(len(kopy3))    
    #quit()
    #a=init_env_data_halfinf['init_LP'].to_ndarray()
    #a= init_env_data_halfinf['init_LP'].__iter__()[0]
    #print(a.shape)
   
    #print(a.shape)
    #quit()
    #EXTRACT CHARGE INFORMATION FROM THIS MPO ENVIROMENT AND THEN SET THE CORRESPONDING CHARGE INFO TO
    #LEFT BOUNDARY
  
    #chargeinfo=ChargeInfo([1],['N'])
    #print(init_env_data_halfinf['init_LP'])
    #quit()
    chargeinfo=init_env_data_halfinf['init_LP'].legs[0].chinfo
    

    legcharges=[]
    #define 3 legs
    qflat=[6]
    legcharge=LegCharge.from_qflat(chargeinfo,qflat,qconj=1)#.bunch()[1]
    legcharges.append(legcharge)
    #READ OFF FROM [:]
    qflat=kopy3
    legcharge=LegCharge.from_qflat(chargeinfo,qflat,qconj=-1)#.bunch()[1]

    legcharges.append(legcharge)
    qflat=[6]
    legcharge=LegCharge.from_qflat(chargeinfo,qflat,qconj=-1)#.bunch()[1]
    legcharges.append(legcharge)
    #print(legcharges[1])
    #print('charges fine')
    #quit()
    #quit()

    #labels=['vR*', 'wR', 'vR']

    #HOW DO YOU GET DATA_FLAT (WHERE DO YOU GET IT FROM???)
    data_flat=np.zeros(len(kopy3))
    data_flat[0]=1
    data_flat=np.reshape(data_flat,(1,60,1))
    #print(data_flat)
    #quit()
    
    array_defined=Array.from_ndarray( data_flat,
                        legcharges,
                        dtype=np.float64,
                        qtotal=None,
                        cutoff=None,
                        labels=labels,
                        raise_wrong_sector=True,
                        warn_wrong_sector=True)
    
    
    print(array_defined)
    #init_env_data_halfinf['init_LP'] =array_defined
    init_env_data_halfinf['age_LP'] = 0


    print('setleft environment',".."*30)
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


def load_right_environment(name,num_site):
   
    #LOAD ENVIRONMENT
    print("loading right environment",'..'*20)
    with open(name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    Bflat=loaded_xxxx['RP_B']
    qflat_list_c=loaded_xxxx['RP_q']
   
    print(Bflat.shape)
    #shape=Bflat.shape
   
    #change the shape of Bflat and qflat
    Bflat=np.transpose(Bflat, (1, 0, 2))
    print(len(qflat_list_c))
    qflat_list=[qflat_list_c[1],qflat_list_c[0],qflat_list_c[2]]
    




    root_config_ = np.array([0,1,0])
    root_config_ = root_config_.reshape(3,1)
    site=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=('N','K'),site_loc=num_site)
    chargeinfo=site.leg.chinfo
    legcharges=[]
    
    #loads right environment
    labels=['vL', 'wL', 'vL*']
    conj_q=[1,1,-1]
    #print()
    print(qflat_list[1])

    #shift K by num_site-2 to get the correct environment
    for i in range(len(qflat_list[0])):
        qflat_list[0][i][0]+=qflat_list[0][i][1]*(num_site-2)
    for i in range(len(qflat_list[2])):
        qflat_list[2][i][0]+=qflat_list[2][i][1]*(num_site-2)
    for i in range(len(qflat_list[1])):
        qflat_list[1][i][0]+=qflat_list[1][i][1]*(num_site-2)

    print(qflat_list[1])
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



def load_left_environment(name):
   
    #LOAD ENVIRONMENT
    print("loading left environment",'..'*20)
    with open(name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    print(loaded_xxxx.keys())
    #quit()
    Bflat=loaded_xxxx['LP1_B']
    qflat_list_c=loaded_xxxx['LP1_q']
   
    
    #shape=Bflat.shape
   
    #change the shape of Bflat and qflat
    Bflat=np.transpose(Bflat, (1, 0, 2))
    #SET IT TO ZEROS BUT WITH CORRECT CHARGES
    a,b,c=Bflat.shape
    Bflat=0*Bflat
    for i in range(a):
           Bflat[i,-1,i]=1
    print(Bflat.shape)
    #print(len(qflat_list_c))
    qflat_list=[qflat_list_c[1],qflat_list_c[0],qflat_list_c[2]]
    root_config_ = np.array([0,1,0])
    root_config_ = root_config_.reshape(3,1)
    site=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=('N','K'),site_loc=-1)
    chargeinfo=site.leg.chinfo
    legcharges=[]
    
    #loads right environment
    labels=['vR*', 'wR', 'vR']
    conj_q=[1,1,-1]

    #shift K
    for i in range(len(qflat_list[0])):
        qflat_list[0][i][0]+=-qflat_list[0][i][1]*1
    for i in range(len(qflat_list[2])):
        qflat_list[2][i][0]+=-qflat_list[2][i][1]*1
    for i in range(len(qflat_list[1])):
        qflat_list[1][i][0]+=-qflat_list[1][i][1]*1

    for i in range(len(qflat_list)):
        #print(i)
        #print(qflat_list[i])
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
    print("left environment is loaded",'..'*20)
  
    return environment

L=14



M,sites=create_segment_DMRG_model(L)
#THIS ONE HAS BOTH N,K conservation
name='qflat_QH_1_3_K_cons'
psi_halfinf=load_data(name,sites)
#print(psi_halfinf._B[0])
#quit()
#quit()
print(len(sites))
name='env_QH_1_3_K_cons'
right_env=load_right_environment(name,len(sites))


#print('babababa')
#print(psi_halfinf._B[-1])
#print(right_env)
#print(M.H_MPO._W[-1])
#quit()
left_environment=load_left_environment(name)
print('bababbab')
print(psi_halfinf._B[0])
print(left_environment)
print(M.H_MPO._W[0])
#quit()

psi_halfinf.canonical_form_finite(cutoff=0.0)
print(psi_halfinf)
#print(a)
#print(b)
#quit()
#psi_halfinf=project_and_find_segment_mps(mps,last)
init_env_data_halfinf={}
#initialize right enviroment
#INSTEAD OF 44 SHOULD JUST BE A LAST ELEMENT?
#what is the second parameter???

init_env_data_halfinf['init_LP'] = left_environment    #DEFINE RIGHT ENVIROMENT
init_env_data_halfinf['age_LP'] =0

init_env_data_halfinf['init_RP'] = right_env   #DEFINE RIGHT ENVIROMENT
init_env_data_halfinf['age_RP'] =0




dmrg_params = {
    'mixer': True,
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 400,
        'svd_min': 1.e-10,
    },
}

eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_halfinf, M, dmrg_params,
                                     resume_data={'init_env_data': init_env_data_halfinf})

print("enviroment works")
print("running DMRG")
eng_halfinf.run()

