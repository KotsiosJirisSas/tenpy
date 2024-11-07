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
    Lx = 14;			# circumference
    LL = 0;			# which Landau level to put in
    mixing_chi = 450; #Bond dimension in initial sweeps
    chi = 450;		#Bond dimension of MPS
    xi = 6;			# The Gaussian falloff for the Coulomb potential
    Veps = 1e-4		# how accurate to approximate the MPO


    NLL = 1; Veps = 1e-4
    xi = 1
    d = 0
    def rvr(r):
        return np.exp(-r/xi)
    V = { 'eps':Veps, 'xiK':xi, 'GaussianCoulomb': {('L','L'):{'v':1, 'xi':xi}} }

    root_config = np.array([0, 1, 0])		# this is how the initial wavefunction looks

    model_par = {
        'verbose': 3,
        'layers': [ ('L', LL) ],
        'Lx': Lx,
        'Vs': V,
        'boundary_conditions': ('infinite', L),
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

    try:
        M = MPOGraph(sites=sites,bc='segment',max_range=None) #: Initialize MPOGRAPH instance

        '''
        M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

        Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
        The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
        '''

        States=QH_Graph_final.obtain_states_from_graphs(G_new,L)
        print("Ordering states",".."*10)

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



        #initialize wavefunction as MPS
    
        
        pos= [[i] for i in range(L)]

        #quit()
        lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="segment")
        x=lattice.mps_sites()
            
        model=MPOModel(lattice, H)
    except:
        model=0

    print("created model",".."*30)
    return model,sites

def load_data(name,sites):
    #name='qflat_QH_1_3-2'
 
    with open(name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    #print(loaded_xxxx['qflat'])
    #quit()
    number=len(sites)//3
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




L=15

M,sites=create_segment_DMRG_model(L)
#THIS ONE HAS BOTH N,K conservation
name='qflat_QH_1_3-3'
psi_halfinf=load_data(name,sites)




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
last=L-1
age=50
init_env_data_halfinf['init_RP'] = MPOEnvironment(psi_halfinf, M.H_MPO, psi_halfinf).init_RP(last, age)    #DEFINE RIGHT ENVIROMENT
init_env_data_halfinf['age_RP'] =0


quit()
'''
M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
'''



#initialize wavefunction as MPS
pstate=["empty", "full","empty"]*broj
psi = MPS.from_product_state(sites, pstate, bc="segment")
print(psi.bc)
#simple_lattice()

#initialize MPOModel

from tenpy.models.lattice import Lattice
from tenpy.models.lattice import IrregularLattice
#lattice=Chain(L=N,site=sites, bc="periodic",  bc_MPS="finite")
#lattice=Chain(N,spin, bc="periodic",  bc_MPS="finite")
pos= [[i] for i in range(L)]
#quit()
lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="segment")
x=lattice.mps_sites()




quit()


N=3
M_i=create_infinite_DMRG_model(N)
name="qflat_QH_1_3-2"

#returns infinite DMRG MPS
mps=load_data(name)

root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
#spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')

spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
sites=[spin]*3
pstate=["empty", "full","empty"]
#mps = MPS.from_product_state(sites, pstate, bc="infinite")


N=5


#DEFINE MPO MODEL M_i for infinite DMRG AND TURN IT INTO A SEGMENT
M_s = M_i.extract_segment(enlarge=N)

first, last = M_s.lat.segment_first_last
#print(last)
#quit()

#MULTIPLY BY CELL SIZE

#psi_halfinf,init_env_data_halfinf=set_infinite_like_segment(mps,M_i,last)
#mps3 = MPS.from_product_state(sites, pstate, bc="infinite")
mps2=mps.extract_segment(0,last)
psi_halfinf = mps2.copy() 

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

age=50
init_env_data_halfinf['init_RP'] = MPOEnvironment(mps, M_i.H_MPO, mps).init_RP(last, age)    #DEFINE RIGHT ENVIROMENT
init_env_data_halfinf['age_RP'] =0
#print(init_env_data_halfinf['init_RP'])
#quit()
#initialize left enviroment
init_env_data_halfinf=set_left_environment(mps,init_env_data_halfinf,M_i.H_MPO)



dmrg_params = {
    'mixer': True,
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 400,
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
eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_halfinf, M_s, dmrg_params,
                                     resume_data={'init_env_data': init_env_data_halfinf})


print("enviroment works")
print("running DMRG")
eng_halfinf.run()

