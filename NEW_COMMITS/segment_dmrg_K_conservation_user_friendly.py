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




np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)
def create_segment_DMRG_model(model_par,L,root_config_,conserve):

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
    M = mod.QH_model(model_par)
    print("Old code finished producing MPO graph",".."*10)


    #load tenpy2 graph into tenpy3 graph
    G=M.MPOgraph
    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
    


    #root_config_ = np.array([0,1,0])
  

    #define Hilbert spaces for each site with appropriate conservation laws
    sites=[]
    for i in range(L):
      
        spin=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=conserve,site_loc=i)
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


    #sort leg charges to make DMRG algortihm quicker
    H.sort_legcharges()
    
    #Define lattice on which MPO is defined
    pos= [[i] for i in range(L)]
    lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="segment")
    x=lattice.mps_sites()
    
    #create model from lattice and MPO
    model=MPOModel(lattice, H)
    print("created model",".."*30)

    assert model.H_MPO.is_equal(model.H_MPO.dagger())
    print('asserted')
    
    return model,sites

def load_data(name,sites):
    """
    loads MPS as segment mps of length len(sites)
    name: Str, name of the .pkl file from which we import
    sites: list of class:Sites, list of hilbert spaces corresponding to each site
    """
    L=len(sites)
    with open(name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    
    #finds length of an infinite unit cell
    Bflat0=loaded_xxxx['Bs']
    infinite_unit_cell=len(Bflat0)

    number=len(sites)//infinite_unit_cell+1
    
    #enlarge the unit cell accordingly,
    #TODO: MAKE THIS PROCESS QUICKER
    Bflat=Bflat0*number

    #load singular values
    Ss=loaded_xxxx['Ss']
    
    #load charge infromation
    qflat2=loaded_xxxx['qflat']

    #change qflat into representation consistent with Tenpy3
    #this is just charges of the leftmost leg
    qflat=[]
    for i in range(len(qflat2)):
        kopy=[]
        for m in range(len(qflat2[i])):
            kopy.append(qflat2[i][m])
        qflat.append(kopy)
    qflat=np.array(qflat)
  

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
    B = psi_halfinf.get_B(0, form='B')
    B.iproject(proj, 'vL')
    psi_halfinf.set_B(0, B, form='B')
    psi_halfinf.set_SL(0, np.ones(1, float))
    psi_halfinf.canonical_form_finite(cutoff=0.0)
    psi_halfinf.test_sanity()

    print('projected MPS',".."*30)
 
    return psi_halfinf


def load_environment(name,location,root_config_,conserve, side='right'):

    #TODO: GENERALIZE TO ALL CONSERVATION LAWS SO THAT IT IS LOADED MORE SMOOTHLY
    """
    loads environment on from old tenpy2 code
    name:   Str, the file name in which the environment is saved
    location: Int, sets location of the site at which environment is located (-1 for left and len(sites) for right usually) 
    side: Str, if right loads right side, otherwise loads left side
    """
    print("loading "+side+ " environment",'..'*20)
    with open(name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    if side =='right':
        Bflat=loaded_xxxx['RP_B']
        qflat_list_c=loaded_xxxx['RP_q']
        #change the shape of Bflat and qflat so that it is consistent with new tenpy
        Bflat=np.transpose(Bflat, (1, 0, 2))
        qflat_list=[qflat_list_c[1],qflat_list_c[0],qflat_list_c[2]]
        labels=['vL', 'wL', 'vL*']
        conj_q=[1,1,-1]
        shift=2
    else:
        Bflat=loaded_xxxx['LP2_B']
        qflat_list_c=loaded_xxxx['LP2_q']
        #transpose bflat and qflat to make legs consistent with TeNpy3
        Bflat=np.transpose(Bflat, (2, 0, 1))
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
    print(environment)
    print("environment is loaded",'..'*20)
    #x=environment.qtotal
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
    print(environment)
    print("vacuum left environment is loaded",'..'*20)
    print(environment.qtotal)

    return environment

def load_param(name):

    with open(name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')
    model_par = loaded_xxxx['model_par']

    root_config_ = model_par['root_config'].reshape(len(model_par['root_config']),1)

    conserve=[]
    if  model_par['cons_C']==total:
        conserve.append('N')

    if model_par['cons_K']:
        conserve.append('K')
    #load root_configuration and conservation laws
    if len(conserve)==1:
        conserve=conserve[0]
    else:
        conserve=tuple(conserve)
    return model_pat,conserve,root_config_



L=14

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
    'boundary_conditions': ('infinite', L),
    'cons_C': 'total', #Conserve number for each species (only one here!)
    'cons_K': False, #Conserve K
    'root_config': root_config, #Uses this to figure out charge assignments
    'exp_approx': '1in', #For multiple orbitals, 'slycot' is more efficient; but for 1 orbital, Roger's handmade code '1in' is slightly more efficient
}
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)

conserve=('N','K')


#model_par,conserve,root_config_=load_param(name)

model_par['boundary_conditions']= ('infinite', L)
M,sites=create_segment_DMRG_model(model_par,L,root_config_,conserve)
#THIS ONE HAS BOTH N,K conservation
name='qflat_QH_1_3_K_cons'
psi_halfinf=load_data(name,sites)

#psi_halfinf.left_environment(0)
psi_halfinf=project_left_side_of_mps(psi_halfinf)
#psi_halfinf.canonical_form_finite()



print(len(sites))
name='env_QH_1_3_K_cons'
right_env=load_environment(name,len(sites),root_config_,conserve, side='right')




#leg_MPS
leg_MPS=psi_halfinf._B[0].get_leg('vL')
#leg_MPO
leg_MPO=M.H_MPO._W[0].get_leg('wL')
left_environment=set_left_environment_to_vacuum(leg_MPO,leg_MPS)


for i in range(1):
    x=psi_halfinf._B[i]#.qtotal
    print(x)
    b=M.H_MPO._W[i]#.qtotal
    print(b)
    a=right_env#.qtotal
    #print(a)

    a=left_environment#.qtotal
    print(a)



init_env_data_halfinf={}


init_env_data_halfinf['init_LP'] = left_environment    #DEFINE LEFT ENVIROMENT
init_env_data_halfinf['age_LP'] =0

init_env_data_halfinf['init_RP'] = right_env   #DEFINE RIGHT ENVIROMENT
init_env_data_halfinf['age_RP'] =0




dmrg_params = {
    'mixer': True,
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 1900,
        'svd_min': 1.e-10,
    },
}

eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_halfinf, M, dmrg_params,
                                     resume_data={'init_env_data': init_env_data_halfinf})
#print(eng_halfinf.chi_max)
#quit()
print("enviroment works")
print("running DMRG")
#quit()
#print("MPS qtotal:", M.qtotal)
#print("MPO qtotal:", psi_halfinf.qtotal)
eng_halfinf.run()

