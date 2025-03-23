#!/usr/bin/env python
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys
print(sys.version)

sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 
#sys.path.append('/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy') 
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
from tenpy.additional_import.Zalatel_import import packVmk

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

#import h5py
#import hdfdict
from tenpy.tools import hdf5_io
#SAVE THE DATA
#import h5py
#from tenpy.tools import hdf5_io
import pickle
import time
np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)
###
#some imports from dom's script#
from pf_apf_newest2 import trim_down_the_MPO,eliminate_elements_from_the_graph,get_old_basis
#from pf_apf_newest import set_MPS_boundary # adds 2+pstate sites/ pf apf 2 adds 1+pstate sites
####
print('='*100)
print('succesfull import of all modules')
print('='*100)
####



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


    #print('Charge of leg to which we project')
    #print(charge)
    

    leg=psi_halfinf._B[0].get_leg('vL')
    flats=leg.to_qflat()
    #print(flats)
    

    
    ind=[]

    
    #finds the schmidt values with the corresponding charge
    for j in range(len(charge)):
        for i in range(len(flats)):
            if np.all(flats[i]==charge[j]):
                if i not in ind:
                    ind.append(i)
                    break;
    
    #get rid of duplicates
    #ind=list(set(ind))
    #print(ind)
    ind=np.array(ind)

    #print(ind)
    print("number of remaining indices:")
    print(len(ind))
    
    
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


class QH_system():
    '''
    An instance of a Quantum Hall model ready to be put into a segment DMRG calculation.
    Key properties:

    MPO
    Initial MPS
    Left_environment
    Right_Environment

    '''
    ##
    ###
    def __init__(self, params):
        '''
        loads whatever it needs to load
        '''
        time0 = time.time()
        self.data_location = params['model data file']
        self.L_env_loc = params['left environment file']
        self.R_env_loc = params['right environment file']
        self.system_length = params['sys length']#what Dom calls L in his code
        self.verbose = params['verbose']
        self.unit_cell = params['unit cell']
        self.pstate = []#,'','full']#params['sites added']

        for i in range(params['sites added']-2):
            self.pstate.append('')
        #self.pstate=['empty','full','empty','full','empty','full','empty']
        self.load_model_params()
        self.graph = [self.graph[0]]*self.system_length #total graph is graph x sys length
        #Q: for multicomponent systems have to change this no? ie --> self.graph[:#_of_components]
        self.model_par['boundary_conditions']= ('infinite', self.system_length)
        self.model_par['layers'] = [('L',self.LL)]
        self.sides_added = params['sites added']
        self.shift_import=[]
        #TODO LOAD ENCIRONMNENTS AT THIS POINT BUT ONLY MATCH THEM LATER
        
        
        time1 = time.time()
        if self.verbose>0:
            print('='*100)
            print("INITIALIZATION TIME:",time1-time0)
            print('='*100)
        ###############################################
        ############ CREATE MPO #######################
        ###############################################
        if self.verbose>0:
            print('='*100)
            print("CREATING MPO MODEL")
            print('='*100)
        self.create_segment_DMRG_model()
        time2 = time.time()
        if self.verbose>0:
            print('='*100)
            print("MPO CONSTRUCTION TIME:",time2-time1)
            print('='*100)
        ###############################################
        ############ GET PERMUTATION ##################
        ###############################################
        if self.verbose>0:
            print('='*100)
            print("GETTING PERMUTATION")
            print('='*100)
        self.load_permutation_of_basis()
        time3 = time.time()
        if self.verbose>0:
            print('='*100)
            print("PERMUTATION TIME:",time3-time2)
            print('='*100)
        ###############################################
        ############ GET WAVEFUNCTIONS ################
        ###############################################
        if self.verbose>0:
            print('='*100)
            print("GETTING LEFT WAVEFUNCTION")
            print('='*100)
        self.load_data(side='left',MPSshift=0)
        if self.verbose>0:
            print('='*100)
            print("GETTING RIGHT WAVEFUNCTION")
            print('='*100)
        self.load_data(side='right',MPSshift=self.sides_added)
        time4 = time.time()
        if self.verbose>0:
            print('='*100)
            print("GETTING WAVEFUNCTIONS TIME:",time4-time3)
            print('='*100)
        ###############################################
        ############ GET ENVIRONMENTS ################
        ###############################################
        if self.verbose>0:
            print('='*100)
            print("GETTING LEFT ENVIRONMENT")
            print('='*100)
        self.left_env = self.load_environment(location=-1,side='left',charge_shift=[0,0])
        if self.verbose>0:
            print('='*100)
            print("GETTING RIGHT ENVIRONMENT")
            print('='*100)
        self.right_env=self.load_environment(location=len(self.sites)-2,side='right')
        time5 = time.time()
        if self.verbose>0:
            print('='*100)
            print("GETTING ENVIRONMNETS TIME:",time5-time4)
            print('='*100)

    
    def create_segment_DMRG_model(self,add=False):
        """
        TODO: This function has conservation of momentum hard-coded
        TODO: This function has # of components = 1 hardcoded
        ----------------------------------------
        Creates MPOModel given graph from tenpy2.
        Uses:
            self.model_par(dict): Contains model parameters
            self.system_length
            self.root_config
            self.conserve
            self.graph, the old tenpy graph
    
        Returns:
            self.mode: MODEL ENCOMPASSING MPO AND LATTICE
            self.sites: contain HILBERT SPACE ON THE LATTICE
            self.ordered_states
            self.graph, the new updated graph
        """
        # construct MPO Model from the Graph using the tenpy2 code
        if self.verbose >0:
            print('.'*100)
            print("transforming MPO from old to new tenpy")
            print('.'*100)
        #load tenpy2 graph into tenpy3 graph
        G=self.graph
        #trim down the MPO
        G, extra_keys=trim_down_the_MPO(G)
        #gives the graph in tenpy3
        G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
        #define Hilbert spaces for each site with appropriate conservation laws.
        #If one adds additional cells to the system, make sure the momentum quantum numbers shift appropriately
        if not add:
            sites=[]
            for i in range(self.system_length):
                #produce a single site
                spin=QH_MultilayerFermionSite_final(N=1,root_config=self.root_config,conserve=('each','K'),site_loc=i)
                sites.append(spin)
        else:
            sites=[]
            #produce sites
            for i in range(-add,self.system_length-add):
                spin=QH_MultilayerFermionSite_final(N=1,root_config=self.root_config,conserve=('each','K'),site_loc=i)
                sites.append(spin)
        #self.sites = sites
        #return
        #TODO: remove above two lines after error fixing


        #initialize MPOGRAPH class instance
        M = MPOGraph(sites=sites,bc='segment',max_range=None)
        '''
        M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

        Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
        The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
        '''
        States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,self.system_length)
        if self.verbose>0:
            print('.'*100)
            print("Ordering states")
            print('.'*100)
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
            States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,self.system_length)
        
    
        M.states = States #: Initialize aux. states in model
        M._ordered_states = QH_G2MPO.set_ordered_states(States) #: sort these states(assign an index to each one)
        
        if self.verbose>0:
            print('.'*100)
            print("Testing sanity of ordered states")
            print('.'*100)
        M.test_sanity()
        M.graph = G_new #: Inppuut the graph in the model 
        if self.verbose>0:
            print('.'*100)
            print("BUILDING GRID...")
        grids =M._build_grids()#:Build the grids from the graph
        if self.verbose>0:
            print("BUILDING MPO...")
        H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
        if self.verbose>0:
            print('.'*100)
            print("MPO BUILT")
            print('.'*100)

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
        pos= [[i] for i in range(self.system_length)]
        lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="segment")
        x=lattice.mps_sites()
        
        #create model from lattice and MPO
        model=MPOModel(lattice, H)
        if self.verbose>0:
            print('.'*100)
            print("MODEL CREATED")
            print('.'*100)
        if self.verbose>0:
            print('.'*100)
            if model.H_MPO.is_equal(model.H_MPO.dagger()):
                print("MPO *IS* HERMITIAN")
            else:
                print("MPO *IS NOT* HERMITIAN")
            print('.'*100)
        self.graph_new = G_new #updated graph
        self.model = model #MODEL ENCOMPASSING MPO AND LATTICE
        self.sites = sites #contain HILBERT SPACE ON THE LATTICE
        self.ordered_states = ordered_states
        return 

    def add_sides_to_Left_MPS(self):
        '''
        adds *num* sites to the left MPS.
        do we get charge mismatch error?
        '''
        num = self.sides_added
        if self.verbose>0:
            print('='*100)
            print('ADDING ',str(num),' sites to the right of LHS MPS')
            print('='*100)
        ###
        #make sure RHS MPS is shifted by 0.
        Bflat = []
        Ss = []
        Ss1 = self.psi_halfinf_L._S
        Bs1 = self.psi_halfinf_L._B
        ###
        # PART 1: CONSTRUCT LHS OF MPS
        for i in range(len(self.psi_halfinf_L._B)):
            Ss.append(Ss1[i])
            Bflat.append(np.transpose(Bs1[i].to_ndarray(),(1,0,2)))
        #PART 2: ADD ADDITIONAL SITES
        leg_physical=[]
        for i in range(num):
            leg_physical.append(self.sites[len(self.psi_halfinf_L._B)+i].leg)

        left_leg = self.psi_halfinf_L._B[-1].get_leg('vR')
        left_Ss = self.psi_halfinf_L._S[-1]
        #Bs_added,Ss_added = self.add_sites_to_L(legL = left_leg,Sleft=left_Ss,leg_physical = leg_physical,numsites=num)
        Bs_added,Ss_added = self.add_sites_to_L_with_pstate(legL = left_leg,Sleft=left_Ss,leg_physical = leg_physical,numsites=num)
        #append extra Bs and Ss into total list
        Ss.append(Ss1[len(self.psi_halfinf_L._B)]) #add one more pair of SVs #TODO Am i adding it in wrong place?
        for i in range(num):
            Ss.append(Ss_added[i])
            Bflat.append(np.transpose(Bs_added[i],(1,0,2)))
        new_length = len(self.psi_halfinf_L._B) + num
        for index in range(new_length):
            print('B,S,S',Bflat[index].shape,len(Ss[index]),len(Ss[index+1]))
        psi=MPS.from_Bflat(self.sites[:new_length],Bflat[:new_length],SVs=Ss[:new_length+1], bc='segment',legL=self.psi_halfinf_L._B[0].get_leg('vL'))
        filling= psi.expectation_value("nOp")
     
        print('filling',filling,'avg nu',np.sum(filling)/new_length)
        #
        quit()
        #psi=MPS.from_Bflat(self.sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
        return
    
    def patch_WFs_trivial(self):
        '''
        patch WFS together w/o adding extra sides inbetween.
        need to change way i get RHS code!
        '''
        if self.verbose>0:
            print('='*100)
            print('PATCHING WAVEFUNCTIONS TOGETHER *WITH NO EXTRA SITES IN BETWEEN*')
            print('='*100)
        ###
        #make sure RHS MPS is shifted by 0.
        self.load_data(side='right',MPSshift=0)
        Bflat = []
        Ss = []
        Ss1 = self.psi_halfinf_L._S
        Ss2 = self.psi_halfinf_R._S
        Bs1 = self.psi_halfinf_L._B
        Bs2 = self.psi_halfinf_R._B
        ###
        # PART 1: CONSTRUCT LHS OF MPS
        for i in range(len(self.psi_halfinf_L._B)):
            Ss.append(Ss1[i])
            Bflat.append(np.transpose(Bs1[i].to_ndarray(),(1,0,2)))
        # PART 2: CONSTRUCT RHS OF MPS
        for i in range(len(self.psi_halfinf_R._B)):
            Ss.append(Ss2[i])
            Bflat.append(np.transpose(Bs2[i].to_ndarray(),(1,0,2)))
        # add final Ss. I guess for N sites we need N+1 Ss?
        Ss.append(Ss2[-1])
        #######
        #reads charges off the left leg of MPS
        left_leg=self.psi_halfinf_L._B[0].get_leg('vL')
        psi=MPS.from_Bflat(self.sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
        self.psi = psi
        #####
        filling= psi.expectation_value("nOp")
        print('filling',filling)
        print('avg nu',np.sum(filling)/(self.system_length))
        ######
        print('flat Bs',len(self.psi._B[-1].to_ndarray()))
        for flatB in range(len(self.psi._B[-1].to_ndarray())):
            print(self.psi._B[-1].to_ndarray()[flatB].shape)
        quit()
        print('-2 ',self.psi._B[-2].get_leg('vR'))

        print('phys',self.sites[-1].leg)
        print('-1',self.psi._B[-1].get_leg('vR'))
        #
        print('q flat',self.psi._B[-1].get_leg('vR').to_qflat())
        #
        for sslice in self.psi._B[-2].get_leg('vR').slices:
            print(self.psi._B[-2].get_leg('vR').charges[sslice])
        return

    def construct_BS_from_right(self,leg_physical,numsites,Ss_full, Qflat_added, qflat_left,Qflat_right_final,Ss_fin,Ss_left_side):
        Qs=[Qflat_right_final]
        Ss=[Ss_fin]
        Bs=[]
        for site in range(numsites):
            qflat=[] #qflat to the right of site
            Ss_temp = []#temporary singular values to the right of site
            ch_physical = leg_physical[numsites-1-site].to_qflat() #physicall charges

            
            if site == 0:
                leg_to_the_right_charges = Qflat_right_final
                Ss_temp_left = Ss_full[-1]
            else:
                leg_to_the_right_charges = Qs[site]
                Ss_temp_left = Ss[site ]
            for j,charge in enumerate(leg_to_the_right_charges):
            
    
                
                charge_to_the_right = charge - ch_physical[0] # does not add a particle
               
                qflat.append(charge_to_the_right)
                Ss_temp.append(Ss_temp_left[j])
                charge_to_the_right = charge - ch_physical[1] # adds a particle
                qflat.append(charge_to_the_right)
                Ss_temp.append(Ss_temp_left[j])


            #intersection = [x for x in qflat if x in Qflat_added[-2-site]]
           
           
            #print('intersection AAAAAA')
            #print(intersection[0])
            #Find only elements that survive
            # print(qflat)
            #quit()
            surviving_indices=[]
            print("START START")
            print('finding the surviving indices!')
            if site!=numsites-1:
                print(len(Qflat_added[-2-site]))
                for i, x in enumerate(Qflat_added[-2-site]):
                    
                    for y in qflat:
                    
                        if x[0]==y[0] and x[1]==y[1]:
                            surviving_indices.append(i)

                surviving_indices=list(set(surviving_indices))
                #print(surviving_indices)
          
                #print(qflat)
                #quit()
                surviving_indices=np.array(surviving_indices)
                # Indices of removed elements
                #print(Qflat_added[-2-site])
                intersection=np.array(Qflat_added[-2-site])[surviving_indices]
            
                Ss_temp=Ss_full[-2-site][surviving_indices]
                qflat=np.array(qflat)
                Ss_temp = np.array(Ss_temp)
                qflat=np.array(intersection)
                #print(qflat.shape)
            if site==numsites-1:
                #put left side SS as the last one
                qflat=qflat_left
                Ss_temp=Ss_left_side
            ####################################################
            ##################### now fill Bs ##################
            ####################################################
            Bflat = np.zeros((len(qflat),2,len(leg_to_the_right_charges)))
            print(Bflat.shape)
            for j,charge in enumerate(leg_to_the_right_charges):
                charge_to_the_right = charge - ch_physical[0]
                ind=np.where((qflat==charge_to_the_right).all(axis=1))[0]
                Bflat[ind,0,j]+=np.random.random(len(ind))
                charge_to_the_right = charge - ch_physical[1]
                ind=np.where((qflat==charge_to_the_right).all(axis=1))[0]
                Bflat[ind,1,j]+=np.random.random(len(ind))
            ####################################################
            ##################### now fill Ss ##################
            ####################################################
            #Alternative choice: get random singular values
         
            #get singular values that are a continuation of the previous one. ie if Q to the left has singular value S, then Q'=Q+qphys to the right will have singular value S
            Ss.append(Ss_temp/np.sum(Ss_temp**2)) # these are the singular values to the *RIGHT* of my site. usual notation would have them to the left.
            Bs.append(Bflat)
            Qs.append(qflat)
        Qs=Qs[::-1]
        print(len(Bs))
        Bs=Bs[::-1]
        Ss=Ss[::-1]

        for i in range(len(Bs)):
            print(Bs[i].shape)
            print(Ss[i].shape)
      
        return Bs,Ss[1:]


    def patch_WFs(self):
        from collections import Counter
        from collections import defaultdict
        '''
        Given the half-inf left and right MPSs, patch them together.
        Can insert cells to move us to different (N,K) sectors
        '''

        '''
        TUESDAY TO DO:
        ------------------
        1)  FOLLOW *ADD SIDES TO THE LEFT MPS* CODE
        2)  THEN, GIVEN THE RIGHT LEG OF LHS (LHS+ADDED SITES) AND THE LEFT LEG OF RHS,
            PROJECT BOTH  TO A SMALLER BOND DIMENSION SUCH THAT ALL THEIR LEGS ARE CONNECTED.
            THIS WILL REQUIRE ERASING SOME BFLATs AND Ss OF BOTH THE LHS AND RHS. ACTUALLY USE Ss OF 
            KEEP LHS LEG W CHARGE Q AND RHS LEG W CHARG Q' IF Q = Q' + Qphys FOR ANY OF THE TWO POSSIBLE QPHYS
        '''
        num = self.sides_added
        if self.verbose>0:
            print('='*100)
            print('ADDING ',str(num),' sites to the right of LHS MPS')
            print('='*100)
        ###
        #make sure RHS MPS is shifted by 0.
        Bflat = []
        Ss = []
        Ss1 = self.psi_halfinf_L._S
        Bs1 = self.psi_halfinf_L._B
        Ss2 = self.psi_halfinf_R._S
        #Bs2 = self.psi_halfinf_R._B
        ###
        # PART 1: CONSTRUCT LHS OF MPS
        for i in range(len(self.psi_halfinf_L._B)):
            Ss.append(Ss1[i])
            Bflat.append(np.transpose(Bs1[i].to_ndarray(),(1,0,2)))

        Ss.append(Ss1[-1])
        #PART 2: ADD ADDITIONAL SITES
        leg_physical=[]
        for i in range(num):
            leg_physical.append(self.sites[len(self.psi_halfinf_L._B)+i].leg)
        
        left_leg=self.psi_halfinf_L._B[-1].get_leg('vR')
        left_Ss = self.psi_halfinf_L._S[-1]
        #Bs_added,Ss_added = self.add_sites_to_L(legL = left_leg,Sleft=left_Ss,leg_physical = leg_physical,numsites=num)
      
        Qflat_added,Ss_added= self.add_sites_to_L_with_pstate(legL = left_leg,Sleft=left_Ss,leg_physical = leg_physical,numsites=num)
        #PART 3: CONNECT TO RHS

       
        
        #first: how many charges agree?
        charges_LHS = Qflat_added[-1]
        charges_RHS = self.psi_halfinf_R._B[0].get_leg('vL').to_qflat()
        largest_charge_LHS = charges_LHS[np.argmax(np.array(Ss_added[-1]))]
        largest_charge_RHS = charges_RHS[np.argmax(np.array(Ss2[0]))]
     

        charges_LHS = np.array(charges_LHS)
        charges_RHS = np.array(charges_RHS)
        shift = np.array(largest_charge_RHS) - np.array(largest_charge_LHS)
        #print(charges_LHS)
        #print(charges_RHS)
        #CHANGE TO SHIFT TO SOME FIXED VALUE!
        #[-460   -8]
        # 443    7
        #[-351   -6]
        #RELEVANT CHARGE SHIFT
        #shift=charges_RHS[self.shift_import[0]]-charges_LHS[self.shift_import[1]]
        #shift=self.shift_import

      
        #FOR 3 UNIT CELL INSERTIONS WE GET THIS
        #shift=np.array([9,0])


        charges_RHS_shifted = charges_RHS - shift
        # Step 1: Store indices of each row in LHS and RHS
        lhs_indices = defaultdict(list)
        rhs_indices = defaultdict(list)
        for i, row in enumerate(charges_LHS):
            lhs_indices[tuple(row)].append(i)
        for j, row in enumerate(charges_RHS_shifted):
            rhs_indices[tuple(row)].append(j)
        # Step 2: Pair indices while respecting the count limit
        matches = []
        for row in lhs_indices.keys() & rhs_indices.keys():  # Only consider common rows
            lhs_list = lhs_indices[row]
            rhs_list = rhs_indices[row]
            # Pair elements up to the minimum occurrences
            for i, j in zip(lhs_list, rhs_list):
                matches.append((i, j))
        # Convert to numpy array
        matches = np.array(matches)
        #print("Index Pairs:\n", matches)
        print("Total Matching Pairs:", len(matches))  # Should match `common_count`
        for match in matches:
            ind_L,ind_R = match
            if not np.allclose(charges_LHS[ind_L],charges_RHS_shifted[ind_R]):
                print('uhhhhh')
        ######
        #if len(matches)<100:return 
        if len(matches)<30:return    


        Ss_left_side=self.psi_halfinf_L._S[-1]
        #remove some  Ss and Bs from RHS MPS anc construct again. see if it kills off dead nodes
        S_LHS = Ss_added[-1].copy()
    
        mask = np.isin(np.arange(len(S_LHS)), matches[:,0])
        #print(mask)
        #quit()
        #B_LHS = B_LHS[:,:,mask]
        
        #construct the Bflat now!!
        qflat_left=self.psi_halfinf_L._B[-1].get_leg('vR').to_qflat()
        print('length of BS')
        print(len(Qflat_added[-1][mask]))
        #quit()
        Bs_added,Ss_added=self.construct_BS_from_right(leg_physical,num,Ss_added, Qflat_added, qflat_left,Qflat_added[-1][mask], S_LHS[mask],Ss_left_side)
        for i in range(len(Bs_added)):
            Bflat.append(np.transpose(Bs_added[i],(1,0,2)))
            Ss.append(Ss_added[i])
        for i in range(len(Bflat)):
            print(Bflat[i].shape)
            print(Ss[i].shape)
        new_length = len(self.psi_halfinf_L._B) + num
        print(new_length)
        
        ########################
        psi=MPS.from_Bflat(self.sites[:new_length],Bflat[:new_length],SVs=Ss[:new_length+1], bc='segment',legL=self.psi_halfinf_L._B[0].get_leg('vL'))
        #########################
        #psi.canonical_form_finite(cutoff=0.0)
        print(psi)
        #quit()
        #now one needs to shift all the charges of the B matrices 
        filling= psi.expectation_value("nOp")
        print('filling',filling)
        print('TOTAL FILLING',np.sum(filling-1/3))
        print('new bond dimensions',[len(S) for S in psi._S])
        


        print(psi)
        #LOAD RHS AGAIN, BUT NOW SHIFT CHARGES 
        self.load_data(side='right',MPSshift=self.sides_added,charge_shift=-shift)
      
        #RHS MPS
        #project RHS MPS TO relevant charges
        self.psi_halfinf_R,charges=project_side_of_mps_given_charges( self.psi_halfinf_R , charges_LHS ) 
        print(self.psi_halfinf_R)
        #########################################
        #create new psi, now by appending projected RHS AND LHS
        Bflat2=[]
        Ssnew=[]
        for i in range(len(Bflat[:new_length])):
            Bflat2.append(np.transpose(psi._B[i].to_ndarray(),(1,0,2)))
            Ssnew.append(psi._S[i])
            
        
        for i in range(len(self.psi_halfinf_R._B)):
            Bflat2.append(np.transpose(self.psi_halfinf_R._B[i].to_ndarray(),(1,0,2)))
            Ssnew.append(self.psi_halfinf_R._S[i])
       
        #Ss2.append(psi._S[-1])
        Ssnew.append(self.psi_halfinf_R._S[-1])
        #now add them together
        #########################################
        psi=MPS.from_Bflat(self.sites,Bflat2,SVs=Ssnew, bc='segment',legL=self.psi_halfinf_L._B[0].get_leg('vL'))
        #########################################

        filling= psi.expectation_value("nOp")
        print('filling',filling)

        print('testing the WF','...'*10)
        print('TOTAL leftover number of electrons',np.sum(filling-1/3))
        K=np.sum(np.arange(len(filling))*(filling-1/3))
        #print(psi)
        print("K sector:",K)

        print(shift)
        z=np.abs(np.sum(filling-1/3))
        #z=0
        if z<0.1:
            with open('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy/NEW_COMMITS/succes='+str(params['sites added'])+'.txt','w') as f:
                f.write(str(shift)+"\n")
                print(shift)
                f.write('excess number of electrons:'+str(np.sum(filling-1/3))+"\n")
                f.write("K sector: "+str(K)+"\n")
        quit()
        #print('success?')
        #filling= psi.expectation_value("nOp")
        #print('filling',filling)
        #print('TOTAL FILLING',np.sum(filling)/len(self.sites),np.sum(filling))
        #############################################
        self.psi_merged = psi
        quit()
        return
        #quit()

        new_length = len(self.psi_halfinf_L._B) + num
        for index in range(new_length):
            print('B,S,S',Bflat[index].shape,len(Ss[index]),len(Ss[index+1]))
        psi=MPS.from_Bflat(self.sites[:new_length],Bflat[:new_length],SVs=Ss[:new_length+1], bc='segment',legL=self.psi_halfinf_L._B[0].get_leg('vL'))
        filling= psi.expectation_value("nOp")
        print()
        print('filling',filling,'avg nu',np.sum(filling)/new_length)
        #
        quit()
        #psi=MPS.from_Bflat(self.sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
        #PART 3:CONSTRUCT RHS OF MPS
        for i in range(len(self.psi_halfinf_R._B)):
            Ss.append(Ss2[i])
            Bflat.append(np.transpose(self.psi_halfinf_R._B[i].to_ndarray(),(1,0,2)))
        Ss.append(Ss2[-1])
        ######
        #reads charges off the left leg of MPS
        left_leg=self.psi_halfinf_L._B[0].get_leg('vL')
        num = 32
        psi=MPS.from_Bflat(self.sites[:num],Bflat[:num],SVs=Ss[:num+1], bc='segment',legL=left_leg)
        filling= psi.expectation_value("nOp")
        print()
        print('filling',filling)
        #print(len(psi._B)*1/2)
        print('TOTAL FILLING',np.sum(filling)/len(self.sites),np.sum(filling))
        print('Patched two wavefunctions together sucessfully')
        self.psi = psi
        return
    #####################################
    #####################################
    ####### LOADING RELATED FUNCS #######
    #####################################
    #####################################
    def load_model_params(self):
        '''
        Loads the pickled parameter file, specifically the model parameters and root config.
        Input:
        name(str):      Path of pickle file
        '''
        with open(self.data_location, 'rb') as f:
            loaded_xxxx = pickle.load(f, encoding='latin1')
        model_par = loaded_xxxx['Model']
        root_config_ = model_par['root_config'].reshape(self.unit_cell,1)
        if model_par['cons_K']:
            conserve = (model_par['cons_C'],'K') 
        else:
            conserve = 'N'
        self.LL = model_par['LL']
        if self.verbose > 0 :
            print('='*100)
            print('MODEL PARAMETER KEYS')
            for key in model_par:
                print(key,':',model_par[key])
            print('='*100)
        self.model_par = model_par
        self.conserve = conserve
        if self.conserve != ('each','K'): raise ValueError
        self.root_config = root_config_
        self.graph = loaded_xxxx['graph']
        self.basis_old = loaded_xxxx['indices'][::-1]
        self.permutation_old = [loaded_xxxx['permutations']]
        self.Bflat_old = loaded_xxxx['MPO_B'][0]
        ##########################################
        #mps dictionaries#
        self.loaded_MPS_L = {'MPS_Ss':loaded_xxxx['MPS_Ss'],'MPS_qflat':loaded_xxxx['MPS_qflat'],'MPS_Bs':loaded_xxxx['MPS_Bs']}
        if 'MPS_2_Ss' in loaded_xxxx.keys():
            self.loaded_MPS_R = {'MPS_Ss':loaded_xxxx['MPS_2_Ss'],'MPS_qflat':loaded_xxxx['MPS_2_qflat'],'MPS_Bs':loaded_xxxx['MPS_2_Bs']}
        else:
            self.loaded_MPS_R = self.loaded_MPS_L.copy()
            print('RHS MPS copied from LHS MPS')
        ########################
        #environment loading#
        # and initialization#
        self.L_env_qs = loaded_xxxx['LP_q']
        self.R_env_qs = loaded_xxxx['RP_q']
        L_env =np.load(self.L_env_loc,allow_pickle=True)
        R_env =np.load(self.R_env_loc,allow_pickle=True)
        self.dict_L = {'LP':L_env['LP'],'LP_q':self.L_env_qs}
        self.dict_R = {'RP':R_env['RP'],'RP_q':self.R_env_qs}
        #return model_par,conserve,root_config_,loaded_xxxx
        return
    def load_data(self,MPSshift=2,side='right',charge_shift=[0,0]):
        """
        loads MPS as segment mps of length len(sites)
        Input:
        name:(str)      name of the .pkl file from which we import
        sites:(list)    list of class:Sites, list of hilbert spaces corresponding to each site
        shift:          Shifts the charges of the MPS according to the 'displacement' of the MPS
        MPSshift:(int)  How much to shift RHS MPS to the right by, to insert an interface in between
        side:           'left' or 'right'
        charge_shift:   
        """
        if side == 'right':
            psi_MPS = self.loaded_MPS_R.copy()
            half=(self.system_length//6)*3
            sites = self.sites[half+len(self.pstate)+MPSshift:]    # why the +2???? system = [LHS]|[added piece][RHS]
                                                            # and the [added piece] is -[]-[pstate]-[]-
            #print('og',self.sites[0].leg)
            #print('shifted',sites[0].leg)
            shift = half+len(self.pstate)+MPSshift
            #change this later on
            sites = self.sites[half+MPSshift:]    # why the +2???? system = [LHS]|[added piece][RHS]
                                                            # and the [added piece] is -[]-[pstate]-[]-
            #print('og',self.sites[0].leg)
            #print('shifted',sites[0].leg)
            shift = half+MPSshift
    
        elif side == 'left':
            half=(self.system_length//6)*3 
            psi_MPS = self.loaded_MPS_L.copy()
            sites = self.sites[:half]
            shift = 0
        else:
            raise ValueError 
        L=len(sites)
        Bflat0 = psi_MPS['MPS_Bs'].copy()
        Ss = psi_MPS['MPS_Ss'].copy()
        qflat2 = psi_MPS['MPS_qflat'].copy()[0] # we are taking first leg charges only
        #print('Qs shapes (at bonds)',len(qflat2),len(qflat2[0]),len(qflat2[1]),len(qflat2[2]))

        print('B FLAT SHAPE:',Bflat0[0].shape,Bflat0[1].shape,Bflat0[2].shape)
        #TODO: remove hardcoded periodicity

        #change qflat into representation consistent with Tenpy3
        #this is just charges of the leftmost leg
        qflat=[]
        for i in range(len(qflat2)):
            kopy=[]
            for m in range(len(qflat2[i])):
                if m==0:
                    shifted=qflat2[i][0]+shift*qflat2[i][1]# a momentum shift for the first charge sector???
                    kopy.append(shifted)
                else:
                    kopy.append(qflat2[i][m])
            qflat.append(kopy)
        print("shift charges. The charge shift is",charge_shift)
        for i in range(len(qflat)):
            qflat[i][0]+=charge_shift[0]
            qflat[i][1]+=charge_shift[1]  
        ####
        #print('Qs shapes (after)',len(qflat),len(qflat[0]),len(qflat[1]),len(qflat[2]))
        qflat=np.array(qflat)
        #print('qflat_shifted',qflat)
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
        if side == 'right':
            self.psi_halfinf_R = mps
        elif side == 'left':
            self.psi_halfinf_L = mps
        print('B FLAT SHAPE AFTER:',mps._B[0].shape,mps._B[1].shape,mps._B[2].shape)
        return 
#
    def load_permutation_of_basis(self):
        '''
        Given the TenPy2 graph and a set of permutations as input,
        permute the basis to make it compatible with the TenPy3 code.
        '''
        #TODO: COMMENT OUT
        #GENERALIZED FOR N LAYERS, BECAUSE EACH BOND HAS DIFFERENT PERMUTATIONS AND BASIS
        def find_permutation(source, target):
            return [source.index(x) for x in target]
        
        ordered_states = self.ordered_states
        G_old,basis_old,permutation_old = self.graph,self.basis_old,self.permutation_old
        old_basis=get_old_basis(G_old,basis_old,permutation_old)
        for i in range(len(old_basis)):
            assert len(old_basis[i])==len(ordered_states[i])

        permutation=[]
        for i in range(len(old_basis)):
            permutation.append(find_permutation(old_basis[i],ordered_states[i]))
        
        for i in range(len(old_basis)):
            #asserts two bases are the same
            assert np.all(ordered_states[i]==np.array(old_basis[i])[permutation[i]])

        if self.verbose>0:
            print('*'*100)
            print('Checking permutation')
            print('!sanity check perm needs modifying!')
            self.sanity_check_permutation(permutation[0],permutation[0])
        self.permutation = permutation
        return
#
    def load_environment(self,location, side='right',charge_shift=[0,0]):
        #TODO: GENERALIZE TO ALL CONSERVATION LAWS SO THAT IT IS LOADED MORE SMOOTHLY
        """
        loads environment on from old tenpy2 code
        name:   Str, the file name in which the environment is saved
        location: Int, sets location of the site at which environment is located (-1 for left and len(sites) for right usually) 
        side: Str, if right loads right side, otherwise loads left side


        TODO: NEED TO ENSURE WE HAVE THE CORRECT NUMBER OF SITE
        """
        permute = self.permutation[0]
        print("loading "+side+ " environment",'..'*20)
        if side =='right':
            Bflat = self.dict_R['RP']
            qflat_list_c = self.dict_R['RP_q']
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
            Bflat = self.dict_L['LP']
            qflat_list_c = self.dict_L['LP_q']
            #transpose bflat and qflat to make legs consistent with TeNpy3
            Bflat=np.transpose(Bflat, (2, 0, 1))
            #permute to be consistent with MPO
            Bflat=Bflat[:,permute,:]
            qflat_list_c[0]=qflat_list_c[0][permute]
            qflat_list=[qflat_list_c[2],qflat_list_c[0],qflat_list_c[1]]
            labels=['vR*', 'wR', 'vR']
            conj_q=[1,-1,-1]
            shift=3
        #create site at the end of the chain
        site=QH_MultilayerFermionSite_final(N=1,root_config=self.root_config,conserve=self.conserve,site_loc=location)
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
        if self.verbose > 1:
            print(environment)
        if self.verbose> 0 :
            print("environment is loaded",'..'*20)
        return environment
    ######
    #helper functions
    def sanity_check_permutation(self,permute,permute2):
        '''
        Checking the permutation; ie that B matrices match in both cases
        For multicomponent systems we need multiple permutations in general.
        '''
        #permute old Bflat and check its the same as new Bflat for hamiltonian
        
        Bflat_old = self.Bflat_old
        B_flat_new = self.model.H_MPO._W[0]#the b flat of the new mpo
        B_flat_new = B_flat_new.to_ndarray()
        #
        #DIFFERENT PERMUTATION SINCE DIFFERENT BASIS ON DIFFERENT LEGS FOR NON TRIVIAL LENGTH OF MPO
        Bflat_old=Bflat_old[permute,:,:,:]
        Bflat_old=Bflat_old[:,permute2,:,:]

        er=np.sum(( Bflat_old-B_flat_new)**2)
        er2=np.sum(( Bflat_old+B_flat_new)**2)
        thresh=10**(-8)
        
        #consistent permutation but need to check differently
        #TODO: better check of chemical potential term!
        counter=0
        for i in range(len(Bflat_old)):
            #one element can differ, ie chemical potential
            for j in range(len(Bflat_old)):
                crit=np.sum((Bflat_old[j,i]-B_flat_new[j,i])**2)
                if crit>0.00001:
                    counter+=1
                    #remove chemical potential difference
                    if counter==1:
                        er-=crit
        print('MPO missmatch:',er/er2)
        if er/er2<thresh:
            print('permutation is consistent')
        else:
            raise ValueError("inconsistent permutation") 
        return 
    def check_environment_location(self):
        '''
        helps identify where to cut the MPS to match the bond at thiwch we have Left/RIght environmnets
        '''
        print('LHS B:')
        for i in range(len(self.psi_halfinf_L._B)):
            print('site',i,self.psi_halfinf_L._B[i].shape)
        print('RHS B:')
        for i in range(len(self.psi_halfinf_R._B)):
            print('site',i,self.psi_halfinf_R._B[i].shape)
        return
    
    def add_sites_to_L(self,legL,Sleft,leg_physical,numsites):
        '''
        Creates the B flat and singular values for the sites to be added to the system
        ------------------------------------------------------------------------------
        Input:
        legL: The leg connected to the left part of the added sites
        leg_physical: The physical legs for the extra sites
        numsites(int): Number of  extra sites
        
        Output:
        Bs
        Ss

        Used to construct the total MPS = LHS_MPS + ADDED_SITES_MPS
        '''

        charges_left = legL.to_qflat()
        print('left MPS charges',len(charges_left))
        if len(leg_physical) != numsites:
            raise ValueError
        #gives corresponding labels to the environment
        Bs = []
        Qs = []
        Ss = []
        #create first leg part
        for site in range(numsites):
            qflat=[] #qflat to the right of site
            Ss_temp = []#temporary singular values to the right of site
            ch_physical = leg_physical[site].to_qflat() #physicall charges
            if site == 0:
                leg_to_the_left_charges = legL.to_qflat()
                Ss_temp_left = Sleft
            else:
                leg_to_the_left_charges = Qs[site-1]
                Ss_temp_left = Ss[site - 1]
            for j,charge in enumerate(leg_to_the_left_charges):
                charge_to_the_right = charge + ch_physical[0] # does not add a particle
                qflat.append(charge_to_the_right)
                Ss_temp.append(Ss_temp_left[j])
                charge_to_the_right = charge + ch_physical[1] # adds a particle
                qflat.append(charge_to_the_right)
                Ss_temp.append(Ss_temp_left[j])
            qflat=np.array(qflat)
            Ss_temp = np.array(Ss_temp)
            sorted_charges = np.lexsort(np.array(qflat).T) 
            qflat= qflat[sorted_charges]
            Ss_temp = Ss_temp[sorted_charges]
            ####################################################
            ##################### now fill Bs ##################
            ####################################################
            Bflat = np.zeros((len(leg_to_the_left_charges),2,len(qflat)))
            print(Bflat.shape)
            for j,charge in enumerate(leg_to_the_left_charges):
                charge_to_the_right = charge + ch_physical[0]
                ind=np.where((qflat==charge_to_the_right).all(axis=1))[0]
                Bflat[j,0,ind]+=np.random.random(len(ind))
                charge_to_the_right = charge + ch_physical[1]
                ind=np.where((qflat==charge_to_the_right).all(axis=1))[0]
                Bflat[j,1,ind]+=np.random.random(len(ind))
            ####################################################
            ##################### now fill Ss ##################
            ####################################################
            #Alternative choice: get random singular values
            Ss_temp = np.random.random(Bflat.shape[2])
            #get singular values that are a continuation of the previous one. ie if Q to the left has singular value S, then Q'=Q+qphys to the right will have singular value S
            Ss.append(Ss_temp/np.sqrt(np.sum(Ss_temp**2))) # these are the singular values to the *RIGHT* of my site. usual notation would have them to the left.
            Bs.append(Bflat)
            Qs.append(qflat)
            #plt.imshow(np.abs(Bflat[:,0,:]),cmap='coolwarm')
            #plt.colorbar()
            #plt.title('added_Bmatrix')
            #plt.savefig('/mnt/users/kotssvasiliou/tenpy/NEW_COMMITS/figures/MPS_figures/B_'+str(site)+'.png',dpi=500)
        ###
        '''
        ch_physical=leg_physical_site.to_qflat()[0]
        ch_physical2=leg_physical_site.to_qflat()[1]
        for i,charge in enumerate(L):
            charge_2=charge+ch_physical
            qflat.append(charge_2)
            charge_2=charge+ch_physical2
            qflat.append(charge_2)
        #remove duplicates
        qflat = list(map(list, set(map(tuple, qflat))))
        qflat=np.array(qflat)
        #sort charges
        sorted_charges = np.lexsort(np.array(qflat).T) 
        qflat= qflat[sorted_charges]
        #duljina_RMPS=len(qflat)
        print("lenght:",duljina_RMPS)
        Bflat=np.zeros(duljina_left*duljina_physical*duljina_RMPS)  
        Bflat=np.reshape(Bflat,(duljina_left,duljina_physical,duljina_RMPS))
        for i,charge in enumerate(L):
            charge_2=charge+ch_physical
            ind=np.where((qflat==charge_2).all(axis=1))[0]
            Bflat[i,0,ind]+=np.random.random(len(ind))

            charge_2=charge+ch_physical2
            ind=np.where((qflat==charge_2).all(axis=1))[0]
            Bflat[i,1,ind]+=np.random.random(len(ind))
        Bs.append(Bflat)
        #Bflat[:,0,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
        #Bflat[:,1,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
        #set left leg to be equal to qflat
        #CREATE FIRST SITE OF THE BOUNDAR
        print('duljina opet',duljina_left)
        for i in range(len(pstate)):
            #get the correct site etc
            leg_physical_site=leg_physical[i+1]
            Bflat=np.zeros(duljina_left*duljina_physical*duljina_left)  
            Bflat=np.reshape(Bflat,(duljina_left,duljina_physical,duljina_left))
            qflat=[]
            if pstate[i]=='empty':
                #add 0 on our site
                ch_physical=leg_physical_site.to_qflat()[0]
                for m,charge in enumerate(L):
                    charge_2=charge+ch_physical
                    qflat.append(charge_2)
                #sort qflat2
                qflat=np.array(qflat)
                sorted_charges = np.lexsort(np.array(qflat).T) 
                qflat= qflat[sorted_charges]
                for m,charge in enumerate(L):
                    charge_2=charge+ch_physical
                    ind=np.where((qflat==charge_2).all(axis=1))[0]
                    #Bflat[m,0,ind]+=np.random.random(len(ind))
                Bflat[:,0,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
                Bflat[:,1,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
                #ALTERNATIVELY JUST FILL ALL OF THEM RANDOMLY SO YOU SHOULD NOT HAVE A PROBLEM AT ALL
                #print(np.random.random(len(ind)))
            elif pstate[i]=='full':
                #add one on our site
                ch_physical=leg_physical_site.to_qflat()[1]
                for m,charge in enumerate(L):
                    charge_2=charge+ch_physical
                    #ind=np.where((L==charge).all(axis=1))[0]
                    #Bflat[m,0,ind]+=np.random.random(len(ind))
                    qflat.append(charge_2)
                #GETS qflat, and sorts it out!
                #sort qflat
                qflat=np.array(qflat)
                sorted_charges = np.lexsort(np.array(qflat).T) 
                qflat= qflat[sorted_charges]
                for m,charge in enumerate(L):
                    charge_2=charge+ch_physical
                    ind=np.where((qflat==charge_2).all(axis=1))[0]
                    #Bflat[m,1,ind]+=np.random.random(len(ind))
                Bflat[:,0,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
                Bflat[:,1,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
                #update the charges
                #Bflat[:,1,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
            L=np.array(qflat)
            Bs.append(Bflat)
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
            #Bflat[i,0,ind]+=np.random.random(len(ind))
            #print(np.random.random(len(ind)))
        print('left-right charges')
        ch_physical=leg_physical_site.to_qflat()[1]
        for i,charge in enumerate(L):
            charge_2=charge+ch_physical
            #produce sites only when these match!!!
            ind=np.where((R==charge_2).all(axis=1))[0]
            print(charge_2)
            print(ind)
            #Bflat[i,1,ind]+=np.random.random(len(ind))
        Bflat[:,0,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
        Bflat[:,1,:]+=np.random.random((Bflat.shape[0],Bflat.shape[2]))
        print(R)
        Bs.append(Bflat)
        return Bs
        '''
        return Bs,Ss
    def add_sites_to_L_with_pstate(self,legL,Sleft,leg_physical,numsites):
        '''
        Creates the B flat and singular values for the sites to be added to the system
        ------------------------------------------------------------------------------
        Input:
        legL: The leg connected to the left part of the added sites
        leg_physical: The physical legs for the extra sites
        numsites(int): Number of  extra sites
        
        Output:
        Bs
        Ss

        Used to construct the total MPS = LHS_MPS + ADDED_SITES_MPS
        '''

        charges_left = legL.to_qflat()
        print('left MPS charges',len(charges_left))
        if len(leg_physical) != numsites:
            raise ValueError
        #gives corresponding labels to the environment
        Bs = []
        Qs = []
        Ss = []
        #create first leg part
        for site in range(numsites):
            qflat=[] #qflat to the right of site
            Ss_temp = []#temporary singular values to the right of site
            ch_physical = leg_physical[site].to_qflat() #physicall charges
            if site == 0:
                leg_to_the_left_charges = legL.to_qflat()
                Ss_temp_left = Sleft
            else:
                leg_to_the_left_charges = Qs[site-1]
                Ss_temp_left = Ss[site - 1]
            for j,charge in enumerate(leg_to_the_left_charges):
           
                if site!=0 and site!=numsites-1:
                    if self.pstate[site-1]=='full':
                        charge_to_the_right = charge + ch_physical[1] # adds a particle
                        qflat.append(charge_to_the_right)
                        Ss_temp.append(Ss_temp_left[j])
                    elif self.pstate[site-1]=='empty':
                        
                        charge_to_the_right = charge + ch_physical[0] # does not add a particle
                        qflat.append(charge_to_the_right)
                        Ss_temp.append(Ss_temp_left[j])
                    elif self.pstate[site-1]=='':
                        #print('INININ')
                        charge_to_the_right = charge + ch_physical[0] # does not add a particle
                        qflat.append(charge_to_the_right)
                        Ss_temp.append(Ss_temp_left[j])
                        charge_to_the_right = charge + ch_physical[1] # does not add a particle
                        qflat.append(charge_to_the_right)
                        Ss_temp.append(Ss_temp_left[j])
                else:
                    charge_to_the_right = charge + ch_physical[0] # does not add a particle
                    qflat.append(charge_to_the_right)
                    Ss_temp.append(Ss_temp_left[j])
                    charge_to_the_right = charge + ch_physical[1] # adds a particle
                    qflat.append(charge_to_the_right)
                    Ss_temp.append(Ss_temp_left[j])
            qflat=np.array(qflat)
            Ss_temp = np.array(Ss_temp)
            sorted_charges = np.lexsort(np.array(qflat).T) 
            qflat= qflat[sorted_charges]
            Ss_temp = Ss_temp[sorted_charges]
    
            ####################################################
            ##################### now fill Ss ##################
            ####################################################
            #Alternative choice: get random singular values
            Ss_temp = np.random.random(len(qflat))

            #Ss_temp[self.shift_import[0]] = 1.2
            #get singular values that are a continuation of the previous one. ie if Q to the left has singular value S, then Q'=Q+qphys to the right will have singular value S
            Ss.append(Ss_temp/np.sum(Ss_temp**2)) # these are the singular values to the *RIGHT* of my site. usual notation would have them to the left.
            
            Qs.append(qflat)
            #plt.imshow(np.abs(Bflat[:,0,:]),cmap='coolwarm')
            #plt.colorbar()
            #plt.title('added_Bmatrix')
            #plt.savefig('/mnt/users/kotssvasiliou/tenpy/NEW_COMMITS/figures/MPS_figures/B_'+str(site)+'.png',dpi=500)
        ###
      
      
        return Qs,Ss



a=int(sys.argv[1])
b=int(sys.argv[2])
####
sides_added = 3
params = {}
params['verbose'] = 1
params['sys length'] = 120
params['unit cell'] = 3#in the case of q=3 laughlin
params['sites added'] = 9#sides added in the middle
#params['model data file'] = "/mnt/users/kotssvasiliou/tenpy_data/laughlin_haldane/Data.pkl"
params['model data file'] = "/mnt/users/dperkovic/quantum_hall_dmrg/data_load/interface_test/Data.pkl"

#params['left environment file'] = "/mnt/users/kotssvasiliou/tenpy_data/laughlin_haldane/env_L.npz"
#params['right environment file'] = "/mnt/users/kotssvasiliou/tenpy_data/laughlin_haldane/env_R.npz"


params['left environment file'] = "/mnt/users/dperkovic/quantum_hall_dmrg/data_load/interface_test/env_L.npz"
params['right environment file'] = "/mnt/users/dperkovic/quantum_hall_dmrg/data_load/interface_test/env_R.npz"
QHsys = QH_system(params=params)

QHsys.shift_import=[a,b]
#
repeat = 10
for _ in range(repeat):
    QHsys.patch_WFs()
#
quit()
QHsys.patch_WFs(num=1)
#QHsys.add_sides_to_Left_MPS(num=3)
quit()
timei = time.time()
QHsys.patch_WFs()
timef = time.time()
#print('patching time:',timef-timei)
quit()
timei = time.time()
QHsys.patch_WFs()
timef = time.time()
print('patching time:',timef-timei)
quit()
#QHsys.check_environment_location()
####
#dmrg setup#
init_env_data_halfinf={}
init_env_data_halfinf['init_LP'] = QHsys.left_env   #DEFINE LEFT ENVIROMENT
init_env_data_halfinf['age_LP'] =0
init_env_data_halfinf['init_RP'] = QHsys.right_env   #DEFINE RIGHT ENVIROMENT
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


quit()
eng_halfinf = dmrg.TwoSiteDMRGEngine(QHsys.psi, QHsys.model,dmrg_params,resume_data={'init_env_data': init_env_data_halfinf})