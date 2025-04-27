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

        self.system_length = params['sys length']#what Dom calls L in his code
        self.verbose = params['verbose']
        self.unit_cell = params['unit cell']
        self.avg_filling = params['avg_filling']
        self.half= params['half']
        self.nlayer=params['N_layer']
        #not needed parameter
        self.pstate = []#,'','full']#params['sites added']
        for i in range(params['sites added']-2):
            self.pstate.append('')
        #self.pstate=['empty','full','empty','full','empty','full','empty']
        self.load_model_params()
        #self.graph = [self.graph[0]]*self.system_length #total graph is graph x sys length
        #Q: for multicomponent systems have to change this no? ie --> self.graph[:#_of_components]
        self.model_par['boundary_conditions']= ('infinite', self.system_length)
        #self.model_par['layers'] = [('L',self.LL)]
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
        self.create_sites()
        time2 = time.time()
        if self.verbose>0:
            print('='*100)
            print("MPO CONSTRUCTION TIME:",time2-time1)
            print('='*100)
   
        time3 = time.time() 
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
    

    
    def create_sites(self,add=False):
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
       
        if not add:
            sites=[]
            for i in range(self.system_length):
                #produce a single site
                spin=QH_MultilayerFermionSite_final(N=self.nlayer,root_config=self.root_config,conserve=('each','K'),site_loc=i)
                sites.append(spin)
        else:
            sites=[]
            #produce sites
            for i in range(-add,self.system_length-add):
                spin=QH_MultilayerFermionSite_final(N=self.nlayer,root_config=self.root_config,conserve=('each','K'),site_loc=i)
                sites.append(spin)
        self.sites=sites
        return 

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
      
        Qflat_added,Ss_added= self.add_sites_to_L(legL = left_leg,Sleft=left_Ss,leg_physical = leg_physical,numsites=num)
        #PART 3: CONNECT TO RHS

       
        
        #first: how many charges agree?
        charges_LHS = Qflat_added[-1]
        charges_RHS = self.psi_halfinf_R._B[0].get_leg('vL').to_qflat()
        largest_charge_LHS = charges_LHS[np.argmax(np.array(Ss_added[-1]))]
        largest_charge_RHS = charges_RHS[np.argmax(np.array(Ss2[0]))]
     

        charges_LHS = np.array(charges_LHS)
        charges_RHS = np.array(charges_RHS)
        shift = np.array(largest_charge_RHS) - np.array(largest_charge_LHS)
        print(shift)
        #quit()
        #if you import shift instead of determining it via method above
        if len(self.shift_import)>0:
            shift=self.shift_import


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
        if len(matches)<20:return    


        Ss_left_side=self.psi_halfinf_L._S[-1]
        #remove some  Ss and Bs from RHS MPS anc construct again. see if it kills off dead nodes
        S_LHS = Ss_added[-1].copy()
    
        mask = np.isin(np.arange(len(S_LHS)), matches[:,0])
      
        
        #construct the Bflat now!!
        qflat_left=self.psi_halfinf_L._B[-1].get_leg('vR').to_qflat()
        print('length of BS')
        print(len(Qflat_added[-1][mask]))
        #quit()
        Bs_added,Ss_added=self.construct_BS_from_right(leg_physical,num,Ss_added, Qflat_added, qflat_left,Qflat_added[-1][mask], S_LHS[mask],Ss_left_side)
        for i in range(len(Bs_added)):
            Bflat.append(np.transpose(Bs_added[i],(1,0,2)))
            Ss.append(Ss_added[i])
       
        new_length = len(self.psi_halfinf_L._B) + num
            
        ########################
        psi=MPS.from_Bflat(self.sites[:new_length],Bflat[:new_length],SVs=Ss[:new_length+1], bc='segment',legL=self.psi_halfinf_L._B[0].get_leg('vL'))
        #########################
        #psi.canonical_form_finite(cutoff=0.0)
        
        print(psi)
        #now one needs to shift all the charges of the B matrices 
        filling= psi.expectation_value("nOp")
        print('filling',filling)
        print('TOTAL FILLING',np.sum(filling-self.avg_filling/self.nlayer))
        print('new bond dimensions',[len(S) for S in psi._S])
    
        
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
        print('TOTAL leftover number of electrons',np.sum(filling-self.avg_filling/self.nlayer))
        K=np.sum(np.arange(len(filling))//self.nlayer*(filling-self.avg_filling/self.nlayer))
        #print(psi)
        print("K sector:",K)

        print(shift)
        z=np.abs(np.sum(filling-self.avg_filling/self.nlayer))
     
        if z<0.01:
            with open('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy/NEW_COMMITS/L='+str(self.system_length)+"_half="+str(self.half)+'_multilayer_with_interlayer_interaction_Haldane_success='+str(self.sides_added)+'.txt','a') as f:
                f.write('shift:'+str(shift)+"\n")
                f.write('excess number of electrons:'+str(np.sum(filling-self.avg_filling/self.nlayer))+"\n")
                f.write("K sector: "+str(K)+"\n")
      
        print('passed')
        self.psi_merged = psi
        
        return psi
        
     
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
        print(loaded_xxxx.keys()) 
        try:      
            model_par = loaded_xxxx['Model']
        except:
            model_par = loaded_xxxx['Parameters']
        print(model_par['root_config'])
        root_config_ = model_par['root_config'].reshape(self.unit_cell,self.nlayer)
        """
        if model_par['cons_K']:
            conserve = (model_par['cons_C'],'K') 
        else:
            conserve = 'N'
        """
        conserve=('each','K')
        #self.LL = model_par['LL']
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
       
        return
    def load_data(self,MPSshift=2,side='right',charge_shift=[]):
        if len(charge_shift)==0:
            charge_shift=np.zeros(self.nlayer+1)
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
        half=self.half
        if side == 'right':
            psi_MPS = self.loaded_MPS_R.copy()
           
            sites = self.sites[half+len(self.pstate)+MPSshift:]    # why the +2???? system = [LHS]|[added piece][RHS]
                                                            # and the [added piece] is -[]-[pstate]-[]-
            #print('og',self.sites[0].leg)
            #print('shifted',sites[0].leg)

            #to shift by correct number of sites we have to divide by nlayer
            shift = (half+len(self.pstate)+MPSshift)//self.nlayer
            #change this later on
            sites = self.sites[half+MPSshift:]    # why the +2???? system = [LHS]|[added piece][RHS]
                                                            # and the [added piece] is -[]-[pstate]-[]-
            #print('og',self.sites[0].leg)
            #print('shifted',sites[0].leg)
            shift = (half+MPSshift)//self.nlayer
    
        elif side == 'left':
  
            psi_MPS = self.loaded_MPS_L.copy()
            sites = self.sites[:half]
            shift = 0
        else:
            raise ValueError 
        L=len(sites)
        Bflat0 = psi_MPS['MPS_Bs'].copy()
        Ss = psi_MPS['MPS_Ss'].copy()
        qflat2 = psi_MPS['MPS_qflat'].copy()
        # we are taking first leg charges only
        #this bit checks whether the full charge or only left leg is given
        if len(qflat2)==self.unit_cell*self.nlayer:
            qflat2=qflat2[0] 
       
        
        #quit()
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
                    #SHIFT THIS TOO
                    shifted=qflat2[i][0]
                    for z in range(self.nlayer):
                    
                        shifted+=shift*qflat2[i][1+z]# a momentum shift for the first charge sector???
                    kopy.append(shifted)
                else:
                    kopy.append(qflat2[i][m])
            qflat.append(kopy)
        print("shift charges. The charge shift is",charge_shift)
        for i in range(len(qflat)):
            qflat[i][0]+=charge_shift[0]
            for z in range(self.nlayer):
                qflat[i][z+1]+=charge_shift[z+1]  
           
        ####
        #print('Qs shapes (after)',len(qflat),len(qflat[0]),len(qflat[1]),len(qflat[2]))
        print(len(qflat))
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

    ######
    #helper functions

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
         
            #eliminate duplicate charges
            qflat = list(map(list, set(map(tuple, qflat))))   
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

            #get singular values that are a continuation of the previous one. ie if Q to the left has singular value S, then Q'=Q+qphys to the right will have singular value S
            Ss.append(Ss_temp/np.sum(Ss_temp**2)) # these are the singular values to the *RIGHT* of my site. usual notation would have them to the left.
            
            Qs.append(qflat)
           
        return Qs,Ss
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
      
        Bs=Bs[::-1]
        Ss=Ss[::-1]
        
        #for i in range(len(Bs)):
        #    print(Bs[i].shape)
        #    print(Ss[i].shape)
      
        return Bs,Ss[1:]




if __name__ == "__main__":
   
   
    num=int(sys.argv[1])
    b=int(sys.argv[2])
    half=int(sys.argv[3])
    #nu for the state we are matching
    avg_filling=1/2
    avg_filling=2/3

    
   
    #half=(L//6)*6
    #half=(L//6)*6-2
    L=num
    ####
    params = {}
    params['verbose'] = 1
    params['sys length'] = L
    params['half']=half
    params['avg_filling']=avg_filling
    params['N_layer']=2
    #params['unit cell'] = 3#in the case of q=3 laughlin
    params['unit cell'] = 3#in the case of pf-apf
    params['sites added'] = b#sides added in the middle
    #params['model data file'] = "/mnt/users/kotssvasiliou/tenpy_data/laughlin_haldane/Data.pkl"
   
    params['model data file'] = '/mnt/users/dperkovic/quantum_hall_dmrg/data_load/multilayer_with_interlayer_interaction_Haldane/Data.pkl'
  
    QHsys = QH_system(params=params)

    #QHsys.shift_import=[a,b]

    QHsys.patch_WFs()
