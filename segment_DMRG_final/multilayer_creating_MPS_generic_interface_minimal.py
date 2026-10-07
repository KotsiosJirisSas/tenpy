#!/usr/bin/env python
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys
print(sys.version)

sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 

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
#from pf_apf_newest2 import trim_down_the_MPO,eliminate_elements_from_the_graph,get_old_basis
#from pf_apf_newest import set_MPS_boundary # adds 2+pstate sites/ pf apf 2 adds 1+pstate sites
####

#patching of LHS and RHS (+ added sites in between)
from patch_mps import patch_mps
#vacuum on the LHS (+ pad), used for type_calculation='vacuum_interface'
from vacuum_interface import vacuum_lhs, vacuum_lhs_window

print('='*100)
print('succesfull import of all modules')
print('='*100)
####


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
        self.verbose=params['verbose']
        self.model_params = params['model_params']
        self.nlayer = len(self.model_params['layers'])
        self.root_config = self.model_params['root_config'].reshape(self.model_params['root_config'].size//self.nlayer, self.nlayer)
        self.unit_cell = self.root_config.shape[0]
        self.system_length=params['sys length']
        self.interface_position=params['interface_position']
        # conservation laws, same cons_K / cons_C block as in make_iMPO and create_segment_DMRG_model
        # (the sites of the MPS must have the same ChargeInfo as the sites of the MPO)
        if self.model_params['cons_K']:
            if self.model_params['cons_C']=='total':
                self.conserve=('N','K')
            else:
                self.conserve=(self.model_params['cons_C'],'K')
        else:
            self.conserve=self.model_params['cons_C']
        self.order_of_species=None
        if self.model_params['cons_C']=='species':
            self.order_of_species=self.model_params['order_of_species']
        self.type_calculation = params['type_calculation']
        
        #not needed parameter
        self.pstate = []#,'','full']#params['sites added']
        for i in range(params['sites added']-2):
            self.pstate.append('')
       
        if  self.type_calculation!='vacuum_interface':
            self.avg_filling = self.nlayer*np.sum(self.root_config)/self.root_config.size
      
        
        #self.model_par['boundary_conditions']= ('infinite', self.system_length)
        self.sides_added = params['sites added']
        self.shift_import=[]
        
        self.load_mps()
        time1 = time.time()
        if self.verbose>0:
            print('='*100)
            print("INITIALIZATION TIME:",time1-time0)
            print('='*100)
        if self.verbose>0:
            print('='*100)
            print("creating hilbert space")
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
        if self.type_calculation!='vacuum_interface':
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
    
    def create_sites(self,add=0):
        """
        TODO: This function has conservation of momentum hard-coded
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
       
        
        sites=[]
        #produce sites
        for i in range(-add,self.system_length-add):
            spin=QH_MultilayerFermionSite_final(N=self.nlayer,root_config=self.root_config,conserve=self.conserve,site_loc=i,order_of_species=self.order_of_species)
            sites.append(spin)
        self.sites=sites
        return 

    def patch_WFs(self,save_dir='/mnt/users/kotssvasiliou/tenpy/NEW_COMMITS/segDMRG_results/coulomb/',max_pad=None,max_offset=0):
        '''
        Given the half-inf left and right MPSs, patch them together.
        Can insert cells to move us to different (N,K) sectors

        charge_condition:           For multilayer case do you allow Q1 != 0 and Q2 != 0?

        The actual patching is done in patch_mps.py. The modes only differ in what was loaded
        (LHS, RHS different or the same) and in whether the shift is imported or determined
        from the Schmidt values. Number of added sites == 0 is a special case.

        vacuum_interface: the LHS is empty (all 0), shift_import is the TARGET CHARGE (K,n_1,..,n_nlayer).
        sites added > 0: a window of random charge-conserving sites between the vacuum and the RHS absorbs the charge
        (it can hold more or fewer particles than the neutral number, so adding and removing are symmetric).
        sites added = 0: the RHS is projected onto the sector with this charge; if there is none, the right end of the
        vacuum is padded with 1010... (see vacuum_interface.py). max_pad / max_offset limit that search.
        '''
        if self.type_calculation=='vacuum_interface':
            if len(self.shift_import)==0:
                raise ValueError('vacuum_interface needs the target charge (K,n_1,...) in shift_import')
            if self.sides_added==0:
                # no window: the right end of the vacuum is padded with 1010... so that the target can be reached
                self.psi_halfinf_L, self.psi_halfinf_R, shift, self.pad = vacuum_lhs(self.sites, self.interface_position, self.nlayer,
                                                        self.psi_halfinf_R, self.shift_import, max_pad, max_offset)
            else:
                # window of added sites between the empty vacuum and the right piece, the window absorbs the charge
                self.psi_halfinf_L, shift = vacuum_lhs_window(self.sites, self.interface_position, self.shift_import)
        else:
            #shift imported (e.g. excitations) or None, then it is determined from the Schmidt values
            shift = self.shift_import if len(self.shift_import)>0 else None
        psi, shift = patch_mps(self.psi_halfinf_L, self.psi_halfinf_R, self.sites, self.sides_added, shift)
        self.shift_final = shift
        print('shift',shift)
        print('charge on the right end of the segment',psi._B[-1].get_leg('vR').to_qflat()[0])

        filling= psi.expectation_value("nOp")
        print('filling',filling)
        print('testing the WF','...'*10)
        for l in range(self.nlayer):
            print('total number of electrons in layer',l,':',np.sum(filling[l::self.nlayer]))
        
        if self.type_calculation=='vacuum_interface':
            self.avg_filling=np.sum(filling)/len(filling)
        
      
        
        K=np.sum(np.arange(len(filling))//self.nlayer*(filling-self.avg_filling/self.nlayer))
        print("K sector:",K)
       
        """
        NOTE: needed only for sanity checking the charges on Pf-APf interface. redundant now.
        ### for reference. not useful otherwise.
        excess = np.sum(filling-self.avg_filling/self.nlayer)
        print('total leftover number of electrons',excess)
        if (np.abs(excess)<0.01):
            with open(save_dir+'L='+str(self.system_length)+'_multilayer_interface_='+str(self.sides_added)+'.txt','a' ) as f:
                f.write('shift:'+str(shift)+"\n")
                f.write('excess number of electrons:'+str(excess)+"\n")
                f.write("K sector: "+str(K)+"\n")
        """
        self.psi_merged = psi
        return psi
        
     
    #####################################
    #####################################
    ####### LOADING RELATED FUNCS #######
    #####################################
    #####################################
    def load_mps(self):
        '''
        Loads the pickled parameter file, specifically the model parameters and root config.
        Input:
        name(str):      Path of pickle file
        '''
        
        with open(self.data_location, 'rb') as f:
            loaded_xxxx = pickle.load(f)

        ##########################################
        #mps dictionaries#
        if self.type_calculation=='excitation':
            self.loaded_MPS_L = loaded_xxxx['mps']

            self.loaded_MPS_R = self.loaded_MPS_L.copy()
        if self.type_calculation=='vacuum_interface':
            self.loaded_MPS_L = None
            self.loaded_MPS_R = loaded_xxxx['mps']
        if self.type_calculation=='interface':
            self.loaded_MPS_L = loaded_xxxx['mps']
            self.loaded_MPS_R = loaded_xxxx['mps_2']
        return
    def load_data(self,MPSshift=2,side='right',charge_shift=[]):
        """
        Loads the infinite MPS (TeNPy3 object) and cuts it into a segment MPS
        living on the corresponding part of self.sites.

        Input:
        MPSshift:(int)  How much to shift the RHS MPS to the right by, to insert an interface in between
        side:           'left' or 'right'
        charge_shift:   charge offsets [K, n_1, ..., n_nlayer] added to the left leg of the MPS

        Uses:           self.loaded_MPS_L / self.loaded_MPS_R, self.sites, self. interface_position, self.pstate
        Sets:           self.psi_halfinf_L or self.psi_halfinf_R
        """
        if len(charge_shift)==0:
            charge_shift=np.zeros(self.nlayer+1)

        interface_position=self.interface_position
        if side == 'right':
            psi_MPS = self.loaded_MPS_R.copy()

            # system = [LHS]|[added piece][RHS], the added piece is -[]-[pstate]-[]-
            # open question: why the +2?
            sites = self.sites[interface_position+len(self.pstate)+MPSshift:]
            #print('og',self.sites[0].leg)
            #print('shifted',sites[0].leg)

            # to shift by the correct number of sites we have to divide by nlayer
            shift = (interface_position+len(self.pstate)+MPSshift)//self.nlayer

            # NOTE: this second pair of assignments overrides the one above
            # (the added sites are not included in the shift)
            sites = self.sites[interface_position+MPSshift:]
            #print('og',self.sites[0].leg)
            #print('shifted',sites[0].leg)
            shift = (interface_position+MPSshift)//self.nlayer

        elif side == 'left':

            psi_MPS = self.loaded_MPS_L.copy()
            sites = self.sites[:interface_position]
            shift = 0
        else:
            raise ValueError
        L=len(sites)

        # psi_MPS is the TeNPy3 infinite MPS, we read one unit cell from it:
        # B tensors as flat arrays with legs (p, vL, vR)
        Bflat0 = [np.transpose(psi_MPS.get_B(i,form='B').to_ndarray(),(1,0,2)) for i in range(psi_MPS.L)]
        # left Schmidt values of every site in the unit cell
        Ss = [np.array(psi_MPS.get_SL(i)) for i in range(psi_MPS.L)]
        # charges of the left leg of the first site (only the first leg is needed)
        qflat2 = psi_MPS.get_B(0,form='B').get_leg('vL').to_qflat()

        #print('Qs shapes (at bonds)',len(qflat2),len(qflat2[0]),len(qflat2[1]),len(qflat2[2]))
        print(len(Bflat0))
        print(len(qflat2))

        print('B FLAT SHAPE:',Bflat0[0].shape,Bflat0[1].shape,Bflat0[2].shape)

        # translate the charges of the left leg:
        # the first charge (momentum) is shifted by the translation of the MPS,
        # the remaining charges stay as they are
        qflat=[]
        print(shift)
        for i in range(len(qflat2)):
            kopy=[]
            for m in range(len(qflat2[i])):
                if m==0:
                    shifted=qflat2[i][0]
                    for z in range(self.nlayer):
                        shifted+=shift*qflat2[i][1+z]# momentum shift of the first charge sector
                    kopy.append(shifted)
                else:
                    kopy.append(qflat2[i][m])
            qflat.append(kopy)

        # add the requested charge offsets on top
        print("shift charges. The charge shift is",charge_shift)
        for i in range(len(qflat)):
            qflat[i][0]+=charge_shift[0]
            for z in range(self.nlayer):
                qflat[i][z+1]+=charge_shift[z+1]

        #print('Qs shapes (after)',len(qflat),len(qflat[0]),len(qflat[1]),len(qflat[2]))
        print(len(qflat))

        qflat=np.array(qflat)

        #print('qflat_shifted',qflat)

        # repeat the unit cell until it covers the segment
        infinite_unit_cell=len(Bflat0)
        print(infinite_unit_cell)
        number=len(sites)//infinite_unit_cell+1

        Bflat=Bflat0*number
        # Schmidt values: repeat the same way, a segment needs one extra value for the rightmost leg
        Ss=Ss*number
        Ss.append(Ss[0])

        # cut the MPS to the wanted size
        Bflat=Bflat[:L]
        Ss=Ss[:L+1]

        # leftmost leg of charges
        chargeinfo=sites[0].leg.chinfo

        left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]

        # create the segment MPS, the charges on all bonds follow from the left leg
        print('INININ')
        print(side)
        print('loading the MPS')
        print(len(sites))

        mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='segment',legL=left_leg)
        print('loaded mps from data',".."*30)

        if side == 'right':
            self.psi_halfinf_R = mps
        elif side == 'left':
            self.psi_halfinf_L = mps

        return