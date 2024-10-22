"""Code for running DMRG """
import sys
import os
import numpy as np
sys.path.append('C:/Users/Kotsios2/Documents/GitHub/tenpy') 
from tenpy.linalg import np_conserved as npc
#from tenpy.models import multilayer_qh_DP_final as mod
import itertools
from tenpy.networks.mps import MPS
from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import QH_MultilayerFermionSite_3
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
import QH_G2MPO
#import QH_Graph_final

'''
Test example for the creation of sites:
While the *sites* variable used for creating the MPO will all be in location 0 (as the unit cell = # of components in most iDMRG cases)
the *Lattice* instance shoudl be made up for different sites with appropriate charges
'''



'''
To do:
1) Add translation functions
2) How do you actually create the *lattice* out of a bunch of sites?

'''
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
for i in range(10):
    spin=QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=('N','K'),site_loc=i)
    print('---------------')
    print('Charges of HIlbert space at location'+str(i)+':')
    print(spin.leg)
# so in principle one can create a chain like:
Chain = []
for i in range(10):
    Chain.append(QH_MultilayerFermionSite_3(N=1,root_config=root_config_,conserve=('N','K'),site_loc=i))
quit()
lattice=Chain(N,spin, bc="periodic",  bc_MPS="infinite")