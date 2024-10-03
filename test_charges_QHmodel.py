'''
Here I am creating the Hamiltonian Hilbert space and checking the charges of the onsite operators
'''
import sys
import os
sys.path.append('/home/v/vasiliou/tenpynew2/tenpy') # add your own path here
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models import multilayer_qh_DP_final as mod
import itertools


from tenpy.networks.mps import MPS
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import QH_MultilayerFermionSite_2
from tenpy.networks.site import kron

root_config = np.array([0,1,0])
root_config = root_config.reshape(3,1)
Site = QH_MultilayerFermionSite_2(root_config,N=1,conserve='N')
print('Qflat vector')
print(Site.leg.to_qflat())
print('-'*50)
print('Onsite operators:')
print(Site.opnames)
print('-'*50)
print('Are the newly created local operators npc instances?',isinstance(Site.aOp,npc.Array))
print('-'*50)
print('Annihilation operator:')
print(Site.aOp)
print('-'*50)
print('Create Bond operators:')
print('-'*50)
AOp_l = kron(Site.aOp,Site.Id)
AOp_r = kron(Site.Id,Site.aOp)
bond_ops = dict(AOp_l=AOp_l,AOp_r=AOp_r)
print('Are the newly created bond operators npc instances?',isinstance(AOp_l,npc.Array))
print('-'*50)
print('How do they look?')
print('AOp_l=',AOp_l)

'''
To do:
1) Fix JW string charge
2) Where do the bond operators play a role?
3) Understand MPO processes
'''
