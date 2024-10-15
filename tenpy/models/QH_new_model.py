import numpy as np

from .model import MPOModel, NearestNeighborModel
from .lattice import Chain
from ..networks.site import QH_MultilayerFermionSite_2
from ..networks.site import QH_MultilayerFermionSite
import numpy as np
from .model import MPOModel, NearestNeighborModel
from .lattice import Chain

from tenpy.networks.mpo import MPO
__all__ = ['QHModel', 'QHChain']



class QHChain( MPOModel):
    def __init__(self,H_MPO, L=2 ):
        spin =QH_MultilayerFermionSite(N=1)# SpinSite(S=S, conserve="Sz")
        # the lattice defines the geometry
    
        lattice = Chain(L, spin, bc="periodic", bc_MPS="infinite")
      
        a=lattice.mps_sites()
        print(a)
        quit()
        # finish initialization
        # generate MPO for DMRG
        MPOModel.__init__(self, lattice, H_MPO)
        # generate H_bond for TEBD
        #NearestNeighborModel.__init__(self, lattice, self.calc_H_bond())
