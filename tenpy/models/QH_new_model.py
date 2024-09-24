import numpy as np

from .model import MPOModel, NearestNeighborModel
from .lattice import Chain
from ..networks.site import SpinHalfSite

import numpy as np
from .model import MPOModel, NearestNeighborModel
from .lattice import Chain
from ..networks.site import SpinHalfSite
from tenpy.networks.mpo import MPO
__all__ = ['QHModel', 'QHChain']

class QHModel(MPOModel):
    def __init__(self, model_params,  lattice,H_MPO):
        """
        Initialize the QHModel with model parameters, the Hamiltonian MPO, and the lattice.

        Parameters:
        - model_params: dict, the parameters for the model.
        - H_MPO: Matrix Product Operator for the Hamiltonian.
        - lattice: The lattice structure (e.g., Chain) for the model.
        """
        

        # Store the model parameters, Hamiltonian MPO, and lattice locally
        self.model_params = model_params
        self.H_MPO = H_MPO
        self.lattice = lattice

        # Initialize sites based on the model parameters
        self.sites =[]
        for i in range(model_params["L"]):
            self.sites.append(self.init_sites(model_params))
        # Initialize the parent class MPOModel with the given parameters and lattice

        HMPO=MPO(self.sites,H_MPO)
        super().__init__(lattice,HMPO)
        
        # Initialize terms (the Hamiltonian MPO is already provided)
        self.init_terms(HMPO)

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity')
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', True)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, H_MPO):
        """Set up the Hamiltonian terms using the given MPO."""
        self.H_MPO = H_MPO  # Store the MPO for the Hamiltonian


class QHChain(QHModel, NearestNeighborModel):
    """The QHModel on a Chain, suitable for TEBD.

    See the :class:`QHModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True

    def __init__(self, model_params, H_MPO):
        """
        Initialize the QHChain with model parameters and the Hamiltonian MPO.

        Parameters:
        - model_params: dict, the parameters for the model.
        - H_MPO: Matrix Product Operator for the Hamiltonian.
        """
        # Set up the chain lattice
        lattice = Chain(model_params["L"], None, bc='periodic')  # Assuming 'L' is the chain length

        # Call the parent class (QHModel) constructor with model_params, H_MPO, and lattice
        super().__init__(model_params,  lattice,H_MPO)