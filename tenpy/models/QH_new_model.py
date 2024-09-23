import numpy as np

from .model import MPOModel, NearestNeighborModel
from .lattice import Chain
from ..networks.site import SpinHalfSite

__all__ = ['QHModel', 'QHChain']
class QHModel(MPOModel):
    def __init__(self, model_params, H_MPO):
        """
        Initialize the QHModel with model parameters and the Hamiltonian MPO.

        Parameters:
        - model_params: dict, the parameters for the model.
        - H_MPO: Matrix Product Operator for the Hamiltonian.
        """
        # Initialize the parent class MPOModel with the given parameters
        super().__init__(model_params, H_MPO)

        # Store the model parameters and Hamiltonian MPO locally
        self.model_params = model_params
        self.H_MPO = H_MPO

        # Initialize sites based on the model parameters
        self.sites = self.init_sites(model_params)

        # Initialize terms (the Hamiltonian MPO is already provided)
        self.init_terms(H_MPO)

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'parity', str)
        assert conserve != 'Sz'
        if conserve == 'best':
            conserve = 'parity'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', True, bool)
        site = SpinHalfSite(conserve=conserve, sort_charge=sort_charge)
        return site

    def init_terms(self, H_MPO):
        """Set up the Hamiltonian terms using the given MPO."""
        self.H_MPO = H_MPO  # Store the MPO for the Hamiltonian


class QHChain(QHModel, NearestNeighborModel):
    """The :class:`TFIModel` on a Chain, suitable for TEBD.

    See the :class:`TFIModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True

    def __init__(self, model_params, H_MPO):
        # Call the parent class (QHModel) constructor with model_params and H_MPO
        super().__init__(model_params, H_MPO)
