"""Nearest-neighbour spin-S models.

Uniform lattice of spin-S sites, coupled by nearest-neighbour interactions.
"""
# Copyright 2018-2023 TeNPy Developers, GNU GPLv3

import numpy as np

from ..networks.site import SpinSite
from .model import CouplingMPOModel, NearestNeighborModel
from .lattice import Chain
from ..tools.params import asConfig

__all__ = ['SpinModel', 'SpinChain', 'DipolarSpinChain']


class SpinModel(CouplingMPOModel):
    r"""Spin-S sites coupled by nearest neighbour interactions.

    The Hamiltonian reads:

    .. math ::
        H = \sum_{\langle i,j\rangle, i < j}
              (\mathtt{Jx} S^x_i S^x_j + \mathtt{Jy} S^y_i S^y_j + \mathtt{Jz} S^z_i S^z_j
            + \mathtt{muJ} i/2 (S^{-}_i S^{+}_j - S^{+}_i S^{-}_j))  \\
            - \sum_i (\mathtt{hx} S^x_i + \mathtt{hy} S^y_i + \mathtt{hz} S^z_i) \\
            + \sum_i (\mathtt{D} (S^z_i)^2 + \mathtt{E} ((S^x_i)^2 - (S^y_i)^2))

    Here, :math:`\langle i,j \rangle, i< j` denotes nearest neighbor pairs.
    All parameters are collected in a single dictionary `model_params`, which
    is turned into a :class:`~tenpy.tools.params.Config` object.

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`SpinModel` below.

    Options
    -------
    .. cfg:config :: SpinModel
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.Site.SpinSite`.
            For ``'best'``, we check the parameters what can be preserved.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        Jx, Jy, Jz, hx, hy, hz, muJ, D, E  : float | array
            Coupling as defined for the Hamiltonian above.

    """
    def init_sites(self, model_params):
        S = model_params.get('S', 0.5)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            # check how much we can conserve
            if not model_params.any_nonzero([('Jx', 'Jy'), 'hx', 'hy', 'E'],
                                            "check Sz conservation"):
                conserve = 'Sz'
            elif not model_params.any_nonzero(['hx', 'hy'], "check parity conservation"):
                conserve = 'parity'
            else:
                conserve = None
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        sort_charge = model_params.get('sort_charge', None)
        site = SpinSite(S, conserve, sort_charge)
        return site

    def init_terms(self, model_params):
        Jx = model_params.get('Jx', 1.)
        Jy = model_params.get('Jy', 1.)
        Jz = model_params.get('Jz', 1.)
        hx = model_params.get('hx', 0.)
        hy = model_params.get('hy', 0.)
        hz = model_params.get('hz', 0.)
        D = model_params.get('D', 0.)
        E = model_params.get('E', 0.)
        muJ = model_params.get('muJ', 0.)

        # (u is always 0 as we have only one site in the unit cell)
        for u in range(len(self.lat.unit_cell)):
            self.add_onsite(-hx, u, 'Sx')
            self.add_onsite(-hy, u, 'Sy')
            self.add_onsite(-hz, u, 'Sz')
            self.add_onsite(D, u, 'Sz Sz')
            self.add_onsite(E * 0.5, u, 'Sp Sp')
            self.add_onsite(E * 0.5, u, 'Sm Sm')
        # Sp = Sx + i Sy, Sm = Sx - i Sy,  Sx = (Sp+Sm)/2, Sy = (Sp-Sm)/2i
        # Sx.Sx = 0.25 ( Sp.Sm + Sm.Sp + Sp.Sp + Sm.Sm )
        # Sy.Sy = 0.25 ( Sp.Sm + Sm.Sp - Sp.Sp - Sm.Sm )
        for u1, u2, dx in self.lat.pairs['nearest_neighbors']:
            self.add_coupling((Jx + Jy) / 4., u1, 'Sp', u2, 'Sm', dx, plus_hc=True)
            self.add_coupling((Jx - Jy) / 4., u1, 'Sp', u2, 'Sp', dx, plus_hc=True)
            self.add_coupling(Jz, u1, 'Sz', u2, 'Sz', dx)
            self.add_coupling(muJ * 0.5j, u1, 'Sm', u2, 'Sp', dx, plus_hc=True)
        # done


class SpinChain(SpinModel, NearestNeighborModel):
    """The :class:`SpinModel` on a Chain, suitable for TEBD.

    See the :class:`SpinModel` for the documentation of parameters.
    """
    default_lattice = Chain
    force_default_lattice = True


class DipolarSpinChain(CouplingMPOModel):
    r"""Dipole conserving H3-H4 spin-S chain.

    The Hamitlonian reads:

    .. math ::
        H = - \mathtt{J3} \sum_{i} (S^+_i (S^-_{i + 1})^2 S^+_{i + 2} + \mathrm{h.c.})
            - \mathtt{J4} \sum_{i} (S^+_i S^-_{i + 1} S^-_{i + 2} S^+_{i + 2} + \mathrm{h.c.})

    Parameters
    ----------
    model_params : :class:`~tenpy.tools.params.Config`
        Parameters for the model. See :cfg:config:`DipolarSpinChain` below.

    Options
    -------
    .. cfg:config :: DipolarSpinChain
        :include: CouplingMPOModel

        S : {0.5, 1, 1.5, 2, ...}
            The 2S+1 local states range from m = -S, -S+1, ... +S.
        conserve : 'best' | 'dipole' | 'Sz' | 'parity' | None
            What should be conserved. See :class:`~tenpy.networks.site.SpinSite`.
            Note that dipole conservation necessarily includes Sz conservation.
            For ``'best'``, we preserve ``'dipole'``.
        sort_charge : bool | None
            Whether to sort by charges of physical legs.
            See change comment in :class:`~tenpy.networks.site.Site`.
        J3, J4 : float | array
            Coupling as defined for the Hamiltonian above.
    """

    def init_lattice(self, model_params):
        """Initialize a 1D lattice"""
        L = model_params.get('L', 64)
        S = model_params.get('S', 1)
        conserve = model_params.get('conserve', 'best')
        if conserve == 'best':
            conserve = 'dipole'
            self.logger.info("%s: set conserve to %s", self.name, conserve)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        bc = 'periodic' if bc_MPS in ['infinite', 'segment'] else 'open'
        bc = model_params.get('bc', bc)
        sort_charge = model_params.get('sort_charge', None)
        site = SpinSite(S=S, conserve=conserve, sort_charge=sort_charge)
        lattice = Chain(L, site, bc=bc, bc_MPS=bc_MPS)
        if conserve == 'dipole':
            site.leg.chinfo.set_lattice(lattice)
        return lattice

    def init_terms(self, model_params):
        """Add the onsite and coupling terms to the model"""
        J3 = model_params.get('J3', 1)
        J4 = model_params.get('J4', 0)

        self.add_multi_coupling(-J3, [('Sp', 0, 0), ('Sm', 1, 0), ('Sm', 1, 0), ('Sp', 2, 0)], plus_hc=True)
        self.add_multi_coupling(-J4, [('Sp', 0, 0), ('Sm', 1, 0), ('Sm', 2, 0), ('Sp', 3, 0)], plus_hc=True)
