'''
[02/09/24]start of QH model class
'''
import numpy as np
import itertools
import copy
from ..networks.site import Site 
class FermionSiteLLL(Site):
    r"""Create a :class:`Site` for spin-less fermions for the Quantum Hall LLL. So the band properties only go into the modification of V.

    Local states are ``empty`` and ``full``.

    .. warning ::
        Using the Jordan-Wigner string (``JW``) is crucial to get correct results,
        otherwise you just describe hardcore bosons!
        Further details in :doc:`/intro/JordanWigner`.

    ==============  ===================================================================
    operator        description
    ==============  ===================================================================
    ==============  ===================================================================

    ============== ====  ===============================
    `conserve`     qmod  *excluded* onsite operators
    ============== ====  ===============================
    ``'N'``        [1]   --
    ``'parity'``   [2]   --
    ``'None'``     []    --
    ============== ====  ===============================

    Parameters
    ----------
    conserve :
    filling : 

    Attributes
    ----------
    conserve :
    filling : 
    """


    '''
    Steps:
    1) Model after chain
    2)
    '''
    #def __init__(self, conserve='N', filling=0.5):