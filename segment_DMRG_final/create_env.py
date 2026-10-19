#!/usr/bin/env python
import numpy as np
import sys
import pickle

sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 
sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy/NEW_COMMITS') 
sys.path.append('/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy') 
sys.path.append('/mnt/users/kotssvasiliou/tenpy')
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg')
np.set_printoptions(precision=5, suppress=True, linewidth=100)

import tenpy
from tenpy.networks.mpo import MPOEnvironment

tenpy.tools.misc.setup_logging(to_stdout="INFO")

#make_iMPO lives in the iDMRG run script
from QHE_run_from_graph import make_iMPO


def calculate_environment(full_name_load,  site, side, type_calculation='excitation', return_bulk=False):
    """
    Environment of the infinite MPS, taken directly from the iMPS and its MPO.

    full_name_load: pickle with the MPO graph of one unit cell
    WF_path:        pickle with 'mps' (infinite TeNPy3 MPS) and 'model_par'
    site:           site index in the frame of the infinite MPS (site 0 = first site of its unit cell)
    type_calculation: 'excitation', 'interface' or 'vacuum_interface'. For 'interface' the right environment
                    is taken from the second state, data['mps_2']
    side:           'left'  -> contraction strictly left of `site`  (labels 'vR*','wR','vR')
                    'right' -> contraction strictly right of `site` (labels 'vL*','wL','vL')
    return_bulk:    if True, also return (eps, cell):
                    eps  = energy per site of the iMPS, computed with the SAME infinite MPO
                           that builds the environment (so it cancels the environment offset exactly)
                    cell = psi.L, the unit cell of the iMPS
    returns:        the environment tensor  (or  env, eps, cell  if return_bulk)
    """
    with open(full_name_load,'rb') as f:
        data=pickle.load(f,encoding='latin1')
    graph=data['graph']
    #for an interface the right hand side is the second state
    if type_calculation=='interface' and side=='right':
        psi=data['mps_2']
    else:
        psi=data['mps']
    model_par=data['model_par']

    #infinite MPO with the same unit cell as the iMPS
    M_inf,sites=make_iMPO(graph,model_par,num_sites=psi.L)
    #no sort_legcharges here: the outer bond of the segment MPO stays in the natural order of the states
    #(sorting the infinite MPO gave a different order of the same states)

    #TeNPy converges the environment of the infinite MPS itself
    env_inf=MPOEnvironment(psi,M_inf.H_MPO,psi)

    if side=='left':
        env=env_inf.get_LP(site)
    elif side=='right':
        env=env_inf.get_RP(site)
    else:
        raise ValueError
    print('environment Array:')
    print(env)
    if return_bulk:
        # for an infinite MPS TeNPy returns the energy PER SITE
        eps = float(np.real(M_inf.H_MPO.expectation_value(psi)))
        print('bulk energy per site eps =', eps, '| iMPS unit cell =', psi.L)
        return env, eps, psi.L
    return env