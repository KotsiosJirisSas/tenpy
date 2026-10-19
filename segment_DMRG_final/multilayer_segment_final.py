#!/usr/bin/env python
import numpy as np
import scipy
import os
import re
import sys
print(sys.executable)
print(sys.path)
print(sys.version)
import matplotlib.pyplot as plt
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg') 
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 

np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import tenpy

from create_env import calculate_environment
tenpy.tools.misc.setup_logging(to_stdout="INFO")
from tenpy.models.model import MPOModel


#from vacuum_interface import vacuum_left_environment
from vacuum_interface import vacuum_left_environment_from_mpo
import pickle
from tenpy.linalg.charges import LegCharge
from tenpy.linalg.charges import ChargeInfo
from tenpy.linalg.np_conserved import Array


from tenpy.models.lattice import Lattice
"""Code for running DMRG """
import sys
import os

import numpy as np


from tenpy.algorithms import dmrg


from tenpy.networks.site import QH_MultilayerFermionSite_final


import QH_G2MPO
import QH_Graph_final



from multilayer_creating_MPS_generic_interface_minimal import QH_system
import pickle
import create_segment_model

from graph_potential import graph_with_potential

from sanity_checks import sanity_check_filling

np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)
print('SUCCESFULLY IMPORTED EVERYTHING \n')



def translate_and_rebind(env, leg_mps, leg_mpo, T, nlayer, name=''):
    """
    Puts the environment `env` (computed from the infinite MPS and MPO) onto the legs of the segment.
    The data of env is not changed, only the charges of its legs are replaced by the segment's own legs.

    leg_mps: MPS leg of the segment at this boundary (left env: psi_total._B[0].get_leg('vL'),
                                                      right env: psi_total._B[-1].get_leg('vR'))
    leg_mpo: MPO leg of the segment at this boundary (left: M.H_MPO._W[0].get_leg('wL'),
                                                      right: M.H_MPO._W[-1].get_leg('wR'))
    T:       number of sites between the frame of the infinite MPS and the segment, K -> K + N*T for every row
             (T=0 for the left environment, T=( interface_position+added_sites)//N_layer for the right one)
    """
    labels = env.get_leg_labels()                 # e.g. ['vL*','wL','vL'] for a right environment
    env_legs = [env.get_leg(l) for l in labels]

    # STEP 1: check that the legs of env are the legs of the segment, translated by K -> K + N*T
    # d = (segment charges) - (env charges) - (translation), it must be ONE constant row for all rows
    for nm, leg_e, leg_s in [('MPS', env_legs[0], leg_mps), ('MPO', env_legs[1], leg_mpo)]:
        qe, qs = leg_e.to_qflat(), leg_s.to_qflat()
        assert len(qe)==len(qs), (name, nm, 'dims differ', len(qe), len(qs))
        N = qe[:,1:1+nlayer].sum(axis=1)          # particle number of every row (column 0 is K)
        d = qs - qe
        d[:,0] -= T*N                             # remove the translation
        same = bool((d==d[0]).all())
        # MPO: offset should be [0,0]. MPS: offset is -(charge shift of the patching), 0 for no shift
        print(name, nm, 'constant offset (K,N):', d[0].tolist(), '| all rows equal:', same)
        if not same:
            raise ValueError(name+' '+nm+': legs are not a translation of each other (order or states differ)')

    # STEP 2 (diagnostic only, can be deleted): charge rule  sum_i qconj_i*q_i  of every non-zero entry
    data = env.to_ndarray()
    nz = np.argwhere(np.abs(data) > 1e-12)
    cj = [int(l.qconj) for l in env_legs]
    old_q = [l.to_qflat() for l in env_legs]
    new_q = [leg_mps.to_qflat(), leg_mpo.to_qflat(), leg_mps.to_qflat()]
    rule = lambda qs: sum(c*q[nz[:,k]] for k,(c,q) in enumerate(zip(cj, qs)))
    print(name, 'qconj of env legs', labels, cj, '| qtotal', env.qtotal.tolist())
    print(name, 'raw charges of legs 0 and 2 equal in env:', bool((old_q[0]==old_q[2]).all()))
    print(name, 'rule values with old charges:', np.unique(rule(old_q),axis=0)[:5].tolist())
    print(name, 'rule values with new charges:', np.unique(rule(new_q),axis=0)[:5].tolist())

    # STEP 3: rebuild env with the segment's legs, same data.
    # `like` gives the segment leg the same orientation (qconj) as the env leg it replaces
    # (this is what was missing before: the orientation of the legs of a right environment is not the one I assumed)
    like = lambda target, ref: target if target.qconj == ref.qconj else target.conj()
    legs = [like(leg_mps, env_legs[0]), like(leg_mpo, env_legs[1]), like(leg_mps, env_legs[2])]
    return Array.from_ndarray(env.to_ndarray(), legs, dtype=np.float64, qtotal=None, cutoff=None,
                              labels=labels, raise_wrong_sector=True, warn_wrong_sector=True)


    
def shift_environment_charges(environment,root_config_,location, Nlayer=2,charge_shift=[0,0,0]):
    """
    Shift the quantum-number charges of a DMRG environment tensor to ensure
    compatibility with the total MPS/MPO charge configuration after merging
    two semi-infinite systems.

    This function applies a constant additive shift to the quantum-number
    labels of selected tensor legs (typically `vL`, `wL`, and their conjugates),
    reconstructs the environment tensor with the updated charge structure, and
    returns the corrected environment object.

    Parameters
    ----------
    environment : Array
        A TeNPy `Array` object representing either the left or right environment
        block of the DMRG calculation (usually obtained from
        `calculate_environment` or loaded from file).
        It must have the legs `vL`, `wL`, and `vL*` (bra, MPO bond, and conjugate).

    root_config_ : np.ndarray
        The root (pattern) configuration of the multilayer quantum Hall unit cell.
        Used to construct the appropriate `QH_MultilayerFermionSite_final`
        and infer the charge structure for each leg.

    location : int
        Site index along the 1D lattice where the environment is defined.
        This is needed to construct the correct site object with appropriate
        charge conventions (momentum and layer-resolved quantum numbers).

    Nlayer : int, optional
        Number of layers (internal degrees of freedom) in the quantum Hall model.
        Default is 2.

    charge_shift : list[int] or np.ndarray, optional
        Constant vector of charge offsets to apply to the auxiliary quantum numbers
        on each leg of the environment tensor. Length must be `Nlayer + 1`.
        For example, `[0, 0, 0]` applies no shift, while `[1, -1, 0]` shifts the
        first two layer charges in opposite directions.

    Returns
    -------
    environment : Array
        The same environment tensor, but with modified charge sectors that reflect
        the applied `charge_shift`. The tensor has the same shape and numerical
        data as the input, but its leg quantum numbers are consistently updated.

    Notes
    -----
    - The legs `'vL'`, `'wL'`, and `'vL*'` correspond respectively to:
        * left MPS bond leg
        * left MPO bond leg
        * conjugate left MPS bond leg
      The conjugation is encoded in `conj_q = [1, 1, -1]`.
    - The function modifies both the incoming (`vL`) and conjugate (`vL*`) legs.
    - The sites charge information is reconstructed using
      `QH_MultilayerFermionSite_final`, ensuring that the charge bookkeeping
      remains consistent with the Hamiltonian MPO.

    Example
    -------
    >>> shifted_env = shift_environment_charges(
    ...     environment=right_env,
    ...     root_config_=root_config_,
    ...     location=len(sites)-1,
    ...     Nlayer=3,
    ...     charge_shift=[0, 0, 1, 0]
    ... )
    print(shifted_env.get_leg('vL'))
    # => Leg charges shifted by [0, 0, 1, 0] relative to original

    See Also
    --------
    calculate_environment : builds the left/right environment tensors
    QH_MultilayerFermionSite_final : defines the multilayer site structure
    Array.from_ndarray : reconstructs a TeNPy Array from raw data and leg charges
    """


    Bflat=environment.to_ndarray()
    qflat_list=[]
   

    labels=['vL', 'wL', 'vL*']
    for label in labels:
        qflat_list.append(environment.get_leg(label).to_qflat())
   
    
  
  
    conj_q=[1,1,-1]
    
   
    for i in range(len(qflat_list[0])):
      
     
        #ADD CONSTANT SHIFT ON AUXILIARY LEGS
        for m in range(Nlayer+1):
            qflat_list[0][i][m]+= charge_shift[m]
       
    for i in range(len(qflat_list[2])):

        #ADD CONSTANT SHIFT ON AUXILIARY LEGS
        for m in range(Nlayer+1):
            qflat_list[2][i][m]+= charge_shift[m]
    
    legcharges=[]
    #creates all three legs of MPO
    site=QH_MultilayerFermionSite_final(N=Nlayer,root_config=root_config_,conserve=('each','K'),site_loc=location)
    chargeinfo=site.leg.chinfo

    for i in range(len(qflat_list)):
        legcharge=LegCharge.from_qflat(chargeinfo,qflat_list[i],qconj=conj_q[i]).bunch()[1]
        legcharges.append(legcharge)


    
    #creates MPO
    environment=Array.from_ndarray( Bflat,
                        legcharges,
                        dtype=np.float64,
                        qtotal=None,
                        cutoff=None,
                        labels=labels,
                        raise_wrong_sector=True,
                        warn_wrong_sector=True)
    #print(environment)
    #print("shifted environment charges",'..'*20)
    return environment

def load_mps(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)          # no latin1, this is a py3 TeNPy object

    psi = data['mps']
    model_par = data['model_par']
    

    return model_par, psi


def compare_leg(name, env_leg, seg_leg):
    qa, qb = env_leg.to_qflat(), seg_leg.to_qflat()
    print(name, 'dims env/segment:', len(qa), len(qb))
    if len(qa)==len(qb):
        d = qb - qa
        print('   difference constant:', bool((d==d[0]).all()), ' distinct rows:', np.unique(d,axis=0)[:4].tolist())

def V_potential(x, x0, w, U0):
    return (1 - np.tanh((x - x0)/w)) * U0/2


def V_orbital_from_real_potential(i, x0, w, U0, L_y):
    """
    Orbital-space potential V(i) = <psi_i| V |psi_i> for lowest-Landau-level orbitals
    on a cylinder of circumference L_y (lengths in ell_B), for the real-space potential

        V(x) = (1 - tanh((x - x0*2pi/L_y) / (w*2pi/L_y))) * U0 / 2 .

    Orbital i is centred at X_i = 2 pi i / L_y with |psi_i(x)|^2 = exp(-(x - X_i)^2)/sqrt(pi)
    (y integrated out), so

        V(i) = int dx V(x) exp(-(x - X_i)^2) / sqrt(pi),

    i.e. V(x) convolved with a Gaussian of variance 1/2 (rms width ell_B/sqrt(2)).
    x0 and w are in units of orbital index: without the convolution one would have
    V(i) = (1 - tanh((i - x0)/w)) * U0/2.  For w = 0 the result is U0/2 * erfc(2pi (i-x0)/L_y).

    The convolution is done exactly in Fourier space (F.T. of tanh step x Gaussian):
        V(i) = U0/2 - (U0/pi) int_0^inf dq sin(q d) exp(-q^2/4) a / sinh(a q),
        d = 2pi (i - x0)/L_y,   a = pi * w * 2pi/L_y / 2 .
    The integrand is smooth and Gaussian-decaying, so a midpoint rule is exponentially
    accurate (~1e-15) and costs the same for any w, including w -> 0.

    Parameters
    ----------
    i : int or array of int   orbital index (pass i // nlayer yourself if you fold layers)
    x0, w : step centre and width in orbital-index units (w >= 0)
    U0 : potential scale
    L_y : cylinder circumference in ell_B

    Returns an array of the same shape as i (0-d array for scalar i).
    """
    i = np.asarray(i, dtype=float)
    k = 2 * np.pi / L_y
    d = k * (i - x0)                  # orbital centre minus step centre, in ell_B
    wx = k * w                        # step width in ell_B
    a = 0.5 * np.pi * wx

    # beyond this distance the result is U0 or 0 to < 1e-15: set directly, don't integrate
    cap = 18.0 * wx + 10.0
    out = np.where(d > cap, 0.0, U0)          # saturated value; active entries overwritten below
    flat = d.ravel()
    active = np.flatnonzero(np.abs(flat) <= cap)
    if active.size == 0:
        return out

    da = flat[active]

    # q grid: midpoint rule. Aliasing images sit at 2pi/h +- d in real space and must land
    # where the (smoothed) step is already saturated: >= 18 wx (tanh tail) + 14 (Gaussian)
    h = 2 * np.pi / (np.abs(da).max() + 18.0 * wx + 14.0)
    Q = 14.0                          # exp(-Q^2/4) ~ 5e-22
    q = (np.arange(int(np.ceil(Q / h))) + 0.5) * h

    if a == 0:
        s = 1.0 / q                                          # a/sinh(aq) -> 1/q
    else:
        s = 2 * a * np.exp(-a * q) / (-np.expm1(-2 * a * q))  # = a/sinh(aq), overflow-safe
    wq = np.exp(-q**2 / 4) * s * h

    # chunk over orbitals so the (chunk x n_q) work array stays ~ 2e7 doubles (~160 MB max)
    step = max(1, int(2e7 // q.size))
    res = np.empty(da.size)
    for j in range(0, da.size, step):
        res[j:j + step] = np.sin(da[j:j + step, None] * q) @ wq
    out.flat[active] = U0 / 2 - (U0 / np.pi) * res
    return out





def run_excitation_dmrg(name_load, name_save, L, interface_position,added_sites=0, shift=0, V_of_site=lambda i: 0*i,chi=1000):
    """
    Perform a DMRG calculation for the *excited state* of a quantum Hall system by 
    stitching together two semi-infinite MPS wavefunctions (left and right environments)
    and optimizing the merged interface.

    The function loads a pre-converged iDMRG ground state (via `.pkl` file), constructs
    the full finite-size system by joining left and right semi-infinite segments, and
    runs a DMRG optimization to relax the interface — yielding a variational approximation
    of a localized excitation (quasihole, quasielectron, domain wall, etc.).

    Parameters
    ----------
    name_load : str
        Base filename (without extension) of the saved iDMRG data to load, e.g.
        `'data_Lx20.0_chi_2_psi_Pf_a_xi_1.77'`. The corresponding file must be
        located at `<name_load>.pkl`.

    name_save : str
        Output filename (without extension) to which the resulting data will be saved
        as a `.pkl` file.

    load_env : bool, optional (default=False)
        If `False`, both left and right environments are constructed from the loaded
        iDMRG MPS using `calculate_environment`.  
        If `True`, precomputed environments are loaded from `.npz` and `.pkl` files
        located at hardcoded paths (for debugging or reuse of prior runs).

    added_sites : int, optional
        Number of unit cells (or physical sites) inserted in the middle of the chain
        to separate left and right domains. This controls the interface width.

    shift : list[int] or np.ndarray, optional
        Charge shift vector applied when matching the quantum number legs between
        the right environment and the total merged MPS. Used to ensure correct
        quantum number consistency across the interface.
    L: length of the segment

    Notes
    -----
    The overall procedure is as follows:
        1. Load iDMRG parameters and the converged infinite MPS (`psi_ interface_positioninf`).
        2. Define total system length `L` and midpoint ` interface_position` corresponding to the
           junction position.
        3. Construct the model Hamiltonian as an MPO using `create_segment_DMRG_model`.
        4. Build a `QH_system` object containing left and right  interface_position-infinite wavefunctions.
        5. Create or load left and right DMRG environments, shifting their charges if necessary.
        6. Merge left and right MPSs into a finite chain (`psi_total`).
        7. Initialize the DMRG engine (`TwoSiteDMRGEngine`) with these environments.
        8. Run a full DMRG optimization to minimize the interface energy.
        9. Extract observables:
           - Local filling density profile ⟨n_i⟩
           - Entanglement entropy and spectrum
           - Total energy of the merged system
        10. Save all relevant data as a Python pickle.

    Outputs
    -------
    A `.pkl` file stored at `<name_save>.pkl` containing a dictionary with:
        {
            "dmrg_params": dict,
            "energy": float,
            "model_par": dict,
            "density": np.ndarray,
            "entanglement_entropy": np.ndarray,
            "entanglement_spectrum": dict or list
        }

    Example
    -------
    >>> run_excitation_dmrg(
    ...     name_load='/path/to/data_Lx20.0_chi_2_psi_Pf_a_xi_1.77',
    ...     name_save='Pf_excitation_test',
    ...     load_env=False,
    ...     added_sites=2,
    ...     shift=[0, 0, 0]
    ... )
    projecting MPS ..............................................................
    loading right env
    creating right environment
    ...
    running DMRG
    Filling: [...]
    entanglement entropy: [...]
    Data saved to Pf_excitation_test.pkl

    """
    
    #iDMRGdata_loc = '/mnt/extraspace/kotssvasiliou/QH_DMRG/data_Lx20.0_chi_2_psi_Pf_a_xi_1.77/'
    file_path=name_load+'.pkl'
   
    #model_par,mps,graph=load_mps_and_graph(file_path)
    with open(name_load+'.pkl','rb') as f:
        data=pickle.load(f)
    model_par=data['model_par']
    graph=data['graph']
   
  
   




    verbose=2

   
    ###
    ## matches the boundary exactly at  interface_position
    #TODO: add the potential
    #graph= add_potential_graph(graph,L)
   
    print('graph',len(graph))
    
    #graph=graph*L
    # wrapper: graph_with_potential passes the site index i, the potential depends on the orbital x = i // N_layer
    #shape of the potential
    potential_values=V_of_site(np.arange(L))
    #print(potential_values)
  
    graph = graph_with_potential(graph, L, V_of_site, mode='add', keys=('R','F'), op_name='nOp')
    L=len(graph)

   
    

    
   

   

    ############################
    ##### testing L, interface_position L #####
    ############################
    print('system length',L)
    print(' interface_position', interface_position)
    ############################
    ############################
    ############################

    model_par['boundary_conditions']= ('infinite', L)
    print('__'*100)
    print(model_par)


    
    print(len(graph))

    M,sites=create_segment_model.create_segment_DMRG_model(model_par,graph)

    
    N_layer=len(model_par['layers'])
    root_config_ = model_par['root_config']
    root_config_ = root_config_.reshape(root_config_.size//N_layer,N_layer)


    
     
    ###############################################################
    #construct the wavefunction with merged interface
    params = {}
    params['verbose'] = 1
    params['sys length'] = L
    params['interface_position'] = interface_position
    params['sites added'] = added_sites#sides added in the middle
    params['model data file'] = name_load+'.pkl'
    params['model_params'] = model_par
    params['type_calculation'] = "vacuum_interface"
    if int(added_sites/2)<interface_position and shift[0]>0 and np.all(shift[1:] % 3 == 0):
        params['window_left'] = 'half'

  
    #params['window_product'] = added_sites > 90
    params['window_product'] = False
    QHsys = QH_system(params=params)
    QHsys.shift_import=shift
   
   

    right_site = L - 1 - (interface_position + QHsys.n_right)
    T = (interface_position + QHsys.n_right)//N_layer

    #if we do not load environments,
    #we create them
    print('£'*50)
    print('creating right environment')
    
    right_env, eps_R, cell_R = calculate_environment(file_path, site=right_site, side='right',
                                                     type_calculation=params['type_calculation'], return_bulk=True)

    ##########################################################################
    # ENVIRONMENT OFFSET BOOKKEEPING
    # get_RP(right_site) is built from the fixed point stored at the END of the iMPS unit cell,
    # so it already contains the energy of the s sites between right_site and that boundary:
    #     l_R = length of the right piece,  s = (-l_R) mod cell
    # The reported DMRG energy therefore counts L + s sites. In vacuum_interface mode s depends on
    # where the right piece starts, (interface_position + n_right - L) mod cell, so runs with different
    # added_sites (mod cell) carry different offsets s*eps. We remove it below.
    ##########################################################################
    l_R = L - (interface_position + QHsys.n_right)
    s_env = (-l_R) % cell_R
    offset = eps_R * s_env
    print('right piece length l_R =', l_R, '| extra sites absorbed by RP: s =', s_env,
          '| eps_R =', eps_R)
    print('OFFSET = eps*s =', offset)


    
   



    print('£'*50)
    print('creating left environment')
  
    print('creating left environment')
    if params['type_calculation'] == "vacuum_interface":
        left_environment = vacuum_left_environment_from_mpo(M.H_MPO)
    else:
        left_environment = calculate_environment(file_path, site=0,side='left',type_calculation=params['type_calculation'] )

    print('Done, created left environment')
    print('£'*50)
    
    
     
    
   
    ########################################
    ####printing the charges on the legs####
    ########################################
    QHsys.patch_WFs(save_dir = '',max_pad=30)
    psi_total=QHsys.psi_merged
    ###############################################################
    print(psi_total)
    print()
   

    filling= psi_total.expectation_value("nOp")
    print('number of electrons:', np.sum(filling))
    
    sanity_check_filling(psi_total, model_par, L,  interface_position, added_sites, shift, params['type_calculation'])
    print('filling prior to the entrance')
    print(filling)
  
    #print(params['avg_filling'])
    #K=np.sum(np.arange(len(total_filling))*(total_filling- params['avg_filling']))
    #diff=np.sum((total_filling- params['avg_filling']))
    if verbose>0:
        print('#'*100)
        print('Initial guess, number of particles and momentum sector')
        #print('N='+str(diff))
        #print('K='+str(K))
        print('#'*100)
    
    







    
    #if np.sum(shift_final)!=0:
    #    print("the right environment has an issue?")
    #    quit()

    
  
   
    #T = ( interface_position+added_sites)//N_layer
    right_env = translate_and_rebind(right_env, psi_total._B[-1].get_leg('vR'), M.H_MPO._W[-1].get_leg('wR'), T, N_layer, name='RP')
    if params['type_calculation'] != "vacuum_interface":
        # if not vacuum shift charges if needed
        left_environment = translate_and_rebind(left_environment, psi_total._B[0].get_leg('vL'), M.H_MPO._W[0].get_leg('wL'), 0, N_layer, name='LP')


    init_env_data_halfinf={}


    init_env_data_halfinf['init_LP'] = left_environment    #DEFINE LEFT ENVIROMENT
    init_env_data_halfinf['age_LP'] =0

    init_env_data_halfinf['init_RP'] = right_env   #DEFINE RIGHT ENVIROMENT
    init_env_data_halfinf['age_RP'] =0



    mixer={'amplitude': 0.05,
           'decay': 2,
           'disable_after':5}
    dmrg_params = {
        'mixer': True,
        'max_E_err': 1.e-8,
        'max_S_err': 1.e-6,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-8,
        },
       
        'max_sweeps': 30,
        'mixer_params':mixer}
    
    eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_total, M, dmrg_params,resume_data={'init_env_data': init_env_data_halfinf})
   
    #print("enviroment works")
    print("Starting the DMRG run")
    #
    #print("MPS qtotal:", M.qtotal)
    #print("MPO qtotal:", psi_halfinf.qtotal)
    E0, psi=eng_halfinf.run()

    ##########################################################################
    # REMOVE THE ENVIRONMENT OFFSET
    # The right environment contains s extra bulk sites (see ENVIRONMENT OFFSET BOOKKEEPING above),
    # so the raw DMRG energy E0 is too high by  offset = eps * s.
    #   energy           = E0 - offset   (offset subtracted)
    #   energy_no_offset = E0            (raw DMRG energy)
    #   offset           = eps * s
    ##########################################################################
    E_offset = E0 - offset
    print('energy_no_offset (raw E0) =', E0, '| offset = eps*s =', offset, '| energy = E0 - offset =', E_offset)

    #calculate and store data
    filling=psi.expectation_value("nOp")
    print('Filling:',filling)


    E_spec=psi.entanglement_spectrum(by_charge=True)[::5]
    #print('entanglement spectrum:',E_spec)



    EE=psi.entanglement_entropy()
    print('entanglement entropy:',EE)


    
    data = { "dmrg_params":dmrg_params,"energy":E_offset, "model_par":model_par,'density':filling,'entanglement_entropy': EE, 'entanglement_spectrum':E_spec,'potential':potential_values,"sweep_stats":eng_halfinf.sweep_stats,
             "energy_no_offset":E0, "offset":offset }


   
    print('Storing the result','.'*40)
    # Save object
    with open(name_save+".pkl", "wb") as f:
        pickle.dump(data, f)
    







##############################################################################################################
# COMMAND LINE ARGUMENTS
#   python3 multilayer_segment_final.py <added_sites> <name_load> <name_save> <L> <interface_position> [shift ...]
##############################################################################################################

# Number of sites in the window between the vacuum / left piece and the right piece (0 = no window).
# The window sites are on the RIGHT side of the interface (they start at interface_position).
# IMPORTANT: for a fixed run (same interface_position and same shift), changing added_sites only changes the
# INITIAL wavefunction (how the charge is distributed at the start). It does NOT change the charges: the charge
# sector, and therefore the total number of particles, is set by interface_position and the charge shift below.
added_sites=int(sys.argv[1])

# Graph+MPS file without '.pkl' (contains 'graph' and 'model_par', 'mps'). 
name_load=str(sys.argv[2])

# Output file name without '.pkl'
name_save=str(sys.argv[3])

# Number of repetitions of the unit-cell graph: the segment has L * len(graph) sites.
L=int(sys.argv[4])

# Interface position in sites: the end of the left piece (the empty vacuum in vacuum_interface).
# The charge shift is measured from this point (K relative to it), and the total number of particles in the
# vacuum_interface mode is  nu*(L - interface_position) - N_shift/q.
# Keep it FIXED when scanning added_sites: the right piece then starts at interface_position + added_sites,
# but the charge sector stays the same.
interface_position=int(sys.argv[5])

chi=int(sys.argv[6])
U0=float(sys.argv[7])

x0=float(sys.argv[8])
### applied shift at the interface
# One integer per charge component: (K, N) for cons_C='total' or (K, n_1, ..., n_nlayer) for 'each'.
# This is what sets the charge sector of the whole run, together with interface_position:
#   - the N entries are in units of q = root_config.size per particle (a site has charge q*n - p),
#   - the sign: a NEGATIVE N entry ADDS particles, a positive one removes them (checked for N = -3, q = 3: +1 particle).
shift_length=len(sys.argv)-9
shift=[]
for i in range(9,shift_length+9):
    shift.append(int(sys.argv[i]))
shift=np.array(shift)

### Applied charge shift at the interface
# For simplicity of input, K is specified relative to the interface (` interface_position`):
#     K -> K -  interface_position * sum(N_i)
# NOTE: output SHIFT must be interpreted with this shift in mind.
# this basically just says that we are removing a number of electrons near " interface_position", that is
# near interface posiiton
shift[0] +=  interface_position * np.sum(shift[1:])

#############################################################
#in orbital units. to convert to real space multiply by 2pi/L_y
"""
if 'SHIFTED_BY_1.5_' in name_save:
    x0=interface_position+1.5
elif 'SHIFTED_BY_3' in name_save:
    x0=interface_position+3
else:
    x0=interface_position
"""

w=15
#print(x0)
#NOTE: hardcoded N_layer here, in anything else N_layer is not hardcoded.
N_layer=1#len(model_par['layers'])
#V_of_site = lambda i: V_potential(i//N_layer, x0, w, U0)

m = re.search(r'Lx=(\d+(?:\.\d+)?)', os.path.basename(name_load))
if m is None:
    raise ValueError("no 'Lx=' found in name_load: " + name_load)
L_y = float(m.group(1))
V_of_site = lambda i: V_orbital_from_real_potential(i//N_layer, x0, w, U0,L_y)
############################################################





print('£'*100)
print('INITIAL PARAMETERS')
print('potential calcualated in real space and converted to V(n)')
print('added_sites        :', added_sites)
print('name_load          :', name_load)
print('name_save          :', name_save)
print('L                  :', L)
print('interface_position :', interface_position)
print('chi                :', chi)
print('U0                 :', U0)
print('x0 (potential)     :', x0)
print('w  (potential)     :', w)
print('interface position:',interface_position)
print('N_layer (hardcoded):', N_layer)
print('shift (K, N...)    :', shift, '  <- K already includes interface_position*sum(N)')
print('£'*100)
print("#"*100)
print("Start the calculation")
print("#"*100)


run_excitation_dmrg(name_load,name_save,L, interface_position,added_sites,shift,V_of_site,chi)

##################
###################
# for new runs, change *name_save* and *iDMRGloc*
###################
####################