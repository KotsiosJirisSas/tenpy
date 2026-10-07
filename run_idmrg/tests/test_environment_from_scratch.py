#!/mnt/users/dperkovic/myenv
#source /mnt/users/dperkovic/myenv/bin/activate
"""Code for running DMRG """
import sys


import pickle


sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 
import re

import h5py
from tenpy.tools import hdf5_io
import numpy as np
import matplotlib.pyplot as plt
from tenpy.models.model import MPOModel
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPOGraph
from tenpy.models.lattice import Lattice
from tenpy.networks.site import QH_MultilayerFermionSite_final
import os

import QH_G2MPO
import  QH_Graph_final
print(sys.executable)
print(sys.version)



import tenpy.tools

#logger is important here, prints INFO for DMRG
tenpy.tools.misc.setup_logging(to_stdout="INFO")

np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

###	Layer naming scheme	###
# See line 22 of multilayer_qh for details
# A layer is a component - like a spin species, or bilayer index. It can have any key as name.
# Each layer can have multiple Landau levels.
# The Hilbert space is specified by
# 'layers': [  ( layer_key, landau level), (layer_key, landau level) , . . . ]
# For example,  [ ('up', 0), ('up', 1), ('down', 1) ]
# will have two species, 'up' and 'down'; 'up' contains LLs of n = 0 and 1, and 'down' a single n = 1 LL.
# For nu = 1/3,   we will use  [ ('L', LL)], with LL specified below.


def eliminate_elements_from_the_graph(G,not_included_couplings):
    """
    Trims down the MPO from the keys that are not connected to any legs, i.e. are connected to zero block.
    
    input: G which is graph, not_included_couplings - list of expelled keys
    output: G without unnecessary keys
    """

    L=len(G)
    for i in range(L):
       
        for element in not_included_couplings:
        
            try:
                G[i].pop(element[0],None)
                
            except:
                pass
            
            for key in G[i].keys():
                try:
                    G[i][key].pop(element[0],None)
                except:
                    pass
    return G

def trim_down_the_MPO(G):
    extra_keys=[]
    for n in range(len(G)):
        extra_keys_row=[]
        for key in G[n].keys():
            #print(key)
            
            #print(G[n][key].keys())
            if list(G[n][key].keys())==[]:
                #print('**'*100)
                #print('what what')
                #print(key)
                extra_keys_row.append(key)
        extra_keys.append(extra_keys_row)
        for k in extra_keys_row:
            #pass
            G[n].pop(k, None)
    #print(extra_keys)
    
    return G, extra_keys

def make_iMPO(G,model_par,num_sites):

    ###############
    ######## produces the MPO graph and MPOModel for iDMRG given model_par
    #############
  
    
    Nlayer=len(model_par['layers'])
    G=G*(num_sites//Nlayer)
    print('nlayer:', Nlayer)

    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
    G_new, extra_keys=trim_down_the_MPO(G_new)



    #
    root_config_ = model_par['root_config']
    root_config_ = root_config_.reshape(root_config_.size//Nlayer,Nlayer)
    print('root configuration:')
    print(root_config_)
  
    L = len(G_new) #System size for finite case, or unit cell for infinite
    print('total length of the unit cell')
    print(L)
    
    sites = []
    for i in range(num_sites):
        if model_par['cons_K']:
            if model_par['cons_C']=='total':
                conserve=('N','K')
            else:
                conserve=(model_par['cons_C'],'K')
        else:
            conserve=model_par['cons_C']

        order_of_species=None
        if model_par['cons_C']=='species':
            order_of_species=model_par['order_of_species']
          
        spin=QH_MultilayerFermionSite_final(N=Nlayer,root_config=root_config_,conserve=conserve,site_loc=i,order_of_species=order_of_species)
        #print(spin.leg)
       
        sites.append(spin)
   
    #print(conserve)
   
    #print(sites[10].leg)
    M = MPOGraph(sites=sites,bc='infinite',max_range=None) #: Initialize MPOGRAPH instance

    '''
    M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

    Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
    The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
    '''


    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)

    print(len(not_included_couplings))


    print("Ordering states",".."*10)
    previous_length=len(States[0])+1
    while len(States[0])!=previous_length:

        print('in'*30)
        previous_length=len(States[0])
  
        not_included_couplings_copy=[]
        for i in range(Nlayer):
            
            for m in not_included_couplings[i]:
            
                not_included_couplings_copy.append(m)
   
        #obtain new basis of keys for reduced graph
        G_new=eliminate_elements_from_the_graph(G_new,not_included_couplings_copy)
        G_new, extra_keys=trim_down_the_MPO(G_new)
        States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)

    
    
    
    States,not_included_couplings_2=QH_Graph_final.obtain_states_from_graphs(G_new,L)

   
    
    M.states = States #: Initialize aux. states in model
    M._ordered_states = QH_G2MPO.set_ordered_states(States) #: sort these states(assign an index to each one)
    print("Finished",".."*10 )



    print("Test sanity"+".."*10)
    M.test_sanity()
    M.graph = G_new #: Inpput the graph in the model 
    print("Test passed!"+".."*10)
    grids =M._build_grids()#:Build the grids from the graph
    
   
    print("Building MPO"+".."*10)
    ##############
    
    H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
    print("Built"+".."*10)
  


    
    #initialize MPOModel

    pos= [[i] for i in range(L)]
    lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="infinite")
    x=lattice.mps_sites()
    model=MPOModel(lattice, H)
  
    #produces MPO with (N,K) conservation
    return model,sites



def create_data_to_save(psi,engine,  E0,dmrg_params,model_par):
    Length=psi.correlation_length()
    print('correlation length:',Length)



    filling=psi.expectation_value("nOp")

    print('Filling:',filling)


    E_spec=psi.entanglement_spectrum(by_charge=True)
    #print('entanglement spectrum:',E_spec)



    EE=psi.entanglement_entropy()
    print('entanglement entropy:',EE)


    pairs = [(i, i + 1) for i in range(0, psi.L-1, 2)]
    print('pairs:',pairs)
    g0= [psi.correlation_function('nOp', 'nOp', [i], [j])[0, 0] for i, j in pairs]

    print('g(0):',g0)

    #######
    #### calculate K sector
    #########3
    Ks=[]
    Cs=[]
    for i in range(len(psi._B)):
        charge_left=psi._B[i].legs[0].to_qflat()
        #print(charge_left)
        charges=charge_left[:,0]
        #print(charges)
        K=np.sum((np.abs(psi._S[i])**2)*charges)/np.sum((np.abs(psi._S[i])**2))
        Ks.append(K)
       
        #####
        ##3 check charge neutrality
        #print(charge_left[0])
        #print(len(charge_left[0]))
        if len(charge_left[0])==2:
            charges=charge_left[:,-1]
            C=np.sum((np.abs(psi._S[i])**2)*charges)/np.sum((np.abs(psi._S[i])**2))
        else:
            charges=charge_left[:,1:]
            C=np.sum((np.abs(psi._S[i])[:,None]**2)*charges, axis=0)/np.sum((np.abs(psi._S[i])**2))
    
        
        Cs.append(C)
        ##### charge neutraility
        print('charge neutrality:',C)
    print('K sector:',Ks)
    print('Cs:',Cs)
  
    data = {"sweep_stats":engine.sweep_stats,"dmrg_param":dmrg_params,"energy":E0, "entanglement_entropy": EE, "density":filling, "entanglement_spectrum": E_spec,"model_par":model_par,'correlation_length': Length,'K':Ks,'C':Cs,'g(0)':g0}
    return data


#MEANS PARTICLE OF TYPE 0 down IN LL 0
#PARTICLE OF TYPE up IS IN LL0
# note this is how we write these things
# layers = [ (mu, a), (mu', a'), . . . (species, band = LL#) ]

if __name__ == "__main__":

        # ============================================================
    # Read command-line arguments.
    #
    # Expected usage is roughly:
    #   python script.py <name_load> <pstate> <name_save> [chi_max]
  
    # ============================================================

    #chemical potential
    name_load='/mnt/users/dperkovic/DMRG_modified/data_load/Previous_project/nu=1_3_haldane/each_Haldane_nu=1_3_Lx=16.0'
  
   
    # Convert the comma-separated product state into a list.
    # The number of entries determines the number of sites in the
    # infinite MPS unit cell.
    pstate=['empty','full','empty']
    num_sites=len(pstate)

    

    #chi of intermediate run.
    chi_int=100
    chi_max=300





    ######################
    ##### Store data from the MPS
    ############


    # Print the supplied input parameters for the job log.
    print("£"*40)
    print('input parameters')
    print(name_load)
    print(pstate)
    print("£"*40)




    


    ###########################
    ######### load weak-pairing/CFL phase that converges fine.

    # ============================================================
    # Construct the filename of a reference calculation that is
    # expected to lie in the weak-pairing/CFL phase and converge
    # reliably.
    #
    # The intention appears to be:
    #
    #   1. Build/converge an iMPS using a "safe" Hamiltonian model1.
    #   2. Use that wavefunction as the initial state for the desired
    #      target Hamiltonian model2.
    #
    # This allows the target calculation to start from an already
    # well-behaved wavefunction rather than directly from the product state.
    # ============================================================



    
    binary_list = [1 if x== 'full' else 0 for x in pstate]
    pstate_coded=str([float(x) if isinstance(x, np.floating) else x for x in binary_list]).replace(' ', '')
                
    
    name_load_weak_graph='/mnt/users/dperkovic/DMRG_modified/data_load/Previous_project/nu=1_3_haldane/each_Haldane_nu=1_3_Lx=16.0'
    






    print("Import graph",".."*10)
    with open(name_load+'.pkl','rb') as f:
        data=pickle.load(f)

    # IMPORTANT:
    # model_par and G are overwritten here. From this point onward,
    # model_par refers to the TARGET Hamiltonian, not the reference
    # weak-pairing/CFL Hamiltonian loaded above.
    model_par=data['model_par']
    G=data['graph']
    print("MPO graph and model parameters imported from a file",".."*10)

    # Construct the target Hamiltonian model2.
    #
    # This is fine only if model1 and model2 have fully compatible Site
    # objects (same local Hilbert space, conserved charges, ordering, etc.).
    model,sites=make_iMPO(G,model_par,num_sites)



    # ============================================================
    # Construct an infinite-boundary-condition MPS from the supplied
    # product state.
    #
    # Despite loading a "weak" calculation above, no MPS is loaded from
    # that file here. Instead a NEW product-state MPS is constructed and
    # subsequently optimized with model.
    # ============================================================

    print('creating initial wavefunction')
    psi0 = MPS.from_product_state(sites, pstate, bc="infinite")




    ##### want to converge the entropy sooner since basically, now want to see
    # if there is some discontinuity

    # Discard singular values below this threshold during truncation.
    svd_min=1.e-7
    
 
    


    #gives me a linear ramp from 200 to chi_max using every 5 steps.

    # Construct a bond-dimension schedule for the first/reference run.
    #
    # At sweep index t, the requested chi is:
    #   200 + (3000 - 200) * t / 60
    #
    # However, t only runs from 0 through 30 here:
    #   0, 5, 10, ..., 30
    #
    # Therefore this schedule DOES NOT actually reach 3000.
    # At t=30 it reaches approximately 1600.
    



    
   
    
    #gives me a linear ramp from 200 to chi_max using every 5 steps.

    # Construct the bond-dimension schedule for the final model2 run.
    #
    # This ramps from 3000 at t=0 to chi_max at t=60 in increments
    # of five sweeps.
    #
    # WARNING:
    # If the user supplies chi_max < 3000, this schedule decreases
    # the bond dimension rather than increasing it.
    chi_list_final = {0:100,40:chi_max}
    #chi_list_final = {t: int(round(chi_int + (chi_max - chi_int) * t / 30)) for t in range(0, 31, 5)}

    # Parameters for the target-Hamiltonian optimization.
    dmrg_params_final= {'max_E_err': 0,
        'max_S_err': 0,
    "trunc_params": {"chi_max": chi_max, "svd_min": svd_min}, 
    "mixer": True, 
    'chi_list':chi_list_final}
 
    # ============================================================
    # TEST:
    #
    # A: uninterrupted run
    #
    # B1: run to TEST_STOP_SWEEP
    #     save ONLY the resulting MPS
    #
    # B2: construct a completely NEW DMRG engine from that MPS.
    #     No resume_data.
    #     No old environment.
    #     No old engine state.
    #
    # The chi_list of B2 is shifted such that its local sweep zero
    # corresponds to global sweep TEST_STOP_SWEEP.
    #
    # This tests whether rebuilding the DMRG environment from the
    # same MPS changes the subsequent calculation.
    # ============================================================

    TEST_TOTAL_SWEEPS = 50
    TEST_STOP_SWEEP = 20

    if not (0 < TEST_STOP_SWEEP < TEST_TOTAL_SWEEPS):
        raise ValueError(
            "Need 0 < TEST_STOP_SWEEP < TEST_TOTAL_SWEEPS"
        )

    print('=' * 80)
    print('TESTING NEW ENVIRONMENT FROM SAME MPS')
    print('continuous run: 0 ->', TEST_TOTAL_SWEEPS)
    print(
        'split run:      0 ->',
        TEST_STOP_SWEEP,
        'then NEW ENGINE ->',
        TEST_TOTAL_SWEEPS
    )
    print('=' * 80)


    # ============================================================
    # Original/global chi schedule
    # ============================================================

    print('\nGLOBAL chi_list:')
    print(chi_list_final)


    # ============================================================
    # Make two completely independent copies of the same starting
    # state.
    # ============================================================

    psi_start = psi0.copy()

    psi_continuous = psi_start.copy()
    psi_split = psi_start.copy()


    # ============================================================
    # RUN A
    #
    # One uninterrupted calculation.
    # ============================================================

    print('\n' + '=' * 80)
    print('RUN A: UNINTERRUPTED')
    print('=' * 80)

    params_continuous = dict(dmrg_params_final)

    params_continuous['max_sweeps'] = TEST_TOTAL_SWEEPS-10
    params_continuous['max_E_err'] = 0.0
    params_continuous['max_S_err'] = 0.0
    params_continuous.pop('max_hours', None)

    # Keep the original/global chi schedule.
    params_continuous['chi_list'] = dict(chi_list_final)

    print('RUN A chi_list:')
    print(params_continuous['chi_list'])

    eng_continuous = dmrg.TwoSiteDMRGEngine(
        psi_continuous,
        model,
        params_continuous
    )

    E_continuous, psi_continuous = eng_continuous.run()


    # Save final quantities from A before doing anything else.

    continuous_sweeps = eng_continuous.sweeps
    continuous_entropy = psi_continuous.entanglement_entropy()
    continuous_chi = list(psi_continuous.chi)
    continuous_S = [
        S.copy() for S in psi_continuous._S
    ]

    print()
    print('RUN A final energy:', E_continuous)
    print('RUN A engine.sweeps:', continuous_sweeps)
    print('RUN A chi:', continuous_chi)
    print('RUN A entropy:', continuous_entropy)


    # ============================================================
    # RUN B1
    #
    # Start from the identical initial state but stop at
    # TEST_STOP_SWEEP.
    # ============================================================

    print('\n' + '=' * 80)
    print('RUN B1: BEFORE STOP')
    print('=' * 80)

    params_part1 = dict(dmrg_params_final)

    params_part1['max_sweeps'] = TEST_STOP_SWEEP-10
    params_part1['max_E_err'] = 0.0
    params_part1['max_S_err'] = 0.0
    params_part1.pop('max_hours', None)

    # B1 uses exactly the same global schedule as A.
    params_part1['chi_list'] = dict(chi_list_final)

    print('RUN B1 chi_list:')
    print(params_part1['chi_list'])

    eng_part1 = dmrg.TwoSiteDMRGEngine(
        psi_split,
        model,
        params_part1
    )

    E_part1, psi_split = eng_part1.run()

    print()
    print('RUN B1 energy:', E_part1)
    print('RUN B1 engine.sweeps:', eng_part1.sweeps)
    print('RUN B1 chi:', list(psi_split.chi))
    print(
        'RUN B1 entropy:',
        psi_split.entanglement_entropy()
    )


    # ============================================================
    # "SAVE" ONLY PSI
    #
    # Make a copy. After this point we deliberately use nothing
    # from eng_part1.
    #
    # In particular:
    #
    #   NO resume_data
    #   NO environments
    #   NO sweep_stats
    #   NO update_stats
    #   NO resume_run()
    #
    # ============================================================

    psi_saved = psi_split.copy()

    # Save B1 statistics for plotting the complete split trajectory.
    stats_part1 = {key: list(value) for key, value in eng_part1.sweep_stats.items()}

    # We are deliberately finished with this engine.
    del eng_part1


    # ============================================================
    # Construct the chi schedule for the NEW engine.
    #
    # New engine:
    #
    #     local sweep 0
    #
    # should correspond to:
    #
    #     global sweep TEST_STOP_SWEEP
    #
    # Therefore:
    #
    #     new_t = old_t - TEST_STOP_SWEEP
    #
    # ============================================================

    chi_list_part2 = {}

    for global_sweep, chi in chi_list_final.items():

        if global_sweep >= TEST_STOP_SWEEP:

            local_sweep = (
                global_sweep - TEST_STOP_SWEEP
            )

            chi_list_part2[local_sweep] = chi


    # If TEST_STOP_SWEEP is not itself an entry in chi_list_final,
    # determine which chi was active at the stopping point.

    if 0 not in chi_list_part2:

        active_chi = None

        for global_sweep in sorted(chi_list_final):

            if global_sweep <= TEST_STOP_SWEEP:
                active_chi = chi_list_final[global_sweep]
            else:
                break

        if active_chi is None:
            active_chi = chi_int

        chi_list_part2[0] = active_chi


    # Sort for readability.
    chi_list_part2 = dict(
        sorted(chi_list_part2.items())
    )


    print('\n' + '=' * 80)
    print('SHIFTED chi_list FOR NEW ENGINE')
    print('=' * 80)

    print('original/global chi_list:')
    print(chi_list_final)

    print()

    print('new/local chi_list:')
    print(chi_list_part2)


    # ============================================================
    # RUN B2
    #
    # COMPLETELY NEW ENGINE.
    #
    # The only calculation state supplied to it is psi_saved.
    #
    # This forces TeNPy to construct the environments again from
    # scratch.
    # ============================================================

    print('\n' + '=' * 80)
    print('RUN B2: NEW ENGINE FROM PSI ONLY')
    print('=' * 80)

    params_part2 = dict(dmrg_params_final)

    # The new engine starts counting sweeps from zero.
    #
    # Therefore it only needs to perform the REMAINING number of
    # sweeps.

    N_REMAINING = (
        TEST_TOTAL_SWEEPS - TEST_STOP_SWEEP
    )

    params_part2['max_sweeps'] = N_REMAINING-10
    params_part2['max_E_err'] = 0.0
    params_part2['max_S_err'] = 0.0
    params_part2.pop('max_hours', None)

    # Critical:
    # use the shifted schedule.
    params_part2['chi_list'] = chi_list_part2

    print(
        'new engine number of sweeps:',
        N_REMAINING
    )

    print(
        'new engine chi_list:',
        params_part2['chi_list']
    )


    # ------------------------------------------------------------
    # THIS is the actual test.
    #
    # Fresh engine.
    # No resume_data.
    # ------------------------------------------------------------
    psi_saved.canonical_form()
    eng_new = dmrg.TwoSiteDMRGEngine(
        psi_saved,
        model,
        params_part2
    )

    print(
        'fresh engine starts with sweeps =',
        eng_new.sweeps
    )

    # Ordinary run(), NOT resume_run().
    E_split, psi_split = eng_new.run()


    split_local_sweeps = eng_new.sweeps

    split_entropy = (
        psi_split.entanglement_entropy()
    )

    split_chi = list(psi_split.chi)


    print()
    print('RUN B2 final energy:', E_split)

    print(
        'RUN B2 local engine.sweeps:',
        split_local_sweeps
    )

    print(
        'RUN B total nominal sweeps:',
        TEST_STOP_SWEEP + split_local_sweeps
    )

    print(
        'RUN B final chi:',
        split_chi
    )

    print(
        'RUN B final entropy:',
        split_entropy
    )


    # ============================================================
    # FINAL COMPARISON
    # ============================================================

    print('\n' + '=' * 80)
    print('FINAL COMPARISON')
    print('=' * 80)

    print()
    print('continuous energy :', E_continuous)
    print('new-env energy    :', E_split)
    print(
        'energy difference :',
        E_split - E_continuous
    )
    print(
        'abs energy diff   :',
        abs(E_split - E_continuous)
    )


    print()
    print(
        'continuous sweeps:',
        continuous_sweeps
    )

    print(
        'split nominal sweeps:',
        TEST_STOP_SWEEP + split_local_sweeps
    )


    print()
    print(
        'continuous chi:',
        continuous_chi
    )

    print(
        'new-env chi   :',
        split_chi
    )

    print(
        'chi identical :',
        continuous_chi == split_chi
    )


    print()
    print('continuous entropy:')
    print(continuous_entropy)

    print('new-env entropy:')
    print(split_entropy)

    print('entropy difference:')
    print(
        split_entropy - continuous_entropy
    )

    print(
        'max abs entropy difference:',
        np.max(
            np.abs(
                split_entropy
                - continuous_entropy
            )
        )
    )


    # ============================================================
    # Compare Schmidt values.
    # ============================================================

    max_S_difference = 0.0
    same_S_shapes = True

    for i, (S_a, S_b) in enumerate(
        zip(
            continuous_S,
            psi_split._S
        )
    ):

        if np.shape(S_a) != np.shape(S_b):

            same_S_shapes = False

            print(
                'Schmidt-value shape differs on bond',
                i,
                np.shape(S_a),
                np.shape(S_b)
            )

            continue

        dS = np.max(
            np.abs(
                np.asarray(S_a)
                - np.asarray(S_b)
            )
        )

        max_S_difference = max(
            max_S_difference,
            dS
        )


    print()
    print(
        'Schmidt-value shapes identical:',
        same_S_shapes
    )

    print(
        'max abs Schmidt-value difference:',
        max_S_difference
    )


    # ============================================================
    # Result
    # ============================================================

    print('\n' + '=' * 80)

    if (
        abs(E_split - E_continuous) < 1.e-12
        and same_S_shapes
        and max_S_difference < 1.e-10
    ):

        print(
            'RESULT: rebuilding the environment gives '
            'the same final state to tight tolerance.'
        )

    else:

        print(
            'RESULT: rebuilding the environment changes '
            'the subsequent DMRG trajectory.'
        )

    print('=' * 80)

    # ============================================================
    # PLOT RUN A VS RUN B (B1 + fresh-engine B2)
    # ============================================================
    stats_A = eng_continuous.sweep_stats
    stats_B2 = eng_new.sweep_stats

    sweep_A = np.asarray(stats_A['sweep'])
    E_A = np.asarray(stats_A['E'])
    S_A = np.asarray(stats_A['S'])
    norm_A = np.asarray(stats_A['norm_err'])
    chi_A = np.asarray(stats_A['max_chi'])

    sweep_B1 = np.asarray(stats_part1['sweep'])
    E_B1 = np.asarray(stats_part1['E'])
    S_B1 = np.asarray(stats_part1['S'])
    norm_B1 = np.asarray(stats_part1['norm_err'])
    chi_B1 = np.asarray(stats_part1['max_chi'])

    sweep_B2_local = np.asarray(stats_B2['sweep'])
    sweep_B2 = sweep_B2_local + TEST_STOP_SWEEP
    E_B2 = np.asarray(stats_B2['E'])
    S_B2 = np.asarray(stats_B2['S'])
    norm_B2 = np.asarray(stats_B2['norm_err'])
    chi_B2 = np.asarray(stats_B2['max_chi'])

    sweep_B = np.concatenate([sweep_B1, sweep_B2])
    E_B = np.concatenate([E_B1, E_B2])
    S_B = np.concatenate([S_B1, S_B2])
    norm_B = np.concatenate([norm_B1, norm_B2])
    chi_B = np.concatenate([chi_B1, chi_B2])

    print('RUN A sweep points:', sweep_A)
    print('RUN B sweep points:', sweep_B)

    for yA, yB, ylabel, title, filename, logy in [
        (E_A, E_B, 'Energy', 'DMRG energy', 'test_environment_energy.png', False),
        (S_A, S_B, 'S', 'Entanglement entropy', 'test_environment_entropy.png', False),
        (norm_A, norm_B, 'norm_err', 'MPS norm error', 'test_environment_norm_err.png', True),
        (chi_A, chi_B, 'max chi', 'Actual maximum bond dimension', 'test_environment_chi.png', False),
    ]:
        plt.figure(figsize=(7, 5))
        if logy:
            plt.semilogy(sweep_A, yA, 'o-', label='A: uninterrupted')
            plt.semilogy(sweep_B, yB, 's--', label='B: rebuilt environment')
        else:
            plt.plot(sweep_A, yA, 'o-', label='A: uninterrupted')
            plt.plot(sweep_B, yB, 's--', label='B: rebuilt environment')
        plt.axvline(TEST_STOP_SWEEP, linestyle=':', label='environment rebuilt')
        plt.xlabel('Sweep')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close()

    print('Saved plots:')
    print('test_environment_energy.png')
    print('test_environment_entropy.png')
    print('test_environment_norm_err.png')
    print('test_environment_chi.png')