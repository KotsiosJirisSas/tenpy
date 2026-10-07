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
    Length=psi.correlation_length(charge_sector=None, return_charges=True)
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
    #
    # name_load : base filename/path of the target V0 calculation
    # pstate    : comma-separated product-state specification
    # name_save : base filename/path used for saving the final result
    # chi_max   : optional maximum MPS bond dimension
    # ============================================================

    #chemical potential
    name_load=str(sys.argv[1])
    pstate=str(sys.argv[2])
    name_save=str(sys.argv[3])


    # Convert the comma-separated product state into a list.
    # The number of entries determines the number of sites in the
    # infinite MPS unit cell.
    pstate=pstate.split(",")
    num_sites=len(pstate)

    # Read the requested maximum bond dimension.
    # If the fourth command-line argument is absent OR cannot be converted
    # to an integer, fall back to chi_max = 3400.
    #
    # WARNING:
    # The bare `except:` catches every exception, not only a missing/invalid
    # command-line argument. This can potentially hide unrelated errors.
  
    chi_max=int(sys.argv[4])
    
    option=str(sys.argv[5])

    #chi of intermediate run.
    chi_int=3000





    ######################
    ##### Store data from the MPS
    ############


    # Print the supplied input parameters for the job log.
    print("£"*40)
    print('input parameters')
    print(name_load)
    print(name_save)
    print(pstate)
    print(option)
    print("£"*40)

    # Extract V0 from name_load.
    #
    # For example:
    #   "...V0=-0.8..." -> V0 = -0.8
    #
    # The regex supports:
    #   V0=-0.8
    #   V0=0.25
    #   V0=1
    #   V0=-1.2e-3
    #
    # If V0 cannot be found/parsing fails, V0 is silently set to zero.
    #
    # WARNING:
    # Again, the bare `except:` can hide errors other than a missing V0.
    try:
        V0 = float(re.search(r"V0=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", name_load).group(1))
    except:
        V0=0.0
    print('V0 extract',V0)



    


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

 
    # Choose the V0 value used for the initial/reference Hamiltonian.
    #
    # For filenames without "RPA", use V0_test = -0.8.
    # For filenames containing "RPA", use V0_test = 0.0.
    if 'RPA' not in name_load:
        V0_test=-0.8
    else:
        V0_test=0.0
    add=''
    # Replace the V0 appearing in name_load by V0_test while leaving
    # the rest of the filename unchanged.
    #
    # Example:
    #   name_load      = "...V0=-1.5..."
    #   V0_test        = -0.8
    #   name_load_weak = "...V0=-0.8..."
    #
    # WARNING:
    # If name_load does NOT contain a substring matching "V0=<number>",
    # re.sub() simply returns name_load unchanged. There is no explicit
    # check here that the replacement actually occurred.
    name_load_weak_graph = re.sub(
    r"V0=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    f"V0={V0_test}",
    name_load)

    name_load_weak_MPS= re.sub(
    r"V0=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
    f"V0={V0_test}",
    name_load)
    name_load_weak_MPS = name_load_weak_MPS.replace("data_load", "infinite_DMRG_data")
    
    
    binary_list = [1 if x== 'full' else 0 for x in pstate]
    pstate_coded=str([float(x) if isinstance(x, np.floating) else x for x in binary_list]).replace(' ', '')
                
    name_load_weak_mps=name_load_weak_MPS+'_pstate='+str(pstate_coded)+'_chi='+str(chi_int)+'_MPS_data_new_tenpy.pkl'
    
    print(name_load_weak_mps)
    if os.path.isfile(name_load_weak_mps) and option=='load_MPS':
        print('x'*20)
        print('loading the MPS initial state')
        print('x'*20)
        with open(name_load_weak_mps,'rb') as f:
            data=pickle.load(f)
        #print(data.keys())
        psi=data['mps']
        add='_with_loaded_psi_weak_paired'
    else:
        print('x'*20)
        print('converged psi for intermediate state not found')
        print('loading the intermediate state graph')
        print('x'*20)
       
       
        # Load the reference/weak-pairing calculation.
        with open(name_load_weak_graph+'.pkl','rb') as f:
            data=pickle.load(f)
        
        # Extract the model parameters and MPO graph corresponding to
        # the reference V0_test Hamiltonian.
        model_par=data['model_par']
        G=data['graph']
        print("MPO graph and model parameters imported from a file",".."*10)


        # Construct model1, the reference Hamiltonian used for the first
        # iDMRG optimization.
        #
        # `sites` is also obtained here and is subsequently used to construct
        # the initial product-state MPS.
        model1,sites=make_iMPO(G,model_par,num_sites)
        add='_with_simulated_psi_weak_paired'

    



    # ============================================================
    # Now load the MPO/model corresponding to the actual requested
    # value of V0 (i.e. name_load rather than name_load_weak).
    # This becomes model2.
    # ============================================================

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
    model2,sites=make_iMPO(G,model_par,num_sites)



    # ============================================================
    # Construct an infinite-boundary-condition MPS from the supplied
    # product state.
    #
    # Despite loading a "weak" calculation above, no MPS is loaded from
    # that file here. Instead a NEW product-state MPS is constructed and
    # subsequently optimized with model1.
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
    
    chi_list_middle_state= {t: int(round(200 + (chi_int - 200) * t / 30)) for t in range(0, 31, 5)}



    
   

    # Parameters for the first DMRG run using model1.
    #
    # This first stage is intended to obtain a converged reference
    # weak-pairing/CFL wavefunction before switching to model2.
    chi_list_final = {t: int(round(chi_int + (chi_max - chi_int) * t / 30)) for t in range(0, 31, 5)}
    print()

    
    dmrg_params_middle_state = {'max_E_err': 1.e-8,
            'max_S_err': 6.e-7,
        "trunc_params": {"chi_max": chi_int, "svd_min": svd_min}, 
        "mixer": True, 
        "max_sweeps":50,
        'chi_list':chi_list_middle_state
    }

    ### uncomment if want non-default mixer.
    ##mixer={'amplitude': 0.05,'decay': 4, 'disable_after':30}
    ##dmrg_params["mixer_params"] = mixer

    # Maximum number of sweeps intended for the later/final stage.
    max_sweeps=200

    # Time limit for the first/reference run, in hours.

    t_break=24*2 ### 2 days
    
    
    # Make a copy of the first-stage parameters and add the wall-time cap.
    # Using dict(...) gives a shallow copy, which is sufficient here as
    # only a top-level key is being added.
    dmrg_params_middle_state["max_hours"] = t_break



    ################ converge the CFL first?
    ############## alternatively, can just load it

    # ============================================================
    # FIRST DMRG STAGE
    #
    # Start from the product-state infinite MPS and optimize using
    # model1, i.e. the reference V0_test Hamiltonian.
    #
    # The intended result is a well-converged weak-pairing/CFL state
    # that can subsequently be used as the initial state for model2.
    # ============================================================
    if not os.path.isfile(name_load_weak_mps) or option!='load_MPS':
        print('x'*30)
        print('starting intermediate DMRG')
        print(dmrg_params_middle_state )
        print("Run DMRG:")
        #quit()
        eng = dmrg.TwoSiteDMRGEngine(psi0, model1, dmrg_params_middle_state)
        E0, psi = eng.run()

  

    # `eng.shelve` indicates that the first engine stopped because it was
    # shelved (for example because the max_hours limit was reached).
    #
    
   
    
    #gives me a linear ramp from 200 to chi_max using every 5 steps.

    # Construct the bond-dimension schedule for the final model2 run.
    #
    # This ramps from 3000 at t=0 to chi_max at t=60 in increments
    # of five sweeps.
    #
    # WARNING:
    # If the user supplies chi_max < 3000, this schedule decreases
    # the bond dimension rather than increasing it.
    chi_list_final = {t: int(round(chi_int + (chi_max - chi_int) * t / 30)) for t in range(0, 31, 5)}

    # Parameters for the target-Hamiltonian optimization.
    dmrg_params_final= {'max_E_err': 1.e-8,
        'max_S_err': 6.e-7,
    "trunc_params": {"chi_max": chi_max, "svd_min": svd_min}, 
    "mixer": True, 
    "max_sweeps":max_sweeps,
    'chi_list':chi_list_final}
 

    print('resuming calculation')
    print('x'*30)

    

    
  

  

    # Set the intended total wall-clock budget to 10 days.
    # Since t_break was allocated to the first stage, the second stage
    # receives the remaining time.
    # 10 days for the total run.
    t_total=10*24 
    dmrg_params_final["max_hours"] = t_total-t_break
   

    # TeNPy will create a fresh engine here, which clears stats,
    # so we restore them immediately afterwards.

    # ============================================================
    # SECOND DMRG STAGE
    #
    # A new DMRG engine is constructed using:
    #
    #   psi    : wavefunction obtained from model1
    #   model2 : target negative-V0 Hamiltonian
    #
    # Thus the intention is to let the already-optimized reference
    # wavefunction flow under the target Hamiltonian.
    #
    print('x'*30)
    print('starting the final DMRG')
    print('x'*30)

    psi.canonical_form()
    eng = dmrg.TwoSiteDMRGEngine(psi, model2, dmrg_params_final)

    # Restore accumulated statistics from the first engine.
    

    # Continue the DMRG run using the target Hamiltonian model2.
    E0, psi = eng.run()


    # ============================================================
    # Package the final MPS, engine information, energy, DMRG
    # parameters, and target-model parameters for saving.
    # ============================================================

    psi.canonical_form()
    data=create_data_to_save(psi,eng,  E0,dmrg_params_final,model_par)
   
    ######################
    ##### Store data from the MPS
    ############

    # Save the final data structure as an HDF5 file.
    #
    # NOTE:
    # The file is opened with mode 'w', so an existing file with the
    # same name will be overwritten.
    with h5py.File(name_save+add+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
   


   

    
