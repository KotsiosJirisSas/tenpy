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


import QH_G2MPO
import  QH_Graph_final
print(sys.executable)
print(sys.version)


import tenpy.tools

from  tenpy.linalg import np_conserved as npc


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
    print(psi.sites[0].nOp)
    print(psi.sites[1].nOp)
    #SzSx_list = [npc.outer(psi.sites[0].nOp.replace_labels(['p', 'p*'], ['p0', 'p0*']), psi.sites[1].nOp.replace_labels(['p', 'p*'], ['p1', 'p1*'])) ]
    nn01 = psi.expectation_value_term([('nOp', 0), ('nOp', 1)])
    #val=psi.expectation_value(SzSx_list)
    print(nn01)

    #pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    pairs = [(i, i + 1) for i in range(0, psi.L-1, 2)]
    print(pairs)
    diag = [psi.correlation_function('nOp', 'nOp', [i], [j])[0, 0] for i, j in pairs]

    print(diag)
    quit()
    E_spec=psi.entanglement_spectrum(by_charge=True)
    #print('entanglement spectrum:',E_spec)



    EE=psi.entanglement_entropy()
    print('entanglement entropy:',EE)



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
  
    data = {"sweep_stats":engine.sweep_stats,"dmrg_param":dmrg_params,"energy":E0, "entanglement_entropy": EE, "density":filling, "entanglement_spectrum": E_spec,"model_par":model_par,'correlation_length': Length,'K':Ks,'C':Cs}
    return data


#MEANS PARTICLE OF TYPE 0 down IN LL 0
#PARTICLE OF TYPE up IS IN LL0
# note this is how we write these things
# layers = [ (mu, a), (mu', a'), . . . (species, band = LL#) ]

if __name__ == "__main__":
    #chemical potential
    name_load=str(sys.argv[1])
    pstate=str(sys.argv[2])
    name_save=str(sys.argv[3])


    pstate=pstate.split(",")
    num_sites=len(pstate)

    try:
        chi_max=int(sys.argv[4])
    except:
        chi_max=3400

    store_MPS=True




    ######################
    ##### Store data from the MPS
    ############


    print("£"*40)
    print('input parameters')
    print(name_load)
    print(name_save)
    print(pstate)
    print("£"*40)
    try:
        V0 = float(re.search(r"V0=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", name_load).group(1))
    except:
        V0=0.0
    print('V0 extract',V0)







    print("Import graph",".."*10)
    with open(name_load+'.pkl','rb') as f:
        data=pickle.load(f)
    model_par=data['model_par']
    G=data['graph']
    print("MPO graph and model parameters imported from a file",".."*10)


    model,sites=make_iMPO(G,model_par,num_sites)
    root_config_=model_par['root_config']





    print('creating initial wavefunction')
    psi = MPS.from_product_state(sites, pstate, bc="infinite")




    ##### want to converge the entropy sooner since basically, now want to see
    # if there is some discontinuity
    if chi_max<4000:
        svd_min=5.e-6
    else:
        svd_min=1.e-6
    #gives me a linear ramp from 200 to chi_max using every 5 steps.
    #chi_list = {t: int(round(200 + (chi_max - 200) * t / 60)) for t in range(0, 61, 5)}
    chi_list = {0: chi_max}
    #V0s=[-0.96,-0.98,-1.02,-1.04]
 

    if 'V_intra=[-0.5,1.0]_V_inter=[-0.5,1.0]' in name_load or 'V_intra=[-1.0,1.0]_V_inter=[-1.0,1.0]' in name_load or V0<-0.8:
        max_sweep=240
    else:
        max_sweep=170
    max_sweep=150

    dmrg_params = {'max_E_err': 1.e-8,
            'max_S_err': 1.e-6,
        "trunc_params": {"chi_max": chi_max, "svd_min": svd_min}, 
        "mixer": True, 
        "max_sweeps":max_sweep,
        'chi_list':chi_list
    }
    ### uncomment if want non-default mixer.
    ##mixer={'amplitude': 0.05,'decay': 4, 'disable_after':30}
    ##dmrg_params["mixer_params"] = mixer
    t_break=24*3 ### 3 days
    
    
    # run with the time limit
    dmrg_params_t = dict(dmrg_params)
    dmrg_params_t["max_hours"] = t_break
    print(dmrg_params)
    print("Run DMRG:")

    eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params_t)
    E0, psi = eng.run()
    print('x'*30)
    print('time: 3 days have passed, saving data, resuming calculation')

    data=create_data_to_save(psi,eng,  E0,dmrg_params,model_par)

    with h5py.File(name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)


    if eng.shelve:
        print('resuming calculation')
        print('x'*30)
        resume_data = eng.get_resume_data()

        checkpoint = {
            "psi": psi,
            "resume_data": resume_data,
            "sweep_stats": eng.sweep_stats,
            "update_stats": eng.update_stats,
            "E0": E0}

      
        # remove the time cap for the continued run
        dmrg_params_full = dict(dmrg_params)
        dmrg_params_full.pop("max_hours", None)
        # 10 days for the total run.
        t_total=10*24 
        dmrg_params_full["max_hours"] = t_total-t_break

        # TeNPy will create a fresh engine here, which clears stats,
        # so we restore them immediately afterwards.
        eng = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params_full, resume_data=resume_data)
        eng.sweep_stats = checkpoint["sweep_stats"]
        eng.update_stats = checkpoint["update_stats"]

        E0, psi = eng.resume_run()
    data=create_data_to_save(psi,eng,  E0,dmrg_params,model_par)
   
    ######################
    ##### Store data from the MPS
    ############
    with h5py.File(name_save+".h5", 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
   


   

    
