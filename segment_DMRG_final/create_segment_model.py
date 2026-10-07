import numpy as np

from tenpy.models.model import MPOModel
from tenpy.models.lattice import Lattice
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.site import QH_MultilayerFermionSite_final

import QH_G2MPO
import QH_Graph_final


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

def create_segment_DMRG_model(model_par, graph, add=0):

    """
    Creates MPOModel given model parameters with L sites on a segment.

    model_par: parameters for creation of MPOModel
    L: int, number of sites on chain
    root_config_: root configuration
    conserve: gives tuple of conserved quantities
   
    returns:
    MPOModel
    Sites - list of sites 
    """

    N_layer=len(model_par['layers'])
    L=len(graph)
    
   
    #SET PARAMETERS
    #TODO: MAKE MODEL PARAMETERS LOAD FROM OLD TENPY2
        

    #root configuration and conservation laws, loaded from model_par as in make_iMPO
    root_config_ = model_par['root_config']
    root_config_ = root_config_.reshape(root_config_.size//N_layer,N_layer)

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


    # construct MPO Model from the Graph using the tenpy2 code
    print("Start model in Old Tenpy",".."*10)
    #M = mod.QH_model(model_par)
    print("Old code finished producing MPO graph",".."*10)

    
    #load tenpy2 graph into tenpy3 graph
    G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(graph)
    #print(len(G_new[0].keys()))

    #trim down the MPO
    G_new, extra_keys=trim_down_the_MPO(G_new)
    
    
    cell=len(root_config_)
    
    print('asserting that the size is compatible with enivornment.....')
    #assert L%cell==0


    #root_config_ = np.array([0,1,0])
  

    #define Hilbert spaces for each site with appropriate conservation laws
    sites=[]
    for i in range(-add,L-add):
        spin=QH_MultilayerFermionSite_final(N=N_layer,root_config=root_config_,conserve=conserve,site_loc=i,order_of_species=order_of_species)
        #print(spin.leg)
        sites.append(spin)
    
    M = MPOGraph(sites=sites,bc='segment',max_range=None) #: Initialize MPOGRAPH instance

    '''
    M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

    Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
    The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
    '''
    print(L)
    

    States,not_included_couplings=QH_Graph_final.obtain_states_from_graphs(G_new,L)

    print(len(not_included_couplings))


    print("Ordering states",".."*10)
    previous_length=len(States[0])+1
    while len(States[0])!=previous_length:

        print('in'*30)
        previous_length=len(States[0])
  
        not_included_couplings_copy=[]
        for i in range(N_layer):
            
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
    M.graph = G_new #: INppuut the graph in the model 
    print("Test passed!"+".."*10)
    grids =M._build_grids()
    #:Build the grids from the graph
    print("Building MPO"+".."*10)


    H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
    print("Built"+".."*10)


   
    
    #Define lattice on which MPO is defined
    pos= [[i] for i in range(L)]
    lattice = Lattice([1], sites,positions=pos, bc="periodic", bc_MPS="segment")
    x=lattice.mps_sites()
    
    #create model from lattice and MPO
    model=MPOModel(lattice, H)
    print("created model",".."*30)

    #assert model.H_MPO.is_equal(model.H_MPO.dagger())
    print('asserted')
   
    
    return model,sites