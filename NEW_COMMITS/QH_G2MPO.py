import pickle
import numpy as np
import sys
import os
sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') #comment out if not me
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
#from tenpy.networks.site import [Placeholder] #this is whatever class captures the QH hilbert space. Should be very close to FermionSite(spinless)
'''
~~~10 Oct 2024~~~
~~~~~~~~KV~~~~~~~

SKELETON FOR MPO CONSTRUCTION FOR QUANTUM HALL SYSTEMS WITH NUMBER CONSERVATION STRAIGHT FROM THE GRAPH G OF THE FINITE STATE MACHINE

INPUTS:
1)  MODEL PARAMETERS. GET FED INTO THE CREATION OF THE HILBERT SPACE AND ALSO THE CREATION OF THE GRAPH
2)  THE GRAPH G. THIS CAN EITHER BE IMPORTED FROM TENPY2 OR PREFERABLY FROM THE CODE DOM ALTERED TO WORK IN TENPY3. 
    POINT IS, THE GRAPH SHOULD BE A LIST OF DICTIONARIES OF DICTIONARIES, WITH THE GRAPH AT INDEX i OF THE LIST LOOKS LIKE
    G[i] = {r:{c:['Operator Name',Operator_strength]}}


STEPS:
0) Change certain aspects of the Graph to make sure it is compatible to use for tenpy3:
    a)changes 'R' and 'F' nodes to 'IdL','IdR'
    b) turns aby tuple nodes to string nodes, so all nodes are represented by straight strings that are sorted lexicographically (except IdL,IdR which automatically get sorted 1st and last respectively)
    c) make sure Operator_strength entries are all floats

1) Create Hilbert space os a single site, along with all the local operators and the charges of the physical legs. Then create the sites: a chain of length L of them (L= size or unit cell)
2) Initialize an MPOgraoh instance: MPOGraph(sites=sites,bc='infinite',max_range=None)
3) Extract the auxilliary Hilber space states (ie the node strings) from the Graph and *manually* feed them into the MPOgraph instance (M), along with their ordering.
   (Maybe even find a way to do it from within the MPOgraph class by writing new class functions)
4) Manually input the graph into the instance: M.graph = G
5) Create the grid. Basically turns G ---> W where W's are grids (matrices) and W_ij contains the name and strength of the onsite operator connecting nodes i and j
6) Call Build_MPO which builds the MPO from the model instance M. Breaking it down:
   a) It first goes through the nodes and calculates the charges for the auxilliary legs
   b) Calls the MPO class function MPO.from_grids() which initializes the MPO based on all the above collected information.
   This is pretty much the MPOgraphs.Build_MPO function with small changes. Maybe incorporate directly
7) Done
'''
#####################################################################################################
#Functions we might need (Some might need minor modification for QH case, as they are based on SPin chain case)
def G2G_map(G):
    '''
    Provides a mapping from tenpy2 to tenpy3 Graph dictionaries. Helpful for building MPO in TenPy3 using Markov Processes from Tenpy2.
    The main difference between the two graphs is the mapping 'R'(ready) <-----> 'IdL' (identities to the left) and 'F'(finished) <-----> 'IdR' (identities to the right)
    Another thing i did is, the more complicated systems has nodes labelled by ('str1','str2',#) eg ('Mk','zz',0) and i turned them into just strings by doing ('str1','str2',#)----> '('str1','str2',#)'.
        ultimately the labels shouldn't matter at all after creating the Graph
    Another thing is the floats for the strength of the operators. While going from tenpy2 (and python2) to tenpy3 (and python3) and pickling the dictionary, the floats et a weird np.float64(float) thing in front of them that i deal with.
        code should work fine even if the pickling works 'correctly'
    
    Input: The Graph at site i
    Output: The mapped Graph at site i
    '''
    #print(len(G))
    #print(G[0])
    #quit()
    G_new = {}
    for key_L in G.keys():
        if key_L == 'R': key_L_new = 'IdL'
        elif key_L == 'F': key_L_new = 'IdR'
        else:
            if isinstance(key_L,tuple): 
                #print(key_L)
                keyara=[]
                for i in range(len(key_L)):
                    if isinstance(key_L[i], str):
                        keyara.append(key_L[i])
                    elif isinstance(key_L[i], np.int64):
                        keyara.append(int(key_L[i]))
                    elif isinstance(key_L[i], np.float64):
                        keyara.append(float(key_L[i]))
                    else:
                        keyara.append(key_L[i])
                
                keyara=tuple(keyara)
                #print(keyara)
                key_L_new = str(keyara) #dumb way of turning all node labels to strings
             
            else: key_L_new = key_L
        if key_L_new not in G_new:
            G_new[key_L_new] = {} # or []?
        ###################################
        for key_R in G[key_L].keys():
            if key_R == 'R': key_R_new = 'IdL'
            elif key_R == 'F': key_R_new = 'IdR'
            else:
                if isinstance(key_R,tuple): 
               
                    #GET RID OF np.int64 from the string
                    keyara=[]
                    for i in range(len(key_R)):
                        if isinstance(key_R[i], str):
                            keyara.append(key_R[i])
                        elif isinstance(key_R[i], np.int64):
                            keyara.append(int(key_R[i]))
                        elif isinstance(key_R[i], np.float64):
                            keyara.append(float(key_R[i]))
                        else:
                            keyara.append(key_R[i])
                    
                    keyara=tuple(keyara)
                    key_R_new = str(keyara) #
                   
                else: key_R_new = key_R
            if key_R_new not in G_new[key_L_new]:
                G_new[key_L_new][key_R_new] = []
            data = G[key_L][key_R]
            #some sanity checks:
            if not isinstance(data,list) or not isinstance(data[0],tuple) : raise ValueError
            if len(data) != 1: raise ValueError
            if not isinstance(data[0][0],str) or not isinstance(data[0][1],float): raise ValueError
            new_data = (data[0][0], float(data[0][1])) #turns np.float64(float) to just float
            #new_data = list(new_data) # i think that's right, keep it a tuple.
            G_new[key_L_new][key_R_new].append(new_data)
    return G_new
def basis_map(basis):
    '''
    Provides a mapping from tenpy2 to tenpy3 Graph dictionaries. Helpful for building MPO in TenPy3 using Markov Processes from Tenpy2.
    The main difference between the two graphs is the mapping 'R'(ready) <-----> 'IdL' (identities to the left) and 'F'(finished) <-----> 'IdR' (identities to the right)
    Another thing i did is, the more complicated systems has nodes labelled by ('str1','str2',#) eg ('Mk','zz',0) and i turned them into just strings by doing ('str1','str2',#)----> '('str1','str2',#)'.
        ultimately the labels shouldn't matter at all after creating the Graph
    Another thing is the floats for the strength of the operators. While going from tenpy2 (and python2) to tenpy3 (and python3) and pickling the dictionary, the floats et a weird np.float64(float) thing in front of them that i deal with.
        code should work fine even if the pickling works 'correctly'
    
    Input: The Graph at site i
    Output: The mapped Graph at site i
    '''
    #print(len(G))
    #print(G[0])
    #quit()
    basis_new= []
    for key_L in basis:
        if key_L == 'R': key_L_new = 'IdL'
        elif key_L == 'F': key_L_new = 'IdR'
        else:
            if isinstance(key_L,tuple): 
                #print(key_L)
                keyara=[]
                for i in range(len(key_L)):
                    if isinstance(key_L[i], str):
                        keyara.append(key_L[i])
                    elif isinstance(key_L[i], np.int64):
                        keyara.append(int(key_L[i]))
                    elif isinstance(key_L[i], np.float64):
                        keyara.append(float(key_L[i]))
                    else:
                        keyara.append(key_L[i])
                
                keyara=tuple(keyara)
                #print(keyara)
                key_L_new = str(keyara) #dumb way of turning all node labels to strings
             
            else: key_L_new = key_L
        basis_new.append(key_L_new)
        
    return basis_new


def get_opnames(G):
    '''
    Given the graph, go through all nodes and prints out the physical operators.
    '''
    Opnames = []
    for k1 in G.keys():
        for k2 in G[k1].keys():
            op = G[k1][k2][0][0]
            if not isinstance(op,str): raise ValueError
            if op not in Opnames:
                Opnames.append(op)
    return Opnames
def _mpo_graph_state_order(key):
    '''
    Assignes an ordering to each MPO node based on the string
    '''
    if isinstance(key, str):
        if key == 'IdL':  # should be first
            return (-2, ) #-2 is higher priority ===> gets sorted first
        if key == 'IdR':  # should be last
            return (2, ) #-2 is lowest priority ===> gets sorted last
        # fallback: compare strings
        return (0, key) #all other strings have same priority 0 and get sorted alphabetically (should be lexicographically?) If it is lexigographically, then there should be some resonable ordering...
    else:
        raise ValueError

def set_ordered_states(states):
    '''
    creates a dictionary for each bond in the system that assigns an index to each node: {'node1':index_1,....}. We always have 'IdL':1,'IdR':-1
    '''
    res = []
    for s in states:
        d = {}
        for i, key in enumerate(sorted(s, key=_mpo_graph_state_order)):
            d[key] = i
        res.append(d)
    
    return res

def build_MPO(Model, Ws_qtotal=None):
    '''
    Input: the MPOgraph instance
    OUtput: The MPO instance
    '''
    Model.test_sanity()
    #M._set_ordered_states()
    grids = Model._build_grids()
    IdL = [s.get('IdL', None) for s in Model._ordered_states]
    IdR = [s.get('IdR', None) for s in Model._ordered_states]
    legs, Ws_qtotal = Model._calc_legcharges(Ws_qtotal)
    H = MPO.from_grids(Model.sites, grids, Model.bc, IdL, IdR, Ws_qtotal, legs, Model.max_range)
    return H

#####################################################################################################
#sketch of how it will look:
#model_parameters = {}
#site = QHSite(model_parameters)
#system size = L
#sites = [site] * L  
#G = function(model_parameters): extract the graph however you do it.
#G = G2G_map(G) #make whatever necessary changes to the graph
#M = MPOGraph(sites=sites,bc='infinite',max_range=None) : Initialize MPOGRAPH instance
#States = [ set(G[0].keys()) , set( [ k for r in G[0].values() for k in r.keys()]  ) ] extract auxilliary state strings from the Graph
#M.states = States : Initialize aux. states in model
#M._ordered_states = set_ordered_states(States) : sort these states(assign an idnex to each one)
#M.graph = G : INppuut the graph in the model 
#grids =M._build_grids():Build the grids from the graph
#H = build_MPO(M,None): BUild the MPO
