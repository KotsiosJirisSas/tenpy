import pickle
import numpy as np
import sys
import os
sys.path.append('/home/v/vasiliou/tenpynew2/tenpy') #comment out if not me
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO

'''
~~~10 Oct 2024~~~
~~~~~~~~KV~~~~~~~
Here I am trying to create  a spin 1/2 model for the Long-range spin Hamiltonian 
H = \\sum_{i,r}( Jz[r] Z_i x Z_{i+r} + J_{r} (X_i X_{i+r}+Y_{i}Y_{i+r})
Where J ~ 1/r^2. The approximation of this as a sum of exponential terms is done with slycot in old tenpy and the Graph G is inputted here (data.pkl) as the starting point.
Dom altered the MPO_Graph code of old tenpy and it can be used here directly for G instead of inputting from old tenpy, I just cant actually run that code cause i get errors when i import the additional_modules firectory


==========================================================================================================================================================================
Notes
==========================================================================================================================================================================
Note 1: Need to take care that the local operators match between old and new Tenpy.
        So eg the data.pkl and data_3.pkl have in the graph local operators 'RSp' and 'RSm' which are rescaled 'Sp' and 'Sm'but 
        current version of the code fails at M.build_grids() step because the site instance we are using does not recognise them. 
        Easiest soln is to add the attributes directly in the sites class (SpinHalfSite class). The alternative is to change the G2G map to take that into account
Note 2: generalize part of the code when W's are more that 1 site. Only the mapping aspect should need updating and the part where i explicitly set the auxilliary space states
Note 3: Dom altered the MPO_Graph code of old tenpy to TenPy3 and it can be used here directly for G instead of inputting from old tenpy, I just cant actually run that code cause i get 
        errors when i import the additional_modules directory...
==========================================================================================================================================================================
Tests to do
==========================================================================================================================================================================
Test 1: Compare this with eg c_mps_mpo.py model which builds directly from the Ws. We can easily recreate the graph G needed for that rather than import it because its 
        a very simple graph.
Test 2: Make sure this works fine with local operators operators with non-zero charge (so the Sp,Sm)
Test 3: Test this works for finite case as well
'''
#data.pkl has non-zero Jz,J_\pm
#data_2.pkl only has non-zero Jz. so its simpler.
#data_3.pkl only has non-zero J\pm.
#All these were created in the haldaneshastry.py model in old Tenpy

with open('data.pkl', 'rb') as f:
    loaded_data = pickle.load(f, encoding='latin1')
with open('data_2.pkl', 'rb') as f:
    loaded_data_2 = pickle.load(f, encoding='latin1')
with open('data_3.pkl', 'rb') as f:
    loaded_data_3 = pickle.load(f, encoding='latin1')

G_old = loaded_data_2

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
    G_new = {}
    for key_L in G.keys():
        if key_L == 'R': key_L_new = 'IdL'
        elif key_L == 'F': key_L_new = 'IdR'
        else:
            if isinstance(key_L,tuple): key_L_new = str(key_L) #dumb way of turning all node labels to strings
            else: key_L_new = key_L
        if key_L_new not in G_new:
            G_new[key_L_new] = {} # or []?
        ###################################
        for key_R in G[key_L].keys():
            if key_R == 'R': key_R_new = 'IdL'
            elif key_R == 'F': key_R_new = 'IdR'
            else:
                if isinstance(key_R,tuple): key_R_new = str(key_R) #dumb way of turning all node labels to strings
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
def get_opnames(G):
    '''
    Given the graph, go through all nodes and print out the physical operators
    '''
    Opnames = []
    for k1 in G.keys():
        for k2 in G[k1].keys():
            op = G[k1][k2][0][0]
            if not isinstance(op,str): raise ValueError
            if op not in Opnames:
                Opnames.append(op)
    return Opnames

print('-'*100)
G_new = G2G_map(G_old[0])
print(get_opnames(G_new))
#So G has been mapped correctly

####################
# ATTACH G TO A MODEL!!!
####################
#STEP1: CREATE HILBERT SPACE AND CHARGE INFORMATION
spin = SpinHalfSite(conserve="Sz")
L = 1  # number of unit cell for infinite system
sites = [spin] * L  # repeat entry of list N times
M = MPOGraph(sites=sites,bc='infinite',max_range=None)
print('='*100)
print('Initialized the model')
print('Inifinite or finite? ',M.bc)
print('Unit cell size: L = ',M.L)
print('Physical Hilbert Space:',M.sites[0].state_labels)
print('Physical leg charge info:',M.sites[0].leg)
print('='*100)
print(M._ordered_states)# These ordered states live on the bonds not on sites, hence having L+1 of them. But really states[-1] == states[0] when it comes to bonds.
#######################
#extract the states explicitly for each bond. Here I am doing this explicitly because there is only L=0 but in general, need to generalize...
#Then, attach a dictionary to them to label their index. Then finally endow the model instance with self.states, self._ordered_states attributes!
G = []
G.append(G_new)
States = [ set(G[0].keys()) , set( [ k for r in G[0].values() for k in r.keys()]  ) ] #formed from rows and columns

def _mpo_graph_state_order(key):
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
    res = []
    for s in states:
        d = {}
        for i, key in enumerate(sorted(s, key=_mpo_graph_state_order)):
            d[key] = i
        res.append(d)
    return res
    
res = set_ordered_states(States)
M.states = States
M._ordered_states = res
M.test_sanity()
IdL = [s.get('IdL', None) for s in res] #[0,0] in our case. Two 0's cause two bonds
IdR = [s.get('IdR', None) for s in res] #[13,13] in our case. Two cause two bonds
#####################################################
#build grid, by first setting the graph attribute.
# At this point we have instead of a dictionary a grid whose elements are tuples that look like W_ij = [('Op_name',Op_strength)]
M.graph = G
grids =M._build_grids()
###################################################
#Now the big step, calculate the leg charges and create the MPO
print('Sp charge',M.sites[0].get_op('Sp').qtotal)
print('Sm charge',M.sites[0].get_op('Sm').qtotal)
print('Sz charge',M.sites[0].get_op('Sz').qtotal)
print('Id charge',M.sites[0].get_op('Id').qtotal)
print('-'*100)
def build_MPO(Model, Ws_qtotal=None):
    Model.test_sanity()
    #M._set_ordered_states()
    grids = Model._build_grids()
    IdL = [s.get('IdL', None) for s in Model._ordered_states]
    IdR = [s.get('IdR', None) for s in Model._ordered_states]
    legs, Ws_qtotal = M._calc_legcharges(Ws_qtotal)
    H = MPO.from_grids(Model.sites, grids, Model.bc, IdL, IdR, Ws_qtotal, legs, Model.max_range)
    return H
H = build_MPO(M,None)

quit()
