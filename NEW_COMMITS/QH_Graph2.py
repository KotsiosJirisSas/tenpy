"""Call of (finite) DMRG."""
import sys
import os
sys.path.append('/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy') 
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models import multilayer_qh_DP_final as mod
import itertools


from tenpy.networks.mps import MPS
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import QH_MultilayerFermionSite
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
import QH_G2MPO

np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

NLL = 1; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)

#Potential data for (single/multilayered) model Laughlin
V = { 'eps':Veps, 'xiK':2., 'rV(r)': { ('L','L'): {'rV': rvr} }, 'coulomb': { ('L','L'):  {'v':-1., 'xi': xi}} }

root_config = [0]*NLL


N=2
model_par = {

	#ahhhh ok ok so it constructs the periodic one, ggwp with 24 sites for some reason
	# - for reason of there being 2 layers!
	'boundary_conditions': ('infinite', N), #for finite replace by periodic here
	'verbose': 2,
	#'layers': [ ('L', l) for l in range(NLL) ],
	'layers':[ ('L', 1)],
	'Lx': 12.,
	'Vs': V,
	'cons_C': 'total',
	'cons_K': False,
	'root_config': root_config,
	'exp_approx': 'slycot',
}

#print ("-"*10 + "Comparing analytic V(q) Yukawa and V(r) Yukawa" +"-"*10)
print("START MODEL",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)


####################################################################################################
G=M.MPOgraph

print("Change the old tenpy MPO graph to new tenpy MPO graph",".."*10)
G_new=[]
for i in range(len(G)):
	G_new.append( QH_G2MPO.G2G_map(G[i])) #G2G maps {}  --> {} so this will now work for graphs which are lists of dictionaries

print("finished",".."*10)
if len(G_new) != len(G):
	print('Something went wrong in creating the list of dictionaries')
	raise ValueError

#IMPORT site
from tenpy.networks.site import FermionSite_testara
spin=FermionSite_testara(conserve='N')
L = len(G_new) #System size for finite case, or unit cell for infinite


sites = [spin] * L 


M = MPOGraph(sites=sites,bc='infinite',max_range=None) #: Initialize MPOGRAPH instance

'''
M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
'''
print("Initialize states ",".."*10 )


States = []

for bond in np.arange(L+1):
	if bond == 0:
		states_from_rows = set()
		states_from_rows = set(G_new[bond].keys())
		States.append(states_from_rows)
	if 0 < bond < L:
		states_from_rows = set()
		states_from_columns = set()
		states = set()
		states_from_rows = set(G_new[bond].keys())
		states_from_columns = set([ k for r in G_new[bond-1].values() for k in r.keys()])
		
		states = states_from_rows & states_from_columns #take intersection
		
		States.append(states)
	if bond == L:
		states_from_rows = set()
		states_from_columns = set([ k for r in G_new[bond-1].values() for k in r.keys()])
		States.append(states_from_columns)
##########################

print("Finished",".."*10 )

print("Ordering states",".."*10)
M.states = States #: Initialize aux. states in model
M._ordered_states = QH_G2MPO.set_ordered_states(States) #: sort these states(assign an index to each one)


print("Finished",".."*10 )

#quit()

print("Test sanity",".."*10)
M.test_sanity()
M.graph = G_new #: INppuut the graph in the model 
print("Test passed!"+".."*10)
grids =M._build_grids()#:Build the grids from the graph
H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
print('bildara')
quit()