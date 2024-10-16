"""Call of (finite) DMRG."""
import sys
import os
sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 
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

print ("-"*10 + "Comparing analytic V(q) Yukawa and V(r) Yukawa" +"-"*10)
print("START MODEL")
M = mod.QH_model(model_par)
print('OLD CODE FINISHED')



G=M.MPOgraph

#G_new=[]
#for i in range(len(G)):
#	G_new.append( QH_G2MPO.G2G_map(G[i])) #make whatever necessary changes to the graph
G_new=[QH_G2MPO.G2G_map(G[0])]
#print(len(G_new[0]))
from tenpy.networks.site import FermionSite_testara
#spin = SpinHalfSite(conserve=None)
spin=FermionSite_testara(conserve='N')
L = 1# number of unit cell for infinite system
sites = [spin] * L 


M = MPOGraph(sites=sites,bc='infinite',max_range=None) #: Initialize MPOGRAPH instance
#lst=G_new[0].keys()

#from collections import Counter
#repeated = [item for item, count in Counter(lst).items() if count > 0]
#print(len(repeated))
#quit()
a=[]
#print(len(G[0].values()))
for r in G_new[0].values():
	for k in r.keys():
		a.append(k)
#print(len(a))
#a=set( a  )
#print(len(a))

#alal=set(G_new[0].keys())|set(a)
#print(len(alal))
#quit()
States = [ set(G_new[0].keys()) , set( [ k for r in G_new[0].values() for k in r.keys()]  ) ] #extract auxilliary state strings from the Graph

#print(len(States[0]))
#quit()
#print(len(States[1]))
#print(States[1])
#quit()
M.states = States #: Initialize aux. states in model
M._ordered_states = QH_G2MPO.set_ordered_states(States) #: sort these states(assign an idnex to each one)
#print(M._ordered_states[1])
#print('done')
#quit()
#print(len(M._ordered_states[0] ))
#quit()
M.test_sanity()
M.graph = G_new #: INppuut the graph in the model 
#print(len(G_new[0]))

grids =M._build_grids()#:Build the grids from the graph
H = QH_G2MPO.build_MPO(M,None)#: BUild the MPO
print('bildara')
quit()

'''
H_MPO=M.H_mpo


model_params={"L": N, "bc_MPS": "infinite", 'site':None, 'bc':'periodic'}
#NEW CODE STARTS HERE

hilber_space_single_site=QH_MultilayerFermionSite(N=1)
#DO WE NEED H_bonds in chain?
define_chain=Chain(N,hilber_space_single_site)
H = MPO.from_grids([hilber_space_single_site] * N, M.H_mpo, bc='infinite', IdL=0, IdR=-1)
#MANAGED TO DEFINE A CHAIN WITH (HOPEFULLY) CORRECT MATRICES
'''