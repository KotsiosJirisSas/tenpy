"""PRODUCES GRAPH AND MPO FOR QUANTUM HALL SYSTEMS"""
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
from tenpy.networks.site import QH_MultilayerFermionSite_2
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
import QH_G2MPO


def obtain_new_tenpy_MPO_graph(G):
	print("Change the old tenpy MPO graph to new tenpy MPO graph",".."*10)
	G_new=[]
	for i in range(len(G)):
		G_new.append( QH_G2MPO.G2G_map(G[i])) #G2G maps {}  --> {} so this will now work for graphs which are lists of dictionaries

	print("finished",".."*10)
	if len(G_new) != len(G):
		print('Something went wrong in creating the list of dictionaries')
		raise ValueError
	return G_new




def obtain_states_from_graphs(G_new,L):
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
		print("Finished",".."*10 )

	return States





