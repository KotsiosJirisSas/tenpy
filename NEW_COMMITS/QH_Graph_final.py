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




def obtain_states_from_graphs(G_new,L, bc="infinite"):
	print("Initialize states ",".."*10 )
	States = []
	not_included_couplings=[]
	if bc=="finite":
		#have yet to implement it fully for finite dmrg counting of zero states even though graph does not read them as zero
		for bond in np.arange(L+1):
			if bond == 0:
				states_from_rows = set()
				states_from_rows = set(G_new[bond].keys())
				
				States.append(states_from_rows)
				not_included_couplings.append([])
			if 0 < bond < L:
				states_from_rows = set()
				states_from_columns = set()
				states = set()
				states_from_rows = set(G_new[bond].keys())
				states_from_columns = set([ k for r in G_new[bond-1].values() for k in r.keys()])
				

				#HMM WHAT GOES WRONG HERE???
				states = states_from_rows & states_from_columns #take intersection - MAYBE NEED UNION INTEAD??

				states2=states_from_rows | states_from_columns
				
				#ADD NEW ZERO ELEMENTS TO THE GRAPH SO THAT IT CAN CALCULATE CHARGES
				row=[]
				non_included=states2-states
				for el in list(non_included):
					#print(el)
					if el in list(states_from_columns):
						#add it to the row

						row.append([el,"column"])
					else:
						#print("row")
						row.append([el,"row"])
				not_included_couplings.append(row)
				#print(non_included)
				#quit()
				#print(not_included_couplings)
				#quit()
					
				States.append(states)
			if bond == L:
				not_included_couplings.append([])
				states_from_rows = set()
				states_from_columns = set([ k for r in G_new[bond-1].values() for k in r.keys()])
				States.append(states_from_columns)
			print("Finished",".."*10 )
		
	elif bc=="infinite":
		for bond in np.arange(L+1):
			#CREATES INFINITE DMRG MODEL
			
			states_from_rows = set()
			states_from_columns = set()
			states = set()
			states_from_rows = set(G_new[bond%L].keys())
			states_from_columns = set([ k for r in G_new[(bond-1)%L].values() for k in r.keys()])
			
			#TAKE THE INTERSECTION
			#TWO SETS SHOULD BE EQUAL BECAUSE YOU MULTIPLY ONE BY ANOTHER IN MATRIX
			#IF THEY ARENT EQUAL CERTAIN ELEMENTS ARE JUST ZERO- HOWEVER GRAPH IS DISCONNECTED AND SO CHARGES WONT
			#BE CALCULATED PROPERLY
			#THUS NEED TO SET THOSE ELEMENTS EXPLICITLY TO ZERO
			#same procedure in finite DMRG, but have yet to implement it fully
			states = states_from_rows & states_from_columns #take intersection - MAYBE NEED UNION INTEAD??
			
			states = states_from_rows & states_from_columns
			#LOOK AT UNION
			states2=states_from_rows | states_from_columns
			#print(len(states))
			#quit()

			row=[]
			#look at elements which are not in row and columns
			non_included=states2-states
			#save all the elements that are not in the row/column
			for el in list(non_included):
				if el in list(states_from_columns):
					row.append([el,"column"])
				else:
					#print("row")
					row.append([el,"row"])
			not_included_couplings.append(row)
			
			#just produce states
			States.append(states)
	
	return States,not_included_couplings





