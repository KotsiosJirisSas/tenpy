#!/usr/bin/env python
"""Code for running DMRG """
import sys
import os
sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 
import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models import multilayer_qh_DP_final as mod
import itertools
from tenpy.networks.mps import MPS
from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg
from tenpy.networks.mpo import MPO
from tenpy.models.lattice import Chain
from tenpy.networks.site import QH_MultilayerFermionSite_2
from tenpy.networks.mpo import MPOGraph
from tenpy.networks.mpo import MPO
import QH_G2MPO
import QH_Graph_final
from tenpy.networks.site import QH_MultilayerFermionSite
print(sys.executable)




###	Layer naming scheme	###
# See line 22 of multilayer_qh for details
# A layer is a component - like a spin species, or bilayer index. It can have any key as name.
# Each layer can have multiple Landau levels.
# The Hilbert space is specified by
# 'layers': [  ( layer_key, landau level), (layer_key, landau level) , . . . ]
# For example,  [ ('up', 0), ('up', 1), ('down', 1) ]
# will have two species, 'up' and 'down'; 'up' contains LLs of n = 0 and 1, and 'down' a single n = 1 LL.
# For nu = 1/3,   we will use  [ ('L', LL)], with LL specified below.
"""
np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)
#########################
Lx = 14;			# circumference
LL = 0;			# which Landau level to put in
mixing_chi = 450; #Bond dimension in initial sweeps
chi = 450;		#Bond dimension of MPS
xi = 6;			# The Gaussian falloff for the Coulomb potential
Veps = 1e-4		# how accurate to approximate the MPO


NLL = 1; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)
V = { 'eps':Veps, 'xiK':xi, 'GaussianCoulomb': {('L','L'):{'v':1, 'xi':xi}} }

root_config = np.array([0, 1, 0])		# this is how the initial wavefunction looks

model_par = {
	'verbose': 3,
	'layers': [ ('L', LL) ],
	'Lx': Lx,
	'Vs': V,
	'boundary_conditions': ('infinite', 1),
	'cons_C': 'total', #Conserve number for each species (only one here!)
	'cons_K': False, #Conserve K
	'root_config': root_config, #Uses this to figure out charge assignments
	'exp_approx': '1in', #For multiple orbitals, 'slycot' is more efficient; but for 1 orbital, Roger's handmade code '1in' is slightly more efficient
}

dmrg_par = {
	'N_STEPS': 2,
	#'STARTING_ENV_FROM_PSI': 21,
	'MAX_STEPS': 36,
	'MIN_STEPS': 16,
	'MAX_ERROR_E' : 1e-6,
	'MAX_ERROR_S' : 1e-4,
	'CHI_LIST': {0:mixing_chi, 12:chi},
	'TRUNC_CUT': 1e-9,
	'LANCZOS_PAR' : {'N_min': 2, 'N_max': 20, 'p_tol': 5e-6, 'p_tol_to_trunc': 1/25., 'cache_v':np.inf},
	'mixer': (0.000001, 2., 10, 'id'),
}
"""

np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

NLL = 1; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)

#Potential data for (single/multilayered) model Laughlin
V = { 'eps':Veps, 'xiK':2., 'rV(r)': { ('L','L'): {'rV': rvr} }, 'coulomb': { ('L','L'):  {'v':-1., 'xi': xi}} }

root_config = [0]*NLL


N=3
model_par = {

	
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
#G_og=[{('a_', 0, 3): {('a_', 0, 4): [('StrOp', 1.0)], ('a', 0, 3, 'A', 0, 0): [('AOp', 1.0)], ('a', 0, 3, 'A', 0, 1): [('AOp', 1.0)]}, ('_a', 0, 3): {('_a', 0, 2): [('StrOp', 1.0)]}, ('A_', 0, 6): {}, ('a', 0, 3, 'A', 0, 1): {('a', 0, 3, 'A', 0, 1): [('Id', 0.34475573503532636)], ('_a', 0, 2): [('AOp', -0.019830660130506394)]}, ('_a', 0, 2): {('_a', 0, 1): [('StrOp', 1.0)]}, ('A', 0, 1, 'a', 0, 0): {('_A', 0, 0): [('aOp', 0.13832145119393363)], ('A', 0, 1, 'a', 0, 0): [('Id', 0.8005881679881267)]}, ('A', 0, 2, 'a', 0, 1): {('_A', 0, 1): [('aOp', -0.01359559860484505)], ('A', 0, 2, 'a', 0, 2): [('Id', 0.21324533869805767)], ('A', 0, 2, 'a', 0, 1): [('Id', 0.3425333055817517)]}, ('_A', 0, 1): {('_A', 0, 0): [('StrOp', 1.0)]}, ('a', 0, 2, 'A', 0, 2): {('_a', 0, 1): [('AOp', 0.013062430119161455)], ('a', 0, 2, 'A', 0, 2): [('Id', 0.3425333055817541)], ('a', 0, 2, 'A', 0, 1): [('Id', -0.21324533869805679)]}, ('_a', 0, 4): {('_a', 0, 3): [('StrOp', 1.0)]}, ('_A', 0, 0): {'F': [('AOp', 1.0)]}, ('a', 0, 3, 'A', 0, 0): {('a', 0, 3, 'A', 0, 0): [('Id', 0.4619497352374192)], ('_a', 0, 2): [('AOp', 0.025248692719033152)]}, ('a', 0, 4, 'A', 0, 1): {('a', 0, 4, 'A', 0, 1): [('Id', 0.33871110444159785)], ('_a', 0, 3): [('AOp', 0.006402357895473034)], ('a', 0, 4, 'A', 0, 0): [('Id', -0.040305948895300314)]}, ('a', 0, 5, 'A', 0, 0): {('a', 0, 5, 'A', 0, 0): [('Id', 0.5138512188414278)], ('_a', 0, 4): [('AOp', 0.00011023908392563828)]}, 'F': {'F': [('Id', 1.0)]}, ('A', 0, 0, 'a', 0, 0): {('A', 0, 0, 'a', 0, 0): [('Id', 0.8337251361291302)], ('A', 0, 0, 'a', 0, 1): [('Id', 0.09379746220264774)], 'F': [('nOp', -0.2660018610388971)]}, ('A', 0, 1, 'a', 0, 3): {('A', 0, 1, 'a', 0, 4): [('Id', 0.23708339098378298)], ('_A', 0, 0): [('aOp', -0.04380586818225882)], ('A', 0, 1, 'a', 0, 3): [('Id', 0.38172630104765676)]}, ('_A', 0, 2): {('_A', 0, 1): [('StrOp', 1.0)]}, ('a_', 0, 2): {('a_', 0, 3): [('StrOp', 1.0)], ('a', 0, 2, 'A', 0, 0): [('AOp', 1.0)], ('a', 0, 2, 'A', 0, 1): [('AOp', 1.4142135623730951)]}, ('A_', 0, 5): {('A', 0, 5, 'a', 0, 0): [('aOp', 1.0)], ('A_', 0, 6): [('StrOp', 1.0)]}, ('_A', 0, 5): {('_A', 0, 4): [('StrOp', 1.0)]}, ('a', 0, 1, 'A', 0, 4): {('a', 0, 1, 'A', 0, 4): [('Id', 0.38172630104763583)], ('_a', 0, 0): [('AOp', 0.030390807643815607)], ('a', 0, 1, 'A', 0, 3): [('Id', -0.23708339098377437)]}, ('a', 0, 1, 'A', 0, 1): {('_a', 0, 0): [('AOp', 0.0061305932314431565)], ('a', 0, 1, 'A', 0, 1): [('Id', 0.7630762915951965)], ('a', 0, 1, 'A', 0, 2): [('Id', 0.14797043763385084)]}, ('a', 0, 4, 'A', 0, 0): {('a', 0, 4, 'A', 0, 1): [('Id', 0.040305948895300314)], ('_a', 0, 3): [('AOp', 0.0006454127172432714)], ('a', 0, 4, 'A', 0, 0): [('Id', 0.33871110444159785)]}, ('_A', 0, 4): {('_A', 0, 3): [('StrOp', 1.0)]}, ('A_', 0, 4): {('A_', 0, 5): [('StrOp', 1.0)], ('A', 0, 4, 'a', 0, 0): [('aOp', 1.4142135623730951)]}, ('A', 0, 0, 'a', 0, 1): {('A', 0, 0, 'a', 0, 0): [('Id', -0.09379746220264774)], ('A', 0, 0, 'a', 0, 1): [('Id', 0.8337251361291302)], 'F': [('nOp', 0.05253891639028042)]}, ('A', 0, 1, 'a', 0, 2): {('_A', 0, 0): [('aOp', -0.01009450230953929)], ('A', 0, 1, 'a', 0, 1): [('Id', -0.14797043763386378)], ('A', 0, 1, 'a', 0, 2): [('Id', 0.7630762915952091)]}, ('a', 0, 2, 'A', 0, 0): {('_a', 0, 1): [('AOp', 0.042544852049328626)], ('a', 0, 2, 'A', 0, 0): [('Id', 0.6410408600755128)]}, ('A', 0, 0, 'a', 0, 5): {('A', 0, 0, 'a', 0, 5): [('Id', 0.3495781422729576)], 'F': [('nOp', -0.23103170492184383)]}, ('A', 0, 2, 'a', 0, 2): {('_A', 0, 1): [('aOp', 0.013062430119161288)], ('A', 0, 2, 'a', 0, 2): [('Id', 0.3425333055817517)], ('A', 0, 2, 'a', 0, 1): [('Id', -0.21324533869805767)]}, ('A', 0, 3, 'a', 0, 1): {('A', 0, 3, 'a', 0, 1): [('Id', 0.34475573503532886)], ('_A', 0, 2): [('aOp', -0.019830660130507678)]}, ('A', 0, 4, 'a', 0, 0): {('A', 0, 4, 'a', 0, 1): [('Id', 0.040305948895300446)], ('_A', 0, 3): [('aOp', 0.0006454127172432719)], ('A', 0, 4, 'a', 0, 0): [('Id', 0.3387111044415981)]}, ('A', 0, 0, 'a', 0, 4): {('A', 0, 0, 'a', 0, 4): [('Id', 0.2943158206367369)], ('A', 0, 0, 'a', 0, 3): [('Id', -0.36314516089054333)], 'F': [('nOp', -0.026162348887412326)]}, ('a', 0, 1, 'A', 0, 0): {('_a', 0, 0): [('AOp', 0.1383214511939045)], ('a', 0, 1, 'A', 0, 0): [('Id', 0.8005881679881443)]}, ('a', 0, 2, 'A', 0, 1): {('_a', 0, 1): [('AOp', -0.013595598604845185)], ('a', 0, 2, 'A', 0, 2): [('Id', 0.21324533869805679)], ('a', 0, 2, 'A', 0, 1): [('Id', 0.3425333055817541)]}, ('a_', 0, 1): {('a', 0, 1, 'A', 0, 1): [('AOp', 1.4142135623730951)], ('a', 0, 1, 'A', 0, 0): [('AOp', 1.0)], ('a', 0, 1, 'A', 0, 3): [('AOp', 1.4142135623730951)], ('a_', 0, 2): [('StrOp', 1.0)]}, ('A_', 0, 1): {('A', 0, 1, 'a', 0, 1): [('aOp', 1.4142135623730951)], ('A', 0, 1, 'a', 0, 0): [('aOp', 1.0)], ('A', 0, 1, 'a', 0, 3): [('aOp', 1.4142135623730951)], ('A_', 0, 2): [('StrOp', 1.0)]}, ('a_', 0, 6): {}, ('_a', 0, 5): {('_a', 0, 4): [('StrOp', 1.0)]}, ('_a', 0, 0): {'F': [('aOp', 1.0)]}, ('A', 0, 1, 'a', 0, 1): {('_A', 0, 0): [('aOp', 0.006130593231424408)], ('A', 0, 1, 'a', 0, 1): [('Id', 0.7630762915952091)], ('A', 0, 1, 'a', 0, 2): [('Id', 0.14797043763386378)]}, ('A', 0, 3, 'a', 0, 0): {('A', 0, 3, 'a', 0, 0): [('Id', 0.4619497352374154)], ('_A', 0, 2): [('aOp', 0.025248692719034436)]}, ('A', 0, 4, 'a', 0, 1): {('A', 0, 4, 'a', 0, 1): [('Id', 0.3387111044415981)], ('_A', 0, 3): [('aOp', 0.006402357895473009)], ('A', 0, 4, 'a', 0, 0): [('Id', -0.040305948895300446)]}, ('a_', 0, 5): {('a', 0, 5, 'A', 0, 0): [('AOp', 1.0)], ('a_', 0, 6): [('StrOp', 1.0)]}, ('A', 0, 5, 'a', 0, 0): {('A', 0, 5, 'a', 0, 0): [('Id', 0.5138512188414278)], ('_A', 0, 4): [('aOp', 0.00011023908392563828)]}, ('a', 0, 1, 'A', 0, 3): {('a', 0, 1, 'A', 0, 4): [('Id', 0.23708339098377437)], ('_a', 0, 0): [('AOp', -0.043805868182256937)], ('a', 0, 1, 'A', 0, 3): [('Id', 0.38172630104763583)]}, ('_A', 0, 3): {('_A', 0, 2): [('StrOp', 1.0)]}, ('A_', 0, 2): {('A_', 0, 3): [('StrOp', 1.0)], ('A', 0, 2, 'a', 0, 0): [('aOp', 1.0)], ('A', 0, 2, 'a', 0, 1): [('aOp', 1.4142135623730951)]}, ('A', 0, 0, 'a', 0, 3): {('A', 0, 0, 'a', 0, 4): [('Id', 0.36314516089054333)], ('A', 0, 0, 'a', 0, 3): [('Id', 0.2943158206367369)], 'F': [('nOp', -0.015888996698317146)]}, ('A', 0, 1, 'a', 0, 4): {('A', 0, 1, 'a', 0, 4): [('Id', 0.38172630104765676)], ('_A', 0, 0): [('aOp', 0.03039080764381981)], ('A', 0, 1, 'a', 0, 3): [('Id', -0.23708339098378298)]}, ('_a', 0, 1): {('_a', 0, 0): [('StrOp', 1.0)]}, 'R': {('A', 0, 0, 'a', 0, 5): [('nOp', 1.0)], ('A', 0, 0, 'a', 0, 0): [('nOp', 1.4142135623730951)], ('A_', 0, 1): [('AOp', 1.0)], ('A', 0, 0, 'a', 0, 3): [('nOp', 1.4142135623730951)], ('a_', 0, 1): [('aOp', 1.0)], 'R': [('Id', 1.0)], ('A', 0, 0, 'a', 0, 2): [('nOp', 1.0)]}, ('A', 0, 2, 'a', 0, 0): {('_A', 0, 1): [('aOp', 0.042544852049328445)], ('A', 0, 2, 'a', 0, 0): [('Id', 0.6410408600755139)]}, ('A_', 0, 3): {('A_', 0, 4): [('StrOp', 1.0)], ('A', 0, 3, 'a', 0, 0): [('aOp', 1.0)], ('A', 0, 3, 'a', 0, 1): [('aOp', 1.0)]}, ('a_', 0, 4): {('a_', 0, 5): [('StrOp', 1.0)], ('a', 0, 4, 'A', 0, 0): [('AOp', 1.4142135623730951)]}, ('a', 0, 1, 'A', 0, 2): {('_a', 0, 0): [('AOp', -0.010094502309538074)], ('a', 0, 1, 'A', 0, 1): [('Id', -0.14797043763385084)], ('a', 0, 1, 'A', 0, 2): [('Id', 0.7630762915951965)]}, ('A', 0, 0, 'a', 0, 2): {('A', 0, 0, 'a', 0, 2): [('Id', 0.8330402460772576)], 'F': [('nOp', 0.8454122673973181)]}}]


print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)
print("Old code finished producing MPO graph",".."*10)
#quit()
G=M.MPOgraph
#dict1=G[0]
#dict2=G_og[0]
#for a in G[0].keys():
#	only_in_dict1 =  dict2[a].keys()#-dict1[a].keys()
#	for b in dict2[a].keys():
#		print(dict1[a][b]==dict2[a][b])
#		print(dict1[a][b])
#		print(dict2[a][b])
#		#print(only_in_dict1)
#quit()

#print(G_og[0]['R']['R'])
#print('BREEAAAAAAAK')
#print(G[0]==G_og[0])
#quit()

#print(G[0].keys())
#x=G[0][('_a', 0, 2)]

#print(x)
#quit()
#v={('Mk', 'AL-6-aL.11', 0): [('Id', np.float64(0.9492759810967927))], ('Mk', 'AL-6-aL.11', 1): [('Id', np.float64(0.272277824486316))], ('Mk', 'AL-6-aL.11', 2): [('Id', np.float64(0.0064367729656733845))], ('Mk', 'AL-6-aL.11', 3): [('Id', np.float64(-0.05451767910804409))], ('Mk', 'AL-6-aL.11', 4): [('Id', np.float64(0.008568176372579986))], ('Mk', 'AL-6-aL.11', 5): [('Id', np.float64(0.019895117063725355))], ('Mk', 'AL-6-aL.11', 6): [('Id', np.float64(-0.0072862507291826676))], ('Mk', 'AL-6-aL.11', 7): [('Id', np.float64(2.9239849781939463e-05))], ('Mk', 'AL-6-aL.11', 8): [('Id', np.float64(-0.0131756138667698))], ('Mk', 'AL-6-aL.11', 9): [('Id', np.float64(-0.0026713558619229027))], ('Mk', 'AL-6-aL.11', 10): [('Id', np.float64(-0.0014369395309839633))], ('Mk', 'AL-6-aL.11', 11): [('Id', np.float64(0.0034659985118308833))], ('Mk', 'AL-6-aL.11', 12): [('Id', np.float64(-0.0011354223380746977))], ('Mk', 'AL-6-aL.11', 13): [('Id', np.float64(0.003598450249208297))], ('Mk', 'AL-6-aL.11', 14): [('Id', np.float64(0.002166924931815965))], ('Mk', 'AL-6-aL.11', 15): [('Id', np.float64(-0.00059835814516991))], ('Mk', 'AL-6-aL.11', 16): [('Id', np.float64(0.0049199443231798924))], ('Mk', 'AL-6-aL.11', 17): [('Id', np.float64(-0.005206551347353976))], ('Mk', 'AL-6-aL.11', 18): [('Id', np.float64(0.004384502049434996))], ('Mk', 'AL-6-aL.11', 19): [('Id', np.float64(0.0006989354883427595))], ('Mk', 'AL-6-aL.11', 20): [('Id', np.float64(0.0006537599421993103))], ('Mk', 'AL-6-aL.11', 21): [('Id', np.float64(-0.0009739624671363384))], ('Mk', 'AL-6-aL.11', 22): [('Id', np.float64(0.002481823083080848))], ('Mk', 'AL-6-aL.11', 23): [('Id', np.float64(-0.0007796396119069333))], ('Mk', 'AL-6-aL.11', 24): [('Id', np.float64(-0.0004180988081148821))], ('Mk', 'AL-6-aL.11', 25): [('Id', np.float64(-0.002297156494485442))], ('Mk', 'AL-6-aL.11', 26): [('Id', np.float64(0.00165281050129148))], ('Mk', 'AL-6-aL.11', 27): [('Id', np.float64(0.0022936103664082425))], ('_A', np.int64(0), np.int64(5)): [('aOp', np.float64(2.257236479579241e-06))]}
#b={('Mk', 'AL-6-aL.11', 0): [('Id', np.float64(0.9492759814609087))], ('Mk', 'AL-6-aL.11', 1): [('Id', np.float64(0.27227782832587044))], ('Mk', 'AL-6-aL.11', 2): [('Id', np.float64(0.006436771511995403))], ('Mk', 'AL-6-aL.11', 3): [('Id', np.float64(-0.05451767433012043))], ('Mk', 'AL-6-aL.11', 4): [('Id', np.float64(0.008568175628151946))], ('Mk', 'AL-6-aL.11', 5): [('Id', np.float64(0.019895114241533944))], ('Mk', 'AL-6-aL.11', 6): [('Id', np.float64(-0.007286249989640748))], ('Mk', 'AL-6-aL.11', 7): [('Id', np.float64(2.9244241133726508e-05))], ('Mk', 'AL-6-aL.11', 8): [('Id', np.float64(-0.013175613823802971))], ('Mk', 'AL-6-aL.11', 9): [('Id', np.float64(-0.0026713574254690737))], ('Mk', 'AL-6-aL.11', 10): [('Id', np.float64(-0.0014369403686344607))], ('Mk', 'AL-6-aL.11', 11): [('Id', np.float64(0.0034660035478183194))], ('Mk', 'AL-6-aL.11', 12): [('Id', np.float64(-0.0011354226598752695))], ('Mk', 'AL-6-aL.11', 13): [('Id', np.float64(0.0035984520851457414))], ('Mk', 'AL-6-aL.11', 14): [('Id', np.float64(-0.002166927982600214))], ('Mk', 'AL-6-aL.11', 15): [('Id', np.float64(-0.0005983483721298887))], ('Mk', 'AL-6-aL.11', 16): [('Id', np.float64(-0.004919933391840824))], ('Mk', 'AL-6-aL.11', 17): [('Id', np.float64(-0.005206555977500087))], ('Mk', 'AL-6-aL.11', 18): [('Id', np.float64(0.004384502391492693))], ('Mk', 'AL-6-aL.11', 19): [('Id', np.float64(-0.0006989330086271956))], ('Mk', 'AL-6-aL.11', 20): [('Id', np.float64(0.000653757624726711))], ('Mk', 'AL-6-aL.11', 21): [('Id', np.float64(0.0009739594645053874))], ('Mk', 'AL-6-aL.11', 22): [('Id', np.float64(0.0024818153108455768))], ('Mk', 'AL-6-aL.11', 23): [('Id', np.float64(0.000779654293089555))], ('Mk', 'AL-6-aL.11', 24): [('Id', np.float64(0.000418121547337863))], ('Mk', 'AL-6-aL.11', 25): [('Id', np.float64(-0.0022971504713420465))], ('Mk', 'AL-6-aL.11', 26): [('Id', np.float64(0.0016528145582151432))], ('Mk', 'AL-6-aL.11', 27): [('Id', np.float64(-0.00229361147678582))], ('_A', np.int64(0), np.int64(5)): [('aOp', np.float64(2.257236377525245e-06))]}
#print(v==G[0][('Mk', 'AL-6-aL.11', 0)])
#quit()
print(G[0][('Mk', 'AL-6-aL.11', 0)])
quit()
G_new=QH_Graph_final.obtain_new_tenpy_MPO_graph(G)
#print(G_new[0][])
#{"('Mk', 'AL-6-aL.11', 0)": [('Id', -0.0022936103664083296)], "('Mk', 'AL-6-aL.11', 1)": [('Id', -0.012011554777185388)], "('Mk', 'AL-6-aL.11', 2)": [('Id', 0.006983104212694983)], "('Mk', 'AL-6-aL.11', 3)": [('Id', 0.03226845207125104)], "('Mk', 'AL-6-aL.11', 4)": [('Id', 0.0052054391215796345)], "('Mk', 'AL-6-aL.11', 5)": [('Id', -0.052969495630489076)], "('Mk', 'AL-6-aL.11', 6)": [('Id', 0.004782068458101691)], "('Mk', 'AL-6-aL.11', 7)": [('Id', -0.009226238884621787)], "('Mk', 'AL-6-aL.11', 8)": [('Id', 0.12181012215361176)], "('Mk', 'AL-6-aL.11', 9)": [('Id', 0.02680274923762687)], "('Mk', 'AL-6-aL.11', 10)": [('Id', 0.02613075320426292)], "('Mk', 'AL-6-aL.11', 11)": [('Id', -0.03887269255818529)], "('Mk', 'AL-6-aL.11', 12)": [('Id', -0.011587382233824599)], "('Mk', 'AL-6-aL.11', 13)": [('Id', -0.051890294683347715)], "('Mk', 'AL-6-aL.11', 14)": [('Id', -0.045196435317422325)], "('Mk', 'AL-6-aL.11', 15)": [('Id', 0.00770847110335915)], "('Mk', 'AL-6-aL.11', 16)": [('Id', -0.08796050017973375)], "('Mk', 'AL-6-aL.11', 17)": [('Id', 0.0524504387817054)], "('Mk', 'AL-6-aL.11', 18)": [('Id', -0.22693959239263164)], "('Mk', 'AL-6-aL.11', 19)": [('Id', -0.07287589757499625)], "('Mk', 'AL-6-aL.11', 20)": [('Id', -0.017177801279661543)], "('Mk', 'AL-6-aL.11', 21)": [('Id', 0.11145661506377952)], "('Mk', 'AL-6-aL.11', 22)": [('Id', -0.21779847616078557)], "('Mk', 'AL-6-aL.11', 23)": [('Id', 0.11180765580960449)], "('Mk', 'AL-6-aL.11', 24)": [('Id', 0.0018771109848217421)], "('Mk', 'AL-6-aL.11', 25)": [('Id', 0.23475438624686132)], "('Mk', 'AL-6-aL.11', 26)": [('Id', -0.15502741255614333)], "('Mk', 'AL-6-aL.11', 27)": [('Id', 0.544024526982654)], "('_A', 0, 5)": [('aOp', 4.342318515562255e-07)]}
print(G_new[0]["('Mk', 'AL-6-aL.11', 27)"])
quit()
#print(G_new[0]["('_a', 0, 2)"])
#quit()
#print(G_new)
#quit()
#print(len(G_new))
#quit()
#G_new=[G_new[0],G_new[0],G_new[0]]
#print(len(G_new))
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
#spin=QH_MultilayerFermionSite(N=1)
L = len(G_new) #System size for finite case, or unit cell for infinite
sites = [spin] * L 
M = MPOGraph(sites=sites,bc='finite',max_range=None) #: Initialize MPOGRAPH instance

'''
M.states holds the keys for the auxilliary states of the MPO. These states live on the bonds.

Bond s is between sites s-1,s and there are L+1 bonds, meaning there is a bond 0 but also a bond L.
The rows of W[s] live on bond s while the columns of W[s] live on bond s+1
'''

States=QH_Graph_final.obtain_states_from_graphs(G_new,L)
print("Ordering states",".."*10)

M.states = States #: Initialize aux. states in model
M._ordered_states = QH_G2MPO.set_ordered_states(States) #: sort these states(assign an index to each one)
print("Finished",".."*10 )



print("Test sanity"+".."*10)
M.test_sanity()
M.graph = G_new #: Inpput the graph in the model 
print("Test passed!"+".."*10)
grids =M._build_grids()#:Build the grids from the graph

#quit()
print("Building MPO"+".."*10)
H = QH_G2MPO.build_MPO(M,None)#: Build the MPO
print("Built"+".."*10)





#initialize wavefunction as MPS
pstate=["empty", "full","empty"]
psi = MPS.from_product_state(sites, pstate, bc="infinite")


#initialize MPOModel
lattice=Chain(L,spin, bc="periodic",  bc_MPS="infinite")
model=MPOModel(lattice, H)

dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True, "max_sweeps":100}

print("Run DMRG:")
engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)  
E0, psi = engine.run()


print("E =", E0)
print("Finished running DMRG")




Length=psi.correlation_length()
print('correlation length:',Length)



filling=psi.expectation_value("nOp")

print('Filling:',filling)


E_spec=psi.entanglement_spectrum()
print('entanglement spectrum:',E_spec)



EE=psi.entanglement_entropy()
print('entanglement entropy:',EE)


#SAVE THE DATA
import h5py
from tenpy.tools import hdf5_io

data = {"psi": psi,  # e.g. an MPS
        "dmrg_params":dmrg_params, "model_par":model_par, "model": model }

name="nu=1_3_charge_conservation_no_K_conservation"
with h5py.File(name+".h5", 'w') as f:
    hdf5_io.save_to_hdf5(f, data)