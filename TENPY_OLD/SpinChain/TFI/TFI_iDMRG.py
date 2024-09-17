'''
H = (J/4) \sum Z_{i} Z_{i+1} + (h_x/2) \sum X_{i}
H = -sum Z_{i} Z_{i+1} - g \sum X_{i}
****
Note that i am using the ZZ+X Hamiltonian insteda of the XX+Z
****
so for |g|<1: GS is Z_2 degenerate Ferromagnet, while for |g|>>1 it is a paramagnet along +x.
The GS you get in the disordered regime depends on your initial state.
The GS at the two limits are product states and so should have zero ES, while near the transition the state is highly entangled.
FOr all g, the initial wavefunction is set up to point up.
'''
###############################################################
#################IMPORTING PACKAGES############################
###############################################################
import sys
import os
parent_directory = '/home/v/vasiliou/Project5/sources/TenPy2' #replace with location of your TenPy package
sys.path.append(parent_directory)
import numpy as np
import scipy.special
from models import spin_chain as mod
from mps.mps import iMPS
from algorithms import DMRG
from algorithms.linalg import np_conserved as npc
from algorithms.linalg import npc_helper
from tools.string import to_mathematica_lists
import matplotlib.pyplot as plt

import time
import cProfile
import warnings 
warnings.simplefilter(action='ignore',category=FutureWarning)#supress futurewarning messages

###############################################################
######################INITIALIZING#############################
###############################################################
g = 0.5
model_par = {
    'L': 2, #local hilbert space dim
	'S': 0.5, #spin one half
	'hx': -2 * g, #U(1) breaking field
	#'hz': 0.,
	'Jz': -4.,		
	'conserve_Z2': False, #we're interested in both Z2 conserving and breaking regimes
	'magnetization': (0, 1),#avg magnetization ???
	'fracture_mpo': False,
	'verbose': 0,
}

sim_par = {
	'CHI_LIST':{0:24},
	'VERBOSE': True,
	'N_STEPS': 10, 
	'STARTING_ENV_FROM_PSI': 1,#!
	'UPDATE_ENV': 1,
	'MAX_ERROR_E' : 10**(-14),
	'MAX_ERROR_S' : 1*10**(-3),
	'MIN_STEPS' : 30,
	'MAX_STEPS' : 800,
	'LANCZOS_PAR' : {'N_min': 2, 'N_max': 20, 'p_tol': 5*10**(-10), 'p_tol_to_trunc': 1/10.},
}
M = mod.spin_chain_model(model_par)
initial_state = np.array( [M.up, M.up])		#Ensures the ground state i get in FM region is gonna be up up.
M = mod.spin_chain_model(model_par)
psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve = M)
psi.convert_to_form('C')#?
Zs = []
Xs = []
Xis = []
Ss = []
Es = []
Es_exact = []

def runDMRG():
	sim_par['SVD_MAX'] = 16
	print('CHI lIST',sim_par['CHI_LIST'])
	dropped, Estat, Sstat, RP, LP = DMRG.ground_state(psi,M,sim_par)
	psi.canonical_form2()
	Sz = psi.site_expectation_value(M.Sz)
	Sx = psi.site_expectation_value(M.Sx)
	S = psi.entanglement_entropy()
	E0 = np.mean(psi.bond_expectation_value(M.H))
	Xi = psi.correlation_length()
	sim_par['RP'] = RP
	sim_par['LP'] = LP
	Zs.append(np.mean(Sz))
	Xs.append(np.mean(Sx))
	Xis.append(Xi)
	Ss.append(np.mean(S))
	Es.append(E0)
def TFI_groundstateenergy(g):
	return -(1.+g) * scipy.special.ellipe(4.*g/(1.+g)**2) * 2. / np.pi

mode = 'order params'
if __name__ == "__main__":
	if mode == 'order params':
		gs =  np.arange(0.5, 1.5, 0.04)
		for g in gs:
			model_par['hx'] = -2*g
			M = mod.spin_chain_model(model_par)
			runDMRG()
		plt.plot(gs,Xis,'.')
		plt.xlabel('$g/J$')
		plt.ylabel('$\\xi$')
		plt.savefig('corr_length.png')
		#plt.plot(gs, Zs, '.-',c='red',label='$\\langle Z\\rangle$')
		#plt.plot(gs, Ss, '.-',c='blue',label='S')
		#plt.plot(gs,Xs,'.-',c='green',label='$\\langle X\\rangle$')
		#plt.plot(gs,np.exp(1/np.array(Xis)),c='green')
		#plt.legend()
		#plt.xlabel('$g/J$')
		#plt.savefig('ord_params.png')
	if mode == 'energy':
		gs =  np.arange(0.5, 1.5, 0.05)
		for g in gs:
			model_par['hx'] = -2*g
			M = mod.spin_chain_model(model_par)
			Es_exact.append(TFI_groundstateenergy(g))
			runDMRG()
		plt.plot(gs,Es,'.',c='red',label='DMRG')
		plt.plot(gs,Es_exact,'--',alpha=0.5,c='blue',label='exact')
		plt.xlabel('$g/J$')
		plt.ylabel('$E$ per site')
		plt.legend()
		plt.savefig('energy.png')







				
				
		
