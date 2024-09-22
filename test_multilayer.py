"""Runs tests of the QH interaction creation code"""

import numpy as np
from tenpy.linalg import np_conserved as npc
from tenpy.models import multilayer_qh as mod
from tenpy.models.multilayer_qh import interlace_zero
#from tenpy.algorithms import simulation
import itertools
#from tenpy.mps.mps import iMPS
np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

NLL = 2; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)

#Potential data for (single/multilayered) model Laughlin
V = { 'eps':Veps, 'xiK':2., 'rV(r)': { ('L','L'): {'rV': rvr} }, 'coulomb': { ('L','L'):  {'v':-1., 'xi': xi}} }

root_config = [0]*NLL

model_par = {
	'verbose': 2,
	'layers': [ ('L', l) for l in range(NLL) ],
	#'layers':[ ('L', 1)],
	'Lx': 12.,
	'Vs': V,
	'cons_C': 'total',
	'cons_K': True,
	'root_config': root_config,
	'exp_approx': 'slycot',
}

print ("-"*10 + "Comparing analytic V(q) Yukawa and V(r) Yukawa" +"-"*10)
M = mod.QH_model(model_par)
print ("Vmk norm: (all should be zero)")
for a in range(NLL):
	for b in range(NLL):
		for c in range(NLL):
			for d in range(NLL):
				print( np.linalg.norm(M.Vmk['L']['L'][a, b, c, d]))



NLL = 3; Veps = 1e-4
#Potential data for (single/multilayered) model Laughlin
V = { 'eps':Veps, 'xiK':2., 'TK': { ('L','L'):[-0.3,1., 0.05] },'TK_vq': { ('L','L'):[0.3,-1., -0.05] } }

root_config = [0]*NLL

model_par = {
	'verbose': 2,
	'layers': [ ('L', l) for l in range(NLL) ],
	#'layers':[ ('L', 1)],
	'Lx': 12.,
	'Vs': V,
	'cons_C': 'total',
	'cons_K': True,
	'root_config': root_config,
	'exp_approx': 'slycot',
}

print ("-"*10 + "Comparing analytic TK and TK from V(q)""" +"-"*10)
M = mod.QH_model(model_par)
print ("Vmk norm: (all should be zero)")
for a in range(NLL):
	for b in range(NLL):
		for c in range(NLL):
			for d in range(NLL):
				print (np.linalg.norm(M.Vmk['L']['L'][a, b, c, d]))



