"""Call of (finite) DMRG."""

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
from tenpy.networks.site import QH_MultilayerFermionSite_2

np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

NLL = 1; Veps = 1e-4
xi = 1
d = 0
def rvr(r):
	return np.exp(-r/xi)

#Potential data for (single/multilayered) model Laughlin
V = { 'eps':Veps, 'xiK':2., 'rV(r)': { ('L','L'): {'rV': rvr} }, 'coulomb': { ('L','L'):  {'v':-1., 'xi': xi}} }

root_config = [0]*NLL


N=1
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
	'cons_K': True,
	'root_config': root_config,
	'exp_approx': 'slycot',
}

print ("-"*10 + "Comparing analytic V(q) Yukawa and V(r) Yukawa" +"-"*10)
print("START MODEL")
M = mod.QH_model(model_par)
print('OLD CODE FINISHED')


H_mpo=M.H_mpo
#print(len(M.MPOgraph[0]))
#quit()

#print(len(H_mpo))
#print(len(H_mpo[0]))
#print(len(H_mpo[0][0]))



#quit()
#quit()
#quit()
model_params={"L": N, "bc_MPS": "infinite", 'site':None, 'bc':'finite'}
#NEW CODE STARTS HERE







# For nu = 1/3,   we will use  [ ('L', LL)], with LL specified below.

np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)
#########################
Lx = 14;			# circumference
LL = 0;			# which Landau level to put in
mixing_chi = 450; #Bond dimension in initial sweeps
chi = 450;		#Bond dimension of MPS
xi = 6;			# The Gaussian falloff for the Coulomb potential
Veps = 1e-4		# how accurate to approximate the MPO


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



G_og=[{('a_', 0, 3): {('a_', 0, 4): [('StrOp', 1.0)], ('a', 0, 3, 'A', 0, 0): [('AOp', 1.0)], ('a', 0, 3, 'A', 0, 1): [('AOp', 1.0)]}, ('_a', 0, 3): {('_a', 0, 2): [('StrOp', 1.0)]}, ('A_', 0, 6): {}, ('a', 0, 3, 'A', 0, 1): {('a', 0, 3, 'A', 0, 1): [('Id', 0.34475573503532636)], ('_a', 0, 2): [('AOp', -0.019830660130506394)]}, ('_a', 0, 2): {('_a', 0, 1): [('StrOp', 1.0)]}, ('A', 0, 1, 'a', 0, 0): {('_A', 0, 0): [('aOp', 0.13832145119393363)], ('A', 0, 1, 'a', 0, 0): [('Id', 0.8005881679881267)]}, ('A', 0, 2, 'a', 0, 1): {('_A', 0, 1): [('aOp', -0.01359559860484505)], ('A', 0, 2, 'a', 0, 2): [('Id', 0.21324533869805767)], ('A', 0, 2, 'a', 0, 1): [('Id', 0.3425333055817517)]}, ('_A', 0, 1): {('_A', 0, 0): [('StrOp', 1.0)]}, ('a', 0, 2, 'A', 0, 2): {('_a', 0, 1): [('AOp', 0.013062430119161455)], ('a', 0, 2, 'A', 0, 2): [('Id', 0.3425333055817541)], ('a', 0, 2, 'A', 0, 1): [('Id', -0.21324533869805679)]}, ('_a', 0, 4): {('_a', 0, 3): [('StrOp', 1.0)]}, ('_A', 0, 0): {'F': [('AOp', 1.0)]}, ('a', 0, 3, 'A', 0, 0): {('a', 0, 3, 'A', 0, 0): [('Id', 0.4619497352374192)], ('_a', 0, 2): [('AOp', 0.025248692719033152)]}, ('a', 0, 4, 'A', 0, 1): {('a', 0, 4, 'A', 0, 1): [('Id', 0.33871110444159785)], ('_a', 0, 3): [('AOp', 0.006402357895473034)], ('a', 0, 4, 'A', 0, 0): [('Id', -0.040305948895300314)]}, ('a', 0, 5, 'A', 0, 0): {('a', 0, 5, 'A', 0, 0): [('Id', 0.5138512188414278)], ('_a', 0, 4): [('AOp', 0.00011023908392563828)]}, 'F': {'F': [('Id', 1.0)]}, ('A', 0, 0, 'a', 0, 0): {('A', 0, 0, 'a', 0, 0): [('Id', 0.8337251361291302)], ('A', 0, 0, 'a', 0, 1): [('Id', 0.09379746220264774)], 'F': [('nOp', -0.2660018610388971)]}, ('A', 0, 1, 'a', 0, 3): {('A', 0, 1, 'a', 0, 4): [('Id', 0.23708339098378298)], ('_A', 0, 0): [('aOp', -0.04380586818225882)], ('A', 0, 1, 'a', 0, 3): [('Id', 0.38172630104765676)]}, ('_A', 0, 2): {('_A', 0, 1): [('StrOp', 1.0)]}, ('a_', 0, 2): {('a_', 0, 3): [('StrOp', 1.0)], ('a', 0, 2, 'A', 0, 0): [('AOp', 1.0)], ('a', 0, 2, 'A', 0, 1): [('AOp', 1.4142135623730951)]}, ('A_', 0, 5): {('A', 0, 5, 'a', 0, 0): [('aOp', 1.0)], ('A_', 0, 6): [('StrOp', 1.0)]}, ('_A', 0, 5): {('_A', 0, 4): [('StrOp', 1.0)]}, ('a', 0, 1, 'A', 0, 4): {('a', 0, 1, 'A', 0, 4): [('Id', 0.38172630104763583)], ('_a', 0, 0): [('AOp', 0.030390807643815607)], ('a', 0, 1, 'A', 0, 3): [('Id', -0.23708339098377437)]}, ('a', 0, 1, 'A', 0, 1): {('_a', 0, 0): [('AOp', 0.0061305932314431565)], ('a', 0, 1, 'A', 0, 1): [('Id', 0.7630762915951965)], ('a', 0, 1, 'A', 0, 2): [('Id', 0.14797043763385084)]}, ('a', 0, 4, 'A', 0, 0): {('a', 0, 4, 'A', 0, 1): [('Id', 0.040305948895300314)], ('_a', 0, 3): [('AOp', 0.0006454127172432714)], ('a', 0, 4, 'A', 0, 0): [('Id', 0.33871110444159785)]}, ('_A', 0, 4): {('_A', 0, 3): [('StrOp', 1.0)]}, ('A_', 0, 4): {('A_', 0, 5): [('StrOp', 1.0)], ('A', 0, 4, 'a', 0, 0): [('aOp', 1.4142135623730951)]}, ('A', 0, 0, 'a', 0, 1): {('A', 0, 0, 'a', 0, 0): [('Id', -0.09379746220264774)], ('A', 0, 0, 'a', 0, 1): [('Id', 0.8337251361291302)], 'F': [('nOp', 0.05253891639028042)]}, ('A', 0, 1, 'a', 0, 2): {('_A', 0, 0): [('aOp', -0.01009450230953929)], ('A', 0, 1, 'a', 0, 1): [('Id', -0.14797043763386378)], ('A', 0, 1, 'a', 0, 2): [('Id', 0.7630762915952091)]}, ('a', 0, 2, 'A', 0, 0): {('_a', 0, 1): [('AOp', 0.042544852049328626)], ('a', 0, 2, 'A', 0, 0): [('Id', 0.6410408600755128)]}, ('A', 0, 0, 'a', 0, 5): {('A', 0, 0, 'a', 0, 5): [('Id', 0.3495781422729576)], 'F': [('nOp', -0.23103170492184383)]}, ('A', 0, 2, 'a', 0, 2): {('_A', 0, 1): [('aOp', 0.013062430119161288)], ('A', 0, 2, 'a', 0, 2): [('Id', 0.3425333055817517)], ('A', 0, 2, 'a', 0, 1): [('Id', -0.21324533869805767)]}, ('A', 0, 3, 'a', 0, 1): {('A', 0, 3, 'a', 0, 1): [('Id', 0.34475573503532886)], ('_A', 0, 2): [('aOp', -0.019830660130507678)]}, ('A', 0, 4, 'a', 0, 0): {('A', 0, 4, 'a', 0, 1): [('Id', 0.040305948895300446)], ('_A', 0, 3): [('aOp', 0.0006454127172432719)], ('A', 0, 4, 'a', 0, 0): [('Id', 0.3387111044415981)]}, ('A', 0, 0, 'a', 0, 4): {('A', 0, 0, 'a', 0, 4): [('Id', 0.2943158206367369)], ('A', 0, 0, 'a', 0, 3): [('Id', -0.36314516089054333)], 'F': [('nOp', -0.026162348887412326)]}, ('a', 0, 1, 'A', 0, 0): {('_a', 0, 0): [('AOp', 0.1383214511939045)], ('a', 0, 1, 'A', 0, 0): [('Id', 0.8005881679881443)]}, ('a', 0, 2, 'A', 0, 1): {('_a', 0, 1): [('AOp', -0.013595598604845185)], ('a', 0, 2, 'A', 0, 2): [('Id', 0.21324533869805679)], ('a', 0, 2, 'A', 0, 1): [('Id', 0.3425333055817541)]}, ('a_', 0, 1): {('a', 0, 1, 'A', 0, 1): [('AOp', 1.4142135623730951)], ('a', 0, 1, 'A', 0, 0): [('AOp', 1.0)], ('a', 0, 1, 'A', 0, 3): [('AOp', 1.4142135623730951)], ('a_', 0, 2): [('StrOp', 1.0)]}, ('A_', 0, 1): {('A', 0, 1, 'a', 0, 1): [('aOp', 1.4142135623730951)], ('A', 0, 1, 'a', 0, 0): [('aOp', 1.0)], ('A', 0, 1, 'a', 0, 3): [('aOp', 1.4142135623730951)], ('A_', 0, 2): [('StrOp', 1.0)]}, ('a_', 0, 6): {}, ('_a', 0, 5): {('_a', 0, 4): [('StrOp', 1.0)]}, ('_a', 0, 0): {'F': [('aOp', 1.0)]}, ('A', 0, 1, 'a', 0, 1): {('_A', 0, 0): [('aOp', 0.006130593231424408)], ('A', 0, 1, 'a', 0, 1): [('Id', 0.7630762915952091)], ('A', 0, 1, 'a', 0, 2): [('Id', 0.14797043763386378)]}, ('A', 0, 3, 'a', 0, 0): {('A', 0, 3, 'a', 0, 0): [('Id', 0.4619497352374154)], ('_A', 0, 2): [('aOp', 0.025248692719034436)]}, ('A', 0, 4, 'a', 0, 1): {('A', 0, 4, 'a', 0, 1): [('Id', 0.3387111044415981)], ('_A', 0, 3): [('aOp', 0.006402357895473009)], ('A', 0, 4, 'a', 0, 0): [('Id', -0.040305948895300446)]}, ('a_', 0, 5): {('a', 0, 5, 'A', 0, 0): [('AOp', 1.0)], ('a_', 0, 6): [('StrOp', 1.0)]}, ('A', 0, 5, 'a', 0, 0): {('A', 0, 5, 'a', 0, 0): [('Id', 0.5138512188414278)], ('_A', 0, 4): [('aOp', 0.00011023908392563828)]}, ('a', 0, 1, 'A', 0, 3): {('a', 0, 1, 'A', 0, 4): [('Id', 0.23708339098377437)], ('_a', 0, 0): [('AOp', -0.043805868182256937)], ('a', 0, 1, 'A', 0, 3): [('Id', 0.38172630104763583)]}, ('_A', 0, 3): {('_A', 0, 2): [('StrOp', 1.0)]}, ('A_', 0, 2): {('A_', 0, 3): [('StrOp', 1.0)], ('A', 0, 2, 'a', 0, 0): [('aOp', 1.0)], ('A', 0, 2, 'a', 0, 1): [('aOp', 1.4142135623730951)]}, ('A', 0, 0, 'a', 0, 3): {('A', 0, 0, 'a', 0, 4): [('Id', 0.36314516089054333)], ('A', 0, 0, 'a', 0, 3): [('Id', 0.2943158206367369)], 'F': [('nOp', -0.015888996698317146)]}, ('A', 0, 1, 'a', 0, 4): {('A', 0, 1, 'a', 0, 4): [('Id', 0.38172630104765676)], ('_A', 0, 0): [('aOp', 0.03039080764381981)], ('A', 0, 1, 'a', 0, 3): [('Id', -0.23708339098378298)]}, ('_a', 0, 1): {('_a', 0, 0): [('StrOp', 1.0)]}, 'R': {('A', 0, 0, 'a', 0, 5): [('nOp', 1.0)], ('A', 0, 0, 'a', 0, 0): [('nOp', 1.4142135623730951)], ('A_', 0, 1): [('AOp', 1.0)], ('A', 0, 0, 'a', 0, 3): [('nOp', 1.4142135623730951)], ('a_', 0, 1): [('aOp', 1.0)], 'R': [('Id', 1.0)], ('A', 0, 0, 'a', 0, 2): [('nOp', 1.0)]}, ('A', 0, 2, 'a', 0, 0): {('_A', 0, 1): [('aOp', 0.042544852049328445)], ('A', 0, 2, 'a', 0, 0): [('Id', 0.6410408600755139)]}, ('A_', 0, 3): {('A_', 0, 4): [('StrOp', 1.0)], ('A', 0, 3, 'a', 0, 0): [('aOp', 1.0)], ('A', 0, 3, 'a', 0, 1): [('aOp', 1.0)]}, ('a_', 0, 4): {('a_', 0, 5): [('StrOp', 1.0)], ('a', 0, 4, 'A', 0, 0): [('AOp', 1.4142135623730951)]}, ('a', 0, 1, 'A', 0, 2): {('_a', 0, 0): [('AOp', -0.010094502309538074)], ('a', 0, 1, 'A', 0, 1): [('Id', -0.14797043763385084)], ('a', 0, 1, 'A', 0, 2): [('Id', 0.7630762915951965)]}, ('A', 0, 0, 'a', 0, 2): {('A', 0, 0, 'a', 0, 2): [('Id', 0.8330402460772576)], 'F': [('nOp', 0.8454122673973181)]}}]


print("Start model in Old Tenpy",".."*10)
M = mod.QH_model(model_par)

#from tenpy.networks.site import FermionSite
root_config = np.array([0,1,0])
root_config = root_config.reshape(3,1)
hilber_space_single_site=QH_MultilayerFermionSite_2(N=1,root_config=root_config,conserve='N')
#hilber_space_single_site=QH_MultilayerFermionSite(N=1)
Id, StrOp, nOp,nOp_shift,AOp,aOp,invnOp= hilber_space_single_site.Id,hilber_space_single_site.StrOp, hilber_space_single_site.nOp, hilber_space_single_site.nOp_shift, hilber_space_single_site.AOp, hilber_space_single_site.aOp, hilber_space_single_site.invnOp
#




H_mpo=M.H_mpo
#print(len(H_mpo))
#quit()
#hilber_space_single_site=FermionSite(conserve='N')

#Id, nOp,nOp_shift,AOp,aOp= hilber_space_single_site.Id, hilber_space_single_site.N, hilber_space_single_site.dN, hilber_space_single_site.Cd, hilber_space_single_site.C
grids=H_mpo
#spin=QH_MultilayerFermionSite(N=1)
Ws=[]
for x in grids:
	copy_2=[]
	for z in x:
		copy=[]
		for m in z:
			#print(m)
			if m!=None:
				copy.append(m)
				
			else:
				op=hilber_space_single_site.Id*0
				copy.append(op)
		copy_2.append(copy)
	Ws.append(copy_2)
"""
J, Delta, hz = 1., 1., 0.2

W_bulk = [[Id, nOp, aOp,AOp , -hz *nOp], 
          [None, None, None, None, 0.532 * J * AOp],
    [None, J * Delta * AOp, None, None, 0.512 * J * aOp], 
    [None, None, None, None, J * Delta * nOp],
          [None, None, None, None, Id]]

W_bulk = [[Id, nOp, nOp,nOp , -hz *AOp], 
          [None, None, None, None, 0.532 * J * nOp],
    [None, J * Delta * nOp, None, None, 0.512 * J * nOp], 
    [None, None, None, None, J * Delta * nOp],
          [None, None, None, None, Id]]

W_bulk = [[Id, 0*Id, 0*Id ], 
          [1.01*aOp, 0*Id, 0*Id],
    [0*Id, AOp, Id] ]





Ws =  [W_bulk] * (N ) 

"""
#hilber_space_single_site=QH_MultilayerFermionSite(N=1)
#DO WE NEED H_bonds in chain?
#define_chain=Chain(N,hilber_space_single_site)
#W_first = [H_mpo[0][0]]  # first row
#W_last = [[row[-1]] for row in H_mpo[-1]]  # last column
#W_first = [W_bulk[0],W_bulk[1]]  # first row
#W_last = [[row[-1]] for row in W_bulk]
#Ws = [W_first] + [W_bulk] * (N -2)  + [W_last]

#Ws =  [W_bulk] * (N ) 

#Ws = [W_first] + H_mpo[1:-1] + [W_last]

#H = MPO.from_grids([hilber_space_single_site] * N,Ws, bc='finite', IdL=0, IdR=-1)

#MANAGED TO DEFINE A CHAIN WITH (HOPEFULLY) CORRECT MATRICES

H = MPO.from_grids([hilber_space_single_site] * N, Ws, bc='infinite', IdL=0, IdR=-1)

print('done')
quit()
N=2

model_params={"L": N, "conserve": "N", "bc_MPS": "infinite"}
#QHModel(model_params,  lattice,H_MPO)

root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)


#print(hilber_space_single_site)
#print(H.sites)
#quit()
#model = QHChain(model_params,root_config_,H_MPO=H)
model = QHChain(H,L=2)
sites = model.lat.mps_sites()

psi = MPS.from_product_state(sites, ['empty'] * N, "infinite")
dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True}
info = dmrg.run(psi, model, dmrg_params)
print("E =", info['E'])
# E = -1.342864022725017
print("max. bond dimension =", max(psi.chi))
# max. bond dimension = 56
print("corr. length =", psi.correlation_length())
# corr. length = 4.915809146764157
print("FINISHED")
