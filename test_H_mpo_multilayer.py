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
	'cons_K': True,
	'root_config': root_config,
	'exp_approx': 'slycot',
}

print ("-"*10 + "Comparing analytic V(q) Yukawa and V(r) Yukawa" +"-"*10)
print("START MODEL")
M = mod.QH_model(model_par)
print('OLD CODE FINISHED')


H_mpo=M.H_mpo
print(len(M.MPOgraph[0]))
quit()

print(len(H_mpo))
print(len(H_mpo[0]))
print(len(H_mpo[0][0]))



#quit()
#quit()
#quit()
model_params={"L": N, "bc_MPS": "infinite", 'site':None, 'bc':'finite'}
#NEW CODE STARTS HERE
root_config = np.array([0,1,0])
root_config = root_config.reshape(3,1)



from tenpy.networks.site import FermionSite

#hilber_space_single_site=QH_MultilayerFermionSite_2(N=1,root_config=root_config,conserve='N')
#hilber_space_single_site=QH_MultilayerFermionSite(N=1)
#Id, StrOp, nOp,nOp_shift,AOp,aOp,invnOp= hilber_space_single_site.Id,hilber_space_single_site.StrOp, hilber_space_single_site.nOp, hilber_space_single_site.nOp_shift, hilber_space_single_site.AOp, hilber_space_single_site.aOp, hilber_space_single_site.invnOp
#




hilber_space_single_site=FermionSite(conserve='N')

Id, nOp,nOp_shift,AOp,aOp= hilber_space_single_site.Id, hilber_space_single_site.N, hilber_space_single_site.dN, hilber_space_single_site.Cd, hilber_space_single_site.C

J, Delta, hz = 1., 1., 0.2
"""
copy=H_mpo.copy()
for i in range(len(H_mpo)):
	

	for j in range(len(H_mpo[0])):
	
		for k in range(len(H_mpo[0][0])):
			if H_mpo[i][j][k]==None:
				copy[i][j][k]=0*Id


print('aaaaa')
num=0

num_col=0
for j in range(len(H_mpo[0])):
	num_row=0
	for k in range(len(H_mpo[0][0])):
		if H_mpo[0][j][k]==None:
			num_row+=1
	#print(num_row)
	if num_row==len(H_mpo[0]):
		num+=1
	
print(num)

for j in range(len(H_mpo[0])):
	num_row=0
	for k in range(len(H_mpo[0][0])):
		if H_mpo[0][k][j]==None:
			num_row+=1
	#print(num_row)
	print(len(H_mpo[0]))
	if num_row==len(H_mpo[0]):
		num+=1
"""	
#print(num)
#quit()
#print(copy[0][0])

#quit()
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
 


#W_bulk=[[-0.6647744*Id, None, None],
#[None,None ,StrOp],
#[None, None, 0.2412*Id]]
#print(W_bulk[2])



#W_bulk = [[ Id,Id, 0*Id], 
#          [ None, None, None],
#    		[ None, None, None]]
Ws =  [W_bulk] * (N ) 


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
H = MPO.from_grids([hilber_space_single_site] * N, Ws, bc='finite', IdL=0, IdR=-1)
print("FINISHED")
