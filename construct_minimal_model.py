"""Initialization of sites, MPS and MPO."""

from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO

spin = SpinHalfSite(conserve="Sz")

N = 6  # number of sites
sites = [spin] * N  # repeat entry of list N times
pstate = ["up", "down"] * (N // 2)  # Neel state
psi = MPS.from_product_state(sites, pstate, bc="finite")
print("<Sz> =", psi.expectation_value("Sz"))
# <Sz> = [ 0.5 -0.5  0.5 -0.5]
print("<Sp_i Sm_j> =", psi.correlation_function("Sp", "Sm"), sep="\n")
# <Sp_i Sm_j> =
# [[1. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0.]]

# define an MPO
Id, Sp, Sm, Sz = spin.Id, spin.Sp, spin.Sm, spin.Sz
J, Delta, hz = 1., 1., 0.2
W_bulk = [[Id, Sp, Sm, Sz, -hz * Sz], 
          [None, None, None, None, 0.5 * J * Sm],
    [None, None, None, None, 0.5 * J * Sp], 
    [None, None, None, None, J * Delta * Sz],
          [None, None, None, None, Id]]


#W_bulk = [[ Id,Id, None], 
#          [ None, None, None],
#    		[ None, None, None]]
#print(W_bulk[0])
W_first = [W_bulk[0]]  # first row
W_last = [[row[-1]] for row in W_bulk]  # last column
Ws = [W_first] + [W_bulk] * (N - 2) + [W_last]

#Ws =  [W_bulk] * (N ) 
#print(Sp)

#print(len(Ws))
H = MPO.from_grids([spin] * N, Ws, bc='finite', IdL=0, IdR=-1)

from tenpy.models.example_chain import Simplest_example_Chain

from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain

lattice=Chain(N,spin, bc="periodic", bc_MPS="finite")
model=MPOModel(lattice, H)
#print('done')
#quit()

#model = Simplest_example_Chain(H,L=N)
#sites = model.lat.mps_sites()
from tenpy.algorithms import dmrg
psi = MPS.from_product_state(sites, ['down'] * N, "finite")
dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True}

info = dmrg.run(psi, model, dmrg_params)
print("E =", info['E'])
print("DONE")
quit()
# E = -1.342864022725017
print("max. bond dimension =", max(psi.chi))
# max. bond dimension = 56
print("corr. length =", psi.correlation_length())
# corr. length = 4.915809146764157
print("FINISHED")
