"""Initialization of sites, MPS and MPO."""

from tenpy.networks.site import QH_MultilayerFermionSite_2
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
import numpy as np
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
spin =  QH_MultilayerFermionSite_2(root_config_,conserve='N')#SpinHalfSite(conserve="Sz")


#build_initial_state(size, states, filling, mode='random', seed=None)
N = 3  # number of sites
sites = [spin] * N  # repeat entry of list N times
#pstate = ["empty", "full"] * (N // 2)  # Neel state
pstate=["empty", "full","empty"]
psi = MPS.from_product_state(sites, pstate, bc="infinite")
import h5py
from tenpy.tools import hdf5_io

data = {"psi": psi,  # e.g. an MPS
        "parameters": {"L": 6, "g": 1.3}}

with h5py.File("file.h5", 'w') as f:
    hdf5_io.save_to_hdf5(f, data)



Length=psi.correlation_length()
print(Length)



filling=psi.correlation_function("AOp", "aOp")
filling=psi.expectation_value("nOp")
print(filling)

print('entanglement spectrum')
E_spec=psi.entanglement_spectrum()
print(E_spec)
#quit()
#root_config_ = np.array([0,1,0])
#spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
##root_config_ = root_config_.reshape(3,1)
#pstate = ['empty', 'full', 'empty']*N


#sites = [spin] * (N)  # repeat entry of list N times


#psi = MPS.from_product_state(sites, pstate, bc="infinite")
print('MPS constructed'+'..'*10)

#quit()
Id, Sp, Sm, Sz = spin.Id, spin.AOp, spin.aOp, spin.nOp
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
#W_first = [W_bulk[0]]  # first row
#W_last = [[row[-1]] for row in W_bulk]  # last column
#Ws = [W_first] + [W_bulk] * (N - 2) + [W_last]

Ws =  [W_bulk] * (N ) 
#print(Sp)

#print(len(Ws))
H = MPO.from_grids([spin]*N , Ws, bc='infinite', IdL=0, IdR=-1)


from tenpy.models.model import MPOModel
from tenpy.models.lattice import Chain

lattice=Chain(N,spin, bc="periodic", bc_MPS="infinite")
model=MPOModel(lattice, H)

print(lattice.bc)
#quit()
#print(spin.AOp)
#print('done')
#quit()

#model = Simplest_example_Chain(H,L=N)
#sites = model.lat.mps_sites()
from tenpy.algorithms import dmrg
#psi = MPS.from_product_state(sites, ['down'] * N, "finite")
dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True, "max_sweeps":10}
print('enter DMRG algorithm:')
#E0, psi  = dmrg.run(psi, model, dmrg_params)
engine = dmrg.TwoSiteDMRGEngine(psi, model, dmrg_params)  # (re)initialize DMRG environment with new model
# this uses the result from the previous DMRG as first initial guess
E0, psi = engine.run()




print("E =", E0)
print("DONE")




Length=psi.correlation_length()
print('correlation length:',Length)



filling=psi.expectation_value("nOp")

print('Filling:',filling)
quit()
# E = -1.342864022725017
print("max. bond dimension =", max(psi.chi))
# max. bond dimension = 56
print("corr. length =", psi.correlation_length())
# corr. length = 4.915809146764157
print("FINISHED")
