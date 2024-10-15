import sys
import os
sys.path.append('/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy') 
from tenpy.networks.mps import MPS
from tenpy.models.QH_new_model import QHChain
from tenpy.algorithms import dmrg

N=2
import numpy as np


root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
model_params={"L": N, "bc_MPS": "infinite"}
#QHModel(model_params,  lattice,H_MPO)
H=[]
model = QHChain(H,L=N)
sites = model.lat.mps_sites()
psi = MPS.from_product_state(sites, ['empty'] * N, "infinite")
dmrg_params = {"trunc_params": {"chi_max": 100, "svd_min": 1.e-10}, "mixer": True}
info = dmrg.run(psi, model, dmrg_params)
print("E =", info['E'])
# E = -1.342864022725017
print("max. bond dimension =", max(psi.chi))
# max. bond dimension = 56
print("corr. length =", psi.correlation_length())