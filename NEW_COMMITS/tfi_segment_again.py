import numpy as np
import scipy
import matplotlib.pyplot as plt
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import os
import sys

sys.path.append('/Users/domagojperkovic/Desktop/git_konstantinos_project/tenpy') 
#sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 
import tenpy
import tenpy.linalg.np_conserved as npc
from tenpy.algorithms import dmrg
from tenpy.networks.mps import MPS
from tenpy.models.xxz_chain import XXZChain
from tenpy.models.tf_ising import TFIChain
import os
import sys
sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 

tenpy.tools.misc.setup_logging(to_stdout="INFO")


from tenpy.models.lattice import TrivialLattice
from tenpy.models.model import MPOModel
from tenpy.networks.mpo import MPOEnvironment
from tenpy.networks.mps import MPSEnvironment

model_params = {
    'J': 1. , 'g': 1.5,
    'L': 2,
    'bc_MPS': 'infinite',
    'conserve': 'best',
}

M_i = TFIChain(model_params)

# first dmrg run for *infinite* lattice
psi0_i = MPS.from_lat_product_state(M_i.lat, [['up']])

dmrg_params = {
    'mixer': True,
    'max_E_err': 1.e-10,
    'trunc_params': {
        'chi_max': 300,
        'svd_min': 1.e-10,
    },
}

#print(psi0_i)
#x=psi0_i
#print(psi0_i.get_B(0)[0])
#quit()
eng0_i = dmrg.TwoSiteDMRGEngine(psi0_i, M_i, dmrg_params)
#quit()
E0_i, _ = eng0_i.run()
#print("rest")
#print(a)
#quit()

resume_psi0_i = eng0_i.get_resume_data(sequential_simulations=True)
#print(psi0_i)

#quit()
#print(resume_psi0_i)

#print(psi0_i.entanglement_entropy())


enlarge = 10  # this is a parameter: how large should the "segment" be?
# beware: if you have gapless excitations, this will induce a "finite-size" gap ~ 1/(enlarge*N_sites_per_unit_cell)

M_s = M_i.extract_segment(enlarge=10)
first, last = M_s.lat.segment_first_last


psi0_s = psi0_i.extract_segment(first, last)
init_env_data = eng0_i.env.get_initialization_data(first, last)


#print(M_i.H_MPO.__dict__)
#quit()
#quit()
#print(init_env_data['init_LP'][1])

#psi0_s.canonical_form_finite()

#print(psi0_s.segment_boundaries)
#quit()

psi_halfinf = psi0_s.copy()  # the oringinal MPS

S = psi0_s.get_SL(0)
proj = np.zeros(len(S), bool)
proj[np.argmax(S)] = True
B = psi_halfinf.get_B(0, form='B')
B.iproject(proj, 'vL')
psi_halfinf.set_B(0, B, form='B')
psi_halfinf.set_SL(0, np.ones(1, float))


#print(psi_halfinf.get_SL(1))
#quit()

#a,b=psi_halfinf.segment_boundaries

psi_halfinf.canonical_form_finite(cutoff=0.0)
#print(psi_halfinf.get_SL(1))
#quit()
psi_halfinf.test_sanity()
#print(psi_halfinf)

#print(psi_halfinf._B[0])
#quit()
print('START ZERO')
B = psi_halfinf.get_B(0, form='B')
#print(B)
#quit()
#print(psi_halfinf.segment_boundaries[0])
#quit()
a,b=psi_halfinf.segment_boundaries
#print(a,b)
#quit()
#quit()
init_env_data_halfinf = init_env_data.copy()
#print(psi_halfinf)
print('MPOOOO')
init_env_data_halfinf['init_LP'] = MPOEnvironment(psi0_i, M_i.H_MPO, psi0_i).init_LP(0, 0)#[4,:,4]
#quit()
#print(init_env_data_halfinf['init_LP'])
#quit()
from tenpy.linalg.np_conserved import Array
from tenpy.linalg.np_conserved import Array
from tenpy.linalg.charges import LegCharge
from tenpy.linalg.charges import ChargeInfo
#chargeinfo=ChargeInfo([1],['N'])
#print(init_env_data_halfinf['init_LP'])
#quit()
chargeinfo=init_env_data_halfinf['init_LP'].legs[0].chinfo
#print(chargeinfo)


legcharges=[]
#define 3 legs
qflat=[0]
legcharge=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
legcharges.append(legcharge)
qflat=[0,1,0]
legcharge=LegCharge.from_qflat(chargeinfo,qflat,qconj=-1).bunch()[1]

legcharges.append(legcharge)
qflat=[0]
legcharge=LegCharge.from_qflat(chargeinfo,qflat,qconj=-1).bunch()[1]
legcharges.append(legcharge)
#print(legcharges[1])

#quit()
#quit()

labels=['vR*', 'wR', 'vR']

data_flat=[1,0,0]
data_flat=np.reshape(data_flat,(1,3,1))

array_defined=Array.from_ndarray( data_flat,
                     legcharges,
                     dtype=np.float64,
                     qtotal=None,
                     cutoff=None,
                     labels=labels,
                     raise_wrong_sector=True,
                     warn_wrong_sector=True)

print(array_defined)
print(array_defined.qtotal)
quit()
#print(array_defined)
#quit()
init_env_data_halfinf['init_LP'] =array_defined#Array(legcharges, dtype=np.float64, qtotal=None, labels=labels)
print('defined')
#print(array_defined)
#quit()
#npc.array()shape=1,3,1
#quit()
#quit()
a=init_env_data_halfinf['init_LP']
#init_env_data_halfinf['init_LP'] = MPOEnvironment(psi0_i, M_i.H_MPO, psi0_i).init_LP(0, 0)
init_env_data_halfinf['age_LP'] = 0

#quit()

#print("START"*100)
#print(init_env_data_halfinf['age_LP'])
#print(init_env_data_halfinf['init_LP'][0])
#print(a[0])

#print("STOP"*100)
#quit()
#print()
#print("START"*100)
#print(init_env_data_halfinf['init_RP'])
a=init_env_data_halfinf['init_RP']
#print(init_env_data_halfinf['age_RP'])
#print("STOP"*100)



init_env_data_halfinf['init_RP'] = MPOEnvironment(psi0_i, M_i.H_MPO, psi0_i).init_RP( 19, 19)


#init_env_data_halfinf['age_RP']=0 
#print("START"*100)
#print(init_env_data_halfinf['age_RP'])
#print(init_env_data_halfinf['init_RP'][3])

#quit()
#print(a[0])

#print("STOP"*100)
"""
[0.      0.13332 0.14834 0.15192 0.15297 0.15331 0.15343 0.15347 0.15348 0.15349 0.15349 0.15349
 0.15349 0.15349 0.15349 0.15349 0.15349 0.15349 0.15349 0.15349 0.15349]
[0.94082 0.88667 0.87957 0.87797 0.87753 0.8774  0.87735 0.87734 0.87733 0.87733 0.87733 0.87733
 0.87733 0.87733 0.87733 0.87733 0.87733 0.87733 0.87733 0.87733]
"""

#quit()
#print(init_env_data_halfinf['age_RP'])
#print("STOP"*100)
#quit()
#init_env_data_halfinf['age_RP'] = 0
print("AAAAAAA"*100)
a,b=psi_halfinf.segment_boundaries
#print(a)
#print(b)
#quit()

#print(init_env_data_halfinf['init_LP'])
#quit()
#print('AAAAAAA')
#print(init_env_data_halfinf['init_RP'])
#quit()
eng_halfinf = dmrg.TwoSiteDMRGEngine(psi_halfinf, M_s, dmrg_params,
                                     resume_data={'init_env_data': init_env_data_halfinf})


print("enviroment works")
quit()
eng_halfinf.run()


#print(psi_halfinf.entanglement_entropy())
#print(psi_halfinf.expectation_value("Sigmaz"))
#quit()
model_params_defect = {
    'J': 1. , 'g': [0.] + [model_params['g']] * (psi_halfinf.L-1),
    'L': psi_halfinf.L,
    'bc_MPS': 'segment',
    'conserve': 'best',
}

M_s_defect = TFIChain(model_params_defect)
