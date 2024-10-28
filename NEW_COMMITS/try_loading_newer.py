import pickle
import sys
sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 
from tenpy.networks.mps import MPS 
import numpy as np
from tenpy.networks.site import QH_MultilayerFermionSite
from tenpy.networks.site import QH_MultilayerFermionSite_2


############################################################
#######LOADING DATA FROM OLD TENPY TO INFINITE MPS##########
############################################################
name="qflat_QH_1_3-2"
#name="wavef_QH_1_3"
with open(name+'.pkl', 'rb') as f:
    loaded_xxxx = pickle.load(f, encoding='latin1')
print(loaded_xxxx.keys())


Bflat=loaded_xxxx['Bs']

Ss=loaded_xxxx['Ss']
qflat2=loaded_xxxx['qflat']
#quit()
print(qflat2)
qflat=[]
for i in range(len(qflat2)):
    qflat.append(qflat2[i][0])
qflat=np.array(qflat)
#print(qflat)
#quit()
#print(len(qflat))
#print(qflat)

#quit()
broj=2
Bflat=[Bflat[0],Bflat[1],Bflat[2]]*broj
#print(Bflat)
#print(len(Ss))
#print(len(Ss[2]))
#print(Bflat[2].shape)
#quit()
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
#spin=QH_MultilayerFermionSite(N=1)
L=3
sites = [spin] * L*broj
#broj=1
#pstate=["empty", "full","empty"]*broj

Ss=[Ss[2],Ss[0],Ss[1]]*broj
from tenpy.linalg.charges import LegCharge
from tenpy.linalg.charges import ChargeInfo
#chargeinfo=ChargeInfo([1],['N'])
chargeinfo=sites[0].leg.chinfo
#print(chargeinfo)
#quit()


left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
#a=left_leg.bunch()
#print(a)
#print(left_leg)
#quit()
#LegCharge(chargeinfo, slices, charges, qconj=1)
#psi=MPS.from_product_state(sites, pstate,bc='infinite')
#print(psi._B)
mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='infinite',legL=left_leg)
print("loaded original MPS")
#print(Bflat[0].shape)

#quit()
print("extracting semi-infinite chain of length 20")

mps2=mps.extract_segment(0, 17)
psi_halfinf = mps2.copy()  # the oringinal MPS

#EXTRACT A SEGMENT
S = mps2.get_SL(0)
proj = np.zeros(len(S), bool)
proj[np.argmax(S)] = True
B = psi_halfinf.get_B(0, form='B')
B.iproject(proj, 'vL')
psi_halfinf.set_B(0, B, form='B')
psi_halfinf.set_SL(0, np.ones(1, float))
psi_halfinf.canonical_form_finite()
psi_halfinf.test_sanity()
#print(mps2.L)

#haha=psi_halfinf.get_B(0)

#print(haha.shape)
#quit()

print(vars(psi_halfinf).keys())
print('Finished extracting half of MPS')
quit()

#initialize enviroment
init_env_data_halfinf = init_env_data.copy()
#init_env_data_halfinf['init_LP'] = None
init_env_data_halfinf['init_LP'] = MPOEnvironment(psi0_i, M_i.H_MPO, psi0_i).init_LP(0, 0)
init_env_data_halfinf['age_LP'] = 0
quit()
#quit()