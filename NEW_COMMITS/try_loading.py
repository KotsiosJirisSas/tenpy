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
qflat=[]
for i in range(len(qflat2)):
    qflat.append(qflat2[i][0])
qflat=np.array(qflat)
#print(len(qflat))
#print(qflat)

#quit()

#print(len(Ss))
#print(len(Ss[2]))
#print(Bflat[2].shape)
#quit()
root_config_ = np.array([0,1,0])
root_config_ = root_config_.reshape(3,1)
spin=QH_MultilayerFermionSite_2(N=1,root_config=root_config_,conserve='N')
#spin=QH_MultilayerFermionSite(N=1)
L=3
sites = [spin] * L
#broj=1
#pstate=["empty", "full","empty"]*broj

Ss=[Ss[2],Ss[0],Ss[1]]
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
print('loaded')
quit()
#quit()

#Bflat= [np.random.rand(4, 4) for _ in range(3)]
#print(Bflat)
d=2
Bflat = [np.random.rand( d, d, 2) for _ in range(3)]
#print(Bflat)
SVs=[np.random.rand(4) for _ in range(3)]
mps=MPS.from_Bflat(sites,Bflat,SVs=None, bc='infinite',legL=left_leg)

print(mps)