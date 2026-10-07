#!/mnt/users/dperkovic/myenv/bin/python
#source /mnt/users/dperkovic/myenv/bin/activate
"""Code for producing DMRG graph"""
import sys
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 

#sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 

import numpy as np
from tenpy.models import multilayer_qh_DP_final as mod

#from tenpy.networks.site import QH_MultilayerFermionSite_final_2


#quit()
import subprocess



np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)





import pickle




def make_iMPO(model_par):
    print("Start model in Old Tenpy",".."*10)

    ###
    # so graph is problematic on the new run, lets load graph and see if there is problem
    ######
    M = mod.QH_model(model_par,'')
    print("Old code finished producing MPO graph",".."*10)

    
      
    G=M.MPOgraph
    
   
    return G








Lx=float(sys.argv[1])
mu=float(sys.argv[2])






#Delta1=8 #in meV
e=1.6*10**(-19)
epsilon0=8.85*10**(-12)

hbar=6.626*10**(-34)/2/np.pi
B=12
l_B=np.sqrt(hbar/(e*B))
d=50/(l_B*10**9)
print(l_B)
epsilonr=6.7
interaction=1000*e/(4*np.pi*epsilon0*epsilonr*l_B)#in meV
#OK so epsilonr=6.7 WILL BE GOOD GG!



#USE ABSOLUTE VALUE SINCE IT SHOULD NOT MATTER WHATSOEVER...
pref=0



#this is easier!
vecs=[('down',0),('up',0)]

layers=vecs

species=[]
for name in vecs:
     species.append(name[0])
species=list(set(species))


root_config_=np.array([0,0, 0,0, 1,1, 0,0, 0,0])

Veps = 1e-5

#isotropic Nlayer interaction
V = { 'eps':Veps, 'xiK':2.5}
print(layers)

#magnetic length in nm
#25*sqrt(1/B)
#B=12



V['haldane']={}
for i,layer in enumerate(species):
    for j,layer2 in enumerate(species):
            print(layer)
            #if i<=j:
            #key=(layer[0],layer2[0])
            key=(layer,layer2)
            if i==j:
                V['haldane'][key]=[1.0,1.0]
            else:
                V['haldane'][key]=[1.0,1.0]
            #else:
            #    V['GaussianCoulomb'][key]={'v':2., 'xi': xi} #because it gets overwritten cool cool.
print(V)



####
## OK, SO PROBLEM IS AT CONSERVATION AT LABELLING EACH BAND DIFFERENTLY
## BECAUSE GRAPH TREATS THEM THE SAME
########
#OK, SO EACH IS BASICALLY SPECIES CONSERVATION.....
# SO CANNOT REALLY DO THIS EACH BUSINESS. CAN JUST DO SPECIES BUSINESS!
model_par={ #for finite replace by periodic here
	'verbose': 2,
    'root_config': root_config_,
            'exp_approx': 'slycot', 
            'Vs': V, 
            'cons_K': True, 
            'cons_C': 'total', #or total
            'Lx': Lx,

            }


model_par['layers']=layers

model_par['pref']=pref

model_par['mu']=[0,mu]

print(model_par)

#name='haldane_1_3_1_3_graph_Lx='+str(Lx)+'xi=5'

path='/mnt/users/dperkovic/DMRG_modified/data_load/nu=2_5_332/'
#name='Haldane_1_3_1_3_no_interlayer_interaction'


#this is one without np.conj on the pref before shooting it into make_IMPO
#name='incorportated_scaled_interaction_4_bands_Delta='+str(Delta1)+'_mu='+print_out+"_Lx="+str(Lx)+'_xi='+str(xi)
name="332_total_N_conserve_parent_hamiltonian_Lx="+str(Lx)+'_mu='+str(mu)
#name='mid_regime_Three_bands_With_prefactor_each_conservation_Delta='+str(Delta1)+'_mu='+print_out+"_Lx="+str(Lx)+'_xi='+str(xi)
name_save=path+name+'.pkl'


graph=make_iMPO(model_par)
import pickle


data={'model_par':model_par,'graph':graph}
with open(name_save,'wb') as f:
    pickle.dump(data,f)



