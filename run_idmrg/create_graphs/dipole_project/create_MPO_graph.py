#!/mnt/users/dperkovic/myenv/bin/python
#source /mnt/users/dperkovic/myenv/bin/activate
"""Code for producing DMRG graph"""
import sys
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 

#sys.path.append('/mnt/users/dperkovic/quantum_hall_dmrg/tenpy') 

import numpy as np
from tenpy.models import multilayer_qh_DP_final as mod

#from tenpy.networks.site import QH_MultilayerFermionSite_final_2
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy/NEW_COMMITS') 


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

Delta1=float(sys.argv[2])


mu2=-0.25

#
"""
TO INTRODUCE A MIX OF LANDAU LEVELS IN BANDS THE NOTATION GOES THIS WAY

layers=[ ('BAND1', [ {0: c1  }, {1: c2  } ]),('BAND2', [ {0: c1  }, {1: c2  } ])]
BAND1 is the name of the band. Band consists of c1|0>+c2|1> where 0,1 are 0LL and 1LL
"""
from LKBP_Hamiltonian.LKBP_hamiltonian import calculate_relevant_eigenvectors



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


#CROSSING EXACTLY AT DELTA=8.3
mus,vecs,pref=calculate_relevant_eigenvectors(Delta1)

mus=mus/interaction

#USE ABSOLUTE VALUE SINCE IT SHOULD NOT MATTER WHATSOEVER...
pref=np.abs(pref)



#this is easier!
vecs=[('down_K-',1),('up_K-',1),('new',0),('down_K-',0)]

#vecs=[('1LL',1),('1LL',0),]
# SO I THINK WE SHOULD REWRITE CODE SO THAT 
# For graphene code works like this
#layers=[ ('band1', [ { 0: 1/np.sqrt(2), 1: 0.3},{2: -1/np.sqrt(2)} ]), ('band1', [ { 1: 1/np.sqrt(2)},{3: -1/np.sqrt(2)} ])]
#Each dictionary represents Landau Level in orbital A_i
#since both states are called the same, it is the same species so it will calculate
#F_NM as in the spinfull case. However, it will calculate it only between the same orbitals A_i
#as other orbitals are deemed to have 0 overlaps!
#you just need to make sure that parity of states is defined consistently!
#its got some sector issues tho!

layers=vecs

species=[]
for name in vecs:
     species.append(name[0])
species=list(set(species))

#eigenvals,eigenvec=calculate_relevant_eigenvectors(Delta1)
#cut eigenvec to contain only 4 LLs

#layers=[ ('band1',0), ('band1', 1)]

#,('down2', 0),('up',1)]#,('empty',0)]
xi=2
#OK, SO THIS SORTS THAT ONE OUT FOR THE LL MIXING!
#layers=[ ('single_layer', [ {0: 0.4  }, {0: 0.4  },{1:np.sqrt(1-0.4**2)  }, {2: 0.04**2} ])]

#layers=[ ('down', 1),('down',0)]

root_config_=np.array([1,1,1,1, 1,1,1,0, 1,1,1,1, 1,1,0,1])
#root_config_=np.array([1,1,1 ,0,1,1, 1,1,1, 0,1,1])


#root_config_=np.array([1,1, 0, 0,0,0,0,0, 0, 0,0,0])
#root_config_=np.array([0,1,1,0])
Veps = 1e-5

#isotropic Nlayer interaction
V = { 'eps':Veps, 'xiK':2.5}
print(layers)

#magnetic length in nm
#25*sqrt(1/B)
#B=12


#SO THIS IS DOUBLE GATED COULUMB
#BUT WE CAN ALSO TRY GAUSSIAN ONLY
"""
V['coulomb_blgated']={}
for i,layer in enumerate(species):
    for j,layer2 in enumerate(species):
            print(layer)
            if i<=j:
                #key=(layer[0],layer2[0])
                key=(layer,layer2)
                if i==j:
                    V['coulomb_blgated'][key]={'v':1., 'z': 0,'d':d,'epsA':1,' betaA':1}
                else:
                    V['coulomb_blgated'][key]={'v':2., 'z': 0,'d':d,'epsA':1,' betaA':1} #because it gets overwritten cool cool.

print(V)
#quit()

"""
V['GaussianCoulomb']={}
for i,layer in enumerate(species):
    for j,layer2 in enumerate(species):
            print(layer)
            #if i<=j:
            #key=(layer[0],layer2[0])
            key=(layer,layer2)
            #if i==j:
            V['GaussianCoulomb'][key]={'v':1., 'xi': xi}
            #else:
            #    V['GaussianCoulomb'][key]={'v':2., 'xi': xi} #because it gets overwritten cool cool.
print(V)

#NEED TO FIX THIS DOUBLE COUNTING OF INTERACTIONS
#isotropic Nlayer interaction
#V = { 'eps':Veps, 'xiK':2}
"""
V['haldane']={}
for layer in layers:
   /mnt/users/dperkovic/DMRG_modified/data_load/nu=2_5_332
    key=(layer[0],layer[0])
    print(key)
    V['haldane'][key]=[0,1.]
V['haldane'][('up','down')]=[0,0.]
"""

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
            'Lx': Lx
            }


model_par['layers']=layers

model_par['pref']=pref

#model_par['species']=species
model_par['mu']=-mus

print(model_par)

#name='haldane_1_3_1_3_graph_Lx='+str(Lx)+'xi=5'

path='/mnt/users/dperkovic/DMRG_modified/data_load/haldane_1_3_1_3/'
path='/mnt/users/dperkovic/DMRG_modified/data_load/4_bands_simple/'
#name='Haldane_1_3_1_3_no_interlayer_interaction'

name='Hybridized_1LL_mix='+str(list(mus))+"_Lx="+str(Lx)+'_xi='+str(xi)

print_out=str([float(x) if isinstance(x, np.floating) else x for x in mus]).replace(' ', '')

#this is one without np.conj on the pref before shooting it into make_IMPO
#name='incorportated_scaled_interaction_4_bands_Delta='+str(Delta1)+'_mu='+print_out+"_Lx="+str(Lx)+'_xi='+str(xi)
name='CONJ_incorportated_scaled_interaction_4_bands_Delta='+str(Delta1)+'_mu='+print_out+"_Lx="+str(Lx)+'_xi='+str(xi)
#name='mid_regime_Three_bands_With_prefactor_each_conservation_Delta='+str(Delta1)+'_mu='+print_out+"_Lx="+str(Lx)+'_xi='+str(xi)
name_save=path+name+'.pkl'


graph=make_iMPO(model_par)
import pickle


data={'model_par':model_par,'graph':graph}
with open(name_save,'wb') as f:
    pickle.dump(data,f)



