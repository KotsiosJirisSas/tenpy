#!/mnt/users/dperkovic/myenv/bin/python
#source /mnt/users/dperkovic/myenv/bin/activate
"""Code for producing DMRG graph"""
import sys
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 

import numpy as np
from tenpy.models import multilayer_qh_DP_final as mod


import pickle




np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)









def make_iMPO(model_par):
    print("Start model in Old Tenpy",".."*10)

    ###
    # so graph is problematic on the new run, lets load graph and see if there is problem
    ######
    M = mod.QH_model(model_par,'')
    print("Old code finished producing MPO graph",".."*10)

    
      
    G=M.MPOgraph
    
   
    return G








Lx=round(float(sys.argv[1]),3)

V0=round(float(sys.argv[2]),3)


######
# the potential fed into is 2pi tanh (qd/2)/q.
# For us it is tanh(qd)/qs
# so d=3 corresponds to our case d=1.5
d=round(float(sys.argv[3]),3)
xi=2.8
xi=4
#
"""
TO INTRODUCE A MIX OF LANDAU LEVELS IN BANDS THE NOTATION GOES THIS WAY

layers=[ ('BAND1', [ {0: c1  }, {1: c2  } ]),('BAND2', [ {0: c1  }, {1: c2  } ])]
BAND1 is the name of the band. Band consists of c1|0>+c2|1> where 0,1 are 0LL and 1LL
"""


### in terms of interaction
mus=[0,0]


#USE ABSOLUTE VALUE SINCE IT SHOULD NOT MATTER WHATSOEVER...
pref=1.0


#this is easier!
vecs=[('down_K-',0),('up_K-',0)]


#vecs=[('up_K-',0)]

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



root_config_=np.array([0,0,1,0,0,1,0,0])

Veps = 1e-5

#isotropic Nlayer interaction
V = { 'eps':Veps, 'xiK':2.5}
#V = { 'eps':Veps, 'xiK':5}
print(layers)

#magnetic length in nm


#SO THIS IS DOUBLE GATED COULUMB
#BUT WE CAN ALSO TRY GAUSSIAN ONLY

if d<10:
    V['coulomb_blgated']={}
    for i,layer in enumerate(species):
        for j,layer2 in enumerate(species):
                print(layer)
                if i<=j:
                    #key=(layer[0],layer2[0])
                    key=(layer,layer2)
                    if i==j:
                        V['coulomb_blgated'][key]={'v':1., 'z': 0,'d':d,'epsA':1,' betaA':1,'V0':V0/2} #NOTE: IMPORTANT; makes it consistent with previous calculations
                    else:
                        V['coulomb_blgated'][key]={'v':2., 'z': 0,'d':d,'epsA':1,' betaA':1,'V0':V0} #because it gets overwritten cool cool.
else:
    V['GaussianCoulomb']={}
    for i,layer in enumerate(species):
        for j,layer2 in enumerate(species):
                print(layer)
                if i<=j:
                    #key=(layer[0],layer2[0])
                    key=(layer,layer2)
                    if i==j:
                        V['GaussianCoulomb'][key]={'v':1.,'xi':xi,  'V0':V0/2} #NOTE: IMPORTANT; makes it consisten with previous calculations
                    else:
                        V['GaussianCoulomb'][key]={'v':2.,'xi':xi, 'V0':V0} #because it gets overwritten cool cool.
print(V)
#quit()


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
            'cons_C': 'each', #or total
            'Lx': Lx
            }


model_par['layers']=layers

model_par['pref']=pref

#model_par['species']=species
model_par['mu']=-np.array(mus)

print(model_par)



path='/mnt/users/dperkovic/DMRG_modified/data_load/Landau_level_2_bands_V0_perturbation_each/'
if d<10:
    name='each_Landau_level_2_bands_perturbed_V0='+str(V0)+'_d='+str(d)+"_Lx="+str(Lx)+'_coulumb_double_gated'
else:    
    path='/mnt/users/dperkovic/DMRG_modified/data_load/Landau_level_2_bands_V0_perturbation_each/Gaussian_Coulomb/'   
    name='each_Landau_level_2_bands_perturbed_V0='+str(V0)+'_xi='+str(xi)+"_Lx="+str(Lx)+'_Gaussian_Coulomb' 

name_save=path+name+'.pkl'


graph=make_iMPO(model_par)
import pickle


data={'model_par':model_par,'graph':graph}
with open(name_save,'wb') as f:
    pickle.dump(data,f)



