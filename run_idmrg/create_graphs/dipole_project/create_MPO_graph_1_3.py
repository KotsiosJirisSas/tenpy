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




#
"""
TO INTRODUCE A MIX OF LANDAU LEVELS IN BANDS THE NOTATION GOES THIS WAY

layers=[ ('BAND1', [ {0: c1  }, {1: c2  } ]),('BAND2', [ {0: c1  }, {1: c2  } ])]
BAND1 is the name of the band. Band consists of c1|0>+c2|1> where 0,1 are 0LL and 1LL
"""

"""
def v_q_screened(q_values):
     

     (1+a*np.tanh(q_values**2*b)/q_values)
"""
from scipy.special import j0
def inverse_radial_ft_2d_sum(F_values, k_values, r_values):
    """
    Discrete inverse radial Fourier transform in 2D using a sum.
    
    Parameters
    ----------
    F_values : array
        Values F(k_i) on k grid.
    k_values : array
        Momentum grid points k_i (uniform spacing).
    r_values : array
        Radial positions r at which to evaluate f(r).
    
    Returns
    -------
    f_r : array
        Function values f(r) at each r.
    """
    dk = k_values[1] - k_values[0]  # assume uniform spacing
    f_r = []

    s = np.sum(F_values * j0(k_values * r_values) * k_values) * dk
 
    return s / (2*np.pi)

def rvr(r):
    return np.exp(-r)


###########################################
###########perform Fourier transform and then multiply via exp(-r^2/(2*xi))
##########################
def RPA_screened_interaction(r):
    q=np.linspace(0.0001,10,1000)
    a=6
    b=0.5
    pi=a*(np.tanh(b*q**2))
    V_q=1/(q+pi)
    V_r=inverse_radial_ft_2d_sum(V_q, q, r)
    xi=2.0
    return r*V_r*np.exp(-r**2/(2.0*xi))



#CROSSING EXACTLY AT DELTA=8.3

#this is easier!
vecs=[('0LL',0),('1LL',1)]


vecs=[('0LL',0)]
#vecs=[('1LL',1)]
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
"""
r_values=np.linspace(0.01,10,1000)
total=[]
for r in r_values:
    total.append(RPA_screened_interaction(r))


import matplotlib.pyplot as plt
plt.figure()
plt.scatter(r_values,np.array(total))

plt.savefig('test.png')
quit()
"""
layers=vecs

species=[]
for name in vecs:
     species.append(name[0])
species=list(set(species))

#eigenvals,eigenvec=calculate_relevant_eigenvectors(Delta1)
#cut eigenvec to contain only 4 LLs

#layers=[ ('band1',0), ('band1', 1)]

#,('down2', 0),('up',1)]#,('empty',0)]

#OK, SO THIS SORTS THAT ONE OUT FOR THE LL MIXING!
#layers=[ ('single_layer', [ {0: 0.4  }, {0: 0.4  },{1:np.sqrt(1-0.4**2)  }, {2: 0.04**2} ])]

#layers=[ ('down', 1),('down',0)]
#layers=[ ('single_layer', [ {0: np.sqrt(1/2)  },{1: np.sqrt(1-0.42-0.5)}, {2: np.sqrt(0.42)} ])]






layers=[ ('d',0)]


root_config_=np.array([0,1,0])




#root_config_=np.array([0,1,0])
Veps = 1e-5

#isotropic Nlayer interaction
#THIS BASICALLY JUST FORCES THE TAIL TO CONVERGE, AND CUTS IT OFF SHARPLY
V = { 'eps':Veps, 'xiK': 2}
print(layers)

#magnetic length in nm
#25*sqrt(1/B)
#B=12


#SO THIS IS DOUBLE GATED COULUMB
#BUT WE CAN ALSO TRY GAUSSIAN ONLY
e=1.6*10**(-19)
epsilon0=8.85*10**(-12)

hbar=6.626*10**(-34)/2/np.pi
B=12
l_B=np.sqrt(hbar/(e*B))
#d=50/(l_B*10**9)

#V['screening']= {'a':1, 'b':0.25} #you messed up by a factor of 2pi. Thats why instead of 7 you have approx 1.
#

#xi=2
#V['GaussianCoulomb']={}

V['haldane']={}
#V['rV(r)']={}


species=['d']
for i,layer in enumerate(species):
    for j,layer2 in enumerate(species):
            print(layer)
            #if i<=j:
            
            #key=(layer[0],layer2[0])
            key=(layer,layer2)
            print(key)
            #'rV', ['tol', 'coarse', 'd']
            #V['rV(r)'][key]={'rV':RPA_screened_interaction}#{rvr,'v':1., 'xi': xi}
            #else:
            V['haldane'][key]=[0,1,0] #because it gets overwritten cool cool.

            #V['GaussianCoulomb'][key]={'v':1., 'xi': xi} #because it gets overwritten cool cool.



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
model_par['pref']=0
#model_par['species']=species
#model_par['mu']=[0,mu]
#model_par['mu']=[0,0]
print(model_par)
#quit()


path='/mnt/users/dperkovic/DMRG_modified/data_load/nu=1_3_shift/'
#name='Haldane_1_3_1_3_no_interlayer_interaction'



#this is one without np.conj on the pref before shooting it into make_IMPO
######THIS IS ACTUAL TWO

#DO IT WITH 1in. BUt for 1in need to fix the indices in the other code
name=model_par['cons_C']+"_Haldane_nu=1_3_shift_Lx="+str(Lx)
name_save=path+name+'.pkl'
#print(xi)
#quit()
print(name)
graph=make_iMPO(model_par)



import pickle
data={'model_par':model_par,'graph':graph}
with open(name_save,'wb') as f:
    pickle.dump(data,f)


