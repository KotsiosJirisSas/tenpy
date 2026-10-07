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
from scipy.special import eval_laguerre

def get_Haldane_pseudopotential(V,m,x_values,LL_level=0):
    #calculates value of m-th Haldane pseudopotential for landau level LL_level
    #given some potential V, here V is actually a fourier transform in 2D of positional space 

  
    dk=x_values[1]-x_values[0]
    coeffs_1=np.zeros(LL_level+1)
    coeffs_1[-1]=1
    coeffs_2=np.zeros(m+1)
    coeffs_2[-1]=1
 
    poly_two=eval_laguerre(m,x_values**2)
    
    poly_one=(eval_laguerre(LL_level,x_values**2/2))**2

    integral=dk*np.sum(np.exp(-x_values**2)*poly_one*poly_two*V*x_values)/(2*np.pi)
  
    return integral


def make_iMPO(model_par):
    print("Start model in Old Tenpy",".."*10)

    ###
    # so graph is problematic on the new run, lets load graph and see if there is problem
    ######
    M = mod.QH_model(model_par,'')
    print("Old code finished producing MPO graph",".."*10)

    
      
    G=M.MPOgraph
    
   
    return G










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

    try:
        dulj=len(r_values)
        s=[]
        for r0 in r_values:
            s.append(np.sum(F_values * j0(k_values * r0) * k_values) * dk)
        s=np.array(s)
    except:
        s = np.sum(F_values * j0(k_values * r_values) * k_values) * dk
 
    return s / (2*np.pi)




###########################################
###########perform Fourier transform and then multiply via exp(-r^2/(2*xi))
##########################
def RPA_screened_interaction(r,a,b):
    q=np.linspace(0.01,1000,10000)
    c=1000000
    pi=a*(np.tanh(b*q**2))*(np.tanh(c*q))
    V_q=(np.tanh(c*q))*np.exp(-0.01*q)/(q+pi)
    V_r=inverse_radial_ft_2d_sum(V_q, q, r)

    return r*V_r*np.exp(-r**2/(2.0*xi))

def LK_interaction(r,a,xi):

    delta=0.01
    V_r=np.cos(a*r)/np.sqrt(r**2+delta**2)
  
    return r*V_r*np.exp(-r**2/(2.0*xi))
def coulumb_added_delta(r,V0,xi):
    epsilon=0.1

    #### add V0 term
    #scales it so that V0 changes the percentage only
    if xi==2:
        pref= 0.6862182437529724
    elif xi==1:
        pref= 0.5114132495511697
        
    A=4*np.pi/(2*np.pi*epsilon)*pref
    return np.exp(-r**2/(2*xi**2))+r*V0*A*np.exp(-r**2/(2*epsilon))
def RPA_screened_interaction_added_delta(r,a,b,pref,V0):
    q=np.linspace(0.01,1000,10000)
    c=1000000
    pi=a*(np.tanh(b*q**2))*(np.tanh(c*q))
    V_q=(np.tanh(c*q))*np.exp(-0.01*q)/(q+pi)
    V_r=inverse_radial_ft_2d_sum(V_q, q, r)
  
    epsilon=0.1
    A=4*np.pi/(2*np.pi*epsilon)*pref
    return r*V_r*np.exp(-r**2/(2.0*xi))+r*V0*A*np.exp(-r**2/(2*epsilon))
#CROSSING EXACTLY AT DELTA=8.3

#this is easier!
#vecs=[('0LL',0),('1LL',1)]


#vecs=[('1LL',0),('2LL',1)]
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


Lx=float(sys.argv[1])

V0=float(sys.argv[2])

a=float(sys.argv[3])

b=float(sys.argv[4])

xi=float(sys.argv[5])

vecs=[ ('d',0),('u',0)]

layers=vecs

species=[]
for name in vecs:
     species.append(name[0])
species=list(set(species))

import matplotlib.pyplot as plt
q=np.linspace(0.01,1000,10000)
c=1000000
print(a,b)
pi=a*(np.tanh(b*q**2))*(np.tanh(c*q))
V_q=(np.tanh(c*q))*np.exp(-0.01*q)/(q+pi)
r=np.linspace(0.01,10,10000)

plt.scatter(q,V_q)
V_r=[]

#pref=get_Haldane_pseudopotential(V_q,0,q,LL_level=0)
#print('V0, pseudopotential', pref)

#V_r=inverse_radial_ft_2d_sum(V_q, q, r)*np.exp(-r**2/(2.0*xi))

print(len(V_r))
#V_q=inverse_radial_ft_2d_sum(V_r, r, q)*((4*np.pi))

########
### OK, so there is issue of V0 being too small, that is we need to scale it up, by a factor of almost 5....
### WRITE THIS DOWN INTO THE NOTES FOR COMPLETENESS.
#######
#r=np.linspace(0.01,10,10000)
#####gives the prefactor
#pref=get_Haldane_pseudopotential(V_q,0,q,LL_level=0)
#print('V0, pseudopotential, after screen',pref)

#pot=RPA_screened_interaction_added_delta(r,a,b,pref,V0)



#plt.figure()

#plt.scatter(r,pot)
#plt.savefig('govno.png')

#quit()

layers=[ ('d',0),('u',0)]


#root_config_=np.array([1,1, 0, 0,0,0,0,0, 0, 0,0,0])
root_config_=np.array([1,1,0,0,0,0,0,0])

Veps = 1e-5

#isotropic Nlayer interaction
#THIS BASICALLY JUST FORCES THE TAIL TO CONVERGE, AND CUTS IT OFF SHARPLY
xiK=3
V = { 'eps':Veps, 'xiK': xiK}
print(layers)

#magnetic length in nm
#25*sqrt(1/B)
#B=123

#SO THIS IS DOUBLE GATED COULUMB
#BUT WE CAN ALSO TRY GAUSSIAN ONLY
e=1.6*10**(-19)
epsilon0=8.85*10**(-12)

hbar=6.626*10**(-34)/2/np.pi
B=12
l_B=np.sqrt(hbar/(e*B))
#d=50/(l_B*10**9)

#V['screening']= {'a':1, 'b':0.25} #you messed up by a factor of 2pi. Thats why instead of 7 you have approx 1.
#V['GaussianCoulomb']={}


#V['rV(r)']={}
V['haldane']={}
species=['d','u']
for i,layer in enumerate(species):
    for j,layer2 in enumerate(species):
            print(layer)
            #if i<=j:
            
            #key=(layer[0],layer2[0])
            key=((layer,layer2),(layer,layer2))
            print(key)
            #key=(layer,layer2)
            #'rV', ['tol', 'coarse', 'd']
            
            V['haldane'][key]=[1,0,0]
            #V['rV(r)'][key]={'rV':lambda r: RPA_screened_interaction_added_delta(r,a,b,pref,V0)}
            #V['haldane'][key]=[-1,1]
            #else:
            #V['GaussianCoulomb'][key]={'v':1., 'xi': xi} #because it gets overwritten cool cool.
print(V)
#quit()


model_par={ #for finite replace by periodic here
	'verbose': 2,
    'root_config': root_config_,
            'exp_approx': '1in', 
            'Vs': V, 
            'cons_K': True, 
            'cons_C': 'each', #or total
            'Lx': Lx
            }


model_par['layers']=layers
model_par['pref']=0
#model_par['species']=species
#model_par['mu']=[0,mu]
model_par['mu']=[0,0]
print(model_par)
#quit()

#name='haldane_1_3_1_3_graph_Lx='+str(Lx)+'xi=5'

path='/mnt/users/dperkovic/DMRG_modified/data_load/haldane_1_3_1_3/'
path='/mnt/users/dperkovic/DMRG_modified/data_load/V0_coulumb/'
#name='Haldane_1_3_1_3_no_interlayer_interaction'



#this is one without np.conj on the pref before shooting it into make_IMPO

#DO IT WITH 1in. BUt for 1in need to fix the indices in the other code
name=model_par['cons_C']+'_xiK='+str(xiK)+'_Coulumb_added_V0_V0='+str(V0)+'_b='+str(b)+"_0LL_Lx="+str(Lx)+'_xi='+str(xi)
#name=model_par['cons_C']+'_a='+str(a)+'_b='+str(b)+'V0'+str(V0)+'Lx='+str(Lx)+'_xi='+str(xi)
name_save=path+name+'.pkl'


graph=make_iMPO(model_par)
import pickle
quit()
model_par['Vs']={'V0':V0,'b':b}
data={'model_par':model_par,'graph':graph}
with open(name_save,'wb') as f:
    pickle.dump(data,f)



