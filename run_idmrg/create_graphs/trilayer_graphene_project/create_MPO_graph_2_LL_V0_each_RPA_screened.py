#!/mnt/users/dperkovic/myenv/bin/python
#source /mnt/users/dperkovic/myenv/bin/activate
"""Code for producing DMRG graph"""
import sys
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 

import numpy as np
from tenpy.models import multilayer_qh_DP_final as mod
from scipy.special import ive



import pickle



np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)





def RPA_screened_Coulomb(q,A,b,epsilon):
    # bare (epsilon-regularized) 2D Coulomb interaction: V0(q) = 2*pi/(|q|+epsilon)
    V_unsc=2*np.pi/(np.abs(q)+epsilon)
    
    # RPA denominator: 1 + V0(q) * A * tanh(b*q^2)  (polarizability-like screening factor)
    screened=1+V_unsc*A*np.tanh(b*q**2)

    # screened interaction V(q) = V0(q) / (1 + V0(q)*A*tanh(b*q^2))
    return V_unsc/screened


def convolve_gaussian_radial(q, xi, Vq, k_max, V0, n=2000):
    """
    Computes I(q) = 2*pi * integral_0^k_max k * Vq(k, epsilon) * exp(-xi^2*(q-k)^2/2) * I0(xi^2*q*k) dk + 4*pi*V0

    q can be a float or an ndarray; output matches (scalar in -> scalar out,
    array in -> array out).

    q      : float or ndarray, points at which to evaluate I(q)
    xi     : float, Gaussian width parameter
    Vq     : callable Vq(k, epsilon), vectorized radial function
    k_max  : float, upper integration cutoff (must be finite)
    V0     : float, added as a constant 4*pi*V0 term (e.g. q=0 correction)
    n      : int, number of grid points
    """
    #NOTE: tested in file test_RPA_interaction.ipynb
    # remember if the caller passed a scalar q, so we can unwrap the result at the end
    scalar_input = np.isscalar(q)
    # promote q to at least a 1D array so the broadcasting below works uniformly
    q = np.atleast_1d(np.asarray(q, dtype=float))

    k = np.linspace(0, k_max, n)        # shared k-grid, shape (n,)
    dk = k[1] - k[0]
    epsilon =  1e-7                     # small regulator passed into Vq to avoid q=0 singularities

    # broadcast: q -> (m,1), k -> (1,n) => z, integrand shape (m,n)
    z = xi**2 * q[:, None] * k[None, :]

    # integrand of the radial convolution integral (Riemann-sum grid version):
    # k * Vq(k) * exp(-xi^2*(q-k)^2/2) * ive(0,z), where ive(0,z)=I0(z)*exp(-|z|)
    # is used instead of I0 directly for numerical stability at large z;
    # the extra xi**2 factor here folds in the prefactor needed when Vq is fed
    # as the "bare" 1/q-type kernel (see chat derivation) rather than 2*pi/q
    integrand = (k[None, :] * Vq(k, epsilon)[None, :]
        * np.exp(-xi**2 * (q[:, None] - k[None, :])**2 / 2.0)
        * ive(0, z))*xi**2

    # Riemann sum over k, plus the constant 4*pi*V0 offset term
    result = np.sum(integrand, axis=1) * dk + 4 * np.pi * V0

    # unwrap back to a scalar if a scalar q was passed in
    return result[0] if scalar_input else result








def make_iMPO(model_par):
    print("Start model in Old Tenpy",".."*10)

    ###
    # so graph is problematic on the new run, lets load graph and see if there is problem
    ######
    M = mod.QH_model(model_par,'')
    print("Old code finished producing MPO graph",".."*10)

    
      
    G=M.MPOgraph
    
   
    return G




#RPA screening parameters


Lx=round(float(sys.argv[1]),3)   # system circumference / length, from command-line arg 1

V0=round(float(sys.argv[2]),4)   # V0 perturbation strength, from command-line arg 2

xi=4    # Gaussian smearing width used in convolve_gaussian_radial



### in terms of interaction
mus=[0,0]     # chemical potentials for the two layers/species


pref=1.0     # overall prefactor for the model


#this is easier!
vecs=[('down_K-',0),('up_K-',0)]     # layer/species definitions: (species name, index)



layers=vecs    # alias: layers = vecs

species=[]
# collect the species/layer names out of the (name, index) tuples in vecs
for name in vecs:
     species.append(name[0])
species=list(set(species))    # unique species names



root_config_=np.array([0,0,1,0,0,1,0,0])    # root configuration / filling pattern for the MPS

Veps = 1e-5    # regularization scale for the bare Coulomb potential used below

#isotropic Nlayer interaction

parameters_for_coulomb= { 'eps':Veps, 'xiK':2.5}    # metadata dict describing the Coulomb params (for saving)

V = { 'eps':Veps, 'xiK':2.5}     # actual params dict that will carry the callables
print(layers)


A=1.95    # RPA screening amplitude parameter
b=0.6     # RPA screening tanh-argument scale parameter



#NOTE: va;ies tested in file test_RPA_interaction.ipynb

## k_max from the intergral must be larger than the DMRG cutoff in k space.
# set k_max to be large enough
# 2xi added to make sure the edge is correct to within 2 sigma
k_max=-np.log(0.1*Veps)*V['xiK']+2*xi
#### fix dk to be small enough, tested in ipynb file
### deltak chosen so that maximal error is <=0.1% at q=0.0
deltak=0.001
deltak=0.0005
# number of grid points needed to keep grid spacing <= deltak, rounded to nearest 1000
n=int(round(k_max/deltak,-3))
V['v(q)']={}
parameters_for_coulomb['v(q)']={}
# loop over all (layer,layer2) pairs to build the interaction matrix v(q)
for i,layer in enumerate(species):
    for j,layer2 in enumerate(species):
            print(layer)
            if i<=j:    # only fill upper triangle (symmetric interaction matrix)
                key=(layer,layer2)

                
                if i==j:
                    # intra-layer interaction: RPA-screened Coulomb convolved with the Gaussian smearing
                    RPA_screened_interaction_fixed = lambda qs,epsilon: RPA_screened_Coulomb(qs,A, b,epsilon)
                    full_RPA_interaction = lambda q:  convolve_gaussian_radial(q, xi, RPA_screened_interaction_fixed , k_max,V0/2, n=n)
                    # save the (picklable) description of this interaction, since the lambda itself can't be pickled
                    parameters_for_coulomb['v(q)'][key]={'vq':'RPA_screened_interaction','A':A,'b':b,'xi':xi,'V0':V0/2,'num_of_points':n,'k_max':k_max} #NOTE: IMPORTANT; makes it consisten with previous calculations
                    V['v(q)'][key]={'vq':full_RPA_interaction} #NOTE: IMPORTANT; makes it consisten with previous calculations
                else:
                    # inter-layer interaction: same as above but doubled (factor 2 for the off-diagonal coupling)
                    RPA_screened_interaction_fixed = lambda qs,epsilon: RPA_screened_Coulomb(qs,A, b,epsilon)
                    full_RPA_interaction = lambda q:  2*convolve_gaussian_radial(q, xi, RPA_screened_interaction_fixed , k_max,V0/2, n=n)
                    parameters_for_coulomb['v(q)'][key]={'vq':'2xRPA_screened_interaction','A':A,'b':b,'xi':xi,'V0':V0/2,'num_of_points':n,'k_max':k_max} #NOTE: IMPORTANT; makes it consisten with previous calculations

                    V['v(q)'][key]={'vq':full_RPA_interaction} #because it gets overwritten cool cool.
print(V)



####
model_par={ #for finite replace by periodic here
	'verbose': 2,
    'root_config': root_config_,
            'exp_approx': 'slycot', 
            'Vs': V, 
            'cons_K': True, 
            'cons_C': 'each', #or total
            'Lx': Lx,
            }


model_par['layers']=layers

model_par['pref']=pref
model_par['mu']=-np.array(mus)

print(model_par)




path='/mnt/users/dperkovic/DMRG_modified/data_load/Landau_level_2_bands_V0_perturbation_each_RPA_screened/'   
name='each_RPA_screened_A='+str(A)+'_b='+str(b)+'_Landau_level_2_bands_perturbed_V0='+str(V0)+'_xi='+str(xi)+"_Lx="+str(Lx)+'_Coulomb_screened_gaussian_envelope' 

name_save=path+name+'.pkl'


graph=make_iMPO(model_par)     # build the DMRG MPO graph with the model parameters (including the V(q) callables)
import pickle
print('done')
### since cannot save lambda function.
model_par['Vs']=parameters_for_coulomb    # swap in the picklable description before saving (lambdas aren't picklable)
data={'model_par':model_par,'graph':graph}
with open(name_save,'wb') as f:
    pickle.dump(data,f)