#!/usr/bin/env python
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import sys
import pickle

print(sys.version)
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import tenpy
from tenpy.networks.mps import TransferMatrix

from tenpy.networks.mps import MPS

tenpy.tools.misc.setup_logging(to_stdout="INFO")



import pickle
from tenpy.linalg.charges import LegCharge

"""Code for running DMRG """
import sys
import os

import numpy as np
from tenpy.networks.mps import MPS

from tenpy.networks.mps import MPS



from tenpy.networks.site import QH_MultilayerFermionSite_final

import h5py
from tenpy.tools import hdf5_io
#SAVE THE DATA
#import h5py
#from tenpy.tools import hdf5_io
import pickle
np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)



def load_data(loaded_xxxx,sites,Na):
    """
    loads MPS as segment mps of length len(sites)
    name: Str, name of the .pkl file from which we import
    sites: list of class:Sites, list of hilbert spaces corresponding to each site
    Na: number of unit cells
    """
    L2=len(sites)
    print(L2)
    #finds length of an infinite unit cell
    Bflat2=loaded_xxxx['B_flat']
    Bflat=[]
    for i in range(len(Bflat2)):
    
        Bflat.append(np.transpose(Bflat2[i],(1,0,2)))
    Bflat=Bflat
    
    #load singular values
    Ss=loaded_xxxx['SS']

    #load charge infromation  
    qflat2=loaded_xxxx['qflat_left']#[0]
    
    

        

    infinite_unit_cell=len(Bflat)



    #change qflat into representation consistent with Tenpy3
    #this is just charges of the leftmost leg
    """
    qflat=[]
    for i in range(len(qflat2)):
        kopy=[]
        for m in range(len(qflat2[i])):
            if m==0:
                kopy.append(qflat2[i][m]+shift*qflat2[i][1])
            else:
                kopy.append(qflat2[i][m])
        qflat.append(kopy)
    """
    qflat=np.array(qflat2)
    print("smece JEDNO")
    """
    for i in range(len(qflat)):
        qflat[i][0]+=charge_shift[0]
        qflat[i][1]+=charge_shift[1]
    #PERMUTE Ss values
    """
  

    #cut the mps into mps of wanted size
    #Ss.append(Ss[0])
    print(len(Ss))
    print(len(Bflat))
    #define left most leg of charges
    chargeinfo=sites[0].leg.chinfo
    print(chargeinfo)
    left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1).bunch()[1]
   
    #print(kopy)
    #create MPS,

    #left_leg=LegCharge.from_qflat(chargeinfo,qflat,qconj=1)
    #charges are calculated from the left leg
    Bflat=Bflat*Na
    Ss=Ss*Na

    

    mps=MPS.from_Bflat(sites,Bflat,SVs=Ss, bc='infinite',legL=left_leg)
    print('loaded mps from data',".."*30)
    
    return mps


def load_param(name):
    print(name)
    path='/mnt/users/dperkovic/quantum_hall_dmrg/data_load/one_third_latest/'
    path='/mnt/users/dperkovic/quantum_hall_dmrg/data_load/multilayer_with_interlayer_interaction_Haldane/'
    with open(path+name+'.pkl', 'rb') as f:
        loaded_xxxx = pickle.load(f, encoding='latin1')

    print(loaded_xxxx.keys())
    #print(loaded_xxxx['graph'][0])
    
    #keys=[ 'exp_approx',  'cons_K',  'MPS_Ss', 'cons_C', 'Lx', 'root_config', 'LL', 'Vs']
    #model_par={}
    #for k in keys:
    #    model_par[k]=loaded_xxxx[k]
    
    model_par = loaded_xxxx['Model']
    
    #print(model_par)
    print(model_par['root_config'])
    #quit()
    N_layer=2
    root_config_ = model_par['root_config'].reshape(3,N_layer)

    conserve=[]
    if  model_par['cons_C']=='total':
        conserve.append('N')

    if model_par['cons_K']:
        conserve.append('K')
    #load root_configuration and conservation laws
    if len(conserve)==1:
        conserve=conserve[0]
    else:
        conserve=tuple(conserve)
    conserve=('each','K')
    return model_par,conserve,root_config_,loaded_xxxx




import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

path = '/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/Landau_level_2_bands_V0_perturbation_each_Ly=14_NEW_corr_length/'

i = 0
js = [0, 1, 2, 3, 4, 5]

for name in os.listdir(path):
    if '_chi=8000_MPS_data_new_tenpy' in name and name.endswith('.pkl'):

        full_path = os.path.join(path, name)
        with open(full_path, 'rb') as f:
            data = pickle.load(f)

        psi = data['mps']
        base_name = name.replace('.pkl', '')

        gs = []
        for j in js:
            g0 = psi.correlation_function('nOp', 'nOp', [i], [j])
            # correlation_function returns a (1,1) array for single-site inputs
            g0 = np.asarray(g0).squeeze()
            gs.append(g0)
            print(f"{base_name}: i={i}, j={j}, g0={g0}")

        # --- save the full gs list for this file ---
        gs_save_path = os.path.join(path, f'gs_list_{base_name}.pkl')
        with open(gs_save_path, 'wb') as f:
            pickle.dump({'i': i, 'js': js, 'gs': gs}, f)

        # --- save + plot each (i,j) pair separately ---
        for j, g_val in zip(js, gs):
            plt.figure()
            plt.bar([f'i={i},j={j}'], [np.real(g_val)])
            plt.ylabel(r'$\langle n_i n_j \rangle$')
            plt.title(base_name)
            plt.tight_layout()

            fig_path = os.path.join(path, f'corr_i={i}_j={j}_{base_name}.png')
            plt.savefig(fig_path, dpi=150)
            plt.close()

            # also save the individual scalar value
            #val_path = os.path.join(path, f'corr_i={i}_j={j}_{base_name}.pkl')
            print(g_val)
            #with open(val_path, 'wb') as f:
            #    pickle.dump(g_val, f)
quit()










print('total corr lenght:',psi.correlation_length())
Length=[]
sectors=psi.correlation_length_charge_sectors()
results=[]
for sec in [sectors[3]]:
    print(sec)
    try:
        TM = TransferMatrix(psi, psi, charge_sector=sec, form='B')
        etas, vecs = TM.eigenvectors(num_ev=3, which='LM')
        eta = etas[0]
        #psi.correlation_length()
           #print('eigenvalue:', eta, ' correlation length:', xi)
        print(etas)
        nontrivial = [e for e in etas if not np.isclose(np.abs(e), 1.0, atol=1e-8)]
        print(nontrivial)
        print(psi.L)
        xi = -psi.L / np.log(np.abs(nontrivial[0]))   # matches psi.correlation_length()'s convention
        print('old way xi',xi)
        xi=psi.correlation_length(charge_sector=sec)
        print('lenght again',xi)
        results.append([sec, eta, xi])
    except Exception as e:
        print('cannot calculate:', e)

#Length=psi.correlation_length()
print('correlation length:',Length)


quit()

filling=psi.expectation_value("nOp")

print('Filling:',filling)


E_spec=psi.entanglement_spectrum(by_charge=True)
#print('entanglement spectrum:',E_spec)
pairs = [(i, i + 1) for i in range(0, psi.L-1, 2)]
g0= [psi.correlation_function('nOp', 'nOp', [i], [j])[0, 0] for i, j in pairs]
print(g0)


EE=psi.entanglement_entropy()
print('entanglement entropy:',EE)

