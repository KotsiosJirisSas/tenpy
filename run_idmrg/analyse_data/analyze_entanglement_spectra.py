
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
import utils
import re
import ast





def find_mu_from_name(name):
    """
    Extract the V0 parameter from a filename
    mu
    The filename is expected to contain a substring of the form
    "V0=<number>", where the number may be positive, negative, decimal,
    or written in scientific notation.

    Parameters
    ----------
    name : str
        Filename or string containing the V0 value.

    Returns
    -------
    float
        Extracted mu value.

    Raises
    ------
    ValueError
        If no mu value is found in the filename.
    """

    # Search for a floating-point number immediately following "V0=".
    #match = re.search(r", name)
    mu = float(ast.literal_eval(re.search(r'_E=(\[[^\]]+\])', name).group(1))[2])

    return mu

def plot_entanglement_spectra(path,labels,nlayer,tol=0.01):
   

    dir_list = os.listdir(path)
    dir_list.sort()
 
    excluded = ["MPS", "Environment", "Envs"]
    for name in dir_list:
        
        if any(label in name for label in labels) and all(x not in name for x in excluded):
            print(name)
         
            data=utils.load_data_from_path(path+name)
        
            print("*"*30)
            print(name)
        
            ES=data['entanglement_spectrum']
           
            print('EE:',data['entanglement_entropy'])
            print('correlation',data['correlation_length'])
            print('density',data['density'])
            
            site=0
            val2=0
            #V0=utils.find_V0_from_name(name)
            #mu='bla'
            mu=utils.find_V0_from_name(name)
            #mu=find_mu_from_name(name)
            print(mu)

            if utils.is_periodic_density(data['density'],nlayer,tolerance=tol):
                #utils.ES_plot_no_fixed_sector(ES[site], 'smece', file_path='/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/analyze_DMRG_data/temporary_figure/'+str(name[:-3])+'_site='+str(site)+'_.png',ylim=False)
                #utils.ES_plot_fix_charge_sector(ES[site],'mu='+str(mu),file_path='/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/analyze_DMRG_data/temporary_figure/'+str(name[:-3])+'_site='+str(site)+'_.png',ylim=20,show_degeneracy=True)
                #"fix_N_only":
            
                #print('ININI')
                #import pickle
                #with open(name+".pkl", "wb") as f:
                #    pickle.dump(ES[site], f)
                utils.ES_plot_fix_charge_sector(ES[site],'mu='+str(mu),file_path='/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/analyze_DMRG_data/temporary_figure/'+str(name[:-3])+'_site='+str(site)+'_.png',ylim=20,param="fix_N_only",show_degeneracy=True)
                #quit()
                #plt.ylim()



if __name__=="__main__":
    path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/Landau_level_2_bands_V0_perturbation_each/'
    path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/Landau_level_mixing_4_bands_empty_bands/'

    path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/Landau_level_2_bands_V0_perturbation_each/Gaussian_Coulomb/'
    labels=['.h5','chi=2000','[0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0]']#,'pstate=[0,0,1,1,0,0,0,0]_']
    
    labels=['.h5','chi=2000','[0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0]']#,'pstate=[0,0,1,1,0,0,0,0]_']

    labels=['.h5','chi=5000','pstate=[0,0,1,1,0,0,0,0]_']#,'pstate=[0,0,1,1,0,0,0,0]_']

    nlayer=2
    #path='/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/trial_bilayer_states_haldane/'

    path="/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/Landau_level_2_bands_V0_perturbation_each_RPA_screened/"
    labels=['xi=2']
    labels=['V0=0.0']
    #labels=['chi=5000','each_Landau_level_2_bands_perturbed_V0=-0.82_xi=4_Lx=18.0_Gaussian_Coulomb_pstate=[0,0,1,1,0,0,0,0]_chi=5000']
    labels=['Lx=17','Lx=16','Lx=18']
    #labels=['each_RPA_screened_A=1.95_b=0.6_Landau_level_2_bands_perturbed_V0=0.0_xi=4_Lx=15.0_Coulomb_screened_gaussian_envelope_pstate=[0,0,1,0,0,0,0,1]_chi=8000']
    plot_entanglement_spectra(path,labels,nlayer,tol=51)


#