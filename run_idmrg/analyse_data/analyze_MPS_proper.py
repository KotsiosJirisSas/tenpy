import os
import re
import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy') 
np.set_printoptions(precision=5, suppress=True, linewidth=100)
plt.rcParams['figure.dpi'] = 150
import tenpy

import os
import pickle
import numpy as np
import h5py
from tenpy.tools import hdf5_io
from tenpy.networks.mps import TransferMatrix

import ast

def is_periodic_density(densities, period, tolerance=0.01):
    """
    Checks whether each unit cell is identical up to a relative tolerance.

    Parameters
    ----------
    densities : array-like
        Density values.
    period : int
        Number of sites per unit cell.
    tolerance : float
        Maximum allowed relative deviation.

    Returns
    -------
    bool
    """

    densities = np.asarray(densities)

    if len(densities) % period != 0:
        raise ValueError("Length of densities must be a multiple of period.")

    cells = densities.reshape(-1, period)

    reference = np.mean(cells, axis=0)

    relative_error = np.abs(cells - reference) / np.maximum(np.abs(reference), 1e-12)

    return relative_error,np.all(relative_error <= tolerance)








def load_data_from_path(path, name):
    """
    Load TeNPy HDF5 data from a file.

    Parameters
    ----------
    path : str
        Directory containing the HDF5 file.
    name : str
        Name of the HDF5 file.

    Returns
    -------
    dict
        Data loaded from the HDF5 file using TeNPy's hdf5_io loader.
    """

    # Open the HDF5 file in read-only mode.
    with h5py.File(path + name, 'r') as h5_file:

        # Load the HDF5 contents into a Python object.
     
        data = hdf5_io.load_from_hdf5(h5_file)
       

    return data




def find_V0_from_name(name):
    """
    Extract the V0 parameter from a filename.

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
        Extracted V0 value.

    Raises
    ------
    ValueError
        If no V0 value is found in the filename.
    """

    # Search for a floating-point number immediately following "V0=".
    match = re.search(r"V0=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", name)

  

    if match:
        # Convert the matched string to a floating-point number.
        V0 = float(match.group(1))

      

    else:
        # Raise an error if no V0 value could be found.
        raise ValueError("V0 not found")

    return V0




def extract_pstate_and_K_sector(name,nlayer):
    """
    Extract the pstate and compute the corresponding K sector from a filename.

    The filename is expected to contain a substring of the form
    "pstate=[...]", where the list contains occupation numbers.

    The K sector is computed relative to the average filling using

        K = sum_i i * (pstate_i - filling)

    Parameters
    ----------
    name : str
        Filename or string containing the pstate list.

    Returns
    -------
    tuple
        A tuple `(pstate, K)`, where `pstate` is the extracted list and `K`
        is the computed momentum-sector value.

    Raises
    ------
    ValueError
        If no pstate list is found in the filename.
    """

    # Search for the pstate list inside the filename.
    pstate_match = re.search(r"pstate=(\[[^\]]*\])", name)

    # If the pstate substring is missing, stop with an informative error.
    if not pstate_match:
        raise ValueError("pstate not found")

    # Safely convert the matched list string into a Python list.
    pstate = ast.literal_eval(pstate_match.group(1))

    # Compute the average filling fraction.
    filling = np.sum(pstate) / len(pstate)

    # Compute the K sector relative to the average filling.
    K = np.sum(np.arange(len(pstate))//nlayer * (np.array(pstate) - filling))

    return pstate, K

def find_chi_from_name(name):
    m = re.search(r'_chi=(\d+)', name)
    if not m:
        raise ValueError("chi not found in name")
    return int(m.group(1))





def find_Lx_from_name(name,verbose):
    """
    Extract the V0 parameter from a filename.

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
        Extracted V0 value.

    Raises
    ------
    ValueError
        If no V0 value is found in the filename.
    """

    # Search for a floating-point number immediately following "V0=".
    match = re.search(r"Lx=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", name)

    # Set verbosity level for printing extracted values.
   

    if match:
        # Convert the matched string to a floating-point number.
        Lx = float(match.group(1))

        # Print the extracted V0 value if verbose output is enabled.
        if verbose > 0:
            print('Lx:', Lx)

    else:
        # Raise an error if no V0 value could be found.
        raise ValueError("Lx not found")

    return Lx


def find_V0_from_name(name):
    """
    Extract the V0 parameter from a filename.

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
        Extracted V0 value.

    Raises
    ------
    ValueError
        If no V0 value is found in the filename.
    """

    # Search for a floating-point number immediately following "V0=".
    match = re.search(r"V0=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", name)

  

    if match:
        # Convert the matched string to a floating-point number.
        V0 = float(match.group(1))

      

    else:
        # Raise an error if no V0 value could be found.
        raise ValueError("V0 not found")

    return V0




def extract_pstate_and_K_sector(name,nlayer):
    """
    Extract the pstate and compute the corresponding K sector from a filename.

    The filename is expected to contain a substring of the form
    "pstate=[...]", where the list contains occupation numbers.

    The K sector is computed relative to the average filling using

        K = sum_i i * (pstate_i - filling)

    Parameters
    ----------
    name : str
        Filename or string containing the pstate list.

    Returns
    -------
    tuple
        A tuple `(pstate, K)`, where `pstate` is the extracted list and `K`
        is the computed momentum-sector value.

    Raises
    ------
    ValueError
        If no pstate list is found in the filename.
    """

    # Search for the pstate list inside the filename.
    pstate_match = re.search(r"pstate=(\[[^\]]*\])", name)

    # If the pstate substring is missing, stop with an informative error.
    if not pstate_match:
        raise ValueError("pstate not found")

    # Safely convert the matched list string into a Python list.
    pstate = ast.literal_eval(pstate_match.group(1))

    # Compute the average filling fraction.
    filling = np.sum(pstate) / len(pstate)

    # Compute the K sector relative to the average filling.
    K = np.sum(np.arange(len(pstate))//nlayer * (np.array(pstate) - filling))

    return pstate, K

def find_chi_from_name(name):
    m = re.search(r'_chi=(\d+)', name)
    if not m:
        raise ValueError("chi not found in name")
    return int(m.group(1))





def find_Lx_from_name(name,verbose):
    """
    Extract the V0 parameter from a filename.

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
        Extracted V0 value.

    Raises
    ------
    ValueError
        If no V0 value is found in the filename.
    """

    # Search for a floating-point number immediately following "V0=".
    match = re.search(r"Lx=([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", name)

    # Set verbosity level for printing extracted values.
   

    if match:
        # Convert the matched string to a floating-point number.
        Lx = float(match.group(1))

        # Print the extracted V0 value if verbose output is enabled.
        if verbose > 0:
            print('Lx:', Lx)

    else:
        # Raise an error if no V0 value could be found.
        raise ValueError("Lx not found")

    return Lx

def g_avg_at_separation(psi, j):
    """Average <n(i) n(i+j)> over all valid reference sites i within
    one unit cell, keeping i+j < L (no wraparound)."""
    g_vals = []
    L=8
    for i in range(L - j):   # ensures i + j < L
        g = psi.correlation_function('nOp', 'nOp', [i], [i + j])
        g_vals.append(np.real(np.asarray(g).squeeze()))
    return np.mean(g_vals)

def get_corr_lengths(psi):
    sec=psi.correlation_length_charge_sectors()[0]

    TM = TransferMatrix(psi, psi, charge_sector=sec, form='B')
    etas, vecs = TM.eigenvectors(num_ev=3, which='LM')
   
    #psi.correlation_length()
        #print('eigenvalue:', eta, ' correlation length:', xi)
    #get_corr_lengths(psi)
    
  
    nontrivial = [e for e in etas if not np.isclose(np.abs(e), 1.0, atol=1e-8)]
  
    xis = -psi.L / np.log(np.abs(nontrivial[0]))   # matches psi.correlation_length()'s convention
    return xis
def extract_all_data_from_folder(path, nlayer, labels, verbose=0, tol=1,
                                  norm_tol=1, js=range(6)):
    """
    Load DMRG data files (.h5) and their companion MPS pickle files (.pkl),
    filter on density periodicity and DMRG norm error, and extract:
    energy, correlation length (computed directly from the MPS), V0,
    K sector, entanglement entropy, g(0) (both file value and full
    wavefunction-computed values for i=0, j in `js`).
    """

    dir_list = os.listdir(path)
    dir_list.sort()

    energies = []
    correlation_lengths = []
    V0s = []
    Ks = []
    EEs = []
    polarizations = []
    chis = []
    Lxs = []
    C_charge = []
    K_visc = []
    g0s_file = []   # g(0) as stored in the .h5 file
    g0s_wf = []     # list of [g(0,j) for j in js] computed directly from psi

    excluded = ["MPS", "Environment", "Envs"]

    for name in dir_list:

        if all(label in name for label in labels) and all(x not in name for x in excluded):

            name_noext = name[:-3]  # strip ".h5"

            print(name)
            data = load_data_from_path(path, name_noext + '.h5')
            

            V0 = find_V0_from_name(name_noext)
            chi = find_chi_from_name(name_noext)
            Lx = find_Lx_from_name(name_noext, verbose)

            norm = data['sweep_stats']['norm_err'][-1]
            print('norm',norm)
            relative_eror,trueth=is_periodic_density(data['density'], nlayer, tolerance=tol)
            print('desnity deviation',relative_eror)
            if  trueth and norm < norm_tol:

                if verbose > 0:
                    print("*" * 30)
                    print(name_noext)
                    print('density:', data['density'])
                    print('V0:', V0)
                    print('energy', data['energy'])

                # --- load companion MPS pickle to compute quantities directly from the wavefunction ---
                pkl_name = name_noext + '_MPS_data_new_tenpy.pkl'
                psi = None
                try:
                    with open(os.path.join(path, pkl_name), 'rb') as f:
                        mps_data = pickle.load(f)
                    psi = mps_data['mps']
                except Exception as e:
                    if verbose > 0:
                        print(f"Could not load MPS file for {name_noext}: {e}")

                # correlation length, computed directly from psi (not from the h5 file)
                if psi is not None:
                    try:
                        corr_length =get_corr_lengths(psi)/(2*np.pi)
                        #= psi.correlation_length() / (2 * np.pi)
                    except Exception as e:
                        if verbose > 0:
                            print(f"correlation_length() failed for {name_noext}: {e}")
                        corr_length = np.nan
                else:
                    corr_length = np.nan

                V0s.append(V0)
                chis.append(chi)
                energies.append(data['energy'])
                correlation_lengths.append(corr_length)
                EEs.append(data['entanglement_entropy'][0])
                polarizations.append(np.abs(data['density'][0] - data['density'][1]))

                pstate, K = extract_pstate_and_K_sector(name_noext, nlayer)
                if verbose > 0:
                    print('K', K, pstate)

               
                g0_file = np.mean(data['g(0)'])
                g0s_file.append(g0_file)

                # g(0) computed directly from the wavefunction, for i=0, j in js
                if psi is not None:
                    g0_wf_list = []
                    for j in js:
                        try:
                            g_val=g_avg_at_separation(psi, j)
                        except Exception as e:
                            if verbose > 0:
                                print(f"correlation_function failed for {name_noext}, j={j}: {e}")
                            g_val = np.nan
                        g0_wf_list.append(g_val)
                else:
                    g0_wf_list = [np.nan] * len(list(js))
                g0s_wf.append(g0_wf_list)

                Lxs.append(Lx)
                Ky = data['K']
                C = data['C']
                C_charge.append(C)
                K_visc.append(Ky)
                Ks.append(K)

    return (Ks, V0s, Lxs, energies, correlation_lengths, EEs, polarizations,chis, C_charge, K_visc, g0s_file, g0s_wf)

nlayer=2
path = '/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/Landau_level_2_bands_V0_perturbation_each_Ly=14_NEW_corr_length/'
labels=['chi=8000','xi=4']

(Ks, V0s, Lxs, energies, correlation_lengths, EEs, polarizations,chis, C_charge, K_visc, g0s_file, g0s_wf) = extract_all_data_from_folder(path, nlayer, labels, verbose=0, tol=0.5, norm_tol=1e-3, js=range(6))

def print_list(name, arr):
    print(f"{name} = {np.array2string(np.asarray(arr), separator=', ', threshold=np.inf)}")

print_list("V0s", V0s)
print_list("correlation_lengths", correlation_lengths)
print_list("chis", chis)
print("Ks =", list(Ks))
print_list("energies", energies)
print_list("g0s_file", g0s_file)
print_list("g0s_wf", g0s_wf)
print_list("EEs", EEs)
print_list("Lxs", Lxs)

#print("K_visc =", list(K_visc))
#print_list("polarizations", polarizations)