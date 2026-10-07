import sys

# Add the local TeNPy installation to the Python path so that it can be imported.
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy')

import re
import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.optimize import curve_fit
from tenpy.tools import hdf5_io
import ast


def linear_model(x, m, c):
    """
    Linear model used for fitting data.

    Parameters
    ----------
    x : float or np.ndarray
        Independent variable.
    m : float
        Gradient of the line.
    c : float
        Intercept of the line.

    Returns
    -------
    float or np.ndarray
        Value of m*x + c.
    """

    # Return the value of a straight line.
    return m * x + c


def save_graph_data(name, dipoles, energies, Ks):
    """
    Save graph data to a text file.

    The output file contains dipole values, energies, and momentum-sector
    values K. This is useful for storing processed data used in plots.

    Parameters
    ----------
    name : str
        Name of the output file, without the ".txt" extension.
    dipoles : list or np.ndarray
        Dipole values to save.
    energies : list or np.ndarray
        Energy values to save.
    Ks : list or np.ndarray
        Momentum-sector values to save.
    """

    # Open a text file with the requested name.
    with open(name + ".txt", "w") as f:

        # Write dipole data.
        f.write("dipole:\n")
        f.write(str(dipoles) + "\n")

        # Write energy data.
        f.write("Energies:\n")
        f.write(str(energies) + "\n")

        # Write momentum-sector data.
        f.write("Ks:\n")
        f.write(str(Ks) + "\n")



def ES_plot_fix_charge_sector(es, title, param='fix_N_and_Sz', file_path='data/ES.png',unit_cell_size=16, ylim=False, show_degeneracy=False):
    """
    Plot the entanglement spectrum for a site.

    The function first identifies the charge sector with the largest total
    weight, then plots the entanglement energies in that sector as a function
    of momentum K.

    Parameters
    ----------
    es : list
        Entanglement spectrum data. Each entry is expected to contain charge
        sector information and corresponding entanglement energies.
    title : str
        Plot title.
    file_path : str, optional
        Path where the generated plot should be saved.
    param : str
        Choice of N, Sz sector. We always fix N, but sometimes allow Sz to change.
    unit_cell_size : int
        Gives periodicity of unit cell so system can tell between even and odd.
    show_degeneracy : bool, optional
        If True, label levels with degeneracy greater than one.
    degeneracy_tol : float, optional
        Energy tolerance used to identify degenerate levels.
    """

    minEE = np.array([
        list(sec[0]) + [np.sum(np.exp(-2 * sec[1]))]
        for sec in es
    ])

    pts = []
    pts_full = []

    if param == "fix_N_and_Sz":
        N = minEE[np.argmax(minEE[:, -1]), 1:-1]

        for sec in es:
            if np.array_equal(sec[0][1:], N):
                pts.extend([[sec[0][0], p] for p in sec[1]])
                pts_full.extend([np.append(sec[0], p) for p in sec[1]])

    elif param == "fix_N_only":
        N = np.sum(minEE[np.argmax(minEE[:, -1]), 1:-1])

        for sec in es:
            if np.sum(sec[0][1:]) == N:
                pts.extend([[sec[0][0], p] for p in sec[1]])
                pts_full.extend([np.append(sec[0], p) for p in sec[1]])

    elif param == "fix_N_different_parity":
        N_max = np.sum(minEE[np.argmax(minEE[:, -1]), 1:-1])
        parity = N_max % unit_cell_size

        mask = (np.sum(minEE[:, 1:-1], axis=1) % unit_cell_size) != parity

        if len(mask) != 0:
            idx2 = np.where(mask)[0][np.argmax(minEE[mask, -1])]
            N2 = np.sum(minEE[idx2, 1:-1])

            for sec in es:
                if np.sum(sec[0][1:]) == N2:
                    pts.extend([[sec[0][0], p] for p in sec[1]])
                    pts_full.extend([np.append(sec[0], p) for p in sec[1]])
    else:
        raise ValueError(f"Unknown param: {param}")

    pts = np.array(pts)
    pts_full = np.array(pts_full)

    if len(pts) == 0:
        pts = np.zeros((0, 2), dtype=int)

    wh, wv = 1, 0.01

    plt_kwargs = {
        'markersize': 10,
        'color': 'navy',
        'marker': [(-wh, -wv), (wh, -wv), (wh, wv), (-wh, wv)],
        'linewidth': 0.
    }

    plt.figure()
    plt.plot(pts[:, 0], pts[:, 1], **plt_kwargs)

    if show_degeneracy:
        degeneracy_tol=5e-2
        for K in np.unique(pts[:, 0]):
            energies = np.sort(pts[pts[:, 0] == K, 1])
            used = np.zeros(len(energies), dtype=bool)

            for i, energy in enumerate(energies):
                if used[i]:
                    continue

                group = np.where(np.abs(energies - energy) < degeneracy_tol)[0]
                group = group[~used[group]]
                used[group] = True

                if len(group) > 1:
                    plt.annotate(str(len(group)), (K, np.mean(energies[group])), xytext=(-8, 0), textcoords='offset points', fontsize=10, ha='right', va='center', clip_on=True)

    plt.xlabel(r'$K$', fontsize=18)
    plt.ylabel(r'$E_i$', fontsize=18, rotation='horizontal')
    plt.title(title)

    if ylim:
        plt.ylim(top=ylim)

    plt.savefig(file_path, dpi=400)

    return pts, pts_full

def load_data_from_path( name):
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
    with h5py.File( name, 'r') as h5_file:

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

    # Set verbosity level for printing extracted values.
    verbose = 1

    if match:
        # Convert the matched string to a floating-point number.
        V0 = float(match.group(1))

        # Print the extracted V0 value if verbose output is enabled.
        if verbose > 0:
            print('V0:', V0)

    else:
        # Raise an error if no V0 value could be found.
        raise ValueError("V0 not found")

    return V0


def extract_pstate_and_K_sector(name):
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
    K = np.sum(np.arange(len(pstate)) * (np.array(pstate) - filling))

    return pstate, K


def plot_in_different_colors(Ks, V0s, energies, name):
    """
    Plot data as a function of V0, using a different curve for each K sector.

    Parameters
    ----------
    Ks : list or np.ndarray
        Momentum-sector values.
    V0s : list or np.ndarray
        V0 parameter values.
    energies : list or np.ndarray
        Values to plot on the y-axis. In the current script this is usually
        the correlation length, despite the variable name.
    name : str
        Output filename without the ".png" extension.
    """

    # Convert inputs to NumPy arrays so that boolean masks work reliably.
    Ks = np.array(Ks)
    V0s = np.array(V0s)
    energies = np.array(energies)

    # Start a new figure.
    plt.figure()

    # Plot one curve for each unique K sector.
    for K in np.unique(Ks):

        # Select only the data belonging to the current K sector.
        mask = Ks == K

        # Sort by V0 so that each curve connects points in increasing V0 order.
        order = np.argsort(V0s[mask])

        # Plot the selected and sorted data.
        plt.plot(
            V0s[mask][order],
            energies[mask][order],
            marker="o",
            label=f"K={K}"
        )

    # Label the axes.
    plt.xlabel("V0")
    plt.ylabel("Correlation length")

    # Set y-axis limits.
    plt.ylim(0, 10)

    # Add a legend identifying each K sector.
    plt.legend()

    # Improve spacing before saving.
    plt.tight_layout()

    # Save the plot as a PNG file.
    plt.savefig(name + '.png')


def plot_energy_against_V0(path):
    """
    Load DMRG data files and plot correlation length against V0.

    For each suitable HDF5 file in the directory, this function extracts:
    - the energy,
    - the correlation length,
    - the V0 value from the filename,
    - the pstate and corresponding K sector from the filename.

    It then plots the correlation length as a function of V0, with different
    colors/curves for different K sectors.

    Parameters
    ----------
    path : str
        Directory containing the HDF5 data files.
    """

    # Get and sort the contents of the directory.
    dir_list = os.listdir(path)
    dir_list.sort()

    # Lists used to collect quantities from each file.
    energies = []
    correlation_lengths = []
    V0s = []
    Ks = []

    # Loop over all files in the directory.
    for name in dir_list:

        # Default fallback value.
        val = ''

        # Skip MPS and environment files.
        if ("MPS" not in name and 'Environment' not in name and 'Envs' not in name):

            # Remove the ".h5" extension from the filename.
            name = name[:-3]

            print("*" * 30)
            print(name)

            # Load the HDF5 data.
            data = load_data_from_path(path, name + '.h5')

            # Extract V0 from the filename.
            V0 = find_V0_from_name(name)

            # Store V0.
            V0s.append(V0)

            # Print diagnostic information.
            print('length', data['correlation_length'] / (2 * np.pi))
            print('energy', data['energy'])

            # Store energy and rescaled correlation length.
            energies.append(data['energy'])
            correlation_lengths.append(data['correlation_length'] / (2 * np.pi))

            # Extract the pstate and compute the K sector.
            pstate, K = extract_pstate_and_K_sector(name)

            # Store K sector.
            Ks.append(K)

            print('K', K)

    # Output directory for generated plots.
    path = '/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/analyze_DMRG_data/temporary_figure/'

    # Output filename.
    # The first assignment is overwritten by the second one.
    name = 'Energy_V0_different_Ks.png'
    name = 'cor_lengts_V0_different_Ks.png'

    # Plot correlation length against V0 for each K sector.
    plot_in_different_colors(Ks, V0s, correlation_lengths, path + name)



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

    return np.all(relative_error <= tolerance)


def ES_plot_no_fixed_sector(es, title, file_path='data/ES.png',unit_cell_size=16,ylim=False):
    """
    Plot the entanglement spectrum for a site.

    The function first identifies the charge sector with the largest total
    weight, then plots the entanglement energies in that sector as a function
    of momentum K.

    Parameters
    ----------
    es : list
        Entanglement spectrum data. Each entry is expected to contain charge
        sector information and corresponding entanglement energies.
    title: str, has title label
    file_path : str, optional
        Path where the generated plot should be saved.
    param: str,
            refers to a choice of N, Sz sector we are choosing. we always fix N,
            but sometimes allow Sz to change
    unit_cell_size: int,
            gives periodicity of unit cell so system can tell between even and odd
    """
    ##TODO: fix so that periodicity is correct.

    # For each charge sector, compute a quantity related to its total weight.
    # sec[0] contains the charge-sector labels.
    # sec[1] contains the entanglement energies.
    minEE = np.array([
        list(sec[0]) + [np.sum(np.exp(-2 * sec[1]))]
        for sec in es
    ])
    


    # Store points to plot as [momentum, entanglement_energy].
  
    pts = []

    

    for sec in es:
    
            pts.extend([[sec[0][0], p] for p in sec[1]])
    
    # Convert list of plot points to a NumPy array.
    pts = np.array(pts)

    # If no points were found, create an empty array with the correct shape.
    if len(pts) == 0:
        pts = np.zeros((0, 2), dtype=int)

    # Define a rectangular marker used for plotting the entanglement levels.
    wh, wv = 1, 0.01

    plt_kwargs = {
        'markersize': 10,
        'color': 'navy',
        'marker': [(-wh, -wv), (wh, -wv), (wh, wv), (-wh, wv)],
        'linewidth': 0.
    }


    plt.figure()
    # Plot entanglement energy against momentum K.
    plt.plot(pts[:, 0], pts[:, 1], **plt_kwargs)

    # Label axes.
    plt.xlabel(r'$K$', fontsize=18)
    plt.ylabel(r'$E_i$', fontsize=18, rotation='horizontal')

    # Add a title containing the supplied parameter values.
    plt.title(title)

    if ylim:
        plt.ylim(top=ylim)

    # Save the figure and close it so future plots do not overlap.
    plt.savefig(file_path, dpi=400)
    

    return pts



if __name__ == "__main__":

    # Directory containing the infinite-DMRG data to analyze.
    path = "/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/Landau_level_2_bands_V0_perturbation/"

    # Plot correlation length against V0, grouped by K sector.
    plot_energy_against_V0(path)

    # Uncomment this to plot entanglement spectra instead.
    # plot_entanglement_spectra(path)

    # TODO: Move exploratory plotting and analysis into a Jupyter notebook.
