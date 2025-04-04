"""Tuning through the phase transition of the transverse field Ising model.

This example uses DMRG to find the ground state of the transverse field Ising model when tuning
through the phase transition by changing the field `g`.
It plots a few observables in the end.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.models.tf_ising import TFIChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import matplotlib.pyplot as plt
import scipy.sparse as sparse
from scipy.sparse.linalg import eigsh
import warnings
import scipy.integrate

def infinite_gs_energy(J, g):
    """For comparison: Calculate groundstate energy density from analytic formula.

    The analytic formula stems from mapping the model to free fermions, see P. Pfeuty, The one-
    dimensional Ising model with a transverse field, Annals of Physics 57, p. 79 (1970). Note that
    we use Pauli matrices compared this reference using spin-1/2 matrices and replace the sum_k ->
    integral dk/2pi to obtain the result in the N -> infinity limit.
    """
    def f(k, lambda_):
        return np.sqrt(1 + lambda_**2 + 2 * lambda_ * np.cos(k))

    E0_exact = -g / (J * 2. * np.pi) * scipy.integrate.quad(f, -np.pi, np.pi, args=(J / g, ))[0]
    return E0_exact


def run_finite(gs):
    
    print("running for gs = ", gs)
    L = 60
    model_params = dict(L=L, J=1., g=gs[0], bc_MPS='finite', conserve=None)
    chi = 100
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'update_env': 5,
        'start_env': 5,
        'max_E_err': 0.0001,
        'max_S_err': 0.0001,
        'max_sweeps': 100,  # NOTE: this is not enough to fully converge at the critical point!
        'mixer': False
    }

    M = TFIChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), (["up", "down"] * L)[:L], M.lat.bc_MPS)

    engine = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    np.set_printoptions(linewidth=120)

    corr_length = []
    fidelity = []
    Sz = []
    E = []
    S = []
    SxSx = []
    old_psi = None
    for g in gs:
        print("-" * 80)
        print("g =", g)
        print("-" * 80)
        model_params['g'] = g
        M = TFIChain(model_params)
        engine.init_env(model=M)  # (re)initialize DMRG environment with new model
        # this uses the result from the previous DMRG as first initial guess
        E0, psi = engine.run()
        E.append(E0/L)
        # psi is modified by engine.run() and now represents the ground state for the current `g`.
        
        S.append(psi.entanglement_entropy()[L//2])
      
        Sz.append(np.sum(psi.expectation_value('Sigmax'))/L)
        print("<Sigmaz>", Sz[-1])
      
        if old_psi is not None:
            fidelity.append(np.abs(old_psi.overlap(psi, understood_infinite=True)))
            print("fidelity", fidelity[-1])
        old_psi = psi.copy()
        dmrg_params['start_env'] = 0  # (some of) the parameters are read out again
    results = {
        'model_params': model_params,
        'dmrg_params': dmrg_params,
        'gs': gs,
       
        'fidelity': np.array(fidelity),
        'Sz': np.array(Sz),
        'E': np.array(E),
        'S': np.array(S),
        
    }

    return results

def run_bunch_fdmrg(Js):
    Es=[]
    for J in Js:
        print('-'*80)
        print('J=',J)
        print('-'*80)
        info=run_finite(J)
        print('information of fDMRG',info.keys())
        Es.append(info['E'])
    
    plt.figure()
    plt.title('fDMRG for N=18 sites')
    plt.xlabel('Jz')
    plt.scatter(Js,np.array(Es)/18)
    plt.ylabel('E/N')
    plt.savefig('Energies_plot_fDMRG'+'.png')




def run_infinite(gs):
    print("running for gs = ", gs)
    L = 2
    model_params = dict(L=L, J=1., g=gs[0], bc_MPS='infinite', conserve=None)
    chi = 100
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'update_env': 5,
        'start_env': 5,
        'max_E_err': 0.0001,
        'max_S_err': 0.0001,
        'max_sweeps': 150,  # NOTE: this is not enough to fully converge at the critical point!
        'mixer': False
    }

    M = TFIChain(model_params)
    print('bc:')
    print(M.lat.__dict__)
    quit()
    psi = MPS.from_product_state(M.lat.mps_sites(), (["up", "down"] * L)[:L], M.lat.bc_MPS)

    engine = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    np.set_printoptions(linewidth=120)

    corr_length = []
    fidelity = []
    Sz = []
    E = []
    S = []
    SxSx = []
    old_psi = None
    Sx=[]
    for g in gs:
        print("-" * 80)
        print("g =", g)
        print("-" * 80)
        model_params['g'] = g
        M = TFIChain(model_params)
        engine.init_env(model=M)  # (re)initialize DMRG environment with new model
        # this uses the result from the previous DMRG as first initial guess
        E0, psi = engine.run()
        E.append(E0)
        # psi is modified by engine.run() and now represents the ground state for the current `g`.
        S.append(psi.entanglement_entropy()[0])
        corr_length.append(psi.correlation_length(tol_ev0=1.e-3))
        print("corr. length", corr_length[-1])
        Sz.append(psi.expectation_value('Sigmaz'))
        Sx.append(psi.expectation_value('Sigmax'))
        print("<Sigmax>", Sx[-1])
        SxSx.append(psi.correlation_function("Sigmax", "Sigmax", [0], 20)[0, :])
        print("<Sigmax_0 Sigmax_i>", SxSx[-1])
        
        if old_psi is not None:
            fidelity.append(np.abs(old_psi.overlap(psi, understood_infinite=True)))
            print("fidelity", fidelity[-1])
        old_psi = psi.copy()
        dmrg_params['start_env'] = 0  # (some of) the parameters are read out again
    results = {
        'model_params': model_params,
        'dmrg_params': dmrg_params,
        'gs': gs,
        'corr_length': np.array(corr_length),
        'fidelity': np.array(fidelity),
        'Sx': np.array(Sx),
        'E': np.array(E),
        'S': np.array(S),
        'SxSx': np.array(SxSx),
        'eval_transfermatrix': np.exp(-1. / np.array(corr_length)),
    }
    return results


def plot(results, filename):
    fig, axes = plt.subplots(2, 3, gridspec_kw=dict(wspace=0.3, hspace=0.3), figsize=(15, 10))
    gs = np.array(results['gs'])
    for ax, key in zip(axes.flatten(),
                       ['E', 'S', 'SxSx', 'Sx', 'eval_transfermatrix', 'fidelity']):
        if key == 'fidelity':
            ax.plot(0.5 * (gs[1:] + gs[:-1]), np.abs(results[key]), 'ro-')
        elif key == 'SxSx':
            cm = plt.cm.get_cmap("viridis")
           
            SxSx = results[key]
            max_j = SxSx.shape[1]
            for j in range(1, max_j):
                label = "j= {j:d}".format(j=j) if j in [1, max_j - 1] else None
                ax.plot(gs[x], SxSx[:, j][x], label=label, color=cm(j / max_j))
            ax.legend()
            
            pass
        elif key=='E':
            A=[]
            for g in gs:
                A.append(infinite_gs_energy(1, g))
            x=np.argsort(gs)
            ax.plot(gs[x],np.array(A)[x])
            
            ax.plot(gs[x],results[key][x],'ro-')
        else:
            ax.plot(gs[x], results[key][x], 'ro-')
        ax.set_xlabel('$g/J$')
        ax.set_ylabel(key)
        ax.set_xlim(min(gs), max(gs))
    plt.savefig(filename)
    print("saved to " + filename)

def plot_finite(results, filename):
    fig, axes = plt.subplots(2, 3, gridspec_kw=dict(wspace=0.3, hspace=0.3), figsize=(15, 10))
    gs = np.array(results['gs'])
    for ax, key in zip(axes.flatten(),
                       ['E', 'S',  'Sz', 'fidelity']):
        if key == 'fidelity':
            ax.plot(0.5 * (gs[1:] + gs[:-1]), np.abs(results[key]), 'ro-')
        elif key == 'SxSx':
            cm = plt.cm.get_cmap("viridis")
            """
            SxSx = results[key]
            max_j = SxSx.shape[1]
            for j in range(1, max_j):
                label = "j= {j:d}".format(j=j) if j in [1, max_j - 1] else None
                ax.plot(gs, SxSx[:, j], label=label, color=cm(j / max_j))
            ax.legend()
            """
            pass
        else:
            ax.plot(gs, results[key], 'ro-')
        ax.set_xlabel('$g/J$')
        ax.set_ylabel(key)
        ax.set_xlim(min(gs), max(gs))
    plt.savefig(filename)
    print("saved to " + filename)

if __name__ == "__main__":
    filename = 'Infinite_DMRG_tfi_phase_transition'
    import pickle
    import os.path
    
    gs = sorted(set(np.linspace(1.1, 1.5, 5)).union(set(np.linspace(0.95, 1.05, 11))))[::-1]
    gs=list(gs)
    gs.extend([0.5,0.6,0.7,0.8,0.9])
    gs=np.linspace(0.5,1.5,11)
    print(gs)
    gs=np.sort(gs)
    #gs=[2,3,5,4]
    #gs=np.linspace(1.1,1.7,10)
    #gs=np.linspace(0.5,1.2,10)[::-1]
    #gs = [0.8,0.9]
      
    results = run_infinite(gs)
    #plot(results, filename + '.png')
    results =run_finite(gs)
    filename = 'finite_DMRG_tfi_phase_transition'
    plot_finite(results, filename + '.png')
    #run_bunch_fdmrg(gs)
