"""Calculate the correlation length of the transverse field Ising model for various h_z.

This example uses DMRG to find the ground state of the transverse field Ising model when tuning
through the phase transition by changing the field `hz`. It uses
:meth:`~tenpy.networks.mps.MPS.correlation_length` to extract the correlation length of the ground
state, and plots it vs. hz in the end.
"""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np

from tenpy.models.spins import SpinChain
from tenpy.networks.mps import MPS
from tenpy.algorithms import dmrg
import matplotlib.pyplot as plt


def run_infinite(Jzs):
    L = 2

    model_params = dict(L=L, Jx=0.0, Jy=0.0, Jz=1.0,hx=-Jzs[0], bc_MPS='infinite')#, conserve='Sz')
    chi = 300
    dmrg_params = {
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        },
        'update_env': 20,
        'start_env': 20,
        'max_E_err': 0.0001,
        'max_S_err': 0.0001,
        'mixer': False
    }

    M = SpinChain(model_params)
    psi = MPS.from_product_state(M.lat.mps_sites(), (["up", "down"] * L)[:L], M.lat.bc_MPS)

    engine = dmrg.TwoSiteDMRGEngine(psi, M, dmrg_params)
    np.set_printoptions(linewidth=120)
    corr_length = []
    energies=[]
    for Jz in Jzs:
        print("-" * 80)
        print('hz=',Jz)
        #print("Jz = {Jz:.4f}".format(Jz))
        print("-" * 80)
        model_params['hz'] = Jz
        M = SpinChain(model_params)
        engine.init_env(model=M)  # (re)initialize DMRG environment with new model
        # this uses the result from the previous DMRG as first initial guess
        E,fin=engine.run()
        # psi is modified by engine.run() and now represents the ground state for the current `Jz`.
        corr_length.append(psi.correlation_length(tol_ev0=1.e-3))
        energies.append(E)
        print("corr. length", corr_length[-1])
        print("<Sz>", psi.expectation_value('Sz'))
        dmrg_params['start_env'] = 0  # (some of) the parameters are read out again
    corr_length = np.array(corr_length)
    results = {
        'model_params': model_params,
        'dmrg_params': dmrg_params,
        'hzs': Jzs,
        'corr_length': corr_length,
        'eval_transfermatrix': np.exp(-1. / corr_length),
    'energies':energies}
    return results


def plot(results, filename):
    corr_length = results['corr_length']
    Jzs = results['hzs']
    plt.figure()
    plt.plot(Jzs, corr_length)
    plt.xlabel(r'$J_z/J_x$')
    plt.ylabel(r'$t = \exp(-\frac{1}{\xi})$')
    plt.savefig(filename)
    print("saved to " + filename)

    Es = results['energies']
    plt.figure()
    plt.plot(Jzs, Es)
    plt.xlabel(r'$J_z/J_x$')
    plt.ylabel(r'$Energy$')
    plt.savefig('energy_'+filename)
    print("saved to " + filename)

def run_finite(J):
    N = 16  # number of sites
    
    
    start={"L": N, "Jz": 1., "Jx": 0.,"Jx":0.,'hx':-J,  "bc_MPS": "finite"}#,"conserve":'Sz'}
    model = SpinChain(start)
    sites = model.lat.mps_sites()
    psi = MPS.from_product_state(model.lat.mps_sites(), (["up", "down"] * N)[:N], model.lat.bc_MPS)
    #psi = MPS.from_product_state(sites, ['up'] * N, "finite")
    dmrg_params = {"trunc_params": {"chi_max": 200, "svd_min": 1.e-10}, "mixer": True}
    info = dmrg.run(psi, model, dmrg_params)
    print("E =", info['E'])
    # E = -20.01638790048513
    print("max. bond dimension =", max(psi.chi))
    return info

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

if __name__ == "__main__":
    filename = 'xxz_corrlength.pkl'
    import pickle
    import os.path
    print('-'*80)
    print('Run finite DMRG')
    print('-'*80)
    Js=np.linspace(0.01,1.5,100)
    #run_bunch_fdmrg(Js)
    #quit()

    results = run_infinite(list(np.arange(4.0, 1.5, -0.25)) + list(np.arange(1.5,-1.5, -0.05)))
    name = 'xxz_corrlength.png'
    plot(results, name)