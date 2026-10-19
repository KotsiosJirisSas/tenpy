#!/mnt/users/dperkovic/myenv/bin/python
#source /mnt/users/dperkovic/myenv/bin/activate
"""Code for producing DMRG graph  --  scan version of create_MPO_graph_2_3.py

Same model as before (0LL, double gated Coulomb d=6, root config [1,0,1], slycot exponential fit).
What is new:
   * --eps   tolerance of the exponential fit                         (default 1e-5 = old value)
   * --xiK   hint for the real-space range of the interaction         (default 2.5  = old value)
   * --maxK  alternative to --xiK: choose xiK so that the interaction range is exactly this many orbitals.
             (maxK = ceil(-ln(0.1*eps) * xiK / kappa), kappa = 2 pi / Lx.  Changing eps alone also changes maxK!)
   * the graph is written to a NEW folder; the file name carries eps, xiK and maxK
   * the achieved fit error ('norm err' of the Markov compression) is stored in the pickle under 'scan_info'

   * --exact  use the EXACT shift-register realisation inside the compression (sets Markov.USE_EXACT_SS_FROMV if the
              patched Markov.py is installed, otherwise swaps Markov.SS_fromV at runtime).  --legacy forces the old route.  Found with dissect_compression.py: control.tf2ss makes the realisation of long sequences
              inexact (error 5e-5 at length 83, 47 at length 98), which then ruins the compression.  With --exact the
              same blocks compress to ~5e-8.  Exact graphs get '_exact' in the file name.

NOTE: WITHOUT --exact, maxK above ~85 gives garbage (norm err >> norm). A warning is printed.

usage:
    python3 create_mpo_graph_scan_2_3.py 15                          # old graph (eps=1e-5, xiK=2.5 -> maxK=83)
    python3 create_mpo_graph_scan_2_3.py 15 --eps 1e-4 --maxK 83
    python3 create_mpo_graph_scan_2_3.py 15 --eps 1e-6 --xiK 2.5 --exact
"""
import sys
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy')

import os
import argparse
import pickle
import numpy as np
from tenpy.models import multilayer_qh_DP_final as mod

np.set_printoptions(linewidth=np.inf, precision=7, threshold=np.inf, suppress=False)

# ------------------------------------------------------------------------------------------
# where the scan graphs go (NEW folder, the old data_load folder is not touched)
# ------------------------------------------------------------------------------------------
PATH_SAVE = '/mnt/users/dperkovic/DMRG_modified/data_load/nu=2_3_antidot_project_idmrg_test/'
MAXK_SAFE = 85      # above this the compression failed in the tests (norm err >> norm)


def max_K(eps, xiK, Lx):
    """the interaction range the model code will use (see make_Vmk in multilayer_qh_DP_final.py)"""
    kappa = 2 * np.pi / Lx
    return int(np.ceil(-np.log(0.1 * eps) * max(1., xiK) / kappa))


def xiK_for_maxK(eps, maxK, Lx):
    """xiK that gives exactly this maxK"""
    kappa = 2 * np.pi / Lx
    return (maxK - 0.5) * kappa / (-np.log(0.1 * eps))


def SS_fromV_exact(V, shiftby_1=True, ignore_r0=None):
    """FALLBACK copy of Markov.SS_fromV_exact, used only if the patched Markov.py is NOT installed
    (then it is swapped in at runtime by use_exact_realization).

    Same contract as Markov.SS_fromV: V[a][b][r] -> state space with D = 0 and C A^r B = V[:, :, r].
    control.tf2ss (the legacy route) is inexact for long sequences; a FIR sequence has an exact shift-register
    realisation:   A = shift,  B = (1, 0, ...),  C = (V_0, V_1, ...).
    ignore_r0 as in the legacy code: V_0 := V_1 (in place), or 0 if the sequence has length 1.
    """
    import control
    if not shiftby_1:
        raise NotImplementedError('only shiftby_1=True is used by compute_ABC')
    V = np.asarray(V)
    if V.ndim == 1:
        V = V.reshape(1, 1, -1)
    p, q, l = V.shape
    zero = np.zeros((p, q), dtype=bool)
    if ignore_r0 is not None:
        ign = np.array(ignore_r0, dtype=bool)
        for a in range(p):
            for b in range(q):
                if ign[a][b]:
                    if l == 1:
                        zero[a, b] = True
                    else:
                        V[a, b, 0] = V[a, b, 1]
    n = l * q
    A = np.zeros((n, n), dtype=V.dtype)
    for k in range(1, l):
        A[k * q:(k + 1) * q, (k - 1) * q:k * q] = np.eye(q)
    B = np.zeros((n, q), dtype=V.dtype)
    B[:q, :] = np.eye(q)
    C = np.hstack([V[:, :, k] for k in range(l)])
    if zero.any():
        C = C.copy()
        for a in range(p):
            for b in range(q):
                if zero[a, b]:
                    C[a, b::q] = 0
    return control.StateSpace(A, B, C, np.zeros((p, q), dtype=V.dtype), 1.)


def use_exact_realization(on=True):
    """Switch the compression to the exact (on=True) or the legacy tf2ss (on=False) realisation.
    With the patched Markov.py this sets Markov.USE_EXACT_SS_FROMV.  With the original Markov.py the exact function
    is swapped in at runtime (on=False then simply means: leave the original)."""
    import tenpy.algorithms.Markov as Markov
    if hasattr(Markov, 'USE_EXACT_SS_FROMV'):
        Markov.USE_EXACT_SS_FROMV = bool(on)
        print('Markov.USE_EXACT_SS_FROMV =', Markov.USE_EXACT_SS_FROMV, '(patched Markov.py)')
    elif on:
        Markov.SS_fromV = SS_fromV_exact
        print('Markov.SS_fromV replaced at runtime by the exact shift-register realisation (original Markov.py)')
    else:
        print('legacy tf2ss realisation (original Markov.py)')


def exact_active():
    """is the exact realisation currently active?"""
    import tenpy.algorithms.Markov as Markov
    if hasattr(Markov, 'USE_EXACT_SS_FROMV'):
        return bool(Markov.USE_EXACT_SS_FROMV)
    return getattr(Markov.SS_fromV, '__name__', '') == 'SS_fromV_exact'


def graph_stem(Lx, d, mu, eps, xiK, maxK, exact=False):
    """file name (without .pkl). create_idmrg_2_3_jobs.py reads eps, xiK, maxK (and _exact) back from it."""
    return '0LL_d=' + str(d) + '_Lx=' + str(Lx) + '_coulumb_double_gated_mu=' + str(mu) \
           + '_eps=%g_xiK=%g_maxK=%d' % (eps, xiK, maxK) + ('_exact' if exact else '')


def make_iMPO(model_par):
    """returns the MPO graph and a dictionary with the quality of the exponential fit"""
    print("Start model in Old Tenpy", ".." * 10)
    M = mod.QH_model(model_par, '')
    print("Old code finished producing MPO graph", ".." * 10)
    G = M.MPOgraph
    info = {}
    try:
        MS = M.Vmk_MS['MS']
        info['norm_err'] = float(MS.info['total_norm_err'])     # the 'norm err' of the log
        info['tol'] = float(MS.info['ABC_tol'])
        info['input_norm'] = float(MS.info['2norm'])
    except Exception as e:
        print('could not read the fit quality from the Markov system:', e)
        info['norm_err'] = info['tol'] = info['input_norm'] = float('nan')
    info['maxK'] = int(getattr(M, 'maxK', -1))
    info['maxM'] = int(getattr(M, 'maxM', -1))
    return G, info


def build_model_par(Lx, eps, xiK, d=6):
    xi = 4                       # only used by the Gaussian branch
    pref = 1.0
    vecs = [('down_K-', 0)]
    layers = vecs
    species = list(set([name[0] for name in vecs]))
    root_config_ = np.array([1, 0, 1])

    # isotropic Nlayer interaction
    V = {'eps': eps, 'xiK': xiK}
    V0 = 0.0
    d1 = 3
    z = 0

    if d < 10 or z != 0:
        V['coulomb_blgated'] = {}
        for i, layer in enumerate(species):
            for j, layer2 in enumerate(species):
                if i <= j:
                    key = (layer, layer2)
                    if i == j:
                        V['coulomb_blgated'][key] = {'v': 1., 'z': z, 'd': d, 'epsA': 1, ' betaA': 1, 'V0': V0 / 2}
                    else:
                        V['coulomb_blgated'][key] = {'v': 2., 'z': z, 'd': d, 'epsA': 1, ' betaA': 1, 'V0': V0}
    else:
        V['GaussianCoulomb'] = {}
        for i, layer in enumerate(species):
            for j, layer2 in enumerate(species):
                if i <= j:
                    key = (layer, layer2)
                    if i == j:
                        V['GaussianCoulomb'][key] = {'v': 1., 'xi': xi, 'V0': V0 / 2}
                    else:
                        V['GaussianCoulomb'][key] = {'v': 2., 'xi': xi, 'V0': V0}
    print(V)

    model_par = {
        'verbose': 2,
        'root_config': root_config_,
        'exp_approx': 'slycot',
        'Vs': V,
        'cons_K': True,
        'cons_C': 'total',
        'Lx': Lx,
    }
    model_par['layers'] = layers
    model_par['pref'] = pref
    mu = 0.0
    model_par['mu'] = -np.array([mu])
    return model_par, mu


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('Lx', type=float)
    ap.add_argument('--eps', type=float, default=1e-5, help='Vs["eps"]  (default 1e-5, the old value)')
    ap.add_argument('--xiK', type=float, default=2.5, help='Vs["xiK"]  (default 2.5, the old value)')
    ap.add_argument('--maxK', type=int, default=None, help='choose xiK so that maxK is exactly this (overrides --xiK)')
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument('--exact', action='store_true', help='use the exact shift-register realisation inside the compression')
    grp.add_argument('--legacy', action='store_true', help='force the legacy tf2ss realisation (matters once Markov.py defaults to exact)')
    ap.add_argument('--d', type=float, default=6, help='gate distance (default 6)')
    ap.add_argument('--outdir', type=str, default=PATH_SAVE)
    args = ap.parse_args()

    Lx = round(args.Lx, 3)
    d = int(args.d) if float(args.d).is_integer() else args.d

    xiK = args.xiK
    if args.maxK is not None:
        xiK = float(round(xiK_for_maxK(args.eps, args.maxK, Lx), 5))
        if xiK < 1.0:
            print('WARNING: --maxK %d needs xiK=%.3f < 1 and the code clamps xiK to >= 1' % (args.maxK, xiK))
    maxK = max_K(args.eps, xiK, Lx)
    if args.maxK is not None and maxK != args.maxK:
        print('WARNING: asked for maxK=%d but eps=%g xiK=%g gives maxK=%d' % (args.maxK, args.eps, xiK, maxK))
    if args.exact:
        use_exact_realization(True)
    elif args.legacy:
        use_exact_realization(False)
    exact = exact_active()               # the real state (library default included), used for names and bookkeeping
    print('exact realisation active:', exact)
    if maxK > MAXK_SAFE and not exact:
        print('WARNING: maxK=%d > %d.  In the tests the tf2ss-based compression failed for such long sequences '
              '(norm err >> norm). Use --exact, or check "norm err / total norm" below.' % (maxK, MAXK_SAFE))
    print('eps=%g  xiK=%g  ->  maxK=%d' % (args.eps, xiK, maxK))

    model_par, mu = build_model_par(Lx, args.eps, xiK, d)
    print(model_par)

    os.makedirs(args.outdir, exist_ok=True)
    name_save = os.path.join(args.outdir, graph_stem(Lx, d, mu, args.eps, xiK, maxK, exact) + '.pkl')

    graph, fit_info = make_iMPO(model_par)

    # 'model_par' and 'graph' are exactly what QHE_run_from_graph*.py reads; 'scan_info' is only bookkeeping
    scan_info = {'eps': args.eps, 'xiK': xiK, 'maxK': maxK, 'Lx': Lx, 'd': d, 'exact': bool(exact)}
    scan_info.update(fit_info)
    data = {'model_par': model_par, 'graph': graph, 'scan_info': scan_info}
    with open(name_save, 'wb') as f:
        pickle.dump(data, f)
    print('saved graph to', name_save)
    print('FIT QUALITY  norm_err = %.4e   input_norm = %.4f   (tol = %.3e)   exact=%s' %
          (fit_info['norm_err'], fit_info['input_norm'], fit_info['tol'], exact))
    # small sidecar so that the job maker can refuse graphs whose fit failed without loading the pickle
    import json
    with open(name_save[:-4] + '.fit.json', 'w') as f:
        json.dump({'norm_err': fit_info['norm_err'], 'input_norm': fit_info['input_norm'], 'tol': fit_info['tol'],
                   'maxK': maxK, 'exact': bool(exact)}, f)
