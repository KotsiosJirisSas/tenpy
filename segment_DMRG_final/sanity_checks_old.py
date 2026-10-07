"""
Sanity check of the particle number of the merged initial state psi_total.
"""
import numpy as np


def bulk_filling_per_site(model_par):
    """
    Average number of particles per site from the root configuration.
    One site is one orbital of one layer, so (particles) = (this number) * (number of sites).
    For one layer and root_config = [1,0,1] this is 2/3.
    """
    return float(np.mean(np.asarray(model_par['root_config'], dtype=float)))


def sanity_check_filling(psi, model_par, L, half, added_sites, shift, type_calculation, model_par_R=None, tol=0.1):
    """
    psi:            merged initial MPS (psi_total)
    model_par:      model parameters of the left state (the root configuration gives its filling)
    model_par_R:    model parameters of the right state, only if its root configuration differs (interface)
    L, half, added_sites: as in run_excitation_dmrg (L = number of sites of the segment)
    shift:          the charge shift (K, N) or (K, n_1, ..., n_nlayer) passed on the command line
    tol:            tolerance for the comparisons

    vacuum_interface:  the left side is empty, so the right piece holds nu_R*(L-half) particles. The total number of
                       particles must be
                           nu_R*(L-half) + N_shift ,   N_shift = sum of the particle number entries of the shift.
                       For shift = 0 this is nu_R*(L-half) and the two numbers must be equal.
                       For shift != 0 the printout also says whether the difference equals +N_shift or -N_shift
                       (this tells the sign convention).
    other modes:       the added sites are random, so only the table of the three regions is printed.
    returns a dict with the numbers.
    """
    filling = np.asarray(psi.expectation_value("nOp"), dtype=float)
    assert len(filling) == L, (len(filling), L)
    nu_L = bulk_filling_per_site(model_par)
    nu_R = bulk_filling_per_site(model_par_R) if model_par_R is not None else nu_L
    shift = np.atleast_1d(np.asarray(shift))
    shift_N = float(np.sum(shift[1:])) if len(shift) > 1 else 0.0

    n_left, n_add, n_right = half, added_sites, L - half - added_sites
    measured = [filling[:half].sum(), filling[half:half + added_sites].sum(), filling[half + added_sites:].sum()]
    total = float(filling.sum())
    out = dict(total=total, left=measured[0], added=measured[1], right=measured[2])

    print('-' * 100)
    print('SANITY CHECK OF THE FILLING   (mode: ' + type_calculation + ', shift: ' + str(shift.tolist()) + ')')
    print('filling per site from the root configuration: left %.6f, right %.6f' % (nu_L, nu_R))

    if type_calculation == 'vacuum_interface':
        expected_base = nu_R * n_right
        diff = total - expected_base
        out.update(expected_base=expected_base, difference=diff)
        print('total number of particles:                 %.6f' % total)
        print('right piece, nu_R*(L-half) = %.6f * %d =  %.6f' % (nu_R, n_right, expected_base))
        print('difference total - nu_R*(L-half):          %.6f' % diff)
        if np.all(shift == 0):
            ok = abs(diff) < tol
            out['ok'] = bool(ok)
            print('shift = 0: the two numbers must be equal ->', 'OK' if ok else 'MISMATCH')
        else:
            out['equals_plus_shift_N'] = bool(abs(diff - shift_N) < tol)
            out['equals_minus_shift_N'] = bool(abs(diff + shift_N) < tol)
            print('N component of the shift: %.3f | difference = +N_shift: %s | difference = -N_shift: %s'
                  % (shift_N, out['equals_plus_shift_N'], out['equals_minus_shift_N']))
    else:
        print('region            sites   measured   nu*sites')
        print('left piece     %8d %10.4f %10.4f' % (n_left, measured[0], nu_L * n_left))
        print('added sites    %8d %10.4f   (random, unconstrained)' % (n_add, measured[1]))
        print('right piece    %8d %10.4f %10.4f' % (n_right, measured[2], nu_R * n_right))
        print('total                   %10.4f' % total)
    print('-' * 100)
    return out
