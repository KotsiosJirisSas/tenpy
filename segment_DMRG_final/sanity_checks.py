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
                           nu_R*(L-half) - (N entries of the shift)/q ,   q = root_config.size
                       (the site charges are q*n - p, so one particle is q units of N charge; a negative N entry adds
                       particles, a positive one removes them). For shift = 0 this is nu_R*(L-half).
                       Prints OK / MISMATCH (tolerance tol; the initial MPS is not an exact N eigenstate, so expect
                       small fractional deviations).
    other modes:       the added sites are random, so only the table of the three regions is printed.
    returns a dict with the numbers.
    """
    filling = np.asarray(psi.expectation_value("nOp"), dtype=float)
    assert len(filling) == L, (len(filling), L)
    nu_L = bulk_filling_per_site(model_par)
    nu_R = bulk_filling_per_site(model_par_R) if model_par_R is not None else nu_L
    shift = np.atleast_1d(np.asarray(shift))
    # charges are relative to the root configuration: a site has charge q*n - p (n = occupation), so one particle
    # is q = root_config.size units of N charge. The N entries of the shift are in these units.
    q = np.asarray(model_par['root_config']).size
    shift_N = float(np.sum(shift[1:])) / q if len(shift) > 1 else 0.0       # in particles

    n_left, n_add, n_right = half, added_sites, L - half - added_sites
    measured = [filling[:half].sum(), filling[half:half + added_sites].sum(), filling[half + added_sites:].sum()]
    total = float(filling.sum())
    out = dict(total=total, left=measured[0], added=measured[1], right=measured[2])

    print('-' * 100)
    print('SANITY CHECK OF THE FILLING   (mode: ' + type_calculation + ', shift: ' + str(shift.tolist()) + ')')
    print('filling per site from the root configuration: left %.6f, right %.6f' % (nu_L, nu_R))

    if type_calculation == 'vacuum_interface':
        # the window (if any) is part of the sites after `half`: the total number does not depend on how it is filled
        expected_base = nu_R * (L - half)
        diff = total - expected_base
        out.update(expected_base=expected_base, difference=diff)
        print('total number of particles:                 %.6f' % total)
        print('nu_R*(L-half) = %.6f * %d =  %.6f' % (nu_R, L - half, expected_base))
        print('difference total - nu_R*(L-half):          %.6f' % diff)
        # the sign is confirmed by a run: a negative N entry of the shift adds particles, a positive one removes them
        expected_diff = -shift_N
        ok = abs(diff - expected_diff) < tol
        out.update(expected_difference=expected_diff, ok=bool(ok))
        print('N part of the shift in particles (N entries / q, q=%d): %.3f -> expected difference %.3f' % (q, shift_N, expected_diff))
        print('measured difference %.6f vs expected %.6f ->' % (diff, expected_diff), 'OK' if ok else 'MISMATCH')
    else:
        print('region            sites   measured   nu*sites')
        print('left piece     %8d %10.4f %10.4f' % (n_left, measured[0], nu_L * n_left))
        print('added sites    %8d %10.4f   (random, unconstrained)' % (n_add, measured[1]))
        print('right piece    %8d %10.4f %10.4f' % (n_right, measured[2], nu_R * n_right))
        print('total                   %10.4f' % total)
    print('-' * 100)
    return out