import numpy as np
import os
import math
from itertools import product

path = "/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=2_3_antidot_project/"
path_save = '/mnt/users/dperkovic/DMRG_modified/segment_DMRG_results/nu=2_3_antidot_project/'

files = []
for name in os.listdir(path):
    if 'MPS_data' in name:
        files.append(name[:-4])

# ---------------------------------------------------------------------------------------------
# smallest window that can reach (K,N) -- same matching as patch_mps / vacuum_lhs_window
# SECTORS: (K, N) of the left-leg sectors of the iMPS at a unit-cell boundary (local coordinates, from the logs)
# ---------------------------------------------------------------------------------------------
SECTORS = [(-27, 0), (-27, 3), (-24, -3), (-24, 0), (-24, 3), (-21, -3), (-21, 0), (-21, 3), (-21, 6), (-18, -3), (-18, 0),
           (-18, 3), (-18, 6), (-15, -6), (-15, -3), (-15, 0), (-15, 3), (-15, 6), (-12, -6), (-12, -3), (-12, 0), (-12, 3),
           (-12, 6), (-9, -6), (-9, -3), (-9, 0), (-9, 3), (-6, -6), (-6, -3), (-6, 0), (-6, 3), (-3, -3), (-3, 0), (-3, 3),
           (0, -3), (0, 0), (0, 3), (3, -3), (3, 0), (3, 3), (6, -3), (6, 0), (6, 3), (9, -3), (9, 0), (9, 3), (12, -3),
           (12, 0), (12, 3), (15, -3), (15, 0), (18, 0)]

def n_left_for(added, half, K):
    """sites of the window to the left of the interface: half-half for K>0 (never more than the vacuum has), none for K<=0.
    Must be the same rule as params['window_left'] in the driver."""
    if K <= 0:
        return 0
    return min(added - 3*math.ceil(added/6), half - half % 3)

def reachable(half, added, K, N):
    n_left = n_left_for(added, half, K)
    n_right = added - n_left
    start = half + n_right
    x_first = half - n_left
    sum_right = sum(range(half, half + n_right))
    Kt = K + half*N                                   # the driver adds interface_position*N to K
    for Kl, Nl in SECTORS:
        Kw = (Kl + start*Nl) - Kt
        Nw = Nl - N
        if (Nw + 2*n_right) % 3 or (Kw + 2*sum_right) % 3:
            continue
        j = (Nw + 2*n_right)//3
        S = (Kw + 2*sum_right)//3
        if 0 <= j <= added and j*x_first + j*(j - 1)//2 <= S <= j*(x_first + added - 1) - j*(j - 1)//2:
            return True
    return False

def min_added(half, K, N, margin=3):
    """smallest window (multiple of 3) that matches, plus `margin`; the result is checked to match too"""
    for added in range(3, 200, 3):
        if reachable(half, added, K, N):
            added += margin
            assert reachable(half, added, K, N)
            return added
    raise ValueError('no window up to 200 sites reaches K=%d N=%d at half=%d' % (K, N, half))

# ---------------------------------------------------------------------------------------------
first = list(range(-800*3, 800*3 + 1, 27*3))

first = list(range(-2000*3, -800*3 + 1, 27*3*3))
second = [0,3]
charges = [[q1, q2] for q1, q2 in product(first, second)]
halves = [ 60]

Ls = [270]
Us = [4.0]
chis = [1000]
x0s = [51]

with open('jobs.sh', 'w') as f:
    for file_load in files:
        if not ('chi=3000' in file_load and 'MPS_data' in file_load and 'd=6' in file_load):
            continue
        print(file_load)
        for x0 in x0s:
            for charge in charges:
                for L in Ls:
                    for U0 in Us:
                        for chi in chis:
                            for half in halves:
                                added_sites = min_added(half, charge[0], charge[1])
                                # nothing is skipped; only a warning if the right piece is short or does not fit in L
                                start = half + added_sites - n_left_for(added_sites, half, charge[0])
                                if L - start < 30:
                                    print('WARNING (job still written): K=%d half=%d added=%d -> right piece starts at %d, only %d sites left in L=%d'
                                          % (charge[0], half, added_sites, start, L - start, L))
                                file_load_full = path + file_load
                                file_save = (path_save + file_load + '_charge=' + str(charge[0]) + '_' + str(charge[1]) + "_"
                                             + '_L=' + str(L) + '_U=' + str(U0) + '_added_sites=' + str(added_sites)
                                             + '_chi_seg=' + str(chi) + '_interface_position=' + str(half) + '_x0=' + str(x0)
                                             + '_width=15')
                                f.write('addqueue  -c "7 hour" -m 14 /usr/bin/python3 '
                                        '/mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/multilayer_segment_final.py '
                                        + str(added_sites) + " " + file_load_full + ' ' + file_save + ' ' + str(L) + ' '
                                        + str(half) + ' ' + str(chi) + ' ' + str(U0) + ' ' + str(x0) + ' '
                                        + str(charge[0]) + " " + str(charge[1]) + '\n')