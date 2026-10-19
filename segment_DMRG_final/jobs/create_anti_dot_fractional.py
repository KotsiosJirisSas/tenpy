import numpy as np
import os
import math

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

# the sectors of YOUR iMPS: if sectors.txt (written by dump_sectors.py) is in the folder you run this from, it is used
# instead of the table above. Nothing else in this file is changed.
if os.path.exists('sectors.txt'):
    SECTORS = [tuple(int(x) for x in line.split()) for line in open('sectors.txt') if line.strip()]
    print('SECTORS read from sectors.txt:', len(SECTORS), 'sectors')
else:
    print('sectors.txt not found: using the SECTORS table in this file (', len(SECTORS), 'sectors )')

UNIT_CELL = 3


def n_left_for(added, half, K_driver, N=0):
    """Window sites to the LEFT of the interface. Mirrors the driver + QH_system exactly (unchanged):
         driver:  window_left='half'  if  int(added/2) < interface_position  and  shift[0] > 0
                  with shift[0] = first + interface_position*second   (= K_driver here)
         QH_system('half'): n_right = min(added, unit_cell*ceil(added/(2*unit_cell))), n_left = added - n_right
    In the half-half case n_right is a multiple of 3 and any remainder of a fractional-charge window sits on
    the left; otherwise the whole window (and the remainder) is on the right."""
    if int(added/2) < half and K_driver > 0 and N % 3 == 0:   # fractional N (N mod 3 != 0): whole window on the right
        return added - min(added, UNIT_CELL*math.ceil(added/(2*UNIT_CELL)))
    return 0


def reachable(half, added, K, N):
    n_left = n_left_for(added, half, K + half*N, N)
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
    """Smallest window that matches, plus `margin` (a multiple of 3, so the residue mod 3 is kept).
    ALL window sizes are tried: integer charge needs added = 0 mod 3, charge N = 1, 2 (mod 3) needs
    added = 2, 1 (mod 3) -- that is what puts the extra e/3 into the window."""
    for added in range(1, 200):
        if reachable(half, added, K, N):
            added += margin
            if reachable(half, added, K, N):
                return added
    raise ValueError('no window up to 200 sites reaches K=%d N=%d at half=%d' % (K, N, half))


def first_and_added(half, Kphys, second):
    """For integer charge first = -K_SCALE*Kphys is always allowed. For fractional charge K - K_bg is
    itself shifted by 1/3 or 2/3, so first must be one of -K_SCALE*Kphys + {0, +1, -1}; take the one
    with the smallest window."""
    best = None
    for d in (0, 1, -1):
        first = -K_SCALE*Kphys + d
        try:
            a = min_added(half, first, second)
        except ValueError:
            continue
        if best is None or a < best[1]:
            best = (first, a)
    if best is None:
        raise ValueError('no first in -3K+{0,1,-1} reachable for K-K_bg=%d, second=%d, half=%d' % (Kphys, second, half))
    return best


# ---------------------------------------------------------------------------------------------
# fractional-N scan near the minimum (x0=51)
#   integer N:  half=63 -> 138, half=60 -> 140;  second=3 -> N-1
#   charge shift: second +1 -> N-1/3, +2 -> N-2/3, -1 -> N+1/3, -2 -> N+2/3
# K windows interpolated from the x0=51 integer-N minima (136: ~-35, 137: ~-20, 138: ~-15, 139: ~0)
# ---------------------------------------------------------------------------------------------
K_SCALE = 3      # first = -K_SCALE * (K - K_bg)  (+0/±1 for fractional charge, see first_and_added)
K_STEP = 6       # step in K - K_bg (3 = finest)
SKIP_EXISTING = True   # don't rewrite jobs whose result file is already in path_save

#        label    : (half, second, Kmin, Kmax)      K = K - K_bg
SCAN = {'136+2/3': (63,  4, -75, 15),    # 137 - 1/3
        '137+1/3': (63,  2, -65, 20),    # 137 + 1/3
        '137+2/3': (63,  1, -60, 25),    # 138 - 1/3
        '138+1/3': (63, -1, -55, 30),    # 138 + 1/3
        '138+2/3': (60,  4, -50, 40),    # 139 - 1/3
        '139+1/3': (60,  2, -45, 50)}    # 139 + 1/3

jobs = []   # (N label, K - K_bg, half, second)
for Nlab, (half, second, kmin, kmax) in SCAN.items():
    for Kphys in range(kmin, kmax + 1, K_STEP):
        jobs.append((Nlab, Kphys, half, second))

Ls = [270]
Us = [4.0]
chis = [1000]
x0s = [51]

n_written = n_skipped = 0
with open('jobs.sh', 'w') as f:
    for file_load in files:
        if not ('chi=3000' in file_load and 'MPS_data' in file_load and 'd=6' in file_load):
            continue
        print(file_load)
        for x0 in x0s:
            for Nlab, Kphys, half, second in jobs:
                for L in Ls:
                    for U0 in Us:
                        for chi in chis:
                            try:
                                first, added_sites = first_and_added(half, Kphys, second)
                            except ValueError as e:
                                print('SKIP N=%s K-K_bg=%d: %s' % (Nlab, Kphys, e))
                                continue
                            n_left = n_left_for(added_sites, half, first + half*second, second)
                            start = half + added_sites - n_left
                            if L - start < 30:
                                print('WARNING (job still written): N=%s K-K_bg=%d half=%d added=%d -> right piece starts at %d, only %d sites left in L=%d'
                                      % (Nlab, Kphys, half, added_sites, start, L - start, L))
                            file_load_full = path + file_load
                            file_save = (path_save + file_load + '_charge=' + str(first) + '_' + str(second) + "_"
                                         + '_L=' + str(L) + '_U=' + str(U0) + '_added_sites=' + str(added_sites)
                                         + '_chi_seg=' + str(chi) + '_interface_position=' + str(half) + '_x0=' + str(x0)
                                         + '_width=15')
                            if SKIP_EXISTING and (os.path.exists(file_save) or os.path.exists(file_save + '.pkl')):
                                n_skipped += 1
                                continue
                            f.write('addqueue  -c "7 hour" -m 17 /usr/bin/python3 '
                                    '/mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/multilayer_segment_final.py '
                                    + str(added_sites) + " " + file_load_full + ' ' + file_save + ' ' + str(L) + ' '
                                    + str(half) + ' ' + str(chi) + ' ' + str(U0) + ' ' + str(x0) + ' '
                                    + str(first) + " " + str(second) + '\n')
                            n_written += 1

print('%d jobs written to jobs.sh, %d skipped (already done)' % (n_written, n_skipped))