# TEST batch after the svd_min fix.
#   x0 = 51   : N = 137 ... 139 in steps of 1/3
#   x0 = 52.5 : N = 136 ... 138 in steps of 1/3
# a few K values each, L = 270, half = 63.
# Same window model and the same job line as create_anti_dot_fractional; only the scan is different.
import numpy as np
import os
import math

path = "/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=2_3_antidot_project/"
path_save = '/mnt/users/dperkovic/DMRG_modified/segment_DMRG_results/nu=2_3_antidot_project/'

files = []
for name in os.listdir(path):
    if 'MPS_data' in name:
        files.append(name[:-4])

SECTORS = [(-27, 0), (-27, 3), (-24, -3), (-24, 0), (-24, 3), (-21, -3), (-21, 0), (-21, 3), (-21, 6), (-18, -3), (-18, 0),
           (-18, 3), (-18, 6), (-15, -6), (-15, -3), (-15, 0), (-15, 3), (-15, 6), (-12, -6), (-12, -3), (-12, 0), (-12, 3),
           (-12, 6), (-9, -6), (-9, -3), (-9, 0), (-9, 3), (-6, -6), (-6, -3), (-6, 0), (-6, 3), (-3, -3), (-3, 0), (-3, 3),
           (0, -3), (0, 0), (0, 3), (3, -3), (3, 0), (3, 3), (6, -3), (6, 0), (6, 3), (9, -3), (9, 0), (9, 3), (12, -3),
           (12, 0), (12, 3), (15, -3), (15, 0), (18, 0)]

# the sectors of YOUR iMPS, if dump_sectors.py has written sectors.txt next to this script
if os.path.exists('sectors.txt'):
    SECTORS = [tuple(int(x) for x in line.split()) for line in open('sectors.txt') if line.strip()]
    print('SECTORS read from sectors.txt:', len(SECTORS), 'sectors')
else:
    print('sectors.txt not found: using the SECTORS table in this file (', len(SECTORS), 'sectors )')

UNIT_CELL = 3
K_SCALE = 3


def n_left_for(added, half, K_driver, N=0):
    """Window sites to the LEFT of the interface. Mirrors the driver + QH_system:
    fractional N (N mod 3 != 0) keeps the whole window on the right."""
    if int(added/2) < half and K_driver > 0 and N % 3 == 0:
        return added - min(added, UNIT_CELL*math.ceil(added/(2*UNIT_CELL)))
    return 0


def reachable(half, added, K, N):
    n_left = n_left_for(added, half, K + half*N, N)
    n_right = added - n_left
    start = half + n_right
    x_first = half - n_left
    sum_right = sum(range(half, half + n_right))
    Kt = K + half*N
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
    for added in range(1, 200):
        if reachable(half, added, K, N):
            added += margin
            if reachable(half, added, K, N):
                return added
    raise ValueError('no window up to 200 sites reaches K=%d N=%d at half=%d' % (K, N, half))


def first_and_added(half, Kphys, second):
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
# the test scan.  second = 2*(L-half) - 3*N  ->  N = (2*(L-half) - second)/3
#   half = 63, L = 270  ->  second = 414 - 3N
# ---------------------------------------------------------------------------------------------
L = 270
half = 63
U0 = 4.0
chi = 1000
SKIP_EXISTING = True          # don't rerun a job whose .pkl is already in path_save

SCAN = [
    # x0 ,  N from .. to (in thirds, Nq = 3N) , K-K_bg values
    (51,   411, 417, [-29, -23, -17]),        # N = 137 ... 139
    (52.5, 408, 414, [-17, -11,  -5]),        # N = 136 ... 138
]

n_written = n_skipped = 0
rows = []
with open('jobs.sh', 'w') as f:
    for file_load in files:
        if not ('chi=3000' in file_load and 'MPS_data' in file_load and 'd=6' in file_load):
            continue
        print(file_load)
        for x0, Nq_min, Nq_max, Kphys_list in SCAN:
            for Nq in range(Nq_min, Nq_max + 1):
                second = 2*(L - half) - Nq
                for Kphys in Kphys_list:
                    try:
                        first, added_sites = first_and_added(half, Kphys, second)
                    except ValueError as e:
                        print('SKIP x0=%s N=%d/3 K-K_bg=%d: %s' % (x0, Nq, Kphys, e))
                        continue
                    file_load_full = path + file_load
                    file_save = (path_save + file_load + '_charge=' + str(first) + '_' + str(second) + "_"
                                 + '_L=' + str(L) + '_U=' + str(U0) + '_added_sites=' + str(added_sites)
                                 + '_chi_seg=' + str(chi) + '_interface_position=' + str(half) + '_x0=' + str(x0)
                                 + '_width=15')
                    if SKIP_EXISTING and (os.path.exists(file_save) or os.path.exists(file_save + '.pkl')):
                        n_skipped += 1
                        continue
                    f.write('addqueue  -c "7 hour" -m 14 /usr/bin/python3 '
                            '/mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/multilayer_segment_final.py '
                            + str(added_sites) + " " + file_load_full + ' ' + file_save + ' ' + str(L) + ' '
                            + str(half) + ' ' + str(chi) + ' ' + str(U0) + ' ' + str(x0) + ' '
                            + str(first) + " " + str(second) + '\n')
                    n_written += 1
                    rows.append((x0, Nq/3.0, second, Kphys, first, added_sites))

print()
print('   x0        N  second   K-K_bg   charge K  added_sites')
for r in rows:
    print('%6s %8.3f %7d %8d %10d %12d' % r)
print()
print('%d jobs written to jobs.sh, %d skipped (already done)' % (n_written, n_skipped))