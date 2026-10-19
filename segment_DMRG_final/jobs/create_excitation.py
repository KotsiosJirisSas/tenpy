import numpy as np
import os
import math

path = "/mnt/users/dperkovic/DMRG_modified/infinite_DMRG_data/nu=2_3_antidot_project/"
path_save = '/mnt/users/dperkovic/DMRG_modified/segment_DMRG_results/nu=2_3_antidot_project/excitation/'

files = []
for name in os.listdir(path):
    if 'MPS_data' in name:
        files.append(name[:-4])

# allowed (K, N) residues mod 3:  N = 0 -> K = 0,  N = 2 (e.g. 2, -1) -> K = 0,  N = 1 (e.g. 1, -2) -> K = 2
# added_sites = (-N) mod 3  (3 if that is 0), so that added_sites + N is a multiple of 3
charges = [[3*k, -1] for k in range(-5, 5)] + [[-1 + 3*k, 1] for k in range(-5, 5)]

charges = [[0, 0]]+[[3*k, -1] for k in range(-5, 5)] + [[-1 + 3*k, 1] for k in range(-5, 5)]

charges += [[3*k, 2] for k in range(-5, 5)] + [[2 + 3*k, -2] for k in range(-5, 5)]

L = 141
half = 60          # must be a multiple of 3 (the driver adds half*N to K)
U0 = 0.0
chi = 2000
x0 = 0

assert half % 3 == 0

def allowed(K, N):
    return (N % 3 == 0 and K % 3 == 0) or (N % 3 == 2 and K % 3 == 0) or (N % 3 == 1 and K % 3 == 2)

with open('jobs.sh', 'w') as f:
    for file_load in files:
        if not ('chi=3000' in file_load and 'MPS_data' in file_load and 'd=6' in file_load):
            continue
        print(file_load)
        for charge in charges:
            added_sites = (-charge[1]) % 3 or 3      # N=0 -> 3, N=+2 -> 1, N=-2 -> 2, N=-1 -> 1, N=+1 -> 2
            assert allowed(charge[0], charge[1]), ('K, N residues not allowed', charge)
            assert (added_sites + charge[1]) % 3 == 0, (added_sites, charge)
            file_load_full = path + file_load
            file_save = (path_save + file_load + 'EXCITATION_charge=' + str(charge[0]) + '_' + str(charge[1])
                         + '_' + '_L=' + str(L) + '_U=' + str(U0) + '_added_sites=' + str(added_sites)
                         + '_chi_seg=' + str(chi) + '_interface_position=' + str(half))
            f.write('addqueue  -c "10 min" -m 24 /usr/bin/python3 '
                    '/mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/multilayer_segment_final_exc.py '
                    + str(added_sites) + " " + file_load_full + ' ' + file_save + ' ' + str(L) + ' '
                    + str(half) + ' ' + str(chi) + ' ' + str(U0) + ' ' + str(x0) + ' '
                    + str(charge[0]) + " " + str(charge[1]) + '\n')