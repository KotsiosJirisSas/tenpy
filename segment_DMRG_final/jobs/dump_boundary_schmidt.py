# Checks the Schmidt values of the iMPS bonds, i.e. the bonds that end up at the fixed right boundary
# of the segment. Run once on the cluster (takes seconds):
#
#   python3 dump_boundary_schmidt.py '<name_load without .pkl>'
#
# What it tells you: for every bond of the 3-site unit cell, the smallest Schmidt value and how many
# are below the svd_min that the segment DMRG uses (1e-6 at the moment). If a bond has Schmidt values
# at or below 1e-6, the first DMRG sweep DISCARDS those states, the state's right leg then has fewer
# sectors than the right environment, and sweep 2 dies with 'incompatible LegCharge'.
import sys, pickle
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy')
import numpy as np

name_load = sys.argv[1]
with open(name_load + '.pkl', 'rb') as f:
    data = pickle.load(f)
psi = data['mps']

print('iMPS: L =', psi.L, '| bc =', psi.bc, '| chi =', psi.chi)
print()
print('bond |  dim |      min S |  #S<1e-6 | #S<1e-5 | #S<1e-4 |   sum S^2 | smallest 6 S')
for b in range(len(psi._S)):
    S = np.sort(np.asarray(psi._S[b], float))[::-1]
    if S.size == 0:
        continue
    print('%4d | %4d | %10.3e | %8d | %7d | %7d | %9.6f | %s'
          % (b, S.size, S.min(), int((S < 1e-6).sum()), int((S < 1e-5).sum()),
             int((S < 1e-4).sum()), float(np.sum(S**2)), np.array2string(S[-6:], precision=2)))
print()
print('The bond that sits at the right boundary of the segment is set by the length of the right piece:')
print('   l_R = L_segment - interface_position - n_right ;  the boundary bond is bond (l_R mod 3) of the unit cell')
print('   integer particle number  -> l_R = 0 mod 3')
print('   N = m + 2/3              -> l_R = 1 mod 3')
print('   N = m + 1/3              -> l_R = 2 mod 3   <- the family that fails')
