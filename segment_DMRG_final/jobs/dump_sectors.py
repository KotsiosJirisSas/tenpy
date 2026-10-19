# Run on the cluster, once per iMPS file:   python3 dump_sectors.py <name_load>      (name_load WITHOUT '.pkl', as in the jobs)
# Writes sectors.txt: the distinct (K, N) charges on the left leg of the first site of the iMPS (a unit-cell boundary),
# i.e. the sectors the right piece can start in. make_jobs_*.py reads it instead of the table I copied from your logs.
import sys, pickle
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy')
import numpy as np

name_load = sys.argv[1]
with open(name_load + '.pkl', 'rb') as f:
    data = pickle.load(f)
psi = data['mps']
q = psi._B[0].get_leg('vL').to_qflat()          # columns as in the logs: column 0 = K, column 1 = N
rows = sorted(set(map(tuple, np.asarray(q).astype(int).tolist())))
print('charge info:', psi.sites[0].leg.chinfo, '| rows on the leg:', len(q), '| distinct (K, N):', len(rows))
for r in rows:
    print(*r)
with open('sectors.txt', 'w') as g:
    for r in rows:
        g.write(' '.join(str(x) for x in r) + '\n')
print('written sectors.txt')
