"""
Writes jobs_graphs.sh : the MPO graphs for the iDMRG test.

    python3 create_2_3_graphs.py
    bash jobs_graphs.sh

WHAT WE LEARNED (dissect_compression.py on the cluster, control 0.10.1 / slycot 0.6.0)
  * The compression of long sequences failed because control.tf2ss (inside SS_fromV) returns an INEXACT realisation:
    it misses the input by 5e-5 at length 83 and by 47 at length 98, before any truncation. The balanced truncation
    afterwards is fine.  Starting from the exact shift-register realisation the same blocks compress to ~5e-8.
  * --exact swaps SS_fromV at runtime (no library file edited).

TWO FAMILIES
  old   : as before, tf2ss realisation, maxK <= 83.  Achieved fit error 1e-3 ... 7.5e-5.  Gives the lever arm for the
          extrapolation of E to zero fit error, and contains your graph so far (eps=1e-5, xiK=2.5).
  exact : --exact.  Expected fit error ~1e-7 or better, also for long ranges, so eps and xiK can now be scanned
          honestly.  The fit quality is printed (FIT QUALITY line) and stored; create_idmrg_2_3_jobs.py refuses graphs
          whose fit failed.  CHECK THE FIT QUALITY LINES BEFORE RUNNING THE IDMRG JOBS.
"""
Lx = 15

# (eps, how to fix the range, exact?)          expected maxK
CASES = [
    (1e-3, '--xiK 2.5', False),     # 55   lever arm, fit error not measured yet
    (1e-4, '--xiK 2.5', False),     # 69   fit error 2.5e-4
    (1e-5, '--xiK 2.5', False),     # 83   fit error 7.5e-5   <- your graph so far (baseline)
    (1e-5, '--xiK 2.5', True),      # 83   same coefficients, exact realisation
    (1e-6, '--xiK 2.5', True),      # 97
    (1e-7, '--xiK 2.5', True),      # 110
    (1e-6, '--xiK 4.0', True),      # 154  longer range at the same eps: is xiK=2.5 long enough?
]

SCRIPT = '/mnt/users/dperkovic/DMRG_modified/tenpy/run_idmrg/create_graphs/create_mpo_graph_scan_2_3.py'
QUEUE = '/usr/bin/python3 '     # graphs build in < 1 s; request is generous

with open('jobs_graphs.sh', 'w') as f:
    for eps, rng, exact in CASES:
        line = QUEUE + SCRIPT + ' ' + str(Lx) + ' --eps ' + str(eps) + ' ' + rng + (' --exact' if exact else '')
        f.write(line + '\n')
        print(line)
