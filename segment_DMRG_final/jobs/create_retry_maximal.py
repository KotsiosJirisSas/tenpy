"""
Reads the python3-*.out logs in a folder and writes jobs_60.sh: one job with added_sites=60 for every (K,N,U)
that FAILED ('no matching charge sectors') and has no run that got past the patch.
usage: python3 retry_60.py <folder with python3-*.out>
"""
import glob, re, sys

folder = sys.argv[1] if len(sys.argv) > 1 else '.'
ADDED = 60
SCRIPT = '/mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/multilayer_segment_final.py'


def val(t, key):
    m = re.search(r'^' + re.escape(key) + r'\s*:\s*(.+?)\s*$', t, re.M)
    return m.group(1) if m else None


runs = {}                                   # key -> list of (status)
info = {}
for f in sorted(glob.glob(folder + '/python3-*.out')):
    t = open(f, errors='replace').read()
    ns, nl = val(t, 'name_save'), val(t, 'name_load')
    L, half, chi, U0 = val(t, 'L'), val(t, 'interface_position'), val(t, 'chi'), val(t, 'U0')
    x0 = val(t, 'x0 (potential)')            # new argument of the driver: position of the potential step
    m = re.search(r'shift \(K, N\.\.\.\)\s*:\s*\[\s*(-?\d+)\s+(-?\d+)\s*\]', t)
    if None in (ns, nl, L, half, chi, U0, x0) or not m:
        continue                            # empty / not started / old-driver log without x0
    N = int(m.group(2))
    K = int(m.group(1)) - int(half) * N     # the K you typed
    label = ns.split('_charge=')[0]          # keeps differently labelled batches (e.g. _SHIFTED_BY_1.5) apart
    key = (nl, label, int(L), int(half), int(chi), U0, float(x0), K, N)
    failed = 'ValueError: no matching charge sectors' in t
    runs.setdefault(key, []).append(failed)
    info[key] = ns

n = 0
with open('jobs_60.sh', 'w') as g:
    for key, res in runs.items():
        if not all(res):                    # some run passed the patch / is running -> not a failed pair
            continue
        nl, _, L, half, chi, U0, x0, K, N = key
        ns = re.sub(r'_added_sites=\d+', '_added_sites=%d' % ADDED, info[key])
        # argument order of the driver: added_sites name_load name_save L interface_position chi U0 x0 K N
        g.write('addqueue  -c "7 hour" -m 8 /usr/bin/python3 %s %d %s %s %d %d %d %s %g %d %d\n'
                % (SCRIPT, ADDED, nl, ns, L, half, chi, U0, x0, K, N))
        n += 1
print(n, 'jobs written to jobs_60.sh')