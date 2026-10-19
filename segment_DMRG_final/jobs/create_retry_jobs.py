"""
Builds jobs_retry.sh: re-queues ONLY the (K,N) jobs that failed in patch_mps with
'no matching charge sectors', each with added_sites increased by `STEP` over the largest value already tried.

Run it again after each round (the failing jobs die in ~3 min, before any DMRG sweep):
  - a job that passed the patch (running / finished / no verdict yet) is never touched or duplicated
  - a job whose every tried added_sites failed gets added_sites = max(tried) + STEP
  - when max(tried)+STEP > MAX_ADDED the pair is only reported, not queued
So each pair ends at the smallest added_sites in {first, first+STEP, ...} that passes the patch.

New: instead of always +STEP, the next value is max(max(tried)+STEP, guess) with
guess = |N| + margin(K,N) rounded up to a multiple of STEP (see margin()). The guess is an empirical rule from the first rounds
(added_sites=12 passes up to |N|=9, 15 passes |N|=12, at the lowest K), NOT a proven formula; if the guess
still fails the script just keeps adding STEP. Pairs it already queued are remembered in retry_submitted.txt
so a second run does not queue the same job again before its log exists.

usage: python3 make_retry_jobs.py <folder with python3-*.out> [STEP] [MAX_ADDED] [EXTRA] [FORCE] [N_LIST]
  FORCE  : if given, every failing pair whose N is in N_LIST is queued once at added_sites=FORCE (instead of the rule);
           pairs with other N are skipped in that run.  N_LIST default = the 9 columns that never passed:
           -51,-39,-27,-21,-18,-15,27,39,51   e.g.   python3 make_retry_jobs.py . 3 60 0 60
"""
import glob, re, sys, collections

folder = sys.argv[1] if len(sys.argv) > 1 else '.'
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MAX_ADDED = int(sys.argv[3]) if len(sys.argv) > 3 else 60
EXTRA = int(sys.argv[4]) if len(sys.argv) > 4 else 0       # extra safety margin on top of the rule below


def margin(K, N):
    """
    Empirical rule, read off the pass/fail maps of added_sites = 12 and 15 (only two sizes, so a GUESS):
      required added_sites ~ |N| + margin
      N > 0 (particles removed): margin 3 at low K (K <= -18), 0 for K >= -12
      N < 0 (particles added)  : margin 3 at low K (K <= -15), 6 for K >= -12
    If the guess is too small the job just fails again and the next run adds STEP.
    """
    if N > 0:
        return 3 if K <= -18 else 0
    return 3 if K <= -15 else 6
FORCE = int(sys.argv[5]) if len(sys.argv) > 5 else None
N_LIST = ([int(x) for x in sys.argv[6].split(',')] if len(sys.argv) > 6
          else [-51, -39, -27, -21, -18, -15, 27, 39, 51])
SUBMITTED = 'retry_submitted.txt'

SCRIPT = '/mnt/users/dperkovic/DMRG_modified/tenpy/segment_DMRG_final/multilayer_segment_final.py'
PYTHON = '/usr/bin/python3'
CPU_TIME, MEM = '7 hour', 8


def val(text, key):
    m = re.search(r'^' + re.escape(key) + r'\s*:\s*(.+?)\s*$', text, re.M)
    return m.group(1) if m else None


n_old = 0
runs = collections.defaultdict(dict)       # key -> {added_sites: 'fail' | 'other'}
info = {}                                  # key -> (name_load, name_save, L, half, chi, U0, K_in, N)
for f in sorted(glob.glob(folder + '/python3-*.out')):
    t = open(f, errors='replace').read()
    a, nl, ns = val(t, 'added_sites'), val(t, 'name_load'), val(t, 'name_save')
    L, half, chi, U0 = val(t, 'L'), val(t, 'interface_position'), val(t, 'chi'), val(t, 'U0')
    x0 = val(t, 'x0 (potential)')            # new argument of the driver: position of the potential step
    m = re.search(r'shift \(K, N\.\.\.\)\s*:\s*\[\s*(-?\d+)\s+(-?\d+)\s*\]', t)
    if None in (a, nl, ns, L, half, chi, U0, x0) or not m:
        if m and x0 is None:
            n_old += 1                     # log of the old driver (no x0): cannot be re-queued with the new command
        continue                           # empty / not started yet: no verdict
    N = int(m.group(2))
    K_in = int(m.group(1)) - int(half) * N  # the printed K contains half*N, undo it: this is the K you typed
    label = ns.split('_charge=')[0]          # path + file + any label such as _SHIFTED_BY_1.5: keeps batches apart
    key = (nl, label, int(L), int(half), int(chi), float(U0), float(x0), K_in, N)
    status = 'fail' if 'ValueError: no matching charge sectors' in t else 'other'
    # 'other' = past the patch, still running, finished, or crashed elsewhere -> do not touch
    runs[key][int(a)] = status
    info[key] = (nl, ns)

import os
done = {}                                   # key string -> largest added_sites already queued by this script
if os.path.exists(SUBMITTED):
    for ln in open(SUBMITTED):
        k, a_ = ln.rsplit('|', 1)
        done[k] = max(done.get(k, 0), int(a_))

lines, giveup, kept, new_sub = [], [], 0, []
for key, tried in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][7], kv[0][8])):
    if any(s != 'fail' for s in tried.values()):
        kept += 1
        continue
    _, _, L, half, chi, U0, x0, K, N = key
    guess = -(-(abs(N) + margin(K, N) + EXTRA) // STEP) * STEP   # rounded up to a multiple of STEP
    nxt = max(max(tried) + STEP, guess)
    if FORCE is not None:
        if N not in N_LIST or max(tried) >= FORCE:
            continue
        nxt = FORCE
    ks = repr(key)
    if done.get(ks, 0) >= nxt:
        kept += 1                                         # already queued at this size, wait for its log
        continue
    nl, ns = info[key]
    if nxt > MAX_ADDED:
        giveup.append((K, N, U0, max(tried)))
        continue
    ns_new = re.sub(r'_added_sites=\d+', '_added_sites=' + str(nxt), ns)
    new_sub.append(ks + '|' + str(nxt) + '\n')
    # argument order of the driver: added_sites name_load name_save L interface_position chi U0 x0 K N
    lines.append('addqueue  -c "%s" -m %d %s %s %d %s %s %d %d %d %s %g %d %d\n'
                 % (CPU_TIME, MEM, PYTHON, SCRIPT, nxt, nl, ns_new, L, half, chi, U0, x0, K, N))

with open('jobs_retry.sh', 'w') as g:
    g.writelines(lines)
with open(SUBMITTED, 'a') as g:
    g.writelines(new_sub)
if n_old:
    print('logs of the old driver (no x0 printed) ignored:', n_old)
print('pairs left alone (passed the patch / running / no verdict):', kept)
print('retry jobs written to jobs_retry.sh:', len(lines))
print('gave up (would exceed MAX_ADDED=%d): %d' % (MAX_ADDED, len(giveup)))
for K, N, U0, a in giveup[:20]:
    print('   K=%d N=%d U=%s  last tried added_sites=%d' % (K, N, U0, a))