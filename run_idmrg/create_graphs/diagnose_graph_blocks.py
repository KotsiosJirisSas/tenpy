#!/mnt/users/dperkovic/myenv/bin/python
"""
Where does the large 'norm err / total norm' come from?   (new file, changes nothing else)

For every case it builds the model exactly like create_mpo_graph_scan_2_3.py does, then looks inside the Markov system
M.Vmk_MS['MS'] and prints, for the worst blocks:
    n_seq    shape of the stored sequence block  (p, q, maxK+1)
    2norm    norm of the input sequences
    N        number of Markov nodes kept
    norm_err ||compressed - input||   (this is the number that is summed into 'norm err / total norm')
    max|eig| largest |eigenvalue| of the compressed A.  > 1 means the compressed system DIVERGES with r.
    last_r   last r at which max|seq| is still above the tolerance tol  (how much of maxK carries information)
Each case is built TWICE; if the two total errors differ the compression is not reproducible.

usage
    python3 diagnose_graph_blocks.py 15                              # the 8 scan cases
    python3 diagnose_graph_blocks.py 15 --cases 1e-5:2.5 1e-6:2.5    # eps:xiK
    python3 diagnose_graph_blocks.py 15 --cases 1e-5@83 1e-6@83 1e-7@83 1e-6@97
    python3 diagnose_graph_blocks.py 15 --cases 1e-5@83 1e-6@97 --residual 2     # + where in r the error sits
    python3 diagnose_graph_blocks.py 15 --exact --cases 1e-5@83 1e-6@97 1e-7@110 1e-6@154   # with the exact realisation
                                                                     # eps@maxK : xiK is chosen so that maxK is exactly that
"""
import sys
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy')
import argparse
import numpy as np
from tenpy.models import multilayer_qh_DP_final as mod
from create_mpo_graph_scan_2_3 import build_model_par, use_exact_realization      # same model as the scan graphs


def xiK_for_maxK(eps, maxK, Lx):
    kappa = 2 * np.pi / Lx
    return (maxK - 0.5) * kappa / (-np.log(0.1 * eps))     # ceil(-ln(0.1 eps) xiK / kappa) == maxK


def parse_case(s, Lx):
    if '@' in s:
        e, k = s.split('@')
        eps, maxK = float(e), int(k)
        xiK = xiK_for_maxK(eps, maxK, Lx)
        if xiK < 1.0:
            print('WARNING: %s needs xiK=%.3f < 1, the code clamps xiK to >= 1, so maxK will be larger' % (s, xiK))
        return eps, xiK
    e, x = s.split(':')
    return float(e), float(x)


def build(Lx, eps, xiK):
    model_par, mu = build_model_par(Lx, eps, xiK, 6)
    model_par['verbose'] = 1
    M = mod.QH_model(model_par, '')
    return M, M.Vmk_MS['MS']


def abc_output(r_max, A, B, C):
    """same as Markov.ABC_output: V[:, :, r] = C A^r B"""
    V = np.zeros((C.shape[0], B.shape[1], r_max))
    X = B
    for r in range(r_max):
        V[:, :, r] = np.dot(C, X)
        X = np.dot(A, X)
    return V


def residual_report(MS, block_id):
    """WHERE in r does the compressed system miss the input sequence?"""
    b = MS.io[block_id]
    seq = b['seq']
    comp = abc_output(seq.shape[2], b['A'], b['B'], b['C']) if b['N'] > 0 else np.zeros_like(seq)
    diff = comp - seq
    has_ign = 'ignore_r0' in b
    if has_ign and diff.shape[2] > 0:
        diff[:, :, 0] *= (1 - np.asarray(b['ignore_r0']))
    d = np.linalg.norm(diff.reshape(-1, diff.shape[2]), axis=0)          # error per r
    tot2 = np.sum(d ** 2)
    print('      block %d: N=%d  norm_err=%.4e  ignore_r0 present: %s' % (block_id, b['N'], np.sqrt(tot2), has_ign))
    print('        seq [r=0..5]      :', np.array2string(seq[0, 0, :6], precision=4))
    print('        compressed [0..5] :', np.array2string(comp[0, 0, :6], precision=4))
    print('        |error| per r [0..5]:', np.array2string(d[:6], precision=3))
    print('        largest error at r=%d (%.3e)' % (int(np.argmax(d)), d.max()))
    for lo, hi in [(0, 2), (2, 10), (10, 30), (30, len(d))]:
        hi = min(hi, len(d))
        if hi <= lo:
            continue
        print('        share of error^2 in r=%d..%d : %.3f' % (lo, hi - 1, np.sum(d[lo:hi] ** 2) / tot2 if tot2 > 0 else 0.0))


def block_table(MS, top=8, residual=0):
    rows = []
    tol = MS.info.get('ABC_tol', np.nan)
    for i, b in enumerate(MS.io):
        if b is None:
            continue
        seq = b['seq']
        A = b['A']
        eig = np.max(np.abs(np.linalg.eigvals(A))) if A.size else 0.0
        mx = np.max(np.abs(seq.reshape(-1, seq.shape[2])), axis=0)
        idx = np.nonzero(mx > tol)[0]
        last_r = int(idx[-1]) if len(idx) else -1
        rows.append(dict(block=i, shape=seq.shape, norm=np.linalg.norm(seq), N=b['N'],
                         err=b['norm_err'], eig=eig, last_r=last_r, tail=mx[-1]))
    rows.sort(key=lambda r: -r['err'])
    print('    %5s %-14s %10s %4s %12s %9s %7s %10s' % ('block', 'seq shape', '2norm', 'N', 'norm_err', 'max|eig|', 'last_r', 'tail'))
    for r in rows[:top]:
        print('    %5d %-14s %10.3e %4d %12.4e %9.4f %7d %10.2e' %
              (r['block'], str(r['shape']), r['norm'], r['N'], r['err'], r['eig'], r['last_r'], r['tail']))
    if residual:
        print('    residual of the %d worst distinct blocks:' % residual)
        done = []
        for r in rows:
            if any(np.isclose(r['err'], e, rtol=1e-9) for e in done):
                continue
            done.append(r['err'])
            residual_report(MS, r['block'])
            if len(done) >= residual:
                break
    nbad = sum(1 for r in rows if r['eig'] > 1.0 + 1e-9)
    print('    blocks: %d, with a divergent compressed A (max|eig|>1): %d' % (len(rows), nbad))
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('Lx', type=float)
    ap.add_argument('--cases', nargs='*', default=None)
    ap.add_argument('--top', type=int, default=8)
    ap.add_argument('--exact', action='store_true', help='compress with the exact shift-register realisation (see create_mpo_graph_scan_2_3.py)')
    ap.add_argument('--residual', type=int, default=0, help='also print where in r the N worst distinct blocks miss the input')
    args = ap.parse_args()
    Lx = round(args.Lx, 3)
    if args.exact:
        use_exact_realization()
    if args.cases:
        cases = [parse_case(c, Lx) for c in args.cases]
    else:
        cases = [(e, x) for x in (2.5, 4.0) for e in (1e-4, 1e-5, 1e-6, 1e-7)]

    for eps, xiK in cases:
        maxK = int(np.ceil(-np.log(0.1 * eps) * max(1., xiK) / (2 * np.pi / Lx)))
        print('=' * 100)
        print('eps=%g  xiK=%.4f  -> maxK=%d' % (eps, xiK, maxK))
        totals = []
        for rep in (1, 2):
            M, MS = build(Lx, eps, xiK)
            totals.append(MS.info['total_norm_err'])
            if rep == 1:
                print('  total_norm_err = %.6e   (tol = %.3e, input 2norm = %.5f)' % (MS.info['total_norm_err'], MS.info['ABC_tol'], MS.info['2norm']))
                block_table(MS, args.top, args.residual)
        print('  rebuilt a second time: total_norm_err = %.6e   -> %s' %
              (totals[1], 'reproducible' if np.isclose(totals[0], totals[1], rtol=1e-9) else 'NOT REPRODUCIBLE'))
