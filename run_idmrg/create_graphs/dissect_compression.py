#!/mnt/users/dperkovic/myenv/bin/python
"""
WHY does the Markov compression fail?   (new file, changes nothing else)

compute_ABC does, for every block:      sys = SS_fromV(seq)                 step 1: exact realisation of the sequence
                                        sys_comp = SS_compress(sys, tol)    step 2: balanced truncation (slycot ab09ad)
This script takes the worst block(s) of a given graph case and tests the two steps separately.

  STEP 1   err_full = || output(sys) - seq ||            should be ~1e-15.  If not, tf2ss already breaks the sequence.
  STEP 2   for several tolerances and for the four (job, equil) variants of ab09ad:
              Nr, err = || output(sys_comp) - seq ||, and the error bound 2*sum(discarded Hankel singular values).
           job   'B' = balance & truncate (what the code uses),  'N' = balancing-free square-root variant
           equil 'S' = scale the system first (what the code uses),  'N' = no scaling
  STEP 1b  the same compression, but starting from the EXACT shift-register realisation of the sequence
           (A = shift, B = e1, C = (v0, v1, ...); reproduces the sequence with zero numerical error). If the errors
           disappear here, the whole failure is in tf2ss and a ~10-line replacement of SS_fromV fixes it.
  HSV      Hankel singular values returned by slycot vs an independent SVD of the Hankel matrix of the sequence.
           If they disagree above ~1e-9 the Gramian computation inside slycot is inaccurate.

Everything is saved to <out>/block_<id>_<case>.npz (seq, A, B, C, hsv, reference hsv) for offline analysis.

usage (next to create_mpo_graph_scan_2_3.py):
    python3 dissect_compression.py 15 --cases 1e-6@97
    python3 dissect_compression.py 15 --cases 1e-5@83 1e-6@97 --nblocks 2
    python3 dissect_compression.py 15 --cases 1e-6@97 --blocks 0 --out dissect_out
"""
import sys
sys.path.append('/mnt/users/dperkovic/DMRG_modified/tenpy')
import os
import argparse
import numpy as np


def output(sys_, r_max, abc_output):
    return abc_output(r_max, sys_.A, sys_.B, sys_.C, dtype=sys_.D.dtype)


def fir_realization(seq):
    """exact realisation of V[:, :, r]:  C A^r B = V_r, with D = 0 (shift register, order l*q)"""
    import types
    p, q, l = seq.shape
    n = l * q
    A = np.zeros((n, n))
    for k in range(1, l):
        A[k * q:(k + 1) * q, (k - 1) * q:k * q] = np.eye(q)
    B = np.zeros((n, q))
    B[:q, :] = np.eye(q)
    C = np.hstack([seq[:, :, k] for k in range(l)])
    return types.SimpleNamespace(A=A, B=B, C=C, D=np.zeros((p, q)))


def per_r_summary(diff):
    d = np.linalg.norm(diff.reshape(-1, diff.shape[2]), axis=0)
    tot2 = np.sum(d ** 2)
    parts = []
    for lo, hi in [(0, 10), (10, 30), (30, len(d))]:
        hi = min(hi, len(d))
        if hi > lo:
            parts.append('r=%d..%d: %.2f' % (lo, hi - 1, np.sum(d[lo:hi] ** 2) / tot2 if tot2 > 0 else 0.0))
    return 'peak at r=%d (%.2e);  share of error^2  %s' % (int(np.argmax(d)), d.max(), '  '.join(parts))


def hankel_ref(seq1d):
    """Hankel singular values of the FIR system with impulse response h[0]=0, h[r+1]=seq[r]:
       singular values of H[i,j] = seq[i+j] (zero beyond the end)."""
    l = len(seq1d)
    H = np.zeros((l, l))
    for i in range(l):
        H[i, :l - i] = seq1d[i:]
    return np.linalg.svd(H, compute_uv=False)


def dissect_block(seq, tol, SS_fromV, SS_compress, ABC_output, tag='', outdir=None):
    """seq: array (1,1,l).  The three functions are those of tenpy.algorithms.Markov."""
    l = seq.shape[2]
    s1d = seq[0, 0, :]
    print('  sequence: length %d, 2norm %.4e, tol %.3e' % (l, np.linalg.norm(s1d), tol))

    # ---------------- step 1
    sys_full = SS_fromV(seq)
    n = np.size(sys_full.A, 0)
    err_full = np.linalg.norm(output(sys_full, l, ABC_output) - seq)
    print('  STEP 1  tf2ss realisation: order %d, ||B||=%.2e, ||C||=%.2e, reproduces the sequence to %.3e' %
          (n, np.linalg.norm(sys_full.B), np.linalg.norm(sys_full.C), err_full))

    print('          where: ' + per_r_summary(output(sys_full, l, ABC_output) - seq))

    # ---------------- reference Hankel singular values
    s_ref = hankel_ref(s1d)

    # ---------------- step 2
    print('  STEP 2  balanced truncation (rows: job/equil/tol multiplier)')
    print('     %-3s %-5s %8s | %4s %12s %12s %9s %10s' % ('job', 'equil', 'tol', 'Nr', 'err', 'bound', 'max|eig|', '||B||*||C||'))
    hsv_keep = None
    for job, equil in [('B', 'S'), ('B', 'N'), ('N', 'S'), ('N', 'N')]:
        for mult in [0.01, 0.1, 1.0, 10.0, 100.0]:
            t = tol * mult
            try:
                sc, hsv = SS_compress(sys_full, '', tol=t, equil=equil, job=job)
            except Exception as e:
                print('     %-3s %-5s %8.1e | failed: %s' % (job, equil, t, str(e)[:60]))
                continue
            nr = np.size(sc.A, 0)
            err = np.linalg.norm(output(sc, l, ABC_output) - seq)
            bound = 2 * np.sum(s_ref[nr:])
            eig = np.max(np.abs(np.linalg.eigvals(sc.A))) if nr else 0.0
            bc = np.linalg.norm(sc.B) * np.linalg.norm(sc.C)
            flag = '  <-- error above bound' if err > 10 * max(bound, 1e-15) and err > t else ''
            print('     %-3s %-5s %8.1e | %4d %12.3e %12.3e %9.4f %10.2e%s' % (job, equil, t, nr, err, bound, eig, bc, flag))
            if job == 'B' and equil == 'S' and mult == 1.0:
                hsv_keep = np.asarray(hsv)
                kept = (np.array(sc.A), np.array(sc.B), np.array(sc.C))

    # ---------------- step 1b : exact realisation, same compression
    fir = fir_realization(seq)
    err_fir = np.linalg.norm(output(fir, l, ABC_output) - seq)
    print('  STEP 1b exact shift-register realisation: order %d, reproduces the sequence to %.3e' % (fir.A.shape[0], err_fir))
    print('     %-3s %-5s %8s | %4s %12s %12s %9s %10s' % ('job', 'equil', 'tol', 'Nr', 'err', 'bound', 'max|eig|', '||B||*||C||'))
    for job, equil in [('B', 'S'), ('N', 'S')]:
        for mult in [0.1, 1.0, 10.0]:
            t = tol * mult
            try:
                sc, hsv = SS_compress(fir, '', tol=t, equil=equil, job=job)
            except Exception as e:
                print('     %-3s %-5s %8.1e | failed: %s' % (job, equil, t, str(e)[:60]))
                continue
            nr = np.size(sc.A, 0)
            err = np.linalg.norm(output(sc, l, ABC_output) - seq)
            bound = 2 * np.sum(s_ref[nr:])
            eig = np.max(np.abs(np.linalg.eigvals(sc.A))) if nr else 0.0
            print('     %-3s %-5s %8.1e | %4d %12.3e %12.3e %9.4f %10.2e' %
                  (job, equil, t, nr, err, bound, eig, np.linalg.norm(sc.B) * np.linalg.norm(sc.C)))

    # ---------------- HSV comparison
    if hsv_keep is not None:
        print('  HSV  slycot vs independent SVD of the Hankel matrix (index: slycot, reference, relative difference)')
        for i in [0, 1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60]:
            if i < len(hsv_keep) and i < len(s_ref):
                rel = abs(hsv_keep[i] - s_ref[i]) / max(s_ref[i], 1e-300)
                print('     %3d: %.4e  %.4e  %.1e' % (i, hsv_keep[i], s_ref[i], rel))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
        fn = os.path.join(outdir, 'block_%s.npz' % tag)
        np.savez(fn, seq=seq, A_full=sys_full.A, B_full=sys_full.B, C_full=sys_full.C,
                 hsv_slycot=hsv_keep if hsv_keep is not None else np.zeros(0), hsv_ref=s_ref,
                 A_comp=kept[0], B_comp=kept[1], C_comp=kept[2], tol=tol)
        print('  saved', fn)


def xiK_for_maxK(eps, maxK, Lx):
    return (maxK - 0.5) * (2 * np.pi / Lx) / (-np.log(0.1 * eps))


def parse_case(s, Lx):
    if '@' in s:
        e, k = s.split('@')
        return float(e), float(xiK_for_maxK(float(e), int(k), Lx))
    e, x = s.split(':')
    return float(e), float(x)


if __name__ == '__main__':
    from tenpy.models import multilayer_qh_DP_final as mod
    import tenpy.algorithms.Markov as Markov
    from tenpy.algorithms.Markov import SS_compress, ABC_output
    # step 1 must be the LEGACY route (tf2ss), also once the patched Markov.py defaults to the exact one
    SS_fromV = getattr(Markov, 'SS_fromV_tf2ss_legacy', Markov.SS_fromV)
    from create_mpo_graph_scan_2_3 import build_model_par

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('Lx', type=float)
    ap.add_argument('--cases', nargs='+', required=True, help='eps@maxK or eps:xiK')
    ap.add_argument('--nblocks', type=int, default=1, help='number of worst distinct blocks to dissect')
    ap.add_argument('--blocks', nargs='*', type=int, default=None, help='explicit block ids instead')
    ap.add_argument('--out', type=str, default='dissect_out')
    args = ap.parse_args()
    Lx = round(args.Lx, 3)
    try:
        import control, slycot, scipy
        print('versions: control', control.__version__, '| slycot', slycot.__version__, '| scipy', scipy.__version__, '| numpy', np.__version__)
    except Exception as e:
        print('could not read versions:', e)

    for case in args.cases:
        eps, xiK = parse_case(case, Lx)
        print('=' * 100)
        print('case', case, ' eps=%g xiK=%.5f' % (eps, xiK))
        model_par, mu = build_model_par(Lx, eps, xiK, 6)
        model_par['verbose'] = 0
        M = mod.QH_model(model_par, '')
        MS = M.Vmk_MS['MS']
        tol = MS.info['ABC_tol']
        print('total_norm_err of the normal pipeline: %.4e  (tol %.3e)' % (MS.info['total_norm_err'], tol))
        if args.blocks:
            ids = args.blocks
        else:
            order = sorted([i for i, b in enumerate(MS.io) if b is not None], key=lambda i: -MS.io[i]['norm_err'])
            ids, seen = [], []
            for i in order:
                if any(np.isclose(MS.io[i]['norm_err'], e, rtol=1e-9) for e in seen):
                    continue
                seen.append(MS.io[i]['norm_err'])
                ids.append(i)
                if len(ids) >= args.nblocks:
                    break
        for i in ids:
            b = MS.io[i]
            print('-' * 100)
            print('block %d   (pipeline: N=%d, norm_err=%.4e)' % (i, b['N'], b['norm_err']))
            dissect_block(np.array(b['seq']), tol, SS_fromV, SS_compress, ABC_output,
                          tag='%d_%s' % (i, case.replace(':', '_').replace('@', 'at')), outdir=args.out)
