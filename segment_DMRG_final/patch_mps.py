"""
Patching of two segment MPS (LHS and RHS) with some added sites in between.

    [ psi_L ][ added sites ][ psi_R ]

The three modes only differ in what is loaded and in how the shift is obtained:
    1) LHS, RHS different:  psi_L, psi_R come from different iMPS, shift=None (computed) or imported
    2) LHS, RHS the same:   psi_R is a copy of psi_L, shift is imported
    3) LHS vacuum:          not implemented yet
Added sites == 0 is a special case: no sites are built, the last bond of psi_L is
projected onto the sectors that connect to psi_R.
"""
import numpy as np
from collections import defaultdict

from tenpy.linalg.charges import LegCharge
from tenpy.networks.mps import MPS


#####################################
#### charge sector helpers (numpy) ##
#####################################
def lexsort(a, axis=-1):
    """wrapper around ``np.lexsort``: allow for trivial case ``a.shape[0] = 0`` without sorting"""
    if any([s == 0 for s in a.shape]):
        return np.arange(a.shape[axis], dtype=np.intp)
    return np.lexsort(a, axis=axis)


def unique_sorted(q):
    """unique charge rows, ordered like the legs in TeNPy (lexsort: last charge is the primary key)"""
    q = np.unique(q, axis=0)
    return q[lexsort(q.T)]


def forward_sectors(q_left, phys_q):
    """
    Charge sectors on the bonds of the added sites, reached from the left.

    q_left: charges of the right leg of psi_L, shape (chi, nq)
    phys_q: list with one array of shape (d, nq) per added site, the charges of its physical states
    returns F, F[0]=q_left and F[s+1]=sectors to the right of added site s
    """
    F = [np.asarray(q_left)]
    for qp in phys_q:
        F.append(unique_sorted(np.concatenate([F[-1] + q for q in qp])))
    return F


def match_rows(q_a, q_b):
    """
    Index pairs (i, j) with q_a[i] == q_b[j] (all charge components).
    The k-th occurrence of a charge in q_a is paired with the k-th occurrence in q_b.
    """
    pos = defaultdict(list)
    for j, row in enumerate(map(tuple, q_b)):
        pos[row].append(j)
    count = defaultdict(int)
    I, J = [], []
    for i, row in enumerate(map(tuple, q_a)):
        k = count[row]
        if k < len(pos.get(row, ())):
            I.append(i)
            J.append(pos[row][k])
            count[row] = k + 1
    return np.array(I, dtype=int), np.array(J, dtype=int)


def connect_sectors(F, end_mask, phys_q):
    """
    Sectors that survive on the bonds of the added sites, such that every sector
    can be reached from the left piece AND leads to the matched sectors on the right.
    All components of the charges are compared.

    F: output of forward_sectors, end_mask: boolean mask of F[-1] with the matched sectors
    returns bonds, bonds[0]=F[0] (full right leg of psi_L), bonds[-1]=F[-1][end_mask]
    """
    num = len(phys_q)
    bonds = [None] * (num + 1)
    bonds[0] = F[0]
    bonds[num] = F[num][end_mask]
    for s in range(num - 1, 0, -1):
        reach = set(map(tuple, np.concatenate([bonds[s + 1] - q for q in phys_q[s]])))
        keep = np.array([tuple(r) in reach for r in F[s]], dtype=bool)
        bonds[s] = F[s][keep]
        if len(bonds[s]) == 0:
            raise ValueError('no charge sector connects the left piece to the right piece through the added sites')
    return bonds


def cap_bonds(bonds, phys_q, max_bond, rng=None):
    """
    Limits the number of charge sectors (= bond dimension of the random window) to about `max_bond`.
    `bonds` is the output of connect_sectors: every sector on it can be reached from the left piece and leads to a
    matched sector on the right. The kept sectors are the union of random paths through the added sites. The paths
    are drawn BACKWARDS, starting at every matched sector on the last bond (the same number of paths for each), so
    ALL matched sectors stay connected to the left piece and none of them is dropped: the RHS is then not projected
    on a smaller set of sectors (dropping some of them changes the legs of the RHS and the right environment no
    longer fits). bonds[0] and bonds[-1] are not changed, the inner bonds have at most max(max_bond, #matched) sectors.
    returns (bonds, None)   (the second entry is kept for compatibility: no matched sector is ever dropped)
    """
    num = len(phys_q)
    if max_bond is None or max_bond <= 0 or num < 2 or max(len(b) for b in bonds[1:-1]) <= max_bond:
        return bonds, None
    rng = np.random.default_rng() if rng is None else rng
    index = [{tuple(r): i for i, r in enumerate(b)} for b in bonds]
    kept = [set() for _ in bonds]
    n_end = len(bonds[num])
    per_end = max(1, max_bond // n_end)
    for e in range(n_end):
        for _ in range(per_end):
            i = e
            for s in range(num - 1, -1, -1):          # step back from bond s+1 to bond s
                q = bonds[s + 1][i]
                prev = [index[s][tuple(q - qp)] for qp in phys_q[s] if tuple(q - qp) in index[s]]
                i = prev[rng.integers(len(prev))]
                kept[s].add(i)
    new = [bonds[0]]
    for s in range(1, num):
        new.append(bonds[s][np.array(sorted(kept[s]), dtype=int)])
    new.append(bonds[num])
    return new, None


def _pairs(bonds, phys_q, s):
    """for every sector r of bonds[s+1]: the list of (l, p) with bonds[s][l] + phys_q[s][p] == bonds[s+1][r]"""
    where = defaultdict(list)
    for l, row in enumerate(map(tuple, bonds[s])):
        where[row].append(l)
    out = []
    for r, qr in enumerate(bonds[s + 1]):
        out.append([(l, p) for p, q in enumerate(phys_q[s]) for l in where.get(tuple(qr - q), ())])
    return out


def sector_tensors(bonds, phys_q):
    """
    Random B tensors (p, vL, vR) of the added sites, non-zero only where charge is conserved:
    bonds[s][l] + phys_q[s][p] == bonds[s+1][r]       (not canonical, used when the left piece is not a single sector)
    """
    Bs = []
    for s, qp in enumerate(phys_q):
        B = np.zeros((len(qp), len(bonds[s]), len(bonds[s + 1])))
        for r, pairs in enumerate(_pairs(bonds, phys_q, s)):
            for l, p in pairs:
                B[p, l, r] = np.random.random()
        Bs.append(B)
    return Bs


def canonical_sector_tensors(bonds, phys_q, w_end):
    """
    Random charge-conserving tensors that form an exactly CANONICAL state (norm 1, no random mismatch at the junction).
    Needs a single sector on bonds[0] (vacuum on the left). Every charge sector is one state, so the Schmidt weights are
    diagonal and can be chosen from the right: the weight w_r of sector r on bond s+1 is split at random among its
    predecessors (l, p),   B[p,l,r]^2 * rho_l = w_r * pi(l, p | r)    with   rho_l = sum_{p,r} w_r pi(l, p | r).
    Then   sum_{p,r} B[p,l,r]^2 = 1  (right-canonical)  and  sum_{p,l} rho_l B[p,l,r]^2 = w_r  (Schmidt weights),
    in particular the last bond carries exactly w_end = S^2 of the left bond of the RHS.
    returns Bs, Ss   (Ss = Schmidt values of the inner bonds 1 .. num-1)
    """
    assert len(bonds[0]) == 1
    num = len(phys_q)
    w = np.asarray(w_end, float)
    w = w / w.sum()
    Bs = [None] * num
    Ss = [None] * (num - 1)
    for s in range(num - 1, -1, -1):
        pairs = _pairs(bonds, phys_q, s)
        B2 = np.zeros((len(phys_q[s]), len(bonds[s]), len(bonds[s + 1])))
        for r, pr in enumerate(pairs):
            pi = np.random.random(len(pr))
            pi = pi / pi.sum()
            for (l, p), x in zip(pr, pi):
                B2[p, l, r] = w[r] * x
        rho = B2.sum(axis=(0, 2))                                  # weights of the bond s
        B2 = B2 / np.maximum(rho, 1e-300)[None, :, None]
        Bs[s] = np.sqrt(B2)
        if s > 0:
            Ss[s - 1] = np.sqrt(rho) / np.linalg.norm(np.sqrt(rho))
        w = rho
    return Bs, Ss


def random_S(n):
    S = np.random.random(n)
    return S / np.linalg.norm(S)


#####################################
#### MPS helpers (TeNPy)           ##
#####################################
def to_flat(psi):
    """B tensors as flat arrays with legs (p, vL, vR)"""
    return [np.transpose(B.to_ndarray(), (1, 0, 2)) for B in psi._B]


def shift_left_leg(psi, sites, charge_shift):
    """rebuilds psi on `sites`, with charge_shift added to the charges of its left leg (all charges follow)"""
    qflat = psi._B[0].get_leg('vL').to_qflat()
    qflat = qflat + np.asarray(charge_shift).astype(qflat.dtype)
    left_leg = LegCharge.from_qflat(sites[0].leg.chinfo, qflat, qconj=1).bunch()[1]
    return MPS.from_Bflat(sites, to_flat(psi), SVs=list(psi._S), bc='segment', legL=left_leg)


def project_left_leg(psi, keep):
    """keeps only the sectors `keep` (indices) on the left leg of psi and brings it to canonical form"""
    psi.segment_boundaries = (None, None)
    S = psi.get_SL(0)
    proj = np.zeros(len(S), bool)
    proj[keep] = True
    B = psi.get_B(0, form='B')
    B.iproject(proj, 'vL')
    psi.set_B(0, B, form='B')
    psi.set_SL(0, S[proj])
    psi.canonical_form_finite(cutoff=0.0)
    psi.test_sanity()
    return psi


#####################################
#### the patching                  ##
#####################################
MAX_BOND = 4000  # default cap on the number of charge sectors of the random window (0 or None: no cap)


def patch_mps(psi_L, psi_R, sites, num_added, shift=None, max_bond=MAX_BOND):
    """
    psi_L:      segment MPS on sites[:half]
    psi_R:      segment MPS on sites[half+num_added:]
    sites:      all sites of the system
    num_added:  number of sites added in between (0 is a special case)
    max_bond:   cap on the number of charge sectors on the bonds of the added sites (see cap_bonds)
    shift:      charge shift of the RHS (K, n_1, ..., n_nlayer). If None it is determined from the
                largest Schmidt values of the two pieces. For num_added>0 the Schmidt values of the
                LHS side are random, so pass the shift explicitly in that case.

    returns the merged MPS and the shift
    """
    half = len(psi_L._B)
    num = num_added
    phys_q = [sites[half + s].leg.to_qflat() for s in range(num)]

    q_L = psi_L._B[-1].get_leg('vR').to_qflat()
    q_R = psi_R._B[0].get_leg('vL').to_qflat()

    # sectors on the LHS of the RHS: either the last bond of psi_L or the last added bond
    F = forward_sectors(q_L, phys_q)
    q_end = F[-1]
    S_end = psi_L._S[-1] if num == 0 else np.random.random(len(q_end))

    if shift is None:
        shift = q_R[np.argmax(psi_R._S[0])] - q_end[np.argmax(S_end)]
    shift = np.rint(np.asarray(shift)).astype(int)

    # which sectors agree?
    q_R_shifted = q_R - shift
    iL, jR = match_rows(q_end, q_R_shifted)
    print("Total Matching Pairs:", len(iL))
    if len(iL) == 0:
        raise ValueError('no matching charge sectors between LHS and RHS, check the shift ' + str(shift))
    mask = np.zeros(len(q_end), dtype=bool)
    mask[iL] = True
    keep_R = np.sort(jR)
    if not np.array_equal(q_end[mask], q_R_shifted[keep_R]):
        raise ValueError('the matched charge sectors of LHS and RHS are not ordered the same way')

    # LHS (+ added sites)
    B_L = to_flat(psi_L)
    S_L = list(psi_L._S)
    if num == 0:
        B_L[-1] = B_L[-1][:, :, mask]
        S_left = S_L[:half]
        B_mid = []
    else:
        bonds = connect_sectors(F, mask, phys_q)
        bonds, sel = cap_bonds(bonds, phys_q, max_bond)
        if sel is not None:
            # drop the matched sectors that no path reached, on the window AND on the RHS
            mask_idx = np.flatnonzero(mask)[sel]
            mask = np.zeros(len(q_end), dtype=bool)
            mask[mask_idx] = True
            keep_R = keep_R[sel]
            print('capped window: matched sectors kept', len(sel))
        print('window bond dimension (charge sectors): max', max(len(b) for b in bonds), '| cap', max_bond)
    # RHS: shift the charges, keep the matched sectors
    psi_R = shift_left_leg(psi_R, sites[half + num:], -shift)
    psi_R = project_left_leg(psi_R, keep_R)

    if num > 0:
        if len(bonds[0]) == 1:
            w_end = np.asarray(psi_R.get_SL(0)) ** 2                 # weights of the left bond of the (projected) RHS
            if len(w_end) != len(bonds[-1]):
                raise ValueError('the left leg of the RHS has %d sectors, the window ends on %d' % (len(w_end), len(bonds[-1])))
            B_mid, S_mid = canonical_sector_tensors(bonds, phys_q, w_end)
            print('window: canonical initial state (norm 1, Schmidt weights consistent with the RHS)')
        else:
            B_mid = sector_tensors(bonds, phys_q)
            S_mid = [random_S(len(b)) for b in bonds[1:-1]]
        S_left = S_L[:half + 1] + S_mid

    # merge: S_left holds the Schmidt values of bonds 0..half+num-1, psi_R._S the rest
    psi = MPS.from_Bflat(sites, B_L + B_mid + to_flat(psi_R), SVs=S_left + list(psi_R._S),
                         bc='segment', legL=psi_L._B[0].get_leg('vL'))
    return psi, shift