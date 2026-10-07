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


def sector_tensors(bonds, phys_q):
    """
    Random B tensors (p, vL, vR) of the added sites, non-zero only where charge is conserved:
    bonds[s][l] + phys_q[s][p] == bonds[s+1][r]
    """
    Bs = []
    for s, qp in enumerate(phys_q):
        qL, qR = bonds[s], bonds[s + 1]
        where = defaultdict(list)
        for l, row in enumerate(map(tuple, qL)):
            where[row].append(l)
        B = np.zeros((len(qp), len(qL), len(qR)))
        for r, qr in enumerate(qR):
            for p, q in enumerate(qp):
                for l in where.get(tuple(qr - q), ()):
                    B[p, l, r] = np.random.random()
        Bs.append(B)
    return Bs


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
def patch_mps(psi_L, psi_R, sites, num_added, shift=None):
    """
    psi_L:      segment MPS on sites[:half]
    psi_R:      segment MPS on sites[half+num_added:]
    sites:      all sites of the system
    num_added:  number of sites added in between (0 is a special case)
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
        B_mid = sector_tensors(bonds, phys_q)
        S_left = S_L[:half + 1] + [random_S(len(b)) for b in bonds[1:-1]]

    # RHS: shift the charges, keep the matched sectors
    psi_R = shift_left_leg(psi_R, sites[half + num:], -shift)
    psi_R = project_left_leg(psi_R, keep_R)

    # merge: S_left holds the Schmidt values of bonds 0..half+num-1, psi_R._S the rest
    psi = MPS.from_Bflat(sites, B_L + B_mid + to_flat(psi_R), SVs=S_left + list(psi_R._S),
                         bc='segment', legL=psi_L._B[0].get_leg('vL'))
    return psi, shift
