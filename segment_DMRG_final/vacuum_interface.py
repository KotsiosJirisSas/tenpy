"""
Mode 3: vacuum on the LHS.

    [ 0 0 0 ... 0 (pad) ][ psi_R ]

Everything on the left is empty, so without a pad the total charge of the segment is the charge of
the sector on the left leg of psi_R. The target charge `target` = (K, n_1, ..., n_nlayer) is reached as follows:

  1) if a sector with charge `target` exists on the left leg of psi_R: project onto it (largest Schmidt weight)
  2) otherwise pad the right end of the vacuum with an alternating 1010... product state. The pad has charge
     Q_pad. Charge is conserved along the chain: the left part (vacuum + pad) has charge Q_vac + Q_pad and psi_R
     is relabeled from its sector s to that charge, so the total charge is  Q_vac - target  independent of the pad,
     if the sector is  s = target + Q_pad.  Small pads are tried first.
     (Charges are relative to the root configuration: an empty site has N charge -p, a filled one q-p, with
     p particles per q sites of the root configuration, and K = (orbital index) * N. So an empty vacuum has a
     nonzero charge Q_vac, and one particle is q units of N charge.)

The vacuum (+ pad) is a product state, patch_mps (with 0 added sites) does the rest.
"""
import numpy as np
from collections import defaultdict

from tenpy.linalg.charges import LegCharge
from tenpy.linalg.np_conserved import Array
from tenpy.networks.mps import MPS

from patch_mps import project_left_leg


#####################################
#### the pad                       ##
#####################################
def compositions(k, caps):
    """all (k_1, ..., k_n) with sum k and 0 <= k_l <= caps[l]"""
    if len(caps) == 1:
        if k <= caps[0]:
            yield (k,)
        return
    for first in range(min(k, caps[0]) + 1):
        for rest in compositions(k - first, caps[1:]):
            yield (first,) + rest


def pad_positions(half, nlayer, offset=0):
    """
    Sites of the pad for every layer: every other orbital, packed against the cut (the right end of
    the vacuum), the pattern ...0101 | psi_R. The whole pattern can be moved to the left by `offset` orbitals.
    Site i belongs to layer i % nlayer.
    """
    pos = []
    for l in range(nlayer):
        sites_l = [i for i in range(half - 1, -1, -1) if i % nlayer == l]   # layer l, from the cut to the left
        pos.append(sites_l[offset::2])
    return pos


def find_pad(dq, half, nlayer, q_R, S_R0, target, max_pad=None, max_offset=0):
    """
    dq:      charge added by filling a site, shape (half, nq)
    q_R:     charges on the left leg of psi_R, S_R0 its Schmidt values
    target:  charge we want (the sector on the RHS leg if there is no pad); the pad needs the sector target + Q_pad
    returns dict with the sites to fill, the charge of the pad and the index j of the sector on the RHS leg
    Smallest pad first (number of particles, then offset), among equals the largest Schmidt weight.
    """
    target = np.asarray(target)
    where = defaultdict(list)
    for j, row in enumerate(map(tuple, q_R)):
        where[row].append(j)
    S_R0 = np.asarray(S_R0)

    pos = {}
    cum = {}
    for o in range(max_offset + 1):
        pos[o] = pad_positions(half, nlayer, o)
        cum[o] = [np.concatenate([np.zeros((1, dq.shape[1]), dq.dtype), np.cumsum(dq[p], axis=0)]) if len(p) > 0
                  else np.zeros((1, dq.shape[1]), dq.dtype) for p in pos[o]]
    kmax = max(sum(len(p) for p in pos[o]) for o in pos)
    if max_pad is not None:
        kmax = min(kmax, max_pad)

    for k in range(kmax + 1):
        best = None
        for o in range(max_offset + 1):
            caps = [len(p) for p in pos[o]]
            for comp in compositions(k, caps):
                Q_pad = sum(cum[o][l][comp[l]] for l in range(nlayer))
                js = where.get(tuple(target + Q_pad))
                if js:
                    j = js[int(np.argmax(S_R0[js]))]
                    if best is None or S_R0[j] > best['weight']:
                        filled = [pos[o][l][m] for l in range(nlayer) for m in range(comp[l])]
                        best = dict(weight=S_R0[j], j=j, filled=sorted(filled), Q_pad=Q_pad,
                                    n_particles=comp, offset=o)
        if best is not None:
            return best
    raise ValueError('no pad with up to ' + str(kmax) + ' particles reaches the target charge ' + str(target)
                     + ' (try a larger max_pad / max_offset, or a different target). Sectors on the RHS left leg (K,N,..): '
                     + str(np.unique(q_R, axis=0)[:30].tolist()))


def product_state(sites_L, filled):
    """
    Product state on sites_L: empty everywhere except on the sites in `filled`.
    The left leg has charge 0, state 0 of a site is empty and state 1 is filled.
    """
    half = len(sites_L)
    nq = sites_L[0].leg.to_qflat().shape[1]
    filled = set(filled)
    Bflat = []
    for i in range(half):
        B = np.zeros((2, 1, 1))
        B[1 if i in filled else 0, 0, 0] = 1.
        Bflat.append(B)
    left_leg = LegCharge.from_qflat(sites_L[0].leg.chinfo, np.zeros((1, nq), dtype=int), qconj=1).bunch()[1]
    return MPS.from_Bflat(sites_L, Bflat, SVs=[np.ones(1) for _ in range(half + 1)], bc='segment', legL=left_leg)


def vacuum_lhs(sites, half, nlayer, psi_R, target, max_pad=None, max_offset=0):
    """
    Builds the LHS (vacuum + pad) for the target charge and projects psi_R onto the matching sector.

    sites:   all sites, the vacuum lives on sites[:half]
    psi_R:   segment MPS on sites[half:]
    returns psi_L, psi_R (projected), the shift to pass to patch_mps, and the info about the pad
    """
    sites_L = sites[:half]
    qphys = [s.leg.to_qflat() for s in sites_L]
    dq = np.array([q[1] - q[0] for q in qphys])
    Q_vac = np.sum([q[0] for q in qphys], axis=0)
    # charges are relative to the root configuration, so the empty vacuum is not neutral; the target is relative to it
    print('charge of the empty vacuum Q_vac:', Q_vac.tolist(), '(one particle = q =', int(round(dq[:, 1:].max())) if dq.shape[1] > 1 else '?', 'units of N charge)')

    q_R = psi_R._B[0].get_leg('vL').to_qflat()
    pad = find_pad(dq, half, nlayer, q_R, psi_R._S[0], target, max_pad, max_offset)
    print('PAD: particles per layer', pad['n_particles'], ', offset', pad['offset'], ', charge', pad['Q_pad'],
          ', sector on the RHS leg', q_R[pad['j']])
    print('filled sites of the vacuum:', pad['filled'])

    psi_L = product_state(sites_L, pad['filled'])
    Q_left = psi_L._B[-1].get_leg('vR').to_qflat()[0]
    # the RHS sector s has to sit on a bond with the charge of the left part;  s = target + Q_pad  so this is  target - Q_vac
    shift = q_R[pad['j']] - Q_left
    psi_R = project_left_leg(psi_R, [pad['j']])
    return psi_L, psi_R, shift, pad


#####################################
#### vacuum left environment       ##
#####################################
def find_identity_channel(W, tol=1e-12):
    """
    W: npc.Array with legs (phys_out, phys_in, a, b)

    Returns:
        (a_id, b_id) or None
    """
   
    dim_a, dim_b ,n_out, n_in= W.shape

    ch_a = W.get_leg(0).to_qflat()
    ch_b = W.get_leg(1).to_qflat()
    #print(ch_a,ch_b)
    #quit()
    #print(n_out, n_in, dim_a, dim_b)
    #quit()
    for a in range(dim_a):
        #print('value of a',a)
        #print(ch_a[a])
        # 1. CHARGE CONDITION; must be on diagonal with charges
        try:
            matches = np.all(ch_b == ch_a[a], axis=1)
            b_ind = np.where(matches)[0]
        except:
            b_ind=[]
       
        for b in b_ind:
        
            #### test diagonal:
           
            # 2. CHECK FULL TENSOR STRUCTURE
            block = W[ a, b,:,:].to_ndarray()
            #print(block)
            
            identity=np.eye(n_in)
            assert np.all(block.shape==identity.shape)
           
            value_test=np.sum(np.abs(block-identity))
           
            if value_test>tol:
                pass
            else:

                #print('value of a',a)
                #print(ch_a[a])
                #print('Inside')

                ############
                #### need the column to be (I,0) basically,
                #### then this works out fine
                block2 = W[ :, b,:,:].to_ndarray()
                #print(block2.shape)
                

                identity2=np.zeros(shape=(dim_a,n_in,n_in))
                identity2[a]=np.eye(n_in)
                assert np.all(block2.shape==identity2.shape)
              
                #print('compare')
                #print(np.sum(np.abs(identity2)))
                #print(np.sum(np.abs(block2)))
               
                value_test2=np.sum(np.abs(block2-identity2))

                #print('value test2:',value_test2)
                #print('x'*20)
                if value_test2<tol:
                    print('row, column indices correspoding to the identity')
                    print(a,b)
                    index_a=a
                  
         
    return index_a


def set_left_environment_to_vacuum(W,leg_HMPO,leg_MPS):
  
    """
    produces left environment which corresponds to vacuum
    so far works only it total charge is zero
    leg_HMPO,leg_MPS: give the legs of MPO and MPS
    """

    index_a=find_identity_channel(W, tol=1e-12)

    #print(index_a)
    #quit()
    duljina_MPS=len(leg_MPS.to_qflat())
    duljina=len(leg_HMPO.to_qflat())


    labels=['vR*', 'wR', 'vR']


    
    #gives qflat for non-trivial leg    
    legcharges=[]
    legcharges.append(leg_MPS)
    legcharges.append(leg_HMPO.conj())
    legcharges.append(leg_MPS.conj())
  

    
    #set data flat
    Bflat=np.zeros(duljina*duljina_MPS*duljina_MPS)
    
    Bflat=np.reshape(Bflat,(duljina_MPS,duljina,duljina_MPS))
    
    
            
    
    a,b,c=Bflat.shape
    Bflat=0*Bflat
    for i in range(a):
        Bflat[i,index_a,i]=1

    #print(np.einsum('abc,b',Bflat,W))
    #quit()
  
  
    
    #define left environment
    environment=Array.from_ndarray( Bflat,
                        legcharges,
                        dtype=np.float64,
                        qtotal=None,
                        cutoff=None,
                        labels=labels,
                        raise_wrong_sector=True,
                        warn_wrong_sector=True)
    #print(environment)
    print("vacuum left environment is loaded",'..'*20)
    #print(environment.qtotal)
    
    return environment


def vacuum_left_environment(psi_L, H_MPO):
    """left environment of the vacuum LHS: identity channel of the first W of the segment MPO"""
    W = H_MPO._W[0]
    return set_left_environment_to_vacuum(W, W.get_leg('wL'), psi_L._B[0].get_leg('vL'))


def vacuum_left_environment_from_mpo(H_MPO):
    """
    left environment of the vacuum LHS, needs only the segment MPO:
    the MPS leg is a single sector with charge 0, the same leg as the left leg of psi_L and psi_total
    """
    W = H_MPO._W[0]
    chinfo = W.chinfo
    leg_mps = LegCharge.from_qflat(chinfo, np.zeros((1, chinfo.qnumber), dtype=int), qconj=1).bunch()[1]
    return set_left_environment_to_vacuum(W, W.get_leg('wL'), leg_mps)


def vacuum_lhs_window(sites, half, target, n_left=0):
    """
    Vacuum LHS for a run with added sites (a window around the interface): no pad, the window absorbs the charge.

    The vacuum is a product state of empty sites. The sites of the window carry random charge-conserving tensors
    (patch_mps), so they can hold more or fewer particles than the neutral number: a window of m sites with j filled
    sites has charge  q*j - p*m, symmetric around the neutral filling. The RHS sectors that are kept are those with
    charge  target + (charge of the window).
    The merge shift is  target - Q_vac , as for the pad (the total charge is  Q_vac - target).

    n_left: number of window sites to the LEFT of the interface `half`. They replace the last n_left sites of the
    vacuum (sites half-n_left ... half-1), so the vacuum product state lives on sites[:half-n_left] only.
    Q_vac is still the charge of the `half` empty sites, so for a fixed half and target the total charge sector is
    the same for every window (n_left = 0 is the old behaviour: the whole window is to the right of the interface).

    returns psi_L (on sites[:half-n_left]) and the shift to pass to patch_mps (with num_added = TOTAL number of
    window sites, left + right)
    """
    if n_left < 0 or n_left > half:
        raise ValueError('n_left = ' + str(n_left) + ' must be between 0 and the interface position ' + str(half))
    Q_vac = np.sum([s.leg.to_qflat()[0] for s in sites[:half]], axis=0)     # charge of `half` empty sites
    psi_L = product_state(sites[:half - n_left], [])
    shift = np.asarray(target) - Q_vac
    print('vacuum with a window of added sites, no pad. Q_vac =', Q_vac.tolist(), ', merge shift = target - Q_vac =', shift.tolist())
    print('window sites to the left of the interface:', n_left, ', vacuum on sites 0 ..', half - n_left - 1)
    return psi_L, shift


#####################################
#### product-state window          ##
#####################################
def window_configuration(x_first, m, j, S):
    """
    j distinct sites among x_first, ..., x_first+m-1 whose positions add up to S (None if there is no such choice).
    The empty sites are spread over the particles as evenly as possible (c_i = number of empty sites to the left of
    particle i grows linearly with i), so the result is a dense-to-dilute strip and not one solid block.
    """
    E = m - j
    if j < 0 or j > m:
        return None
    if j == 0:
        return [] if S == 0 else None
    D = S - (j * x_first + j * (j - 1) // 2)       # = sum_i c_i
    if D < 0 or D > j * E:
        return None
    if j == 1:
        c = np.array([D])
    else:
        a = min(E / (j - 1.), 2. * D / (j * (j - 1.)), 2. * (j * E - D) / (j * (j - 1.)))
        b = D / float(j) - a * (j - 1) / 2.
        c = np.clip(np.rint(b + a * np.arange(j)).astype(int), 0, E)
        diff = D - int(c.sum())
        while diff > 0:                              # raise the last entry that has room (stays non-decreasing)
            i = int(np.max(np.where(c < E)[0]))
            c[i] += 1
            diff -= 1
        while diff < 0:                              # lower the first non-zero entry (stays non-decreasing)
            i = int(np.min(np.where(c > 0)[0]))
            c[i] -= 1
            diff += 1
    pos = [int(x_first + i + c[i]) for i in range(j)]
    assert len(set(pos)) == j and sum(pos) == S and pos[0] >= x_first and pos[-1] < x_first + m
    return pos


def vacuum_lhs_product_window(sites, half, psi_R, target, n_left, n_right):
    """
    Window around the interface as a PRODUCT STATE (bond dimension 1), for large K where the random window of
    vacuum_lhs_window becomes too expensive (its dense tensors grow like m^3 in the number of window sites m).

    Window = sites half-n_left ... half+n_right-1 (n_left sites replace the end of the vacuum), psi_R starts at
    half+n_right. Everything to the left of psi_R is one product state: empty except for the filled window sites.
    The charge sector of the run is the same as in vacuum_lhs_window for the same half and target:
    total charge Q_vac(half) - target. Among the RHS sectors s that some filling of the window can reach
    (s = target + charge of the window, see vacuum_lhs_window) the one with the largest Schmidt weight is used.

    returns psi_L (on sites[:half+n_right]), psi_R (projected on the chosen sector), the shift for patch_mps
    (call patch_mps with num_added = 0), and a dict with the filled sites
    """
    start = half + n_right
    m = n_left + n_right
    x_first = half - n_left
    sites_L = sites[:start]
    qphys = [s.leg.to_qflat() for s in sites_L]
    dq = np.array([q[1] - q[0] for q in qphys])
    if dq.shape[1] != 2 or len(set(dq[:, 1].tolist())) != 1:
        raise ValueError('the product-state window is implemented for one layer with charges (K, N)')
    dN = int(dq[0, 1])
    # the sites of the window to the right of the interface count with their absolute charge
    Q_empty_R = np.sum([q[0] for q in qphys[half:]], axis=0)
    target_eff = np.asarray(target) + Q_empty_R

    q_R = psi_R._B[0].get_leg('vL').to_qflat()
    S_R0 = np.asarray(psi_R._S[0])
    best = None
    for jr, row in enumerate(q_R):
        Qex = row - target_eff                      # charge the filled window sites have to add to the empty ones
        if Qex[1] % dN or Qex[0] % dN:
            continue
        cfg = window_configuration(x_first, m, int(Qex[1] // dN), int(Qex[0] // dN))
        if cfg is None:
            continue
        if best is None or S_R0[jr] > best[0]:
            best = (S_R0[jr], jr, cfg)
    if best is None:
        raise ValueError('no filling of the window (' + str(m) + ' sites from ' + str(x_first) + ') reaches the target charge '
                         + str(np.asarray(target).tolist()) + ', use more added sites. Sectors on the RHS left leg: '
                         + str(np.unique(q_R, axis=0)[:30].tolist()))
    _, j, filled = best
    print('product-state window: sites', x_first, '...', start - 1, ', filled', len(filled), 'sites:', filled)
    print('RHS sector', q_R[j].tolist(), ', Schmidt weight', float(S_R0[j]))

    psi_L = product_state(sites_L, filled)
    Q_left = psi_L._B[-1].get_leg('vR').to_qflat()[0]
    shift = q_R[j] - Q_left
    print('vacuum + product-state window, Q_left =', Q_left.tolist(), ', merge shift =', shift.tolist())
    psi_R = project_left_leg(psi_R, [j])
    return psi_L, psi_R, shift, dict(filled=filled, j=j)