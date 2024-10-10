'''
Mainly copies from c_mps_mpo.py example.
I want to understand how exactly the leg charges work for an MPO W matrix.
'''



'''
    >>> s = tenpy.networks.site.SpinHalfSite(conserve='Sz')
    >>> Id, Splus, Sminus, Sz = s.Id, s.Sp, s.Sm, s.Sz
    >>> J = 1.
    >>> leg_wR = npc.LegCharge.from_qflat(s.leg.chinfo,
    ...                                   [op.qtotal for op in [Id, Splus, Sminus, Sz, Id]],
    ...                                   qconj=-1)
    >>> W_mpo = npc.grid_outer([[Id, Splus, Sminus, Sz, None],
    ...                         [None, None, None, None, J*0.5*Sminus],
    ...                         [None, None, None, None, J*0.5*Splus],
    ...                         [None, None, None, None, J*Sz],
    ...                         [None, None, None, None, Id]],
    ...                        grid_legs=[leg_wR.conj(), leg_wR],
    ...                        grid_labels=['wL', 'wR'])
    >>> W_mpo.shape
    (5, 5, 2, 2)
    >>> W_mpo.get_leg_labels()
    ['wL', 'wR', 'p', 'p*']
    """
'''

import sys
import os
sys.path.append('/home/v/vasiliou/tenpynew2/tenpy')
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.linalg import np_conserved as npc
import numpy as np



'''
First, lets analyse a single W matrix, rather than a full MPO = W^{[0]}W^{[1]} \dots W^{[N]}
'''
#initialize site
spin = SpinHalfSite(conserve="Sz")

####
chinfo = spin.leg.chinfo
print('chinfo',chinfo)
print('chinfo',[chinfo.make_valid()] * 6)
quit()
#####

#extract the operators from the site instance
Id, Sp, Sm, Sz = spin.Id, spin.Sp, spin.Sm, spin.Sz
#set parameter values
J, Delta, hz = 1., 1., 0.2
#create the W matrix of the bulk system. This is a list of lists. W[i] gives you the i'th row and W[i][j] gives you the physical operator(npc instance) of i'th row,  j'th column.
W_bulk = [[Id, Sp, Sm, Sz, -hz * Sz], 
          [None, None, None, None, 0.5 * J * Sm],
    [None, None, None, None, 0.5 * J * Sp], 
    [None, None, None, None, J * Delta * Sz],
          [None, None, None, None, Id]]
'''
W matrix structure:

                p*
                ^
                |
         wL ->- W ->- wR
                |
                ^
                p

We know the charge info of p,p^* legs as they match the charge info of the site instance. 
(Eg the basis state 'up' has charge +1 and the  basis state '-' has charge -1)
But what is the charge info of the auxilliary legs wR and wL?
For that, we need to call op.qtotal for the operators, to calculate their total charge.
'''
print('--'*100)
print('What is the charge structure of the physical legs?')
print(spin.leg.chinfo)
print('--'*100)
print('WHat are the total charges of the operators?')
for op in [Id, Sp, Sm, Sz, Id]:
    q_op = op.qtotal
    print('The total charge of operator is'+str(q_op))
print('--'*100)
'''
As expected, Id and Sz have charge 0 (both commute with w*Sz)
while Sp and Sm have charge \pm2 as they do not conserve Sz (flipping a spin from down to up means increasing 2*Sz by 2)
            W = npc.grid_outer(grids[i], [legs[i], legs[i + 1].conj()], Ws_qtotal[i], ['wL', 'wR'])
'''
print('Content of grid:')
print('')
print('')
grid_shape, entries = npc._nontrivial_grid_entries(W_bulk)
print('grid shape',grid_shape)
print('non-zero entries',entries)
print('')
print('')
idx, entry = entries[1]  # first non-trivial entry
print('First non trivial entry:')
print('index',idx)
print(entry)
print('--'*100)
'''
so this identifies non-zero (non-None) entries in the grid, and then finds the first non-trivial one. In our case, it is at indices (0,0) and it is the Id operator
'''
leg_wR = npc.LegCharge.from_qflat(spin.leg.chinfo,[op.qtotal for op in [Id, Sp, Sm, Sz, Id]],qconj=-1)
grid_legs=[leg_wR.conj(), leg_wR]
grid_labels=['wL', 'wR']
chinfo = entry.chinfo #ChargeInfo([1], ['2*Sz']) for Sz symmetry
dtype = np.result_type(*[e.dtype for _, e in entries])
legs = list(grid_legs) + entry.legs
labels = entry._labels[:]
labels = grid_labels + labels
print('legs:',legs)
print('labels:',labels)
print('--'*100)
'''
By this point, we grouped the physical and auxilliary legs and labels together
'''
print('')
print('')
grid_charges = [l.get_charge(l.get_qindex(i)[0]) for i, l in zip(idx, grid_legs)]
print('grid_charges',grid_charges)
#quit()
print('entry q tot',entry.qtotal)
qtotal = chinfo.make_valid(np.sum(grid_charges + [entry.qtotal], axis=0))
print('qtotal',qtotal)
'''
The above ^ is the part of the code that i cannot understand... l.get_qindex(i) says: At each leg, get the q_index from the flat index. 
Flat index is 0 in our case since the non-trivial element we have is Id which is at (0,0) of our grid. Then, the q_index is the block that this corresponds to. From this we can get the charge associated with that index.
Ie if the index 4 of the W grid corresponds to the 2nd charge block, then the charge of the 4th index will be the charge of the 2nd block.
'''

'''
Now at this point, one creates an empty npc array with the correct total charge, correct legs and correct labels, and fills it up with the elements from the grid
'''
res = npc.Array(legs, dtype, qtotal, labels)
# main work: iterate over all non-trivial entries to fill `res`.
for idx, entry in entries:
    res[idx] = entry  # insert the values with Array.__setitem__ partial slicing.
    if labels is not None and entry._labels != labels:
        labels = None
res.test_sanity()
quit()
