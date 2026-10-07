"""
Position dependent on-site potential V(i), built into the MPO graph.

The MPO graph is a list with one entry per site. Entry i is a nested dict

    graph[i][key_left][key_right] = [(opname, strength), ...]

(the transition from the MPO state `key_left` on the left bond to `key_right` on the right bond).
The on-site term of a site is the transition from the "nothing yet" state to the "all done" state, in the
usual MPO convention  IdL -> IdR  with the number operator.  A potential V(i) is the strength of that term.
"""
import copy
import numpy as np


def graph_with_potential(graph_cell, n_cells, V, mode='add', keys=('IdL', 'IdR'), op_name=None):
    """
    Replacement for  graph = graph*L  with a potential built in.

    graph_cell: the graph of one unit cell (list, one entry per site)
    n_cells:    number of repetitions (the L of graph*L), the system has len(graph_cell)*n_cells sites
    V:          function V(i) -> value on site i (i = 0, 1, ... is the site index in the segment), or an array
                with one value per site
    mode:       'add'     V is added to the existing on-site term (e.g. the uniform chemical potential of the iDMRG)
                'replace' the existing on-site term is overwritten by V
    keys:       (left key, right key) of the on-site transition in the graph
    op_name:    name of the on-site operator; if None, the operator of the existing on-site term is used

    The environments of the segment are computed from the uniform Hamiltonian, so V should go to the value of
    the iDMRG Hamiltonian (V -> 0 in mode 'add') at both ends of the segment.
    """
    if mode not in ('add', 'replace'):
        raise ValueError("mode must be 'add' or 'replace'")
    n_cell = len(graph_cell)
    L = n_cell * n_cells
    # graph*L repeats the SAME dict object L times, changing one site would change all of them -> copy every site
    graph = [copy.deepcopy(graph_cell[i % n_cell]) for i in range(L)]

    values = np.array([V(i) for i in range(L)], dtype=float) if callable(V) else np.asarray(V, dtype=float)
    if len(values) != L:
        raise ValueError('V has ' + str(len(values)) + ' values, the system has ' + str(L) + ' sites')

    key_l, key_r = keys
    for i, site in enumerate(graph):
        v = float(values[i])
        terms = site.get(key_l, {}).get(key_r)
        if terms is None:
            if v == 0:
                continue                      # nothing to add and nothing there
            if op_name is None:
                raise KeyError('site %d has no on-site term %s -> %s, give op_name to create it' % (i, key_l, key_r))
            site.setdefault(key_l, {})[key_r] = [(op_name, v)]
            continue

        names = [t[0] for t in terms]
        name = op_name if op_name is not None else (names[0] if len(names) == 1 else None)
        if name is None:
            raise KeyError('site %d: on-site term %s -> %s has several operators %s, give op_name' % (i, key_l, key_r, names))
        for k, t in enumerate(terms):
            if t[0] == name:
                new = v if mode == 'replace' else t[1] + v
                terms[k] = [name, new] if isinstance(t, list) else (name, new)
                break
        else:
            terms.append((name, v))
    return graph


def show_onsite(graph, site=0, op_name=None):
    """prints the transitions of one site that contain op_name (all transitions if None), to find the keys"""
    for key_l, row in graph[site].items():
        for key_r, terms in row.items():
            if op_name is None or any(t[0] == op_name for t in terms):
                print(repr(key_l), '->', repr(key_r), ':', terms)
