# Segment DMRG for quantum Hall interfaces, excitations and vacuum edges

Finite-segment DMRG built from converged iDMRG states. The segment is assembled from a left piece, an optional
window of added sites, and a right piece. The two ends of the segment are attached to environments computed from the
infinite states (or to the vacuum).

```
|<-- interface_position -->|<- added_sites ->|<------ right piece ------>|
[ left piece / vacuum      ][ window (random) ][ right piece (iMPS)       ]
 left environment                                             right environment
 (site 0 of the segment)                                      (last site of the segment)
```

Three modes, selected by `params['type_calculation']`:

| mode | left piece | right piece | needs `mps_2` | use |
|---|---|---|---|---|
| `excitation` | iMPS `mps` | the same `mps` | no | charged excitations / charge gaps in one state |
| `interface` | iMPS `mps` | a second iMPS `mps_2` | **yes, only here** | interface between two different states |
| `vacuum_interface` | empty (all 0) | iMPS `mps` | no | edge of a state against vacuum |

## Files

| file | purpose |
|---|---|
| `multilayer_segment_final.py` | driver: command line, builds the MPO, merged MPS, environments, runs DMRG |
| `multilayer_creating_MPS_generic_interface_minimal.py` | `QH_system`: loads the iMPS, builds the sites, cuts the pieces, `patch_WFs` |
| `patch_mps.py` | merges left piece, added sites and right piece (charge matching and projection) |
| `vacuum_interface.py` | vacuum mode: pad search, window, vacuum left environment |
| `create_segment_model.py` | segment MPO (`create_segment_DMRG_model`) from `model_par` and the graph |
| `create_env.py` | `calculate_environment`: environment of the infinite MPS at a given site |
| `graph_potential.py` | builds the long graph with a site-dependent potential V(i) |
| `sanity_checks.py` | checks the particle number of the initial state |
| `jobs/create_jobs.py` | writes `jobs.sh` for a queue (excitation mode) |

Not in this folder but required on `sys.path`: the TeNPy fork (`QH_MultilayerFermionSite_final`, `QH_G2MPO`,
`QH_Graph_final`) and `QHE_run_from_graph.py` (provides `make_iMPO`).

## Usage

```
python3 multilayer_segment_final.py <added_sites> <name_load> <name_save> <L> <interface_position> [shift ...]
```

| argument | meaning |
|---|---|
| `added_sites` | number of sites in the window between the left piece and the right piece (0 = no window) |
| `name_load` | graph file **without** `.pkl` (the script appends it) |
| `name_save` | output file without `.pkl` |
| `L` | number of repetitions of the unit-cell graph, the segment has `L * len(graph)` sites |
| `interface_position` | end of the left piece in sites (in `vacuum_interface`: the length of the empty vacuum) |
| `shift ...` | integers, one per charge component, see Charges |

The right piece starts at `interface_position + added_sites`. The right environment is taken at the site
`right_site = L_sites - 1 - (interface_position + added_sites)` of the infinite MPS, with
`L_sites = L * len(graph)`.

The mode and the wavefunction file are set in `run_excitation_dmrg`, not on the command line:

```python
params['type_calculation'] = "vacuum_interface"      # or "excitation" or "interface"
WF_path = '.../<name>_MPS_data_new_tenpy.pkl'          # wavefunction file (hardcoded)
```

Example (`vacuum_interface`, 3 added sites, ν = 2/3, one layer, shift (K, N) = (0, -3)):

```
python3 multilayer_segment_final.py 3 /path/graph_file /path/out 60 20 0 -3
```

## Input files

Two pickles are read in every mode: **file A** (graph file, `name_load`) and **file B** (wavefunction file, `WF_path`).

| | `excitation` | `interface` | `vacuum_interface` |
|---|---|---|---|
| file A | `graph`, `model_par` | `graph`, `model_par` | `graph`, `model_par` |
| file B | `mps`, `model_par` | `mps`, **`mps_2`**, `model_par` | `mps`, `model_par` |
| left piece / left environment | `mps` | `mps` | vacuum, built in code |
| right piece / right environment | `mps` | `mps_2` | `mps` |
| shift | the charge to insert | given, or omitted to be computed (`added_sites=0` only) | the target charge, required |
| `added_sites` | any | any | any: 0 = pad, > 0 = window |

### File A (graph file)

| key | content |
|---|---|
| `graph` | MPO graph of **one unit cell**, repeated `L` times. A list with one entry per site, each a nested dict `graph[i][state_left][state_right] = [(opname, strength), ...]`. The start state is `'R'` and the end state is `'F'` (both have identity self-loops). |
| `model_par` | dict with `layers` (its length is `N_layer`), `root_config`, `cons_K`, `cons_C`, and `order_of_species` if `cons_C == 'species'` |

`calculate_environment` reads this file with `encoding='latin1'` and the driver without, so it must load both ways
(a Python 3 pickle does).

### File B (wavefunction file)

| key | needed in | content |
|---|---|---|
| `mps` | all modes | infinite TeNPy 3 MPS (`bc='infinite'`) from iDMRG |
| `mps_2` | `interface` only | second infinite MPS, same requirements as `mps`, right side |
| `model_par` | all modes | used to build the infinite MPO for the environments |
| `dmrg_params` | optional | bookkeeping |

Requirements for each iMPS: canonical form, one unit cell of `psi.L` sites with `site_loc = 0, ..., L-1`, the
conservation laws of `model_par`, charges ordered K first. `make_iMPO` repeats the graph `psi.L // N_layer` times,
so the unit-cell graph then has to have `psi.L` entries. `model_par` in A and B must agree in `layers`, `root_config`,
`cons_K`, `cons_C`.

For `interface`, two iDMRG runs give two files, so merge them once:

```python
import pickle
a = pickle.load(open('state_left.pkl', 'rb'))
b = pickle.load(open('state_right.pkl', 'rb'))
pickle.dump({'mps': a['mps'], 'mps_2': b['mps'], 'model_par': a['model_par']}, open('interface_states.pkl', 'wb'))
```

Both states must use the same `layers`, `cons_K` and `cons_C`.

### Output (`<name_save>.pkl`)

```
{ 'dmrg_params', 'energy', 'model_par', 'density', 'entanglement_entropy', 'entanglement_spectrum' }
```

## Charges

The conservation laws come from `model_par` everywhere (the MPS sites, the segment MPO and `make_iMPO`), otherwise
TeNPy refuses to combine the legs (`leg has different ChargeInfo`, e.g. `'N'` against `'N1'`):

| `cons_K` | `cons_C` | `conserve` | charge components |
|---|---|---|---|
| True | `'total'` | `('N','K')` | 2: (K, N) |
| True | `'each'` | `('each','K')` | 1 + N_layer: (K, n_1, ..., n_N_layer) |
| True | `'species'` | `('species','K')` | as above, with `order_of_species` |
| False | any | `cons_C` | no K |

Charges are **relative to the root configuration**. With `p` particles in `q = root_config.size` entries:

- a site has N charge `C = q*n - p` (`n` = occupation), and K charge `K = (orbital index) * C`,
- so **one particle is `q` units of N charge**, and an empty site has charge `-p`: an empty vacuum has a nonzero charge `Q_vac`,
- in the data the order is K first (column 0), then N (or the per-layer n),
- translating a state by `T` sites changes every row of a leg as `K -> K + N*T` (N here is the N charge of that row),
  with `T = (interface_position + added_sites) // N_layer` for the right piece.

### The shift

The shift is one integer per charge component, passed on the command line. Together with `interface_position` it sets
the charge sector of the whole run. In `patch_WFs` the charges of the right piece are shifted by `-shift`.

- A **negative N entry adds particles, a positive one removes them** (checked for N = -3, q = 3: the measured
  change was +1.001 particles in `vacuum_interface`).
- If your script converts K with `shift[0] += interface_position * np.sum(shift[1:])`, then K is given relative to the interface.
- **Fixed `interface_position`, varying `added_sites`:** the charge sector is the same, because the merge shift
  does not depend on `added_sites`. `added_sites` only changes the **initial wavefunction**, not the charges. This holds for
  `vacuum_interface`, which I checked with stand-in objects; for `excitation` and `interface` it has not been analyzed.

### Environments and `translate_and_rebind`

The environments are computed in the frame of the infinite MPS. `translate_and_rebind` (in the driver) checks that
each environment leg equals the segment leg translated by `K -> K + N*T` plus one constant offset, and then replaces
the environment legs by the segment's own legs (the data is unchanged). The MPO leg offset should be `[0, 0]`.
The MPS leg offset is minus the charge of the shift. Neither MPO is sorted by leg charge (no `sort_legcharges`);
keep it that way on both sides, otherwise the bonds hold the same states in a different order.

## The three modes

### `excitation`

- `mps` is used for both sides; `mps_2` is not read.
- `shift` is the charge inserted at the interface.
- With `added_sites > 0` the window is filled with random charge-conserving tensors and relaxed by DMRG.
  With `added_sites = 0` there is no window: the last bond of the left piece and the left leg of the right piece are
  projected onto the matching sectors.
- Environments: left at site 0, right at `right_site`, both from `mps`.

```
python3 multilayer_segment_final.py 8 <graph_file> <out> 96 97 0 0 2
```

### `interface`

- Left piece and left environment from `mps`, right piece and right environment from `mps_2`
  (`calculate_environment(..., type_calculation='interface')`). Both states belong to one Hamiltonian.
- Without a shift on the command line, it is computed from the largest Schmidt values of the two pieces. This is only
  meaningful with `added_sites = 0`: with a window the Schmidt values of that side are random.

```
python3 multilayer_segment_final.py 0 <graph_file> <out> 40 20            # shift computed
python3 multilayer_segment_final.py 2 <graph_file> <out> 40 20 0 1 0       # shift given
```

### `vacuum_interface`

- The left piece is a product state of empty sites on the first `interface_position` sites, with left charge 0.
- The left environment is the identity channel of the first MPO tensor (`vacuum_left_environment_from_mpo(M.H_MPO)`).
  `translate_and_rebind` is not applied to it.
- `shift` is the **target charge**, required. The total number of particles is
  `nu * (L - interface_position) - N_target / q`, with `nu = mean(root_config)`.
- The merge shift is `target - Q_vac`, whatever `added_sites` is.

**`added_sites > 0` (window).** The window sits between the empty vacuum and the right piece and belongs to the right
side of the interface. Its sites carry random charge-conserving tensors, so a window of `m` sites with `j` filled
sites has charge `q*j - p*m`: it can hold more or fewer particles than the neutral number, so adding and removing
particles are symmetric. The sectors kept on the right piece's left leg are `target + q*j - p*m`. Size the window so
that the particle number you need lies between 0 and `m`.

**`added_sites = 0` (pad).**

1. If the right piece has a sector with charge `target` on its left leg, project onto it (largest Schmidt weight).
2. Otherwise pad the right end of the vacuum with an alternating `...0101` product state, every other orbital in
   each layer, packed against the cut. The pad has charge `Q_pad`, and the sector needed is `target + Q_pad`. The
   smallest pad that works is taken; among equals the largest Schmidt weight.
3. The vacuum plus pad is merged with `patch_mps`.

A pad can only add particles to an empty vacuum, so removing a particle needs a large pad. Prefer the window for
removal. `max_pad` and `max_offset` (slide the pad to the left by whole orbitals) are arguments of `patch_WFs`.
If no pad works, the error lists the available sectors on the right piece's left leg.

```
python3 multilayer_segment_final.py 0 <graph_file> <out> 40 20 K N      # pad
python3 multilayer_segment_final.py 3 <graph_file> <out> 40 20 K N      # window of 3 sites
```

Printed checks: the pad (or the window message), `Total Matching Pairs`, the charge at the right end, the excess
electrons and K sector, and the filling sanity check.

## Potential V(x)

`run_excitation_dmrg(name_load, name_save, L, interface_position, added_sites, shift, V_of_site)`. `V_of_site(i)` is
the on-site potential on site `i` of the segment (default `0*i`). It replaces `graph = graph*L`:

```python
graph = graph_with_potential(graph, L, V_of_site, mode='add', keys=('R','F'), op_name='nOp')
L = len(graph)
```

- The on-site term is the transition `graph[i]['R']['F'] = [('nOp', V(i))]`. The uniform model has no on-site term, so
  it is created where `V(i) != 0`. `graph*L` repeats the same dict, so every site is deep-copied first.
- Example: `V_of_site = lambda i: (1 - np.tanh((i//N_layer - x0)/w)) * U0/2` (a step from U0 on the left to 0 on the right).
- The environments are computed from the uniform Hamiltonian, so `V` must go to the iDMRG value (0 in `'add'` mode)
  at the right end of the segment (and at the left end in `excitation` and `interface`). In `vacuum_interface`
  the left end can carry a constant, because the left side is empty.
- `show_onsite(graph, site, 'nOp')` prints the transitions that contain an operator, to find the keys.

## Sanity check

```python
from sanity_checks import sanity_check_filling
sanity_check_filling(psi_total, model_par, L, interface_position, added_sites, shift, params['type_calculation'])
```

- `vacuum_interface`: the total number of particles must equal `nu*(L - interface_position) - N_shift/q`, `q = root_config.size`.
  It prints `OK` or `MISMATCH` (tolerance `tol=0.1`). Fractional deviations of about 1e-3 are normal, because the
  initial MPS is not an exact eigenstate of the particle number.
- Other modes: a table of the filling of the left piece, the window and the right piece.

## Steps of a run

1. Load graph and `model_par`, build the long graph (`graph_with_potential`), build the segment MPO.
2. `QH_system` loads `mps` / `mps_2`, builds the sites, and cuts the left piece and the right piece (its left leg is
   translated by `K -> K + N*T`).
3. Environments: right (and left, except in the vacuum mode) from the infinite MPS; vacuum environment in the vacuum mode.
4. `patch_WFs` merges the pieces in the charge sector set by the shift. This gives `psi_total`.
5. Sanity check, then `translate_and_rebind` for each environment.
6. `TwoSiteDMRGEngine(psi_total, M, ..., resume_data={'init_env_data': ...})`.

## Troubleshooting

| message | meaning |
|---|---|
| `dropping init_LP/init_RP with incompatible MPO legs` | the environment's MPO leg differs from the segment MPO (order, states or charges); the environment is ignored and the run is invalid. Do not sort the leg charges of only one of the two MPOs. |
| `dropping ... incompatible MPS legs` | the environment's MPS leg does not match the left/right leg of `psi_total` |
| `leg has different ChargeInfo` | MPS sites and MPO sites use different conservation laws; both must come from `model_par` |
| `wrong sector with non-zero entries` | the environment data is not consistent with the new leg charges (orientation `qconj`, or different states) |
| `no matching charge sectors between LHS and RHS` | the shift is not reachable with this window / left piece; change the shift, or make the window longer |
| `no pad with up to N particles reaches the target charge` | pad mode: choose a target from the listed sectors, raise `max_pad` / `max_offset`, or use a window |
| `MISMATCH` in the sanity check | the particle number differs from `nu*(L-interface_position) - N_shift/q`; check the sign and units (N entries are in units of `q`) |

## Before a production run

- `params['type_calculation']` and `WF_path` are hardcoded in `run_excitation_dmrg`.
- `run_excitation_dmrg` calls `quit()` right after the DMRG run, so the results are not saved. Remove it to store the output.
- `N_layer` is hardcoded to 1 in the line that defines `V_of_site` in the driver.
- `patch_WFs` writes a small log into `save_dir` (here the current directory) when the excess charge is below 0.01.
- `jobs/create_jobs.py` writes excitation-mode jobs with 3-component charges; adapt it to your number of charge components.
- The module parameter is still called `half`. In the driver it is `interface_position`: pass it as `params['half'] = interface_position`.

## What was checked and what was not

The patching, the pad search, the window and the sanity check were tested with stand-in TeNPy objects (charge
conservation along the merged chain, matching, bookkeeping). On the real fork, the N sign (`-N_shift/q`) was confirmed
for one shift with no pad and no K part. Not yet verified on a real run: the K part of a shift, a case where the pad is
actually used, the window runs, and the claim that the sector is independent of `added_sites` for `excitation` and `interface`.
