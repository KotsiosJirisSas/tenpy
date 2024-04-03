"""TODO module docstring"""
from __future__ import annotations
from typing import Iterator
import numpy as np
from .symmetries import Symmetry, Sector, SectorArray, FusionStyle
from .spaces import ProductSpace, VectorSpace

__all__ = ['FusionTree', 'fusion_trees', 'all_fusion_trees', 'allowed_coupled_sectors']
# TODO write tests!


class FusionTree:
    r"""A fusion tree, which represents the map from uncoupled to coupled sectors.

    .. warning ::
        Should think of FusionTrees as immutable.
        Do not act on their attributes with inplace operations, unless you know *exactly* what you
        are doing.

    TODO expand docstring. maybe move drawing to module level docstring.

    Example fusion tree with
        uncoupled = [a, b, c, d]
        are_dual = [False, True, True, False]
        inner_sectors = [x, y]
        multiplicities = [m0, m1, m2]

    |    |
    |    coupled
    |    |
    |    m2
    |    |  \
    |    x   \
    |    |    \
    |    m1    \
    |    |  \   \
    |    y   \   \
    |    |    \   \
    |    m0    \   \
    |    |  \   \   \
    |    a   b   c   d
    |    |   |   |   |
    |    |   Z   Z   |
    |    |   |   |   |
    

    """

    def __init__(self, symmetry: Symmetry,
                 uncoupled: SectorArray | list[Sector],  # N uncoupled sectors
                 coupled: Sector,
                 are_dual: np.ndarray | list[bool],  # N flags: is there a Z isomorphism below the uncoupled sector
                 inner_sectors: SectorArray | list[Sector],  # N - 2 internal sectors
                 multiplicities: np.ndarray | list[int] = None,  # N - 1 multiplicity labels; all 0 per default
                 ):
        # OPTIMIZE demand SectorArray / ndarray (not list) and skip conversions?
        self.symmetry = symmetry
        self.uncoupled = np.asarray(uncoupled)
        self.num_uncoupled = len(uncoupled)
        self.num_vertices = num_vertices = max(len(uncoupled) - 1, 0)
        self.num_inner_edges = max(len(uncoupled) - 2, 0)
        self.coupled = coupled
        self.are_dual = np.asarray(are_dual)
        self.inner_sectors = np.asarray(inner_sectors)
        if multiplicities is None:
            multiplicities = np.zeros((num_vertices,), dtype=int)
        self.multiplicities = np.asarray(multiplicities)
        self.fusion_style = symmetry.fusion_style
        self.is_abelian = symmetry.is_abelian
        self.braiding_style = symmetry.braiding_style

    def test_sanity(self):
        assert self.symmetry.are_valid_sectors(self.uncoupled)
        assert self.symmetry.is_valid_sector(self.coupled)
        assert len(self.are_dual) == self.num_uncoupled
        assert len(self.inner_sectors) == self.num_inner_edges
        assert self.symmetry.are_valid_sectors(self.inner_sectors)
        assert len(self.multiplicities) == self.num_vertices

        # check that inner_sectors and multiplicities are consistent with the fusion rules
        for vertex in range(self.num_vertices):
            # the two sectors below this vertex
            a = self.uncoupled[0] if vertex == 0 else self.inner_sectors[vertex - 1]
            b = self.uncoupled[vertex + 1]
            # the sector above this vertex
            c = self.inner_sectors[vertex] if vertex < self.num_inner_edges else self.uncoupled
            N = self.symmetry.n_symbol(a, b, c)
            # this also checks if a and b can fuse to c. If they can not, we have N == 0.
            assert 0 <= self.multiplicities[vertex] < N

    def __hash__(self) -> int:
        if self.fusion_style == FusionStyle.single:
            # inner sectors are completely determined by uncoupled, all multiplicities are 0
            unique_identifier = (self.are_dual, self.coupled, self.uncoupled)
        elif self.fusion_style == FusionStyle.multiple_unique:
            # all multiplicities are 0
            unique_identifier = (self.are_dual, self.coupled, self.uncoupled, self.inner_sectors)
        else:
            unique_identifier = (self.are_dual, self.coupled, self.uncoupled, self.inner_sectors, self.multiplicities)

        return hash(unique_identifier)

    def __eq__(self, other) -> bool:
        if not isinstance(other, FusionTree):
            return False
        return all(
            np.all(self.coupled == other.coupled),
            np.all(self.uncoupled == other.uncoupled),
            np.all(self.inner_sectors == other.inner_sectors),
            np.all(self.multiplicities == other.multiplicities),
        )

    def __str__(self) -> str:
        raise NotImplementedError  # TODO review, include are_dual!
        uncoupled_str = '(' + ', '.join(self.symmetry.sector_str(a) for a in self.uncoupled) + ')'
        entries = [f'{self.symmetry.sector_str(self.coupled)} ⟵ {uncoupled_str}']
        if self.fusion_style in [FusionStyle.multiple_unique, FusionStyle.general] and self.num_inner_edges > 0:
            inner_sectors_str = ', '.join(self.symmetry.sector_str(x) for x in self.inner_sectors)
            entries.append(f'({inner_sectors_str})')
        if self.fusion_style == FusionStyle.general and self.num_vertices > 0:
            mults_str = ', '.join(self.multiplicities)
            entries.append(f'({mults_str})')
        entries = ', '.join(entries)
        return f'FusionTree[{str(self.symmetry)}]({entries})'

    def __repr__(self) -> str:
        return (f'FusionTree({self.symmetry}, {self.uncoupled}, {self.coupled}, {self.are_dual}, '
                f'{self.inner_sectors}, {self.multiplicities})')

    def copy(self, deep=False) -> FusionTree:
        """Return a shallow (or deep) copy."""
        if deep:
            return FusionTree(self.symmetry, self.uncoupled.copy(), self.coupled.copy(),
                              self.are_dual.copy(), self.inner_sectors.copy())
        return FusionTree(self.symmetry, self.uncoupled, self.coupled, self.are_dual,
                          self.inner_sectors)

    def insert(self, t2: FusionTree) -> FusionTree:
        """Insert a tree `t2` below the first uncoupled sector.

        See Also
        --------
        insert_at
            Inserting at general position
        split
        """
        return FusionTree(
            symmetry=self.symmetry,
            uncoupled=np.concatenate([t2.uncoupled, self.uncoupled[1:]]),
            coupled=self.coupled,
            are_dual=np.concatenate([t2.are_dual, self.are_dual[1:]]),
            inner_sectors=np.concatenate([t2.inner_sectors, self.uncoupled[:1], self.inner_sectors]),
            multiplicities=np.concatenate([t2.multiplicities, self.multiplicities])
        )
        
    def insert_at(self, n: int, t2: FusionTree) -> dict[FusionTree, complex]:
        r"""Insert a tree `t2` below the `n`-th uncoupled sector.

        The result is (in general) not a canonical tree::

            TODO draw
        
        We transform it to canonical form via a series of F moves.
        This yields the result as a linear combination of canonical trees.
        We return a dictionary, with those trees as keys and the prefactors as values.

        Parameters
        ----------
        n : int
            The position to insert at. `t2` is inserted below ``t1.uncoupled[n]``.
            We must have have ``self.are_dual[n] is False``, as we can not have a Z between trees.
        t2 : :class:`FusionTree`
            The fusion tree to insert

        Returns
        -------
        coefficients : dict
            Trees and coefficients that form the above map as a linear combination.
            Abusing notation (``FusionTree`` instances can not actually be scaled or added),
            this means ``map = sum(c * t for t, c in coefficient.items())``.

        See Also
        --------
        insert
            The same insertion, but restricted to ``n=0``, and returns that tree directly, no dict.
        split
        """
        assert self.symmetry == t2.symmetry
        assert np.all(self.uncoupled[n] == t2.coupled)
        assert not self.are_dual[n]

        if t2.num_vertices == 0:
            # t2 has no actual fusion, it is either identity or a Z iso
            if t2.are_dual[0]:
                res = self.copy()
                res.are_dual = self.are_dual.copy()
                res.are_dual[n] = True
                return {res: 1}
            return {self: 1}

        if self.num_vertices == 0:
            return {t2: 1}

        if n == 0:
            # result is already a canonical tree -> no need to do F moves
            return {self.insert(t2): 1}
        
        if t2.num_vertices == 1:
            # inserting a single X tensor
            raise NotImplementedError # TODO
            # - can assume n > 0
            # - do F moves right to left

        # remaining case: t1 has at least 1 vertex and t2 has at least 2.
        # recursively insert: split t2 into a 1-vertex tree and a rest.
        raise NotImplementedError # TODO

    def split(self, n: int) -> tuple[FusionTree, FusionTree]:
        """Split into two separate fusion trees.

        TODO cartoon?

        Parameters
        ----------
        n : int
            Where to split. Must fulfill ``2 <= n < self.num_uncoupled``.

        Returns
        -------
        t1 : :class:`FusionTree`
            The part that fuses the ``uncoupled_sectors[:n]`` to ``inner_sectors[n - 2]``
        t2 : :class:`FusionTree`
            The part that fuses ``inner_sectors[n - 2]`` and ``uncoupled_sectors[n:]``
            to ``coupled``.

        See Also
        --------
        insert
        """
        if n < 2:
            raise ValueError('Left tree has no vertices (n < 2)')
        if n >= self.num_uncoupled:
            raise ValueError('Right tree has no vertices (n >= num_uncoupled)')
        cut_sector = self.inner_sectors[n - 2]
        t1 = FusionTree(
            self.symmetry,
            uncoupled=self.uncoupled[:n],
            coupled=cut_sector,
            are_dual=self.are_dual[:n],
            inner_sectors=self.inner_sectors[:n - 2],
            multiplicities=self.multiplicities[:n - 1],
        )
        t2 = FusionTree(
            self.symmetry,
            uncoupled=np.concatenate([cut_sector[None, :], self.uncoupled[n:]]),
            coupled=self.coupled,
            are_dual=np.insert(self.are_dual[n:], 0, False),
            inner_sectors=self.inner_sectors[n - 1:],
            multiplicities=self.multiplicities[n - 1:],
        )
        return t1, t2
            

class fusion_trees:
    """Iterator over all :class:`FusionTree`s with given uncoupled and coupled sectors.

    This custom iterator has efficient implementations of ``len`` and :meth:`index`, which
    avoid generating all intermediate trees.
    """
    def __init__(self, symmetry: Symmetry, uncoupled: SectorArray | list[Sector], coupled: Sector,
                 are_dual=None):
        # DOC: coupled = None means trivial sector
        self.symmetry = symmetry
        self.uncoupled = np.asarray(uncoupled)  # OPTIMIZE demand SectorArray (not list) and skip?
        self.num_uncoupled = num_uncoupled = len(uncoupled)
        self.coupled = coupled
        if are_dual is None:
            are_dual = np.zeros((num_uncoupled,), bool)
        else:
            are_dual = np.asarray(are_dual)
        self.are_dual = are_dual

    def __iter__(self) -> Iterator[FusionTree]:
        if len(self.uncoupled) < 1:
            raise RuntimeError
        if len(self.uncoupled) == 1:
            yield FusionTree(self.symmetry, self.uncoupled, self.coupled, self.are_dual, [], [])
        elif len(self.uncoupled) == 2:
            # OPTIMIZE does handling of multiplicities introduce significant overhead?
            #          could do a specialized version for multiplicity-free fusion
            for mu in range(self.symmetry._n_symbol(*self.uncoupled, self.coupled)):
                yield FusionTree(self.symmetry, self.uncoupled, self.coupled, self.are_dual, [], [mu])
        else:
            a1 = self.uncoupled[0]
            a2 = self.uncoupled[1]
            for b in self.symmetry.fusion_outcomes(a1, a2):
                uncoupled = np.concatenate([b[None, :], self.uncoupled[2:]])
                # set multiplicity index to 0 for now. will adjust it later.
                left_tree = FusionTree(self.symmetry, self.uncoupled[:2], b, self.are_dual[:2],
                                       [], [0])
                for rest_tree in fusion_trees(self.symmetry, uncoupled, self.coupled):
                    tree = rest_tree.insert(left_tree)
                    for mu in range(self.symmetry._n_symbol(a1, a2, b)):
                        res = tree.copy()
                        res.multiplicities = res.multiplicities.copy()
                        res.multiplicities[0] = mu
                        yield res

    def __len__(self) -> int:
        # OPTIMIZE caching ?
        
        if len(self.uncoupled) == 1:
            return 1

        if len(self.uncoupled) == 2:
            return self.symmetry._n_symbol(*self.uncoupled, self.coupled)

        a1 = self.uncoupled[0]
        a2 = self.uncoupled[1]
        count = 0
        for b in self.symmetry.fusion_outcomes(a1, a2):
            uncoupled = np.concatenate([b[None, :], self.uncoupled[2:]])
            num_subtrees = len(fusion_trees(self.symmetry, uncoupled, self.coupled))
            count += self.symmetry._n_symbol(a1, a2, b) * num_subtrees
        return count

    def index(self, tree: FusionTree) -> int:
        # TODO check compatibility first (same symmetry, same uncoupled, same coupled)
        # TODO inefficient dummy implementation, can exploit __len__ of iterator over subtrees
        # to know how many we need to skip.
        for n, t in enumerate(self):
            if t == tree:
                return n
        raise ValueError(f'tree not in {self}: {tree}')


def all_fusion_trees(space: VectorSpace, coupled: Sector = None) -> Iterator[FusionTree]:
    """Yield all fusion trees from the uncoupled sectors of space to the given coupled sector
    (if not None) or to all possible coupled sectors (default)"""
    raise NotImplementedError  # TODO consider duality!
    if coupled is None:
        for coupled in allowed_coupled_sectors(space):
            yield from all_fusion_trees(space, coupled=coupled)
    else:
        for uncoupled in space.sectors:
            yield from fusion_trees(uncoupled, coupled)


def allowed_coupled_sectors(codomain: ProductSpace, domain: ProductSpace) -> SectorArray:
    """The coupled sectors which are admitted by both codomain and domain"""
    raise NotImplementedError  # TODO think about duality!
    codomain_coupled = codomain._non_dual_sectors
    domain_coupled = domain._non_dual_sectors
    # OPTIMIZE: find the sectors which appear in both codomain_coupled and domain_coupled
    #  can probably be done much more efficiently, in particular since they are sorted.
    #  look at np.intersect1d for inspiration?
    are_equal = codomain_coupled[:, None, :] == domain_coupled[None, :, :]  # [c_codom, c_dom, q]
    mask = np.any(np.all(are_equal, axis=2), axis=0)  # [c_dom]
    return domain_coupled[mask]