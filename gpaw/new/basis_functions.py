import numpy as np
from gpaw.gpu import cupy as cp, as_xp, cupy_is_fake
from functools import cached_property
from dataclasses import dataclass, field
from gpaw.spline import Spline
from gpaw.core import UGDesc
from gpaw.old.grid_descriptor import GridBoundsError
from gpaw.new.timer import trace
from gpaw import GPAW_NO_C_EXTENSION
from gpaw.setup import Setups, Setup
from typing import override, cast
from enum import Enum, auto
from abc import ABC, abstractmethod
import itertools
from collections import defaultdict

from gpaw.sphere.spherical_harmonics import Y

from typing import Protocol, runtime_checkable, TypeAlias

@runtime_checkable
class SplineLike(Protocol):
    """"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...


"""
Each atom has list of basis functions. A basis function is defined by its
angular quantum numbers l, m and a radial function f(r). We also require a
radial cutoff r_c so that f(r) = 0 for r >= r_c. Each basis function evaluates
to f(r) * Y_lm, r being the distance from the atom's center.

We use a combined index mu, or M, to enumerate basis functions of different
atoms and l,m: mu = {a, l, m}. In practice, allowed values of m depend on l so
we mainly use a and l when grouping basis functions. The radial function and
cutoff are the same for all m that share {a, l} indices. Mu is ordered such
that first are l=0 functions of atom a=0, then l=1 funcs of a=0, etc.

Class BasisFunctionCollection describes the full set of all basis functions
phi_mu and provides methods for calculating stuff with them.
"""


def sphere_overlaps_box(
    radius: float,
    center: np.ndarray,   # 3D array, real space pos of sphere center
    box_min: np.ndarray,  # 3D array, box "start" corner in real units
    box_max: np.ndarray,  # 3D array, box "end" corner in real units
    ) -> bool:
    """Exact sphere-box overlap check."""
    # closest point in the box to sphere center
    closest = np.maximum(box_min, np.minimum(center, box_max))
    dist2 = np.sum((closest - center)**2)
    return dist2 <= radius*radius


AtomTypeId: TypeAlias = int
"""Type alias used for labeling atom types"""

@dataclass
class BasisFunctionDesc:
    """Data for defining a basis function phi(x)."""

    l: int
    """Angular momentum quantum number."""
    cutoff: float
    """Cutoff for the radial part. phi(r) = 0 for r > cutoff"""
    f_r: np.ndarray
    """Discrete samples of the radial function f(r) on [0, cutoff].
    These values are used to construct a continuous spline. The
    final value at r = cutoff MUST be exactly zero."""

    @staticmethod
    def from_legacy_spline(spline: Spline) -> "BasisFunctionDesc":
        """"""
        breakpoints = np.linspace(0.0,
                                  spline.get_cutoff(),
                                  spline.get_npoints())
        f_r = spline.map(breakpoints)
        assert(f_r[-1] == 0.0)
        return BasisFunctionDesc(spline.get_angular_momentum_number(),
                                 spline.get_cutoff(),
                                 f_r)


@dataclass
class LFCAtomDefinition:
    id: AtomTypeId
    basis_functions: list[BasisFunctionDesc]

    def __hash__(self):
        return hash(self.id)


@dataclass
class LFCAtomInstance:
    type: LFCAtomDefinition
    position: np.ndarray
    """Real-space position of the atom. 3D array."""


@dataclass
class LFCSystem:
    grid: UGDesc
    atom_types: list[LFCAtomDefinition]
    atoms: list[LFCAtomInstance]

    def __post_init__(self):
       """"""
       # check that all atom IDs are unique
       ids = []
       for atom_type in self.atom_types:
            if atom_type.id in ids:
                raise ValueError("Atom type IDs must be unique")
            ids.append(atom_type.id)


BlockCoords: TypeAlias = tuple[int, int, int]

@dataclass
class GridBlock:
    """Blocks are identified by their 3D indices on a "grid of blocks", ie.
    block (0, 0, 0) starts at grid.start_c and extends BLOCK_SIZE points
    in xyz dirs. Block (1, 0, 0) starts at
        grid.start_c + BLOCK_SIZE * [1, 0, 0],
    and so on.

    If the MPI domain does not divide evenly into blocks, we allow the last
    blocks in each direction to be smaller as to not "overflow" into the next
    domain."""

    grid: UGDesc
    coords: BlockCoords

    # these give the range of grid points this block represents
    start_c: np.ndarray
    """Start indices to the (x,y,z) grid for this block. Inclusive."""
    end_c: np.ndarray
    """End indices to the (x,y,z) grid for this block. Exclusive!"""
    phi_j: list["BasisFunctionInstance"] = field(default_factory=list)
    """List of all basis functions that have overlap with this block.
    Note indexing: each phi_j actually contains many values of m.
    On periodic systems, this includes basis funcs from unit cells that extend
    to this cell."""
    M_m: list[int] = field(default_factory=list)
    """Maps block-local phi index 'm' to global mu"""
    evaluated_phi_mg: np.ndarray | None = None
    """phi_mu(x) precalculated on the grid block. NOTE: the XYZ shape can be
    smaller for some blocks if the grid blocking was uneven.
    See also get_block_shape()"""

    def __post_init__(self):
        """"""
        assert all(x >= 0 for x in self.coords), \
            "Block coords are relative to MPI domain and must be >= 0"
        assert np.all(self.end_c - self.start_c > 0)
        assert np.all(self.start_c >= self.grid.start_c)
        assert np.all(self.end_c <= self.grid.end_c)

    def get_block_xyz(self) -> np.ndarray:
        """Gives real-space XYZ points for the given block.
        4D array of shape (Nx, Ny, Nz, 3)."""
        # See UGDesc.xyz()
        shape = self.end_c - self.start_c
        indices_Rc = np.indices(tuple(shape)).transpose((1, 2, 3, 0))
        indices_Rc += self.start_c
        return indices_Rc @ (self.grid.cell_cv.T / self.grid.size_c).T

    def get_local_start_end_c(self) -> tuple[np.ndarray, np.ndarray]:
        """start_c and end_c for looping over block-local XYZ arrays."""
        local_start_c = np.asarray([0, 0, 0])
        local_end_c = self.end_c - self.start_c
        return local_start_c, local_end_c

    def get_block_shape(self) -> tuple[int, int, int]:
        """Gives real shape of the block."""
        return tuple(self.end_c - self.start_c)


class BasisFunction():
    """"""

    spline: SplineLike

    def __init__(self, desc: BasisFunctionDesc, dtype, use_gpu: bool = False):
        """"""
        assert not use_gpu
        assert dtype is np.float64

        assert desc.l >= 0
        assert desc.cutoff > 0
        assert len(desc.f_r) > 2, "Need at least 3 points to define a spline"
        self.l = desc.l
        self.cutoff = desc.cutoff

        self._init_spline(desc.f_r)

    def _init_spline(self, f_r: np.ndarray) -> None:
        """Initializes spline on the CPU.
        """

        from scipy.interpolate import CubicSpline
        self.spline = CubicSpline(
            np.linspace(0.0, self.cutoff, len(f_r), endpoint=True),
            as_xp(f_r, xp=np),
            bc_type='clamped',  # first derivatives are zero at boundaries
            extrapolate=False
        )

    def get_num_mu(self) -> int:
        """How many mu indices does this basis function span.
        In practice, returns 2*l + 1.
        """
        return 2 * self.l + 1

    def evaluate_spline(self, r: np.ndarray) -> np.ndarray:
        """"""
        # extrapolate=False means we get NaN for r > r_max,
        # so mask that region out. Could also use np.nan_to_num instead
        mask = (r < self.cutoff)
        out = np.zeros_like(r)
        out[mask] = self.spline(r[mask])
        return out


@dataclass
class BasisFunctionInstance:
    """Instance of a basis function with fixed position and index range."""
    phi_class: BasisFunction
    parent_atom: int
    first_mu: int
    position: np.ndarray
    """Real-space position"""
    phi_mG: np.ndarray | None = None
    """Precalculated values on the grid"""

    blocks_with_overlap: list[GridBlock] = field(default_factory=list)

    def get_cutoff(self) -> float:
        """Get the radial cutoff"""
        return self.phi_class.cutoff

    def get_angular_momentum_number(self) -> int:
        """Get l"""
        return self.phi_class.l

    def get_num_mu(self) -> int:
        return self.phi_class.get_num_mu()

    def find_overlapping_points(self, R_Gv: np.ndarray) -> np.ndarray:
        """"""
        assert R_Gv.ndim == 4
        diff = R_Gv - self.position
        distance = (np.sum(diff**2, axis=3))**0.5
        mask = (distance <= self.get_cutoff())
        return R_Gv[mask]


class LFCAtom:
    """"""
    rank: int
    """MPI rank of the grid domain that holds the center of this atom."""


    def __init__(self, index: int, first_mu: int, phi_j: list[BasisFunction],
                 relpos_c: np.ndarray, grid: UGDesc):
        """"""
        self.index = index
        self.first_mu = first_mu
        self.phi_j = phi_j
        self.phi_instances: list[BasisFunctionInstance] = []
        """Includes periodic copies"""
        self.grid = grid

        # Rank and position are properly set in set_position(),
        self.rank = -1
        self.relpos_c = np.asarray([np.nan, np.nan, np.nan])
        position_properly_set = self.set_position(relpos_c)
        assert position_properly_set

    def set_position(self, new_relpos_c: np.ndarray) -> bool:
        """Returns true if the atom migrated to a different rank."""

        if np.all(self.relpos_c == new_relpos_c):
            # No movement at all, nothing to do here
            return False

        self.relpos_c = new_relpos_c
        new_rank = self.grid._gd.get_rank_from_position(new_relpos_c)
        rank_changed = (self.rank != new_rank)
        self.rank = new_rank


        """Handle periodic boundaries: basis funcs from other unit cells
        can extend to the "main" cell. Treat this as a problem of N
        spheres (phi) in a box, and we want to know which spheres overlap
        with a smaller box in the middle ("main" cell). We first have
        to construct the "periodic images" of spheres, then perform
        sphere-box overlap checks.
        """
        position_v = new_relpos_c @ self.grid.cell_cv
        L = np.diag(self.grid.cell_cv)  # shape = (3,)

        # Main cell "min" and "max" points. In principle just (0, 0, 0) and
        # (L, L, L), but in practice the last grid point can be one spacing
        # earlier, and that is the last point we need for overlap checks.
        # FIXME: (L, L, L) is safer because it also sees possible phi that
        # have radius smaller than the grid spacing. So go with L for now.
        # TODO: 1. Warn about small-radius phi
        #       2. Drop such phi from computations. More optimal and allows
        #          assertions checks like "did we only include phi that
        #          contribute"

        main_cell_start_v = np.asarray([0, 0, 0])
        main_cell_end_v = main_cell_start_v + L

        # Enforce no overlap in directions without periodic b.c.
        pbc_mask = ~np.array(self.grid.pbc_c)

        self.phi_instances = []
        mu_work = self.first_mu
        for phi in self.phi_j:

            # NOTE: there is a helper function in GridDescriptor called
            # get_boxes(), but it seems to give false positives. The following
            # does better.
            r = phi.cutoff
            center = position_v

            # How many cells does the sphere overlap in given direction
            n_min = np.floor((center - r) / L).astype(int)
            n_max = np.floor((center + r) / L).astype(int)

            # TODO test if this actually works with non-pbc
            n_min = np.where(pbc_mask, 0, n_min)
            n_max = np.where(pbc_mask, 0, n_max)

            """Build a 3D shift grid. This Numpy magic is equivalent to:
            for nx, ny, nz in itertools.product(
                range(n_min[0], n_max[0] + 1),
                range(n_min[1], n_max[1] + 1),
                range(n_min[2], n_max[2] + 1),
            ):
                shift = np.array([nx, ny, nz])
                ...
            """
            ranges = [np.arange(n_min[d], n_max[d] + 1) for d in range(3)]
            grids = np.meshgrid(*ranges, indexing="ij")
            shifts = np.stack(grids, axis=-1).reshape(-1, 3)
            # includes (0, 0, 0), ie. the main cell
            assert shifts.size > 0

            # Where the periodic copy sphere would be
            translated_centers = center - shifts * L

            # Which shifted sphere actually overlap with the (0,0,0) cell.

            mask = np.asarray(
                [sphere_overlaps_box(r, sphere_pos,
                                     main_cell_start_v,
                                     main_cell_end_v)
                 for sphere_pos in translated_centers]
            )
            assert np.any(mask)

            for atom_pos in translated_centers[mask]:
                self.phi_instances.append(
                    BasisFunctionInstance(phi, self.index, mu_work, atom_pos)
                )

            mu_work += phi.get_num_mu()

        # TODO: the above can still give phi instances that technically
        # overlap the cell, but don't overlap with any grid point...?

        # Precalculate. TODO do in separate function
        # TEMP TEMP keep this for testing because it at least worked

        # get all grid points in this domain in real units.
        # 4D array of shape (Nx, Ny, Nz, 3)
        x_Gv = self.grid.xyz()
        # For now I compute these everywhere on the grid, not just in overlap regions
        grid_shape = (x_Gv.shape[0], x_Gv.shape[1], x_Gv.shape[2])

        for phi in self.phi_instances:
            shape = (phi.get_num_mu(), *grid_shape)
            phi_mG = np.zeros(shape)

            d_Gv = x_Gv - phi.position
            d_G = (d_Gv**2).sum(axis=3)**0.5
            # Points that are too far from the atom are left at value 0
            mask_G = (d_G < phi.get_cutoff())
            f_R_masked = phi.phi_class.spline(d_G[mask_G])
            d_Rv_masked = d_Gv[mask_G]

            l = phi.get_angular_momentum_number()
            for m in range(0, phi.get_num_mu()):
                phi_mG[m, mask_G] = (
                    f_R_masked * Y(l**2 + m, *d_Rv_masked.T)
                )

            phi.phi_mG = phi_mG

        return rank_changed


def build_lfc_system(setups: Setups,
                    grid: UGDesc,
                    relpos_ac: np.ndarray) -> LFCSystem:
    """Old version of BasisFunctions takes just splines_aj: list[list[Spline]]
    as input, which is a list spanning all atoms. In practice, it's
    constructed as [setup.basis_functions_J for setup in setups].
    With just this it's hard to know which splines are shared between atoms.

    We also cannot directly use the existing splines as they do not support
    GPU or single-precision evaluation.

    This function reorganizes GPAW setups to a simpler LFC-specific format,
      de-construct the provided splines
    """

    unique_phi_lists: list[list[Spline]] = []
    atom_types: list[LFCAtomDefinition] = []
    atoms = []

    num_atoms = len(setups)

    relpos_ac = np.asanyarray(relpos_ac)
    assert len(relpos_ac) == num_atoms

    # atom positions in real units
    pos_av = relpos_ac @ grid.cell_cv

    # Iterate over all atoms, not just unique species. This assumes that
    # atoms of the same type have the exact same list of basis functions
    for atom_idx, setup in enumerate(setups):
        setup = cast(Setup, setup)

        my_basis_funcs: list[Spline] = setup.basis_functions_J

        if my_basis_funcs not in unique_phi_lists:
            # Found new atom type
            unique_phi_lists.append(my_basis_funcs)

            phi_descs = []
            for phi in my_basis_funcs:
                phi = cast(Spline, phi)
                phi_descs.append(BasisFunctionDesc.from_legacy_spline(phi))

            atom_types.append(LFCAtomDefinition(len(atom_types), phi_descs))

        # Figure out type of this atom
        atom_type_idx = unique_phi_lists.index(my_basis_funcs)
        atom_type = atom_types[atom_type_idx]

        atoms.append(LFCAtomInstance(atom_type, pos_av[atom_idx]))

    assert len(atoms) == num_atoms
    assert len(unique_phi_lists) == len(atom_types)
    # check that we didn't accidentally include duplicate types.
    # Crazy oneliner for converting list[list[...]] to unique set and back
    assert len(unique_phi_lists) \
        == len(set(tuple(inner) for inner in unique_phi_lists))

    assert len(atom_types) == len(set(atom_types))

    return LFCSystem(grid, atom_types, atoms)


class PrecalculationMode(Enum):
    eIfPossible = auto()
    eAlways = auto()
    eNever = auto()


@dataclass
class LFCMatrixDistributionRules:
    """Used to describe how calculate_potential_matrix should be parallelized.
    We will parallelize by rows, ie. one rank does rows [mu_start, mu_end).
    """
    mu_start: int
    mu_end: int


# Should keep the API as close to the old BasisFunctions as possible,
# so use raw np or cp arrays instead of UGArray/Matrix in the following

class BasisFunctionCollectionBase(ABC):
    """"""

    # Prefer struct-of-arrays type data storage here. Spline storage is left
    # for subclasses to decide. We always index atoms as 0, 1, ... N.

    """The following containers are "global" and "constant": they describe the
    entire system independently of possible domain decomposition, and do not
    change after being initialized in the constructor. IE: We do not change
    atom or basis function indexing at any point.
    """
    atoms_a: list[LFCAtom]
    """List of all atoms in the system, including those which reside in other
    MPI domains."""
    _atom_type_a: list[AtomTypeId]
    """Type id of atom 'a'. This can be used to look up basis functions for
    a given atom instance."""
    _basis_functions_t: list[list[BasisFunction]]
    """BasisFunctions for a given atom type"""
    _mu_range_a: list[range]
    """Range of mu indices for atom 'a'."""
    _atom_for_mu: dict[int, int]
    """Gives index of the atom that contributes to given mu."""

    # This is local to each MPI rank in the grid communicator
    my_atom_indices: list[int]
    """Indices of all atoms whose center resides in this MPI rank (domain
    decomposition)."""

    _matrix_distribution_rules: LFCMatrixDistributionRules
    """FIXME: why does this have to be stored?! Old code passes the M ranges
    to C functions anyway. Outside users do seems to access Mstart, Mend,
    why?!?"""

    block_size: int

    blocks: list[GridBlock]
    # Store just those blocks that actually have phi overlap

    #_phi_aj: list[list[BasisFunctionInstance]]
    #_virtual_phi_aj: dict[int, list[BasisFunctionInstance]]

    def __init__(self,
                 system: LFCSystem,
                 use_gpu: bool = False,
                 mode: PrecalculationMode = PrecalculationMode.eIfPossible,
                 block_size: int | None = 8):
        """
        Initialize a BasisFunctionCollection.

        Parameters
        ----------
        system : LFCSystem
            Defines the grid and atom content on the grid. Should contain all
            atoms in the main unit cell even if the grid is set to use domain
            decomposition.
        use_gpu : bool, optional
            If True, basis functions will be computed and stored
            **exclusively** on GPU.
            Default is False.
        mode : PrecalculationMode, optional
            Control whether basis functions should be precalculated on the
            grid. Precalculation occurs whenever atom positions are changed.
            Warning: For realistic systems this may require extreme
            amounts of memory!
            Default is eIfPossible (precalculate if memory allows).
        block_size: int, optional
            Size of each grid block. None means no blocking.
        """
        self.grid = system.grid

        if block_size is not None:
            self.block_size = min(block_size, np.min(self.grid.mysize_c))
        else:
            self.block_size = np.max(self.grid.mysize_c)

        num_atoms = len(system.atoms)
        if num_atoms == 0:
            raise RuntimeError("No atoms?!")

        if use_gpu and cupy_is_fake:
            raise RuntimeError(
                "Can't use GPU BasisFunctionCollection with fake CuPy")

        self.xp = cp if use_gpu else np

        self.dtype = np.float64

        self._init_basis_functions(system.atom_types)

        # Build lookup lists:
        #   atom -> type ID; atom -> mu range; mu -> atom idx.
        # Also build the atom position array.
        mu = 0
        self._atom_type_a = []
        self._mu_range_a = []
        self._atom_for_mu = {}
        position_av = np.empty((num_atoms, 3), np.float64)

        for atom_idx, atom in enumerate(system.atoms):
            if len(atom.type.basis_functions) == 0:
                raise RuntimeError(
                    f"Missing basis functions for atom {atom_idx}")

            self._atom_type_a.append(atom.type.id)

            mu_a_start = mu
            mu_a_end = mu_a_start
            for desc in atom.type.basis_functions:
                mu_a_end += 2*desc.l + 1

            self._mu_range_a.append(range(mu_a_start, mu_a_end))
            mu = mu_a_end

            for m in range(mu_a_start, mu_a_end):
                assert m not in self._atom_for_mu
                self._atom_for_mu[m] = atom_idx

            position_av[atom_idx] = atom.position

        relpos_ac = position_av @ np.linalg.inv(self.grid.cell_cv)
        self._init_atoms(relpos_ac)
        self._update_grid_blocks()

        # Default is to always compute do the full V_munu matrix
        self.set_matrix_distribution(0, self.Mmax)

        # TODO: actually check if we can/should precalculate
        if mode != PrecalculationMode.eNever:
            # if self.get_cache_size_estimate() > too_much...
            self.should_precalculate = True
        else:
            self.should_precalculate = False

        if self.should_precalculate:
            self.precalculate_on_grid()

    def _init_basis_functions(self,
                              atom_types: list[LFCAtomDefinition]) -> None:
        """Initializes all basis functions for the system.
        """
        self._basis_functions_t = []
        for atom in atom_types:
            phi_j = []
            for desc in atom.basis_functions:
                phi_j.append(
                    BasisFunction(desc,
                    self.dtype,
                    use_gpu=False)
                )
            self._basis_functions_t.append(phi_j)
        #

    def _init_atoms(self, relpos_ac: np.ndarray):
        """"""
        # TODO init mu ranges here
        self.atoms_a = []
        self.my_atom_indices = []
        for a in range(self.num_atoms):
            mu = self.get_mu_range_a(a).start
            phi_j = self._basis_functions_t[self._atom_type_a[a]]
            self.atoms_a.append(
                LFCAtom(a, mu, phi_j, relpos_ac[a], self.grid)
            )

            if self.atoms_a[a].rank == self.grid.comm.rank:
                self.my_atom_indices.append(a)

    @trace
    def _update_grid_blocks(self) -> None:
        """Actually just fully rebuilds block info."""

        self.blocks = []
        grid_spacing_v = np.diag(self.grid._gd.h_cv)

        """For each phi, find which blocks they overlap. We start with an AABB
        (Axis Aligned Bounding Box) overlap test to find nearby boxes with
        possible overlap, then remove false positives by a proper spherical
        overlap test.
        """

        # Round division up. The last block is allowed to be smaller
        num_blocks_c = -(self.grid.mysize_c // -self.block_size).astype(int)
        real_block_size_v = self.block_size * grid_spacing_v

        self._block_map: dict[BlockCoords, GridBlock] = {}

        for phi in self.get_phi_instances():
            # Compute AABB extents for the phi-sphere (real space units)
            radius = phi.get_cutoff()
            aabb_min = phi.position - radius
            aabb_max = phi.position + radius

            # Convert to be relative to origin of this grid domain
            rel_min_v = aabb_min - self.grid.start_c * grid_spacing_v
            rel_max_v = aabb_max - self.grid.start_c * grid_spacing_v

            # Compute block indices that the AABB overlaps
            min_block = np.floor(rel_min_v / real_block_size_v)
            max_block = np.floor(rel_max_v / real_block_size_v)

            min_block = np.maximum(min_block, 0).astype(int)
            max_block = np.minimum(max_block, num_blocks_c - 1).astype(int)

            # AABB must overlap with some block (see below)
            assert np.all(max_block >= min_block)

            for bx, by, bz in itertools.product(
                range(min_block[0], max_block[0] + 1),
                range(min_block[1], max_block[1] + 1),
                range(min_block[2], max_block[2] + 1),
            ):
                # Exact overlap test: get block corners
                block_start_c = (self.grid.start_c
                                 + self.block_size * np.asarray([bx, by, bz]))

                # Clip end corner so that it stays within the MPI domain
                block_end_c = block_start_c + self.block_size
                block_end_c = np.minimum(block_end_c, self.grid.mysize_c)

                # FIXME can happen that a sphere DOES overlap the box,
                # but NOT any grid points in the box => not needed.
                # Happens already in LFCAtom.set_positions...
                if sphere_overlaps_box(
                    radius,
                    phi.position,
                    block_start_c * grid_spacing_v,
                    block_end_c * grid_spacing_v
                ):

                    block_coords = (bx, by, bz)
                    if block_coords in self._block_map:
                        block = self._block_map[block_coords]
                    else:
                        # first time in this block
                        block = GridBlock(
                            self.grid,
                            block_coords,
                            block_start_c,
                            block_end_c)
                        self._block_map[block_coords] = block

                    block.phi_j.append(phi)
                    phi.blocks_with_overlap.append(block)
                # end block loop

            # Sort the block list for convenience
            # phi.blocks_with_overlap = sorted(phi.blocks_with_overlap,
            #                                  key=lambda x : x.coords)

        # end phi loop

        # Construct mapping for block-local phi_m => global phi_mu
        for block in self._block_map.values():
            block.M_m = []
            block.phi_j = sorted(block.phi_j, key=lambda x : x.first_mu)
            for phi in block.phi_j:
                mu = phi.first_mu
                for m in range(phi.get_num_mu()):
                    block.M_m.append(mu + m)
            #
        #

    def get_relevant_blocks(self) -> list[GridBlock]:
        """Returns list of grid blocks that contribute, ie. have overlap with
        some phi."""
        return list(self._block_map.values())

    @trace
    @abstractmethod
    def precalculate_on_grid(self) -> None:
        raise NotImplementedError

    @trace
    @abstractmethod
    def add_to_density(
            self,
            nt_sG: np.ndarray | cp.ndarray,
            f_asi: dict[int, np.ndarray] | dict[int, cp.ndarray]) -> None:
        r"""Add linear combination of squared localized functions to density.
        Note that different atoms can have different number of basis funcs, so
        the range of `i` can vary. Therefore, f_asi is a dict (or a list) of
        arrays, not an array itself.
        :::

          ~        ---   a    a   2
          n (r) += >    f   [Φ(r)]
           s       ---   si   i
                   a,i
        """
        raise NotImplementedError

    @trace
    @abstractmethod
    def calculate_potential_matrix(
            self,
            vt_G: np.ndarray | cp.ndarray,
            out: np.ndarray | cp.ndarray | None = None) \
            -> np.ndarray | cp.ndarray:
        """Calculate lower part of the potential matrix.

        ::

                      /
            ~         |     *  _  ~ _        _   _
            V      =  |  Phi  (r) v(r) Phi  (r) dr    for  mu >= nu
             mu nu    |     mu            nu
                      /

        Parameters
        ----------
        vt_G : np.ndarray | cp.ndarray
            The potential array in real space (GPU or CPU).
        out : np.ndarray | cp.ndarray | None, optional
            Optional output array to store the result. If None, a new array is
            created. Default is None. TODO shape requirements?

        Returns
        -------
        np.ndarray | cp.ndarray
            The potential matrix with shape (Mstop - Mstart, MMax), where:
            - Rows correspond to basis functions [Mstart, Mstop)
            (if set_matrix_distribution was called, otherwise all rows).
            - Only the lower triangle and diagonal are guaranteed to be
            correct. Upper triangle elements may contain undefined or stale
            values.

            If 'out' was given, return value will be that same array.

        Notes
        -----
        - Only the lower triangle and diagonal elements are mathematically meaningful
        - If set_matrix_distribution() has been called, only rows [Mstart, Mstop) are computed
        - The returned array type (CPU or GPU) matches the input vt_G type

        Specifically, only the lower triangle and the diagonal are guaranteed
        to be sensible. Upper triangle may still be modified (old CPU code in
        fact does do this).

        If set_matrix_distribution() has been called, will only do rows
        [Mstart, Mstop).

        The returned array will have shape (Mstop - Mstart, MMax)
        """

        raise NotImplementedError

    @abstractmethod
    def has_precalculated_phi(self) -> bool:
        """"""
        raise NotImplementedError

    @property
    def num_atoms(self) -> int:
        """"""
        return len(self._atom_type_a)

    @cached_property
    def mu_range(self) -> range:
        """Range of mu indices"""
        # atom indices for lookups are always in range 0, ... N
        mu_start = self._mu_range_a[0].start
        mu_end = self._mu_range_a[-1].stop
        return range(mu_start, mu_end)

    @property
    def Mmax(self) -> int:
        """Last global index mu."""
        return self.mu_range.stop

    def get_atoms_for_mu(self, mu_start: int, mu_end: int) -> list[LFCAtom]:
        """Returns list of atoms that contribute to indices in range
        [mu_start, mu_end), exclusive."""
        assert mu_start <= mu_end
        mu_start = max(mu_start, self.mu_range.start)
        mu_end = min(mu_end, self.mu_range.stop)

        res = []
        # auxiliary set for faster "if not in list ..."
        seen = set()

        for mu in range(mu_start, mu_end):
            a = self._atom_for_mu[mu]
            if a not in seen:
                res.append(self.atoms_a[a])
                seen.add(a)

        return res

    def get_phi_for_mu(self, mu_start: int, mu_end: int) \
        -> list[BasisFunctionInstance]:
        """Returns list of basis functions that contribute to indices in range
        [mu_start, mu_end), exclusive"""

        atoms = self.get_atoms_for_mu(mu_start, mu_end)
        res = []
        for atom in atoms:
            # only include phi that really overlap with the wanted range
            for phi in atom.phi_instances:
                if (phi.first_mu < mu_end and
                    phi.first_mu + phi.get_num_mu() > mu_start
                    ):
                    res.append(phi)
        return res

    def get_mu_range_a(self, atom_index: int) -> range:
        """Get range of mu indices for specified atom."""
        return self._mu_range_a[atom_index]

    def get_num_phi_a(self, atom_index: int) -> int:
        """Get number of basis functions for specified atom"""
        mu_range = self._mu_range_a[atom_index]
        return mu_range.stop - mu_range.start

    def is_gpu(self) -> bool:
        """True if this object is using GPU."""
        return self.xp is not np

    def num_basis_functions(self) -> int:
        """"""
        return self.mu_range.stop - self.mu_range.start

    def get_phi_instances(self) -> list[BasisFunctionInstance]:
        """Returns all basis function instances that contribute in this unit
        cell, including periodic copies extending from other cells"""
        out = []
        for atom in self.atoms_a:
            out += atom.phi_instances
        return out

    # ??? These need to be exposed directly, outside code accesses them...
    @property
    def Mstart(self) -> int:
        """"""
        return self._matrix_distribution_rules.mu_start

    @property
    def Mstop(self) -> int:
        """"""
        return self._matrix_distribution_rules.mu_end
    # end ???

    @trace
    def set_positions(self, relpos_ac: np.ndarray) -> bool:
        """Updates atom positions. Return value is True if any atom migrated
        to another MPI rank in grid domain decomposition."""
        relpos_ac = np.asanyarray(relpos_ac)
        if len(relpos_ac) != self.num_atoms:
            raise ValueError("Wrong atom position array shape")

        has_migrations = False
        for a, atom in enumerate(self.atoms_a):
            try:
                has_migrations |= atom.set_position(relpos_ac[a])
            except GridBoundsError as e:
                e.args = (f'Atom {a} too close to edge: {e}',)
                raise

        self._update_grid_blocks()
        if self.should_precalculate:
            self.precalculate_on_grid()
        return has_migrations

    def set_matrix_distribution(self, Mstart: int, Mstop: int) -> None:
        """Specifies that calculate_potential_matrices should only do rows
        [Mstart, Mstop); Mstop is exclusive. Used for parallelization.
        See dataclass LFCMatrixDistributionRules."""
        # FIXME: why do we need stateful Mstart, Mstop?
        # Why not just pass them as inputs to functions that use them?

        Mstart = max(Mstart, self.mu_range.start)
        Mstop = min(Mstop, self.mu_range.stop)
        self._matrix_distribution_rules \
            = LFCMatrixDistributionRules(Mstart, Mstop)

    def calculate_potential_matrices(
            self,
            vt_G: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """Alias for calculate_potential_matrix"""
        return self.calculate_potential_matrix(vt_G)

    def get_cache_size_estimate(self) -> int:
        """Estimate of how many bytes are needed to precalculate and cache the
        basis functions on the grid.
        FIXME: not correct atm.
        """
        # FIXME does not include periodic copies
        num_sites = int(np.prod(self.grid.size))
        # TODO single precision. Do we need to allow complex phi?
        return num_sites * self.num_basis_functions() * 8


    def to_json(self, filename: str):
        """Debugging routine. Stores info about the grid and basis functions
        (spheres) to a JSON file which can be plotted in an external program.
        """
        import json
        data = {}

        def make_phi_data(phi: BasisFunctionInstance) -> dict:
            out = {}
            out["center"] = tuple(phi.position)
            out["radius"] = phi.get_cutoff()
            # just some ID for grouping phi of same spline type
            out["type_id"] = id(phi.phi_class)
            return out

        def make_block_data(block: GridBlock) -> dict:
            out = {}
            points = block.get_block_xyz()
            # flatten to a nested list
            points.reshape(-1, 3).tolist()
            out["points"] = points.reshape(-1, 3).tolist()
            return out

        data["spheres"] = [make_phi_data(phi)
                           for phi in self.get_phi_instances()]

        data["blocks"] = [make_block_data(block)
                          for block in self._block_map.values()]

        with open(filename, "w") as f:
            json.dump(data, f)
        print("Exported BasisFunctions to " + filename)


class BasisFunctionCollection(BasisFunctionCollectionBase):
    """Version that uses GPAW C++ extension."""

    @override
    def __init__(self,
                 system: LFCSystem,
                 use_gpu: bool = False,
                 mode: PrecalculationMode = PrecalculationMode.eIfPossible):
        """"""
        if GPAW_NO_C_EXTENSION:
            raise RuntimeError("Cannot construct BasisFunctionCollection" \
            "without GPAW C++ extension." \
            "Use BasisFunctionCollection_PurePython instead.")

        if not use_gpu:
            raise RuntimeError("Only GPU version implemented for now")
        super().__init__(system, use_gpu, mode)
        # TODO: construct C++ side object that holds splines

    @override
    def precalculate_on_grid(self) -> None:
        """"""
        raise NotImplementedError("WIP")

    @override
    def add_to_density(
        self,
        nt_sG: np.ndarray | cp.ndarray,
        f_asi: dict[int, np.ndarray] | dict[int, cp.ndarray]) -> None:
        """"""

        # Copy the f_asi dict to a single contiguous array
        num_spins = nt_sG.shape[0]
        f_sM = self.xp.empty((num_spins, self.num_basis_functions()))

        for atom_idx in range(self.num_atoms):
            atom_mu_range = self._mu_range_a[atom_idx]
            f_sM[:, atom_mu_range.start:atom_mu_range.stop] = (
                f_asi[atom_idx]
            )

        # TODO call cpp function


    @override
    def calculate_potential_matrices(
        self,
        vt_G: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """"""
        raise NotImplementedError("WIP")

    @override
    def has_precalculated_phi(self) -> bool:
        """"""
        return False
