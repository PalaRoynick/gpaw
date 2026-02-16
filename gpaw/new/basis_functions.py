import numpy as np
from gpaw.gpu import cupy as cp, as_xp, cupy_is_fake
from functools import cached_property
from dataclasses import dataclass
from gpaw.spline import Spline
from gpaw.core import UGDesc
from gpaw.new.timer import trace
from gpaw import GPAW_NO_C_EXTENSION
from gpaw.setup import Setups, Setup
from typing import override, cast
from enum import Enum, auto
from abc import ABC, abstractmethod

from gpaw.sphere.spherical_harmonics import Y

from typing import Protocol, runtime_checkable

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
Note that integrals of phi_mu over the full grid can be optimized because
the basis functions have a finite range => can avoid a lot of redundant
"addition of 0". Challenge is to do this in a smart way.
"""

"""NOTE TO SELF: I quite like the current BasisFunctionCollection setup where
the base class handles indexing but does not dictate how the basis functions
(splines) should be stored. However the BasisFunction class is very annoying,
seems hard to handle both CPU and GPU cases?
Perhaps we can reduce BasisFunction to a much simpler "dataclass" that is
defined within the PurePython version of BasisFunctionCollection.
For the C++ version we should create a pure-C++ class that holds the splines,
and the python implementation wraps around it. This makes sense because we
need to pass splines to kernels, which is easier to do fully from C++.
So we'd change the following so that BasisFunction is always purepython
"""

AtomTypeId = int
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

@dataclass
class BasisFunctionInstance:
    """Instance of a basis function with fixed position and index range."""
    phi_class: BasisFunction
    parent_atom: int
    first_mu: int
    position: np.ndarray
    phi_mG: np.ndarray | None = None
    """Precalculated values on the grid"""

    def get_cutoff(self) -> float:
        """Get the radial cutoff"""
        return self.phi_class.cutoff

    def get_angular_momentum_number(self) -> int:
        """Get l"""
        return self.phi_class.l

    def get_num_mu(self) -> int:
        return self.phi_class.get_num_mu()


class LFCAtom:
    """"""

    def __init__(self, index: int, first_mu: int, phi_j: list[BasisFunction],
                 relpos_c: np.ndarray, grid: UGDesc):
        """"""
        self.index = index
        self.first_mu = first_mu
        self.phi_j = phi_j
        self.phi_instances: list[BasisFunctionInstance] = []
        """Includes periodic copies"""

        self.phi_wmG: list[np.ndarray] | None = None
        """Cached phi_mu(x) on the grid"""

        self.set_position(relpos_c, grid)

    def set_position(self, new_relpos_c: np.ndarray, grid: UGDesc):
        """"""
        self.phi_instances = []

        self.mu_range_w: list[range] = []
        # yeah...

        mu_work = self.first_mu
        for phi in self.phi_j:

            radius = phi.cutoff

            """Use helper from GridDescriptor to find all units cells that this
            spline overlaps with, including the main cell.
            We get XYZ coords for box start and end corners, and relative
            displacement that brings this atom to correct position in the
            other box."""
            # TODO actually not 100% sure what the first two outputs are

            # dunno what this is
            cut: bool = True
            boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray]] \
                = grid._gd.get_boxes(new_relpos_c, radius, cut)

            # Loop includes the "real" atom:
            for beg_c, end_c, relative_displacement_c in boxes:
                # real-space position of the replicated atom
                pos_v = np.dot(new_relpos_c - relative_displacement_c,
                               grid.cell_cv)
                self.phi_instances.append(
                    BasisFunctionInstance(phi, self.index, mu_work, pos_v)
                )

            mu_work += phi.get_num_mu()

        # Precalculate. TODO do in separate function

        # get all grid points in real units. 4D array of shape (Nx, Ny, Nz, 3)
        x_Gv = grid.xyz()
        # For now I compute these everywhere on the grid, not just in overlap regions
        grid_shape = (x_Gv.shape[0], x_Gv.shape[1], x_Gv.shape[2])

        self.phi_wmG = []
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

            self.phi_wmG.append(phi_mG)

            mu = phi.first_mu
            self.mu_range_w.append(range(mu, mu + phi.get_num_mu()))

            # Give the phi instance a view to its contribution to the phi_wmG
            # array. Makes looping over all phi easier. In fact the whole
            # atom.phi_wmG is pretty dumb because its a list of arrays,
            # not a single contiguous array!
            phi.phi_mG = phi_mG


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

    def __post_init__(self):
        """Validate"""
        if self.mu_start >= self.mu_end:
            raise ValueError


# Should keep the API as close to the old BasisFunctions as possible,
# so use raw np or cp arrays instead of UGArray/Matrix in the following

class BasisFunctionCollectionBase(ABC):
    """"""

    # Prefer struct-of-arrays type data storage here. Spline storage is left
    # for subclasses to decide. We always index atoms as 0, 1, ... N.
    # With domain parallelization this will require index mapping (TODO)

    _atom_type_a: list[AtomTypeId]
    """Type id of atom 'a'. This can be used to look up basis functions for
    a given atom instance."""
    _basis_functions_t: list[list[BasisFunction]]
    """BasisFunctions for a given atom type"""
    _mu_range_a: list[range]
    """Range of mu indices for atom 'a'."""
    _phi_MG_cache: np.ndarray | cp.ndarray | None
    """Cached values of basis funcs, phi_mu(x).
    This is 4D: first index is mu (M), then x, y, z indices.
    """
    _matrix_distribution_rules: LFCMatrixDistributionRules
    """FIXME: why does this have to be stored?! Old code passes the M ranges
    to C functions anyway. Outside users do seems to access Mstart, Mend,
    why?!?"""

    atoms_a: list[LFCAtom]

    _phi_aj: list[list[BasisFunctionInstance]]
    _virtual_phi_aj: dict[int, list[BasisFunctionInstance]]

    def __init__(self,
                 system: LFCSystem,
                 use_gpu: bool = False,
                 mode: PrecalculationMode = PrecalculationMode.eIfPossible):
        """
        Initialize a BasisFunctionCollection.

        Parameters
        ----------
        system : LFCSystem
            Defines the grid and atom content on the grid.
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
        """
        self.grid = system.grid

        num_atoms = len(system.atoms)
        if num_atoms == 0:
            raise RuntimeError("No atoms?!")

        if use_gpu and cupy_is_fake:
            raise RuntimeError(
                "Can't use GPU BasisFunctionCollection with fake CuPy")

        self.xp = cp if use_gpu else np

        self.dtype = np.float64

        self._init_basis_functions(system.atom_types)

        # Build lookup lists: atom -> type ID; atom -> mu range.
        # Also build the atom position array
        mu = 0
        self._atom_type_a = []
        self._mu_range_a = []
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

            position_av[atom_idx] = atom.position

        # Default to trivial matrix distribution (ie. one rank does all mu)
        self.set_matrix_distribution(0, self.Mmax)

        relpos_ac = position_av @ np.linalg.inv(self.grid.cell_cv)
        self._place_atoms(relpos_ac)

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

    def _place_atoms(self, relpos_ac: np.ndarray):
        """"""
        # TODO init mu ranges here
        self.atoms_a = []
        for a in range(self.num_atoms):
            mu = self.get_mu_range_a(a).start
            phi_j = self._basis_functions_t[self._atom_type_a[a]]
            self.atoms_a.append(
                LFCAtom(a, mu, phi_j, relpos_ac[a], self.grid)
            )


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
            vt_G: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """Calculate lower part of the potential matrix.

        ::

                      /
            ~         |     *  _  ~ _        _   _
            V      =  |  Phi  (r) v(r) Phi  (r) dr    for  mu >= nu
             mu nu    |     mu            nu
                      /

        Returns Vt_MM.
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

    @cached_property
    def my_atom_indices(self) -> range:
        """TODO handle distributed grid"""
        return range(self.num_atoms)

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
    def set_positions(self, relpos_ac: np.ndarray) -> None:
        """Updates atom positions"""
        relpos_ac = np.asanyarray(relpos_ac)
        if len(relpos_ac) != self.num_atoms:
            raise ValueError("Wrong atom position array shape")

        # TODO: check if atoms moved to another MPI rank

        for a, atom in enumerate(self.atoms_a):
            atom.set_position(relpos_ac[a], self.grid)

        # if self.should_precalculate:
        #     self.precalculate_on_grid()

    def set_matrix_distribution(self, Mstart: int, Mstop: int) -> None:
        """"""
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
        """
        num_sites = int(np.prod(self.grid.size))
        # TODO single precision. Do we need to allow complex phi?
        return num_sites * self.num_basis_functions() * 8

    def _construct_periodic_images(self) -> None:
        """For periodic boundary conditions it can happen that basis functions
        from other unit cells overlap with grid points or basis
        functions in this unit cell. This function creates "virtual images" of
        basis functions that exist outside our unit cell to handle the
        overlaps.

        Must be called whenever atom positions change, which MUST happen only
        from set_positions."""
        self._virtual_phi_aj = {}

        if not np.any(self.grid.pbc_c):
            # No periodic boundaries, nothing to do
            return

        L = np.diag(self.grid.cell_cv)  # shape = (3,)

        """Goal is to place "virtual" atoms in neighbor cells, including
        only the basis funcs that have overlap with this "main" cell.
        We first compute how many periodic unit cells a basis func
        overlaps in each direction, giving a list of integer "shifts" like
        [-2, -1, 0, 1] if the phi extends 2 cells in negative dir and 1 in
        positive dir. Doing this for all 3 dimensions and taking cartesian
        product gives a 3D grid of all possible shifts to cells to which
        this phi may extend. We then check the overlap separately for each
        shifted cell, handling possible virtual atoms in diagonal
        directions.
        """

        for atom_idx, phi_j in enumerate(self._phi_aj):
            center = self._position_av[atom_idx]

            virtual_copies: list[BasisFunctionInstance] = []

            for phi in phi_j:

                r = phi.get_cutoff()

                # How many boxes does the sphere overlap in given direction
                n_min = np.floor((center - r) / L).astype(np.int64)
                n_max = np.floor((center + r) / L).astype(np.int64)

                # Enforce no overlap in directions without periodic b.c.
                pbc_mask = ~np.array(self.grid.pbc_c)
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

                # remove (0,0,0)
                shifts = shifts[np.any(shifts != 0, axis=1)]
                if shifts.size == 0:
                    continue

                # Where the virtual atoms would be
                translated_centers = center - shifts * L

                lower = np.minimum(translated_centers, 0.0)
                upper = np.maximum(translated_centers - L, 0.0)
                d2 = np.sum(lower**2 + upper**2, axis=1)

                # which virtual atoms actually overlap with the (0,0,0) cell
                mask = (d2 <= r * r)
                if not np.any(mask):
                    continue

                for shift_vec, virtual_pos in zip(shifts[mask],
                                                  translated_centers[mask]):
                    # Virtual phi gets the same mu indices as the real one
                    virtual_copies.append(
                        BasisFunctionInstance(
                            phi_class=phi.phi_class,
                            parent_atom=atom_idx,
                            first_mu=phi.first_mu,
                            position=virtual_pos.copy(),
                        )
                    )

            if virtual_copies:
                self._virtual_phi_aj[atom_idx] = virtual_copies

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
