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
                raise RuntimeError("Atom type IDs must be unique")
            ids.append(atom_type.id)


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
    # check that we didn't accidentally include duplicate types
    assert len(unique_phi_lists) == len(set(unique_phi_lists))
    assert len(atom_types) == len(set(atom_types))

    return LFCSystem(grid, atom_types, atoms)


class PrecalculationMode(Enum):
    eIfPossible = auto()
    eAlways = auto()
    eNever = auto()


class BasisFunctionCollectionBase(ABC):
    """"""

    # Prefer struct-of-arrays type data storage here. Spline storage is left
    # for subclasses to decide. We always index atoms as 0, 1, ... N.
    # With domain parallelization this will require index mapping (TODO)

    _atom_type_a: list[AtomTypeId]
    """Type id of atom 'a'. This can be used to look up basis functions for
    a given atom instance."""
    _position_av: np.ndarray
    """Real-space positions of atoms, so that position_av[a] is a 3D
    array."""
    _mu_range_a: list[range]
    """Range of mu indices for atom 'a'."""
    _phi_MG_cache: np.ndarray | cp.ndarray | None
    """Cached values of basis funcs, phi_mu(x).
    This is 4D: first index is mu (M), then x, y, z indices.
    """

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

        # Build lookup lists: atom -> type ID; atom -> mu range.
        # Also build the atom position array
        mu = 0
        self._atom_type_a = []
        self._mu_range_a = []
        self._position_av = np.empty((num_atoms, 3), np.float64)

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

            self._position_av[atom_idx] = atom.position

        # TODO: check if we can/should precalculate
        if mode != PrecalculationMode.eNever:
            # if self.get_cache_size_estimate() > too_much...
            self.should_precalculate = True
        else:
            self.should_precalculate = False

        self._init_basis_functions(system.atom_types)

        if self.should_precalculate:
            self.precalculate_on_grid()

    # Should keep the API as close to the old BasisFunctions as possible,
    # so use raw np or cp arrays instead of UGArray/Matrix in the following

    @trace
    @abstractmethod
    def _init_basis_functions(self,
                              atom_types: list[LFCAtomDefinition]) -> None:
        """Initializes all basis functions for the system. Subclasses can
        choose how this should be done, and how the basis functions should
        be stored.
        """
        raise NotImplementedError

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

    @trace
    def set_positions(self, relpos_ac: np.ndarray) -> None:
        """Updates atom positions"""
        relpos_ac = np.asanyarray(relpos_ac)
        if relpos_ac.shape != self._position_av.shape:
            raise ValueError("Wrong atom position array shape")

        # convert to real units
        self._position_av = relpos_ac @ self.grid.cell_cv

        # TODO: recalculate overlaps; check if atoms moved to another MPI rank

        if self.should_precalculate:
            self.precalculate_on_grid()

    def set_matrix_distribution(self, Mstart: int, Mstop: int) -> None:
        """TODO!"""
        self.Mstart = Mstart
        self.Mstop = Mstop

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

    def _update_boundaries(self) -> None:
        """"""
        if not np.any(self.grid.pbc_c):
            # No periodic boundaries, nothing to do
            return

        #for atom_idx in range(self.num_atoms):


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
    def _init_basis_functions(
        self,
        atom_types: list[list[LFCAtomDefinition]]) -> None:
        """"""
        raise NotImplementedError("WIP")

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
