import numpy as np
from gpaw.gpu import cupy as cp, as_xp, cupy_is_fake
from functools import cached_property
from dataclasses import dataclass
from gpaw.spline import Spline
from gpaw.core import UGDesc
from gpaw.sphere.spherical_harmonics import Y
from gpaw.new.timer import trace
from gpaw import GPAW_NO_C_EXTENSION
from typing import Protocol, runtime_checkable, override
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

# BasisFunction manages splines: chooses C++ vs Python, mirrors to GPU, etc


class PrecalculationMode(Enum):
    eIfPossible = auto()
    eAlways = auto()
    eNever = auto()


@runtime_checkable
class SplineLike(Protocol):
    """"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...


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


class BasisFunction:
    """Defines a basis function: phi_lm(x) = f(r)*Y_lm.
    """

    spline: SplineLike

    def __init__(self,
                 desc: BasisFunctionDesc,
                 purepython: bool = False):
        """
        Initialize a basis function phi_lm(x) = f(r) * Y_lm.

        Parameters
        ----------
        l : int
            Angular momentum number.
        cutoff : float
            Radial cutoff beyond which the basis function is zero.
        f_r : np.ndarray or cp.ndarray
            Discrete samples of the radial function f(r) on [0, cutoff].
            These values are used to construct a continuous spline; the
            final value at r = cutoff must be exactly zero.
        purepython : bool
            If True, will use splines from Scipy/Cupy instead using GPAW C++
            extension. This option is forcibly turned on if
            GPAW_NO_C_EXTENSION is True.
        """
        assert desc.l >= 0
        assert desc.cutoff > 0
        assert len(desc.f_r) > 2, "Need at least 3 points to define a spline"
        self.l = desc.l
        self.cutoff = desc.cutoff
        self.purepython = True if GPAW_NO_C_EXTENSION else purepython

        # TODO handle single precision
        self._init_spline(desc.f_r)

    @staticmethod
    def make_random(l: int,
                    cutoff: float,
                    rng: np.random.Generator,
                    num_points: int = 100,
                    purepython: bool = False) -> "BasisFunction":
        """Generates a basis function with randomly generated radial part
        using the provided RNG.
        """
        values = rng.random(size=num_points)
        # last value must be 0
        values[-1] = 0.0
        return BasisFunction(l, cutoff, values, purepython)

    def _init_spline(self, f_r: np.ndarray) -> None:
        """Initializes spline on the CPU.
        """

        if self.purepython:
            from scipy.interpolate import CubicSpline
            self.spline = CubicSpline(
                np.linspace(0.0, self.cutoff, len(f_r), endpoint=True),
                as_xp(f_r, xp=np),
                bc_type='clamped',  # first derivatives are zero at boundaries
                extrapolate=False
            )
        else:
            from gpaw.cgpaw.gpu import Spline32, Spline64
            self.spline = Spline64(self.cutoff, f_r)

        # # build a cubic spline. Cupyx does not have a direct CubicSpline
        # # replacement, so we build it on CPU first, then get it on GPU as a
        # # PPoly object (generic piecewise polynomial).
        # # TODO handle fake cupy. Would need a PPoly that returns cp.ndarray
        # if self.xp is cp:
        #     from cupyx.scipy.interpolate import PPoly
        #     self.spline = PPoly(cp.asarray(spline_cpu.c),
        #                         cp.asarray(spline_cpu.x),
        #                         extrapolate=False)

    def num_m(self) -> int:
        """How many values of m does this describe. IE: just 2*l + 1."""
        return 2 * self.l + 1

    @staticmethod
    def from_old_spline(spline: Spline, purepython = False) -> "BasisFunction":
        """"""
        num_x: int = spline.get_npoints()
        x = np.linspace(0.0, spline.get_cutoff(), num_x)

        return BasisFunction(spline.get_angular_momentum_number(),
                             spline.get_cutoff(),
                             spline.map(x),
                             purepython)


@dataclass
class AtomData:
    """Does not store basis functions themselves, just useful metadata for
    atoms."""

    mu_range: range
    """Values of the combined index mu spanned by basis functions of this
    atom."""
    index: int
    """Index of the atom (integer between 0 and num_atoms)"""
    position_v: np.ndarray | None = None
    """Position of the atom in real units."""

    @cached_property
    def num_basis_functions(self) -> int:
        """Number of basis functions for this atom, ie. how many {l,m}
        index combinations.
        """
        return self.mu_range.stop - self.mu_range.start

    def set_position(self, new_pos_v: np.ndarray) -> None:
        """"""
        self.position_v = new_pos_v


class BasisFunctionCollectionBase(ABC):
    """Draft implementation of basis functions. This one uses the C++ backend.
    Keep in mind while designing:
        - Should eventually work also without C-extension => try to keep data
        structures on Python side.
    """

    def __init__(self, grid: UGDesc,
                 phi_aj: list[list[BasisFunctionDesc]],
                 use_gpu: bool = False,
                 mode: PrecalculationMode = PrecalculationMode.eIfPossible):
        """
        Initialize a BasisFunctionCollection.

        Parameters
        ----------
        grid : UGDesc
            Grid descriptor.
        phi_aj : list[list[BasisFunctionDesc]]
            List of basis function lists, one list per atom. Each inner list
            contains BasisFunctionDesc objects for that atom.
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
        self.grid = grid

        assert len(phi_aj) > 0, "No atoms?!"

        self.atoms: list[AtomData] = []

        # Assign mu indices to basis functions
        mu = 0
        for a, desc_j in enumerate(phi_aj):
            if len(desc_j) == 0:
                raise RuntimeError(f"Missing basis functions for atom {a}")

            mu_a_start = mu
            mu_a_end = mu_a_start
            for desc in desc_j:
                mu_a_end += 2*desc.l + 1

            self.atoms.append(AtomData(range(mu_a_start, mu_a_end), a))
            mu = mu_a_end

        self._phi_MG_cache: np.ndarray | cp.ndarray | None
        """Cached values of basis funcs, phi_mu(x).
        This is 4D: first index is mu (M), then x, y, z indices.
        """
        self.atoms_placed = False  # Set to True once positions have been set

        if use_gpu and cupy_is_fake:
            raise RuntimeError(
                "Can't use GPU BasisFunctionCollection with fake CuPy")

        self.xp = cp if use_gpu else np

        # TODO: check if we can/should precalculate
        if mode != PrecalculationMode.eNever:
            # if self.get_cache_size_estimate() > too_much...
            self.should_precalculate = True
        else:
            self.should_precalculate = False

        self.create_basis_functions(phi_aj)

    # Should keep the API as close to the old BasisFunctions as possible,
    # so use raw np or cp arrays instead of UGArray/Matrix in the following

    @trace
    @abstractmethod
    def create_basis_functions(
        self,
        phi_desc_aj: list[list[BasisFunctionDesc]]) -> None:
        """Initializes all basis functions for the system. Subclasses can
        choose how this should be done, and how the basis functions should
        be stored."""
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

    @property
    def num_atoms(self) -> int:
        """"""
        return len(self.atoms)

    @cached_property
    def mu_range(self) -> range:
        """Range of mu indices"""
        mu_start = 0
        mu_end = mu_start
        for atom in self.atoms:
            mu_end += atom.num_basis_functions
        return range(mu_start, mu_end)

    @cached_property
    def mu_range_a(self) -> list[range]:
        """Range of mu indices for a given atom."""
        mu = 0
        res = []
        for atom in self.atoms:
            res.append(range(mu, mu + atom.num_basis_functions))
            mu += atom.num_basis_functions
        return res

    @property
    def Mmax(self) -> int:
        """Last global index mu."""
        return self.mu_range.stop

    @cached_property
    def my_atom_indices(self) -> range:
        """TODO handle distributed grid"""
        return range(self.num_atoms)

    def is_gpu(self) -> bool:
        """True if this object is using GPU."""
        return self.xp is not np

    def have_atoms_been_placed(self) -> bool:
        """"""
        return self.atoms_placed

    def num_basis_functions(self) -> int:
        """"""
        return self.mu_range.stop - self.mu_range.start

    @trace
    def set_positions(self, relpos_ac: np.ndarray) -> None:
        """Sets atom positions"""
        relpos_ac = np.asanyarray(relpos_ac)
        assert relpos_ac.shape[0] == self.num_atoms

        # atom positions in real units
        pos_av = relpos_ac @ self.grid.cell_cv

        for a, atom in enumerate(self.atoms):
            atom.position_v = pos_av[a]
        self.atoms_placed = True

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


class BasisFunctionCollection_PurePython(BasisFunctionCollectionBase):
    """Pure Python implementation of BasisFunctionCollection.
    Not very optimized.
    """

    @override
    def create_basis_functions(
        self,
        phi_desc_aj: list[list[BasisFunctionDesc]]) -> None:
        """"""

        self.phi_aj: list[list[BasisFunction]] = []
        for desc_j in phi_desc_aj:
            self.phi_aj.append([BasisFunction(desc, purepython=True)
                                for desc in desc_j])


    @override
    def precalculate_on_grid(self) -> None:
        """Computes and caches phi_mu(x) = f_al(r) * Y_l^m everywhere on the
        grid.
        TODO: Boundary conditions! f(r) can contribute to neighboring cells
        """
        assert self.have_atoms_been_placed(), "Must call set_positions() first"

        num_mu = self.mu_range.stop - self.mu_range.start
        # get all grid points in real units. 4D array of shape (Nx, Ny, Nz, 3)
        x_Gv = self.grid.xyz()

        shape = (num_mu, x_Gv.shape[0], x_Gv.shape[1], x_Gv.shape[2])

        # running work index
        mu_work = self.mu_range.start

        # TODO precalculation on GPU. For now, do it on CPU and send to GPU
        phi_MG_cache = np.zeros(shape, dtype=np.float64)
        for a, atom in enumerate(self.atoms):
            d_Gv = x_Gv - atom.position_v
            d_G = (d_Gv**2).sum(axis=3)**0.5

            for phi in self.phi_aj[a]:
                # Points that are too far from the atom are left at value 0
                mask_G = (d_G < phi.cutoff)

                f_R_masked = phi.spline(d_G[mask_G])
                d_Rv_masked = d_Gv[mask_G]

                for m in range(0, phi.num_m()):
                    phi_MG_cache[mu_work + m, mask_G] = (
                        f_R_masked * Y(phi.l**2 + m, *d_Rv_masked.T)
                    )
                mu_work += phi.num_m()

        # TODO check not-NaN. Splines should be set to NOT extrapolating,
        # so if we attempted to compute splines outside their range we should
        # have gotten an error/NaN/None
        # TODO actually, I guess the intended behavior would be to return 0
        # for r > r_c. This is what the old Spline class does
        self.phi_MG_cache = as_xp(phi_MG_cache, xp=self.xp)

    @override
    def add_to_density(
        self,
        nt_sG: np.ndarray | cp.ndarray,
        f_asi: dict[int, np.ndarray] | dict[int, cp.ndarray]) -> None:
        """"""
        num_spins = nt_sG.shape[0]
        f_sM = self.xp.empty((num_spins, self.num_basis_functions()))

        if self.phi_MG_cache is not None:
            # Copy the f_asi dict to a single array
            for atom_index in range(len(self.phi_aj)):
                atom_mu_range = self.mu_range_a[atom_index]
                f_sM[:, atom_mu_range.start:atom_mu_range.stop] = (
                    f_asi[atom_index]
                )

            nt_sG += self.xp.einsum('sm, mxyz -> sxyz',
                                    f_sM, self.phi_MG_cache**2,  # type:ignore
                                    optimize=True)  # type:ignore
        else:
            raise NotImplementedError(
                "PurePython BasisFunctions without precalculation")

    @override
    def calculate_potential_matrix(
        self,
        vt_G: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """"""
        if self.phi_MG_cache is not None:
            # v(x) * phi(x)
            phit_vt_MG = vt_G * self.phi_MG_cache

            # "integrate": ie. contract the grid indices.
            res = self.xp.einsum("mxyz, nxyz -> mn",
                                 self.phi_MG_cache.conj(),  # type:ignore
                                 phit_vt_MG)                # type:ignore

            # FIXME: Drop half since we promised just the lower half.
            # Very redundant, obviously
            return self.xp.tril(res) * self.grid.dv  # type:ignore
        else:
            raise NotImplementedError(
                "PurePython BasisFunctions without precalculation")