import numpy as np
from gpaw.gpu import cupy as cp, as_xp
from functools import cached_property
from dataclasses import dataclass
from gpaw.spline import Spline
from gpaw.core import UGDesc
from gpaw.sphere.spherical_harmonics import Y
from gpaw.new.timer import trace

from typing import TypeAlias, Literal, Optional
from scipy.interpolate import CubicSpline


GridPoint: TypeAlias = np.ndarray[int, tuple[Literal[3]]]
"""Alias for 3-element Numpy array. Used for indexing XYZ grid points."""

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

Class BasisFunctionsNew describes the full set of all basis functions phi_mu
and provides methods for calculating stuff with them. Note that integrals of
phi_mu over the full grid are can be optimized because the basis
functions have a finite range => can avoid a lot of redundant "addition of 0".
Challenge is to do this in a smart way.

In the following, Splines are objects that describe the radial functions in an
interpolated form, while also knowing its associated angular momentum number l.
"""


@dataclass
class SplineData:
    breakpoints: cp.ndarray
    """Breakpoints {r_i}"""
    values: cp.ndarray
    """Spline values at the breakpoints, {S_i = S(r_i)}"""


class BasisFunction:
    """Defines a basis function: phi_lm(x) = f(r)*Y_lm.
    """

    def __init__(self, l: int, cutoff: float, f_r: np.ndarray | cp.ndarray):
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
        """
        assert l >= 0
        assert cutoff > 0
        assert len(f_r) >= 2
        self.l = l
        self.cutoff = cutoff

        self.xp = cp if isinstance(f_r, cp.ndarray) else np
        self._init_spline(f_r)

    @staticmethod
    def make_random(l: int,
                    cutoff: float,
                    rng: np.random.Generator | cp.random.Generator,
                    num_points: int = 100) -> "BasisFunction":
        """Generates a basis function with randomly generated radial part
        using the provided RNG (numpy or cupy). Result will be on GPU if the
        provided RNG is from Cupy.
        """
        values = rng.random(size=num_points)
        # last value must be 0
        values[-1] = 0.0
        return BasisFunction(l, cutoff, values)

    def _init_spline(self, f_r: np.ndarray | cp.ndarray) -> None:
        """"""

        # build a cubic spline. Cupyx does not have a direct CubicSpline
        # replacement, so we build it on CPU first, then get it on GPU as a
        # PPoly object (generic piecewise polynomial).

        spline_cpu = CubicSpline(
            np.linspace(0.0, self.cutoff, len(f_r), endpoint=True),
            as_xp(f_r, xp=np),
            bc_type='clamped',  # first derivatives are zero at boundaries
            extrapolate=False
        )

        # TODO handle fake cupy. Would need a PPoly that returns cp.ndarray
        if self.xp is cp:
            from cupyx.scipy.interpolate import PPoly
            self.spline = PPoly(cp.asarray(spline_cpu.c),
                                cp.asarray(spline_cpu.x),
                                extrapolate=False)
        else:
            # TODO typing. Note that CubicSpline is also a PPoly
            self.spline = spline_cpu
        #

    def num_m(self) -> int:
        """How many values of m does this describe. IE: just 2*l + 1."""
        return 2 * self.l + 1

    def get_radial_breakpoints(self) -> np.ndarray | cp.ndarray:
        """Returns breakpoints {r_i} defining the radial spline domain."""
        return self.spline.x

    def get_spline_data(self) -> SplineData:
        """"""
        x = self.get_radial_breakpoints()
        y = self.spline(x)
        return SplineData(breakpoints=x, values=y)

    @staticmethod
    def from_old_spline(spline: Spline) -> "BasisFunction":
        """"""

        # TODO double check if old spline always starts from 0
        num_x: int = spline.get_npoints()
        x = np.linspace(0.0, spline.get_cutoff(), num_x)

        return BasisFunction(spline.get_angular_momentum_number(),
                             spline.get_cutoff(),
                             spline.map(x))


class BasisFunctionsNew():
    """Draft implementation of basis functions. This one uses the C++ backend.
    Keep in mind while designing:
        - Should eventually work also without C-extension => try to keep data
        structures on Python side.
    """

    @dataclass
    class AtomData:
        basis_functions: list[BasisFunction]
        index: int  # index of the atom (integer between 0 and num_atoms)
        position_v: Optional[np.ndarray] = None
        """Position of the atom in real units."""

        @cached_property
        def num_basis_functions(self) -> int:
            """Number of basis functions for this atom, ie. how many {l,m}
            index combinations.
            """
            res = 0
            for phi in self.basis_functions:
                res += 2 * phi.l + 1
            return res

        def set_position(self, new_pos_v: np.ndarray) -> None:
            """"""
            self.position_v = new_pos_v

    def __init__(self, grid: UGDesc,
                 phi_aj: list[list[BasisFunction]]):
        """"""
        self.grid = grid
        self.atoms: list[BasisFunctionsNew.AtomData] = []

        assert len(phi_aj) > 0, "No atoms?!"

        for a, phi_M in enumerate(phi_aj):
            assert len(phi_M) > 0, f"Missing basis functions for atom {a}"
            self.atoms.append(BasisFunctionsNew.AtomData(phi_M, a))
        self.num_atoms = len(self.atoms)

        # check that all basis functions are consistently on either CPU or GPU
        xp = self.atoms[0].basis_functions[0].xp
        for atom in self.atoms:
            for phi in atom.basis_functions:
                assert phi.xp is xp, "Basis function mismatch: mixing np, cp"

        self.xp = xp

        self.phi_MG_cache: Optional[np.ndarray | cp.ndarray] = None
        """Cached values of basis funcs, phi_mu(x).
        This is 4D: first index is mu (M), then x, y, z indices.
        """
        self.atoms_placed = False  # Set to True once positions have been set

    @cached_property
    def mu_range(self) -> range:
        """Range of mu indices"""
        mu_start = 0
        mu_end = mu_start
        for atom in self.atoms:
            mu_end += atom.num_basis_functions
        return range(mu_start, mu_end)

    def num_basis_functions(self) -> int:
        """"""
        return self.mu_range.stop - self.mu_range.start

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
        return range(len(self.atoms))

    @trace
    def set_positions(self, relpos_ac: np.ndarray) -> None:
        """Sets atom positions"""
        relpos_ac = np.asanyarray(relpos_ac)
        assert relpos_ac.shape[0] == self.num_atoms

        # atom positions in real units
        pos_av = relpos_ac @ self.grid.cell_cv
        for atom in self.atoms:
            atom.set_position(pos_av[atom.index])

        self.atoms_placed = True

        # TODO: make precalculation optional
        self.precalculate_on_grid()

    def set_matrix_distribution(self, Mstart: int, Mstop: int) -> None:
        """TODO!"""
        self.Mstart = Mstart
        self.Mstop = Mstop

    @trace
    def precalculate_on_grid(self) -> None:
        """Computes and caches phi_mu(x) = f_al(r) * Y_l^m everywhere on the
        grid.
        TODO: Boundary conditions! f(r) can contribute to neighboring cells
        TODO: Probably need faster way of handling values beyond the r cutoff
        TODO: Use GPU. Currently this computes on CPU and copies it to GPU
        """
        assert self.atoms_placed, "Must call set_positions() first"

        num_mu = self.mu_range.stop - self.mu_range.start
        # get all grid points in real units. 4D array of shape (Nx, Ny, Nz, 3)
        x_Gv = self.grid.xyz()

        shape = (num_mu, x_Gv.shape[0], x_Gv.shape[1], x_Gv.shape[2])

        """Compute on CPU. In principle the following could work on GPU with
        small modifications (spline should already work), however:
        - The masking is probably not optimal
        - Should implement spherical harmonics for GPU and fuse the computation
        """

        phi_MG_cache = np.zeros(shape, dtype=np.float64)

        # running work index
        mu_work = self.mu_range.start

        for atom in self.atoms:
            d_Gv = x_Gv - atom.position_v
            d_G = (d_Gv**2).sum(axis=3)**0.5

            for phi in atom.basis_functions:
                # Points that are too far from the atom are left at value 0
                mask_G = (d_G < phi.cutoff)

                f_R_masked = as_xp(phi.spline(d_G[mask_G]), xp=np)  # FIXME
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

    # Should keep the API as close to the old BasisFunctions as possible,
    # so use raw np or cp arrays instead of UGArray/Matrix in the following

    @trace
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

        num_spins = nt_sG.shape[0]
        f_sM = self.xp.empty((num_spins, self.num_basis_functions()))

        if self.phi_MG_cache is not None:
            # Copy the f_asi dict to a single array
            for atom in self.atoms:
                atom_mu_range = self.mu_range_a[atom.index]
                f_sM[:, atom_mu_range.start:atom_mu_range.stop] = (
                    f_asi[atom.index]
                )

            nt_sG += self.xp.einsum('sm, mxyz -> sxyz',
                                    f_sM, self.phi_MG_cache**2,  # type:ignore
                                    optimize=True)  # type:ignore
        else:
            raise NotImplementedError("BasisFunctions without precalculation")

    @trace
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
            raise NotImplementedError("BasisFunctions without precalculation")

    def calculate_potential_matrices(
            self,
            vt_G: np.ndarray | cp.ndarray) -> np.ndarray | cp.ndarray:
        """Alias for calculate_potential_matrix"""
        return self.calculate_potential_matrix(vt_G)

    def cache_memory_estimate(self) -> int:
        """Estimate of how many bytes are needed to precalculate and cache the
        basis functions on the grid.
        """
        num_sites = int(np.prod(self.grid.size))
        # TODO single precision. Do we need to allow complex phi?
        return num_sites * self.num_basis_functions() * 8
