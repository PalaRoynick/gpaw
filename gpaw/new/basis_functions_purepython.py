from gpaw.new.basis_functions import (BasisFunctionCollectionBase,
                                      BasisFunctionDesc, AtomTypeId,
                                      LFCAtomDefinition)
from gpaw.gpu import cupy as cp, cupy_is_fake, as_xp
from gpaw.spline import Spline
from gpaw.sphere.spherical_harmonics import Y

from typing import Protocol, runtime_checkable, override
import numpy as np

@runtime_checkable
class SplineLike(Protocol):
    """"""

    def __call__(self, x: np.ndarray) -> np.ndarray:
        ...


class BasisFunction:
    """Defines a basis function: phi_lm(x) = f(r)*Y_lm.
    """

    spline: SplineLike
    spline_gpu = None

    def __init__(self,
                desc: BasisFunctionDesc):
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
        assert desc.l >= 0
        assert desc.cutoff > 0
        assert len(desc.f_r) > 2, "Need at least 3 points to define a spline"
        self.l = desc.l
        self.cutoff = desc.cutoff

        # TODO handle single precision
        self._init_spline(desc.f_r)

    @staticmethod
    def make_random(l: int,
                    cutoff: float,
                    rng: np.random.Generator,
                    num_points: int = 100) -> "BasisFunction":
        """Generates a basis function with randomly generated radial part
        using the provided RNG.
        """
        values = rng.random(size=num_points)
        # last value must be 0
        values[-1] = 0.0
        return BasisFunction(l, cutoff, values)

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


    def num_m(self) -> int:
        """How many values of m does this describe. IE: just 2*l + 1."""
        return 2 * self.l + 1

    def to_gpu(self,
            keep_cpu: bool) -> None:
        """Creates a GPU version of the spline."""

        # Cupyx does not have a direct CubicSpline replacement, so we use
        # a PPoly object (generic piecewise polynomial).
        # TODO handle fake cupy?
        from cupyx.scipy.interpolate import PPoly
        self.spline_gpu = PPoly(cp.asarray(self.spline.c),
                                cp.asarray(self.spline.x),
                                extrapolate=False)

        if not keep_cpu:
            self.spline = None

    @staticmethod
    def from_old_spline(spline: Spline, purepython = False) -> "BasisFunction":
        """"""
        num_x: int = spline.get_npoints()
        x = np.linspace(0.0, spline.get_cutoff(), num_x)

        return BasisFunction(spline.get_angular_momentum_number(),
                             spline.get_cutoff(),
                             spline.map(x),
                             purepython)


class BasisFunctionCollection_PurePython(BasisFunctionCollectionBase):
    """Pure Python implementation of BasisFunctionCollection.
    Not very optimized.
    """

    def get_phi_list(self, atom_idx: int) -> list[BasisFunction]:
        """Return basis functions for the given atom."""
        return self._phi_t_j.get(self._atom_type_a[atom_idx])  # type: ignore

    @override
    def _init_basis_functions(
        self,
        atom_types: list[LFCAtomDefinition]) -> None:
        """"""

        # Dict that maps atom type ID -> list of basis funcs
        self._phi_t_j: dict[AtomTypeId, list[BasisFunction]] = {}

        for atom_type in atom_types:
            self._phi_t_j[atom_type.id] \
                = [BasisFunction(desc)
                   for desc in atom_type.basis_functions]

    @override
    def precalculate_on_grid(self) -> None:
        """Computes and caches phi_mu(x) = f_al(r) * Y_l^m everywhere on the
        grid.
        TODO: Boundary conditions! f(r) can contribute to neighboring cells
        """

        num_mu = self.mu_range.stop - self.mu_range.start
        # get all grid points in real units. 4D array of shape (Nx, Ny, Nz, 3)
        x_Gv = self.grid.xyz()

        shape = (num_mu, x_Gv.shape[0], x_Gv.shape[1], x_Gv.shape[2])

        # running work index
        mu_work = self.mu_range.start

        # TODO precalculation on GPU. For now, do it on CPU and send to GPU
        phi_MG_cache = np.zeros(shape, dtype=np.float64)
        for atom_idx in range(self.num_atoms):
            d_Gv = x_Gv - self._position_av[atom_idx]
            d_G = (d_Gv**2).sum(axis=3)**0.5

            for phi in self.get_phi_list(atom_idx):
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
    def has_precalculated_phi(self) -> bool:
        """"""
        return self.phi_MG_cache is not None

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
            for atom_idx in range(self.num_atoms):
                atom_mu_range = self._mu_range_a[atom_idx]
                f_sM[:, atom_mu_range.start:atom_mu_range.stop] = (
                    f_asi[atom_idx]
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