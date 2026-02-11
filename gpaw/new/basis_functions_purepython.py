from gpaw.new.basis_functions import (BasisFunctionCollectionBase,
                                      BasisFunctionDesc, AtomTypeId,
                                      LFCAtomDefinition)
from gpaw.gpu import cupy as cp, cupy_is_fake, as_xp
from gpaw.spline import Spline
from gpaw.sphere.spherical_harmonics import Y

from typing import Protocol, runtime_checkable, override
import numpy as np
from dataclasses import dataclass

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


@dataclass
class VirtualAtom:
    """"""
    real_atom_idx: int
    phi_j: list[BasisFunction]
    """List of just those basis funcs that contribute from this image"""
    position: np.ndarray
    """Real-space position of the image. This is always outside the "main"
    cell"""

class BasisFunctionCollection_PurePython(BasisFunctionCollectionBase):
    """Pure Python implementation of BasisFunctionCollection.
    Not very optimized.
    """

    _virtual_atoms_a: dict[int, list[VirtualAtom]]
    """Dict mapping atom index -> list of virtual images of that atom"""

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

        # TEMP TEMP
        self._update_boundaries()

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

        # Handle virtual atoms from neighboring cells
        for atom_idx in range(self.num_atoms):
            virtual_atoms = self._virtual_atoms_a.get(atom_idx)
            if virtual_atoms:
                for atom in virtual_atoms:
                    d_Gv = x_Gv - atom.position
                    d_G = (d_Gv**2).sum(axis=3)**0.5

                    for phi in self.get_phi_list(atom_idx):
                        # Points that are too far from the atom are left at value 0
                        mask_G = (d_G < phi.cutoff)

                        f_R_masked = phi.spline(d_G[mask_G])
                        d_Rv_masked = d_Gv[mask_G]

                        mu = self.get_mu_range_a(atom.real_atom_idx).start

                        for m in range(0, phi.num_m()):
                            phi_MG_cache[mu + m, mask_G] = (
                                f_R_masked * Y(phi.l**2 + m, *d_Rv_masked.T)
                            )

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


    def _update_boundaries(self) -> None:
        """"""
        self._virtual_atoms_a = {}

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

        for atom_idx in range(self.num_atoms):

            type_id = self._atom_type_a[atom_idx]
            center = self._position_av[atom_idx]

            # Holds virtual copies of this atom in each shifted cell
            virtual_copies: dict[tuple[int, int, int], VirtualAtom] = {}

            for phi in self._phi_t_j[type_id]:

                r = phi.cutoff

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
                    key = (
                        int(shift_vec[0]),
                        int(shift_vec[1]),
                        int(shift_vec[2])
                    )

                    if key not in virtual_copies:
                        virtual_copies[key] = VirtualAtom(
                            real_atom_idx=atom_idx,
                            phi_j=[phi],
                            position=virtual_pos.copy()
                        )
                    else:
                        # virtual atom already created; append this phi
                        virtual_copies[key].phi_j.append(phi)
            # end phi loop

            if virtual_copies:
                self._virtual_atoms_a[atom_idx] = list(virtual_copies.values())
