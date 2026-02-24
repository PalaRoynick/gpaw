from gpaw.new.basis_functions import (BasisFunctionCollectionBase,
                                      AtomTypeId,
                                      LFCAtomDefinition,
                                      GridBlock)
from gpaw.core import UGDesc
from gpaw.gpu import cupy as cp, cupy_is_fake, as_xp
from gpaw.sphere.spherical_harmonics import Y

from typing import override, TYPE_CHECKING
import numpy as np
import itertools


    # def to_gpu(self,
    #         keep_cpu: bool) -> None:
    #     """Creates a GPU version of the spline."""

    #     # Cupyx does not have a direct CubicSpline replacement, so we use
    #     # a PPoly object (generic piecewise polynomial).
    #     # TODO handle fake cupy?
    #     from cupyx.scipy.interpolate import PPoly
    #     self.spline_gpu = PPoly(cp.asarray(self.spline.c),
    #                             cp.asarray(self.spline.x),
    #                             extrapolate=False)

    #     if not keep_cpu:
    #         self.spline = None


class BasisFunctionCollection_PurePython(BasisFunctionCollectionBase):
    """Pure Python implementation of BasisFunctionCollection.
    Not very optimized.
    """

    @override
    def precalculate_on_grid(self) -> None:
        """
        """

        def get_block_xyz(block: GridBlock) -> np.ndarray:
            """Gives real-space XYZ points for the given block.
            Does NOT give points that extend beyond the grid domain boundary,
            which can happen if the division into blocks was uneven.
            4D array of shape (Nx, Ny, Nz, 3)."""
            # See UGDesc.xyz()
            shape = block.clipped_end_c - block.start_c
            indices_Rc = np.indices(tuple(shape)).transpose((1, 2, 3, 0))
            indices_Rc += block.start_c
            return indices_Rc @ (block.grid.cell_cv.T / block.grid.size_c).T

        # TODO precalculation on GPU
        for block in self._block_map.values():
            num_m = len(block.M_m)
            block_shape = (self.block_size, self.block_size, self.block_size)
            """Blocks at end boundary can be "too large" => treat still as
            having the same shape as other blocks, leave phi_mg empty in
            the extra points. This simplifies array operations between
            different blocks.
            """
            block.evaluated_phi_mg = np.zeros((num_m, *block_shape),
                                              dtype=np.float64)

            x_Gv = get_block_xyz(block)

            mu_local = 0

            for phi in block.phi_j:
                # Get distance from phi (atom) center for each XYZ point
                d_Gv = x_Gv - phi.position
                d_G = (d_Gv**2).sum(axis=3)**0.5
                f_r = phi.phi_class.spline(d_G)

                # purepython splines return NaN if r > rmax => change to 0.0
                f_r = np.nan_to_num(f_r, nan=0.0)

                l = phi.get_angular_momentum_number()
                for m in range(0, phi.get_num_mu()):
                    block.evaluated_phi_mg[mu_local + m] = (
                        f_r * Y(l**2 + m, *d_Gv.T)
                    )
                mu_local += phi.get_num_mu()

            assert np.all(np.isfinite(block.evaluated_phi_mg))


    @override
    def has_precalculated_phi(self) -> bool:
        """"""
        #return self.phi_MG_cache is not None
        return True

    @override
    def add_to_density(
        self,
        nt_sG: np.ndarray | cp.ndarray,
        f_asi: dict[int, np.ndarray] | dict[int, cp.ndarray]) -> None:
        r"""Add linear combination of squared localized basis functions to
        density:

            nt_s(x) += \sum_a \sum_i f_{asi} |\phi_{ai}|^2

        where i runs over all basis functions for said atom.

        Parameters
        ----------
        nt_sG : np.ndarray | cp.ndarray
            The density array to which contributions are added. Must be domain
            aware: if (nx, ny, nz) is the shape of this MPI grid domain, nt_sG
            must have shape (num_spins, nx, ny, nz).
            Modified in-place.
        f_asi : dict[int, np.ndarray] | dict[int, cp.ndarray]
            Dictionary that maps atom indices to occupation coefficient
            arrays. Each array has shape (num_spins, n_a), if n_a is the
            number of basis functions for that atom.
        Raises
        ------
        AssertionError
            If input array/dict shapes are incorrect.
        """

        num_spins = nt_sG.shape[0]
        assert np.all(self.grid.mysize_c == nt_sG.shape[1:])

        if self.has_precalculated_phi():
            # Flatten f_asi dict to a single array, ordered by global mu
            f_sM = np.zeros((num_spins, self.Mmax))

            for atom_idx in range(self.num_atoms):
                atom_mu_range = self._mu_range_a[atom_idx]
                f_sM[:, atom_mu_range.start:atom_mu_range.stop] = (
                    f_asi[atom_idx]
                )

            for block in self._block_map.values():
                assert block.evaluated_phi_mg is not None

                sx, sy, sz = block.start_c
                ex, ey, ez = block.end_c

                phi2_mg = block.evaluated_phi_mg**2

                for local_m, mu in enumerate(block.M_m):

                    for s in range(num_spins):
                        nt_sG[s, sx:ex, sy:ey, sz:ez] += (
                            f_sM[s, mu] * phi2_mg[local_m]
                        )

        else:
            raise NotImplementedError(
                "PurePython BasisFunctions without precalculation")

        #     for atom in self.atoms_a:
        #         assert atom.index in f_asi
        #         assert f_asi[atom.index].shape[0] == num_spins

        #         """Loop over all basis funcs of the atom, including those
        #         that extend from other cells. Compute also the range of
        #         'i' indices corresponding to the phi; this is necessary
        #         because some phi_j of the original atom may not extend to the
        #         main cell from other cells."""
        #         for phi in atom.phi_instances:
        #             mu_start = phi.first_mu
        #             mu_end = mu_start + phi.get_num_mu()  # exclusive
        #             num_mu = mu_end - mu_start

        #             i_start = mu_start - atom.first_mu
        #             i_end = i_start + num_mu
        #             nt_sG += np.einsum(
        #                 "sm, mxyz -> sxyz",
        #                 f_asi[atom.index][:, i_start:i_end],
        #                 phi.phi_mG**2,
        #                 optimize=True)

                    # for m in range(mu_end - mu_start):
                    #     nt_sG += f_sM[:, mu_start + m] * phi_mG[m, :]**2


        # Copy the f_asi dict to a single array
        # f_sM = self.xp.empty((num_spins, self.num_basis_functions()))
        # for atom_idx in range(self.num_atoms):
        #     atom_mu_range = self._mu_range_a[atom_idx]
        #     f_sM[:, atom_mu_range.start:atom_mu_range.stop] = (
        #         f_asi[atom_idx]
        #     )
        #     # does not work, periodic copies are not accounted for
        #     nt_sG += self.xp.einsum('sm, mxyz -> sxyz',
        #                             f_sM, self.phi_MG_cache**2,  # type:ignore
        #                             optimize=True)  # type:ignore


    @override
    def calculate_potential_matrix(
        self,
        vt_G: np.ndarray | cp.ndarray,
        out: np.ndarray | cp.ndarray | None = None) \
        -> np.ndarray | cp.ndarray:
        """"""
        xp = cp if isinstance(vt_G, cp.ndarray) else np

        num_work_rows = (self._matrix_distribution_rules.mu_end
                         - self._matrix_distribution_rules.mu_start)
        if (num_work_rows <= 0):
            # nothing to do
            if out:
                return out
            else:
                return xp.empty((0))

        if out:
            if out.ndim != 2:
                raise ValueError("out array must be 2D and have enough rows")
            rows, cols = out.shape
            if rows < num_work_rows or cols != self.num_basis_functions():
                raise ValueError("Not enough rows or columns in out array")
            out[:] = 0

        M = self.num_basis_functions()
        res = out or xp.zeros((num_work_rows, M))

        if self.has_precalculated_phi():
            self._potential_matrix_with_precalculation(vt_G, res,
                self._matrix_distribution_rules.mu_start,
                self._matrix_distribution_rules.mu_end)
            return res
        else:
            raise NotImplementedError(
                "PurePython BasisFunctions without precalculation")

        # if self.has_precalculated_phi():
        #     """Double loop over all phi_mu that contribute.
        #     We only need the lower half of the matrix, however this is quite
        #     tricky to achieve without one extra loop (slow).
        #     So instead we do some elements in the upper half too,
        #     while avoiding those that are easy to drop. Seems consistent with
        #     what the old CPU-LFC code does (comparison tests match exactly).
        #     """
        #     all_phi = self.get_phi_instances()

        #     first_mu = self._matrix_distribution_rules.mu_start
        #     # inclusive last_mu
        #     last_mu = self._matrix_distribution_rules.mu_end + 1
        #     phi_for_mu = self.get_phi_for_mu(first_mu, last_mu - 1)

        #     for phi1, phi2 in itertools.product(phi_for_mu, all_phi):
        #         assert phi1.phi_mG is not None
        #         assert phi2.phi_mG is not None

        #         mu = phi1.first_mu
        #         mu_end = mu + phi1.get_num_mu()
        #         nu = phi2.first_mu
        #         nu_end = nu + phi2.get_num_mu()

        #         # Clamp mu range to drop some nu > mu.
        #         mu = max(mu, nu)
        #         if mu >= mu_end:
        #             continue

        #         # Depending on mu_start, mu_end it can happen that phi1
        #         # contains mu indices outside this range => clamp
        #         mu = clamp(mu, first_mu, last_mu)
        #         mu_end = clamp(mu_end, first_mu, last_mu)
        #         # can't fail if our phi_for_mu was obtained correctly:
        #         assert mu_end > mu

        #         p1_m = mu - phi1.first_mu
        #         p1_m_end = mu_end - phi1.first_mu

        #         p1_vt_mG = phi1.phi_mG[p1_m:p1_m_end] * vt_G
        #         # "Integrate", ie. contract the grid indices
        #         res[(mu - first_mu):(mu_end - first_mu), nu:nu_end] \
        #             += np.einsum("mxyz, nxyz -> mn", p1_vt_mG, phi2.phi_mG,
        #                          optimize=True)

        #     return res * self.grid.dv
        # else:
        #     raise NotImplementedError(
        #         "PurePython BasisFunctions without precalculation")

    def _potential_matrix_with_precalculation(
        self,
        vt_G: np.ndarray | cp.ndarray,
        out: np.ndarray | cp.ndarray,
        mu_start: int,
        mu_end: int) -> None:
        """"""

        # FIXME seems to be broken if row_range was non-default

        all_phi = self.get_phi_instances()
        for block in self._block_map.values():
            assert block.evaluated_phi_mg is not None

            # Get slice of the potential in this block
            vt_g = vt_G[block.start_c[0]:block.clipped_end_c[0],
                        block.start_c[1]:block.clipped_end_c[1],
                        block.start_c[2]:block.clipped_end_c[2]]

            phi_mg = block.evaluated_phi_mg
            phi_mu_vt_g = phi_mg * vt_g
            # Integrate, ie. contract grid indices. Dense block-sized matrix
            V_mn = np.einsum("mxyz, nxyz -> mn", phi_mu_vt_g, phi_mg,
                             optimize=True)

            for m, mu in enumerate(block.M_m):
                for n, nu in enumerate(block.M_m):

                    if mu >= nu and mu in range(mu_start, mu_end):
                        # account for row-distributed output matrix
                        mu -= mu_start
                        out[mu, nu] += V_mn[m, n]
        #
        out *= self.grid.dv
