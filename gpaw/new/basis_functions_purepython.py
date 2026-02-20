from gpaw.new.basis_functions import (BasisFunctionCollectionBase,
                                      AtomTypeId,
                                      LFCAtomDefinition)
from gpaw.gpu import cupy as cp, cupy_is_fake, as_xp
from gpaw.sphere.spherical_harmonics import Y

from typing import override
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
        """Computes and caches phi_mu(x) = f_al(r) * Y_l^m everywhere on the
        grid.
        """
        return
        num_mu = self.mu_range.stop - self.mu_range.start
        # get all grid points in real units. 4D array of shape (Nx, Ny, Nz, 3)
        x_Gv = self.grid.xyz()

        shape = (num_mu, x_Gv.shape[0], x_Gv.shape[1], x_Gv.shape[2])

        # TODO precalculation on GPU. For now, do it on CPU and send to GPU
        phi_MG_cache = np.zeros(shape, dtype=self.dtype)

        for atom_idx in range(self.num_atoms):
            phi_j = self._phi_aj[atom_idx]

            virtual_phi_j = self._virtual_phi_aj.get(atom_idx)
            if virtual_phi_j:
                phi_j += virtual_phi_j

            for phi in phi_j:
                # could potentially optimize by computing the position
                # difference per-atom instead of per-phi. But doing it per-phi
                # lets us handle virtual basis funcs in the same loop.
                d_Gv = x_Gv - phi.position
                d_G = (d_Gv**2).sum(axis=3)**0.5
                # Points that are too far from the atom are left at value 0
                mask_G = (d_G < phi.get_cutoff())

                f_R_masked = phi.phi_class.spline(d_G[mask_G])
                d_Rv_masked = d_Gv[mask_G]

                mu = phi.first_mu
                l = phi.get_angular_momentum_number()
                for m in range(0, phi.phi_class.get_num_mu()):
                    phi_MG_cache[mu + m, mask_G] = (
                        f_R_masked * Y(l**2 + m, *d_Rv_masked.T)
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
        #return self.phi_MG_cache is not None
        return True

    @override
    def add_to_density(
        self,
        nt_sG: np.ndarray | cp.ndarray,
        f_asi: dict[int, np.ndarray] | dict[int, cp.ndarray]) -> None:
        """"""
        num_spins = nt_sG.shape[0]

        if self.has_precalculated_phi():
            for atom in self.atoms_a:

                """Loop over all basis funcs of the atom, including those
                that extend from other cells. Compute also the range of
                'i' indices corresponding to the phi; this is necessary
                because some phi_j of the original atom may not extend to the
                main cell from other cells."""
                for phi in atom.phi_instances:
                    mu_start = phi.first_mu
                    mu_end = mu_start + phi.get_num_mu()  # exclusive
                    num_mu = mu_end - mu_start

                    i_start = mu_start - atom.first_mu
                    i_end = i_start + num_mu
                    nt_sG += np.einsum("sm, mxyz -> sxyz",
                                       f_asi[atom.index][:, i_start:i_end],
                                       phi.phi_mG**2,
                                       optimize=True)

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

        else:
            raise NotImplementedError(
                "PurePython BasisFunctions without precalculation")

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

        def clamp(val: int, min_val: int, max_val: int) -> int:
            return max(min_val, min(val, max_val))

        if self.has_precalculated_phi():
            """Double loop over all phi_mu that contribute.
            We only need the lower half of the matrix, however this is quite
            tricky to achieve without one extra loop (slow).
            So instead we do some elements in the upper half too,
            while avoiding those that are easy to drop. Seems consistent with
            what the old CPU-LFC code does (comparison tests match exactly).
            """
            all_phi = self.get_phi_instances()

            first_mu = self._matrix_distribution_rules.mu_start
            # inclusive last_mu
            last_mu = self._matrix_distribution_rules.mu_end + 1
            phi_for_mu = self.get_phi_for_mu(first_mu, last_mu - 1)

            for phi1, phi2 in itertools.product(phi_for_mu, all_phi):
                assert phi1.phi_mG is not None
                assert phi2.phi_mG is not None

                mu = phi1.first_mu
                mu_end = mu + phi1.get_num_mu()
                nu = phi2.first_mu
                nu_end = nu + phi2.get_num_mu()

                # Clamp mu range to drop some nu > mu.
                mu = max(mu, nu)
                if mu >= mu_end:
                    continue

                # Depending on mu_start, mu_end it can happen that phi1
                # contains mu indices outside this range => clamp
                mu = clamp(mu, first_mu, last_mu)
                mu_end = clamp(mu_end, first_mu, last_mu)
                # can't fail if our phi_for_mu was obtained correctly:
                assert mu_end > mu

                p1_m = mu - phi1.first_mu
                p1_m_end = mu_end - phi1.first_mu

                p1_vt_mG = phi1.phi_mG[p1_m:p1_m_end] * vt_G
                # "Integrate", ie. contract the grid indices
                res[(mu - first_mu):(mu_end - first_mu), nu:nu_end] \
                    += np.einsum("mxyz, nxyz -> mn", p1_vt_mG, phi2.phi_mG,
                                 optimize=True)

            return res * self.grid.dv
        else:
            raise NotImplementedError(
                "PurePython BasisFunctions without precalculation")
