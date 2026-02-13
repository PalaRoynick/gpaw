from gpaw.new.basis_functions import (BasisFunctionCollectionBase,
                                      AtomTypeId,
                                      LFCAtomDefinition)
from gpaw.gpu import cupy as cp, cupy_is_fake, as_xp
from gpaw.sphere.spherical_harmonics import Y

from typing import override
import numpy as np


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

        # Copy the f_asi dict to a single array
        f_sM = self.xp.empty((num_spins, self.num_basis_functions()))
        for atom_idx in range(self.num_atoms):
            atom_mu_range = self._mu_range_a[atom_idx]
            f_sM[:, atom_mu_range.start:atom_mu_range.stop] = (
                f_asi[atom_idx]
            )

        if self.has_precalculated_phi():
            for atom in self.atoms_a:

                assert atom.phi_wmG is not None
                for w, phi_mG in enumerate(atom.phi_wmG):
                    mu_start = atom.mu_range_w[w].start
                    mu_end = atom.mu_range_w[w].stop
                    for m in range(mu_end - mu_start):
                        nt_sG += f_sM[:, mu_start + m] * phi_mG[m, :]**2


        # if self.phi_MG_cache is not None:
        #     nt_sG += self.xp.einsum('sm, mxyz -> sxyz',
        #                             f_sM, self.phi_MG_cache**2,  # type:ignore
        #                             optimize=True)  # type:ignore


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
