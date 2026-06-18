from gpaw.new.basis_functions import (BasisFunctionCollectionBase,
                                      LFCSystemDesc, BasisFunctionDesc,
                                      PrecalculationMode)

from gpaw import GPAW_NO_C_EXTENSION
from gpaw.typing import override
from gpaw.gpu import cupy as cp
import numpy as np


class BasisFunctionCollection(BasisFunctionCollectionBase):
    """Version that uses GPAW C++ extension."""

    @override
    def __init__(self,
                 system: LFCSystemDesc,
                 use_gpu: bool = False,
                 mode: PrecalculationMode = PrecalculationMode.eIfPossible,
                 block_size: int | None = 8):
        """"""
        if GPAW_NO_C_EXTENSION:
            raise RuntimeError(
                "Cannot construct BasisFunctionCollection"
                "without GPAW C++ extension."
                "Use BasisFunctionCollection_PurePython instead.")

        if not use_gpu:
            raise RuntimeError("Only GPU version implemented for now")

        # TEMP Force no precalculation, so that we can test stuff
        mode = PrecalculationMode.eNever

        super().__init__(system, use_gpu=use_gpu, mode=mode,
                         block_size=block_size)

    @override
    def _init_splines(self, phi_i: list[BasisFunctionDesc]) -> None:
        """"""
        from gpaw.cgpaw.lfc import LFCBuilder

        # Same LFCBuilder class currently builds both splines and the LFC impl
        builder = LFCBuilder()
        for phi in phi_i:
            builder.add_spline(phi.cutoff, phi.f_r)

        self._spline_pool_cpu = builder.build_spline_pool()
        """Cached reference to the CPU spline pool."""

        if self.uses_gpu():
            from gpaw.cgpaw.lfc import SplinePoolGpu
            self._spline_pool_gpu = SplinePoolGpu(self._spline_pool_cpu)
            """Cached reference to the GPU spline pool."""
        else:
            self._spline_pool_gpu = None

    def _build_cpp_object(self):
        """"""
        from gpaw.cgpaw.lfc import LFCBuilder
        assert self._spline_pool_cpu is not None

        builder = LFCBuilder()
        builder.define_grid(self.grid.start_c, self.grid.end_c,
                            self.geometry.h_cv, self.grid.dv)
        builder.set_block_size(self.geometry.block_size_c)

        # phi_idx_map must be built in the same order as add_phi so that block
        # data indices match the C++ phi data array
        phi_idx_map = {}
        num_phi = 0
        for a in self.atom_indices:
            for phi in self.get_phi_instances_for_atom(a):
                builder.add_phi(phi.desc.l, phi.spline_index, phi.position,
                                phi.first_mu, phi.cell_coords)
                phi_idx_map[id(phi)] = num_phi
                num_phi += 1

        # TODO figure out a fast way of getting required block data
        # From set_positions() we already have:
        # - self.phi_instances_ai: all relevant phi_i for given atom
        # - self._block_overlaps_a: list of dicts, one for each atom.
        #   The dict maps BlockCoords to overlapping phi_i for that atom

        # We need:
        # - start_c for each block (easy)
        # - map block-local phi_j to global phi_i (hard)
        # Note that depending on a global phi index is not very scalable as it
        # practically necessitates a full rebuild of associated arrays
        # whenever any atom moves.

        # This is hacky and inefficient!
        for block_coords, block_phi_j in self.get_block_to_phi_map().items():
            block_phi_idx_to_global = []
            for phi in block_phi_j:
                block_phi_idx_to_global.append(phi_idx_map[id(phi)])

            if block_phi_idx_to_global:
                builder.add_block(self.geometry.block_start_Bc[block_coords],
                                  block_phi_idx_to_global)

        if self.uses_gpu():
            assert self._spline_pool_gpu is not None
            self._impl = builder.build_gpu(self._spline_pool_gpu)
        else:
            raise NotImplementedError("New LFC only support GPU for now")

    @override
    def set_positions(self, relpos_ac: np.ndarray, force_update: bool) -> bool:
        """"""
        has_migrations = super().set_positions(relpos_ac, force_update)
        # Must rebuild the C++ side object whenever atoms move
        self._build_cpp_object()
        return has_migrations

    @override
    def precalculate_on_grid(self) -> None:
        """"""
        raise NotImplementedError("WIP")

    @override
    def add_to_density(
        self,
        nt_sG: np.ndarray | cp.ndarray,
        f_asi: dict[int, np.ndarray] | dict[int, cp.ndarray]
    ) -> None:
        """"""

        num_spins = nt_sG.shape[0]
        assert np.all(self.grid.mysize_c == nt_sG.shape[1:])

        # Flatten f_asi dict to a single array, ordered by global mu.
        xp = np if isinstance(f_asi, np.ndarray) else cp
        f_sM = xp.zeros((num_spins, self.Mmax))

        for atom_idx in range(self.num_atoms):
            atom_mu_range = self.get_mu_range_a(atom_idx)
            f_sM[:, atom_mu_range.start:atom_mu_range.stop] = (
                f_asi[atom_idx]
            )

        self._impl.add_to_density(nt_sG, f_sM)

    @override
    def calculate_potential_matrix(
        self,
        vt_G: np.ndarray | cp.ndarray,
        out: np.ndarray | cp.ndarray | None = None
    ) -> np.ndarray | cp.ndarray:
        """"""
        assert self._impl is not None
        xp = cp if isinstance(vt_G, cp.ndarray) else np

        assert np.all(vt_G.shape == self.grid.mysize_c)

        num_work_rows = (self._matrix_distribution_rules.mu_end
                         - self._matrix_distribution_rules.mu_start)
        if (num_work_rows <= 0):
            # nothing to do
            if out:
                return out
            else:
                return xp.empty((0))

        M = self.Mmax
        if out is not None:
            if out.ndim != 2:
                raise ValueError("out array must be 2D and have enough rows")
            rows, cols = out.shape
            if rows < num_work_rows or cols != M:
                raise ValueError("Not enough rows or columns in out array")
            out[:] = 0
            res = out
        else:
            res = xp.zeros((num_work_rows, M))

        if self.has_precalculated_phi():
            raise NotImplementedError("BasisFunctionCollection with precalc")
        else:
            # FIXME won't work currently if using nontrivial matrix dist
            self._impl.calculate_potential_matrix(vt_G, res)
        return res

    @override
    def construct_density(self,
                          rho_MM,
                          nt_G: np.ndarray | cp.ndarray,
                          q):
        """"""
        raise NotImplementedError("WIP")

    @override
    def has_precalculated_phi(self) -> bool:
        """"""
        return False
