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
            raise RuntimeError("Cannot construct BasisFunctionCollection" \
            "without GPAW C++ extension." \
            "Use BasisFunctionCollection_PurePython instead.")

        # if not use_gpu:
        #     raise RuntimeError("Only GPU version implemented for now")

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
        """Cached reference to the created spline pool."""


    def _build_cpp_object(self):
        """"""
        from gpaw.cgpaw.lfc import LFCBuilder
        assert self._spline_pool_cpu is not None

        builder = LFCBuilder()
        builder.set_block_size(self.geometry.block_size_c)

        for a in self.atom_indices:
            for phi in self.get_phi_instances_for_atom(a):
                builder.add_phi(phi.desc.l, phi.spline_index, phi.position,
                                phi.first_mu, phi.cell_coords)

        #

        # from gpaw.cgpaw.gpu import (BasisFunctionCollection_cpp,
        #                             BasisFunctionCollectionBuilder)
        # builder = BasisFunctionCollectionBuilder()

        # builder.set_block_size(self.block_size)
        # builder.enable_gpu(self.uses_gpu())
        # builder.enable_single_precision(self.is_single_precision())

        # builder.set_spline_pool(self.spline_pool)
        # builder.define_grid(self.grid.start_c,
        #                     self.grid.end_c,
        #                     np.diag(self.grid._gd.h_cv),  # FIXME handle more general grid shape
        #                     self.grid.dv)

        # blocks = self.get_relevant_blocks()
        # phi_i = self.get_phi_instances()

        # builder.reserve(len(phi_i), len(blocks))

        # for block in blocks:
        #     builder.add_block(
        #         tuple(block.start_c), block.shape, block.M_m, block.phi_idx_j)

        # for phi in phi_i:
        #     builder.add_phi(phi.get_angular_momentum_number(),
        #                     phi.index, phi.position, phi.first_mu)

        # self._impl = builder.build()

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
        f_asi: dict[int, np.ndarray] | dict[int, cp.ndarray]) -> None:
        """"""

        # Copy the f_asi dict to a single contiguous array
        # TODO is this something we could do once and cache?
        num_spins = nt_sG.shape[0]
        f_sM = self.xp.empty((num_spins, self.num_basis_functions()))

        for atom_idx in range(self.num_atoms):
            atom_mu_range = self._mu_range_a[atom_idx]
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
        raise NotImplementedError("WIP")

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
