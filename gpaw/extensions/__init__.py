from __future__ import annotations

from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.poisson import PoissonSolver
from gpaw.core import UGArray


class Extension:
    """Extension object.

    Used for jellium, solvation, solvated jellium model, D3, ...
    """

    name = 'unnamed extension'
    charge = 0.0

    def get_energy_contributions(self) -> dict[str, float]:
        return {}

    def force_contribution(self, nt_r, vHt_r):
        return 0.0

    def stress_contribution(self):
        return 0.0

    def move_atoms(self, relpos_ac) -> None:
        return

    def update_non_local_hamiltonian(self,
                                     D_sii,
                                     setup,
                                     atom_index,
                                     dH_sii) -> float:
        return 0.0

    def build(self, builder):
        return self

    def create_poisson_solver(self,
                              grid,
                              pw,
                              *,
                              charge,
                              xp) -> PoissonSolver | None:
        return None

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             nelectrons: float,
                             occ_calc,
                             mixer,
                             log) -> bool:
        """Allow for environment to "converge"."""
        return True

    def update1(self, nt_r) -> None:
        """Hook called right before solving the Poisson equation."""
        pass

    def update1pw(self, nt_g) -> None:
        """PW-mode hook called right before solving the Poisson equation."""
        pass

    def update2(self, nt_r, vHt_r, vt_sr) -> float:
        """Calculate environment energy."""
        return 0.0

    def update_potential(self,
                         vt_sR: UGArray,
                         density) -> float:
        return 0.0
