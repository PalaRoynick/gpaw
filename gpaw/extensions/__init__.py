from __future__ import annotations

from gpaw.core import UGArray
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.new.poisson import PoissonSolver


class Extension:
    """Extension object.

    Used for jellium, solvation, solvated jellium model, D3, ...
    """

    name = 'unnamed extension'

    #: Charge the extension puts into the cell, in electrons -- but note the
    #: sign: it is *added* to the electron count (PotentialCalculator.charge,
    #: gpaw/new/scf.py), so positive means electrons added, the opposite of
    #: GPAW's own ``charge=`` keyword, where +1 removes one.  An extension
    #: that balances it with a counter charge subtracts it from the density
    #: (gpaw/extensions/jellium.py).
    #:
    #: TODO: rename to ``excess_electrons`` -- here, in the two sums over
    #: extensions (gpaw/new/pot_calc.py, gpaw/new/fd/pot_calc.py) and in
    #: JelliumExtension.  That would end the ambiguity and let SJM drop the
    #: charge/excess_electrons alias it needs today, but it changes the
    #: released Jellium(charge=...) keyword, so it needs a deprecation.
    charge = 0.0

    # Extra keys this extension adds to dft.results.  Mirrors
    # ExtensionInput.properties: either kind of object can be handed to the
    # extensions keyword (SpinDirectionConstraint is an Extension).
    properties: list[str] = []

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

    def update1pw(self, nt_g) -> None:
        """PW-mode hook called right before solving the Poisson equation."""

    def update2(self, nt_r, vHt_r, vt_sr) -> None:
        """Hook called right after solving the Poisson equation."""

    def update2pw(self, nt_g, vHt_h, vt_sr) -> None:
        """PW-mode hook called right after solving the Poisson equation."""

    def post_converged(self, dft) -> None:
        """Hook called after the SCF-loop has converged.

        Called from :meth:`gpaw.new.calculation.DFT.calculate_energy`
        once the canonical energies have been put into ``dft.results``.
        Can be used to write to ``dft.log`` and to add or replace entries
        in ``dft.results`` (which are handed on to ASE).  Any new key must
        have a unit-conversion factor in ``gpaw.new.calculation.units`` and
        be listed in the ``properties`` attribute of the input extension
        (:class:`gpaw.dft.ExtensionInput`), which is what puts it into
        ``ASECalculator.implemented_properties``.
        """

    def update_potential(self,
                         vt_sR: UGArray,
                         density) -> float:
        return 0.0
