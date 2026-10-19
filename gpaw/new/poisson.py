from math import nan

from ase.units import Ha


class PoissonSolver:
    def solve(self,
              vHt,
              rhot) -> float:
        raise NotImplementedError

    def vacuum_levels(self, potential) -> tuple[float, float]:
        """The xy-averaged electrostatic potential (Ha) in the vacuum at the
        top and at the bottom of the cell, for the work functions.

        Returns NaN if there is no vacuum; raises NotImplementedError if there
        is no dipole correction.
        """
        raise NotImplementedError


class PoissonSolverWrapper(PoissonSolver):
    """Adapt an old-GPAW grid Poisson solver to the PoissonSolver interface.

    Created by the old solvers' build() methods (gpaw.poisson and
    gpaw.dipole_correction.DipoleCorrection). If the wrapped solver has a
    ``correction`` attribute, it is taken to be half the potential step of
    a dipole correction whose step sits at the cell boundary.
    """

    def __init__(self, solver):
        self.description = solver.get_description()
        self.solver = solver

    def __str__(self):
        return self.description

    def solve(self,
              vHt,
              rhot) -> float:
        self.solver.solve(vHt.data, rhot.data)
        return nan

    def vacuum_levels(self, potential) -> tuple[float, float]:
        try:
            correction = self.solver.correction
        except AttributeError:
            raise NotImplementedError
        v0 = potential.get_vacuum_level() / Ha
        return v0 - correction, v0 + correction
