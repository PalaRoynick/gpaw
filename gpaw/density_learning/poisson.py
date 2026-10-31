"""Stage FD electrostatics on the host while keeping boundary corrections."""

import numpy as np

from gpaw.poisson import _PoissonSolver, PoissonSolver


class HostStagedPoisson(_PoissonSolver):
    """One-rank FD adapter; no GPU-aware MPI or rewritten physical solver."""

    def __init__(self, parameters):
        self.parameters = parameters

    def todict(self):
        return {'name': 'HostStagedPoisson', 'solver': self.parameters}

    def get_description(self):
        return 'FD electrostatics on CPU with explicit host/device transfers'

    def build(self, grid, xp):
        from gpaw.new.poisson import PoissonSolver as NewPoisson

        solver = PoissonSolver(**self.parameters).build(grid, np)

        class StagedSolver(NewPoisson):
            def __str__(self):
                return 'Host-staged GPU FD Poisson:\n' + str(solver)

            def solve(self, potential, charge):
                host_potential = potential.to_xp(np)
                result = solver.solve(host_potential, charge.to_xp(np))
                if potential.xp is not np:
                    potential.data[:] = potential.xp.asarray(host_potential.data)
                return result

            def vacuum_levels(self, potential):
                return solver.vacuum_levels(potential)

        return StagedSolver()
