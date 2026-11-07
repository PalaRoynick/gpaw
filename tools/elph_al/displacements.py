"""Use independently initialized symmetry for every displaced geometry."""
from pathlib import Path

from gpaw.elph import DisplacementRunner


class FreshDisplacementRunner(DisplacementRunner):
    """Avoid reusing a little group when changing the displacement direction.

    With point-group symmetry enabled, an x displacement and a y displacement
    need different symmetry groups. Reusing a converged GPAW calculator can
    otherwise raise 'Broken symmetry!'. Fresh calculators retain all physical
    settings and permit symmetry reduction for each geometry independently.
    """

    def calculate(self, atoms_N, disp):
        log = Path(self.name).parent / f'scf-{disp.name}.txt'
        atoms_N.calc = self.calc.new(txt=str(log))
        return self(atoms_N)
