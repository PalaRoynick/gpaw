# snippet-define-atoms-start
from ase import Atoms
from ase.visualize import view

a = 7.0  # lattice constant
c = 10.0  # c-axis
cell = [[a, 0, 0],
        [-0.5 * a, 3**0.5 / 2 * a, 0],
        [0, 0, c]]  # unit cell in Å
scaled_pos = [[2 / 3., 1 / 3., 0.5],
              [1 / 3., 2 / 3., 0.5],
              [0.6, 1.0, 0.35],
              [0.4, 0.4, 0.35],
              [0.0, 0.6, 0.35],
              [0.4, 1.0, 0.65],
              [0.6, 0.6, 0.65],
              [0.0, 0.4, 0.65]]  # positions in unit cell vectors
a = Atoms('Cr2I6', cell=cell, scaled_positions=scaled_pos, pbc=True)
view(a)
# snippet-define-atoms-end

S = 3 / 2  # Cr spin

# snippet-relax-start
from ase.optimize import BFGS
from ase.filters import UnitCellFilter

from gpaw import GPAW, PW

# Define a variable S for the Cr spin here

m = S * 2  # magnetic moment in Bohr magnetons
a.set_initial_magnetic_moments([m, m, 0, 0, 0, 0, 0, 0])

Nk = 2
calc = GPAW(mode=PW(200),
            xc='PBE',
            kpts=(Nk, Nk, 1))
a.calc = calc

uf = UnitCellFilter(a, mask=[1, 1, 0, 0, 0, 1])
opt = BFGS(uf)
opt.run(fmax=0.2)
calc.write('CrI3_relaxed.gpw')
# snippet-relax-end
