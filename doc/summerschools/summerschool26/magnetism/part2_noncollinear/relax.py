S = 3 / 2

# snippet-relax-start
from ase.io import read
from ase.visualize import view
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from gpaw import GPAW, PW

# Define a variable S for the V spin here
# S = ???

m = S * 2
layer = read('VI2.xyz')
layer.set_initial_magnetic_moments([m, 0, 0])
view(layer)

Nk = 6
layer.calc = GPAW(
    mode=PW(600),
    xc='LDA',
    kpts={'size': (Nk, Nk, 1), 'gamma': True},
    txt='VI2_relax.txt')

uf = UnitCellFilter(layer, mask=[1, 1, 0, 0, 0, 1])
opt = BFGS(uf)
opt.run(fmax=0.1)

layer.get_potential_energy()

layer.calc.write('VI2_relaxed.gpw')
# snippet-relax-end
