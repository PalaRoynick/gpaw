from ase.io import read
from ase.visualize import view
from ase.optimize import BFGS
from ase.filters import UnitCellFilter
from gpaw import GPAW, PW

S = 3 / 2  # student: S = ???
m = S * 2
layer = read('VI2.xyz')
layer.set_initial_magnetic_moments([m, 0, 0])
view(layer)

layer.calc = GPAW(mode=PW(400), xc='LDA', kpts=(4, 4, 1))

uf = UnitCellFilter(layer, mask=[1, 1, 0, 0, 0, 1])
opt = BFGS(uf)
opt.run(fmax=0.1)

# layer.calc = layer.calc.new(symmetry='off')
layer.get_potential_energy()

layer.calc.write('VI2_relaxed.gpw')
