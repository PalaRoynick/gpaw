from ase.io import read
from gpaw import GPAW, PW

layer = read('VI2_relaxed.gpw')
layer_afm = layer.repeat((2, 1, 1))
m = 3.0
layer_afm.set_initial_magnetic_moments([m, 0, 0, -m, 0, 0])
calc = GPAW(mode=PW(400),
            xc='LDA',
            kpts=(2, 4, 1),  # student: kpts=???,
            txt='V2I4_afm.txt')
layer_afm.calc = calc
layer_afm.get_potential_energy()
calc.write('V2I4_afm.gpw')
# Do the same for ferromagnetic structure:
layer_afm = layer.repeat((2, 1, 1))
m = 3.0
layer_afm.set_initial_magnetic_moments([m, 0, 0, m, 0, 0])
calc = GPAW(mode=PW(400),
            xc='LDA',
            kpts=(2, 4, 1),
            txt='V2I4_fm.txt')
layer_afm.calc = calc
layer_afm.get_potential_energy()
calc.write('V2I4_fm.gpw')
