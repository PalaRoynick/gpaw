kpts = (2, 4, 1)

# snippet-afm-start
from ase.io import read
from gpaw import GPAW, PW

m = 3.0

# DFT calculation for the collinear antiferromagnetic structure

layer = read('VI2_relaxed.gpw')

# Define a variable 'kpts' here for the k-point sampling
# kpts = ???

layer_afm = layer.repeat((2, 1, 1))
layer_afm.set_initial_magnetic_moments([m, 0, 0, -m, 0, 0])
calc = GPAW(
    mode=PW(400),
    xc='LDA',
    kpts=kpts,
    txt='V2I4_afm.txt')
layer_afm.calc = calc

layer_afm.get_potential_energy()
calc.write('V2I4_afm.gpw')

# Do the same for the ferromagnetic structure
# snippet-afm-end

layer_fm = layer.repeat((2, 1, 1))
layer_fm.set_initial_magnetic_moments([m, 0, 0, m, 0, 0])
calc = GPAW(
    mode=PW(400),
    xc='LDA',
    kpts=(2, 4, 1),
    txt='V2I4_fm.txt')
layer_fm.calc = calc

layer_fm.get_potential_energy()
calc.write('V2I4_fm.gpw')
