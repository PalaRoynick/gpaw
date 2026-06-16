import numpy as np
from ase.io import read
from ase.visualize import view
from gpaw import GPAW, PW, MixerDif

layer = read('VI2_relaxed.gpw')
m = 3
cell_cv = layer.get_cell()
layer_nc = layer.repeat((3, 1, 1))
new_cell_cv = [[3 * cell_cv[0, 0] / 2, 3**0.5 * cell_cv[0, 0] / 2, 0.0],
               [3 * cell_cv[0, 0] / 2, -3**0.5 * cell_cv[0, 0] / 2, 0.0],
               [0.0, 0.0, cell_cv[2, 2]]]
layer_nc.set_cell(new_cell_cv)
view(layer_nc)

magmoms = np.zeros((len(layer_nc), 3), float)
magmoms[0] = [m, 0, 0]
magmoms[3] = [m * np.cos(2 * np.pi / 3), m * np.sin(2 * np.pi / 3), 0]
magmoms[6] = [m * np.cos(2 * np.pi / 3), -m * np.sin(2 * np.pi / 3), 0]

calc = GPAW(mode=PW(400),
            xc='LDA',
            mixer=MixerDif(),
            symmetry='off',
            magmoms=magmoms,
            soc=False,
            kpts=(2, 2, 1))
layer_nc.calc = calc
layer_nc.get_potential_energy()
calc.write('nc_nosoc.gpw')

# Include similar calculations here for a ferromagnetic state using
# the non-collinear framework
# ...
# snippet-stop
magmoms[:] = [m, 0, 0]
calc = GPAW(mode=PW(400),
            xc='LDA',
            mixer=MixerDif(),
            symmetry='off',
            magmoms=magmoms,
            soc=False,
            kpts=(2, 2, 1))
layer_nc.calc = calc
layer_nc.get_potential_energy()
calc.write('nc_nosoc_fm.gpw')
