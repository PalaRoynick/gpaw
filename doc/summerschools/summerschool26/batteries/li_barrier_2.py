from ase.io import read
from gpaw import GPAW, PW

images = read('neb.traj@-7:')
IS_image = images[0]
TS_image = images[3]
# snippet-barrier
epot_IS = IS_image.get_potential_energy()
epot_TS = TS_image.get_potential_energy()
barrier = epot_TS - epot_IS
print('Energy barrier:', barrier)
# snippet-strain
for strain in [-0.03, 0.03]:
    cell = IS_image.get_cell()
    cell[2] *= 1 + strain
    IS_image97 = IS_image.copy()
    IS_image97.set_cell(cell, scale_atoms=True)
    TS_image97 = TS_image.copy()
    TS_image97.set_cell(cell, scale_atoms=True)
    # Create GPAW calculator and calculate energies:
    # snippet-gpaw
    calc = GPAW(
        mode=PW(500),
        kpts=(5, 5, 6),
        xc='LDA',
        symmetry={'point_group': False})
    IS_image97.calc = calc
    epot_IS97 = IS_image97.get_potential_energy()
    TS_image97.calc = calc
    epot_TS97 = TS_image97.get_potential_energy()
    barrier97 = epot_TS97 - epot_IS97
    print("Energy barrier:", barrier97)
