from ase.dft.bee import BEEFEnsemble
from ase.io import read
from ase.parallel import paropen
from gpaw import GPAW, PW, FermiDirac

li_metal = read('Li-metal-DFTD3.traj')  # Change file name accordingly

calc = GPAW(mode=PW(500),
            kpts=(8, 8, 8),
            occupations=FermiDirac(0.15),
            nbands=-10,
            txt=None,
            xc='BEEF-vdW')

li_metal.calc = calc
li_metal.get_potential_energy()
li_metal.write('li_metal.traj')
# snippet-beef
ens = BEEFEnsemble(calc)
li_metal_ens_cell= ens.get_ensemble_energies(2000)
with paropen('ensemble_li_metal.dat', 'a') as result:
    for e in li_metal_ens_cell:
        print(e, file=result)
