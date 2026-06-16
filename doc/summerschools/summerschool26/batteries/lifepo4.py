from ase.dft.bee import BEEFEnsemble
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.parallel import paropen
from gpaw import GPAW, PW, FermiDirac, Mixer

lifepo4_wo_li = read('lifepo4_wo_li.traj')
lifepo4 = lifepo4_wo_li.copy()
cell = lifepo4.get_cell()
for coords in [(0, 0, 0), (0, 0.5, 0), (0.5, 0.5, 0.5), (0.5, 0, 0.5)]:
    lifepo4.append('Li')
    lifepo4.positions[-1] = coords @ cell
write('lifepo4.traj', lifepo4)
# Read in the structure you made and wrote to file above
lifepo4 = read('lifepo4.traj')
params = dict(
    mode=PW(500),  # plane wave energy cutoff
    nbands=-40,  # the number on empty bands had the system been spin-paired
    kpts={'size': (2, 4, 5),  # k-point mesh
          'gamma': True},
    spinpol=True,  # performing spin polarized calculations
    xc='BEEF-vdW',  # exchange-correlation functional
    occupations=FermiDirac(
        width=0.1),  # smearing
    convergence={'eigenstates': 1.0e-4,  # eV^2 / electron
                 'energy': 2.0e-4,  # eV / electron
                 'density': 1.0e-3},
    mixer=Mixer(0.1, 5, weight=100.0))  # mixer used during SCF optimization
params['setups'] = {'Fe': ':d,4.3'}  # U=4.3 applied to d orbitals
calc = GPAW(txt='lifepo4.txt', **params)
lifepo4.calc = calc
epot_lifepo4_cell = lifepo4.get_potential_energy()
print('E_Pot=', epot_lifepo4_cell)

traj = Trajectory('lifepo4_out.traj', mode='w', atoms=lifepo4)
traj.write()

ens = BEEFEnsemble(calc)
dE = ens.get_ensemble_energies(2000)
result = paropen('ensemble_lifepo4.dat', 'a')
for e in dE:
    print(e, file=result)
result.close()
