from ase.io import write
from ase.parallel import paropen
from gpaw import GPAW, FermiDirac, Mixer, PW
from ase.dft.bee import BEEFEnsemble
from ase import Atoms
fepo4 = Atoms(
    'Fe4O16P4',
    positions=[
        [2.73015081, 1.46880951, 4.56541172],
        [2.23941067, 4.40642872, 2.14957739],
        [7.20997230, 4.40642925, 0.26615813],
        [7.70070740, 1.46880983, 2.68199421],
        [1.16033403, 1.46881052, 3.40240205],
        [3.80867172, 4.40642951, 0.98654342],
        [8.77981469, 4.40642875, 1.42923946],
        [6.13142032, 1.46881092, 3.84509827],
        [4.37288562, 1.46880982, 0.81812712],
        [0.59764596, 4.40643021, 3.23442747],
        [5.56702590, 4.40642886, 4.01346264],
        [9.34268360, 1.46880929, 1.59716233],
        [1.64001691, 0.26061277, 1.17298291],
        [3.32931769, 5.61463705, 3.58882629],
        [8.30013707, 3.19826250, 3.65857000],
        [6.61076951, 2.67698811, 1.24272700],
        [8.30013642, 5.61459688, 3.65856912],
        [6.61076982, 0.26063178, 1.24272567],
        [1.64001666, 2.67700652, 1.17298270],
        [3.32931675, 3.19822249, 3.58882660],
        [0.90585688, 1.46880966, 1.89272372],
        [4.06363530, 4.40642949, 4.30853266],
        [9.03398503, 4.40642957, 2.93877879],
        [5.87676435, 1.46881009, 0.52297232]],
    cell=[9.94012, 5.87524, 4.83157],
    pbc=[1, 1, 1])
for atom in fepo4:
    if atom.symbol == 'Fe':
        atom.magmom = 5.0
magmoms = fepo4.get_initial_magnetic_moments()
print(magmoms)
# snippet-params
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
# snippet-u
params['setups'] = {'Fe': ':d,4.3'}  # U=4.3 applied to d orbitals
# snippet-calc
calc = GPAW(**params)
fepo4.calc = calc
epot_fepo4_cell = fepo4.get_potential_energy()
print(epot_fepo4_cell)
write('fepo4_out.traj', fepo4)
# snippet-beef
ens = BEEFEnsemble(calc)
dE = ens.get_ensemble_energies(2000)
# snippet-ensemble
with paropen('ensemble_fepo4.dat', 'a') as result:
    for e in dE:
        print(e, file=result)
