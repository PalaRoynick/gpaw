from ase.io import read, write
from gpaw import GPAW, PW, FermiDirac, Mixer
from ase.dft.bee import BEEFEnsemble
from ase.parallel import paropen

fepo4 = read('fepo4_out.traj')
fepo4_1li = fepo4.copy()
print(fepo4_1li.get_initial_magnetic_moments())
# snippet-li
fepo4_1li.append('Li')
fepo4_1li.positions[-1] = (0, 0, 0)  # not really needed
# snippet-gpaw
params = dict(
    mode=PW(500),
    nbands=-40,
    kpts={'size': (2, 4, 5),
          'gamma': True},
    spinpol=True,
    xc='BEEF-vdW',
    occupations=FermiDirac(width=0.1),
    convergence={'eigenstates': 1.0e-4,
                 'energy': 2.0e-4,
                 'density': 1.0e-3},
    mixer=Mixer(0.1, 5, weight=100.0),
    setups={'Fe': ':d,4.3'})
calc = GPAW(**params)
fepo4_1li.calc = calc
epot_fepo4_1li_cell = fepo4_1li.get_potential_energy()
print('E_Pot=', epot_fepo4_1li_cell)
write('fepo4_1li_out.traj', fepo4_1li)
ens = BEEFEnsemble(calc)
dE = ens.get_ensemble_energies(2000)
with paropen('ensemble_fepo4_1li.dat', 'w') as fd:
    for i in range(len(dE)):
        print(dE[i], file=fd)
# snippet-results
li_metal = read('li_metal.traj')
fepo4 = read('fepo4_out.traj')
fepo4_1li = read('fepo4_1li_out.traj')
epot_li_metal = li_metal.get_potential_energy() / len(li_metal)
epot_fepo4 = ...
epot_fepo4_1li = ...
# snippet-results-end
epot_fepo4 = fepo4.get_potential_energy()
epot_fepo4_1li = fepo4_1li.get_potential_energy()
li_cost = epot_fepo4_1li - epot_fepo4 - epot_li_metal
assert abs(li_cost - 0.3333333333) < 1e-10
