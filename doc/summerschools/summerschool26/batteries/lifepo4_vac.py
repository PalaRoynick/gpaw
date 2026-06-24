from ase.dft.bee import BEEFEnsemble
from ase.io import read, write
from ase.parallel import paropen
from gpaw import GPAW, PW, FermiDirac

lifepo4_vac = read('lifepo4.traj')
del lifepo4_vac[-4]
params = dict(
    mode=PW(500),
    nbands=-40,
    kpts={'size': (2, 4, 5),
          'gamma': True},
    xc='BEEF-vdW',
    occupations=FermiDirac(width=0.1),
    convergence={'eigenstates': 1.0e-4,
                 'energy': 2.0e-4,
                 'density': 1.0e-3},
    setups={'Fe': ':d,4.3'})
calc = GPAW(**params)
lifepo4_vac.calc = calc
epot_lifepo4_vac = lifepo4_vac.get_potential_energy()
print('E_Pot =', epot_lifepo4_vac)
write('lifepo4_vac_out.traj', lifepo4_vac)

ens = BEEFEnsemble(calc)
dE = ens.get_ensemble_energies(2000)
with paropen('ensemble_lifepo4_vac.dat', 'a') as fd:
    for de in dE:
        print(de, file=fd)

# snippet-results
# Loading in files from exercise day 3.
li_metal = read('li_metal.traj')   # you should have already read this in above
lifepo4 = read('lifepo4_out.traj')
epot_li_metal = li_metal.get_potential_energy() / len(li_metal)
epot_lifepo4 = lifepo4.get_potential_energy()
vac_cost = ...
print(vac_cost)
# snippet-results-end
vac_cost = epot_lifepo4_vac - epot_lifepo4 + epot_li_metal
print(vac_cost)
