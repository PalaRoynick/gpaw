# web-page: eq_pot.svg
import numpy as np
import matplotlib.pyplot as plt
# snippet-read
from ase.io import read
fepo4 = read('fepo4_out.traj')
lifepo4 = read('lifepo4_out.traj')
li_metal = read('li_metal.traj')
epot_fepo4_cell = fepo4.get_potential_energy()
epot_lifepo4_cell = lifepo4.get_potential_energy()
epot_li_metal_cell = li_metal.get_potential_energy()
print('epot_fepo4_cell =', epot_fepo4_cell)
print('epot_lifepo4_cell =', epot_lifepo4_cell)
print('epot_li_metal_cell =', epot_li_metal_cell)
# snippet-read-end
epot_fepo4 = epot_fepo4_cell / len(fepo4) * 6
epot_lifepo4 = epot_lifepo4_cell / len(lifepo4) * 7
epot_li_metal = epot_li_metal_cell / len(li_metal)
V_eq = epot_lifepo4 - epot_fepo4 - epot_li_metal
print(V_eq)
# snippet-beef
epot_fepo4_ens_cell = np.genfromtxt('ensemble_fepo4.dat', max_rows=2000)
epot_lifepo4_ens_cell = np.genfromtxt('ensemble_lifepo4.dat', max_rows=2000)
epot_li_metal_ens_cell = np.genfromtxt('ensemble_li_metal.dat', max_rows=2000)
print('number of functionals in ensemble=', len(epot_fepo4_ens_cell))
print('number of functionals in ensemble=', len(epot_lifepo4_ens_cell))
print('number of functionals in ensemble=', len(epot_li_metal_ens_cell))
# snippet-beef-end
epot_fepo4_ens = epot_fepo4_ens_cell / len(fepo4) * 6
epot_lifepo4_ens = epot_lifepo4_ens_cell / len(lifepo4) * 7
epot_li_metal_ens = epot_li_metal_ens_cell / len(li_metal)
V_eq_ens = epot_lifepo4_ens - epot_fepo4_ens - epot_li_metal_ens
error = np.std(V_eq_ens)
print(error)
assert abs(V_eq_ens.mean() - 3.6) < 0.1
assert abs(error - 0.6) < 0.1
plt.hist(V_eq_ens, 50)
plt.grid(True)
plt.savefig('eq_pot.svg')
