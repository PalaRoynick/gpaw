# web-page: CrI3.svg
from gpaw import GPAW
from gpaw.spinorbit import soc_eigenstates
import matplotlib.pyplot as plt
import numpy as np

calc_fm = GPAW('CrI3_fm.gpw')
thetas = np.linspace(0, 180, 13)

energies = np.zeros(13)
for n, theta in enumerate(thetas):
    soc = soc_eigenstates(calc_fm, theta=theta, phi=0)
    energies[n] += soc.calculate_band_energy() / 2

energies = energies - energies[0]

plt.figure()
plt.plot(thetas, energies * 1000, 'o-')
plt.xlabel(r'$\theta$', size=18)
plt.ylabel('E [meV]', size=18)
plt.show()
plt.savefig('CrI3.svg')
