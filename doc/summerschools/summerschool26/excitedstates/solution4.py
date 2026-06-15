# snippet-plot-rpa-start
# Plot the absorption spectrum:
import numpy as np
import matplotlib.pyplot as plt
plt.figure()

# Of course you need to input the name you gave the files earlier own yourself and only plot the components you calculated.
# Here is only showed hot to plot the x-component.

absox = np.loadtxt('Si_rpa_x.csv', delimiter=',')  # student: absox = np.loadtxt('???_rpa_x.csv', delimiter=',')
plt.plot(absox[:, 0], absox[:, 4], label='RPA_x Si 12x12x4', lw=2, color='b')

plt.xlabel(r'$\hbar\omega\;[eV]$', size=20)
plt.ylabel(r'$Im(\epsilon)$', size=20)
plt.xticks(size=16)
plt.yticks(size=16)
plt.tight_layout()
plt.axis([0.0, 10.0, None, None])
plt.legend()

# plt.show()
plt.savefig('rpa_Si.png')
# snippet-plot-rpa-end

# snippet-bse-groundstate-start
from ase.build import bulk
from gpaw import GPAW
from gpaw import PW
from gpaw.occupations import FermiDirac

Si = bulk('Si', 'diamond', a=5.4)  # student:
atoms = Si  # student: atoms = ???

calc = GPAW(mode=PW(400),  # student: mode=PW(???),
            xc='PBE',
            occupations=FermiDirac(width=0.001),
            parallel={'domain': 1, 'band': 1},
            kpts={'size': (8, 8, 8), 'gamma': True},  # student: kpts={'size': (?, ?, ?), 'gamma': True},
            txt='gs_Si.txt')  # student: txt='gs_???.txt'

atoms.calc = calc
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian(nbands=100)  # student: nbands=???
calc.write('gs_Si.gpw', mode='all')  # student: calc.write('gs_???.gpw', mode='all')
# snippet-bse-groundstate-end

# snippet-bse-spectrum-start
import numpy as np
from gpaw.response.bse import BSE
from gpaw.response.df import DielectricFunction

ecut = 50  # student: ecut = ???
eta = 0.2

df = DielectricFunction('gs_Si.gpw',  # student: 'gs_???.gpw',
                        ecut=ecut,
                        frequencies=np.linspace(0, 10, 1001),
                        nbands=8,  # student: nbands=??,
                        intraband=False,
                        hilbert=False,
                        eta=eta,
                        txt='rpa_Si.txt')  # student: txt='rpa_???.txt'

df.get_dielectric_function(filename='eps_rpa_Si.csv')  # student: filename='eps_rpa_???.csv'

bse = BSE('gs_Si.gpw',  # student: 'gs_???.gpw',
          ecut=ecut,
          valence_bands=range(0, 4),  # student: valence_bands=range(?, ?),
          conduction_bands=range(4, 8),  # student: conduction_bands=range(?, ?),
          nbands=50,  # student: nbands=???,
          mode='BSE',
          integrate_gamma='sphere',
          txt='bse_Si.txt')  # student: txt='bse_???.txt'

bse.get_dielectric_function(filename='eps_bse_Si.csv',  # student: filename='eps_bse_???.csv',
                            eta=eta,
                            write_eig='bse_Si_eig.dat',  # student: write_eig='bse_???_eig.dat',
                            w_w=np.linspace(0.0, 10.0, 10001))
# snippet-bse-spectrum-end

# snippet-plot-bse-start
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

a = np.loadtxt('eps_rpa_Si.csv', delimiter=',')  # student: a = np.loadtxt('eps_rpa_???.csv', delimiter=',')
plt.plot(a[:, 0], a[:, 4], label='RPA', lw=2)

a = np.loadtxt('eps_bse_Si.csv', delimiter=',')  # student: a = np.loadtxt('eps_bse_???.csv', delimiter=',')
plt.plot(a[:, 0], a[:, 2], label='BSE', lw=2)

plt.xlabel(r'$\hbar\omega\;[eV]$', size=24)
plt.ylabel(r'$\epsilon_2$', size=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.axis([2.0, 6.0, None, None])
plt.legend()

# plt.show()
plt.savefig('bse_Si.png')  # student: plt.savefig('bse_???.png')
# snippet-plot-bse-end
