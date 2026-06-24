# snippet-plot-rpa-start
# Plot the absorption spectrum:
import numpy as np
import matplotlib.pyplot as plt
plt.figure()

# Of course you need to input the name you gave the files earlier own
# yourself and only plot the components you calculated.
# Here is only showed hot to plot the x-component.

absox = np.loadtxt('Si_rpa_x.csv', delimiter=',')
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

from ase.build import bulk
atoms = bulk('Si', 'diamond', a=5.4)
pw_cut_off = 400
kpts_grid = (8, 8, 8)
gamma = True
nbands = 100

# snippet-bse-groundstate-start
from gpaw import GPAW
from gpaw import PW
from gpaw.occupations import FermiDirac

calc = GPAW(mode=PW(pw_cut_off),
            xc='PBE',
            occupations=FermiDirac(width=0.001),
            parallel={'domain': 1, 'band': 1},
            kpts={'size': kpts_grid, 'gamma': gamma},
            txt='gs_atoms.txt')

atoms.calc = calc
atoms.get_potential_energy()

calc.diagonalize_full_hamiltonian(nbands=nbands)
calc.write('gs_atoms.gpw', mode='all')
# snippet-bse-groundstate-end


ecut = 50
nbands = 8
valence_bands = range(0, 4)
conduction_bands = range(4, 8)
nbands = 50

# snippet-bse-spectrum-start
import numpy as np
from gpaw.response.bse import BSE
from gpaw.response.df import DielectricFunction

eta = 0.2

df = DielectricFunction('gs_atoms.gpw',
                        ecut=ecut,
                        frequencies=np.linspace(0, 10, 1001),
                        nbands=nbands,
                        intraband=False,
                        hilbert=False,
                        eta=eta,
                        txt='rpa_atoms.txt')

df.get_dielectric_function(filename='eps_rpa_atoms.csv')

bse = BSE('gs_atoms.gpw',
          ecut=ecut,
          valence_bands=valence_bands,
          conduction_bands=conduction_bands,
          nbands=nbands,
          mode='BSE',
          integrate_gamma='sphere',
          txt='bse_atoms.txt')

bse.get_dielectric_function(filename='eps_bse_atoms.csv',
                            eta=eta,
                            write_eig='bse_atoms_eig.dat',
                            w_w=np.linspace(0.0, 10.0, 10001))
# snippet-bse-spectrum-end

# snippet-plot-bse-start
import matplotlib.pyplot as plt
import numpy as np

plt.figure()

a = np.loadtxt('eps_rpa_atoms.csv', delimiter=',')
plt.plot(a[:, 0], a[:, 4], label='RPA', lw=2)

a = np.loadtxt('eps_bse_atoms.csv', delimiter=',')
plt.plot(a[:, 0], a[:, 2], label='BSE', lw=2)

plt.xlabel(r'$\hbar\omega\;[eV]$', size=24)
plt.ylabel(r'$\epsilon_2$', size=24)
plt.xticks(size=20)
plt.yticks(size=20)
plt.tight_layout()
plt.axis([2.0, 6.0, None, None])
plt.legend()

# plt.show()
plt.savefig('bse_atoms.png')
# snippet-plot-bse-end
