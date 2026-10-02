# web-page: pdos_w_and_wo_soc.svg
import matplotlib.pyplot as plt
import numpy as np
from ase.build import bulk
from gpaw import GPAW, PW

atoms = bulk('Au', 'fcc')

k = 10
calc = GPAW(
    mode=PW(600),
    xc='PBE',
    nbands=20,
    convergence={'bands': 'CBM+2.5'},
    kpts={'size': [k, k, k], 'gamma': True},
    txt='fcc_Au_gs.txt')
atoms.calc = calc
atoms.get_potential_energy()

calc.write('fcc_Au_gs.gpw')
calc = GPAW('fcc_Au_gs.gpw')

emin = -10
emax = 2.5
energies = np.linspace(emin, emax, 200)
dos_kwargs = {'energies': energies, 'width': 0.0}

fig, axes = plt.subplots(
    figsize=(4, 4),
    ncols=2,
    layout='constrained',
    sharex=True,
    sharey=True)
fig.suptitle('FCC Au (P)DOS')
color = 'teal'

for ax, soc in zip(axes, [False, True]):
    dos = calc.dos(soc=soc)
    pdos_l = [dos.raw_pdos(energies, a=0, l=l, width=0.0)
              for l in range(3)]
    tdos = dos.raw_dos(energies, width=0.0)

    ax.axhline(0, color='grey', linewidth=0.8)
    ax.fill_betweenx(energies, tdos, label='Total', color='lightgrey')
    for label, pdos, ls in zip('spd', pdos_l, ['--', '-', '-.']):
        ax.plot(pdos, energies, label=label, color=color, linestyle=ls)
    ax.set_title('With SOC' if soc else 'Without SOC')
    ax.set_xlabel('DOS [1/eV]')
    ax.legend(loc='lower right')
    color = 'purple'

axes[0].set_ylabel('$E - E_{Fermi}$ (eV)')
# updates both because they are shared
axes[0].set_ylim((emin, emax))
axes[0].set_xlim((0, None))
fig.savefig('pdos_w_and_wo_soc.svg')
# plt.show()
