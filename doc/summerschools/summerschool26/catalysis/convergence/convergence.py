# web-page: results.json
import json
from pathlib import Path

from ase import Atoms
from ase.build import hcp0001
from ase.constraints import FixAtoms
from ase.optimize import BFGSLineSearch
from gpaw import GPAW, PW
from gpaw.mpi import world

a = 2.72
c = 1.58 * a
vacuum = 4.0


def adsorb(height: float = 1.2,
           nlayers: int = 3,
           nkpts: int = 7,
           ecut: float = 400) -> tuple[float, float, float]:
    """Adsorb nitrogen in hcp-site on Ru(0001) surface.

    Do calculations for N/Ru(0001), Ru(0001) and a nitrogen atom
    if they have not already been done.

    height: float
        Height of N-atom above top Ru-layer.
    nlayers: int
        Number of Ru-layers.
    nkpts: int
        Use a (nkpts * nkpts) Monkhorst-Pack grid that includes the
        Gamma point.
    ecut: float
        Cutoff energy for plane waves.

    Returns height, energy of N/Ru(0001) and clean Ru(0001).
    """

    name = f'Ru{nlayers}-{nkpts}x{nkpts}-{ecut:.0f}'

    parameters = dict(
        mode=PW(ecut),
        poissonsolver={'dipolelayer': 'xy'},
        kpts={'size': (nkpts, nkpts, 1), 'gamma': True},
        xc='PBE')

    # N/Ru(0001):
    slab = hcp0001('Ru', a=a, c=c, size=(1, 1, nlayers))
    z = slab.positions[:, 2].max() + height
    x, y = [2 / 3, 2 / 3] @ slab.cell[:2, :2]
    slab.append('N')
    slab.positions[-1] = [x, y, z]
    slab.center(vacuum=vacuum, axis=2)  # 2: z-axis

    # Fix first nlayer atoms:
    slab.constraints = FixAtoms(indices=list(range(nlayers)))

    slab.calc = GPAW(txt=f'N{name}.txt',
                     **parameters)
    optimizer = BFGSLineSearch(slab, logfile=f'N{name}.opt')
    optimizer.run(fmax=0.01)
    height = slab.positions[-1, 2] - slab.positions[:-1, 2].max()
    enru = slab.get_potential_energy()

    # Clean surface (single point calculation):
    del slab[-1]  # remove nitrogen atom
    slab.calc = GPAW(txt=f'{name}.txt',
                     **parameters)
    eru = slab.get_potential_energy()

    return height, enru, eru


def atom_energy(ecut: float = 400) -> float:
    """Calculate energy of spin-polarized nitrogen atom."""
    molecule = Atoms('N', magmoms=[3])
    molecule.center(vacuum=4.0)
    parameters = dict(
        mode=PW(ecut),
        xc='PBE')
    name = f'N-{ecut:.0f}'
    molecule.calc = GPAW(txt=f'{name}.txt', **parameters)
    en = molecule.get_potential_energy()
    return en


def main():
    h = 1.2  # initial guess

    # N-atom energies:
    eatom = {}
    for ecut in range(350, 801, 50):
        en = atom_energy(ecut)
        eatom[ecut] = en

    # Collect (layers, k-points, ecut, height, adsorption energy)
    # tuple in a list:
    results: list[tuple[int, int, int, float, float]] = []

    # ecut convergence:
    for ecut in range(350, 801, 50):
        h, enru, eru = adsorb(h, 2, 7, ecut)
        results.append((2, 7, ecut, h, enru - eru - eatom[ecut]))

    # Number of layers convergence:
    for n in range(1, 10):
        if n == 2:
            continue  # already done
        h, enru, eru = adsorb(h, n, 7, 400)
        results.append((n, 7, 400, h, enru - eru - eatom[ecut]))

    # Number of k-points convergence:
    for k in range(4, 18):
        if k == 7:
            continue  # already done
        h, enru, eru = adsorb(h, 2, k, 400)
        results.append((2, k, 400, h, enru - eru - eatom[ecut]))

    if world.rank == 0:
        Path('results.json').write_text(
            json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
