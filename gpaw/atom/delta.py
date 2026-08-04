import json
from pathlib import Path

import numpy as np
from ase.collections import dcdft
from gpaw import GPAW, setup_paths
from gpaw.dft import Parameters
from gpaw.mpi import world
from gpaw.utilities.acwf import fit


def delta():
    setup_paths[:0] = ['.']
    name = Path.cwd().name
    symbol = name.split('.')[0]
    atoms = dcdft[symbol]
    params = dict(
        xc='PBE',
        occupations=dict(name='fermi-dirac', width=0.0612),
        kpts=dict(density=6.0, gamma=True),
        convergence=dict(density=1e-6),
        mode=dict(name='pw', ecut=1000))
    builder = Parameters(**params).dft_component_builder(atoms, log=None)
    params['kpts'] = builder.ibz.bz.size_c
    params['gpts'] = builder.grid.size
    atoms.calc = GPAW(txt=f'{name}.txt', **params)
    cell_cv = atoms.get_cell()
    energies = []
    maxstrain = 0.06
    npoints = 7
    strains = np.linspace(-maxstrain, maxstrain, npoints)
    for i, strain in enumerate(strains):
        atoms.set_cell((1.0 + strain) * cell_cv, scale_atoms=True)
        e = atoms.get_potential_energy()
        energies.append(e)

    strain0, e0, d2eds2 = fit(strains, energies)
    v0 = abs(np.linalg.det(cell_cv))
    volume = v0 * (1 + strain0)**3
    data = dict(
        volume=volume,
        v0=v0,
        strains=strains.tolist(),
        energies=energies,
        energy=e0,
        strain=strain0,
        d2eds2=d2eds2)
    if world.rank == 0:
        Path('result.json').write_text(json.dumps(data, indent=1))


def analyse(paths):
    D = []
    for path in paths:
        try:
            data = json.loads((path / 'result.json').read_text())
        except FileNotFoundError:
            continue
        D.append((path.name.split('.')[0], path, abs(data['strain'])))
    for s, p, e in sorted(D):
        print(s, p, e)
    return D


if __name__ == '__main__':
    import sys
    from ase.data import atomic_numbers
    if 0:
        analyse((Path(x) for x in sys.argv[1:]))
    else:
        d1 = analyse(Path('old').glob('*/'))
        d2 = analyse(Path('4').glob('*/'))
        x1, y1 = np.array(sorted([(atomic_numbers[s], e)
                                  for s, p, e in d1])).T
        x2, y2 = np.array(sorted([(atomic_numbers[s], e)
                                  for s, p, e in d2])).T
        import matplotlib.pyplot as plt
        plt.plot(x1, np.log10(y1), '*-')
        plt.plot(x2, np.log10(y2), 'o-')
        plt.show()
