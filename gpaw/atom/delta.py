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
        kpts=dict(density=6.0),
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
        strains=strains.tolist(),
        energies=energies,
        energy=e0,
        strain=strain0,
        d2eds2=d2eds2)
    if world.rank == 0:
        Path('result.json').write_text(json.dumps(data))


