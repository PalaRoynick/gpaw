"""Tools for testing PAW-potentials."""
import json
from pathlib import Path

import numpy as np
from gpaw import setup_paths
from gpaw.dft import DFT, PW, MonkhorstPack, Parameters
from gpaw.utilities.acwf import reference_structure, work


def eos(structure, symbol):
    """EOS helper function."""
    setup_paths[:1] = ['.']
    work(structure, symbol, backwards_compatible=False)


def ecuts(symbol: str,
          name: str = 'FCC',
          _bc=False) -> list[float]:
    """Plane-wave cutoff-energy convergence test."""
    setup_paths[:1] = ['.']
    energies = list(range(300, 1300, 100))
    results = []
    for name in ['FCC', 'Diamond']:
        atoms = reference_structure(symbol, name)
        dft = None
        for ecut in energies:
            if dft is None:
                dft = DFT(
                    atoms,
                    xc='PBE',
                    mode=PW(ecut),
                    txt=f'ecuts-{name}.txt',
                    kpts=MonkhorstPack(density=3.0),
                    experimental={'backwards_compatible': _bc})
                dft.converge()
            else:
                params = Parameters(**dict(dft.params.todict(), mode=PW(ecut)))
                dft = dft.new(atoms, params, log=dft.log)
                dft.converge()
            e = dft.calculate_energy() / len(atoms)
            results.append(e)
    if dft.comm.rank == 0:
        n = len(energies)
        Path('ecuts.json').write_text(
            '[' +
            ',\n '.join(f'[{ecut:6.1f}, {e1}, {e2 - e1}]'
                        for ecut, e1, e2 in zip(energies,
                                                results[:n],
                                                results[n:])) +
            ']\n')
    return results


def get_ecut(ecuts: np.ndarray,
             energies: np.ndarray,
             de: float) -> float:
    """Get ecut where energy has converged to within de.

    >>> get_ecuts([300, 400, 500], [0.3, 0.1, 0.0], 0.2)
    350.0
    """
    y = abs(energies - energies[-1])
    i = len(energies) - 2
    while y[i] < de:
        i -= 1
    return (ecuts[i] * (de - y[i + 1]) +
            ecuts[i + 1] * (y[i] - de)) / (y[i] - y[i + 1])


def convergence_check(dir: Path) -> None:
    """Write summary."""
    ecuts, efcc, de = np.array(
        json.loads((dir / 'ecuts.json').read_text())).T
    ecut1 = get_ecut(ecuts, efcc, 0.1)
    ecut2 = get_ecut(ecuts, efcc, 0.01)
    ecut3 = get_ecut(ecuts, efcc, 0.001)
    ecut1d = get_ecut(ecuts, de, 0.1)
    ecut2d = get_ecut(ecuts, de, 0.01)
    ecut3d = get_ecut(ecuts, de, 0.001)

    print(f'{dir.name:15}',
          f'({ecut1d:4.0f} {ecut2d:4.0f} {ecut3d:4.0f}) ',
          f'({ecut1:4.0f} {ecut2:4.0f} {ecut3:4.0f}) ')


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('path', nargs='+')
    args = parser.parse_args()
    for path in args.path:
        convergence_check(Path(path))
