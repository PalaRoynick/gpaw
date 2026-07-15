"""Tools for testing PAW-potentials."""
import json
from pathlib import Path

import numpy as np
from gpaw import setup_paths
from gpaw.dft import DFT, PW, MonkhorstPack, Parameters
from gpaw.utilities.acwf import reference_structure, work, acwf_structures
from gpaw.mpi import world


def eos(structure, symbol=None):
    """EOS helper function."""
    setup_paths[:1] = ['.']
    symbol = symbol or Path.cwd().name.split('.')[0]
    work(structure, symbol, backwards_compatible=False)


def ecuts(symbol: str = None,
          _bc=False) -> list[float]:
    """Plane-wave cutoff-energy convergence test."""
    setup_paths[:1] = ['.']
    symbol = symbol or Path.cwd().name.split('.')[0]
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
    if world.rank == 0:
        n = len(energies)
        Path('ecuts.json').write_text(
            '[' +
            ',\n '.join(f'[{ecut:6.1f}, {e1}, {e2 - e1}]'
                        for ecut, e1, e2 in zip(energies,
                                                results[:n],
                                                results[n:])) +
            ']\n')
    return results


def workflow():
    from myqueue.workflow import run
    for name in acwf_structures:
        if not Path(f'{name}.json').is_file():
            run(function=eos, args=[name], name=name, cores=24)
    if not Path('ecuts.json').is_file():
        run(function=ecuts, cores=24)


def get_ecut(ecuts: np.ndarray,
             energies: np.ndarray,
             de: float) -> float:
    """Get ecut where energy has converged to within de.

    >>> get_ecut(np.array([300, 400, 500]),
    ...          np.array([0.5, 0.25, 0.0]), 0.375)
    350.0
    """
    y = abs(energies - energies[-1])
    i = len(energies) - 2
    while y[i] < de:
        if i == 0:
            return ecuts[0]
        i -= 1
    return float(
        (ecuts[i] * (de - y[i + 1]) +
         ecuts[i + 1] * (y[i] - de)) / (y[i] - y[i + 1]))


def convergence_check(dir: Path,
                      plot=False,
                      ) -> tuple[np.ndarray,
                                 np.ndarray,
                                 np.ndarray,
                                 list[float],
                                 list[float]]:
    """Write summary."""
    ecuts, efcc, de = np.array(
        json.loads((dir / 'ecuts.json').read_text())).T
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(ecuts[:-1], np.log10(abs(efcc[:-1] - efcc[-1])), label=dir)
    ecuts_tot = [get_ecut(ecuts, efcc, x) for x in [0.1, 0.01, 0.001]]
    ecuts_diff = [get_ecut(ecuts, de, x) for x in [0.1, 0.01, 0.001]]

    return ecuts, efcc, de, ecuts_tot, ecuts_diff


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('path', nargs='+')
    parser.add_argument('-p', '--plot', action='store_true')
    args = parser.parse_args()
    results = []
    for path in args.path:
        path = Path(path)
        try:
            _, _, _, ecuts_tot, ecuts_diff = convergence_check(
                Path(path), args.plot)
        except FileNotFoundError:
            print(path.name)
        ecut1, ecut2, ecut3 = ecuts_tot
        ecut1d, ecut2d, ecut3d = ecuts_diff
        print(f'{path.name:25}',
              f'({ecut1d:4.0f} {ecut2d:4.0f} {ecut3d:4.0f}) ',
              f'({ecut1:4.0f} {ecut2:4.0f} {ecut3:4.0f}) ')
        results.append((ecut3, ecut3d, path))
    for e, de, p in sorted(results):
        print(f'{e:7.1f} {de:7.1f} {p}')
    if args.plot:
        import matplotlib.pyplot as plt
        plt.legend()
        plt.show()
