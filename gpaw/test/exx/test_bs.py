import numpy as np
import pytest
from ase.units import Ha

from gpaw.band_structure import band_structure
from gpaw.dft import GPAW
from gpaw.mpi import world


def par(size):
    kb = []
    for k in range(1, 3):
        if size % k != 0:
            continue
        for b in range(1, size // k + 1):
            if (size // k) % b == 0:
                kb.append((k, b))
    return kb


@pytest.mark.parametrize('xc', ['LDA', 'HSE06'])
@pytest.mark.parametrize(
    'kb',
    [pytest.param((k, b), id=f'k{k}b{b}') for k, b in par(world.size)])
def test_all(xc, kb, gpw_files):
    k, b = kb
    calc = GPAW(gpw_files[f'li2_pw_{xc.lower()}'],
                parallel={'kpt': k, 'band': b})
    eig_in = calc.eigenvalues()[0]
    kpts = np.zeros((4, 3))
    kpts[:, 2] = np.linspace(-0.25, 0.5, 4)
    band_wfs = band_structure(calc.dft, kpts, convergence={'bands': 2})
    for wfs in band_wfs:
        i = int(round(abs(wfs.kpt_c[2]) * 4))
        # The hybrid eigenvalues of this system are pathologically sensitive
        # to the input: changing a single Gaunt coefficient by one ulp moves
        # them by several times 1e-6 eV, which is far more than the default
        # relative tolerance of pytest.approx allows for.
        assert wfs.eig_n[:2] * Ha == pytest.approx(eig_in[i, :2], abs=1e-4)


if __name__ == '__main__':
    import sys
    test_all([int(x) for x in sys.argv[1:]])
    # print(par(2))
