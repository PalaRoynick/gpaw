"""check if fhc can be disabled for libxc >= 7.x.x (needed for mggas)."""
import pytest
from ase.build import molecule

import gpaw.cgpaw as cgpaw
from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell

vacuum = 4.0
h = 0.2


@pytest.mark.mgga
@pytest.mark.libxc
def test_mgga_lxc_fhc():
    libxc_version = getattr(cgpaw, 'libxc_version', '2.x.y')
    if int(libxc_version.split('.')[0]) < 7:
        pytest.skip('libxc too old')
    cluster = molecule('CO')
    adjust_cell(cluster, border=vacuum, h=h)
    calc = GPAW(xc='PBE',
                mode='fd',
                h=h,
                eigensolver={'niter': 5, 'rr_modulo': 2},
                maxiter=24,
                )
    cluster.calc = calc
    cluster.get_potential_energy()
    calc.dft.change(xc='MGGA_X_TPSS+MGGA_C_TPSS')
    calc.dft.converge()
