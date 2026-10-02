import pytest

from gpaw.mpi import world
from gpaw import GPAW


@pytest.mark.parametrize('name, tol',
                         [('bcc_li_pw', 1e-3),
                          ('bcc_li_fd', 1e-3),
                          ('bcc_li_lcao', 2e-3)])
def test_ae_k(gpw_files, name, tol):
    """Test normalization of non gamma-point wave functions."""
    if world.size > 1:
        return
    calc = GPAW(gpw_files[name])
    ae = calc.dft.ibzwfs.get_all_electron_wave_function(
        1, kpt=1, grid_spacing=0.1)
    assert ae.norm2() == pytest.approx(1.0, abs=tol)
