from gpaw.cli.main import main
import pytest


@pytest.mark.ci
@pytest.mark.serial
@pytest.mark.parametrize('P', [1, 2])
def test_python_warnings(P):
    """Convert warning from numpy to error"""
    with pytest.raises(RuntimeError):
        main(
            ['-P',
             str(P),
             'python',
             '-We',
             '-c',
             'import numpy as np; 1 / np.zeros(2)'])
