from gpaw.cli.main import main
import pytest


@pytest.mark.ci
@pytest.mark.serial
def test_python_warnings():
    """Convert warning from numpy to error"""
    with pytest.raises(RuntimeError):
        main(
            ['python',
             '-We',
             '-c',
             'import numpy as np; 1 / np.zeros(2)'])
