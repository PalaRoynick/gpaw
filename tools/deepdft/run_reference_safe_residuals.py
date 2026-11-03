"""Run reference generation with a CuPy workaround for oversized residual grids.

Use the installed CUDA environment's Python from outside the source checkout.
Arguments are the same as for ``gpaw.density_learning.cli reference``.
This process-local workaround does not modify the installed package or native
extension and does not fix native LFC memory leaks.
"""

from functools import wraps
import json
from math import prod
import sys


MAX_NATIVE_GRID_POINTS = 16 * 65535


def install_workaround():
    from gpaw.new.pwfd import eigensolver
    from gpaw.purepython import calculate_residuals_gpu

    original = eigensolver.calculate_residuals_gpu

    @wraps(original)
    def safe_residuals(residual, eigenvalues, wavefunctions):
        if prod(residual.shape[1:]) > MAX_NATIVE_GRID_POINTS:
            # On CuPy inputs this remains on GPU. The existing implementation
            # works one band at a time, avoiding a full-array temporary.
            return calculate_residuals_gpu(residual, eigenvalues, wavefunctions)
        return original(residual, eigenvalues, wavefunctions)

    eigensolver.calculate_residuals_gpu = safe_residuals
    return original


def main(argv=None):
    from gpaw.density_learning.cli import main as cli_main
    from gpaw.new.pwfd import eigensolver

    original = install_workaround()
    print(json.dumps({'event': 'residual_kernel_workaround',
                      'native_grid_point_limit': MAX_NATIVE_GRID_POINTS,
                      'large_grid_backend': 'cupy'}), flush=True)
    try:
        cli_main(['reference', *(sys.argv[1:] if argv is None else argv)])
    finally:
        eigensolver.calculate_residuals_gpu = original


if __name__ == '__main__':
    main()
