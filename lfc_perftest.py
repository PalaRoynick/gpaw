from gpaw.core import UGArray, UGDesc
from gpaw.lfc import BasisFunctions
from gpaw.new.basis_functions import BasisFunctionsNew, BasisFunction, BasisFunctionsLauri
from gpaw.test.lfc.test_basis import (AtomSystem, BasisFunctionDesc,
                                      make_basis,
                                      make_legacy_basis_functions)

import numpy as np
from gpaw.gpu import cupy as cp, as_xp, as_np, cupy_is_fake

import time
import cProfile
import pstats
import io
import argparse


xp = cp

def make_test_system() -> AtomSystem:
    """"""
    n = 64
    a = 3.0  # grid size in Bohr units ("unit cell"). NOT the lattice spacing
    grid = UGDesc(cell=[a, a, a], size=(n, n, n))

    relpos_ac = np.asarray([(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]])

    # Generate random radial functions
    rng = np.random.default_rng(404)
    basis_functions_aj = []

    for relpos in relpos_ac:
        # FIXME: handle boundary conds if any radial part extends outside the grid.
        # Only the old LFC code handles this at the moment.
        # For now, test with small cutoffs like 0.2*a

        s = BasisFunctionDesc.make_random(0, 0.18 * a, rng, num_points=100)
        p = BasisFunctionDesc.make_random(1, 0.25 * a, rng, num_points=100)
        d = BasisFunctionDesc.make_random(2, 0.20 * a, rng, num_points=100)
        basis_functions_aj.append([s, p, d])

    return AtomSystem(grid, relpos_ac, basis_functions_aj)


def init_lfc(system: AtomSystem) -> BasisFunctionsLauri:
    basis = make_basis(system, xp=xp)
    basis.precalculate_on_grid()
    return basis


def init_lfc_legacy(system: AtomSystem, xp) -> BasisFunctions:
    return make_legacy_basis_functions(system, xp)


def compute_matrix_elements(vt_R: UGArray, basis_funcs: BasisFunctionsLauri, repeat: int = 10) -> None:
    for i in range(repeat):
        basis_funcs.calculate_potential_matrix(vt_R)


def compute_matrix_elements_legacy(vt_R: UGArray, basis_funcs: BasisFunctions, repeat: int = 10) -> None:
    for i in range(repeat):
        basis_funcs.calculate_potential_matrices(vt_R.data)


def perftest_lfc(old_code: bool = False) -> None:

    if old_code:
        print("Profiling old LFC")

    #from gpaw.new.timer import global_timer
    # remember to set envvar GPAW_TRACE = 1

    system = make_test_system()

    # test array for calculate_potential_matrix
    vt_R: UGArray = system.grid.zeros(xp=xp)
    rng = xp.random.default_rng(222)
    noise = rng.random(vt_R.data.shape)
    vt_R.data += noise

    profiler = cProfile.Profile()

    # Initialization
    start = time.perf_counter()
    profiler.enable()

    if not old_code:
        basis = init_lfc(system)
    else:
        basis = init_lfc_legacy(system, xp)

    profiler.disable()
    end = time.perf_counter()

    init_time = end - start
    print(f"Initialization: {init_time:.6f} seconds")

    # Compute matrix elements
    repeat = 10
    start = time.perf_counter()
    profiler.enable()

    if not old_code:
        compute_matrix_elements(vt_R, basis, repeat=repeat)
    else:
        compute_matrix_elements_legacy(vt_R, basis, repeat=repeat)

    profiler.disable()
    end = time.perf_counter()

    avg_time = (end - start) / repeat
    print(f"compute_matrix_elements: {avg_time:.6f} seconds")

    # Print cprofile stats
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)

    print(stream.getvalue())


def parse_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old",
        action="store_true",
        help="Run with old LFC code"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Don't use GPU"
    )
    return parser.parse_args()


if __name__ == "__main__":
    """"""
    args = parse_cmd_args()
    if args.cpu:
        xp = np

    perftest_lfc(old_code=args.old)
