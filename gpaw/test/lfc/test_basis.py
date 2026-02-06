from collections.abc import Callable
import numpy as np
import pytest
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING

from gpaw.core import UGArray, UGDesc
from gpaw.core.matrix import Matrix
from gpaw.lfc import BasisFunctions
from gpaw.new.basis_functions import BasisFunctionsNew, BasisFunction
from gpaw.sphere.spherical_harmonics import Y
from gpaw.spline import Spline
from gpaw.mpi import world
from gpaw.gpu import cupy as cp, as_xp, as_np, cupy_is_fake

if TYPE_CHECKING:
    from gpaw.mpi import MPIComm
    world = cast(MPIComm, world)


def xp_params_no_cpupy():
    """Use as @pytest.mark.parametrize("xp", xp_params_no_cpupy())
    to run both xp=np and xp=cp cases, but skipping fake Cupy.
    """
    return [
        pytest.param(np, id="numpy"),
        pytest.param(
            cp,
            id="cupy",
            marks=[
                pytest.mark.gpu,
                pytest.mark.skipif(cupy_is_fake, reason="Fake Cupy"),
            ],
        ),
    ]


@dataclass
class BasisFunctionDesc:
    """Helper dataclass that just contains the data needed to define a radial
    spline. This is useful for tests that need both new and old splines.
    """
    l: int
    radial_cutoff: float
    f_r: np.ndarray

    @staticmethod
    def make_random(l: int,
                    cutoff: float,
                    rng: np.random.Generator,
                    num_points: int = 100) -> "BasisFunctionDesc":
        """"""
        values = rng.random(size=num_points)
        # last value must be 0
        values[-1] = 0.0
        return BasisFunctionDesc(l, cutoff, values)


@dataclass
class AtomSystem:
    grid: UGDesc
    relpos_ac: np.ndarray
    basis_functions_aj: list[list[BasisFunctionDesc]]

    def num_atoms(self) -> int:
        """"""
        return len(self.relpos_ac)


def make_test_system() -> AtomSystem:
    """"""
    n = 24
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


@pytest.fixture(scope="module")
def fixt_atom_system() -> AtomSystem:
    """"""
    return make_test_system()


def make_phi(system: AtomSystem, xp=np) -> list[list[BasisFunction]]:
    """"""
    phi_aj = []
    for phi_j in system.basis_functions_aj:
        funcs = [BasisFunction(phi.l,
                               phi.radial_cutoff,
                               as_xp(phi.f_r, xp=xp))
                 for phi in phi_j]
        phi_aj.append(funcs)
    return phi_aj


def make_legacy_phi(system: AtomSystem) -> list[list[Spline]]:
    """"""
    phi_aj = []
    for phi_j in system.basis_functions_aj:
        funcs = [ Spline.from_data(phi.l,
                                   phi.radial_cutoff,
                                   phi.f_r)
                     for phi in phi_j]
        phi_aj.append(funcs)
    return phi_aj


def make_legacy_basis_functions(system: AtomSystem, xp) -> BasisFunctions:
    """"""
    phi_aj = make_legacy_phi(system)
    basis = BasisFunctions(
        system.grid._gd,
        phi_aj,
        xp=xp)
    basis.set_positions(system.relpos_ac)
    return basis


def calculate_potential_matrix_old_code(
        system: AtomSystem,
        vt_R: UGArray) -> Matrix:
    """Uses old CPU LFC code so should be both reliable and reasonably fast.
    Result matrix will be on GPU if vt_R is a Cupy array, otherwise on CPU.
    """

    xp = vt_R.xp
    basis = make_legacy_basis_functions(system, xp=xp)

    N = cast(int, basis.Mmax)
    # this still only works with numpy
    V_MM = Matrix(N, N, xp=np)
    basis.calculate_potential_matrix(as_np(vt_R.data), V_MM.data, 0)
    V_MM = V_MM.to_xp(xp=xp)

    # other one of these does the full matrix instead of lower-half
    V_MM.data = xp.tril(V_MM.data)
    return V_MM


def make_basis(system: AtomSystem, xp=np) -> BasisFunctionsNew:
    """"""
    phi_aj = make_phi(system, xp=xp)
    basis = BasisFunctionsNew(system.grid, phi_aj)
    basis.set_positions(system.relpos_ac)
    return basis


@pytest.mark.parametrize("xp", xp_params_no_cpupy())
def test_basis_creation(fixt_atom_system: AtomSystem, xp):
    """"""
    system = fixt_atom_system
    basis = make_basis(system, xp=xp)

    assert basis.num_atoms == len(system.relpos_ac)

    # TODO check index range if using multiple MPI ranks?
    assert basis.mu_range.start == 0

    # check number of basis functions (indices {a,l,m})
    num_mu = 0
    for phi_j in system.basis_functions_aj:
        for phi in phi_j:
            num_mu += 2 * phi.l + 1

    assert basis.num_basis_functions() == num_mu

    # test mu index ordering
    mu = 0
    for atom in basis.atoms:
        mu_range = basis.mu_range_a[atom.index]
        mu_start, mu_end = mu_range.start, mu_range.stop

        assert mu_start == mu
        assert mu_end - mu_start == atom.num_basis_functions
        mu += atom.num_basis_functions


@pytest.mark.skipif(world.size > 1, reason="MPI is TODO")
@pytest.mark.parametrize("xp", xp_params_no_cpupy())
def test_potential_matrix(fixt_atom_system: AtomSystem, xp):
    """"""

    system = fixt_atom_system
    basis = make_basis(system, xp=xp)

    basis.precalculate_on_grid()

    # Test array that we sandwich between basis funcs and integrate
    vt_R: UGArray = system.grid.zeros(xp=xp)

    rng = xp.random.default_rng(222)
    noise = rng.random(vt_R.data.shape)
    vt_R.data += noise

    v_MN = basis.calculate_potential_matrix(vt_R.data)

    # Repeat with old code and compare
    v_MN_ref = calculate_potential_matrix_old_code(system, vt_R)
    v_MN_ref = v_MN_ref.data

    xp.testing.assert_allclose(v_MN,
                               v_MN_ref,
                               rtol=1e-10,
                               atol=1e-12)


@pytest.mark.skipif(world.size > 1, reason="MPI is TODO")
@pytest.mark.parametrize("xp", xp_params_no_cpupy())
def test_add_to_density(fixt_atom_system: AtomSystem, xp):
    """"""
    system = fixt_atom_system
    basis = make_basis(system, xp=xp)

    basis.precalculate_on_grid()

    num_spins = 2
    # Test density
    rng = xp.random.default_rng(432)
    sG_shape = (num_spins, *system.grid.size)
    nt_sG = rng.random(sG_shape)

    nt_sG_copy = nt_sG.copy()

    # Make test "array" f_asi. This is actually a dict of arrays, because
    # the number of basis functions (range of i) can vary per atom.
    f_asi = {}
    for atom in basis.atoms:
        shape_si = (num_spins, atom.num_basis_functions)
        f_asi[atom.index] = rng.random(shape_si)

    basis.add_to_density(nt_sG, f_asi)

    # Repeat with old code.
    # this still only works with numpy (density is converted internally)
    f_asi = {a: as_np(f_asi[a]) for a in f_asi.keys()}
    basis = make_legacy_basis_functions(system, xp=np)
    basis.add_to_density(nt_sG_copy, f_asi)

    xp.testing.assert_allclose(nt_sG, nt_sG_copy, rtol=1e-10, atol=1e-12)
