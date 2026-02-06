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


def simple(vt_R: UGArray,
           basis_functions_a: list[list[tuple[int, float, Callable]]],
           pos_av: np.ndarray
           ) -> Matrix:
    """Simple, but slow, implementation of potential matrix-elements."""
    N = sum(2 * l + 1
            for basis_functions in basis_functions_a
            for l, _, _ in basis_functions)
    grid = vt_R.desc
    R_Rv = grid.xyz()
    M = 0
    phit_MR = grid.zeros(N)
    for R_v, basis_functions in zip(pos_av, basis_functions_a):
        # find distance from each grid point to the current atom,
        # and mask points that are further than `rc` from the atom.
        d_Rv = R_Rv - R_v
        d_R = (d_Rv**2).sum(axis=3)**0.5
        for l, rc, func in basis_functions:
            mask_R = d_R < rc
            d_r = d_R[mask_R]
            d_rv = d_Rv[mask_R]

            f_r = func(d_r)
            for m in range(2 * l + 1):
                phit_MR.data[M + m, mask_R] = Y(l**2 + m, *d_rv.T) * f_r
            M += 2 * l + 1

    vtpsit_MR = phit_MR.copy()
    vtpsit_MR.data *= vt_R.data
    return Matrix(N, N, data=phit_MR.integrate(vtpsit_MR))


def ccode(vt_R: UGArray,
          basis_functions_a: list[list[tuple[int, float, Callable]]],
          pos_av: np.ndarray
          ) -> Matrix:
    grid = vt_R.desc
    basis = BasisFunctions(
        grid._gd,
        [[Spline.from_data(l, rc, func(np.linspace(0, rc, 1000)))
          for l, rc, func in basis_functions]
         for basis_functions in basis_functions_a])

    basis.set_positions(np.linalg.solve(grid.cell_cv.T, pos_av.T).T)
    N = basis.Mmax
    V_MM = Matrix(N, N)
    basis.calculate_potential_matrix(vt_R.data, V_MM.data, 0)
    return V_MM


def func(r):
    """radial function (goes to 0.0 at r=1.0)."""
    return 1 - 3 * r**2 + 2 * r**3


@pytest.mark.skip(reason="Skipping old tester")
@pytest.mark.serial
def test_pot_mat_elements():
    """Calculate overlap between s and p function."""
    n = 24
    a = 3.0
    grid = UGDesc(cell=[a, a, a], size=(n, n, n), pbc=[False, False, False])
    vt_R = grid.zeros()
    vt_R.data += 0.5

    """radial cutoff. NB: simple() doesn't account for boundary conditions so
    it will give wrong results if a basis function extends outside the box.
    Happens here if rc > 1.0
    """
    rc = 0.8

    def spline_func(r):
        """Goes to 0.0 at r=rc"""
        x = r / rc
        return 1 - 3 * x**2 + 2 * x**3

    # basis_functions_a = [[(0, rc, func)], [(1, rc, func), (2, rc, func)]]
    basis_functions_a = [[(0, rc, spline_func)], [(1, rc, spline_func)]]

    # Positions in real units
    pos_av = np.array([[1, 1, 1], [2.0, 2, 1]])
    V1_MM = simple(vt_R, basis_functions_a, pos_av)
    V2_MM = ccode(vt_R, basis_functions_a, pos_av)
    V2_MM.tril2full()
    assert V1_MM.data == pytest.approx(V2_MM.data)


def pot_mat_simple_reference(
        vt_R: UGArray,
        basis_functions_a: list[list[Spline]],
        pos_av: np.ndarray) -> Matrix:
    """Like simple() above but with spline input. Input positions are in REAL
    units. FIXME: does not work if the basis funcs extend outside the grid.
    """

    MMax = 0
    for splines_j in basis_functions_a:
        for phi in splines_j:
            l = phi.get_angular_momentum_number()
            MMax += 2 * l + 1

    grid = vt_R.desc
    R_Rv = grid.xyz()

    # Evaluate basis funcs on the grid. For now this has to be done on CPU
    # because we don't have splines on the GPU
    phit_MR = grid.zeros(MMax, xp=np)
    M = 0

    for R_v, splines_j in zip(pos_av, basis_functions_a):
        # find distance from each grid point to the current atom,
        # and mask points that are further than `rc` from the atom.

        d_Rv = R_Rv - R_v
        d_R = (d_Rv**2).sum(axis=3)**0.5

        for phi in splines_j:
            mask_R = d_R < phi.get_cutoff()  # cutoff is in real units
            d_r = d_R[mask_R]
            d_rv = d_Rv[mask_R]

            f_r = phi.map(d_r)
            l = phi.get_angular_momentum_number()

            for m in range(2 * l + 1):
                phit_MR.data[M + m, mask_R] = Y(l**2 + m, *d_rv.T) * f_r
            M += 2 * l + 1

    # Move to GPU if the input matrix is there
    phit_MR = phit_MR.to_xp(vt_R.xp)

    # sandwich vt_R and integrate
    vtpsit_MR = phit_MR.copy()
    vtpsit_MR.data *= vt_R.data
    return Matrix(MMax, MMax, data=phit_MR.integrate(vtpsit_MR))


@pytest.mark.skipif(world.size > 1, reason="MPI is TODO")
@pytest.mark.parametrize("basis_dtype", [float])
@pytest.mark.parametrize("xp", xp_params_no_cpupy())
def test_new_basis_functions(basis_dtype, xp):
    """"""
    pytest.skip()

    # Grid
    n = 26
    a = 3.0  # grid size in Bohr units. Note confusing terminology
    grid = UGDesc(cell=[a, a, a], size=(n, n, n))

    # Atoms and their positions. Relative pos means all coords in range [0, 1]
    relpos_ac = [(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]]
    relpos_ac = np.asarray(relpos_ac)
    # Same in real units
    pos_av = (grid.cell_cv @ relpos_ac.T).T

    # Generate random splines for the atoms
    rng = np.random.default_rng(321)
    atom_splines = []
    for relpos in relpos_ac:

        """
        syntax: l, r_max, f_g
        r_max is the radial cutoff in real units.
        f_g is a R^+ -> R function that must go to 0 at r_max. Give its values
        as an array. interpolated f_g will be the radial part of basis funcs:
        phi_mu ~ Y * f_g
        """

        # FIXME: handle boundary conds if any radial part to neighbor cell.
        # Only the old LFC code handles this at the moment.
        # For now, test with small cutoffs like 0.2*a

        s = Spline.from_data(0, 0.18 * a, [rng.random(), rng.random(), 0.0])
        p = Spline.from_data(1, 0.25 * a, [rng.random(), rng.random(), 0.0])
        d = Spline.from_data(2, 0.2 * a, [rng.random(), rng.random(), 0.0])

        atom_splines.append([s, p, d])

    # Test array that we sandwich between basis funcs and integrate
    vt_R: UGArray = grid.zeros(xp=xp)
    noise = as_xp(rng.random(vt_R.data.shape), xp)
    vt_R.data += noise

    # Calculate matrix elements with old code for reference
    V_MM_ref = pot_mat_old_code_reference(vt_R, atom_splines, relpos_ac)
    V_MM_ref.tril2full()

    V_MM_ref_other = pot_mat_simple_reference(vt_R, atom_splines, pos_av)
    V_MM_ref_other.tril2full()

    xp.testing.assert_allclose(V_MM_ref.data, V_MM_ref_other.data,
                               err_msg="Mismatch in two simple methods??")

    ### New stuff
    # Create and init basis functions
    basis = BasisFunctionsNew(grid, atom_splines, dtype=basis_dtype, xp=xp)

    # Basis funcs can be pre-evaluated everywhere on the grid once atom
    # positions are given. This is initiated from set_positions()
    basis.set_positions(relpos_ac)
    # MPI distribution. NB: create_basis uses band_comm instead of world here
    comm = world
    myM = (basis.Mmax + comm.size - 1) // comm.size
    basis.set_matrix_distribution(
        min(comm.rank * myM, basis.Mmax),
        min((comm.rank + 1) * myM, basis.Mmax))

    V_MM = basis.calculate_potential_matrix(vt_R)
    xp.testing.assert_allclose(V_MM.data, V_MM_ref.data, atol=1e-10)


@pytest.mark.parametrize("xp", xp_params_no_cpupy())
def test_basis_function_spline(xp):
    """"""
    l, cutoff = 1, 2.0
    f_r = xp.asarray([0.0, -0.25, 1.75, 0.35, 0.0])
    phi = BasisFunction(l, cutoff, f_r)

    # Test that values at breakpoints match the input
    spline_data = phi.get_spline_data()
    xp.testing.assert_allclose(spline_data.values, f_r,
                               rtol=1e-12, atol=1e-12)

    # Compare with old Spline class
    rng = xp.random.default_rng(55)
    phi = BasisFunction.make_random(l, cutoff, rng)

    spline_data = phi.get_spline_data()
    spline_old = Spline.from_data(l, cutoff, as_np(spline_data.values))

    x = rng.random((16), dtype=np.float64)
    y = phi.spline(x)
    y_old = spline_old.map(as_np(x))
    xp.testing.assert_allclose(y, y_old, rtol=1e-12, atol=1e-12)

    # TODO! the new spline current returns NaN outside its range
    # # Test out-of-bounds evaluation (spline should give 0)
    # x = x + cutoff
    # y = phi.spline(x)
    # y_old = spline_old.map(as_np(x))
    # xp.testing.assert_allclose(y, y_old, atol=0)


@dataclass
class BasisFunctionDesc:
    """Helper dataclass that just contains the data needed to define a radial
    spline. This is useful for testing when using both new and old
    splines objects.
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
        funcs = [BasisFunction(phi.l, phi.radial_cutoff, as_xp(phi.f_r, xp=xp))
                 for phi in phi_j]
        phi_aj.append(funcs)
    return phi_aj


def make_legacy_phi(system: AtomSystem) -> list[list[Spline]]:
    """"""
    phi_aj = []
    for phi_j in system.basis_functions_aj:
        funcs = [ Spline.from_data(phi.l, phi.radial_cutoff, phi.f_r)
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


def pot_mat_old_code_reference(
        system: AtomSystem,
        vt_R: UGArray,
        use_purepython: bool = False) -> Matrix:
    """Uses old CPU LFC code so should be both reliable and reasonably fast.
    Result matrix will be on GPU if vt_R is a Cupy array, otherwise on CPU.
    """

    xp = vt_R.xp

    if use_purepython:
        phi_aj = make_legacy_phi(system)
        pos_av = (system.grid.cell_cv @ system.relpos_ac.T).T
        V_MM = pot_mat_simple_reference(vt_R, phi_aj, pos_av)
    else:
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
    v_MN_ref = pot_mat_old_code_reference(system, vt_R, use_purepython=False)
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
