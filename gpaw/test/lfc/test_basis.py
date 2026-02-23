from collections.abc import Callable
import numpy as np
import pytest
from dataclasses import dataclass
from typing import cast, TYPE_CHECKING
import copy

from gpaw.core import UGArray, UGDesc
from gpaw.core.matrix import Matrix
from gpaw.lfc import BasisFunctions
from gpaw.new.basis_functions import (BasisFunctionDesc,
                                      BasisFunctionCollectionBase,
                                      LFCAtomDefinition, LFCAtomInstance,
                                      LFCSystem, AtomTypeId)
from gpaw.new.basis_functions_purepython \
    import BasisFunctionCollection_PurePython
from gpaw.sphere.spherical_harmonics import Y
from gpaw.spline import Spline
from gpaw.mpi import world, MPIComm, broadcast_exception
from gpaw.gpu import cupy as cp, as_xp, as_np, cupy_is_fake
from gpaw import GPAW_NO_C_EXTENSION


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

# def parametrize_purepython():
#     """Use as @pytest.mark.parametrize("purepython", parametrize_purepython())
#     to run both purepython and non-purepython versions, skipping the latter if
#     not available.
#     """
#     return [True,
#             pytest.param(
#                 False,
#                 marks=[pytest.mark.skipif(GPAW_NO_C_EXTENSION,
#                                           reason="No C extension")]
#             )
#         ]

def parametrize_purepython():
    return [True]

def make_random_phi(
        l: int,
        cutoff: float,
        rng: np.random.Generator,
        num_points: int = 100) -> BasisFunctionDesc:
    """"""
    values = rng.random(size=num_points)
    # last value must be 0
    values[-1] = 0.0
    return BasisFunctionDesc(l, cutoff, values)


def make_test_system() -> LFCSystem:
    """"""
    n = 24
    a = 3.0  # grid size in Bohr units ("unit cell"). NOT the lattice spacing

    # Grid does automatic domain decomp if running with MPI
    global world
    world = cast(MPIComm, world)
    grid = UGDesc(cell=[a, a, a], size=(n, n, n), comm=world)

    rng = np.random.default_rng(404)

    # Make system of 3 atoms, 2 of which are identical (same basis funcs).
    # Also give the other atom type a different number of basis funcs
    atom_defs = []
    for i in range(2):
        s = make_random_phi(0, 0.98 * a, rng, num_points=100)
        p = make_random_phi(1, 1.35 * a, rng, num_points=100)
        d = make_random_phi(2, 0.1 * a, rng, num_points=100)

        funcs = [s, p, d] if i == 0 else [s, d]
        atom_defs.append(LFCAtomDefinition(i, funcs))

    relpos_ac = np.asarray([(0.5, 0.5, 0.25 + 0.25 * i) for i in [0, 1, 2]])
    pos_av = relpos_ac @ grid.cell_cv

    atom_instances = []
    for a in range(3):
        atom_type = atom_defs[0] if a < 2 else atom_defs[1]
        atom_instances.append(LFCAtomInstance(atom_type, pos_av[a]))

    return LFCSystem(grid, atom_defs, atom_instances)


def make_test_system_small() -> LFCSystem:
    """"""
    n = 4
    a = 2.0
    global world
    world = cast(MPIComm, world)
    grid = UGDesc(cell=[a, a, a], size=(n, n, n), comm=world)

    rng = np.random.default_rng(404)

    atom_defs = []
    for i in range(2):
        s = make_random_phi(0, 0.9 * a, rng, num_points=50)
        p = make_random_phi(1, 0.75 * a, rng, num_points=50)
        d = make_random_phi(2, 2.20 * a, rng, num_points=50)

        #funcs = [s, p, d] if i == 0 else [s, d]
        funcs = [p] if i == 0 else [s]
        atom_defs.append(LFCAtomDefinition(i, funcs))

    relpos_ac = np.asarray([(0.5, 0.5, 0.50 + 0.25 * i) for i in [0, 1]])
    pos_av = relpos_ac @ grid.cell_cv

    atom_instances = []
    for a in range(2):
        atom_type = atom_defs[a]
        atom_instances.append(LFCAtomInstance(atom_type, pos_av[a]))

    return LFCSystem(grid, atom_defs, atom_instances)


@pytest.fixture(scope="module")
def fixt_lfc_system() -> LFCSystem:
    """"""
    return make_test_system()
    #return make_test_system_small()


def make_legacy_basis_functions(system: LFCSystem, xp) -> BasisFunctions:
    """"""
    # Build legacy basis functions (Spline objects)
    phi_t_j: dict[AtomTypeId, list[Spline]] = {}
    for atom_type in system.atom_types:
        funcs = [ Spline.from_data(phi.l,
                                   phi.cutoff,
                                   phi.f_r)
                    for phi in atom_type.basis_functions
                ]
        phi_t_j[atom_type.id] = funcs

    phi_aj = []
    for atom in system.atoms:
        phi_aj.append(phi_t_j.get(atom.type.id))

    basis = BasisFunctions(
        system.grid._gd,
        phi_aj,
        xp=xp)

    pos_av = np.empty((len(system.atoms), 3))
    for a, atom in enumerate(system.atoms):
        pos_av[a] = atom.position

    # annoying, need to transform back to relative position
    relpos_ac = pos_av @ np.linalg.inv(system.grid.cell_cv)

    basis.set_positions(relpos_ac)
    return basis


def make_basis(system: LFCSystem,
               xp,
               purepython: bool) -> BasisFunctionCollectionBase:
    """"""
    if purepython:
        basis = BasisFunctionCollection_PurePython(
            system,
            use_gpu = (xp is not np)
            )
    else:
        assert "TODO"

    return basis

@pytest.mark.parametrize("xp", xp_params_no_cpupy())
@pytest.mark.parametrize("purepython", parametrize_purepython())
def test_basis_creation(fixt_lfc_system: LFCSystem, xp, purepython: bool):
    """"""
    system = fixt_lfc_system
    basis = make_basis(system, xp, purepython)

    assert basis.num_atoms == len(system.atoms)

    # TODO check index range if using multiple MPI ranks?
    assert basis.mu_range.start == 0

    # check number of basis functions (indices {a,l,m})
    num_mu = 0
    for atom in system.atoms:
        for phi in atom.type.basis_functions:
            num_mu += 2 * phi.l + 1

    assert basis.num_basis_functions() == num_mu

    # Both of these should be the full list of basis function instances:
    all_phi = basis.get_phi_instances()
    maybe_all_phi = basis.get_phi_for_mu(0, basis.num_basis_functions())
    assert len(all_phi) == len(maybe_all_phi)
    # Compare with "is", not with "==" because these should be same objects
    for phi in all_phi:
        assert any(p is phi for p in maybe_all_phi)


@pytest.mark.parametrize("xp", xp_params_no_cpupy())
@pytest.mark.parametrize("purepython", parametrize_purepython())
@pytest.mark.parametrize("row_range", [None, range(1, 4), range(0, 100000)])
def test_potential_matrix(fixt_lfc_system: LFCSystem, xp, purepython: bool,
    row_range: range | None):
    """"""

    system = fixt_lfc_system
    basis = make_basis(system, xp, purepython)

    if row_range:
        assert row_range.step == 1
        basis.set_matrix_distribution(row_range.start, row_range.stop)

    # Test array that we sandwich between basis funcs and integrate
    vt_G: UGArray = system.grid.zeros(xp=xp)

    rng = xp.random.default_rng(222)
    noise = rng.random(vt_G.data.shape)
    vt_G.data += noise

    V_MN = basis.calculate_potential_matrix(vt_G.data)

    # Repeat with old code and compare. Need to manually create output array
    basis_old = make_legacy_basis_functions(system, xp=xp)
    if row_range:
        basis_old.set_matrix_distribution(row_range.start, row_range.stop)
        out_shape = (row_range.stop - row_range.start, basis_old.Mmax)
    else:
        out_shape = (basis_old.Mmax, basis_old.Mmax)

    V_MN_ref = xp.zeros(out_shape)
    basis_old.calculate_potential_matrix(vt_G.data, V_MN_ref, 0)

    xp.testing.assert_allclose(V_MN,
                               V_MN_ref,
                               rtol=1e-10,
                               atol=1e-12)


@pytest.mark.parametrize("xp", xp_params_no_cpupy())
@pytest.mark.parametrize("purepython", parametrize_purepython())
@pytest.mark.parametrize("num_spins", [1, 2])
def test_add_to_density(fixt_lfc_system: LFCSystem,
                        xp,
                        purepython: bool,
                        num_spins: int):
    """"""
    system = fixt_lfc_system
    basis = make_basis(system, xp, purepython)
    # Matrix distribution does not matter here.
    # TODO separate test that it indeed does not matter?

    # Test density
    rng = xp.random.default_rng(432)
    sG_shape = (num_spins, *system.grid.mysize_c)
    nt_sG = rng.random(sG_shape)

    nt_sG_ref = nt_sG.copy()

    # Make test "array" f_asi. This is actually a dict of arrays, because
    # the number of basis functions (range of i) can vary per atom.
    f_asi = {}
    for atom_idx in range(basis.num_atoms):
        shape_si = (num_spins, basis.get_num_phi_a(atom_idx))
        f_asi[atom_idx] = rng.random(shape_si)

    basis.add_to_density(nt_sG, f_asi)

    # Repeat with old code.
    # this still only works with numpy (density is converted internally)
    f_asi = {a: as_np(f_asi[a]) for a in f_asi.keys()}
    basis_old = make_legacy_basis_functions(system, xp=np)
    basis_old.set_matrix_distribution(1, 2)
    basis_old.add_to_density(nt_sG_ref, f_asi)

    xp.testing.assert_allclose(nt_sG, nt_sG_ref, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("xp", xp_params_no_cpupy())
@pytest.mark.parametrize("purepython", parametrize_purepython())
@pytest.mark.parametrize("num_spins", [1, 2])
@pytest.mark.skipif(world.size < 2, reason="Not enough MPI ranks")
def test_domain_decomposition(
    fixt_lfc_system: LFCSystem,
    xp,
    purepython: bool,
    num_spins: int):
    """"""
    system = fixt_lfc_system
    basis = make_basis(system, xp, purepython)

    global world
    world = cast(MPIComm, world)
    # Did we construct with the right comm
    assert system.grid.comm is world

    serial_comm = world.new_communicator([0])

    serial_system = copy.copy(system)
    serial_system.grid = system.grid.new(comm=serial_comm)
    serial_basis = make_basis(serial_system, xp, purepython)

    assert serial_basis.num_atoms == basis.num_atoms
    assert serial_basis.num_basis_functions() == basis.num_basis_functions()

    # add_to_density. We want to compare against a pure-serial computation,
    # so each rank should use the same f_asi input => use same seed
    rng = xp.random.default_rng(999)
    f_asi = {}
    for atom_idx in range(basis.num_atoms):
        shape_si = (num_spins, basis.get_num_phi_a(atom_idx))
        f_asi[atom_idx] = rng.random(shape_si)

    """Goal is to test that add_to_density on one rank (full grid) is
    equivalent to add_to_density on the domain-decomposed grid. We need to
    generate a random input density array on the full grid, then decompose it.
    Easiest is to generate it on all ranks with same seed, then just copy
    correct parts of it as the decomposed input. This avoids need for scatterv
    or similar.
    """

    # FIXME: mpi_gather and mpi_scatter crash or deadlock if the nt_sG arrays
    # are not same shape on all ranks. Such calls are invalid but we should
    # raise exception, not crash/deadlock!

    # Re-seed just in case
    rng = xp.random.default_rng(841)
    nt_sG_serial = rng.random((num_spins, *system.grid.size))
    # Decomposed density array:
    sG_shape = (num_spins, *system.grid.mysize_c)
    nt_sG = xp.empty(sG_shape)

    def manual_scatter(full_nt_sG, decomp_nt_sG):
        start_c = system.grid.start_c
        end_c = system.grid.end_c
        decomp_nt_sG[:] = full_nt_sG[:,
                                     start_c[0]:end_c[0],
                                     start_c[1]:end_c[1],
                                     start_c[2]:end_c[2]
                                    ]

    manual_scatter(nt_sG_serial, nt_sG)

    basis.add_to_density(nt_sG, f_asi)
    # Do the serial part separately in all ranks because this test doesn't
    # actually use MPI for scattering the reference result
    serial_basis.add_to_density(nt_sG_serial, f_asi)

    nt_sG_ref = xp.empty_like(nt_sG)
    # Scatter the serial result back to all ranks
    manual_scatter(nt_sG_serial, nt_sG_ref)

    xp.testing.assert_allclose(nt_sG, nt_sG_ref, rtol=1e-10, atol=1e-12)


# def test_grid():
#     """TEMP"""

#     global world
#     n = 4
#     a = 2.0
#     grid = UGDesc(cell=[a, a, a], size=(n, n, n), comm=world)

#     def mpiprint(s):
#         print(f"Rank {world.rank}: {s}", flush=True)

#     # size is 3D tuple, full grid shape. size_c is the same but as np array.
#     # myshape is 3D tuple shape of the decomposed domain, mysize_c is the same but as np array
#     # mypos_c is the location of this domain on "the grid of domains".
#     # start_c and end_c tell what range of indices of the full grid are included in this domain
#     # Old code uses grid._gc.n_c which seems to be the same as mysize_c
#     mpiprint(f"{grid.size=}, {grid.size_c=}, {grid.mysize_c=}, {grid.mypos_c=}, {grid.start_c=}, {grid.end_c=}, {grid._gd.n_c=}")
