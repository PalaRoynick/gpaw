from contextlib import nullcontext

import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from gpaw.dft import Parameters
from gpaw.new.ase_interface import GPAW
from gpaw.new.extensions import Extension
from gpaw.new import scf


def build_potential(comm, mode: str, parallel: dict):
    atoms = Atoms('H2',
                  positions=[(1.35, 1.5, 1.5), (1.65, 1.5, 1.5)],
                  cell=[3.0, 3.0, 3.0],
                  pbc=True)

    modeobj = {'name': 'fd'} if mode == 'fd' else {'name': 'pw', 'ecut': 80}
    params = Parameters(mode=modeobj,
                        h=None if mode == 'fd' else 0.5,
                        gpts=(16, 16, 16) if mode == 'fd' else None,
                        nbands=2,
                        kpts={'size': [4, 1, 1], 'gamma': True},
                        symmetry='off',
                        xc='LDA',
                        parallel=parallel)
    builder = params.dft_component_builder(atoms, log=None, comm=comm)
    basis = builder.create_basis_set()
    density = builder.density_from_superposition(basis)
    pot_calc = builder.create_potential_calculator()
    density.normalize(pot_calc.charge)
    potential, _, _ = pot_calc.calculate_without_orbitals(
        density,
        kpt_band_comm=builder.communicators['D'])
    return builder, potential


def redist_potential(potential, source_builder, target_builder):
    """Move potential between two GPAW k/b/d parallel layouts."""
    return potential.redist(target_builder.grid,
                            target_builder.electrostatic_potential_desc,
                            target_builder.atomdist,
                            source_builder.communicators['D'],
                            target_builder.communicators['D'])


def gather_array(array):
    if array is None:
        return None
    gathered = array.to_xp(np).gather(broadcast=True)
    assert gathered is not None
    return gathered.data.copy()


def snapshot(potential):
    dH_asii = potential.dH_asii.gather(broadcast=True, copy=True)
    assert dH_asii is not None
    return (gather_array(potential.vt_sR),
            dH_asii.data.copy(),
            gather_array(potential.dedtaut_sR),
            gather_array(potential.vHt_x),
            potential.e_stress)


def assert_same_potential(potential, reference):
    data = snapshot(potential)
    for actual, expected in zip(data, reference):
        if actual is None:
            assert expected is None
        else:
            np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.ci
@pytest.mark.parametrize('mode', ['fd', 'pw'])
def test_potential_redist_between_parallel_layouts(comm, mode):
    if comm.size != 4:
        pytest.skip('Test needs exactly 4 MPI ranks')

    source_builder, source_potential = build_potential(
        comm, mode, {'domain': 2, 'kpt': 2, 'band': 1})
    target_builders = [
        build_potential(comm, mode, {'domain': 4, 'kpt': 1, 'band': 1})[0],
        build_potential(comm, mode, {'domain': 1, 'kpt': 1, 'band': 4})[0],
        build_potential(comm, mode, {'domain': 1, 'kpt': 4, 'band': 1})[0],
        build_potential(comm, mode, {'domain': 2, 'kpt': 1, 'band': 2})[0]]

    assert source_builder.grid.comm.size == 2
    assert source_builder.communicators['D'].size == 2
    assert [builder.grid.comm.size for builder in target_builders] == [
        4, 1, 1, 2]
    assert [builder.communicators['D'].size
            for builder in target_builders] == [1, 4, 4, 2]

    source_data = snapshot(source_potential)

    for target_builder in target_builders:
        target_potential = redist_potential(source_potential,
                                            source_builder,
                                            target_builder)
        assert target_potential.vt_sR.desc.comm.size == (
            target_builder.grid.comm.size)
        assert target_potential.dH_asii.layout.atomdist.comm.size == (
            target_builder.atomdist.comm.size)
        assert_same_potential(target_potential, source_data)

        roundtrip_potential = redist_potential(target_potential,
                                               target_builder,
                                               source_builder)
        assert roundtrip_potential.vt_sR.desc.comm.size == (
            source_builder.grid.comm.size)
        assert_same_potential(roundtrip_potential, source_data)


@pytest.mark.ci
def test_potential_calculator_rejects_mismatched_density(comm):
    if comm.size != 4:
        pytest.skip('Test needs exactly 4 MPI ranks')

    source_builder, source_potential = build_potential(
        comm, 'fd', {'domain': 2, 'kpt': 2, 'band': 1})
    target_builder, _ = build_potential(
        comm, 'fd', {'domain': 4, 'kpt': 1, 'band': 1})

    basis = source_builder.create_basis_set()
    density = source_builder.density_from_superposition(basis)
    mismatched_density = density.redist(
        target_builder.grid,
        target_builder.grid,
        target_builder.atomdist,
        source_builder.communicators['D'],
        target_builder.communicators['D'])

    pot_calc = source_builder.create_potential_calculator()
    with pytest.raises(AssertionError):
        pot_calc.calculate_without_orbitals(
            mismatched_density,
            vHt_x=source_potential.vHt_x,
            kpt_band_comm=source_builder.communicators['D'])


@pytest.mark.ci
@pytest.mark.parametrize('mode', ['fd', 'pw'])
def test_scf_uses_full_potential_calculator(comm, mode):
    if comm.size != 4:
        pytest.skip('Test needs exactly 4 MPI ranks')

    atoms = molecule('H2', vacuum=1.0)
    if mode == 'fd':
        modeobj = 'fd'
        kwargs = {'gpts': (16, 16, 16)}
    else:
        modeobj = {'name': 'pw', 'ecut': 80}
        kwargs = {}

    calc = GPAW(mode=modeobj,
                xc='LDA',
                nbands=2,
                convergence={'maximum iterations': 2},
                parallel={'domain': 2, 'band': 2, 'potential': 'full'},
                communicator=comm,
                txt=None,
                **kwargs)
    atoms.calc = calc
    atoms.get_potential_energy()

    full_pot_calc = calc.dft.pot_calc._full_potential_calculator
    assert full_pot_calc.grid.comm.size == comm.size
    assert full_pot_calc.density_atomdist.comm.size == comm.size
    assert calc.dft.potential.vt_sR.desc.comm.size == 2


@pytest.mark.ci
def test_full_potential_with_extensions_warns_and_falls_back(comm):
    if comm.size != 4:
        pytest.skip('Test needs exactly 4 MPI ranks')

    atoms = molecule('H2', vacuum=1.0)
    calc = GPAW(mode='fd',
                xc='LDA',
                nbands=2,
                gpts=(16, 16, 16),
                convergence={'maximum iterations': 2},
                parallel={'domain': 2, 'band': 2, 'potential': 'full'},
                extensions=[Extension()],
                communicator=comm,
                txt=None)
    atoms.calc = calc

    warning = pytest.warns(RuntimeWarning, match='Falling back')
    with warning if comm.rank == 0 else nullcontext():
        atoms.get_potential_energy()

    assert not hasattr(calc.dft.pot_calc, '_full_potential_calculator')


@pytest.mark.ci
def test_full_potential_interpolator_error_warns_and_falls_back(monkeypatch):
    class Comm:
        size = 2
        rank = 0

        def sum_scalar(self, value):
            return value

    class Nt:
        data = object()

    class Density:
        ncomponents = 1
        nt_sR = Nt()

    class Potential:
        vHt_x = None

    class Ibzwfs:
        kpt_band_comm = Comm()

    class PotCalc:
        extensions = []

        def calculate(self, density, ibzwfs, vHt_x):
            return 'potential', 'energies', None

    def raise_interpolator_error(pot_calc, full_density):
        raise ValueError(
            'Cannot construct interpolator.  Grid 12 x 12 x 12 may be '
            'too small.')

    monkeypatch.setattr(scf, 'distribute_density_to_all_ranks',
                        lambda density, kpt_band_comm, world_comm: density)
    monkeypatch.setattr(scf, 'create_full_potential_calculator',
                        raise_interpolator_error)

    with pytest.warns(RuntimeWarning, match='Cannot construct interpolator'):
        potential, energies = scf.calculate_potential(
            Density(), Potential(), PotCalc(), Ibzwfs(), Comm(), 'full')

    assert potential == 'potential'
    assert energies == 'energies'


@pytest.mark.ci
def test_full_potential_charge_center_warns_and_falls_back(monkeypatch):
    class Comm:
        size = 2
        rank = 0

    class Nt:
        data = object()

    class Density:
        ncomponents = 1
        nt_sR = Nt()

    class Potential:
        vHt_x = None

    class Ibzwfs:
        kpt_band_comm = Comm()

    class Solver:
        use_charge_center = True

    class PoissonSolver:
        solver = Solver()

    class PotCalc:
        extensions = []
        poisson_solver = PoissonSolver()

        def calculate(self, density, ibzwfs, vHt_x):
            return 'potential', 'energies', None

    def unsupported(*args):
        raise AssertionError('full potential path should not be reached')

    monkeypatch.setattr(scf, 'distribute_density_to_all_ranks', unsupported)

    with pytest.warns(RuntimeWarning, match='charge-center compensation'):
        potential, energies = scf.calculate_potential(
            Density(), Potential(), PotCalc(), Ibzwfs(), Comm(), 'full')

    assert potential == 'potential'
    assert energies == 'energies'


@pytest.mark.ci
def test_full_potential_spin_density_warns_and_falls_back(monkeypatch):
    class Comm:
        size = 2
        rank = 0

    class Nt:
        data = object()

    class Density:
        ncomponents = 2
        nt_sR = Nt()

    class Potential:
        vHt_x = None

    class Ibzwfs:
        kpt_band_comm = Comm()

    class PotCalc:
        extensions = []

        def calculate(self, density, ibzwfs, vHt_x):
            return 'potential', 'energies', None

    def unsupported(*args):
        raise AssertionError('full potential path should not be reached')

    monkeypatch.setattr(scf, 'distribute_density_to_all_ranks', unsupported)

    with pytest.warns(RuntimeWarning, match='spin-dependent densities'):
        potential, energies = scf.calculate_potential(
            Density(), Potential(), PotCalc(), Ibzwfs(), Comm(), 'full')

    assert potential == 'potential'
    assert energies == 'energies'


@pytest.mark.ci
def test_full_potential_qna_warns_and_falls_back(monkeypatch):
    class Comm:
        size = 2
        rank = 0

    class Nt:
        data = object()

    class Density:
        ncomponents = 1
        nt_sR = Nt()

    class Potential:
        vHt_x = None

    class Ibzwfs:
        kpt_band_comm = Comm()

    class XC:
        name = 'QNA'

    class PotCalc:
        extensions = []
        xc = XC()

        def calculate(self, density, ibzwfs, vHt_x):
            return 'potential', 'energies', None

    def unsupported(*args):
        raise AssertionError('full potential path should not be reached')

    monkeypatch.setattr(scf, 'distribute_density_to_all_ranks', unsupported)

    with pytest.warns(RuntimeWarning, match='QNA is not supported'):
        potential, energies = scf.calculate_potential(
            Density(), Potential(), PotCalc(), Ibzwfs(), Comm(), 'full')

    assert potential == 'potential'
    assert energies == 'energies'


def test_create_full_xc_preserves_qna_parameters():
    from gpaw.core import UGDesc
    from gpaw.new.xc import create_functional
    from gpaw.xc import XC

    qna = {'alpha': 2.0,
           'name': 'QNA',
           'orbital_dependent': False,
           'parameters': {'H': (0.1485, 0.005)},
           'setup_name': 'PBE',
           'type': 'qna-gga'}
    atoms = Atoms('H', pbc=True, cell=[5, 5, 5])
    grid = UGDesc(cell=[5, 5, 5], size=(8, 8, 8))
    xc = create_functional(XC(qna, atoms=atoms), grid)

    full_xc = xc.new(grid=grid, xp=np)

    assert full_xc.name == 'QNA'
    assert full_xc.xc.todict()['parameters'] == qna['parameters']
    assert full_xc.xc.todict()['setup_name'] == 'PBE'
