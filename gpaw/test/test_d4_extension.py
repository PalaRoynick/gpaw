import numpy as np
import pytest
from ase import Atoms

from gpaw import restart
from gpaw.new.ase_interface import GPAW
from gpaw.extensions.d4 import D4

pytest.importorskip("dftd4")


def _get_h2():
    from ase.build import molecule

    atoms = molecule('H2')
    atoms[0].position[1] += 1
    atoms.center(vacuum=2)
    atoms.set_pbc((True, True, True))
    return atoms


def _get_dftd4():
    # Upstream DFTD4 calculator used as the reference dispersion-only
    # correction for the GPAW extension tests below.
    return pytest.importorskip('dftd4.ase').DFTD4


def _d4_energy_forces(atoms, method='PBE'):
    dftd4 = _get_dftd4()
    atoms = atoms.copy()
    atoms.calc = dftd4(method=method)
    return atoms.get_potential_energy(), atoms.get_forces()


def _d4_energy_stress(atoms, method='PBE'):
    dftd4 = _get_dftd4()
    atoms = atoms.copy()
    atoms.calc = dftd4(method=method)
    return atoms.get_potential_energy(), atoms.get_stress(voigt=False)


def test_d4_input_roundtrip():
    from gpaw.dft import ExtensionInput

    ext = D4(xc='PBE', model='d4s')
    restored = ExtensionInput.from_input(
        {'name': 'd4', 'xc': 'PBE', 'model': 'd4s'})

    assert ext.todict() == {'xc': 'PBE', 'model': 'd4s'}
    assert isinstance(restored, D4)
    assert restored.todict() == {'xc': 'PBE', 'model': 'd4s'}


def test_d4_rejects_reserved_name():
    with pytest.raises(TypeError, match='reserved keyword'):
        D4(xc='PBE', name='foo')


@pytest.mark.parametrize('parallel', [(1, 1), (1, 2), (2, 2), (2, 1)])
@pytest.mark.parametrize('mode', [{'name': 'pw', 'ecut': 300}, 'lcao'])
def test_d4_extension_energy_forces(mode, parallel, in_tmp_dir, mpi):
    domain, band = parallel
    if mpi.comm.size < domain * band:
        pytest.skip('Not enough cores for this test.')
    if mpi.comm.size > domain * band * 2:
        pytest.skip('Too many cores for this test.')

    atoms = _get_h2()
    atoms.calc = mpi.NewGPAW(
        extensions=[D4(xc='PBE')],
        symmetry='off',
        parallel={'band': band, 'domain': domain},
        kpts=(2, 1, 1),
        mode=mode)
    e, f = atoms.get_potential_energy(), atoms.get_forces()

    d4_e, d4_f = _d4_energy_forces(atoms)

    atoms.calc = mpi.NewGPAW(mode=mode, kpts=(2, 1, 1), symmetry='off')
    e0, f0 = atoms.get_potential_energy(), atoms.get_forces()

    # The extension result should match plain GPAW plus the standalone D4
    # correction from the reference calculator above.
    assert e == pytest.approx(e0 + d4_e)
    assert f == pytest.approx(f0 + d4_f, abs=5e-5)


def test_d4_extension_move_atoms(in_tmp_dir, mpi):
    atoms = _get_h2()
    atoms.calc = mpi.NewGPAW(
        extensions=[D4(xc='PBE')],
        symmetry='off',
        kpts=(2, 1, 1),
        mode={'name': 'pw', 'ecut': 300})
    atoms.get_potential_energy()

    atoms.positions[0, 2] -= 0.1
    moved_e, moved_f = atoms.get_potential_energy(), atoms.get_forces()
    d4_e, d4_f = _d4_energy_forces(atoms)

    atoms.calc = mpi.NewGPAW(
        symmetry='off',
        kpts=(2, 1, 1),
        mode={'name': 'pw', 'ecut': 300})
    e0, f0 = atoms.get_potential_energy(), atoms.get_forces()

    assert moved_e == pytest.approx(e0 + d4_e)
    assert moved_f == pytest.approx(f0 + d4_f, abs=1.5e-4)


def test_d4_restart(in_tmp_dir, mpi):
    atoms = _get_h2()
    atoms.calc = mpi.NewGPAW(
        extensions=[D4(xc='PBE')],
        symmetry='off',
        kpts=(2, 1, 1),
        mode={'name': 'pw', 'ecut': 300})
    e = atoms.get_potential_energy()
    f = atoms.get_forces()
    atoms.calc.write('calc.gpw')

    atoms, calc = restart('calc.gpw', Class=GPAW, communicator=mpi.comm)
    assert e == pytest.approx(atoms.get_potential_energy())
    assert f == pytest.approx(atoms.get_forces())


@pytest.mark.parametrize('parallel', [(1, 1), (1, 2), (2, 2), (2, 1)])
def test_d4_stress(parallel, in_tmp_dir, mpi):
    from ase.build import bulk

    domain, band = parallel
    if mpi.comm.size < domain * band:
        pytest.skip('Not enough cores for this test.')
    if mpi.comm.size > domain * band * 2:
        pytest.skip('Too many cores for this test.')

    atoms = bulk('C', a=3.5)
    atoms.calc = mpi.NewGPAW(
        extensions=[D4(xc='PBE')],
        symmetry='off',
        parallel={'band': band, 'domain': domain},
        kpts=(2, 2, 2),
        convergence={'density': 1e-5},
        mode={'name': 'pw', 'ecut': 300})
    e, s = atoms.get_potential_energy(), atoms.get_stress(voigt=False)

    d4_e, d4_s = _d4_energy_stress(atoms)

    atoms.calc = mpi.NewGPAW(
        symmetry='off',
        kpts=(2, 2, 2),
        convergence={'density': 1e-5},
        mode={'name': 'pw', 'ecut': 300})
    e0, s0 = atoms.get_potential_energy(), atoms.get_stress(voigt=False)

    # Check the same additive contract for energy and stress.
    assert e == pytest.approx(e0 + d4_e)
    assert s == pytest.approx(s0 + d4_s, abs=5e-5)


def test_d4_model_d4s_smoke(in_tmp_dir, mpi):
    atoms = _get_h2()
    atoms.calc = mpi.NewGPAW(
        extensions=[D4(xc='PBE', model='d4s')],
        symmetry='off',
        kpts=(2, 1, 1),
        mode={'name': 'pw', 'ecut': 300})

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (2, 3)


def test_d4_isolated_atom(in_tmp_dir, mpi):
    atoms = Atoms('He')
    atoms.center(vacuum=3)
    atoms.calc = mpi.NewGPAW(
        xc='PBE',
        extensions=[D4(xc='PBE')],
        mode='pw')

    # Check the D4 correction for an isolated atom.
    atoms.get_potential_energy()
    assert np.allclose(atoms.get_forces(), 0, atol=1e-5)
