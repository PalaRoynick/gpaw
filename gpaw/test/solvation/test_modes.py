import pickle
from pathlib import Path

import ase.parallel
import numpy as np
import pytest
from ase import Atoms
from ase.units import Bohr, Ha

from gpaw import GPAW, FermiDirac
from gpaw.extensions.electrochemistry import SJM
from gpaw.mpi import broadcast, synchronize_atoms, world
from gpaw.solvation.sjm import SJM as OldSJM


@pytest.mark.parametrize('mode', ['pw', 'fd'])
@pytest.mark.parametrize('gpaw_new', [False, True])
def test_h(gpaw_new, mode, in_tmp_dir):
    if mode == 'pw' and not gpaw_new:
        pytest.skip('PW-mode not implemented for old GPAW')

    atoms, params, sjm, solvation = h_atom_inputs()
    params['mode'] = mode

    if gpaw_new:
        sjm_input = SJM(**sjm, **solvation)
        # dipolelayer and zero_vacuum are not constructor arguments: SJM.build
        # sets them from the mode, so before a calculation they sit at
        # the held Solvation's default.
        assert sjm_input.solvation.dipolelayer is False
        assert sjm_input.solvation.zero_vacuum is False
        atoms.calc = GPAW(extensions=[sjm_input], **params)
        energy = atoms.get_potential_energy()
        # Solvation implements the dipole layer in pw mode only; in fd mode
        # SJMPoissonSolver corrects the dipole itself and Solvation must not
        # be asked for it again.
        assert sjm_input.solvation.dipolelayer is (mode == 'pw')
        assert sjm_input.solvation.zero_vacuum is (mode == 'pw')
        pot = -atoms.calc.get_fermi_level()
    else:
        atoms.calc = OldSJM(**solvation, sj=sjm, **params)
        energy = atoms.get_potential_energy()
        pot = atoms.calc.get_electrode_potential()

    assert pot == pytest.approx(7.5, abs=0.01)

    if gpaw_new:
        check_grand_canonical_results(atoms, energy)

    atoms.write('h.traj')
    if gpaw_new:
        atoms.calc.write('h.gpw')
        calc = GPAW('h.gpw', object_hooks={'extensions': lambda exts: []})
        # The grand-potential energies must survive a gpw round-trip:
        for name in ['energy', 'free_energy', 'excess_electrons',
                     'electrode_potential']:
            assert calc.results[name] == pytest.approx(
                atoms.calc.results[name], abs=1e-10)


def h_atom_inputs():
    """The one-atom SJM system that the tests in this file share."""
    a = 1.4
    atoms = Atoms('H', cell=[a, a, 13.0], pbc=(1, 1, 0))
    atoms.positions[0, 2] = 6.0

    k = 2
    params = dict(kpts=(k, k, 1), occupations=FermiDirac(0.2))
    sjm = {'target_potential': 7.5,
           'excess_electrons': -0.0045,
           'jelliumregion': {'top': 11.0, 'bottom': 9.0},
           'tol': 0.01}
    solvation = dict(cavity=NoCavity(), dielectric=Vacuum(), interactions=[])
    return atoms, params, sjm, solvation


def check_grand_canonical_results(atoms, energy):
    """The grand potential - not the canonical energy - goes to ASE."""
    calc = atoms.calc
    sjm = calc.dft.sjm
    results = calc.results

    assert results['excess_electrons'] == pytest.approx(sjm.excess_electrons)
    assert results['electrode_potential'] == pytest.approx(
        sjm.get_electrode_potential())

    # Omega = E - N mu, with mu = -workfunction = fermi level:
    mu_N = sjm.excess_electrons * calc.get_fermi_level()
    canonical = calc.dft.energies.total_extrapolated * Ha
    canonical_free = calc.dft.energies.total_free * Ha
    assert energy == pytest.approx(canonical - mu_N)
    assert results['energy'] == pytest.approx(canonical - mu_N)
    assert results['free_energy'] == pytest.approx(canonical_free - mu_N)
    # The excess charge is small but non-zero, so the two must differ:
    assert abs(energy - canonical) > 1e-6

    sjm.write_sjm_traces(props=('potential', 'background_charge'))
    z, v = np.loadtxt('sjm_traces/potential.txt').T
    assert len(z) == len(v) and 0.0 <= z[0] < z[-1] < atoms.cell[2, 2]
    # The potential trace is referenced to the Fermi level, so it equals the
    # work function in the vacuum where the Poisson solver zeroes it.  Both
    # modes zero it at the last z index -- the top of the cell, above the
    # jellium -- so this is one-sided:
    pot = -calc.get_fermi_level()
    assert abs(v[-1] - pot) < 0.05
    z, rho = np.loadtxt('sjm_traces/background_charge.txt').T
    # The counter charge (in e/Bohr^3) integrates to minus the excess
    # electrons; z and the cell are in Angstrom:
    dV = (z[1] - z[0]) * abs(np.linalg.det(atoms.cell[:2, :2])) / Bohr**3
    assert (rho.sum() * dV) == pytest.approx(-sjm.excess_electrons, abs=1e-6)

    # Cube traces are written through ase.io.write(), which runs on the master
    # rank and then broadcasts the outcome over the whole world, so every rank
    # has to reach it.  Writing them must leave the communicators in step: a
    # rank-0-only call left its broadcast unmatched, and the stale message was
    # then picked up by the next collective -- synchronize_atoms() at the start
    # of the following calculation, which is where it showed up as a hang.
    sjm.write_sjm_traces(path='sjm_traces_cube', style='cube',
                         props=('potential', 'background_charge'))
    sentinel = ('sync', 42) if world.rank == 0 else None
    assert broadcast(sentinel, root=0, comm=world) == ('sync', 42)
    synchronize_atoms(atoms, world)
    if world.rank == 0:
        for prop in ['potential', 'background_charge']:
            assert Path(f'sjm_traces_cube/{prop}.cube').is_file()


class OneRankOfTwo:
    """Stands in for a two-rank world, seen from the rank that is not master.

    ASE's @parallel_function short-circuits at ``world.size == 1``, so in a
    serial run nothing is broadcast and writing a cube file from rank 0 alone
    looks harmless.  Claiming size 2 and rank 1 puts the wrapper on its
    parallel path, where a call some rank never reaches is a broadcast that
    never happens -- the desync itself -- and that is what gets recorded here.
    """

    size = 2
    rank = 1

    def __init__(self):
        self.broadcasts = []
        # What rank 0 would send after a successful write: ase.io.write()
        # returns None, so parallel_function broadcasts (None, None), as a
        # length header first and then the pickle itself.
        self.payload = np.frombuffer(
            pickle.dumps((None, None), pickle.HIGHEST_PROTOCOL), np.int8)

    def broadcast(self, a, root):
        assert root == 0
        self.broadcasts.append(a.dtype)
        if a.dtype == np.int8:
            a[:] = self.payload
        else:
            a[0] = len(self.payload)

    def barrier(self):
        pass


@pytest.mark.ci
@pytest.mark.serial
def test_cube_traces_are_written_collectively(monkeypatch, in_tmp_dir):
    """Every rank has to reach the ase.io.write() behind a cube trace.

    It is a collective: ASE runs the write on the master rank and broadcasts
    the outcome to the rest.  Guarding the call ourselves left that broadcast
    unmatched, and the stale message was picked up by the next collective --
    synchronize_atoms() at the start of the following calculation -- so a
    relaxation that wrote cube traces hung instead of failing.

    A real MPI run catches it for free -- check_grand_canonical_results() does,
    with a sentinel broadcast -- but only the nightly jobs run one, so the
    world is faked here to keep the regression inside the serial suite too.
    """
    atoms, params, sjm, solvation = h_atom_inputs()
    atoms.calc = GPAW(mode='fd', extensions=[SJM(**sjm, **solvation)],
                      **params)
    atoms.get_potential_energy()
    extension = atoms.calc.dft.sjm

    fake = OneRankOfTwo()
    monkeypatch.setattr(ase.parallel.world, 'comm', fake)
    monkeypatch.setattr(extension.solvation, 'comm', fake)

    extension.write_sjm_traces(path='cubes', style='cube',
                               props=('potential', 'background_charge'))
    # Two broadcasts per property -- the length header and the pickle -- from
    # a rank that writes no file of its own.  None at all would mean the call
    # was skipped on this rank, which is the bug.
    assert fake.broadcasts == [np.dtype(int), np.dtype(np.int8)] * 2

    # The z-style traces go through our own writer, which is not collective,
    # so there the rank-0 guard is the right thing and must stay:
    fake.broadcasts.clear()
    extension.write_sjm_traces(path='zs', style='z', props=('potential',))
    assert fake.broadcasts == []


class NoCavity:
    depends_on_el_density = False

    def set_grid_descriptor(self, gd):
        pass

    def allocate(self):
        pass

    def update_atoms(self, atoms, log):
        pass

    def update(self, atoms, density):
        pass

    def communicate_vol_surf(self, comm):
        pass

    def summary(self, log):
        pass

    def todict(self):
        return {}


class Vacuum:
    def set_grid_descriptor(self, gd):
        self.eps_gradeps = [gd.zeros() for _ in range(4)]
        self.eps_gradeps[0][:] = 1.0

    def allocate(self):
        pass

    def todict(self):
        return {}


if __name__ == '__main__':
    import sys
    test_h(int(sys.argv[1]), sys.argv[2], 1)
