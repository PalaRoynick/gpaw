"""SJM against the legacy calculator, in fd and pw mode.

agts.py runs each function here as a separate task.
"""
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111
from ase.units import Bohr, Ha

from gpaw import GPAW, FermiDirac
from gpaw.extensions.electrochemistry import SJM
from gpaw.mpi import broadcast, synchronize_atoms, world
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            LinearDielectric, SurfaceInteraction)
from gpaw.solvation.sjm import SJM as OldSJM
from gpaw.solvation.sjm import SJMPower12Potential
from gpaw.test.solvation.test_sjm_keywords import NoCavity, Vacuum


def test_h(gpaw_new, mode):
    atoms, params, sjm, solvation = h_atom_inputs()
    params['mode'] = mode

    if gpaw_new:
        extensions = [SJM(solvation=solvation, potentiostat=sjm)]
        atoms.calc = GPAW(extensions=extensions, **params)
        energy = atoms.get_potential_energy()
        pot = -atoms.calc.get_fermi_level()
    else:
        atoms.calc = OldSJM(**solvation, sj=sjm, **params)
        energy = atoms.get_potential_energy()
        pot = atoms.calc.get_electrode_potential()

    assert pot == pytest.approx(7.5, abs=0.01)

    if gpaw_new:
        check_grand_canonical_results(atoms, energy, extensions[0], mode)

    atoms.write(f'h-{gpaw_new}-{mode}.traj')
    if gpaw_new:
        atoms.calc.write(f'h-{mode}.gpw')
        calc = GPAW(f'h-{mode}.gpw',
                    object_hooks={'extensions': lambda exts: []})
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


def check_grand_canonical_results(atoms, energy, sjm, mode):
    """The grand potential - not the canonical energy - goes to ASE."""
    calc = atoms.calc
    results = calc.results
    N = sjm.excess_electrons

    assert results['excess_electrons'] == pytest.approx(N)
    assert results['electrode_potential'] == pytest.approx(
        sjm.electrode_potential)

    # Omega = E - N mu, with mu = -workfunction = fermi level:
    mu_N = N * calc.get_fermi_level()
    canonical = calc.dft.energies.total_extrapolated * Ha
    canonical_free = calc.dft.energies.total_free * Ha
    assert energy == pytest.approx(canonical - mu_N)
    assert results['energy'] == pytest.approx(canonical - mu_N)
    assert results['free_energy'] == pytest.approx(canonical_free - mu_N)
    # The excess charge is small but non-zero, so the two must differ:
    assert abs(energy - canonical) > 1e-6

    sjm.write_sjm_traces(path=f'traces-{mode}',
                         props=('potential', 'background_charge'))
    z, v = np.loadtxt(f'traces-{mode}/potential.txt').T
    assert len(z) == len(v) and 0.0 <= z[0] < z[-1] < atoms.cell[2, 2]
    # The potential trace is referenced to the Fermi level, so it equals the
    # work function in the vacuum where the Poisson solver zeroes it.  Both
    # modes zero it at the last z index -- the top of the cell, above the
    # jellium -- so this is one-sided:
    pot = -calc.get_fermi_level()
    assert abs(v[-1] - pot) < 0.05
    z, rho = np.loadtxt(f'traces-{mode}/background_charge.txt').T
    # The counter charge (in e/Bohr^3) integrates to minus the excess
    # electrons; z and the cell are in Angstrom:
    dV = (z[1] - z[0]) * abs(np.linalg.det(atoms.cell[:2, :2])) / Bohr**3
    assert (rho.sum() * dV) == pytest.approx(-N, abs=1e-6)

    # Cube traces are written through ase.io.write(), which runs on the master
    # rank and then broadcasts the outcome over the whole world, so every rank
    # has to reach it.  Writing them must leave the communicators in step: a
    # rank-0-only call left its broadcast unmatched, and the stale message was
    # then picked up by the next collective -- synchronize_atoms() at the start
    # of the following calculation, which is where it showed up as a hang.
    sjm.write_sjm_traces(path=f'cube-traces-{mode}', style='cube',
                         props=('potential', 'background_charge'))
    sentinel = ('sync', 42) if world.rank == 0 else None
    assert broadcast(sentinel, root=0, comm=world) == ('sync', 42)
    synchronize_atoms(atoms, world)
    if world.rank == 0:
        for prop in ['potential', 'background_charge']:
            assert Path(f'cube-traces-{mode}/{prop}.cube').is_file()


def test_sjm(gpaw_new, mode):
    """An Au slab; agts.py runs fd only, and old GPAW on one core.

    PW is not working at the moment, and old GPAW fails in parallel:
    https://gitlab.com/gpaw/gpaw/-/issues/1381
    """
    # Solvent parameters
    u0 = 0.180  # eV
    epsinf = 78.36  # dielectric constant of water at 298 K
    gamma = 0.00114843767916  # 18.4*1e-3 * Pascal* m
    T = 298.15  # K

    # Structure is created
    atoms = fcc111('Au', size=(1, 1, 3))
    atoms.cell[2][2] = 15
    atoms.translate([0, 0, 6 - min(atoms.positions[:, 2])])

    # SJM parameters
    potential = 4.5
    tol = 0.02
    sj = {'target_potential': potential,
          'excess_electrons': -0.045,
          'jelliumregion': {'top': 14.5},
          'tol': tol}

    convergence = {
        'energy': 0.05 / 8.,
        'density': 1e-4,
        'eigenstates': 1e-4}

    params = dict(
        mode=mode,
        kpts=(2, 2, 1),
        xc='PBE',
        convergence=convergence,
        occupations=FermiDirac(0.1),
        txt=f'{gpaw_new}-{mode}.txt')

    solvation = dict(
        cavity=EffectivePotentialCavity(
            effective_potential=SJMPower12Potential(u0=u0,
                                                    unsolv_backside=False),
            temperature=T,
            surface_calculator=GradientSurface()),
        dielectric=LinearDielectric(epsinf=epsinf),
        interactions=[SurfaceInteraction(surface_tension=gamma)])

    if not gpaw_new:
        atoms.calc = OldSJM(**params, sj=sj, **solvation)
        atoms.get_potential_energy()
        pot = atoms.calc.get_electrode_potential()
    else:
        atoms.calc = GPAW(
            **params,
            extensions=[SJM(solvation=solvation, potentiostat=sj)])
        atoms.get_potential_energy()
        pot = -atoms.calc.get_fermi_level()

    assert abs(pot - potential) < tol

    atoms.write(f'Au-{gpaw_new}-{mode}.traj')

    atoms.calc.write(f'Au-{gpaw_new}-{mode}.gpw')
    if gpaw_new:
        calc = GPAW(f'Au-{gpaw_new}-{mode}.gpw')
        assert isinstance(calc.params.extensions[0], SJM)
        # The live settings are written, the equilibrated charge among them:
        written = atoms.calc.params.extensions[0].potentiostat
        read = calc.params.extensions[0].potentiostat
        assert read.todict() == written.todict()
        assert read.charging.region.top == 14.5
