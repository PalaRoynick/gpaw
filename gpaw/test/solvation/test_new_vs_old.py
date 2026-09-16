"""Check that the energies in the new solvation implementation
match those in the old.

Note the old only had finite-difference mode, the new has both
finite-difference and plane-wave. We don't expect total energies
to match between the modes, so for pw we only check interaction energies.
"""
import pytest
from ase.build import molecule
from ase.units import Hartree, Pascal, m

from gpaw import PW
from gpaw.solvation import (KB51Volume, SolvationGPAW, SurfaceInteraction,
                            VolumeInteraction, get_HW14_water_kwargs)
from gpaw.utilities.adjust_cell import adjust_cell

H = 0.24  # AA, FD grid spacing
VAC = 4.0  # AA, cell padding


def make_atoms():
    atoms = molecule('H2O')
    adjust_cell(atoms, border=VAC, h=H)
    return atoms


def make_calc(mode, legacy, txt, interactions=None,):
    kwargs = dict(legacy_gpaw=legacy, xc='LDA', symmetry='off', txt=txt,
                  **get_HW14_water_kwargs())
    kwargs['interactions'] = interactions or []
    kwargs['cavity'].volume_calculator = KB51Volume()
    if not legacy:
        kwargs['experimental'] = {'backwards_compatible': False}
    if mode == 'pw':
        kwargs.update(mode=PW(400.))
    else:
        kwargs.update(dict(mode='fd', h=H))
    return SolvationGPAW(**kwargs)


def get_interaction_energies(calc, legacy):
    """Returns interaction energies as {subscript: energy} in eV,
    either mode."""
    if legacy:
        h = calc.hamiltonian
        e_ia = {
            ia.subscript: getattr(h, 'e_' + ia.subscript) * Hartree
            for ia in h.interactions}
        return e_ia
    ext = calc.dft.solvation
    e_ia = {k: v * Hartree for k, v in ext.interaction_energies.items()}
    return e_ia


def test_fd_energy_new_vs_old(in_tmp_dir):
    """New-FD must reproduce old-FD."""
    ia = [SurfaceInteraction(surface_tension=18.4e-3 * Pascal * m)]

    atoms = make_atoms()
    atoms.calc = make_calc('fd', True, 'old_fd.txt', ia)
    e_old = atoms.get_potential_energy()
    atoms.calc = make_calc('fd', False, 'new_fd.txt', ia)
    e_new = atoms.get_potential_energy()
    assert e_new == pytest.approx(e_old, abs=1e-4)


def test_interaction_energy_consistency(in_tmp_dir):
    """Check that the interaction energies are consistent between the
    new (plane-wave and finite-difference) and old (finite-difference
    only) implementations.
    """
    ia = [SurfaceInteraction(surface_tension=18.4e-3 * Pascal * m),
          VolumeInteraction(pressure=-1e9 * Pascal)]

    atoms = make_atoms()
    atoms.calc = make_calc('fd', True, 'ia_old.txt', ia)
    atoms.get_potential_energy()
    e_ia_old = get_interaction_energies(atoms.calc, legacy=True)

    atoms.calc = make_calc('fd', False, 'ia_fd.txt', ia)
    atoms.get_potential_energy()
    e_ia_fd = get_interaction_energies(atoms.calc, legacy=False)

    atoms.calc = make_calc('pw', False, 'ia_pw.txt', ia)
    atoms.get_potential_energy()
    e_ia_pw = get_interaction_energies(atoms.calc, legacy=False)

    assert set(e_ia_old) == {'surf', 'vol'}
    for subscript, e_old in e_ia_old.items():
        assert e_ia_fd[subscript] == pytest.approx(e_old, abs=1e-4)
        assert e_ia_pw[subscript] == pytest.approx(e_old, abs=1e-3)
