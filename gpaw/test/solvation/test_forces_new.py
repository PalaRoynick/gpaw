"""Forces for implicit solvation (new GPAW); PW & FD.

Using an NaCl system, checks that the forces match numerical
forces, match each other, and that the solvation term is not
negligible.
"""
import pytest
from ase import Atoms
from ase.calculators.fd import calculate_numerical_forces

from gpaw import PW, FermiDirac
from gpaw.new.ase_interface import GPAW
from gpaw.new.solvation import Solvation
from gpaw.solvation import (EffectivePotentialCavity, GradientSurface,
                            KB51Volume, LinearDielectric, Power12Potential)
from gpaw.solvation.interactions import SurfaceInteraction, VolumeInteraction


def make_calc(mode, txt, solvated=True, interactions=None):
    mode_kwargs = (dict(mode=PW(400.)) if mode == 'pw'
                   else dict(mode='fd', h=0.18))
    extensions = []
    if solvated:
        extensions = [Solvation(
            cavity=EffectivePotentialCavity(
                effective_potential=Power12Potential(u0=0.180),
                temperature=298.15,
                surface_calculator=GradientSurface(),
                volume_calculator=KB51Volume()),
            dielectric=LinearDielectric(epsinf=80.0),
            interactions=interactions or [])]
    return GPAW(
        **mode_kwargs,
        kpts=(1, 1, 1),
        xc='PBE',
        setups={'Na': '1'},
        occupations=FermiDirac(0.1),
        txt=txt,
        experimental={'backwards_compatible': False},
        extensions=extensions)


def test_solvation_forces(in_tmp_dir):
    # Atomic system is a vertically oriented NaCl.
    atoms = Atoms('NaCl', positions=[(0., 0., 0.), (0., 0., 2.4)])
    atoms.set_cell([10., 10., 12.])
    atoms.center()
    atoms.pbc = True

    # PW: forces match physical limits (only z, anti-symmetric).
    atoms.calc = make_calc('pw', 'nacl_pw.txt')
    F_pw = atoms.get_forces()
    assert F_pw[:, :2] == pytest.approx(0.0, abs=5e-3)
    assert F_pw[0, 2] == pytest.approx(-F_pw[1, 2], abs=0.03)

    # PW: forces match -dE/dz of the energy.
    f_num = calculate_numerical_forces(atoms, eps=0.03, iatoms=[0],
                                       icarts=[2], force_consistent=True)
    assert F_pw[0, 2] == pytest.approx(f_num[0, 0], abs=0.03)

    # PW: the solvation term is not negligible.
    atoms.calc = make_calc('pw', 'nacl_pw_bare.txt', solvated=False)
    F_bare = atoms.get_forces()
    assert abs(F_pw[0, 2] - F_bare[0, 2]) > 0.1

    # PW/FD: plane-wave forces match finite-difference mode forces.
    atoms.calc = make_calc('fd', 'nacl_fd.txt')
    F_fd = atoms.get_forces()
    assert F_pw[0, 2] == pytest.approx(F_fd[0, 2], abs=0.03)

    # FD: forces match -dE/dz of the energy.
    f_num = calculate_numerical_forces(atoms, eps=0.03, iatoms=[0],
                                       icarts=[2], force_consistent=True)
    assert F_fd[0, 2] == pytest.approx(f_num[0, 0], abs=0.03)

    # FD: the solvation term is not negligible.
    atoms.calc = make_calc('fd', 'nacl_fd_bare.txt', solvated=False)
    F_bare = atoms.get_forces()
    assert abs(F_fd[0, 2] - F_bare[0, 2]) > 0.1


def test_pw_solvation_interaction_forces(in_tmp_dir):
    """Surface/Volume interaction forces in PW: analytic vs -dE/dz.

    Both interactions are density-independent, so they add nothing to the
    KS potential -- the SCF is unchanged and the constants below are
    deliberately exaggerated to make the -gamma*dA/dz - p*dV/dz force term
    resolvable against the numerical-force tolerance.
    """
    ia = [SurfaceInteraction(surface_tension=0.10),   # eV / Ang^2
          VolumeInteraction(pressure=-0.02)]          # eV / Ang^3

    atoms = Atoms('NaCl', positions=[(0., 0., 0.), (0., 0., 2.4)])
    atoms.set_cell([10., 10., 12.])
    atoms.center()
    atoms.pbc = True

    atoms.calc = make_calc('pw', 'nacl_pw_ia.txt', interactions=ia)
    F_ia = atoms.get_forces()

    # The interaction-force term is actually present (not swamped by tol).
    atoms.calc = make_calc('pw', 'nacl_pw_noia.txt')
    F_only_ia = F_ia[0, 2] - atoms.get_forces()[0, 2]
    assert abs(F_only_ia) > 0.5

    # PW force matches -dE/dz of the (interaction-inclusive) energy.
    atoms.calc = make_calc('pw', 'nacl_pw_ia_fd.txt', interactions=ia)
    f_num = calculate_numerical_forces(atoms, eps=0.02, iatoms=[0],
                                       icarts=[2], force_consistent=True)
    assert F_ia[0, 2] == pytest.approx(f_num[0, 0], abs=0.01)

    # PW matches FD mode for the interaction-inclusive force too.
    atoms.calc = make_calc('fd', 'nacl_fd_ia.txt', interactions=ia)
    assert F_ia[0, 2] == pytest.approx(atoms.get_forces()[0, 2], abs=0.01)
