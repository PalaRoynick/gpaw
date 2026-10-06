"""The solvation cavity must follow the atoms when they move.
"""
import numpy as np
import pytest
from ase.build import molecule

from gpaw.new.ase_interface import GPAW
from gpaw.solvation import Solvation
from gpaw.solvation import get_HW14_water_kwargs
from gpaw.utilities.adjust_cell import adjust_cell

H = 0.3  # AA, grid spacing
VAC = 3.0  # AA, cell padding
SHIFT = 0.2  # AA, how much to perturb the moving atom


def make_atoms():
    atoms = molecule('H2O')
    adjust_cell(atoms, border=VAC, h=H)
    return atoms


def make_calc(mode, label):
    kwargs = dict(
        mode=mode,
        xc='LDA',
        txt=f'{mode}-{label}.txt',
        symmetry='off',  # displacement will break symmetry
        convergence={'energy': 1e-4, 'density': 1e-5,
                     'eigenstates': 1e-7},
        extensions=[Solvation(**get_HW14_water_kwargs())])
    if mode == 'fd':
        kwargs['h'] = H
    return GPAW(**kwargs)


def test_cavity_follows_atoms(in_tmp_dir):
    for mode in ['pw', 'fd']:
        check_cavity_follows_atoms(mode)


def check_cavity_follows_atoms(mode):
    atoms = make_atoms()
    atoms.calc = make_calc(mode, 'same_calc')
    atoms.get_potential_energy()

    solvation = atoms.calc.dft.solvation
    g_g = solvation.cavity.g_g.copy()
    eps_g = solvation.dielectric.eps_gradeps[0].copy()

    # Displace one atom on the *same* calculator.
    atoms.positions[0, 0] += SHIFT
    e_same = atoms.get_potential_energy()

    assert not np.allclose(solvation.cavity.g_g, g_g), \
        'cavity did not follow the atoms'
    assert not np.allclose(solvation.dielectric.eps_gradeps[0], eps_g), \
        'dielectric did not follow the cavity'

    # A calculator built at the displaced geometry must agree.
    fresh = make_atoms()
    fresh.positions[0, 0] += SHIFT
    fresh.calc = make_calc(mode, 'fresh_calc')
    e_fresh = fresh.get_potential_energy()
    assert e_same == pytest.approx(e_fresh, abs=5e-3)
