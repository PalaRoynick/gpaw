import numpy as np
import pytest
from ase import Atoms

from gpaw.new.symmetry import (Symmetries, SymmetryAnalysisBug,
                               create_symmetries_object)
from gpaw.utilities.symmetry import (find_set_of_lattice_symmetries,
                                     prune_symmetries)


def test_incomplete_point_group1():

    cell_cv = [[3.0787358319029257, 0.0, 0.0],
               [-1.4504494987687535, 2.7156602464528716, 0.0],
               [-1.4504494987687535, -0.976306364588358, 2.534094800244998]]

    kwargs = {'cell': cell_cv,
              'pbc': (1, 1, 1),
              'tolerance': 1e-1,
              'relative_positions': [[0.5, 0.75, 0.25],
                                     [0.5, 0.25, 0.75],
                                     [0., 0., 0.]],
              'ids': [0, 0, 1],
              'symmorphic': True,
              '_backwards_compatible': False}

    with pytest.raises(SymmetryAnalysisBug):
        sym = Symmetries.from_cell_and_atoms(**kwargs, guarantee_group=False)
    sym = Symmetries.from_cell_and_atoms(**kwargs, guarantee_group=True)
    assert len(sym) == 16


def test_incomplete_point_group2():

    cell_cv = [[6.484107793708888, 0.021988810132164173, 0.033790665556415],
               [-2.725837683619327, 5.883358353028015, -0.03379073060296549],
               [-2.7258376201572068, -2.4138174313122733, 5.365494543828883]]
    position_av = [[-1.36291905, 0.86738539, 1.33292615],
                   [-4.08875625, 2.60215553, 3.99877766],
                   [-1.90903349, 0.04228007, 4.60240183],
                   [1.87913321, -0.023255, 1.93654713],
                   [-0.84670072, 3.51478474, 3.42894735],
                   [2.94146598, 3.44924966, 0.76309265]]

    atoms = Atoms('Sr2Te4', cell=cell_cv, positions=position_av, pbc=True)

    kwargs = {'tolerance': 10**(-5.45), 'symmorphic': True,
              '_backwards_compatible': False}

    with pytest.raises(SymmetryAnalysisBug):
        sym = create_symmetries_object(atoms, **kwargs, guarantee_group=False)
    sym = create_symmetries_object(atoms, **kwargs, guarantee_group=True)
    assert len(sym) == 2

def test_incomplete_point_group_after_prune_symmetries():

    cell = [[5.398741468147337, 0.0, 0.0],
            [2.699370734073669, 4.675447259880091, 0.0],
            [2.699370734073669, 1.5584824199600305, 4.408053950055032]]
    positions = [[5.3987368040710075, 4.492750721011745, 3.1768545010081546],
                 [6.590204659302528, 3.804856434063041, 1.2311956408445004],
                 [4.207268948839487, 3.8048564340630406, 1.2311956408445002],
                 [6.590204659302529, 2.4290678601656315, 3.1768545010081555],
                 [4.207268948839486, 2.4290678601656315, 3.1768545010081555],
                 [5.3987368040710075, 1.7411735732169276, 1.2311956408445006],
                 [5.398736804071007, 3.1169621471143363, 2.2040250709263285],
                 [0.0, 0.0, 0.0]]

    atoms = Atoms('F6SiYb', cell=cell, positions=positions, pbc=True)
    cell_cv = atoms.cell
    relpos_ac = atoms.get_scaled_positions()
    id_a = atoms.numbers

    tol = 1e-5

    U_scc = find_set_of_lattice_symmetries(
        cell_cv, np.array([True, True, True]),
        tolerance=tol, guarantee_group=True)

    with pytest.raises(SymmetryAnalysisBug):
        rotation_scc, translation_sc, atommap_sa = prune_symmetries(
            U_scc, cell_cv, relpos_ac, id_a,
            tolerance=tol, guarantee_group=False)
        Symmetries(cell=cell_cv,
                   rotations=rotation_scc,
                   translations=translation_sc,
                   atommaps=atommap_sa,
                   tolerance=tol)

    rotation_scc, translation_sc, atommap_sa = prune_symmetries(
        U_scc, cell_cv, relpos_ac, id_a,
        tolerance=tol, guarantee_group=True)

    sym = Symmetries(cell=cell_cv,
                     rotations=rotation_scc,
                     translations=translation_sc,
                     atommaps=atommap_sa,
                     tolerance=tol)
    assert len(sym) == 24
