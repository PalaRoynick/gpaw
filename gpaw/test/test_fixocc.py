import numpy as np
from ase.build import molecule
from ase.parallel import parprint
from gpaw import GPAW
from gpaw.utilities.adjust_cell import adjust_cell
import pytest


def test_fixocc_varmom():
    # test variable_magmom keyword in fixed-uniform

    f_t = [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 0, 0, 0, 0]]
    f_t = np.array(f_t)

    h = 0.2
    box = 2
    nbands = 8
    charge = -1
    magmoms = [1, -1]

    atoms = molecule('HF')
    adjust_cell(atoms, box, h)
    atoms.set_initial_magnetic_moments(magmoms)

    convergence = {'energy': 0.01, 'eigenstates': 0.001, 'density': 0.01}
    occupations = {'name': 'fixed-uniform', 'variable_magmom': True}

    params = dict(mode='fd',
                  h=h,
                  nbands=nbands,
                  charge=charge,
                  spinpol=True,
                  convergence=convergence,
                  occupations=occupations)

    calc = GPAW(**params)
    atoms.calc = calc
    atoms.get_potential_energy()

    def get_occupations(calc, kpt=0):
        nspins = calc.get_number_of_spins()
        f_sn = [calc.get_occupation_numbers(spin=spin, kpt=kpt,
                                            raw=True).copy()
                for spin in range(nspins)]
        return np.array(f_sn)

    f_sn = get_occupations(atoms.calc)
    assert f_sn == pytest.approx(f_t, abs=1.e-6)


def test_fixocc():
    h = 0.4
    box = 2
    nbands = 2
    txt = '-'
    txt = None

    H2 = molecule('H2')
    adjust_cell(H2, box, h)
    convergence = {'energy': 0.01, 'eigenstates': 0.001, 'density': 0.01}

    base_kwargs = dict(
        mode='fd',
        h=h,
        nbands=nbands,
        convergence=convergence,
        txt=txt)

    if 1:
        # test ZeroKelvin vs FixedOccupations
        c = GPAW(**base_kwargs, occupations={'width': 0.0})
        H2.calc = c
        E_zk = H2.get_potential_energy()

        c = GPAW(**base_kwargs,
                 occupations=dict(name='fixed', numbers=[[1, 0]]))
        H2.calc = c
        E_fo = H2.get_potential_energy()
        parprint(E_zk, E_fo)
        assert E_zk == pytest.approx(E_fo, abs=1.e-10)

    if 1:
        # test spin-paired vs spin-polarized
        c = GPAW(**base_kwargs,
                 occupations={'name': 'fixed', 'numbers': [[0.5, 0.5]]})
        H2.calc = c
        E_ns = H2.get_potential_energy()
    if 1:
        c = GPAW(**base_kwargs,
                 spinpol=True,
                 occupations={'name': 'fixed', 'numbers': [[0.5, 0.5]] * 2})
        H2.calc = c
        E_sp = H2.get_potential_energy()
        parprint(E_ns, E_sp)
        assert E_ns == pytest.approx(E_sp, abs=1.e-6)


if __name__ == "__main__":
    test_fixocc_varmom()
