import pytest
import numpy as np
from ase.lattice.hexagonal import Graphene
from gpaw.dft import DFT


def test_xc():

    atoms = Graphene(symbol='B',
                     latticeconstant={'a': 2.5, 'c': 1.0},
                     size=(1, 1, 1))
    atoms[0].symbol = 'N'
    atoms.set_pbc(True)
    atoms.center(axis=2, vacuum=3.0)

    ppcg = {'name': 'ppcg',
            'niter': 30,
            'min_niter': 2}

    params = {'xc': 'PBE',
              'mode': {'name': 'pw', 'ecut': 400, 'force_complex_dtype': False},
              'nbands': '120%',
              'eigensolver': ppcg,
              'convergence': {'eigenstates': 1e-6,
                              'density': 1e-4,
                              'forces': 1e-2}}

    xc_hse = {'name': 'HYB_GGA_XC_HSE06',
              'fraction': 0.26,
              'omega': 0.11,
              'backend': 'pw'}

    # preconverge with PBE
    dft = DFT(atoms, **params)
    dft.converge()

    dft.change_xc(xc_hse)

    # fixed_density
    if 0:
        dft.scf_loop.update_density_and_potential = False
        dft.converge(steps=1)
        dft.scf_loop.update_density_and_potential = True

    ase_calc = dft.ase_calculator()
    etot_xc = ase_calc.get_potential_energy(atoms)
    forces_xc = ase_calc.get_forces(atoms)

    if 1:
        from gpaw.new.ase_interface import GPAW
        # check against HSE
        params['xc'] = xc_hse
        calc = GPAW(**params)
        atoms.calc = calc
        etot_hse = atoms.get_potential_energy()
        forces_hse = atoms.get_forces()

    assert etot_xc == pytest.approx(etot_hse, abs=1e-4)
    assert forces_xc == pytest.approx(forces_hse, abs=5e-2)


if __name__ == "__main__":
    test_xc()
