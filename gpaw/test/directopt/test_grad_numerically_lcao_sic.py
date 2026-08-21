import numpy as np
import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.derivatives import Derivatives
from gpaw.new.lcao.derivatives import DerivativesNew
from ase import Atoms


def h3_maker(vacuum=2.0, pbc=True):
    atoms = Atoms('H3', positions=[(0, 0, 0),
                                    (0.59, 0, 0),
                                    (1.1, 0, 0)])
    atoms.center(vacuum=vacuum)
    atoms.set_pbc(pbc)
    return atoms

def h3_do_num_lcao(legacy_gpaw):
        atm = h3_maker()
        atm.calc = GPAW(
            legacy_gpaw=legacy_gpaw,
            mode=LCAO(force_complex_dtype=True),
            basis='sz(dzp)',
            h=0.3,
            spinpol=False,
            convergence={'eigenstates': 10.0,
                         'density': 10.0,
                         'energy': 10.0},
            occupations={'name': 'fixed-uniform'},
            eigensolver={'name': 'etdm-lcao',
                         'matrix_exp': 'egdecomp'},
            mixer={'backend': 'no-mixing'},
            nbands='nao',
            symmetry='off',
            txt=None)
        atm.get_potential_energy()
        return atm.calc

@pytest.mark.do
@pytest.mark.parametrize('legacy_gpaw', [True])
@pytest.mark.parametrize("representation", ['full', 'sparse'])
@pytest.mark.parametrize("matrix_exp", ['egdecomp', 'pade-approx'])
@pytest.mark.parametrize("localizationtype", ['pm', 'pz'])

def test_gradient_numerically_lcao(gpw_files, legacy_gpaw, representation,
                                   matrix_exp, localizationtype):
    """
    test exponential transformation
    direct minimization method for KS-DFT in LCAO
    :param in_tmp_dir:
    :return:
    """

    calc = h3_do_num_lcao(legacy_gpaw)
    atoms = calc.atoms
    atoms.calc = calc

    calc = calc.new(eigensolver={'name': 'etdm-lcao',
                     'representation': representation,
                     'matrix_exp': matrix_exp,
                     'functional': {'name': 'PZ-SIC', 'scaling_factor': (1. ,1.)},
                     'localizationtype': localizationtype,
                     },)

    atoms.calc = calc
    atoms.get_potential_energy()
    ham = calc.hamiltonian
    wfs = calc.wfs
    dens = calc.density

    if legacy_gpaw:

        # Do we get consistent results regardless of randomization?
        rngs = [False,
                np.random.default_rng(123456)]
        num_ders = [Derivatives(wfs.eigensolver,
                                wfs,
                                random_amat=rng,
                                update_c_ref=True)
                    for rng in rngs]

        analytical_results = [
            der.get_analytical_derivatives(
                wfs.eigensolver, ham, wfs, dens, what2calc='gradient')[0]
            for der in num_ders]
        numerical_results = [
            der.get_numerical_derivatives(
                wfs.eigensolver, ham, wfs, dens, what2calc='gradient')[0]
            for der in num_ders]
        for x, *ys in zip(*analytical_results, *numerical_results):
            assert np.array(ys).real == pytest.approx(np.array(ys).real, abs=1.0e-2)
            assert np.array(ys).imag == pytest.approx(x.imag, abs=1.0e-2)

    # else:

    #     eigensolver = calc._dft.scf_loop.eigensolver
    #     pot_calc = calc._dft.pot_calc
    #     potential = calc._dft.potential

    #     shuttle = {'ibzwfs': wfs.ibzwfs,
    #             'density': calc._dft.density,
    #             'ref_orbitals': None,
    #             'pot_calc': pot_calc,
    #             'potential': potential,
    #             'hamiltonian': wfs.hamiltonian,
    #             'a_vec_u': None
    #             }

    #     # Do we get consistent results regardless of randomization?
    #     seeds = [42, 56, 77]
            
    #     num_ders = [DerivativesNew(eigensolver, shuttle,
    #                                 random_amat=True,
    #                                 seed=seed,
    #                                 update_c_ref=True) for seed in seeds]
        
    #     analytical_results = [
    #         der.get_analytical_derivatives(
    #             eigensolver, shuttle, what2calc='gradient')[0]
    #         for der in num_ders]
        
    #     numerical_results = [
    #         der.get_numerical_derivatives(
    #             eigensolver, shuttle, what2calc='gradient')[0]
    #         for der in num_ders]
        
    #     for x, *ys in zip(*analytical_results, *numerical_results):
    #         assert np.array(ys).real == pytest.approx(x.real, abs=1.0e-2)
    #         assert np.array(ys).imag == pytest.approx(x.imag, abs=1.0e-2)