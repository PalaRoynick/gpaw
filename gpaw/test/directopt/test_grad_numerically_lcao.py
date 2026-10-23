import numpy as np
import pytest

from gpaw import GPAW
from gpaw.directmin.derivatives import Derivatives
from gpaw.new.lcao.derivatives import DerivativesNew


@pytest.mark.do
@pytest.mark.parametrize('legacy_gpaw', [False])
def test_gradient_numerically_lcao(in_tmp_dir, gpw_files, legacy_gpaw):
    """
    test exponential transformation
    direct minimization method for KS-DFT in LCAO
    :param in_tmp_dir:
    :return:
    """

    calc = GPAW(gpw_files['h3_do_num_lcao'])
    atoms = calc.atoms
    atoms.calc = calc

    params = [{'name': 'etdm-lcao',
               'representation': 'full',
               'matrix_exp': 'egdecomp'},
              {'name': 'etdm-lcao',
               'representation': 'full',
               'matrix_exp': 'pade-approx'},
              {'name': 'etdm-lcao',
               'representation': 'sparse',
               'matrix_exp': 'egdecomp'},
              {'name': 'etdm-lcao',
               'representation': 'sparse',
               'matrix_exp': 'pade-approx'},
              {'name': 'etdm-lcao',
               'representation': 'u-invar',
               'matrix_exp': 'egdecomp'},
              {'name': 'etdm-lcao',
               'representation': 'u-invar',
               'matrix_exp': 'egdecomp-u-invar'}]

    for eigsolver in params:
        print('IN PROGRESS: ', eigsolver)

        calc = calc.new(eigensolver=eigsolver)
        atoms.calc = calc
        if eigsolver['representation'] == 'u-invar':
            with pytest.warns(UserWarning,
                              match="Use representation == 'sparse'"):
                atoms.get_potential_energy()
        else:
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
                assert np.array(ys).real == pytest.approx(x.real, abs=1.0e-2)
                assert np.array(ys).imag == pytest.approx(x.imag, abs=1.0e-2)

        else:

            eigensolver = calc._dft.scf_loop.eigensolver
            pot_calc = calc._dft.pot_calc
            potential = calc._dft.potential

            shuttle = {'ibzwfs': wfs.ibzwfs,
                       'density': calc._dft.density,
                       'ref_orbitals': None,
                       'pot_calc': pot_calc,
                       'potential': potential,
                       'hamiltonian': wfs.hamiltonian,
                       'a_vec_u': None
                       }

            # Do we get consistent results regardless of randomization?
            seeds = [42, 56, 77]

            num_ders = [DerivativesNew(eigensolver, shuttle,
                                       random_amat=True,
                                       seed=seed,
                                       update_c_ref=True) for seed in seeds]

            analytical_results = [
                der.get_analytical_derivatives(
                    eigensolver, shuttle, what2calc='gradient')[0]
                for der in num_ders]

            numerical_results = [
                der.get_numerical_derivatives(
                    eigensolver, shuttle, what2calc='gradient')[0]
                for der in num_ders]

            for x, *ys in zip(*analytical_results, *numerical_results):
                assert np.array(ys).real == pytest.approx(x.real, abs=1.0e-2)
                assert np.array(ys).imag == pytest.approx(x.imag, abs=1.0e-2)
