import numpy as np
import pytest

from gpaw import GPAW
from gpaw.directmin.derivatives import Derivatives
from gpaw.new.lcao.derivatives import DerivativesNew


@pytest.mark.do
def test_hess_numerically_lcao(in_tmp_dir, gpw_files, gpaw_new):
    """
    Test complex numerical Hessian
    w.r.t rotation parameters in LCAO

    :param in_tmp_dir:
    :return:
    """

    if gpaw_new == 0:

        calc = GPAW(gpw_files['h_hess_num_lcao'])
        calc.set_positions()
        calc.wfs.eigensolver.initialize_dm_helper(
            calc.wfs, calc.hamiltonian,
            calc.density, calc.log
        )
        atoms = calc.atoms
        atoms.calc = calc
        numder = Derivatives(calc.wfs.eigensolver, calc.wfs)

        hess_n = numder.get_numerical_derivatives(
            calc.wfs.eigensolver,
            calc.hamiltonian,
            calc.wfs,
            calc.density,
            what2calc='hessian'
        )
        hess_a = numder.get_analytical_derivatives(
            calc.wfs.eigensolver,
            calc.hamiltonian,
            calc.wfs,
            calc.density,
            what2calc='hessian'
        )
        hess_nt = np.asarray([[1.32720630e+00, -1.93947467e-11],
                             [3.95786680e-09, 1.14599176e+00]])
        assert hess_n == pytest.approx(hess_nt, abs=1e-4)
        assert hess_a == pytest.approx(hess_nt, abs=0.2)

        a_mat_u = {0: [np.sqrt(2) * np.pi / 4.0 + 1.0j * np.sqrt(2) * np.pi / 4.0]}
        c_nm_ref = calc.wfs.eigensolver.dm_helper.reference_orbitals
        calc.wfs.eigensolver.rotate_wavefunctions(calc.wfs, a_mat_u, c_nm_ref)
        calc.wfs.eigensolver.update_ks_energy(
            calc.hamiltonian, calc.wfs, calc.density)
        calc.wfs.eigensolver.get_canonical_representation(
            calc.hamiltonian, calc.wfs, calc.density, sort_eigenvalues=True)
        c_nm = {x: calc.wfs.kpt_u[x].C_nM.copy()
                for x in range(len(calc.wfs.kpt_u))}

        numder = Derivatives(calc.wfs.eigensolver, calc.wfs, c_ref=c_nm)

        hess_n = numder.get_numerical_derivatives(
            calc.wfs.eigensolver,
            calc.hamiltonian,
            calc.wfs,
            calc.density,
            what2calc='hessian'
        )

        hess_a = numder.get_analytical_derivatives(
            calc.wfs.eigensolver,
            calc.hamiltonian,
            calc.wfs,
            calc.density,
            what2calc='hessian'
        )
        hess_nt = np.asarray([[-1.08209601e+00, -1.11022302e-09],
                             [8.50014503e-10, -9.37664521e-01]])
        assert hess_n == pytest.approx(hess_nt, abs=1e-4)
        assert hess_a == pytest.approx(hess_nt, abs=0.2)

    else:

        calc = GPAW(gpw_files['h_hess_num_lcao'])
        #calc.set_positions()
        eigensolver = calc._dft.scf_loop.eigensolver
        pot_calc = calc._dft.pot_calc
        potential = calc._dft.potential

        shuttle = {'ibzwfs': calc.wfs.ibzwfs,
               'density': calc._dft.density,
               'ref_orbitals': None,
               'pot_calc': pot_calc,
               'potential': potential,
               'hamiltonian': calc.wfs.hamiltonian,
               'a_vec_u': None
               }
        
        eigensolver.initialize(shuttle['ibzwfs'], shuttle['density'],
                                    shuttle['potential'],
                                    shuttle['hamiltonian'],
                                    shuttle['pot_calc'])
        
        numder = DerivativesNew(eigensolver, shuttle)

        hess_n = numder.get_numerical_derivatives(
            eigensolver, shuttle,
            what2calc='hessian'
        )

        hess_a = numder.get_analytical_derivatives(
            eigensolver, shuttle,
            what2calc='hessian'
        )

        hess_nt = np.asarray([[1.32720630e+00, -1.93947467e-11],
                             [3.95786680e-09, 1.14599176e+00]])
        assert hess_n == pytest.approx(hess_nt, abs=1e-4)
        assert hess_a == pytest.approx(hess_nt, abs=0.2)


