import numpy as np
import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.derivatives import Derivatives
from gpaw.new.lcao.derivatives import DerivativesNew
from gpaw.directmin.etdm_lcao import LCAOETDM
from ase import Atoms


def h2o_maker(vacuum, t=np.pi / 180 * 104.51, eps=0, **kwargs):
    # d = 0.9575
    # H2O = Atoms(
    #     "OH2",
    #     positions=[(0, 0, 0), (d + eps, 0, 0), (d * np.cos(t), d * np.sin(t), 0)],
    #     **kwargs,
    # )
    # H2O.center(vacuum=vacuum)

    atoms = Atoms("H", positions=[[0, 0, 0]])
    atoms.center(vacuum=5.0)
    atoms.set_pbc(False)

    return atoms


def h2o_cdo_lcao_sic():

    calc = GPAW(
        xc="PBE",
        mode=LCAO(force_complex_dtype=True),
        h=0.25,
        basis="dz(dzp)",
        spinpol=False,
        eigensolver=LCAOETDM(representation="u-invar"),
        # eigensolver={"name": "etdm-lcao", "representation": "u-invar"},
        occupations={"name": "fixed-uniform"},
        mixer={"backend": "no-mixing"},
        nbands="nao",
        symmetry="off",
    )

    # calc = GPAW(
    #     xc="PBE",
    #     mode=LCAO(force_complex_dtype=True),
    #     h=0.22,
    #     basis="dz(dzp)",
    #     spinpol=True,
    #     # eigensolver={'name': 'etdm-lcao',
    #     #              'representation': 'u-invar'},
    #     eigensolver=LCAOETDM(
    #         functional={"name": "PZ-SIC", "scaling_factor": (0.5, 0.5)},
    #         localizationseed=42,
    #         localizationtype="PM_PZ",
    #         subspace_convergence=1e-3,
    #         # grad_tol_pz_localization=1.0e-3,
    #         # maxiter_pz_localization=200,
    #         # converge_unocc=True,
    #     ),
    #     occupations={"name": "fixed-uniform"},
    #     mixer={"backend": "no-mixing"},
    #     convergence={"maximum iterations": max_iter},
    #     nbands="nao",
    #     symmetry="off",

    # )
    return calc


# max_iters = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# max_iters = [5, 6, 7, 8, 9, 10]
# max_iters = [200]


@pytest.mark.do
@pytest.mark.parametrize("gpaw_new", [False])
# @pytest.mark.parametrize("max_iter", max_iters)
def test_hess_numerically_lcao(in_tmp_dir, gpw_files, gpaw_new):
    """
    Test complex numerical Hessian
    w.r.t rotation parameters in LCAO

    :param in_tmp_dir:
    :return:
    """

    atoms = h2o_maker(vacuum=4.0)
    calc = h2o_cdo_lcao_sic()
    atoms.calc = calc
    atoms.get_potential_energy()

    # Numerical derivatives need full position-dependent state (nct_G,
    # vbar, etc.) which read() does not provide for density/hamiltonian.
    calc.initialize_positions()
    numder = Derivatives(calc.wfs.eigensolver, calc.wfs)

    hess_n = numder.get_numerical_derivatives(
        calc.wfs.eigensolver,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc="hessian",
    )
    hess_a = numder.get_analytical_derivatives(
        calc.wfs.eigensolver,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc="hessian",
    )
    hess_nt = np.asarray(
        [[1.32720630e00, -1.93947467e-11], [3.95786680e-09, 1.14599176e00]]
    )

    print("hessian comparison")
    # print(max_iter)
    print(hess_a)
    print(hess_n)

    assert hess_n == pytest.approx(hess_nt, abs=1e-4)
    assert hess_a == pytest.approx(hess_nt, abs=0.2)

    # assert hess_n == pytest.approx(hess_nt, abs=1e-4)
    # assert hess_a == pytest.approx(hess_nt, abs=0.2)

    # assert hess_n == pytest.approx(hess_a, abs=0.2)

    a_mat_u = {0: [np.sqrt(2) * np.pi / 4.0 + 1.0j * np.sqrt(2) * np.pi / 4.0]}
    c_nm_ref = calc.wfs.eigensolver.dm_helper.reference_orbitals
    calc.wfs.eigensolver.rotate_wavefunctions(calc.wfs, a_mat_u, c_nm_ref)
    calc.wfs.eigensolver.update_ks_energy(calc.hamiltonian, calc.wfs, calc.density)
    calc.wfs.eigensolver.get_canonical_representation(
        calc.hamiltonian, calc.wfs, calc.density, sort_eigenvalues=True
    )
    c_nm = {x: calc.wfs.kpt_u[x].C_nM.copy() for x in range(len(calc.wfs.kpt_u))}

    numder = Derivatives(calc.wfs.eigensolver, calc.wfs, c_ref=c_nm)

    hess_n = numder.get_numerical_derivatives(
        calc.wfs.eigensolver,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc="hessian",
    )

    hess_a = numder.get_analytical_derivatives(
        calc.wfs.eigensolver,
        calc.hamiltonian,
        calc.wfs,
        calc.density,
        what2calc="hessian",
    )
    hess_nt = np.asarray(
        [[-1.08209601e00, -1.11022302e-09], [8.50014503e-10, -9.37664521e-01]]
    )
    assert hess_n == pytest.approx(hess_nt, abs=1e-4)
    assert hess_a == pytest.approx(hess_nt, abs=0.2)
    # print(hess_a)
    # print(hess_n)
    # assert hess_n == pytest.approx(hess_a, abs=0.2)
