import numpy as np
import pytest

from gpaw import GPAW, LCAO
from gpaw.directmin.derivatives import Derivatives
from gpaw.new.lcao.derivatives import DerivativesNew
from gpaw.directmin.etdm_lcao import LCAOETDM
from ase import Atoms


def h2o_maker(vacuum, t=np.pi / 180 * 104.51, eps=0, **kwargs):
    d = 0.9575
    H2O = Atoms(
        "OH2",
        positions=[(0, 0, 0), (d + eps, 0, 0), (d * np.cos(t), d * np.sin(t), 0)],
        **kwargs,
    )
    H2O.center(vacuum=vacuum)
    return H2O


def h2o_cdo_lcao_sic():

    calc = GPAW(
        mode=LCAO(force_complex_dtype=True),
        h=0.22,
        occupations={"name": "fixed-uniform"},
        # eigensolver={
        #     'name': 'etdm-lcao',
        #     'localizationtype': 'PM_PZ',
        #     'localizationseed': 42,
        #     'subspace_convergence': 1e-3,
        #     'functional': {'name': 'PZ-SIC', 'scaling_factor': (0.5, 0.5)},
        # },
        eigensolver=LCAOETDM(
            functional={"name": "PZ-SIC", "scaling_factor": (0.5, 0.5)},
            localizationseed=42,
            localizationtype="PM_PZ",
            # grad_tol_pz_localization=1.0e-3,
            # maxiter_pz_localization=200,
            # converge_unocc=True,
        ),
        convergence={"eigenstates": 1e-4},
        mixer={"backend": "no-mixing"},
        nbands="nao",
        symmetry="off",
        spinpol=True,
    )
    # atm.get_potential_energy()
    return calc


@pytest.mark.do
@pytest.mark.parametrize("gpaw_new", [False])
def test_gradient_numerically_lcao(in_tmp_dir, gpw_files, gpaw_new):
    """
    test exponential transformation
    direct minimization method for KS-DFT in LCAO
    :param in_tmp_dir:
    :return:
    """
    atoms = h2o_maker(vacuum=4.0)
    calc = h2o_cdo_lcao_sic()
    atoms.calc = calc
    atoms.get_potential_energy()

    ham = calc.hamiltonian
    wfs = calc.wfs
    dens = calc.density

    # Do we get consistent results regardless of randomization?
    rngs = [False, np.random.default_rng(123456)]
    num_ders = [
        Derivatives(wfs.eigensolver, wfs, random_amat=rng, update_c_ref=True)
        for rng in rngs
    ]

    analytical_results = [
        der.get_analytical_derivatives(
            wfs.eigensolver, ham, wfs, dens, what2calc="gradient"
        )[0]
        for der in num_ders
    ]
    numerical_results = [
        der.get_numerical_derivatives(
            wfs.eigensolver, ham, wfs, dens, what2calc="gradient"
        )[0]
        for der in num_ders
    ]
    for x, *ys in zip(*analytical_results, *numerical_results):
        assert np.array(ys).real == pytest.approx(x.real, abs=1.0e-2)
        assert np.array(ys).imag == pytest.approx(x.imag, abs=1.0e-2)
