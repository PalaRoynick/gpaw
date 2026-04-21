# dump_h2o_multipoles_and_polar_tensors.py

import numpy as np
from ase import Atoms
from gpaw import GPAW
from gpaw.eigensolvers import RMMDIIS
from gpaw.multipoles import MomTensor, PolarTensor
from ase.parallel import parprint

# --- Common geometry ---------------------------------------------------------

H2O_SYMBOLS = ['O', 'H', 'H']
H2O_POSITIONS = [
    [0.00000000,  0.00000000,  0.06664447],
    [0.76804750,  0.00000000, -0.52889132],
    [-0.76804750, 0.00000000, -0.52889132],
]
PBC = [False, False, False]


def build_h2o_atoms():
    mol = Atoms(
        symbols=H2O_SYMBOLS,
        positions=H2O_POSITIONS,
        pbc=PBC,
    )
    mol.center(vacuum=3.5)
    return mol


def build_water_moments():
    """Return (monopole, dipole, quad, octu, hexa) for H2O."""
    gpaw_args = dict(
        mode="fd",
        convergence={'eigenstates': 1e-1, 'density': 1e-1},
        h=0.25,
        xc="LDA",
        eigensolver=RMMDIIS(niter=5),
    )

    molecule = build_h2o_atoms()
    calc = GPAW(**gpaw_args)
    molecule.calc = calc

    mom = MomTensor(molecule)
    return mom.get_multipole_moments()


def build_polar_tensors():
    """Return (dd, dq, qq) for H2O."""
    gpaw_args = dict(
        mode="fd",
        convergence={'eigenstates': 1e-1, 'density': 1e-1},
        h=0.25,
        xc='LDA',
        eigensolver=RMMDIIS(niter=5),
    )

    molecule = build_h2o_atoms()
    calc = GPAW(**gpaw_args)
    molecule.calc = calc

    pol = PolarTensor(molecule)
    dd = pol.calc_dd_pol()
    dq = pol.calc_dq_pol()
    qq = pol.calc_qq_pol()
    return dd, dq, qq


def np_literal(a, varname):
    """Return a string 'varname = np.array([...])' suitable for copy‑paste."""
    body = np.array2string(
        a,
        separator=', ',
        max_line_width=120,
        precision=8,
        floatmode='maxprec_equal',
    )
    return f"{varname} = np.array({body})"


if __name__ == "__main__":
    monopole, dipole, quad, octu, hexa = build_water_moments()
    dd, dq, qq = build_polar_tensors()

    # Scalar monopole as simple float
    parprint(f"ref_monopole = np.float64({monopole:.14f})\n")

    # Vector/tensor quantities as np.array literals
    parprint(np_literal(dipole, "ref_dipole") + "\n")
    parprint(np_literal(quad, "ref_quad") + "\n")
    parprint(np_literal(octu, "ref_octu") + "\n")
    parprint(np_literal(hexa, "ref_hexa") + "\n")

    parprint(np_literal(dd, "dd_ref") + "\n")
    parprint(np_literal(dq, "dq_ref") + "\n")
    parprint(np_literal(qq, "qq_ref") + "\n")