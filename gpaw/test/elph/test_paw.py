"""Independent PAW operator checks (not fitted reference matrix entries)."""
from types import SimpleNamespace

import numpy as np
import pytest
from ase.build import bulk

from gpaw.elph import Supercell, ElectronPhononMatrix
from gpaw.utilities import pack_hermitian


@pytest.mark.serial
@pytest.mark.elph
def test_al_bloch_phases_against_grid_integral(in_tmp_dir):
    """Non-real Bloch phases: compare contraction to real-space integration."""
    from ase.units import Hartree, Bohr
    from ase.utils.timing import Timer
    from gpaw.old.calculator import GPAW
    from gpaw.utilities.tools import tri2full

    atoms = bulk('Al', 'fcc', a=4.04)
    calc = GPAW(mode={'name': 'lcao', 'force_complex_dtype': True},
                basis='sz(dzp)', gpts=(24, 8, 8), kpts=[[0, 0, 0]],
                symmetry='off', txt=None)
    calc.initialize(atoms * (3, 1, 1))
    calc.initialize_positions(atoms * (3, 1, 1))
    wfs = calc.wfs
    rng = np.random.default_rng(22)
    potential = rng.normal(size=tuple(wfs.gd.n_c))
    H = np.zeros((12, 12), complex)
    wfs.basis_functions.calculate_potential_matrix(potential, H, q=0)
    tri2full(H, 'L')
    kernel = H.reshape(3, 4, 3, 4).swapaxes(1, 2)
    c1 = rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))
    c2 = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    R = np.array([-1, 0, 1])
    k = np.array([1 / 3, 0, 0])
    q = np.array([1 / 3, 0, 0])

    def grid(c, wavevector):
        coeff = (np.exp(2j * np.pi * wavevector[0] * R)[None, :, None]
                 * c[:, None]).reshape(len(c), 12).copy()
        psi = wfs.gd.zeros(len(c), dtype=complex)
        wfs.basis_functions.lcao_to_grid(coeff, psi, 0)
        return psi

    psi1, psi2 = grid(c1, k), grid(c2, k + q)
    expected = np.array([[wfs.gd.integrate(p2.conj() * potential * p1)
                          for p1 in psi1] for p2 in psi2])
    elph = ElectronPhononMatrix.__new__(ElectronPhononMatrix)
    elph.world = SimpleNamespace(rank=1)
    elph.comm = elph.world
    elph.indices = [0]
    elph.timer = Timer()
    elph.R_cN = np.array([R, np.zeros(3), np.zeros(3)])
    elph.phonon = SimpleNamespace(band_structure=lambda *a, **kw: (
        np.ones((1, 3)), np.eye(3).reshape(1, 3, 1, 3)))
    elph._yield_g_NNMM = lambda x, s: kernel if x == 0 else np.zeros_like(
        kernel)
    result = elph._bloch_matrix(c1, c2, k, q, prefactor=False, s=0)
    assert result[0] / (Hartree / Bohr) == pytest.approx(expected, abs=1e-12)


@pytest.mark.serial
@pytest.mark.elph
def test_al_projector_derivative_against_finite_difference(in_tmp_dir):
    """Move only the Al projector center; keep all basis functions fixed."""
    from copy import copy
    from ase.units import Bohr
    from gpaw.old.calculator import GPAW

    atoms = bulk('Al', 'fcc', a=4.04) * (3, 1, 1)
    calc = GPAW(mode='lcao', basis='dzp', gpts=(24, 8, 8),
                kpts=(2, 1, 1), symmetry='off', txt=None)
    calc.initialize(atoms)
    calc.initialize_positions(atoms)
    wfs = calc.wfs
    # Duplicate centers: the first half carries projectors, the second half
    # carries a frozen copy of the basis. No SCF calculation is involved.
    expansions = copy(wfs.tciexpansions)
    expansions.I_a = np.tile(expansions.I_a, 2)
    for name in ('P_expansions', 'O_expansions', 'T_expansions'):
        expansion = copy(getattr(expansions, name))
        expansion.I1_a = np.tile(expansion.I1_a, 2)
        expansion.I2_a = np.tile(expansion.I2_a, 2)
        setattr(expansions, name, expansion)
    positions = np.tile(atoms.get_scaled_positions(), (2, 1))
    a = 1
    step = 1e-4  # Bohr

    def overlaps(displacement):
        pos = positions.copy()
        pos[a] += displacement @ np.linalg.inv(atoms.cell.array / Bohr)
        tci = expansions.get_tci_calculator(
            wfs.gd.cell_cv, pos, atoms.pbc, wfs.kd.ibzk_qc, wfs.dtype)
        blocks = [tci.P(a, len(atoms) + b).swapaxes(-1, -2).conj()
                  for b in range(len(atoms))]
        return np.concatenate(blocks, axis=1)

    P = overlaps(np.zeros(3))
    assert P == pytest.approx(wfs.P_aqMi[a], abs=1e-12)
    analytic = wfs.manytci.P_aqMi([a], derivative=True)[a]
    for v in range(3):
        shift = np.eye(3)[v] * step
        finite = (overlaps(shift) - overlaps(-shift)) / (2 * step)
        assert analytic[:, v] == pytest.approx(finite, abs=2e-6)

    # A malformed cache must fail on metadata before attempting to read
    # missing displacement arrays, even if their grid sizes could coincide.
    from ase.utils.filecache import MultiFileJSONCache
    cache = MultiFileJSONCache('bad-fd')
    cache['info'] = dict(natom=1, supercell=(7, 1, 1), delta=.01,
                         dr_version=1)
    sc = Supercell(bulk('Al', 'fcc', a=4.04), supercell=(3, 1, 1))
    with pytest.raises(ValueError, match='supercells must match'):
        sc.calculate_supercell_matrix(calc, fd_name='bad-fd')


@pytest.mark.elph
def test_degenerate_average_is_gauge_invariant():
    from gpaw.elph.gmatrix import degenerate_average as reduce_degenerate

    rng = np.random.default_rng(7)
    g = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    U = np.linalg.qr(rng.normal(size=(3, 3))
                     + 1j * rng.normal(size=(3, 3)))[0]
    W = np.linalg.qr(rng.normal(size=(2, 2))
                     + 1j * rng.normal(size=(2, 2)))[0]
    rotated = np.einsum('ab,bmn,ni->ami', W, g, U)
    select = (np.arange(2), np.arange(3), np.arange(3))
    assert reduce_degenerate(g, *select) == pytest.approx(
        reduce_degenerate(rotated, *select), abs=1e-14)


@pytest.mark.elph
def test_centered_projectors_spin_and_complex_convention():
    """Differentiate a frozen nonlocal operator at the centered atom.

    Complex projector overlaps catch accidental conjugation; unequal spin
    Hamiltonians catch a leaked spin-loop variable. The corner atom carries
    different overlaps, so using atom 0 instead of the center fails.
    """
    rng = np.random.default_rng(2401)
    P = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    dP = rng.normal(size=P.shape) + 1j * rng.normal(size=P.shape)
    H = np.array([[[2., .3], [.3, 1.]], [[-1., .2], [.2, 3.]]])
    Q = np.array([[.4, .1], [.1, .2]])
    R = np.array([[0., .2], [-.2, 0.]])

    class Setups:
        nao = 2

        def __getitem__(self, a):
            return SimpleNamespace(
                dO_ii=Q, nabla_iiv=np.repeat(-R[:, :, None], 3, axis=2))

    setups = Setups()
    bfs = SimpleNamespace(
        my_atom_indices=[0, 1, 2], gamma=False,
        ibzk_qc=np.zeros((1, 3)), sdisp_xc=np.zeros((1, 3)),
        calculate_potential_matrices=lambda V: np.zeros((1, 2, 2)))
    kpts = [SimpleNamespace(s=s, q=0) for s in range(2)]
    wfs = SimpleNamespace(
        kpt_u=kpts, setups=setups, basis_functions=bfs, dtype=complex,
        nspins=2, P_aqMi={a: P[a][None] for a in range(3)},
        manytci=SimpleNamespace(P_aqMi=lambda *a, **kw: {
            a: np.repeat(dP[a][None, None], 3, axis=1)
            for a in range(3)}))
    dH = {a: np.array([pack_hermitian(h) for h in H]) for a in range(3)}
    sc = Supercell(bulk('Al', 'fcc', a=4.04), supercell=(3, 1, 1))
    parts = sc._calculate_supercell_entry(0, 0, np.zeros((2, 1, 1, 1)),
                                          {}, wfs, dH, components=True)
    step = 1e-5
    for s in range(2):
        pp, pm = P[1] + step * dP[1], P[1] - step * dP[1]
        finite = (pp.conj() @ H[s] @ pp.T
                  - pm.conj() @ H[s] @ pm.T) / (2 * step)
        assert parts['P'][s, 0] == pytest.approx(finite, abs=1e-9)
        assert parts['Q'][s, 0] == pytest.approx(dP[1].conj() @ Q @ P[1].T)
        assert parts['R'][s, 0] == pytest.approx(P[1].conj() @ R @ P[1].T)


@pytest.mark.elph
def test_paw_energy_gauge_and_hermiticity():
    """Full coupling is invariant to H -> H+cS, epsilon -> epsilon+c."""
    rng = np.random.default_rng(91)

    def z():
        return rng.normal(size=(3, 4, 4)) + 1j * rng.normal(size=(3, 4, 4))

    def adj(x):
        return x.swapaxes(-1, -2).conj()
    V, D, P, Q, R = [z() for _ in range(5)]
    kernels = dict(V=V + adj(V), D=D + adj(D), P=P + adj(P),
                   Q=Q, Qright=adj(Q), R=R - adj(R))
    eps = np.array([-.2, .1, .1, .8])
    g = ElectronPhononMatrix._assemble_paw(kernels, eps, eps)
    assert g['total'] == pytest.approx(adj(g['total']))
    shift = 2.7
    shifted = dict(kernels, P=kernels['P'] + shift * (Q + adj(Q)))
    gs = ElectronPhononMatrix._assemble_paw(shifted, eps + shift, eps + shift)
    assert gs['total'] == pytest.approx(g['total'], abs=1e-13)
    assert np.allclose(np.diagonal(g['R'], axis1=1, axis2=2), 0)


@pytest.mark.serial
@pytest.mark.elph
def test_al_local_kernel_against_complex_grid_integral(in_tmp_dir):
    """Check g(V) at a complex Bloch point, before Fourier averaging."""
    from gpaw.old.calculator import GPAW
    atoms = bulk('Al', 'fcc', a=4.04)
    supercell = (3, 1, 1)
    calc = GPAW(mode='lcao', basis='sz(dzp)', gpts=(24, 8, 8),
                kpts=[[.17, .11, .07]], symmetry='off', txt=None)
    calc.initialize(atoms * supercell)
    calc.initialize_positions(atoms * supercell)
    wfs = calc.wfs
    potential = np.random.default_rng(200).normal(size=tuple(wfs.gd.n_c))
    zeros = {a: np.zeros((1, setup.ni * (setup.ni + 1) // 2))
             for a, setup in enumerate(wfs.setups)}
    kernels = Supercell(atoms, supercell=supercell)._calculate_supercell_entry(
        0, 0, potential[None], {}, wfs, zeros, components=True)
    psi = wfs.gd.zeros(wfs.setups.nao, dtype=complex)
    wfs.basis_functions.lcao_to_grid(np.eye(wfs.setups.nao, dtype=complex),
                                     psi, 0)
    expected = np.array([[wfs.gd.integrate(p2.conj() * potential * p1)
                          for p1 in psi] for p2 in psi])
    assert kernels['V'][0, 0] == pytest.approx(expected, abs=1e-12)
