"""Decomposition of Kohn-Sham eigenvalues into expectation values.

For a converged plane-wave PAW calculation, every Kohn-Sham eigenvalue can
be written as a sum of expectation values of the individual terms of the
Hamiltonian::

    eps_n = <T>_n + <v_H>_n + <v_xc>_n + <v_bar>_n

Each term consists of a smooth ("pseudo") part evaluated with the pseudo
wave functions, and a one-center PAW correction evaluated with the
projections, e.g. for the kinetic energy::

                ~     1  2  ~      --  --  a*   a   a
    <T>_n   =  <ψ_n| --- ∇ |ψ_n> +  >   >  P   dK   P
                      2             --  --  ni   ij  nj
                                    a   ij

with ``dK_ij = <φ_i|T|φ_j> - <φ~_i|T|φ~_j>``.  The sum of the two parts is
the all-electron expectation value (within the frozen-core PAW
approximation).

The four terms are:

``kinetic``
    Kinetic energy.  PAW correction from ``setup.K_p``.
``hartree``
    Electrostatic potential.  Note that in the PAW formalism this is the
    potential from *all* charges: valence and core electrons as well as the
    nuclei (the latter two enter through the compensation charges and the
    one-center terms ``setup.M_p``, ``setup.M_pp`` and ``setup.Delta_pL``).
``xc``
    Semi-local exchange-correlation potential (plus Hubbard-U correction
    if present).  PAW correction from ``xc.calculate_paw_correction()``.
``zero``
    The PAW "zero potential" (local pseudo potential) ``v_bar``.  Its
    pseudo part and its PAW correction (``setup.MB_p``) cancel to within
    the completeness of the partial-wave basis, so the total should be
    close to zero.  It is reported as a sanity check.

The electrostatic term is additionally split into the potentials from
the three kinds of charges (``hartree = hartree_valence + hartree_core +
hartree_nuclear``):

``hartree_valence``
    Hartree potential of the valence electrons, ``v_H[n_v]``.
``hartree_core``
    Hartree potential of the (frozen) core electrons, ``v_H[n_c]``.
``hartree_nuclear``
    Potential from the nuclei, ``-sum_a Z_a / |r - R_a|``.

and ``hartree_electrons = hartree_valence + hartree_core`` is the Hartree
potential of all electrons.  Each of the three pieces is not neutral, so
the G=0 component of its pseudo potential is set to zero (uniform
neutralizing background); the pieces add up to the neutral total.

By default, the pseudo part of each electrostatic term is the potential
of the smooth density *plus* the corresponding compensation charges,
so that it has the correct multipole moments and the PAW correction is
a pure one-center term.  With ``compensation_charges=False``, the pseudo
part is the potential of the smooth density alone (for the valence term
this is the Hartree potential of the pseudo valence density, ``v_H[n~_v]``,
as used in norm-conserving pseudopotential codes).  The total is the same
in both cases; the ``paw`` part is then ``total - pseudo`` and also
contains the (not strictly one-center) compensation-charge contribution.
Note that the nuclei are represented by compensation charges only, so
``hartree_nuclear.pseudo`` is zero in that case.

This module is the per-state analogue of the splitting of the atomic
Hamiltonian corrections done in :func:`gpaw.new.pot_calc.
calculate_non_local_potential1`.

Example::

    from gpaw import GPAW
    from gpaw.new.pw.ks_decomposition import ks_decomposition

    calc = GPAW('si.gpw')  # must contain wave functions (mode='all')
    d = ks_decomposition(calc, n1=0, n2=8)
    print(d)                             # summary of all terms
    t_skn = d.kinetic.total              # all-electron <T> in eV
    vH_skn = d.hartree_valence.total     # all-electron <v_H[n_v]> in eV
    print(abs(d.residual_skn).max())     # should be ~1e-10 eV
    x = d.ks4()                          # symGW features (Hartree)
    x.x_sknf                             # [t, vh, vxc, eps - eps_VBM]

Only plane-wave mode with LDA/GGA functionals (including +U) is supported.

symGW features
--------------

symGW (N. Pant, V.-A. Ha, D. Kim and F. Giustino, Phys. Rev. B 114,
105112 (2026)) fits the GW self-energy ``Sigma_nk = E^GW_nk - eps_nk +
<v_xc>_nk`` to a polynomial in the "KS4" vector of the state::

    x_nk = [<T>_nk, <v_H>_nk, <v_xc>_nk, eps_nk - eps_VBM]

in Hartree units, where ``v_H`` is the Hartree potential of the valence
electrons with zero average over the unit cell, and ``eps_VBM`` is the
valence band maximum.  :meth:`KSDecomposition.ks4` returns these vectors.
The reference implementation (a patch for Quantum ESPRESSO) evaluates all
three expectation values with norm-conserving pseudo wave functions.  With
PAW there are two self-consistent choices, selected with the
``convention`` argument (see :data:`KS4_CONVENTIONS`):

``'ae'`` (default)
    All-electron (frozen-core) expectation values: ``kinetic.total``,
    ``hartree_valence.total`` and ``xc.total``.  These are well-defined
    physical quantities that do not depend on the pseudization, and
    ``xc.total`` is the ``<v_xc>`` used by GPAW's G0W0 code, so the
    self-energy defined above is consistent with the feature vector.
    Recommended when training on, or predicting, GPAW GW data.

``'pseudo'``
    Everything evaluated with the pseudo wave functions and the smooth
    potentials, like a norm-conserving pseudopotential code:
    ``kinetic.pseudo``, ``hartree_valence.pseudo`` (the Hartree potential
    of the pseudo valence density alone, which requires
    ``compensation_charges=False``) and ``xc.pseudo``.  This is formally the
    closest analogue of the features used in the symGW paper, but PAW
    pseudo quantities are *not* norm-conserving ones (different
    pseudization, and the PAW pseudo wave functions are not normalized),
    so equations fitted to norm-conserving data must be benchmarked before
    they are applied to these features.

Other combinations can be given as a dict, e.g. ``{'t': 'kinetic.total',
'vh': 'hartree_valence.pseudo', 'vxc': 'xc.total'}``.  Whatever the choice,
the same ``<v_xc>`` must be used in the feature vector and in the
quasiparticle energy ``E_nk = eps_nk - <v_xc>_nk + f(x_nk)``.

Charged and non-periodic systems
--------------------------------

For periodic systems each non-neutral piece of the electrostatic
potential is solved with a uniform neutralizing background (zero G=0
component), exactly as the SCF solver does for charged periodic cells.
For charged non-periodic systems (molecules with ``charge != 0``) GPAW
neutralizes the density with a Gaussian charge and subtracts its analytic
potential; here each piece gets its own Gaussian carrying the net charge
of that piece (the valence piece absorbs the tiny non-neutrality of the
SCF pseudo density), so that the pieces still add up to the SCF
potential.

Memory and parallelization
--------------------------

The bands are processed in blocks (``blocksize``) so that the real-space
wave functions of at most about ``DEFAULT_BLOCK_BYTES`` are held at a
time.  k-points are distributed as in the SCF calculation, every band
rank processes the bands it owns, and with domain parallelization each
block is gathered to the domain master.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from math import pi, prod, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Iterator, Sequence

import numpy as np
from ase.units import Ha

from gpaw.core import PWArray, UGArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.mpi import broadcast_float, serial_comm
from gpaw.new.c import add_to_density
from gpaw.new.pw.poisson import ChargedPWPoissonSolver
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.setup import Setup
from gpaw.typing import Array1D, Array3D
from gpaw.utilities import pack_density, unpack_hermitian

if TYPE_CHECKING:
    from gpaw.dft import DFT

TERM_NAMES = ['kinetic', 'hartree', 'xc', 'zero']
"""Terms that add up to the eigenvalue."""
HARTREE_PARTS = ['hartree_valence', 'hartree_core', 'hartree_nuclear']
"""Terms that add up to ``hartree``."""
ALL_TERM_NAMES = TERM_NAMES + HARTREE_PARTS

DEFAULT_BLOCK_BYTES = 256 * 1024**2
"""Default upper bound (per rank) on the memory for the real-space wave
functions of one block of bands, see
:meth:`KSDecompositionCalculator.calculate`."""


@dataclass
class Term:
    """Expectation values of one term of the Hamiltonian for all states.

    Arrays have shape ``(nspins, nkpts, nbands)`` and are in eV.
    """
    pseudo: Array3D
    """Smooth part: ``<ψ~_n|O~|ψ~_n>``."""
    paw: Array3D
    """PAW correction: ``sum_a P^a+ ΔO^a P^a`` (see module docstring for
    the electrostatic terms with ``compensation_charges=False``)."""

    @property
    def total(self) -> Array3D:
        """All-electron expectation value (``pseudo + paw``)."""
        return self.pseudo + self.paw

    def __add__(self, other: Term) -> Term:
        return Term(self.pseudo + other.pseudo, self.paw + other.paw)


@dataclass
class KSDecomposition:
    """Result of :func:`ks_decomposition`.

    All arrays have shape ``(nspins, nkpts, nbands)`` where ``nkpts`` is
    the number of requested IBZ k-points and ``nbands = n2 - n1``.  Units
    are eV.
    """
    eps_skn: Array3D
    """Kohn-Sham eigenvalues."""
    kinetic: Term
    hartree: Term
    xc: Term
    zero: Term
    hartree_valence: Term
    hartree_core: Term
    hartree_nuclear: Term
    n1: int
    n2: int
    kpt_indices: list[int]
    """IBZ k-point indices."""
    compensation_charges: bool = True
    e_vbm: float | None = None
    """Valence band maximum (highest occupied eigenvalue) in eV, or None
    if the calculation has no occupation numbers."""

    @property
    def hartree_electrons(self) -> Term:
        """Hartree potential of all (valence + core) electrons."""
        return self.hartree_valence + self.hartree_core

    @property
    def terms(self) -> dict[str, Term]:
        """The terms that add up to the eigenvalues."""
        return {name: getattr(self, name) for name in TERM_NAMES}

    @property
    def all_terms(self) -> dict[str, Term]:
        """All terms (including the electrostatic sub-terms)."""
        dct = {name: getattr(self, name) for name in ALL_TERM_NAMES}
        dct['hartree_electrons'] = self.hartree_electrons
        return dct

    @property
    def residual_skn(self) -> Array3D:
        """``eps_skn`` minus sum of all (total) terms.  Should be ~0."""
        return self.eps_skn - sum(term.total for term in self.terms.values())

    def todict(self) -> dict[str, np.ndarray]:
        """Flat dict of arrays (e.g. for ``np.savez``).

        Keys: ``eps``, ``kpt_indices``, ``bands`` (``[n1, n2]``),
        ``compensation_charges`` and ``<term>_pseudo``, ``<term>_paw``,
        ``<term>_total`` for each term.
        """
        dct = {'eps': self.eps_skn,
               'kpt_indices': np.array(self.kpt_indices),
               'bands': np.array([self.n1, self.n2]),
               'compensation_charges': np.array(self.compensation_charges),
               'e_vbm': np.array(np.nan if self.e_vbm is None
                                 else self.e_vbm)}
        for name, term in self.all_terms.items():
            dct[f'{name}_pseudo'] = term.pseudo
            dct[f'{name}_paw'] = term.paw
            dct[f'{name}_total'] = term.total
        return dct

    def ks4(self,
            convention: str | dict[str, str] = 'ae',
            e_vbm: float | None = None) -> KS4:
        """symGW feature vectors ``[t, vh, vxc, eps - eps_VBM]``.

        See the section "symGW features" of the module docstring.

        Parameters
        ----------
        convention:
            ``'ae'`` (all-electron expectation values, default),
            ``'pseudo'`` (pseudo wave functions and smooth potentials,
            like a norm-conserving pseudopotential code; requires
            ``compensation_charges=False``) or a dict mapping ``'t'``,
            ``'vh'`` and ``'vxc'`` to ``'<term>.<part>'`` strings, e.g.
            ``{'t': 'kinetic.total', 'vh': 'hartree_valence.pseudo',
            'vxc': 'xc.total'}``.  See :data:`KS4_CONVENTIONS`.
        e_vbm:
            Energy reference in eV.  Default: the valence band maximum
            (highest occupied eigenvalue) of the calculation.

        Returns a :class:`KS4` object with all quantities in Hartree.
        """
        if isinstance(convention, str):
            if convention not in KS4_CONVENTIONS:
                raise ValueError(f'Unknown convention: {convention!r}.  '
                                 f'Use one of {list(KS4_CONVENTIONS)} '
                                 'or a dict.')
            if convention == 'pseudo' and self.compensation_charges:
                raise ValueError(
                    "The 'pseudo' convention needs the Hartree potential "
                    'of the pseudo valence density without compensation '
                    'charges: use compensation_charges=False.')
            fields = KS4_CONVENTIONS[convention]
        else:
            fields = dict(convention)
        if set(fields) != {'t', 'vh', 'vxc'}:
            raise ValueError("Convention must specify 't', 'vh' and 'vxc', "
                             f'got {sorted(fields)}')
        all_terms = self.all_terms
        x_skn = {}
        for key, spec in fields.items():
            name, _, part = spec.partition('.')
            if name not in all_terms or part not in ['pseudo', 'paw',
                                                     'total']:
                raise ValueError(f'Bad specification for {key!r}: {spec!r} '
                                 f'(expected <term>.<part> with term in '
                                 f'{list(all_terms)} and part in '
                                 '[pseudo, paw, total])')
            x_skn[key] = getattr(all_terms[name], part) / Ha
        if e_vbm is None:
            e_vbm = self.e_vbm
            if e_vbm is None:
                raise ValueError('The calculation has no occupation numbers '
                                 'so the valence band maximum is unknown: '
                                 'give e_vbm explicitly.')
        return KS4(t_skn=x_skn['t'],
                   vh_skn=x_skn['vh'],
                   vxc_skn=x_skn['vxc'],
                   eps_skn=(self.eps_skn - e_vbm) / Ha,
                   e_vbm=e_vbm / Ha,
                   convention=fields,
                   kpt_indices=self.kpt_indices,
                   n1=self.n1,
                   n2=self.n2)

    def __str__(self) -> str:
        lines = [f'Kohn-Sham eigenvalue decomposition '
                 f'(bands {self.n1}-{self.n2 - 1}, '
                 f'{len(self.kpt_indices)} k-points, '
                 f'compensation_charges={self.compensation_charges}) [eV]:']
        for name, term in self.all_terms.items():
            lines.append(f'  {name:17}: '
                         f'pseudo in [{term.pseudo.min():9.3f}, '
                         f'{term.pseudo.max():9.3f}], '
                         f'paw in [{term.paw.min():9.3f}, '
                         f'{term.paw.max():9.3f}]')
        lines.append(f'  max |residual|: {abs(self.residual_skn).max():.2e}')
        return '\n'.join(lines)


KS4_CONVENTIONS: dict[str, dict[str, str]] = {
    'ae': {'t': 'kinetic.total',
           'vh': 'hartree_valence.total',
           'vxc': 'xc.total'},
    'pseudo': {'t': 'kinetic.pseudo',
               'vh': 'hartree_valence.pseudo',
               'vxc': 'xc.pseudo'}}
"""Named choices of expectation values for the symGW KS4 vector.
See the module docstring (section "symGW features")."""


@dataclass
class KS4:
    """symGW feature vectors ``x_nk = [t, vh, vxc, eps - eps_VBM]``.

    Result of :meth:`KSDecomposition.ks4`.  All arrays have shape
    ``(nspins, nkpts, nbands)`` and all energies are in **Hartree**
    (the unit used by the symGW equations).
    """
    t_skn: Array3D
    """Kinetic energy ``<T>``."""
    vh_skn: Array3D
    """Hartree potential of the valence electrons ``<v_H[n_v]>`` (zero
    cell average for periodic systems)."""
    vxc_skn: Array3D
    """xc-potential ``<v_xc>``."""
    eps_skn: Array3D
    """Kohn-Sham eigenvalue relative to the valence band maximum."""
    e_vbm: float
    """Energy reference that was subtracted from the eigenvalues."""
    convention: dict[str, str]
    """Which expectation values were used (``'<term>.<part>'``)."""
    kpt_indices: list[int]
    n1: int
    n2: int

    @property
    def x_sknf(self) -> np.ndarray:
        """Feature vectors; last axis is ``[t, vh, vxc, eps]``."""
        return np.stack([self.t_skn, self.vh_skn, self.vxc_skn,
                         self.eps_skn], axis=-1)

    def todict(self) -> dict[str, np.ndarray]:
        """Flat dict of arrays (e.g. for ``np.savez``)."""
        return {'t': self.t_skn,
                'vh': self.vh_skn,
                'vxc': self.vxc_skn,
                'eps': self.eps_skn,
                'e_vbm': np.array(self.e_vbm),
                'kpt_indices': np.array(self.kpt_indices),
                'bands': np.array([self.n1, self.n2]),
                'convention': np.array([self.convention[key]
                                        for key in ['t', 'vh', 'vxc']])}


def is_paw_setup(setup) -> bool:
    """True for real PAW setups (with partial waves and core density)."""
    lc = getattr(setup, 'local_corr', None)
    return lc is not None and getattr(lc, 'n_qg', None) is not None


def coulomb_correction_parts(setup: Setup) -> tuple[Array1D, Array1D]:
    """Split ``setup.M_p`` into core and nuclear parts.

    ``setup.M_p`` is the linear (in the atomic density matrix) one-center
    Coulomb correction.  It describes the interaction of the valence
    partial-wave densities with the core electrons and the nucleus
    (there is no valence-valence contribution, that is in ``M_pp``).
    Returns ``(M_core_p, M_nuc_p)`` with ``M_core_p + M_nuc_p == M_p``.

    Follows the conventions of :meth:`gpaw.setup.Setup.
    calculate_coulomb_corrections`.
    """
    if not is_paw_setup(setup):
        # All-electron H or norm-conserving pseudopotential:
        # nothing to split, all of M_p is "ion".
        M_p = np.asarray(setup.M_p, dtype=float)
        return np.zeros_like(M_p), M_p.copy()

    lc = setup.local_corr
    rgd = lc.rgd2
    r_g = rgd.r_g
    dr_g = rgd.dr_g
    rdr_g = r_g * dr_g

    def H(n_g):
        return rgd.poisson(n_g, 0) * rdr_g

    n_qg = lc.n_qg
    nt_qg = lc.nt_qg
    nc_g = lc.nc_g
    nct_g = lc.nct_g
    g0_g = setup.g_lg[0]
    wg0_g = H(g0_g)
    wn_qg = np.array([H(n_g) for n_g in n_qg])
    wnt_qg = np.array([H(nt_g) for nt_g in nt_qg])
    wnc_g = H(nc_g)
    wnct_g = H(nct_g)
    Delta_q = lc.Delta_lq[0]
    T_qp = lc.T_Lqp[0]

    # Split of the l=0 compensation charge coefficient,
    # Delta0 = int (n_c - n~_c) r^2 dr - Z / sqrt(4 pi):
    Delta0_nuc = -setup.Z / sqrt(4 * pi)
    Delta0_core = setup.Delta0 - Delta0_nuc

    # Nucleus: point charge in the all-electron part, Gaussian
    # compensation charge in the pseudo part:
    A_q = -sqrt(4 * pi) * setup.Z * (n_qg @ rdr_g)
    A_q -= 0.5 * Delta0_nuc * (wnt_qg @ g0_g + nt_qg @ wg0_g)
    A_q -= Delta0_nuc * (g0_g @ wg0_g) * Delta_q
    M_nuc_p = A_q @ T_qp

    # Core electrons: n_c in the all-electron part, n~_c plus
    # compensation charge in the pseudo part:
    mc_g = nct_g + Delta0_core * g0_g
    wmc_g = wnct_g + Delta0_core * wg0_g
    A_q = 0.5 * (wn_qg @ nc_g + n_qg @ wnc_g)
    A_q -= 0.5 * (wnt_qg @ mc_g + nt_qg @ wmc_g)
    A_q -= 0.5 * (mc_g @ wg0_g + g0_g @ wmc_g) * Delta_q
    M_core_p = A_q @ T_qp

    return M_core_p, M_nuc_p


class KSDecompositionCalculator:
    """Calculate per-state kinetic, Hartree, xc and zero-potential terms.

    The potentials and one-center matrices are set up once in the
    constructor; :meth:`calculate` can then be called for different bands
    and k-points.  Works with k-point, band and domain parallelization.
    See the module docstring for the meaning of ``compensation_charges``.
    """

    def __init__(self, dft: DFT, compensation_charges: bool = True):
        self.dft = dft
        self.ibzwfs = dft.ibzwfs
        self.density = dft.density
        self.potential = dft.potential
        self.pot_calc = dft.pot_calc
        self.setups = dft.setups
        self.compensation_charges = compensation_charges

        self._check_supported()

        self.ibzwfs.make_sure_wfs_are_read_from_gpw_file()
        if not self.ibzwfs.has_wave_functions():
            raise ValueError(
                'No wave functions available.  The gpw-file must be '
                "written with mode='all'.")

        grid = self.density.nt_sR.desc
        self.domain_comm = grid.comm
        self.grid0 = grid.new(comm=None)  # serial coarse grid

        # Pseudo potentials on the (serial) coarse grid.  Only available
        # on the master of the domain communicator (empty dicts
        # elsewhere).  Electrostatic potentials without compensation
        # charges go into self.vbare_xR (only used if
        # compensation_charges is False):
        self.vt_xR: dict[str, UGArray] = {}
        self.vbare_xR: dict[str, UGArray] = {}
        # Integrals of compensation charges times electrostatic
        # potentials (all atoms, all ranks):
        self.V_xaL: dict[str, AtomArrays] = {}
        self._pseudo_potentials()

        # One-center corrections for all atoms (on all ranks):
        self.dX_axii: dict[str, dict[int, np.ndarray]] = (
            self._atomic_corrections())

        # Valence band maximum (highest occupied eigenvalue, eV) used as
        # the energy reference of the symGW features:
        self.e_vbm: float | None
        try:
            homo = self.ibzwfs.get_homo_lumo()[0]
        except ValueError:  # no occupation numbers
            self.e_vbm = None
        else:
            self.e_vbm = float(homo * Ha)

    @classmethod
    def from_gpw_file(cls,
                      filename: str | Path,
                      comm=None,
                      **kwargs) -> KSDecompositionCalculator:
        from gpaw import GPAW
        calc = GPAW(filename, legacy_gpaw=False, communicator=comm, txt=None)
        return cls(calc.dft, **kwargs)

    def _check_supported(self) -> None:
        if not isinstance(self.ibzwfs, PWFDIBZWaveFunctions):
            raise NotImplementedError('Only PW mode is supported')
        if not isinstance(self.potential.vHt_x, PWArray):
            raise NotImplementedError('Only PW mode is supported')
        if self.ibzwfs.ncomponents == 4:
            raise NotImplementedError('Non-collinear calculations '
                                      'are not supported')
        xc = self.pot_calc.xc
        if xc.type == 'MGGA' or self.potential.dedtaut_sR is not None:
            raise NotImplementedError('MGGA functionals are not supported')
        if xc.exx_fraction != 0.0:
            raise NotImplementedError('Hybrid functionals are not supported')
        if self.pot_calc.extensions:
            raise NotImplementedError('External potentials/extensions '
                                      'are not supported')
        if self.pot_calc.soc:
            raise NotImplementedError('Spin-orbit coupling is not supported')
        if type(self.pot_calc.poisson_solver).__name__.startswith('Bloechl'):
            raise NotImplementedError(
                "The fast (Bloechl) Poisson solver (poissonsolver={'fast': "
                'True}) is not supported')

    def _to_coarse_grid(self, v_h: PWArray) -> UGArray | None:
        """Map electrostatic potential to the (serial) coarse grid.

        The potential is first mapped onto the plane-wave sphere of the
        density (the SCF potential only contains those components).
        Returns None except on the domain master.
        """
        v_h0 = v_h.gather()
        if self.domain_comm.rank != 0:
            return None
        assert v_h0 is not None
        v_r = v_h0.ifft(grid=v_h0.desc.minimal_uniform_grid())
        v_g = v_r.fft(pw=self.pot_calc.pw.new(comm=None))
        return v_g.ifft(grid=self.grid0)

    def _solve_poisson(self,
                       nt_R: UGArray | None,
                       Q_aL: AtomArrays,
                       charge: float | None = None
                       ) -> tuple[PWArray, AtomArrays, float]:
        """Electrostatic potential from pseudo density + comp. charges.

        Uses the same Poisson solver as the SCF calculation (see
        :meth:`_component_poisson_solver` for charged, non-periodic
        systems, where ``charge`` can be given to override the net charge
        that is neutralized).  Returns the potential (plane-wave
        coefficients), the integrals of the compensation charges times
        the potential (all atoms, all ranks) and the net charge that was
        neutralized (0.0 for periodic systems).
        """
        ps = self.pot_calc.poisson_solver
        nt0_R = nt_R.gather() if nt_R is not None else None
        if self.domain_comm.rank == 0:
            nt0_g = ps.pwg0.zeros()
            if nt0_R is not None:
                nt0_R.fft(out=nt0_g)
            vt0_g = ps.pwg0.zeros()  # dummy
        else:
            nt0_g = None
            vt0_g = None
        ps, charge = self._component_poisson_solver(nt0_g, Q_aL, charge)
        _, v_h, V_aL = ps.solve(nt0_g, Q_aL, vt0_g, None)
        return v_h, V_aL.gather(broadcast=True), charge

    def _component_poisson_solver(self,
                                  nt0_g: PWArray | None,
                                  Q_aL: AtomArrays,
                                  charge: float | None = None):
        """PAW Poisson solver for one (in general non-neutral) component.

        The valence, core and nuclear pieces of the charge density are
        not neutral.  For periodic systems the Poisson solver simply
        drops the G=0 component (uniform neutralizing background), which
        is linear in the density, so the SCF solver can be used for the
        pieces as well.  For charged, non-periodic systems, GPAW's
        :class:`~gpaw.new.pw.poisson.ChargedPWPoissonSolver` neutralizes
        the density with a Gaussian charge, solves, and subtracts the
        analytic potential of the Gaussian again.  This is linear too,
        but only if the Gaussian carries the net charge of the density
        that is actually being solved for, so each piece gets its own
        solver with its own net charge (same width ``alpha``; by default
        the net charge of the density, or ``charge`` if given).  The
        pieces then add up to the SCF potential exactly.

        Returns the solver and the net charge (0.0 if the SCF solver is
        used as is).
        """
        ps = self.pot_calc.poisson_solver
        inner = ps.poisson_solver
        if not isinstance(inner, ChargedPWPoissonSolver):
            return ps, 0.0

        if charge is None:
            # Net charge in GPAW's convention (number of missing
            # electrons, i.e. minus the integral of the electronic +
            # compensation charge density):
            rhot_h = ps.pwh.zeros()
            ps.ghat_aLh.add_to(rhot_h, Q_aL)
            charge = -rhot_h.integrate()
            if nt0_g is not None:
                charge -= nt0_g.integrate()
            charge = broadcast_float(float(charge), self.domain_comm)

        inner_x = ChargedPWPoissonSolver(inner.pw,
                                         self.pot_calc.fine_grid,
                                         charge,
                                         inner.strength,
                                         alpha=inner.alpha)
        ps_x = copy.copy(ps)
        ps_x.poisson_solver = inner_x
        return ps_x, charge

    def _pseudo_potentials(self) -> None:
        """Split the pseudo potential into v_bar, v_H and v_xc.

        The three potentials are calculated in exactly the same way as
        in :meth:`gpaw.new.pw.pot_calc.PlaneWavePotentialCalculator.
        calculate_pseudo_potential` so that their sum equals the
        potential used in the SCF calculation (``potential.vt_sR``).
        The electrostatic potential is further split into valence, core
        and nuclear parts.
        """
        pot_calc = self.pot_calc
        density = self.density
        setups = self.setups
        master = self.domain_comm.rank == 0

        # XC-potential (calculated on the fine grid, restricted to
        # the coarse grid):
        nt_sr = pot_calc.interpolate(density.nt_sR)
        _, vxct_sr, _ = pot_calc.xc.calculate(nt_sr)
        vxct_sR = pot_calc.restrict(vxct_sr).gather()
        if master:
            self.vt_xR['xc'] = vxct_sR
            self.vt_xR['zero'] = pot_calc.vbar0_g.ifft(grid=self.grid0)

        # Full electrostatic potential from the SCF calculation:
        vHt_h = self.potential.vHt_x
        ghat_aLh = setups.create_compensation_charges(
            vHt_h.desc, self.dft.relpos_ac, density.D_asii.layout.atomdist)
        self.V_xaL['hartree'] = ghat_aLh.integrate(vHt_h).gather(
            broadcast=True)
        vHt_R = self._to_coarse_grid(vHt_h)
        if master:
            assert vHt_R is not None
            self.vt_xR['hartree'] = vHt_R

        # Split compensation charges into valence, core and nuclear
        # parts (Q_L = Delta_pL D_p + Delta0 delta_L0 with
        # Delta0 = int (n_c - n~_c) r^2 dr - Z / sqrt(4 pi)):
        Q_aL = density.calculate_compensation_charge_coefficients()
        Q_xaL = {name: Q_aL.new(data=np.zeros_like(Q_aL.data))
                 for name in HARTREE_PARTS}
        for a, Q_L in Q_aL.items():
            setup = setups[a]
            delta0 = density.delta0_a[a]
            if is_paw_setup(setup):
                Q_nuc = -setup.Z / sqrt(4 * pi)
            else:
                Q_nuc = delta0  # all of it is "ion"
            Q_xaL['hartree_nuclear'][a][0] = Q_nuc
            Q_xaL['hartree_core'][a][0] = delta0 - Q_nuc
            Q_xaL['hartree_valence'][a][:] = Q_L
            Q_xaL['hartree_valence'][a][0] -= delta0

        # Smooth densities: valence, core, nothing.  Note that
        # density.nct_R is the smooth core density *per spin channel*
        # (n~_c / ndensities, see gpaw.new.density.Density.nct_R), so the
        # total smooth core density is ndensities * nct_R:
        ndensities = density.ndensities
        nct_R = density.nct_R.new(data=ndensities * density.nct_R.data)
        nt_R = nct_R.new(data=density.nt_sR.data[:ndensities].sum(axis=0))
        nvt_R = nct_R.new(data=nt_R.data - nct_R.data)
        nt_xR: dict[str, UGArray | None] = {'hartree_valence': nvt_R,
                                            'hartree_core': nct_R,
                                            'hartree_nuclear': None}
        Q0_aL = Q_aL.new(data=np.zeros_like(Q_aL.data))

        # For charged non-periodic systems, the Gaussian neutralizing
        # charges of the three pieces must add up to the one of the SCF
        # solver in order to reproduce the SCF potential: the SCF pseudo
        # density is neutral only to within the SCF tolerance and the
        # G=0 residual is silently dropped by the solver.  The valence
        # piece is therefore done last and gets whatever is left of the
        # SCF charge (see _component_poisson_solver):
        inner = pot_calc.poisson_solver.poisson_solver
        remaining = (inner.charge
                     if isinstance(inner, ChargedPWPoissonSolver) else None)

        for name in ['hartree_core', 'hartree_nuclear', 'hartree_valence']:
            charge = None
            if remaining is not None and name == 'hartree_valence':
                charge = remaining
            v_h, V_aL, charge = self._solve_poisson(nt_xR[name],
                                                    Q_xaL[name],
                                                    charge)
            if remaining is not None:
                remaining -= charge
            self.V_xaL[name] = V_aL
            v_R = self._to_coarse_grid(v_h)
            if master:
                assert v_R is not None
                self.vt_xR[name] = v_R
            if not self.compensation_charges:
                v_h, _, _ = self._solve_poisson(nt_xR[name], Q0_aL)
                v_R = self._to_coarse_grid(v_h)
                if master:
                    assert v_R is not None
                    self.vbare_xR[name] = v_R
        if not self.compensation_charges:
            v_h, _, _ = self._solve_poisson(nt_R, Q0_aL)
            v_R = self._to_coarse_grid(v_h)
            if master:
                assert v_R is not None
                self.vbare_xR['hartree'] = v_R

    def _atomic_corrections(self) -> dict[str, dict[int, np.ndarray]]:
        """One-center PAW corrections, split like in
        :func:`gpaw.new.pot_calc.calculate_non_local_potential1`."""
        density = self.density
        setups = self.setups
        xc = self.pot_calc.xc
        V_xaL = self.V_xaL

        D_asii = density.D_asii.to_xp(np).gather(broadcast=True)

        dX_axii: dict[str, dict[int, np.ndarray]] = {
            name: {} for name in ALL_TERM_NAMES}
        ndensities = density.ndensities
        for a, D_sii in D_asii.items():
            setup = setups[a]
            D_sp = np.array([pack_density(D_ii.real) for D_ii in D_sii])
            D_p = D_sp[:ndensities].sum(0)
            M_core_p, M_nuc_p = coulomb_correction_parts(setup)
            dX_axii['kinetic'][a] = unpack_hermitian(setup.K_p)
            dX_axii['zero'][a] = unpack_hermitian(setup.MB_p)
            dX_axii['hartree'][a] = unpack_hermitian(
                setup.M_p + 2.0 * setup.M_pp @ D_p +
                setup.Delta_pL @ V_xaL['hartree'][a])
            dX_axii['hartree_valence'][a] = unpack_hermitian(
                2.0 * setup.M_pp @ D_p +
                setup.Delta_pL @ V_xaL['hartree_valence'][a])
            dX_axii['hartree_core'][a] = unpack_hermitian(
                M_core_p + setup.Delta_pL @ V_xaL['hartree_core'][a])
            dX_axii['hartree_nuclear'][a] = unpack_hermitian(
                M_nuc_p + setup.Delta_pL @ V_xaL['hartree_nuclear'][a])
            dxc_sp = np.zeros_like(D_sp)
            xc.calculate_paw_correction(setup, D_sp, dxc_sp, a=a)
            dxc_sii = unpack_hermitian(dxc_sp)
            if setup.hubbard_u is not None:
                _, dHU_sii = setup.hubbard_u.calculate(setup, D_sii)
                dxc_sii += dHU_sii
            dX_axii['xc'][a] = dxc_sii

        return dX_axii

    def calculate(self,
                  n1: int = 0,
                  n2: int = 0,
                  kpt_indices: Sequence[int] | None = None,
                  blocksize: int | None = None
                  ) -> KSDecomposition:
        """Calculate decomposition for bands ``n1:n2`` and IBZ k-points.

        ``n2 <= 0`` counts from the total number of bands.  All IBZ
        k-points are used if ``kpt_indices`` is ``None``.

        The bands are processed in blocks of ``blocksize`` bands so that
        the memory needed for the real-space wave functions stays
        bounded.  The default block size is chosen so that one block
        uses at most about ``DEFAULT_BLOCK_BYTES`` per rank.  With band
        parallelization every band rank processes the bands it owns
        (blocks never cross band ranks), and with domain parallelization
        each block is gathered to the domain master.  The result does
        not depend on ``blocksize``.
        """
        ibzwfs = self.ibzwfs
        n2 = n2 if n2 > 0 else ibzwfs.nbands + n2
        if not 0 <= n1 < n2 <= ibzwfs.nbands:
            raise ValueError(f'Bad band range: {n1}:{n2} '
                             f'(nbands={ibzwfs.nbands})')
        if kpt_indices is None:
            kpt_indices = range(len(ibzwfs.ibz))
        kpt_indices = list(kpt_indices)
        if blocksize is None:
            bytes_per_band = 16 * prod(self.grid0.size_c)
            blocksize = max(1, DEFAULT_BLOCK_BYTES // bytes_per_band)
        if blocksize < 1:
            raise ValueError(f'Bad blocksize: {blocksize}')

        shape = (ibzwfs.nspins, len(kpt_indices), n2 - n1)
        names = ['eps'] + [f'{name}_{part}'
                           for name in ALL_TERM_NAMES
                           for part in ['pseudo', 'paw']]
        x_skn = {name: np.zeros(shape) for name in names}

        for i, k in enumerate(kpt_indices):
            for spin in range(ibzwfs.nspins):
                if ibzwfs.rank_ks[k, spin] != ibzwfs.kpt_comm.rank:
                    continue
                wfs = ibzwfs._get_wfs(k, spin)
                for b1, b2, wfs_b in self._blocks(wfs, n1, n2, blocksize):
                    for name, x_n in self._one_kpt(wfs_b).items():
                        x_skn[name][spin, i, b1 - n1:b2 - n1] = x_n

        for x in x_skn.values():
            ibzwfs.comm.sum(x)

        return KSDecomposition(
            eps_skn=x_skn['eps'] * Ha,
            n1=n1,
            n2=n2,
            kpt_indices=kpt_indices,
            compensation_charges=self.compensation_charges,
            e_vbm=self.e_vbm,
            **{name: Term(pseudo=x_skn[f'{name}_pseudo'] * Ha,
                          paw=x_skn[f'{name}_paw'] * Ha)
               for name in ALL_TERM_NAMES})

    def _blocks(self,
                wfs: PWFDWaveFunctions,
                n1: int,
                n2: int,
                blocksize: int
                ) -> Iterator[tuple[int, int, PWFDWaveFunctions]]:
        """Serial wave-function objects for blocks of bands.

        Yields ``(b1, b2, wfs_b)`` for the blocks of at most
        ``blocksize`` bands in ``n1:n2`` that are owned by this band
        rank, with ``wfs_b`` collected to the domain master (nothing is
        yielded on the other domain ranks).  This is
        :meth:`gpaw.new.pwfd.wave_functions.PWFDWaveFunctions.
        collect_bands_and_domain` for one block, except that the bands
        stay on the band rank that owns them instead of being sent to
        the band master.
        """
        domain_comm = wfs.psit_nX.desc.comm
        a = max(n1, wfs.n1)
        b = min(n2, wfs.n2)
        pt_aiX = None  # reuse projectors for all blocks of this k-point
        for b1 in range(a, b, blocksize):
            b2 = min(b1 + blocksize, b)
            psit_nX = wfs.psit_nX.desc.empty(
                (b2 - b1, *wfs.psit_nX.dims[1:]), xp=wfs.psit_nX.xp)
            psit_nX.data[:] = wfs.psit_nX.data[b1 - wfs.n1:b2 - wfs.n1]
            atomdist = wfs.atomdist
            if domain_comm.size > 1:
                psit_nX = psit_nX.gather()
                atomdist = atomdist.gather()
                if domain_comm.rank != 0:
                    continue
            wfs_b = PWFDWaveFunctions.from_wfs(wfs,
                                               psit_nX,
                                               atomdist=atomdist,
                                               domain_band_comm=serial_comm)
            wfs_b.eig_n = wfs.eig_n[b1:b2]
            wfs_b._pt_aiX = pt_aiX
            yield b1, b2, wfs_b
            pt_aiX = wfs_b.pt_aiX

    def _one_kpt(self, wfs: PWFDWaveFunctions) -> dict[str, np.ndarray]:
        """Expectation values for one k-point and spin (units: Ha).

        ``wfs`` must be serial (bands and domain collected).
        """
        assert self.vt_xR, 'Potentials only available on domain master'
        spin = wfs.spin
        psit_nG = wfs.psit_nX
        assert isinstance(psit_nG, PWArray)
        P_ani = wfs.P_ani
        nbands = len(psit_nG)

        result = {'eps': np.array(wfs.eig_n, dtype=float),
                  'kinetic_pseudo': psit_nG.norm2('kinetic')}

        # Pseudo densities of individual states on the coarse grid:
        ut_nR = self.grid0.new(dtype=psit_nG.desc.dtype).empty(nbands)
        psit_nG.ifft(out=ut_nR, periodic=False)
        nt_R = self.grid0.empty()
        pseudo_xn = {name: np.zeros(nbands)
                     for name in ALL_TERM_NAMES if name != 'kinetic'}
        bare_xn = {name: np.zeros(nbands) for name in self.vbare_xR}
        for n, ut_R in enumerate(ut_nR.data):
            nt_R.data[:] = 0.0
            add_to_density(1.0, ut_R, nt_R.data)
            for name, x_n in pseudo_xn.items():
                v_xR = self.vt_xR[name]
                v_R = v_xR[spin] if v_xR.dims else v_xR
                x_n[n] = v_R.integrate(nt_R)
            for name, x_n in bare_xn.items():
                x_n[n] = self.vbare_xR[name].integrate(nt_R)
        for name, x_n in pseudo_xn.items():
            result[f'{name}_pseudo'] = x_n

        # One-center PAW corrections:
        for name, dX_aii in self.dX_axii.items():
            paw_n = np.zeros(nbands)
            for a, dX_ii in dX_aii.items():
                if dX_ii.ndim == 3:
                    dX_ii = dX_ii[spin]
                paw_n += expectation_values(P_ani[a], dX_ii)
            result[f'{name}_paw'] = paw_n

        # Without compensation charges in the pseudo part, their
        # contribution is moved to the "paw" part:
        for name, x_n in bare_xn.items():
            result[f'{name}_paw'] += result[f'{name}_pseudo'] - x_n
            result[f'{name}_pseudo'] = x_n

        return result


def expectation_values(P_ni: np.ndarray, M_ii: np.ndarray) -> np.ndarray:
    """Calculate ``sum_ij P_ni^* M_ij P_nj`` for all n."""
    return np.einsum('ni, ij, nj -> n', P_ni.conj(), M_ii, P_ni).real


def ks_decomposition(calc,
                     n1: int = 0,
                     n2: int = 0,
                     kpt_indices: Sequence[int] | None = None,
                     compensation_charges: bool = True,
                     blocksize: int | None = None
                     ) -> KSDecomposition:
    """Decompose Kohn-Sham eigenvalues into kinetic, Hartree, xc and
    zero-potential expectation values.

    Parameters
    ----------
    calc:
        GPAW calculator, :class:`gpaw.dft.DFT` object or name of a
        gpw-file (must contain wave functions, i.e. written with
        ``mode='all'``).
    n1, n2:
        Band range.  ``n2 <= 0`` counts from the total number of bands.
    kpt_indices:
        IBZ k-point indices.  Default is all.
    compensation_charges:
        Include compensation charges in the pseudo part of the
        electrostatic terms (default).  See module docstring.
    blocksize:
        Number of bands processed at a time (default: as many as fit
        in ``DEFAULT_BLOCK_BYTES``).  Does not affect the result.

    See :class:`KSDecomposition` for a description of the result and
    the module docstring for details of the decomposition.
    """
    if isinstance(calc, (str, Path)):
        kdc = KSDecompositionCalculator.from_gpw_file(
            calc, compensation_charges=compensation_charges)
    elif hasattr(calc, 'ibzwfs'):  # DFT object
        kdc = KSDecompositionCalculator(calc, compensation_charges)
    elif hasattr(calc, 'dft'):  # new-style ASE calculator
        kdc = KSDecompositionCalculator(calc.dft, compensation_charges)
    else:
        raise TypeError(
            'Expected a gpw-filename, a gpaw.dft.DFT object or a '
            'new-style GPAW calculator (legacy_gpaw=False), '
            f'got {type(calc)}')
    return kdc.calculate(n1, n2, kpt_indices, blocksize)
