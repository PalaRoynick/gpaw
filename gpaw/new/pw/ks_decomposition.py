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

This module is the per-state analogue of the splitting of the atomic
Hamiltonian corrections done in :func:`gpaw.new.pot_calc.
calculate_non_local_potential1`.

Example::

    from gpaw import GPAW
    from gpaw.new.pw.ks_decomposition import ks_decomposition

    calc = GPAW('si.gpw')  # must contain wave functions (mode='all')
    d = ks_decomposition(calc, n1=0, n2=8)
    t_skn = d.kinetic.total        # all-electron <T> in eV
    vH_skn = d.hartree.pseudo      # pseudo <v_H> in eV
    print(abs(d.residual_skn).max())  # should be ~1e-10 eV

Only plane-wave mode with LDA/GGA functionals (including +U) is supported.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
from ase.units import Ha

from gpaw.core import PWArray, UGArray
from gpaw.core.atom_arrays import AtomArrays
from gpaw.new.c import add_to_density
from gpaw.new.pwfd.ibzwfs import PWFDIBZWaveFunctions
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array3D
from gpaw.utilities import pack_density, unpack_hermitian

if TYPE_CHECKING:
    from gpaw.dft import DFT

TERM_NAMES = ['kinetic', 'hartree', 'xc', 'zero']


@dataclass
class Term:
    """Expectation values of one term of the Hamiltonian for all states.

    Arrays have shape ``(nspins, nkpts, nbands)`` and are in eV.
    """
    pseudo: Array3D
    """Smooth part: ``<ψ~_n|O~|ψ~_n>``."""
    paw: Array3D
    """One-center PAW correction: ``sum_a P^a+ ΔO^a P^a``."""

    @property
    def total(self) -> Array3D:
        """All-electron expectation value (``pseudo + paw``)."""
        return self.pseudo + self.paw


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
    n1: int
    n2: int
    kpt_indices: list[int]
    """IBZ k-point indices."""

    @property
    def terms(self) -> dict[str, Term]:
        return {name: getattr(self, name) for name in TERM_NAMES}

    @property
    def residual_skn(self) -> Array3D:
        """``eps_skn`` minus sum of all (total) terms.  Should be ~0."""
        return self.eps_skn - sum(term.total for term in self.terms.values())

    def todict(self) -> dict[str, np.ndarray]:
        """Flat dict of arrays (e.g. for ``np.savez``).

        Keys: ``eps``, ``kpt_indices``, ``bands`` (``[n1, n2]``) and
        ``<term>_pseudo``, ``<term>_paw``, ``<term>_total`` for each
        term.
        """
        dct = {'eps': self.eps_skn,
               'kpt_indices': np.array(self.kpt_indices),
               'bands': np.array([self.n1, self.n2])}
        for name, term in self.terms.items():
            dct[f'{name}_pseudo'] = term.pseudo
            dct[f'{name}_paw'] = term.paw
            dct[f'{name}_total'] = term.total
        return dct

    def __str__(self) -> str:
        lines = [f'Kohn-Sham eigenvalue decomposition '
                 f'(bands {self.n1}-{self.n2 - 1}, '
                 f'{len(self.kpt_indices)} k-points) [eV]:']
        for name, term in self.terms.items():
            lines.append(f'  {name:8}: pseudo in [{term.pseudo.min():9.3f}, '
                         f'{term.pseudo.max():9.3f}], '
                         f'paw in [{term.paw.min():9.3f}, '
                         f'{term.paw.max():9.3f}]')
        lines.append(f'  max |residual|: {abs(self.residual_skn).max():.2e}')
        return '\n'.join(lines)


class KSDecompositionCalculator:
    """Calculate per-state kinetic, Hartree, xc and zero-potential terms.

    The potentials and one-center matrices are set up once in the
    constructor; :meth:`calculate` can then be called for different bands
    and k-points.  Works with k-point, band and domain parallelization.
    """

    def __init__(self, dft: DFT):
        self.dft = dft
        self.ibzwfs = dft.ibzwfs
        self.density = dft.density
        self.potential = dft.potential
        self.pot_calc = dft.pot_calc
        self.setups = dft.setups

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
        # on the master of the domain communicator (None elsewhere):
        self.vbar_R: UGArray | None
        self.vHt_R: UGArray | None
        self.vxct_sR: UGArray | None
        self.vbar_R, self.vHt_R, self.vxct_sR = self._pseudo_potentials()

        # One-center corrections for all atoms (on all ranks):
        (self.dK_aii, self.dH_aii, self.dB_aii,
         self.dxc_asii) = self._atomic_corrections()

    @classmethod
    def from_gpw_file(cls,
                      filename: str | Path,
                      comm=None) -> KSDecompositionCalculator:
        from gpaw import GPAW
        calc = GPAW(filename, legacy_gpaw=False, communicator=comm, txt=None)
        return cls(calc.dft)

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

    def _pseudo_potentials(self) -> tuple[UGArray | None,
                                          UGArray | None,
                                          UGArray | None]:
        """Split the pseudo potential into v_bar, v_H and v_xc.

        The three potentials are calculated in exactly the same way as
        in :meth:`gpaw.new.pw.pot_calc.PlaneWavePotentialCalculator.
        calculate_pseudo_potential` so that their sum equals the
        potential used in the SCF calculation (``potential.vt_sR``).
        """
        pot_calc = self.pot_calc
        density = self.density

        # XC-potential (calculated on the fine grid, restricted to
        # the coarse grid):
        nt_sr = pot_calc.interpolate(density.nt_sR)
        _, vxct_sr, _ = pot_calc.xc.calculate(nt_sr)
        vxct_sR = pot_calc.restrict(vxct_sr).gather()

        # Electrostatic potential (PW coefficients):
        vHt_h = self.potential.vHt_x.gather()

        if self.domain_comm.rank != 0:
            return None, None, None

        assert vxct_sR is not None and vHt_h is not None
        grid0 = self.grid0
        vbar_R = pot_calc.vbar0_g.ifft(grid=grid0)
        # Map v_H onto the plane-wave sphere of the density
        # (the SCF potential only contains those components):
        vHt_r = vHt_h.ifft(grid=vHt_h.desc.minimal_uniform_grid())
        vHt_g = vHt_r.fft(pw=pot_calc.pw.new(comm=None))
        vHt_R = vHt_g.ifft(grid=grid0)
        return vbar_R, vHt_R, vxct_sR

    def _atomic_corrections(self) -> tuple[dict[int, np.ndarray],
                                           dict[int, np.ndarray],
                                           dict[int, np.ndarray],
                                           dict[int, np.ndarray]]:
        """One-center PAW corrections, split like in
        :func:`gpaw.new.pot_calc.calculate_non_local_potential1`."""
        density = self.density
        setups = self.setups
        xc = self.pot_calc.xc

        D_asii = density.D_asii.to_xp(np).gather(broadcast=True)

        # Integrals of compensation charges times electrostatic potential:
        vHt_h = self.potential.vHt_x
        ghat_aLh = setups.create_compensation_charges(
            vHt_h.desc, self.dft.relpos_ac, density.D_asii.layout.atomdist)
        V_aL: AtomArrays = ghat_aLh.integrate(vHt_h).gather(broadcast=True)

        dK_aii = {}
        dH_aii = {}
        dB_aii = {}
        dxc_asii = {}
        ndensities = density.ndensities
        for a, D_sii in D_asii.items():
            setup = setups[a]
            D_sp = np.array([pack_density(D_ii.real) for D_ii in D_sii])
            D_p = D_sp[:ndensities].sum(0)
            dK_aii[a] = unpack_hermitian(setup.K_p)
            dH_aii[a] = unpack_hermitian(setup.M_p +
                                         2.0 * setup.M_pp @ D_p +
                                         setup.Delta_pL @ V_aL[a])
            dB_aii[a] = unpack_hermitian(setup.MB_p)
            dxc_sp = np.zeros_like(D_sp)
            xc.calculate_paw_correction(setup, D_sp, dxc_sp, a=a)
            dxc_sii = unpack_hermitian(dxc_sp)
            if setup.hubbard_u is not None:
                _, dHU_sii = setup.hubbard_u.calculate(setup, D_sii)
                dxc_sii += dHU_sii
            dxc_asii[a] = dxc_sii

        return dK_aii, dH_aii, dB_aii, dxc_asii

    def calculate(self,
                  n1: int = 0,
                  n2: int = 0,
                  kpt_indices: Sequence[int] | None = None
                  ) -> KSDecomposition:
        """Calculate decomposition for bands ``n1:n2`` and IBZ k-points.

        ``n2 <= 0`` counts from the total number of bands.  All IBZ
        k-points are used if ``kpt_indices`` is ``None``.
        """
        ibzwfs = self.ibzwfs
        n2 = n2 if n2 > 0 else ibzwfs.nbands + n2
        if not 0 <= n1 < n2 <= ibzwfs.nbands:
            raise ValueError(f'Bad band range: {n1}:{n2} '
                             f'(nbands={ibzwfs.nbands})')
        if kpt_indices is None:
            kpt_indices = range(len(ibzwfs.ibz))
        kpt_indices = list(kpt_indices)

        shape = (ibzwfs.nspins, len(kpt_indices), n2 - n1)
        names = ['eps'] + [f'{name}_{part}'
                           for name in TERM_NAMES
                           for part in ['pseudo', 'paw']]
        x_skn = {name: np.zeros(shape) for name in names}

        for i, k in enumerate(kpt_indices):
            for spin in range(ibzwfs.nspins):
                if ibzwfs.rank_ks[k, spin] != ibzwfs.kpt_comm.rank:
                    continue
                wfs = ibzwfs._get_wfs(k, spin)
                wfs = wfs.collect_bands_and_domain(n1, n2)
                if wfs is None:
                    continue  # not band and domain master
                for name, x_n in self._one_kpt(wfs).items():
                    x_skn[name][spin, i] = x_n

        for x in x_skn.values():
            ibzwfs.comm.sum(x)

        return KSDecomposition(
            eps_skn=x_skn['eps'] * Ha,
            n1=n1,
            n2=n2,
            kpt_indices=kpt_indices,
            **{name: Term(pseudo=x_skn[f'{name}_pseudo'] * Ha,
                          paw=x_skn[f'{name}_paw'] * Ha)
               for name in TERM_NAMES})

    def _one_kpt(self, wfs: PWFDWaveFunctions) -> dict[str, np.ndarray]:
        """Expectation values for one k-point and spin (units: Ha).

        ``wfs`` must be serial (bands and domain collected).
        """
        assert self.vHt_R is not None
        assert self.vxct_sR is not None
        assert self.vbar_R is not None
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
        vH_n = np.zeros(nbands)
        vxc_n = np.zeros(nbands)
        vbar_n = np.zeros(nbands)
        for n, ut_R in enumerate(ut_nR.data):
            nt_R.data[:] = 0.0
            add_to_density(1.0, ut_R, nt_R.data)
            vH_n[n] = self.vHt_R.integrate(nt_R)
            vxc_n[n] = self.vxct_sR[spin].integrate(nt_R)
            vbar_n[n] = self.vbar_R.integrate(nt_R)
        result['hartree_pseudo'] = vH_n
        result['xc_pseudo'] = vxc_n
        result['zero_pseudo'] = vbar_n

        # One-center PAW corrections:
        for name, dX_aii in [('kinetic', self.dK_aii),
                             ('hartree', self.dH_aii),
                             ('zero', self.dB_aii)]:
            result[f'{name}_paw'] = sum(
                (expectation_values(P_ani[a], dX_ii)
                 for a, dX_ii in dX_aii.items()),
                np.zeros(nbands))
        result['xc_paw'] = sum(
            (expectation_values(P_ani[a], dxc_sii[spin])
             for a, dxc_sii in self.dxc_asii.items()),
            np.zeros(nbands))

        return result


def expectation_values(P_ni: np.ndarray, M_ii: np.ndarray) -> np.ndarray:
    """Calculate ``sum_ij P_ni^* M_ij P_nj`` for all n."""
    return np.einsum('ni, ij, nj -> n', P_ni.conj(), M_ii, P_ni).real


def ks_decomposition(calc,
                     n1: int = 0,
                     n2: int = 0,
                     kpt_indices: Sequence[int] | None = None
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

    See :class:`KSDecomposition` for a description of the result and
    the module docstring for details of the decomposition.
    """
    if isinstance(calc, (str, Path)):
        kdc = KSDecompositionCalculator.from_gpw_file(calc)
    elif hasattr(calc, 'ibzwfs'):  # DFT object
        kdc = KSDecompositionCalculator(calc)
    elif hasattr(calc, 'dft'):  # new-style ASE calculator
        kdc = KSDecompositionCalculator(calc.dft)
    else:
        raise TypeError(
            'Expected a gpw-filename, a gpaw.dft.DFT object or a '
            'new-style GPAW calculator (legacy_gpaw=False), '
            f'got {type(calc)}')
    return kdc.calculate(n1, n2, kpt_indices)
