from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from pathlib import Path
from warnings import warn

import numpy as np
from ase.units import Ha

from gpaw import GPAW
from gpaw.hybrids.wstc import WignerSeitzTruncatedCoulomb
from gpaw.mpi import world
from gpaw.old.pw.lfc import PWLFC
from gpaw.utilities import pack_density
from gpaw.utilities.tools import tri2full
from gpaw.old.matrix import matrix_matrix_multiply as mmm

if TYPE_CHECKING:
    from typing import Any, Callable
    from gpaw.old.calculator import GPAW as GPAWCalc
    from gpaw.old.kpoint import KPoint
    from gpaw.iks.fdpweri import FDPWERI
    pyscf_t = Any


@dataclass(frozen=True)
class PairBlock:
    rhot_xG: np.ndarray
    D_axp: dict[int, np.ndarray]


class ReciprocalCoulomb:
    name = 'isolated Wigner-Seitz truncated reciprocal-space'

    def __init__(self, pd):
        wstc = WignerSeitzTruncatedCoulomb(
            pd.gd.cell_cv, np.ones(3, dtype=int))
        kernel_G = pd.zeros()
        wstc._get_potential(
            pd.myQ_qG[0],
            pd.kd.bzk_kc[0],
            pd.gd.N_c,
            pd.dtype,
            pd.get_reciprocal_vectors(add_q=True),
            kernel_G)
        kernel_G = kernel_G.real
        if kernel_G.min() < -1.0e-12:
            raise ValueError('Wigner-Seitz Coulomb kernel is not positive.')
        self.sqrt_kernel_G = np.sqrt(np.maximum(kernel_G, 0.0))


class PWERI:
    """Construct F and ERIs with a reciprocal-space Coulomb kernel."""

    def __init__(self, calc: GPAWCalc, cache_bands: int | None = None):
        if cache_bands is not None and cache_bands < 0:
            raise ValueError('cache_bands must be non-negative.')

        if calc.wfs.spos_ac is None:
            calc.set_positions()

        self.calc = calc
        self.wfs = calc.wfs
        self.density = calc.density
        self.hamiltonian = calc.hamiltonian
        self.nspins = self.wfs.nspins
        self.degeneracy = 2 // self.nspins
        self.kpt_s = self.wfs.kpt_u
        self.nbands = self.wfs.bd.nbands
        self.gd = self.wfs.gd
        self.pd = self.wfs.pd
        self.setups = self.wfs.setups
        self.cache_bands = cache_bands

        self.validate_reference()

        self.ghat = PWLFC(
            [setup.ghat_l for setup in self.setups], self.pd)
        self.ghat.set_positions(self.calc.spos_ac, self.wfs.atom_partition)
        self.coulomb = ReciprocalCoulomb(self.pd)
        self.psit_snG = [self.get_realspace_psit(kpt)
                         for kpt in self.kpt_s]

    def validate_reference(self) -> None:
        assert self.wfs.mode == 'pw'
        assert self.wfs.nspins in {1, 2}
        assert len(self.wfs.kpt_u) == self.wfs.nspins
        assert self.wfs.kd.gamma
        # if np.any(self.calc.atoms.pbc):
        #     raise ValueError('Reciprocal-space ERIs require a non-periodic system.')
        assert np.dtype(self.wfs.dtype).kind == 'f'
        assert self.wfs.bd.comm.size == 1
        assert self.wfs.kpt_u[0].psit_nG is not None

        xc = self.hamiltonian.xc
        fraction = getattr(xc, 'exx_fraction', None)
        if fraction is None or fraction != 1.0:
            # raise ValueError('Use xc="EXX:backend=pw".')
            warn(
                "The double counting correction is only implemented for EXX."
                "The hopping (h1) matrix is going to include xc terms."
            )
            self.sub_exx = False
        else:
            self.sub_exx = True

    def get_realspace_psit(self, kpt) -> np.ndarray:
        psit_nG = np.asarray(
            [self.wfs.pd.ifft(psit_G, q=kpt.q) for psit_G in kpt.psit_nG]
        )
        return psit_nG

    def make_pair_block(
            self, pairs: list[tuple[int, int, int]]) -> PairBlock:
        """Build reciprocal-space pair densities."""
        rhot_xG = self.pd.empty(len(pairs))
        D_axp: dict[int, np.ndarray] = {}
        Q_axL: dict[int, np.ndarray] = {}
        nt_R = self.gd.empty(dtype=float)

        for x, (spin, i, j) in enumerate(pairs):
            kpt = self.kpt_s[spin]
            psit_nR = self.psit_snG[spin]
            np.multiply(psit_nR[i], psit_nR[j], out=nt_R)
            rhot_xG[x] = self.pd.fft(nt_R)

            for atom, P_ni in kpt.P_ani.items():
                D_p = pack_density(np.outer(P_ni[i], P_ni[j]))
                Q_L = D_p @ self.setups[atom].Delta_pL
                if atom not in D_axp:
                    D_axp[atom] = np.empty((len(pairs), len(D_p)))
                    Q_axL[atom] = np.empty((len(pairs), len(Q_L)))
                D_axp[atom][x] = D_p
                Q_axL[atom][x] = Q_L

        self.ghat.add(rhot_xG, Q_axL)
        rhot_xG *= self.coulomb.sqrt_kernel_G
        return PairBlock(rhot_xG, D_axp)

    def calculate_eri(
            self, progress: Callable[[str], None] | None = None) -> np.ndarray | None:
        """Calculate all real chemists'-notation ``(ij|kl)`` integrals."""
        pairs = [
            (spin, i, j)
            for spin in range(self.nspins)
            for i in range(self.nbands)
            for j in range(i, self.nbands)
        ]

        cached_bands = self.nbands
        if self.cache_bands is not None:
            cached_bands = min(cached_bands, self.cache_bands)
        cache_count = self.nspins * cached_bands * (cached_bands + 1) // 2

        if cache_count >= len(pairs):
            block_size = len(pairs)
        else:
            block_size = max(1, cache_count // 2)

        max_contraction_bytes = 2**30
        max_block_size = int(
            (max_contraction_bytes // np.dtype(float).itemsize)**0.5)
        block_size = min(block_size, max_block_size)

        nblocks = (len(pairs) + block_size - 1) // block_size
        contractions = nblocks * (nblocks + 1) // 2
        cache_all = cache_count >= len(pairs)
        resident = (len(pairs) if cache_all else
                    min(len(pairs), 2 * block_size))
        cached = resident * self.pd.empty().nbytes
        if progress is not None:
            progress(
                f'Using {nblocks} reciprocal-space blocks of up to '
                f'{block_size} pair densities for {cached_bands} cache bands '
                f'({cached / 2**30:.2f} GiB per domain rank).')
            packed = len(pairs)**2 * np.dtype(float).itemsize
            dense = (self.nspins**2 * self.nbands**4 *
                     np.dtype(float).itemsize)
            progress(
                f'Rank 0 global ERI storage: {packed / 2**30:.2f} GiB '
                f'packed + {dense / 2**30:.2f} GiB expanded.')

        pair_eri = (np.empty((len(pairs), len(pairs)))
                    if world.rank == 0 else None)
        ranges = [(start, min(start + block_size, len(pairs)))
                  for start in range(0, len(pairs), block_size)]
        blocks = None
        if cache_all:
            blocks = [self.make_pair_block(pairs[start:stop])
                      for start, stop in ranges]

        contraction = 0
        for ai, (astart, astop) in enumerate(ranges):
            block_a = (blocks[ai] if blocks is not None else
                       self.make_pair_block(pairs[astart:astop]))

            for bi in range(ai, len(ranges)):
                bstart, bstop = ranges[bi]
                if blocks is not None:
                    block_b = blocks[bi]
                elif bstart == astart:
                    block_b = block_a
                else:
                    block_b = self.make_pair_block(pairs[bstart:bstop])

                smooth = self.pd.integrate(block_a.rhot_xG, block_b.rhot_xG)
                if bstart == astart:
                    tri2full(smooth, 'U')
                paw = np.zeros(smooth.shape, dtype=smooth.dtype)
                for atom, D1_pp in block_a.D_axp.items():
                    D2_pp = block_b.D_axp[atom]
                    paw += D1_pp @ self.setups[atom].M_pp @ D2_pp.T
                self.wfs.atom_partition.comm.sum(paw)

                if world.rank == 0:
                    values = np.asarray(
                        smooth.real + 2.0 * paw, dtype=float)
                    if bstart == astart:
                        values = 0.5 * (values + values.T)
                    pair_eri[astart:astop, bstart:bstop] = values
                    pair_eri[bstart:bstop, astart:astop] = values.T

                contraction += 1
                if progress is not None:
                    progress(
                        f'Reciprocal contractions: {contraction}/{contractions}')

                if blocks is None and block_b is not block_a:
                    del block_b

            if blocks is None:
                del block_a

        if blocks is not None:
            del blocks

        if world.rank == 0:
            return self.expand_pair_matrix(pair_eri, pairs)
        return None

    def expand_pair_matrix(
            self, pair_eri: np.ndarray, pairs: list[tuple[int, int, int]]
    ) -> np.ndarray:
        """Expand the packed pair matrix using the eight real ERI symmetries."""
        eri = np.empty((self.nspins, self.nspins) + (self.nbands,) * 4)
        pair_index_snn = np.empty(
            (self.nspins, self.nbands, self.nbands), dtype=int)
        for a, (spin1, i, j) in enumerate(pairs):
            pair_index_snn[spin1, i, j] = a
            pair_index_snn[spin1, j, i] = a

        for a, (spin1, i, j) in enumerate(pairs):
            for spin2, pair_index_nn in enumerate(pair_index_snn):
                values_nn = pair_eri[a, pair_index_nn]
                eri[spin1, spin2, i, j] = values_nn
                eri[spin1, spin2, j, i] = values_nn
        return eri

    def calculate_fock_matrix(self, kpt: KPoint) -> np.ndarray:
        """Build <phi_i|F_HF|phi_j>."""
        ham = self.hamiltonian
        wfs = self.wfs
        psit = kpt.psit

        work = psit.new(buf=wfs.work_array)
        H_nn = wfs.work_matrix_nn
        H_nn.array.fill(0.0)
        P2 = kpt.projections.new()

        apply_h = partial(wfs.apply_pseudo_hamiltonian, kpt, ham)

        psit.matrix_elements(
            operator=apply_h,
            result=work,
            out=H_nn,
            symmetric=True,
            cc=True
        )

        ham.dH(kpt.projections, out=P2)
        mmm(
            1.0,
            kpt.projections, 'N',
            P2, 'C',
            1.0,
            H_nn,
            symmetric=True
        )

        # no-op for EXX
        ham.xc.correct_hamiltonian_matrix(kpt, H_nn.array)

        self.gd.comm.sum(H_nn.array)

        fock = np.array(H_nn.array)
        tri2full(fock)
        fock = np.asarray(np.real_if_close(fock), dtype=float)
        return 0.5 * (fock + fock.T)

    def run(
            self, progress: Callable[[str], None] | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
        eri = self.calculate_eri(progress=progress)
        fock = np.asarray([self.calculate_fock_matrix(kpt)
                           for kpt in self.kpt_s])[
                               :, :self.nbands, :self.nbands]
        if world.rank != 0:
            return None, None, fock

        occupations = np.asarray(
            [kpt.f_n[:self.nbands] for kpt in self.kpt_s], dtype=float)
        h1 = remove_reference_mean_field(
            fock, eri, occupations, self.degeneracy, self.sub_exx)
        return h1, eri, fock


def reference_fock(
    h1: np.ndarray, eri: np.ndarray, occupations: np.ndarray, degeneracy: int
) -> np.ndarray:
    """Return the reference Fock matrices for spin-resolved occupations."""
    fock = np.empty_like(h1)
    for spin1 in range(len(occupations)):
        coulomb = sum(
            np.einsum('i,pqii->pq', occupations[spin2],
                      eri[spin1, spin2], optimize=True)
            for spin2 in range(len(occupations)))
        exchange = np.einsum(
            'i,piiq->pq', occupations[spin1], eri[spin1, spin1], optimize=True
        ) / degeneracy
        fock[spin1] = h1[spin1] + coulomb - exchange
    return 0.5 * (fock + fock.transpose(0, 2, 1))


def remove_reference_mean_field(
    fock: np.ndarray, eri: np.ndarray, occupations: np.ndarray, degeneracy: int, sub_exx=True,
) -> np.ndarray:
    """Return h = F - J[D] + K[D] for each spin channel."""
    h1 = np.empty_like(fock)
    for spin1 in range(len(occupations)):
        coulomb = sum(
            np.einsum('i,pqii->pq', occupations[spin2],
                      eri[spin1, spin2], optimize=True)
            for spin2 in range(len(occupations)))
        if sub_exx:
            exchange = np.einsum(
                'i,piiq->pq', occupations[spin1], eri[spin1, spin1],
                optimize=True) / degeneracy
        else:
            exchange = 0.0
        h1[spin1] = fock[spin1] - coulomb + exchange
    return 0.5 * (h1 + h1.transpose(0, 2, 1))


def validate_results(
    h1: np.ndarray,
    eri: np.ndarray,
    fock: np.ndarray,
    occupations: np.ndarray,
    eigenvalues: np.ndarray,
        degeneracy: int) -> dict[str, float]:
    """Validate tensor symmetries, and HF reconstruction."""
    eri_errors = {'i<->j': 0.0, 'k<->l': 0.0, '(ij)<->(kl)': 0.0}
    eri_ji = eri.transpose(0, 1, 3, 2, 4, 5)
    eri_lk = eri.transpose(0, 1, 2, 3, 5, 4)
    eri_pair = eri.transpose(1, 0, 4, 5, 2, 3)
    for i in range(eri.shape[2]):
        eri_errors['i<->j'] = max(
            eri_errors['i<->j'],
            float(np.max(np.abs(eri[:, :, i] - eri_ji[:, :, i]))))
        eri_errors['k<->l'] = max(
            eri_errors['k<->l'],
            float(np.max(np.abs(eri[:, :, :, :, i] -
                                eri_lk[:, :, :, :, i]))))
        eri_errors['(ij)<->(kl)'] = max(
            eri_errors['(ij)<->(kl)'],
            float(np.max(np.abs(eri[:, :, i] - eri_pair[:, :, i]))))
    worst_eri = float(max(eri_errors.values()))
    assert worst_eri < 1e-09, f'ERI symmetry check failed: {eri_errors}'

    reconstructed_fock = reference_fock(h1, eri, occupations, degeneracy)
    fock_reconstruction = float(np.max(np.abs(reconstructed_fock - fock)))
    assert fock_reconstruction < 1e-9, f'Fock reconstruction failed: {fock_reconstruction}'

    diagonal = np.zeros_like(fock)
    indices = np.arange(fock.shape[-1])
    diagonal[:, indices, indices] = fock[:, indices, indices]
    max_offdiag = float(np.max(np.abs(fock - diagonal)))
    eigenvalue_error = float(
        np.max(np.abs(np.diagonal(fock, axis1=1, axis2=2) - eigenvalues)))

    return {
        'max_eri_symmetry_error_ha': worst_eri,
        'max_fock_reconstruction_error_ha': fock_reconstruction,
        'max_fock_offdiagonal_ha': max_offdiag,
        'max_fock_eigenvalue_error_ha': eigenvalue_error,
    }


def hf_energy(
        h1: np.ndarray, eri: np.ndarray, occupations: np.ndarray, degeneracy: int
) -> float:
    """Return the HF energy of the exported determinant."""
    one_body = np.einsum('si,sii->', occupations, h1, optimize=True)
    coulomb = 0.0
    for spin1 in range(len(occupations)):
        for spin2 in range(len(occupations)):
            coulomb += 0.5 * np.einsum(
                'i,j,iijj->', occupations[spin1], occupations[spin2],
                eri[spin1, spin2], optimize=True)

    exchange = 0.0
    for spin in range(len(occupations)):
        exchange -= 0.5 / degeneracy * np.einsum(
            'i,j,ijji->', occupations[spin], occupations[spin],
            eri[spin, spin], optimize=True)
    return float(one_body + coulomb + exchange)


def calculate_integrals(
        pweri: PWERI | FDPWERI, nbands: int | None = None, progress: Callable[[str], None] | None = None
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray,
           np.ndarray, np.ndarray]:
    """Run PWERI and return a lowest-band orbital space."""
    if nbands is None:
        nbands = pweri.nbands

    assert 1 <= nbands <= pweri.nbands

    all_occupations = np.asarray([kpt.f_n for kpt in pweri.kpt_s], dtype=float)
    assert np.all(
        np.abs(all_occupations[:, nbands:]) < 1e-10), "excludes occupied!"

    occupations = np.array(all_occupations[:, :nbands])
    integer_occupations = (np.isclose(occupations, 0.0, atol=1.0e-10)
                           | np.isclose(occupations, pweri.degeneracy, atol=1.0e-10))
    assert np.all(integer_occupations)

    pweri.nbands = nbands
    h1, eri, fock = pweri.run(progress=progress)
    eigenvalues = np.asarray(
        [kpt.eps_n[:nbands] for kpt in pweri.kpt_s], dtype=float)
    return h1, eri, fock, occupations, eigenvalues


def integral_metadata(
        pweri: PWERI,
        h1: np.ndarray,
        eri: np.ndarray,
        occupations: np.ndarray,
        eigenvalues: np.ndarray,
        diagnostics: dict[str, float] | None = None) -> dict:
    """Return metadata for calculated integrals."""

    if diagnostics is None:
        diagnostics = {}

    return {
        'occupations': (occupations[0] if pweri.nspins == 1 else occupations),
        'hf_energy': hf_energy(h1, eri, occupations, pweri.degeneracy),
        'gpaw_energy': pweri.calc.results['energy'] / Ha,
        'eigenvalues': (eigenvalues[0] if pweri.nspins == 1 else eigenvalues),
        'electron_count': occupations.sum(),
        'nspins': pweri.nspins,
        'spin_electron_count_difference': (
            0.0 if pweri.nspins == 1
            else occupations[0].sum() - occupations[1].sum()),
        'orbital_count': h1.shape[-1],
        'scf_orbital_count': pweri.wfs.bd.nbands,
        'frozen_core_electron_count': sum(
            setup.Nc for setup in pweri.setups),
        'units': 'Hartree',
        'eri_convention': 'chemist (ij|kl), non-antisymmetrized',
        'orbital_basis': (
            'GPAW spatial orbitals' if pweri.nspins == 1
            else 'spin-resolved GPAW spatial orbitals'),
        'coulomb_boundary': pweri.coulomb.name,
        'eri_backend': 'reciprocal-space block contraction',
        'pair_density_cutoff_ha': pweri.pd.ecut,
        'gpaw_mode': 'pw',
        'eigensolver': type(pweri.wfs.eigensolver).__name__,
        **diagnostics,
    }


def write_integrals(
        outdir: Path,
        h1: np.ndarray,
        eri: np.ndarray,
        fock: np.ndarray,
        metadata: dict) -> None:
    """Write calculated integrals and prepared metadata."""
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / 'H_nn.npy', h1)
    np.save(outdir / 'U_ijkl.npy', eri)
    np.save(outdir / 'Fock_nn.npy', fock)
    np.savez(outdir / 'hf_fci_metadata.npz', **metadata)


def exx_diagnostics(
        calc: GPAWCalc,
        eri: np.ndarray,
        occupations: np.ndarray,
        degeneracy: int) -> dict[str, float]:
    """Compare ERI exchange with GPAW valence-valence EXX."""
    eri_exchange = 0.0
    for spin in range(len(occupations)):
        eri_exchange -= 0.5 / degeneracy * np.einsum(
            'i,j,ijji->', occupations[spin], occupations[spin],
            eri[spin, spin], optimize=True)
    xc = calc.hamiltonian.xc
    gpaw_ecc = float(xc.ecc)
    gpaw_ecv = float(xc.evc)
    gpaw_evv = float(xc.evv)
    gpaw_exx = gpaw_ecc + gpaw_ecv + gpaw_evv
    return {
        'gpaw_exx_energy_ha': gpaw_exx,
        'gpaw_exx_core_core_ha': gpaw_ecc,
        'gpaw_exx_core_valence_ha': gpaw_ecv,
        'gpaw_exx_valence_valence_ha': gpaw_evv,
        'eri_exchange_energy_ha': float(eri_exchange),
        'exx_eri_mismatch_ha': float(
            gpaw_evv - eri_exchange),
    }


def write_fcidump(
        outdir: Path, h1: np.ndarray, eri: np.ndarray, norb: int, nelec: int, spin: int = 0
) -> Path:
    """Write a restricted-orbital FCIDUMP file."""
    from pyscf.tools import fcidump

    assert h1.ndim == 2 and eri.ndim == 4
    dump = outdir / 'FCIDUMP'
    fcidump.from_integrals(dump, h1, eri, norb, nelec, nuc=0.0, ms=spin)
    return dump


def make_pyscf_hf(
        dump: Path, fock_gpaw: np.ndarray, occupations: np.ndarray, nelec: int, model_hf: float
) -> tuple[pyscf_t, float, float, float]:
    """Run PySCF from an existing (restricted) FCIDUMP file."""
    from pyscf.tools import fcidump

    mf = fcidump.to_scf(dump)
    mf.conv_tol = 1.0e-11
    mf.max_cycle = 100

    reference_dm = np.diag(occupations)
    vj, vk = mf.get_jk(dm=reference_dm)
    pyscf_fock = mf.get_hcore() + vj - 0.5 * vk
    fock_error = float(np.max(np.abs(pyscf_fock - fock_gpaw)))
    assert fock_error < 1e-8

    mf.kernel(dm0=reference_dm)
    assert mf.converged
    hf_error = float(mf.e_tot - model_hf)
    assert abs(hf_error) < 1e-7

    occupied = np.flatnonzero(occupations > 1.0e-10)
    virtual = np.flatnonzero(occupations < 2.0 - 1.0e-10)
    gap = float(mf.mo_energy[virtual[0]] - mf.mo_energy[occupied[-1]])
    return mf, fock_error, hf_error, gap


def make_ccsd(
        h1: np.ndarray, eri: np.ndarray, fock: np.ndarray, occupations: np.ndarray
) -> tuple[pyscf_t, pyscf_t]:
    """Prepare CCSD in the GPAW MO basis."""
    from pyscf import ao2mo, gto, lib, scf
    from pyscf.cc import ccsd

    norb = h1.shape[-1]
    reference = scf.RHF(gto.M())
    reference.mo_coeff = np.eye(norb)
    reference.mo_occ = occupations
    reference.mo_energy = np.diag(fock)
    reference.e_tot = hf_energy(
        h1[None], eri[None, None], occupations[None], 2)
    reference.converged = True
    reference.make_rdm1 = lambda *args: np.diag(occupations)
    reference.get_veff = lambda *args, **kwargs: None
    reference.get_fock = lambda *args, **kwargs: fock

    mycc = ccsd.CCSD(reference)
    eris = ccsd._ChemistsERIs()
    eris._common_init_(mycc)
    nocc = eris.nocc
    nvir = norb - nocc
    occupied = slice(None, nocc)
    virtual = slice(nocc, None)
    eris.oooo = np.array(eri[occupied, occupied, occupied, occupied])
    eris.ovoo = np.array(eri[occupied, virtual, occupied, occupied])
    eris.ovvo = np.array(eri[occupied, virtual, virtual, occupied])
    eris.ovov = np.array(eri[occupied, virtual, occupied, virtual])
    eris.oovv = np.array(eri[occupied, occupied, virtual, virtual])
    ovvv = eri[occupied, virtual, virtual, virtual]
    eris.ovvv = lib.pack_tril(
        ovvv.reshape(-1, nvir, nvir)).reshape(nocc, nvir, -1)
    eris.vvvv = ao2mo.restore(
        4, eri[virtual, virtual, virtual, virtual], nvir)
    return mycc, eris


def make_uccsd(
        h1: np.ndarray, eri: np.ndarray, fock: np.ndarray, occupations: np.ndarray
) -> tuple[pyscf_t, pyscf_t]:
    """Prepare UCCSD in the GPAW MO bases."""
    from pyscf import gto, scf
    from pyscf.cc import uccsd

    norb = h1.shape[-1]
    reference = scf.UHF(gto.M())
    reference.mo_coeff = (np.eye(norb), np.eye(norb))
    reference.mo_occ = tuple(occupations)
    reference.mo_energy = tuple(np.diag(f_s) for f_s in fock)
    reference.e_tot = hf_energy(h1, eri, occupations, 1)
    reference.converged = True
    reference.make_rdm1 = lambda *args: np.asarray(
        [np.diag(f_s) for f_s in occupations])
    reference.get_veff = lambda *args, **kwargs: None
    reference.get_fock = lambda *args, **kwargs: fock

    mycc = uccsd.UCCSD(reference)
    same_spin = iter((eri[0, 0], eri[1, 1]))

    def ao2mofn(mo_coeff):
        if isinstance(mo_coeff, tuple):
            return eri[0, 1]
        return next(same_spin)

    eris = uccsd._make_eris_incore(mycc, ao2mofn=ao2mofn)
    return mycc, eris


def run_ccsd(
        h1: np.ndarray, eri: np.ndarray, fock: np.ndarray, occupations: np.ndarray,
        nspins: int, metadata: bool = False
) -> tuple[np.ndarray, dict | None]:
    """Run CCSD or UCCSD and return P_snn in the GPAW MO bases."""
    if nspins == 1:
        mycc, eris = make_ccsd(
            h1[0], eri[0, 0], fock[0], occupations[0])
    else:
        mycc, eris = make_uccsd(h1, eri, fock, occupations)

    mycc.conv_tol = 1.0e-12
    mycc.conv_tol_normt = 1.0e-10
    mycc.max_cycle = 200
    mycc.kernel(eris=eris)
    if not mycc.converged:
        raise RuntimeError('CCSD did not converge.')

    mycc.solve_lambda(eris=eris)
    if not mycc.converged_lambda:
        raise RuntimeError('CCSD Lambda equations did not converge.')

    P_snn = np.asarray(mycc.make_rdm1())
    if nspins == 1:
        P_snn = P_snn[None]
    data = None
    if metadata:
        data = {
            'method': 'CCSD' if nspins == 1 else 'UCCSD',
            'file': 'P_snn.npy',
            'energy_ha': float(mycc.e_tot),
            'correlation_energy_ha': float(mycc.e_corr),
            'spin_convention': (
                'spin-summed' if nspins == 1 else 'spin-resolved'),
            'orbital_basis': 'original GPAW HF molecular orbitals',
            'electron_count': float(np.trace(
                P_snn, axis1=-2, axis2=-1).real.sum()),
        }
    return P_snn, data


def run_fci(
        h1: np.ndarray, eri: np.ndarray, occupations: np.ndarray,
        nspins: int, metadata: bool = False
) -> tuple[np.ndarray, dict | None]:
    """Run FCI and return P_snn in the GPAW MO bases."""
    from pyscf import fci

    norb = h1.shape[-1]
    if nspins == 1:
        nelec = int(round(occupations.sum()))
        solver = fci.direct_spin1.FCI()
        solver.conv_tol = 1.0e-12
        solver.max_cycle = 200
        energy, ci_vector = solver.kernel(
            h1[0], eri[0, 0], norb, nelec)
        P_snn = np.asarray(
            [solver.make_rdm1(ci_vector, norb, nelec)])
    else:
        nelec = tuple(int(round(f_n.sum())) for f_n in occupations)
        solver = fci.direct_uhf.FCI()
        solver.conv_tol = 1.0e-12
        solver.max_cycle = 200
        energy, ci_vector = solver.kernel(
            (h1[0], h1[1]),
            (eri[0, 0], eri[0, 1], eri[1, 1]),
            norb, nelec)
        P_snn = np.asarray(
            solver.make_rdm1s(ci_vector, norb, nelec))

    if not solver.converged:
        raise RuntimeError('FCI did not converge.')

    data = None
    if metadata:
        reference_energy = hf_energy(
            h1, eri, occupations, 2 // nspins)
        data = {
            'method': 'FCI',
            'file': 'P_snn.npy',
            'energy_ha': float(energy),
            'correlation_energy_ha': float(energy - reference_energy),
            'spin_convention': (
                'spin-summed' if nspins == 1 else 'spin-resolved'),
            'orbital_basis': 'original GPAW HF molecular orbitals',
            'electron_count': float(np.trace(
                P_snn, axis1=-2, axis2=-1).real.sum()),
        }
    return P_snn, data


def validate_density_matrix(
        P_snn: np.ndarray, occupations: np.ndarray
) -> dict[str, float]:
    """Validate P_snn"""
    trace_error = float(
        np.trace(P_snn, axis1=-2, axis2=-1).real.sum() - occupations.sum())
    hermiticity_error = float(np.max(np.abs(
        P_snn - P_snn.conj().transpose(0, 2, 1))))
    assert abs(trace_error) < 1.0e-8
    assert hermiticity_error < 1.0e-8
    return {
        'density_matrix_trace_error': trace_error,
        'density_matrix_hermiticity_error': hermiticity_error,
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('restart', type=Path,
                        help='Converged old-interface PW EXX/HF .gpw file.')
    parser.add_argument('--outdir', type=Path, default=Path.cwd(),
                        help='Directory for integral arrays and metadata.')
    parser.add_argument('--nbands', type=int,
                        help='Number of lowest orbitals to calculate and write.')
    parser.add_argument('--cache-bands', type=int,
                        help='Limit Coulomb workspace to the size needed for'
                             ' this many bands. Defaults to all bands; use 0'
                             ' for minimum memory.')
    parser.add_argument('--diagnostics', action='store_true',
                        help='Validate and print integral diagnostics.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print ERI calculation progress.')
    parser.add_argument('--fcidump', action='store_true',
                        help='Write a restricted-orbital FCIDUMP file.')
    solver = parser.add_mutually_exclusive_group()
    solver.add_argument('--ccsd', action='store_true',
                        help='Run CCSD/UCCSD and write P_snn.npy.')
    solver.add_argument('--fci', action='store_true',
                        help='Run FCI and write P_snn.npy.')
    parser.add_argument('--ccsd-metadata', action='store_true',
                        help='Write P_snn_metadata.json with --ccsd.')
    parser.add_argument('--fci-metadata', action='store_true',
                        help='Write P_snn_metadata.json with --fci.')
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    calc = GPAW(args.restart)
    pweri = PWERI(calc, cache_bands=args.cache_bands)
    progress = print if args.verbose and world.rank == 0 else None
    h1, eri, fock, occupations, eigenvalues = calculate_integrals(
        pweri, nbands=args.nbands, progress=progress)

    if world.rank != 0:
        return

    diagnostics = None
    if args.diagnostics:
        diagnostics = validate_results(h1, eri, fock, occupations,
                                       eigenvalues, pweri.degeneracy)
        for name, value in diagnostics.items():
            print(f'{name}: {value:.3e}')

    metadata = integral_metadata(
        pweri, h1, eri, occupations, eigenvalues, diagnostics=diagnostics
    )
    write_integrals(args.outdir, h1, eri, fock, metadata)

    if args.fcidump:
        if pweri.nspins != 1:
            raise ValueError(
                'FCIDUMP is only implemented for spin-paired.')
        write_fcidump(
            args.outdir, h1[0], eri[0, 0], h1.shape[-1],
            int(round(occupations.sum())))

    P_snn = None
    P_snn_metadata = None
    nspins = pweri.nspins
    if args.ccsd or args.fci:
        import gc
        del calc, pweri
        gc.collect()

    if args.ccsd:
        P_snn, P_snn_metadata = run_ccsd(
            h1, eri, fock, occupations,
            nspins=nspins,
            metadata=args.ccsd_metadata)
    elif args.fci:
        P_snn, P_snn_metadata = run_fci(
            h1, eri, occupations,
            nspins=nspins,
            metadata=args.fci_metadata)

    if P_snn is not None:
        if args.diagnostics:
            P_snn_diagnostics = validate_density_matrix(
                P_snn, occupations)
            for name, value in P_snn_diagnostics.items():
                print(f'{name}: {value:.3e}')
            if P_snn_metadata is not None:
                P_snn_metadata.update(P_snn_diagnostics)
        np.save(args.outdir / 'P_snn.npy', P_snn)
        if P_snn_metadata is not None:
            (args.outdir / 'P_snn_metadata.json').write_text(
                json.dumps(P_snn_metadata, indent=2) + '\n')

    print(f'Wrote integrals to {args.outdir.resolve()}')


if __name__ == '__main__':
    from gpaw import setup_paths
    if '.' not in setup_paths:
        setup_paths.append('.')

    raise SystemExit(main())
