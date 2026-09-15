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
from gpaw.lfc import LFC
from gpaw.mpi import world
from gpaw.poisson import PoissonSolver
from gpaw.transformers import Transformer
from gpaw.utilities import pack_density
from gpaw.utilities.tools import tri2full
from gpaw.old.matrix import matrix_matrix_multiply as mmm

if TYPE_CHECKING:
    from typing import Any, Callable
    from gpaw.old.calculator import GPAW as GPAWCalc
    from gpaw.old.kpoint import KPoint
    from gpaw.old.grid_descriptor import GridDescriptor
    pyscf_t = Any


@dataclass(frozen=True)
class PairDensity:
    spin: int
    i: int
    j: int
    rhot_g: np.ndarray
    D_ap: dict[int, np.ndarray]


class CoulombPotential:
    name = 'isolated FPS'

    def __init__(self, finegd: GridDescriptor, physical_pbc: np.ndarray):
        if np.any(physical_pbc):
            raise ValueError(
                'Periodic ERIs need an explicit reciprocal-space G=0 convention.')
        self.finegd = finegd
        self.poisson = PoissonSolver('fast', use_charge_center=True,
                                     use_charged_periodic_corrections=True)
        self.poisson.set_grid_descriptor(finegd)
        self.potential_g = finegd.empty(dtype=float)

    def potential(self, density_g: np.ndarray) -> np.ndarray:
        self.potential_g.fill(0.0)
        self.poisson.solve(
            self.potential_g,
            density_g,
            charge=None,
            zero_initial_phi=True
        )
        return self.potential_g

    def integrate(self, density_g: np.ndarray, potential_g: np.ndarray) -> float:
        value = self.finegd.integrate(density_g, potential_g)
        return float(np.real(value))


class PWERI:
    """Construct F, bare Coulomb ERIs, and h = F - G[D]."""

    def __init__(self, calc: GPAWCalc):
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
        self.finegd = self.density.finegd
        self.setups = self.wfs.setups

        self.validate_reference()

        self.interpolator = Transformer(self.gd, self.finegd, 3)
        self.ghat = LFC(
            self.finegd, [setup.ghat_l for setup in self.setups])
        self.ghat.set_positions(self.calc.spos_ac, self.wfs.atom_partition)
        self.coulomb = CoulombPotential(
            self.finegd, [False, False, False])
        self.psit_snG = [self.get_realspace_psit(kpt)
                         for kpt in self.kpt_s]

    def validate_reference(self) -> None:
        assert self.wfs.mode == 'pw'
        assert self.wfs.nspins in {1, 2}
        assert len(self.wfs.kpt_u) == self.wfs.nspins
        assert self.wfs.kd.gamma
        # assert not np.any(self.calc.atoms.pbc)
        assert np.dtype(self.wfs.dtype).kind == 'f'
        assert self.wfs.bd.comm.size == 1

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
        psit_nG = kpt.psit_nG
        if self.wfs.mode == "pw":
            psit_nG = np.asarray(
                [self.wfs.pd.ifft(psit_G, q=kpt.q) for psit_G in psit_nG]
            )
        return psit_nG

    def make_pair_density(self, spin: int, i: int, j: int) -> PairDensity:
        """Build psi_i*psi_j."""
        kpt = self.kpt_s[spin]
        psit_nG = self.psit_snG[spin]
        nt_G = self.gd.empty(dtype=float)
        np.multiply(psit_nG[i], psit_nG[j], out=nt_G)

        rhot_g = self.finegd.empty(dtype=float)
        self.interpolator.apply(nt_G, rhot_g)

        D_ap: dict[int, np.ndarray] = {}
        Q_aL: dict[int, np.ndarray] = {}
        for atom, P_ni in kpt.P_ani.items():
            D_ii = np.outer(P_ni[i], P_ni[j])
            D_p = pack_density(D_ii)
            D_ap[atom] = D_p
            Q_aL[atom] = D_p @ self.setups[atom].Delta_pL

        self.ghat.add(rhot_g, Q_aL)
        return PairDensity(spin, i, j, rhot_g, D_ap)

    def calculate_eri(self, progress: Callable[[str], None] | None = None) -> np.ndarray:
        """Calculate all real chemists'-notation ``(ij|kl)`` integrals."""
        pairs = [
            (spin, i, j)
            for spin in range(self.nspins)
            for i in range(self.nbands)
            for j in range(i, self.nbands)
        ]

        cached = len(pairs) * self.finegd.zeros().nbytes
        if progress is not None:
            progress(f'Caching {len(pairs)} unique pair densities '
                     f'({cached / 2**30:.2f} GiB per domain rank).')

        pair_data = [
            self.make_pair_density(spin, i, j)
            for spin, i, j in pairs
        ]
        pair_eri = np.empty((len(pairs), len(pairs)))

        for a, pair_a in enumerate(pair_data):
            potential_g = self.coulomb.potential(pair_a.rhot_g)
            for b in range(a, len(pair_data)):
                pair_b = pair_data[b]
                smooth = self.coulomb.integrate(
                    pair_b.rhot_g, potential_g)

                paw_local = 0.0
                for atom, D1_p in pair_a.D_ap.items():
                    D2_p = pair_b.D_ap[atom]
                    paw_local += (
                        D1_p @ self.setups[atom].M_pp @ D2_p)

                paw = self.wfs.atom_partition.comm.sum_scalar(
                    paw_local)

                value = float(np.real(smooth + 2.0 * paw))
                pair_eri[a, b] = value
                pair_eri[b, a] = value

            if progress is not None:
                progress(f'Poisson solves: {a + 1}/{len(pair_data)}')

        return self.expand_pair_matrix(pair_eri, pairs)

    def expand_pair_matrix(
            self, pair_eri: np.ndarray, pairs: list[tuple[int, int, int]]
    ) -> np.ndarray:
        """Expand the packed pair matrix using the eight real ERI symmetries."""
        eri = np.empty((self.nspins, self.nspins) + (self.nbands,) * 4)
        for a, (spin1, i, j) in enumerate(pairs):
            for b, (spin2, k, l) in enumerate(pairs):
                value = pair_eri[a, b]
                for p, q in {(i, j), (j, i)}:
                    for r, s in {(k, l), (l, k)}:
                        eri[spin1, spin2, p, q, r, s] = value
                        eri[spin2, spin1, r, s, p, q] = value
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eri = self.calculate_eri(progress=progress)
        fock = np.asarray([self.calculate_fock_matrix(kpt)
                           for kpt in self.kpt_s])
        occupations = np.asarray([kpt.f_n for kpt in self.kpt_s],
                                 dtype=float)
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
    eri_errors = {
        'i<->j': np.max(np.abs(eri - eri.transpose(0, 1, 3, 2, 4, 5))),
        'k<->l': np.max(np.abs(eri - eri.transpose(0, 1, 2, 3, 5, 4))),
        '(ij)<->(kl)': np.max(
            np.abs(eri - eri.transpose(1, 0, 4, 5, 2, 3))),
    }
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
        pweri: PWERI, nbands: int | None = None, progress: Callable[[str], None] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    h1, eri, fock = pweri.run(progress=progress)
    h1 = np.array(h1[:, :nbands, :nbands])
    fock = np.array(fock[:, :nbands, :nbands])
    eri = np.array(eri[:, :, :nbands, :nbands, :nbands, :nbands])
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
        'scf_orbital_count': pweri.nbands,
        'frozen_core_electron_count': sum(
            setup.Nc for setup in pweri.setups),
        'units': 'Hartree',
        'eri_convention': 'chemist (ij|kl), non-antisymmetrized',
        'orbital_basis': (
            'GPAW spatial orbitals' if pweri.nspins == 1
            else 'spin-resolved GPAW spatial orbitals'),
        'coulomb_boundary': pweri.coulomb.name,
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
    from pyscf import gto, scf
    from pyscf.cc import ccsd

    norb = h1.shape[-1]
    reference = scf.RHF(gto.M())
    reference.mo_coeff = np.eye(norb)
    reference.mo_occ = occupations
    reference.mo_energy = np.diag(fock)
    reference._eri = eri
    reference.e_tot = hf_energy(
        h1[None], eri[None, None], occupations[None], 2)
    reference.converged = True
    reference.make_rdm1 = lambda *args: np.diag(occupations)
    reference.get_veff = lambda *args, **kwargs: None
    reference.get_fock = lambda *args, **kwargs: fock

    mycc = ccsd.CCSD(reference)
    eris = mycc.ao2mo()
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
    """Validate P_snn without writing it."""
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
                        help='Number of lowest orbitals to write.'
                             '  PWERI still evaluates every SCF band.')
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
    pweri = PWERI(calc)
    progress = print if args.verbose and world.rank == 0 else None
    h1, eri, fock, occupations, eigenvalues = calculate_integrals(
        pweri, nbands=args.nbands, progress=progress)

    diagnostics = None
    if args.diagnostics:
        diagnostics = validate_results(h1, eri, fock, occupations,
                                       eigenvalues, pweri.degeneracy)
        if world.rank == 0:
            for name, value in diagnostics.items():
                print(f'{name}: {value:.3e}')

    if world.rank == 0:
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
        if args.ccsd:
            P_snn, P_snn_metadata = run_ccsd(
                h1, eri, fock, occupations,
                nspins=calc.density.nspins,
                metadata=args.ccsd_metadata)
        elif args.fci:
            P_snn, P_snn_metadata = run_fci(
                h1, eri, occupations,
                nspins=calc.density.nspins,
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
