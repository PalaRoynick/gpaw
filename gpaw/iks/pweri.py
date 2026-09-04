import argparse
import json
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Callable
from pathlib import Path

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
    from gpaw.old.calculator import GPAW as GPAWCalc


@dataclass(frozen=True)
class PairDensity:
    spin: int
    i: int
    j: int
    rhot_g: np.ndarray
    D_ap: dict[int, np.ndarray]


class CoulombPotential:
    name = 'isolated FPS'

    def __init__(self, finegd, physical_pbc):
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
            self.finegd, self.calc.atoms.pbc)
        self.psit_snG = [self.get_realspace_psit(kpt)
                         for kpt in self.kpt_s]

    def validate_reference(self) -> None:
        assert self.wfs.mode == 'pw'
        assert self.wfs.nspins in {1, 2}
        assert len(self.wfs.kpt_u) == self.wfs.nspins
        assert self.wfs.kd.gamma
        assert not np.any(self.calc.atoms.pbc)
        assert np.dtype(self.wfs.dtype).kind == 'f'
        assert self.wfs.bd.comm.size == 1

        xc = self.hamiltonian.xc
        fraction = getattr(xc, 'exx_fraction', None)
        if fraction is None or fraction != 1.0:
            raise ValueError('Use xc="EXX:backend=pw".')

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

        local_bytes = len(pairs) * self.finegd.zeros().nbytes
        if progress is not None:
            progress(f'Caching {len(pairs)} unique pair densities '
                     f'({local_bytes / 2**30:.2f} GiB per domain rank).')

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
            self,
            pair_eri: np.ndarray,
            pairs: list[tuple[int, int, int]]
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

    def calculate_fock_matrix(self, kpt) -> np.ndarray:
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
            self,
            progress: Callable[[str], None] | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        eri = self.calculate_eri(progress=progress)
        fock = np.asarray([self.calculate_fock_matrix(kpt)
                           for kpt in self.kpt_s])
        occupations = np.asarray([kpt.f_n for kpt in self.kpt_s],
                                 dtype=float)
        h1 = remove_reference_mean_field(
            fock, eri, occupations, self.degeneracy)
        return h1, eri, fock


def reference_fock(
        h1: np.ndarray,
        eri: np.ndarray,
        occupations: np.ndarray,
        degeneracy: int) -> np.ndarray:
    """Return the reference Fock matrices for spin-resolved occupations."""
    fock = np.empty_like(h1)
    for spin1 in range(len(occupations)):
        coulomb = sum(
            np.einsum('i,pqii->pq', occupations[spin2],
                      eri[spin1, spin2], optimize=True)
            for spin2 in range(len(occupations)))
        exchange = np.einsum(
            'i,piiq->pq', occupations[spin1], eri[spin1, spin1],
            optimize=True) / degeneracy
        fock[spin1] = h1[spin1] + coulomb - exchange
    return 0.5 * (fock + fock.transpose(0, 2, 1))


def remove_reference_mean_field(
        fock: np.ndarray,
        eri: np.ndarray,
        occupations: np.ndarray,
        degeneracy: int) -> np.ndarray:
    """Return h = F - J[D] + K[D] for each spin channel."""
    h1 = np.empty_like(fock)
    for spin1 in range(len(occupations)):
        coulomb = sum(
            np.einsum('i,pqii->pq', occupations[spin2],
                      eri[spin1, spin2], optimize=True)
            for spin2 in range(len(occupations)))
        exchange = np.einsum(
            'i,piiq->pq', occupations[spin1], eri[spin1, spin1],
            optimize=True) / degeneracy
        h1[spin1] = fock[spin1] - coulomb + exchange
    return 0.5 * (h1 + h1.transpose(0, 2, 1))


def validate_results(
        h1: np.ndarray,
        eri: np.ndarray,
        fock: np.ndarray,
        occupations: np.ndarray,
        eigenvalues: np.ndarray,
        degeneracy: int) -> dict[str, float]:
    """Validate tensor symmetries, HF reconstruction, and canonicality."""
    eri_errors = {
        'i<->j': np.max(np.abs(eri - eri.transpose(0, 1, 3, 2, 4, 5))),
        'k<->l': np.max(np.abs(eri - eri.transpose(0, 1, 2, 3, 5, 4))),
        '(ij)<->(kl)': np.max(
            np.abs(eri - eri.transpose(1, 0, 4, 5, 2, 3))),
    }
    worst_eri = float(max(eri_errors.values()))
    assert worst_eri < 1e-09, f'ERI symmetry check failed: {eri_errors}'

    reconstructed_fock = reference_fock(
        h1, eri, occupations, degeneracy)
    fock_reconstruction = float(
        np.max(np.abs(reconstructed_fock - fock)))
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


def hf_energy(h1: np.ndarray,
              eri: np.ndarray,
              occupations: np.ndarray,
              degeneracy: int) -> float:
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
        pweri: PWERI,
        nbands: int | None = None,
        progress: Callable[[str], None] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Run PWERI and return a lowest-band orbital space without writing."""
    if nbands is None:
        nbands = pweri.nbands

    assert 1 <= nbands <= pweri.nbands

    all_occupations = np.asarray([kpt.f_n for kpt in pweri.kpt_s],
                                 dtype=float)
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
    """Return metadata for calculated integrals without writing."""

    if diagnostics is None:
        diagnostics = {}

    return {
        'occupations': (
            occupations[0] if pweri.nspins == 1 else occupations),
        'hf_energy': hf_energy(
            h1, eri, occupations, pweri.degeneracy),
        'gpaw_energy': pweri.calc.results['energy'] / Ha,
        'eigenvalues': (
            eigenvalues[0] if pweri.nspins == 1 else eigenvalues),
        'electron_count': occupations.sum(),
        'spin_count': pweri.nspins,
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
    gpaw_exx_core_core = float(xc.ecc)
    gpaw_exx_core_valence = float(xc.evc)
    gpaw_exx_valence_valence = float(xc.evv)
    gpaw_exx = (gpaw_exx_core_core + gpaw_exx_core_valence +
                gpaw_exx_valence_valence)
    return {
        'gpaw_exx_energy_ha': gpaw_exx,
        'gpaw_exx_core_core_ha': gpaw_exx_core_core,
        'gpaw_exx_core_valence_ha': gpaw_exx_core_valence,
        'gpaw_exx_valence_valence_ha': gpaw_exx_valence_valence,
        'eri_exchange_energy_ha': float(eri_exchange),
        'exx_eri_mismatch_ha': float(
            gpaw_exx_valence_valence - eri_exchange),
    }


def write_fcidump(
        outdir: Path,
        h1: np.ndarray,
        eri: np.ndarray,
        norb: int,
        nelec: int,
        spin: int = 0) -> Path:
    """Write a restricted-orbital FCIDUMP file."""
    from pyscf.tools import fcidump

    assert h1.ndim == 2 and eri.ndim == 4
    dump = outdir / 'FCIDUMP'
    fcidump.from_integrals(
        str(dump), h1, eri, norb, nelec, nuc=0.0, ms=spin)
    return dump


def make_pyscf_hf(
        dump: Path,
        fock_gpaw: np.ndarray,
        occupations: np.ndarray,
        nelec: int,
        model_hf: float):
    """Run PySCF from an existing restricted-orbital FCIDUMP file."""
    from pyscf.tools import fcidump

    mf = fcidump.to_scf(str(dump))
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


def prepare_p_nm(
        p_nm: np.ndarray,
        method: str,
        norb: int,
        nelec: int) -> tuple[np.ndarray, dict]:
    """Validate a spin-summed 1-RDM without writing it."""
    p_nm = np.asarray(np.real_if_close(p_nm))
    assert not np.iscomplexobj(p_nm)
    p_nm = np.asarray(p_nm, dtype=float)
    assert p_nm.shape == (norb, norb)

    trace_error = float(np.trace(p_nm) - nelec)
    hermiticity_error = float(np.max(np.abs(p_nm - p_nm.T)))
    assert abs(trace_error) < 1.0e-8
    assert hermiticity_error < 1.0e-8

    p_nm = 0.5 * (p_nm + p_nm.T)
    metadata = {
        'method': method,
        'file': 'P_nm.npy',
        'orbital_basis': 'exported GPAW spatial orbitals',
        'spin_convention': 'spin-summed',
        'density_convention':
            'n(r) = sum_nm P_nm psi_n*(r) psi_m(r)',
        'norb': norb,
        'nelec': nelec,
        'trace': float(np.trace(p_nm)),
        'trace_error': trace_error,
        'hermiticity_error': hermiticity_error,
    }
    return p_nm, metadata


def write_p_nm(
        outdir: Path,
        p_nm: np.ndarray,
        metadata: dict) -> dict:
    """Write a prepared spin-summed 1-RDM and its metadata."""
    np.save(outdir / 'P_nm.npy', p_nm)
    (outdir / 'P_nm_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n')
    return {
        'density_matrix_method': metadata['method'],
        'density_matrix_file': 'P_nm.npy',
        'density_matrix_trace': metadata['trace'],
        'density_matrix_trace_error': metadata['trace_error'],
        'density_matrix_hermiticity_error': metadata['hermiticity_error'],
    }


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    from gpaw import setup_paths
    if '.' not in setup_paths:
        setup_paths.append('.')

    args = parse_args()
    calc = GPAW(args.restart)
    pweri = PWERI(calc)
    progress = print if args.verbose and world.rank == 0 else None
    h1, eri, fock, occupations, eigenvalues = calculate_integrals(
        pweri, nbands=args.nbands, progress=progress)

    diagnostics = None
    if args.diagnostics:
        diagnostics = validate_results(
            h1, eri, fock, occupations, eigenvalues, pweri.degeneracy)
        if world.rank == 0:
            for name, value in diagnostics.items():
                print(f'{name}: {value:.3e}')

    if world.rank == 0:
        metadata = integral_metadata(
            pweri, h1, eri, occupations, eigenvalues,
            diagnostics=diagnostics)
        write_integrals(
            args.outdir, h1, eri, fock, metadata)
        print(f'Wrote integrals to {args.outdir.resolve()}')


if __name__ == '__main__':
    main()
