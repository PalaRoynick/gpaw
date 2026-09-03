# Future TODO: def bandstructure_interpolate(calc, bandpath, shift_kn):
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.linalg
from ase.spectrum.band_structure import BandStructure
from ase.units import Ha
from gpaw.calculator import GPAW
from gpaw.mpi import world as default_world
from gpaw.utilities import pack_density


def mpower(O_MM, power):
    if power == 0:
        return np.eye(len(O_MM))
    elif power == 1 / 2:
        return scipy.linalg.sqrtm(O_MM)
    elif power == 1:
        return O_MM
    raise ValueError('Provided s_power is not implemented. Use 1, 1/2 or 0.')


@dataclass
class LCAOOperator:
    """
    Class for k-point dependent LCAO matrix operator operator_kMM

    Attributes:
    - kpt_kc: k-point coordinates
    - cell_cv: real-space unit-cell vectors

    Methods:
    - Fourier transform O_kMM to O_RMM; returns RealSpaceOperator
    - Fourier interpolation

    """

    kpt_kc: np.ndarray
    cell_cv: np.ndarray

    def get_operator(self, k: int):
        raise NotImplementedError

    # Add caching
    def construct(self):
        operator_KMM = []
        for k, k_c in enumerate(self.kpt_kc):
            operator_KMM.append(self.get_operator(k))
        return operator_KMM

    def fourier_transform(self, cutoff, N_c, atoms):
        operator_KMM = np.array(self.construct())

        from ase.neighborlist import NeighborList

        assert np.allclose(self.cell_cv, atoms.cell)

        cutoffs = [cutoff / 2] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=True, bothways=True)
        nl.update(atoms)

        offsets_neighbors = []
        for i in range(len(atoms)):
            indices, offsets = nl.get_neighbors(i)
            offsets_neighbors.extend((offsets + N_c // 2) % N_c - N_c // 2)

        offsets_neighbors = np.array(offsets_neighbors)
        offsets_Rc = np.unique(offsets_neighbors, axis=0)

        nao = operator_KMM.shape[-1]
        NR = len(offsets_Rc)
        operator_RMM = np.zeros((NR, nao, nao), dtype=complex)
        NK = len(self.kpt_kc)

        for R, R_c in enumerate(offsets_Rc):
            for u1, k_c in enumerate(self.kpt_kc):
                phase = np.exp(2j * np.pi * np.dot(R_c, k_c))
                operator_RMM[R] += phase * operator_KMM[u1] / NK

        return RealSpaceOperator(self.cell_cv, offsets_Rc, operator_RMM)

    def fourier_interpolate(self, higher_k_mesh, cutoff, N_c, atoms):
        # Take kMM operator, and convert to real space
        operator_RMM = self.fourier_transform(cutoff, N_c, atoms)
        # Convert back to fourier space with higher k-mesh
        return operator_RMM.inverse_fourier_transform(higher_k_mesh)


@dataclass
class GPAWLCAOOperator(LCAOOperator):
    """Construct"""

    calc_bz: GPAW

    @property
    def operator_kMM(self):
        return np.array(
            [self.get_operator(k) for k in range(len(self.calc_bz.wfs.kpt_u))])

    @classmethod
    def create(cls, calc_bz, **kwargs):
        ibzk_qc = calc_bz.wfs.kd.ibzk_qc
        kpt_kc = [ibzk_qc[kpt.q] for kpt in calc_bz.wfs.kpt_u]
        cell_cv = calc_bz.atoms.cell
        return cls(kpt_kc, cell_cv, calc_bz, **kwargs)

    def get_operator(self, k: int):
        return self.build(
            self.calc_bz.wfs.kpt_u[k].S_MM, self.calc_bz.wfs.kpt_u[k].C_nM, k)

    def build(self, S_MM, C_nM, u):
        raise NotImplementedError


@dataclass
class ReciprocalSpace:
    """Conversion for IBZ to BZ kpoint mapping

    Attributes:
        bz2ibz_K: mapping table which BZ k-point K is mapped to IBZ k-point k
        either provided by user or given by GPAW calc.wfs.kd.bz2ibz_k
        kpt_Kc: Optional for plotting; BZ k-point coordinates
        from GPAW calc.wfs.kd.bzk_kc

    """

    bz2ibz_K: np.ndarray
    kpt_Kc: np.ndarray | None = None

    @classmethod
    def from_gpaw_calculator(cls, calc):
        return cls(calc.wfs.kd.bz2ibz_k, calc.wfs.kd.bzk_kc)

    def ibz2bz(self, delta_ibz_kn):
        "Mapping scalar IBZ k-points to BZ K-points"
        return np.array([delta_ibz_kn[k] for k in self.bz2ibz_K])


def _run_on_rank0(calc, world):
    """Check that calculator-backed interpolation is serial on rank zero."""
    if world.rank != 0:
        return False
    if calc is None:
        raise ValueError('A calculator is required on rank zero')
    calc_world = calc.wfs.world
    if calc_world.size != 1:
        raise ValueError(
            'LCAO band-structure interpolation runs on rank zero. Load '
            'the calculator with communicator=mpi.serial_comm.'
        )
    return True


def operators_from_calc(calc_bz, spin=0, world=default_world):
    if not _run_on_rank0(calc_bz, world):
        return None
    # Get the full BZ Hamiltonian
    from gpaw.lcao.tools import get_lcao_hamiltonian

    H_skMM, S_kMM = get_lcao_hamiltonian(calc_bz)
    if not 0 <= spin < len(H_skMM):
        raise ValueError(
            f'Spin index {spin} outside available range 0:{len(H_skMM)}'
        )
    k_bz = calc_bz.get_ibz_k_points()
    #  Reciprocoal LCAO Operator and fourier trafo to real space H_R
    cell_cv = calc_bz.atoms.cell
    hamiltonian_kMM = H_skMM[spin]
    H_kMM = ReciprocalLCAOOperator(
        kpt_kc=k_bz, cell_cv=cell_cv, operator_kMM=hamiltonian_kMM)

    overlap_kMM = S_kMM
    S_kMM = ReciprocalLCAOOperator(
        kpt_kc=k_bz, cell_cv=cell_cv, operator_kMM=overlap_kMM)

    return H_kMM, S_kMM


@dataclass
class RealSpaceOperator:
    """ """

    cell_cv: np.ndarray
    offsets_Rc: np.ndarray
    operator_RMM: np.ndarray

    def inverse_fourier_transform(self, kpt_kc):

        NR, nao, nao2 = self.operator_RMM.shape
        NK = len(kpt_kc)
        operator_kMM = np.zeros((NK, nao, nao), dtype=complex)
        for u1, K_c in enumerate(kpt_kc):
            for R, R_c in enumerate(self.offsets_Rc):
                phase = np.exp(-2j * np.pi * np.dot(R_c, K_c))
                operator_kMM[u1] += phase * self.operator_RMM[R]
        for operator_MM in operator_kMM:
            operator_MM[:] = (operator_MM + operator_MM.T.conj()) / 2

        return ReciprocalLCAOOperator(kpt_kc, self.cell_cv, operator_kMM)


@dataclass
class ScissorOperator(GPAWLCAOOperator):
    """Construct scissor operator"""

    delta_Kn: np.ndarray
    s_power: float = 1 / 2

    def build(self, S_MM, C_nM, u):
        S_MM_pow = mpower(S_MM, self.s_power)
        X_nM = C_nM.conj() @ S_MM_pow.conj()
        delta = self.delta_Kn[u]
        if delta.ndim == 1:  # delta_n
            return X_nM.conj().T @ (delta[:, None] * X_nM)
        else:  # delta_nn
            return X_nM.conj().T @ delta @ X_nM

    @classmethod
    def from_operators(cls, H_kMM, S_kMM, delta_Kn, s_power):
        scissor_kMM = []
        for k in range(len(H_kMM.operator_kMM)):
            H_MM = H_kMM.operator_kMM[k]
            S_MM = S_kMM.operator_kMM[k]
            S_MM_pow = mpower(S_MM, s_power)
            e_n, C_Mn = scipy.linalg.eigh(
                H_MM, S_MM)  # this here is C_Mn not C_nM thats why
            C_Mn = C_Mn[:, : delta_Kn.shape[1]]
            X_nM = C_Mn.conj().T @ S_MM_pow
            delta = delta_Kn[k]
            if delta.ndim == 1:  # delta_n
                scissor_kMM.append(X_nM.conj().T @ (delta[:, None] * X_nM))
            else:  # delta_nn
                scissor_kMM.append(X_nM.conj().T @ delta @ X_nM)
        return ReciprocalLCAOOperator(
            kpt_kc=H_kMM.kpt_kc,
            cell_cv=H_kMM.cell_cv,
            operator_kMM=np.array(scissor_kMM))


@dataclass
class ReciprocalLCAOOperator(LCAOOperator):
    operator_kMM: np.ndarray

    def get_operator(self, k: int):
        return self.operator_kMM[k]

    def __iadd__(self, other):
        self.operator_kMM += other.operator_kMM
        assert np.allclose(self.kpt_kc, other.kpt_kc)
        assert np.allclose(self.cell_cv, other.cell_cv)
        return self


def create_realspace_lcao_operators(
    calc_ibz, cutoff, shift_kn, s_power, spin=0, world=default_world
):
    if not _run_on_rank0(calc_ibz, world):
        return None
    # Setup BZ operators needed to interpolate the scissor correction.
    calc_bz = calc_ibz.fixed_density(
        symmetry='off',
        **(

            dict(
                nbands=shift_kn.shape[1],
                convergence={'bands': shift_kn.shape[1]})
            if shift_kn is not None
            else {}))

    scissor_RMM = None
    if shift_kn is None:
        return calc_bz, scissor_RMM

    H_kMM, S_kMM = operators_from_calc(
        calc_bz, spin=spin, world=world
    )
    N_c = calc_bz.wfs.kd.N_c

    reciprocal = ReciprocalSpace.from_gpaw_calculator(calc_ibz)
    delta_Kn = reciprocal.ibz2bz(shift_kn)
    scissor_kMM = ScissorOperator.from_operators(
        H_kMM,
        S_kMM,
        delta_Kn,
        s_power)
    scissor_RMM = scissor_kMM.fourier_transform(
        cutoff=cutoff, atoms=calc_bz.atoms, N_c=N_c)
    return calc_bz, scissor_RMM


def exact_lcao_operators(
    calc, kpts, shift_kn=None, spin=0, world=default_world
):
    if not _run_on_rank0(calc, world):
        return None
    kwargs = {}
    if shift_kn is not None:
        nbands = shift_kn.shape[1]
        kwargs.update(nbands=nbands, convergence={'bands': nbands})

    calc_bandpath = calc.fixed_density(
        symmetry='off',
        kpts=kpts,
        **kwargs)
    H_kMM, S_kMM = operators_from_calc(
        calc_bandpath, spin=spin, world=world
    )
    assert np.allclose(H_kMM.kpt_kc, kpts)
    assert np.allclose(S_kMM.kpt_kc, kpts)
    return H_kMM, S_kMM


def create_bandstructure(calc,
                         bandpath,
                         cutoff=13,
                         shift_kn=None,
                         s_power=1 / 2,
                         spin=0,
                         world=default_world):

    if not _run_on_rank0(calc, world):
        return None

    scissor_power_RMM = None
    if shift_kn is not None:
        _calc_bz, scissor_power_RMM = create_realspace_lcao_operators(
            calc,
            cutoff=cutoff,
            shift_kn=shift_kn,
            s_power=s_power,
            spin=spin,
            world=world)

    H_bandpath_kMM, S_bandpath_kMM = exact_lcao_operators(
        calc,
        bandpath.kpts,
        shift_kn=shift_kn,
        spin=spin,
        world=world)

    scissor_power_bandpath_kMM = None
    if shift_kn is not None:
        scissor_power_bandpath_kMM = (
            scissor_power_RMM.inverse_fourier_transform(bandpath.kpts))
    return bandstructure_from_reciprocal_operators(
        H_bandpath_kMM,
        S_bandpath_kMM,
        bandpath,
        correction_power_kMM=scissor_power_bandpath_kMM,
        s_power=s_power)


def diagonalize_soc_lcao(H_skMM, S_kMM, V_kvMM):
    """Diagonalize a full spin-polarized LCAO+SOC Hamiltonian."""
    nspins, nkpts, nao, _ = H_skMM.shape
    if nspins not in {1, 2}:
        raise ValueError(f'Expected one or two spins, got {nspins}')

    energies_kn = []
    sz_kn = []
    for k in range(nkpts):
        S_MM = S_kMM[k]
        H_up_MM = H_skMM[0, k]
        H_down_MM = H_skMM[-1, k]
        H_NN = np.zeros((2 * nao, 2 * nao), complex)
        H_NN[:nao, :nao] = H_up_MM
        H_NN[nao:, nao:] = H_down_MM

        x_MM, y_MM, z_MM = V_kvMM[k]
        H_NN[:nao, :nao] += z_MM
        H_NN[:nao, nao:] += x_MM + 1j * y_MM
        H_NN[nao:, :nao] += x_MM - 1j * y_MM
        H_NN[nao:, nao:] -= z_MM

        S_NN = np.kron(np.eye(2), S_MM)
        H_NN = (H_NN + H_NN.T.conj()) / 2
        S_NN = (S_NN + S_NN.T.conj()) / 2
        energies_n, C_Nn = scipy.linalg.eigh(H_NN, S_NN)
        Sz_NN = np.zeros_like(S_NN)
        Sz_NN[:nao, :nao] = S_MM
        Sz_NN[nao:, nao:] = -S_MM
        sz_n = np.einsum(
            'Ni,NM,Mi->i', C_Nn.conj(), Sz_NN, C_Nn
        ).real
        energies_kn.append(energies_n)
        sz_kn.append(sz_n)

    return np.asarray(energies_kn), np.asarray(sz_kn)


def create_soc_bandstructure(
    calc,
    bandpath,
    cutoff=13,
    shift_skn=None,
    s_power=1 / 2,
    world=default_world,
):
    """Interpolate and diagonalize the LCAO+SOC spinor Hamiltonian."""
    if not _run_on_rank0(calc, world):
        return None
    kwargs = {}
    if shift_skn is not None:
        nbands = shift_skn.shape[2]
        kwargs.update(nbands=nbands, convergence={'bands': nbands})
    calc_bandpath = calc.fixed_density(
        symmetry='off', kpts=bandpath.kpts, **kwargs
    )

    from gpaw.lcao.tools import get_lcao_hamiltonian
    from gpaw.spinorbit import soc as soc_terms

    H_skMM, S_kMM = get_lcao_hamiltonian(calc_bandpath)
    H_skMM = np.asarray(H_skMM).copy()
    S_operator = ReciprocalLCAOOperator(
        bandpath.kpts, calc.atoms.cell, np.asarray(S_kMM)
    )

    if shift_skn is not None:
        if len(shift_skn) != len(H_skMM):
            raise ValueError(
                'GW corrections and LCAO Hamiltonian have different '
                'numbers of spin channels'
            )
        for spin, shift_kn in enumerate(shift_skn):
            _, correction_RMM = create_realspace_lcao_operators(
                calc, cutoff, shift_kn, s_power, spin=spin, world=world
            )
            correction_power_kMM = correction_RMM.inverse_fourier_transform(
                bandpath.kpts
            )
            H_skMM[spin] += correction_to_covariant(
                correction_power_kMM, S_operator, s_power
            )

    density = calc.dft.density
    V_avii = []
    for a, D_sii in density.D_asii.items():
        D_sp = np.array([pack_density(D_ii.real) for D_ii in D_sii])
        setup = calc.dft.pot_calc.setups[a]
        xc = calc.dft.pot_calc.xc.xc
        V_avii.append(soc_terms(setup, xc, D_sp) * Ha)

    V_kvMM = []
    for wfs in calc_bandpath.dft.ibzwfs:
        V_vMM = np.zeros((3, len(S_kMM[0]), len(S_kMM[0])), complex)
        for a, V_vii in enumerate(V_avii):
            P_Mi = wfs.P_aMi[a]
            V_vMM += np.einsum(
                'Mi,vij,Tj->vMT', P_Mi.conj(), V_vii.conj(), P_Mi
            )
        V_kvMM.append(V_vMM)

    energies_kn, sz_kn = diagonalize_soc_lcao(
        H_skMM, np.asarray(S_kMM), np.asarray(V_kvMM)
    )
    return (
        BandStructure(
            path=bandpath,
            energies=energies_kn[np.newaxis],
            reference=0,
        ),
        sz_kn,
    )


def create_non_ft_bandstructure(calc, bandpath, world=default_world):
    if not _run_on_rank0(calc, world):
        return None
    calc_bz = calc.fixed_density(symmetry='off', kpts=bandpath.kpts)
    eps_kn = []
    for kpt in range(len(bandpath.kpts)):
        eps_n = calc_bz.get_eigenvalues(kpt=kpt)
        eps_kn.append(eps_n)
    eps_kn = np.array(eps_kn)
    band_structure = BandStructure(
        path=bandpath, energies=eps_kn[np.newaxis, :, :], reference=0)
    return band_structure


def match_kpoints_tol(kpts1_kc, kpts2_kc, tol=1e-12):
    i1 = []
    i2 = []

    for k1, kc1 in enumerate(kpts1_kc):
        for k2, kc2 in enumerate(kpts2_kc):
            if np.allclose(kc1, kc2, atol=tol, rtol=0):
                i1.append(k1)
                i2.append(k2)

    return np.array(i1), np.array(i2)


def plot_soc_bandstructures(
    bandstructures,
    filename,
    reference=0.0,
    emin=None,
    emax=None,
    world=default_world,
):
    """Plot SOC bands colored by their spin-z projections.

    ``bandstructures`` is a sequence of ``(label, bandstructure, sz_kn)``
    tuples. Plotting and file output are performed only on rank zero.
    """
    if world.rank != 0:
        return None
    if not bandstructures:
        raise ValueError('At least one band structure is required')

    fig, axes = plt.subplots(
        1,
        len(bandstructures),
        sharex=True,
        sharey=True,
        squeeze=False,
        figsize=(5 * len(bandstructures), 5),
        constrained_layout=True,
    )
    points = None
    for ax, (label, bs, sz_kn) in zip(axes[0], bandstructures):
        energies_kn = bs.energies[0] - reference
        sz_kn = np.asarray(sz_kn)
        if energies_kn.shape != sz_kn.shape:
            raise ValueError(
                f'Energy and Sz shapes differ for {label}: '
                f'{energies_kn.shape} != {sz_kn.shape}'
            )
        x_k, x_special, labels = bs.path.get_linear_kpoint_axis()
        ax.plot(x_k, energies_kn, color='0.75', linewidth=0.8)
        for energy_k, sz_k in zip(energies_kn.T, sz_kn.T):
            points = ax.scatter(
                x_k,
                energy_k,
                c=sz_k,
                cmap='viridis',
                vmin=-1,
                vmax=1,
                s=5,
            )
        for x in x_special:
            ax.axvline(x, color='0.85', linewidth=0.8)
        ax.axhline(0.0, color='black', linewidth=0.8)
        ax.set(
            title=label,
            xticks=x_special,
            xticklabels=labels,
            xlim=(x_k[0], x_k[-1]),
            ylim=(emin, emax),
        )

    axes[0, 0].set_ylabel('Energy (eV)')
    fig.colorbar(points, ax=axes[0], label=r'$\langle S_z\rangle$')
    fig.savefig(filename, dpi=200)
    return fig, axes[0]


def plot_g0w0_bandstructure(
    calc_gs,
    bandpath,
    filename_g0w0_pw_from_lcao,
    world=default_world,
):
    if not _run_on_rank0(calc_gs, world):
        return None
    import pickle
    results = pickle.load(open(f'{filename_g0w0_pw_from_lcao}', 'rb'))
    delta_ibz_kn = (
        results['qp'][0, :, :] - results['eps'][0, :, :])
    bandpath = calc_gs.atoms.cell.bandpath(npoints=250)
    cutoff = 100
    bandstructure_plot(
        calc_gs,
        bandpath,
        cutoff=cutoff,
        delta_ibz_kn=delta_ibz_kn,
        emin=-10,
        emax=20,
        filename='bs.png',
        plot_also_unshifted=True,
        world=world)
    plt.clf()


def bandstructure_plot(calc_ibz,
                       bandpath,
                       cutoff=14,
                       delta_ibz_kn=None,
                       emin=-10,
                       emax=5,
                       filename='bs.png',
                       plot_also_unshifted=True,
                       plot_also_non_FT=False,
                       s_power=1 / 2,
                       world=default_world):

    if not _run_on_rank0(calc_ibz, world):
        return None

    bs_interpolate_fromRspace = bandstructure(
        calc_ibz, cutoff, bandpath, delta_ibz_kn,
        s_power=s_power, world=world)
    return bs_interpolate_fromRspace.plot(
        emin=emin,
        emax=emax,
        filename=filename,
        plot_also_unshifted=plot_also_unshifted,
        plot_also_non_FT=plot_also_non_FT)


def bandstructure(
    calc_ibz,
    cutoff,
    bandpath,
    delta_ibz_kn,
    s_power=1 / 2,
    world=default_world,
):
    if not _run_on_rank0(calc_ibz, world):
        return None
    bs_shifted_fromRSpace = create_bandstructure(
        calc=calc_ibz,
        bandpath=bandpath,
        cutoff=cutoff,
        shift_kn=delta_ibz_kn,
        s_power=s_power,
        world=world)
    bs_shifted_fromRSpace.write(f'bs_fromRSpace_{cutoff}.json')
    reciprocal = ReciprocalSpace.from_gpaw_calculator(calc_ibz)

    k_ibz_kc = calc_ibz.get_ibz_k_points()

    if delta_ibz_kn.ndim == 2:  # delta_kn
        exact_ibz_eps_kn = np.array(
            [
                calc_ibz.get_eigenvalues(kpt=kpt)[
                    : delta_ibz_kn.shape[1]
                ]  # if eps_kn more bands than correction
                + delta_ibz_kn[kpt]
                for kpt in range(len(k_ibz_kc))])
    else:  # delta_knn
        exact_ibz_eps_kn = np.array(
            [
                np.linalg.eigvalsh(
                    np.diag(
                        calc_ibz.get_eigenvalues(kpt=kpt)[
                            : delta_ibz_kn.shape[1]
                        ]
                    )
                    + delta_ibz_kn[kpt]
                )
                for kpt in range(len(k_ibz_kc))])

    k_bz_kc = reciprocal.kpt_Kc
    exact_bz_eps_kn = reciprocal.ibz2bz(exact_ibz_eps_kn)

    no_shift_bs = create_bandstructure(
        calc_ibz,
        bandpath,
        cutoff=cutoff,
        shift_kn=None,
        world=world)

    bs_no_FT = create_non_ft_bandstructure(
        calc_ibz, bandpath, world=world
    )

    bp_ind, mesh_ind = match_kpoints_tol(bandpath.kpts, k_bz_kc)
    point_e_kn = exact_bz_eps_kn[mesh_ind]
    return BandStructureInterpolationPlot(
        exact_ibz_eps_kn,
        exact_bz_eps_kn,
        bs_shifted_fromRSpace,
        bs_no_FT,
        no_shift_bs,
        k_bz_kc,
        bp_ind,
        point_e_kn,
        calc_ibz,
        world=world)


def correction_to_covariant(correction_power_kMM, S_kMM, s_power):
    if correction_power_kMM is None:
        return None
    if s_power == 1:
        return correction_power_kMM.operator_kMM

    correction_cov_kMM = []
    for S_MM, correction_power_MM in zip(S_kMM.operator_kMM,
                                         correction_power_kMM.operator_kMM):
        S_MM_pow = mpower(S_MM, 1 - s_power)
        correction_cov_kMM.append(S_MM_pow @ correction_power_MM @ S_MM_pow)
    return np.array(correction_cov_kMM)


def eigenvalues_from_reciprocal_operators(H_kMM,
                                          S_kMM,
                                          correction_power_kMM=None,
                                          s_power=1 / 2):
    correction_cov_kMM = correction_to_covariant(
        correction_power_kMM,
        S_kMM,
        s_power)
    H_operator_kMM = H_kMM.operator_kMM
    if correction_cov_kMM is not None:
        H_operator_kMM = H_operator_kMM + correction_cov_kMM

    eigenvalues = []
    for H_MM, S_MM in zip(H_operator_kMM, S_kMM.operator_kMM):
        H_MM = (H_MM + H_MM.T.conj()) / 2
        S_MM = (S_MM + S_MM.T.conj()) / 2
        eigenvalues.append(scipy.linalg.eigh(H_MM, S_MM, eigvals_only=True))
    return np.asarray(eigenvalues)


def bandstructure_from_reciprocal_operators(H_kMM,
                                            S_kMM,
                                            bandpath,
                                            reference=0,
                                            correction_power_kMM=None,
                                            s_power=1 / 2):

    eps_kn = eigenvalues_from_reciprocal_operators(
        H_kMM,
        S_kMM,
        correction_power_kMM,
        s_power)

    return BandStructure(
        path=bandpath,
        energies=eps_kn[np.newaxis, :, :],
        reference=reference)


def bandstructure_from_realspace(H_RMM,
                                 S_RMM,
                                 bandpath,
                                 reference=0,
                                 correction_power_RMM=None,
                                 s_power=1 / 2):
    H_kMM = H_RMM.inverse_fourier_transform(bandpath.kpts)
    S_kMM = S_RMM.inverse_fourier_transform(bandpath.kpts)
    correction_power_kMM = None
    if correction_power_RMM is not None:
        correction_power_kMM = correction_power_RMM.inverse_fourier_transform(
            bandpath.kpts)
    return bandstructure_from_reciprocal_operators(
        H_kMM,
        S_kMM,
        bandpath,
        reference,
        correction_power_kMM=correction_power_kMM,
        s_power=s_power)


@dataclass
class BandStructureInterpolationPlot:
    """
    BandStructureInterpolationPlot
    """

    exact_ibz_eps_kn: np.ndarray
    exact_bz_eps_kn: np.ndarray
    bs_shifted_fromRSpace: object
    bs_no_FT: object
    no_shift_bs: object
    k_bz_kc: np.ndarray
    bp_ind: np.ndarray
    point_e_kn: np.ndarray
    calc_ibz: object
    world: object = default_world

    def plot(self,
             emin=-10,
             emax=5,
             filename='bs.png',
             plot_also_unshifted=True,
             plot_also_non_FT=False):
        if self.world.rank != 0:
            return None
        from ase.spectrum.band_structure import BandStructurePlot

        bsp_shifted_fromRSpace = BandStructurePlot(self.bs_shifted_fromRSpace)
        ax = bsp_shifted_fromRSpace.plot(
            show=False, emin=emin, emax=emax, colors='k')

        if plot_also_unshifted:
            bsp_unshifted = BandStructurePlot(self.no_shift_bs)
            bsp_unshifted.plot(ax=ax, emin=emin, emax=emax, colors='b')

        if plot_also_non_FT:
            bsp_no_FT = BandStructurePlot(self.bs_no_FT)
            bsp_no_FT.plot(
                ax=ax,
                emin=emin,
                emax=emax,
                colors='r',
                linestyle='dashed')

        point_x_kn = np.repeat(
            bsp_shifted_fromRSpace.xcoords[self.bp_ind, None],
            self.exact_bz_eps_kn.shape[1],
            axis=1)

        ax.scatter(
            point_x_kn.ravel(),
            self.point_e_kn.ravel(),
            s=16,
            facecolors='none',
            edgecolors='k',
            linewidths=0.7,
            label='Exact mesh values')
        ax.legend(loc='best')
        ax.figure.savefig(filename, dpi=300, bbox_inches='tight')
        return ax
