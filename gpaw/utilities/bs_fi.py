import numpy as np
from ase.spectrum.band_structure import BandStructure

# Future TODO: def bandstructure_interpolate(calc, bandpath, shift_kn):
from dataclasses import dataclass
import scipy.linalg
from ase import Atoms
import scipy
import matplotlib.pyplot as plt
from gpaw.calculator import GPAW

def mpower(O_MM, power):
    if power == 0:
        return np.eye(len(O_MM))
    elif power == 1 / 2:
        return scipy.linalg.sqrtm(O_MM)
    elif power == 1:
        return O_MM
    raise ValueError("Provided s_power is not implemented. Use 1, 1/2 or 0.")


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

    def fourier_transform(self, cutoff, N_c, atoms=None):
        operator_KMM = np.array(self.construct())

        from ase.neighborlist import NeighborList

        if atoms is None:
            atoms = Atoms("H", cell=self.cell_cv, pbc=True)
        else:
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

    def fourier_interpolate(self, higher_k_mesh, cutoff, N_c):
        # Take kMM operator, and convert to real space
        operator_RMM = self.fourier_transform(cutoff, N_c)
        # Convert back to fourier space with higher k-mesh
        return operator_RMM.inverse_fourier_transform(higher_k_mesh)


@dataclass
class GPAWLCAOOperator(LCAOOperator):
    """Construct"""

    calc_bz: GPAW

    @property
    def operator_kMM(self):
        return np.array(
            [self.get_operator(k) for k in range(len(self.calc_bz.wfs.kpt_u))]
        )

    @classmethod
    def create(cls, calc_bz, **kwargs):
        ibzk_qc = calc_bz.wfs.kd.ibzk_qc
        kpt_kc = [ibzk_qc[kpt.q] for kpt in calc_bz.wfs.kpt_u]
        cell_cv = calc_bz.atoms.cell
        return cls(kpt_kc, cell_cv, calc_bz, **kwargs)

    def get_operator(self, k: int):
        return self.build(
            self.calc_bz.wfs.kpt_u[k].S_MM, self.calc_bz.wfs.kpt_u[k].C_nM, k
        )

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


def operators_from_calc(calc_bz):
    # Get the full BZ Hamiltonian
    from gpaw.lcao.tools import get_lcao_hamiltonian

    H_skMM, S_kMM = get_lcao_hamiltonian(calc_bz)
    assert len(H_skMM) == 1
    k_bz = calc_bz.get_ibz_k_points()
    #  Reciprocoal LCAO Operator and fourier trafo to real space H_R
    cell_cv = calc_bz.atoms.cell
    hamiltonian_kMM = H_skMM[0]
    H_kMM = ReciprocalLCAOOperator(
        kpt_kc=k_bz, cell_cv=cell_cv, operator_kMM=hamiltonian_kMM
    )
    #
    overlap_kMM = S_kMM
    S_kMM = ReciprocalLCAOOperator(
        kpt_kc=k_bz, cell_cv=cell_cv, operator_kMM=overlap_kMM
    )
    #
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
                H_MM, S_MM
            )  # this here is C_Mn not C_nM thats why
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
            operator_kMM=np.array(scissor_kMM),
        )


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


def create_realspace_lcao_operators(calc_ibz, cutoff, shift_kn, s_power):
    # Setup BZ operators needed to interpolate the scissor correction.
    calc_bz = calc_ibz.fixed_density(
        symmetry="off",
        **(

            dict(
                nbands=shift_kn.shape[1],
                convergence={"bands": shift_kn.shape[1]})
            if shift_kn is not None
            else {}
        ),
    )

    scissor_RMM = None
    if shift_kn is None:
        return calc_bz, scissor_RMM

    H_kMM, S_kMM = operators_from_calc(calc_bz)
    N_c = calc_bz.wfs.kd.N_c

    reciprocal = ReciprocalSpace.from_gpaw_calculator(calc_ibz)
    delta_Kn = reciprocal.ibz2bz(shift_kn)
    scissor_kMM = ScissorOperator.from_operators(
        H_kMM,
        S_kMM,
        delta_Kn,
        s_power)
    scissor_RMM = scissor_kMM.fourier_transform(
        cutoff=cutoff, atoms=calc_bz.atoms, N_c=N_c
    )
    return calc_bz, scissor_RMM


def exact_lcao_operators(calc, kpts, shift_kn=None):
    kwargs = {}
    if shift_kn is not None:
        nbands = shift_kn.shape[1]
        kwargs.update(nbands=nbands, convergence={"bands": nbands})

    calc_bandpath = calc.fixed_density(
        symmetry="off",
        kpts=kpts,
        **kwargs,
    )
    H_kMM, S_kMM = operators_from_calc(calc_bandpath)
    assert np.allclose(H_kMM.kpt_kc, kpts)
    assert np.allclose(S_kMM.kpt_kc, kpts)
    return H_kMM, S_kMM


def create_bandstructure(
    calc,
    bandpath,
    cutoff=13,
    shift_kn=None,
    s_power=1 / 2,
):

    scissor_power_RMM = None
    if shift_kn is not None:
        _calc_bz, scissor_power_RMM = create_realspace_lcao_operators(
            calc,
            cutoff=cutoff,
            shift_kn=shift_kn,
            s_power=s_power,
        )

    H_bandpath_kMM, S_bandpath_kMM = exact_lcao_operators(
        calc,
        bandpath.kpts,
        shift_kn=shift_kn,
    )

    scissor_power_bandpath_kMM = None
    if shift_kn is not None:
        scissor_power_bandpath_kMM = (
            scissor_power_RMM.inverse_fourier_transform(bandpath.kpts)
        )
    return bandstructure_from_reciprocal_operators(
        H_bandpath_kMM,
        S_bandpath_kMM,
        bandpath,
        correction_power_kMM=scissor_power_bandpath_kMM,
        s_power=s_power,
    )


def create_non_FT_bandstructure(calc, bandpath):
    calc_bz = calc.fixed_density(symmetry="off", kpts=bandpath.kpts)
    eps_kn = []
    for kpt in range(len(bandpath.kpts)):
        eps_n = calc_bz.get_eigenvalues(kpt=kpt)
        eps_kn.append(eps_n)
    eps_kn = np.array(eps_kn)
    band_structure = BandStructure(
        path=bandpath, energies=eps_kn[np.newaxis, :, :], reference=0
    )
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


def plot_G0W0_bandstructure(calc_gs, bandpath, filename_g0w0_pw_from_lcao):
    import pickle
    results = pickle.load(open(f'{filename_g0w0_pw_from_lcao}', 'rb'))
    delta_ibz_kn = (
        results["qp"][0, :, :] - results["eps"][0, :, :]
    )
    bandpath = calc_gs.atoms.cell.bandpath(npoints=250)
    cutoff = 100
    bandstructure_plot(
        calc_gs,
        bandpath,
        cutoff=cutoff,
        delta_ibz_kn=delta_ibz_kn,
        emin=-10,
        emax=20,
        filename="bs.png",
        plot_also_unshifted=True,
    )
    plt.clf()


def bandstructure_plot(
    calc_ibz,
    bandpath,
    cutoff=14,
    delta_ibz_kn=None,
    emin=-10,
    emax=5,
    filename="bs.png",
    plot_also_unshifted=True,
    plot_also_non_FT=False,
    s_power=1 / 2,
):

    bs_interpolate_fromRspace = bandstructure(
        calc_ibz, cutoff, bandpath, delta_ibz_kn, s_power=s_power
    )
    return bs_interpolate_fromRspace.plot(
        emin=emin,
        emax=emax,
        filename=filename,
        plot_also_unshifted=plot_also_unshifted,
        plot_also_non_FT=plot_also_non_FT,
    )


def bandstructure(calc_ibz, cutoff, bandpath, delta_ibz_kn, s_power=1 / 2):
    bs_shifted_fromRSpace = create_bandstructure(
        calc=calc_ibz,
        bandpath=bandpath,
        cutoff=cutoff,
        shift_kn=delta_ibz_kn,
        s_power=s_power,
    )
    bs_shifted_fromRSpace.write(f"bs_fromRSpace_{cutoff}.json")
    reciprocal = ReciprocalSpace.from_gpaw_calculator(calc_ibz)

    k_ibz_kc = calc_ibz.get_ibz_k_points()

    if delta_ibz_kn.ndim == 2:  # delta_kn
        exact_ibz_eps_kn = np.array(
            [
                calc_ibz.get_eigenvalues(kpt=kpt)[
                    : delta_ibz_kn.shape[1]
                ]  # if eps_kn more bands than correction
                + delta_ibz_kn[kpt]
                for kpt in range(len(k_ibz_kc))
            ]
        )
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
                for kpt in range(len(k_ibz_kc))
            ]
        )

    k_bz_kc = reciprocal.kpt_Kc
    exact_bz_eps_kn = reciprocal.ibz2bz(exact_ibz_eps_kn)

    no_shift_bs = create_bandstructure(
        calc_ibz,
        bandpath,
        cutoff=cutoff,
        shift_kn=None)

    bs_no_FT = create_non_FT_bandstructure(calc_ibz, bandpath)

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
    )


def correction_to_covariant(correction_power_kMM, S_kMM, s_power):
    if correction_power_kMM is None:
        return None
    if s_power == 1:
        return correction_power_kMM.operator_kMM

    correction_cov_kMM = []
    for S_MM, correction_power_MM in zip(
        S_kMM.operator_kMM, correction_power_kMM.operator_kMM
    ):
        S_MM_pow = mpower(S_MM, 1 - s_power)
        correction_cov_kMM.append(S_MM_pow @ correction_power_MM @ S_MM_pow)
    return np.array(correction_cov_kMM)


def eigenvalues_from_reciprocal_operators(
    H_kMM,
    S_kMM,
    correction_power_kMM=None,
    s_power=1 / 2,
):
    correction_cov_kMM = correction_to_covariant(
        correction_power_kMM,
        S_kMM,
        s_power,
    )
    H_operator_kMM = H_kMM.operator_kMM
    if correction_cov_kMM is not None:
        H_operator_kMM = H_operator_kMM + correction_cov_kMM

    eigenvalues = []
    for H_MM, S_MM in zip(H_operator_kMM, S_kMM.operator_kMM):
        H_MM = (H_MM + H_MM.T.conj()) / 2
        S_MM = (S_MM + S_MM.T.conj()) / 2
        eigenvalues.append(scipy.linalg.eigh(H_MM, S_MM, eigvals_only=True))
    return np.asarray(eigenvalues)


def bandstructure_from_reciprocal_operators(
    H_kMM,
    S_kMM,
    bandpath,
    reference=0,
    correction_power_kMM=None,
    s_power=1 / 2,
):

    eps_kn = eigenvalues_from_reciprocal_operators(
        H_kMM,
        S_kMM,
        correction_power_kMM,
        s_power,
    )

    return BandStructure(
        path=bandpath,
        energies=eps_kn[np.newaxis, :, :],
        reference=reference,
    )


def bandstructure_from_realspace(
    H_RMM,
    S_RMM,
    bandpath,
    reference=0,
    correction_power_RMM=None,
    s_power=1 / 2
):
    H_kMM = H_RMM.inverse_fourier_transform(bandpath.kpts)
    S_kMM = S_RMM.inverse_fourier_transform(bandpath.kpts)
    correction_power_kMM = None
    if correction_power_RMM is not None:
        correction_power_kMM = correction_power_RMM.inverse_fourier_transform(
            bandpath.kpts
        )
    return bandstructure_from_reciprocal_operators(
        H_kMM,
        S_kMM,
        bandpath,
        reference,
        correction_power_kMM=correction_power_kMM,
        s_power=s_power,
    )


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

    def plot(
        self,
        emin=-10,
        emax=5,
        filename="bs.png",
        plot_also_unshifted=True,
        plot_also_non_FT=False,
    ):
        from ase.spectrum.band_structure import BandStructurePlot

        bsp_shifted_fromRSpace = BandStructurePlot(self.bs_shifted_fromRSpace)
        ax = bsp_shifted_fromRSpace.plot(
            show=False, emin=emin, emax=emax, colors="k")

        if plot_also_unshifted:
            bsp_unshifted = BandStructurePlot(self.no_shift_bs)
            bsp_unshifted.plot(ax=ax, emin=emin, emax=emax, colors="b")

        if plot_also_non_FT:
            bsp_no_FT = BandStructurePlot(self.bs_no_FT)
            bsp_no_FT.plot(
                ax=ax,
                emin=emin,
                emax=emax,
                colors="r",
                linestyle="dashed")

        point_x_kn = np.repeat(
            bsp_shifted_fromRSpace.xcoords[self.bp_ind, None],
            self.exact_bz_eps_kn.shape[1],
            axis=1,
        )

        ax.scatter(
            point_x_kn.ravel(),
            self.point_e_kn.ravel(),
            s=16,
            facecolors="none",
            edgecolors="k",
            linewidths=0.7,
            label="Exact mesh values",
        )
        ax.legend(loc="best")
        ax.figure.savefig(filename, dpi=300, bbox_inches="tight")
        return ax
