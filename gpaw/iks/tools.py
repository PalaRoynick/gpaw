from __future__ import annotations
from math import sqrt, pi

import warnings
from collections.abc import MutableMapping
from typing import TYPE_CHECKING

import numpy as np
from gpaw.sphere.gaunt import gaunt, super_gaunt
from gpaw.sphere.lebedev import Y_nL, weight_n
from gpaw.utilities import pack_density, unpack_density
from gpaw.xc import XC
from gpaw.xc.numericalxc import calculate_angular_density
from gpaw.new.backwards_compatibility import FakeDensity

if TYPE_CHECKING:
    from typing import Any

    from gpaw.old.arraydict import ArrayDict
    from gpaw.atom.radialgd import RadialGridDescriptor
    from gpaw.old.density import Density
    from gpaw.old.grid_descriptor import GridDescriptor
    from gpaw.old.logger import GPAWLogger
    from gpaw.setup import Setup, Setups
    from gpaw.typing import Array1D, Array2D, Array3D, Array4D, ArrayND
    from gpaw.old.wavefunctions.base import WaveFunctions
    from gpaw.xc.pawcorrection import PAWXCCorrection
    NaN = float


class LogFormatter:
    def __init__(self, log: GPAWLogger, width: int = 88):
        self.log = log
        self.width = width

    def separator(self, char: str = "-") -> None:
        self.log(char * self.width)

    def boxed_title(self, title: str, char: str = "*") -> None:
        self.separator(char)
        self.log(f"{char}{' ' * (self.width - 2)}{char}")
        self.log(f"{char}{title:^{self.width - 2}}{char}")
        self.log(f"{char}{' ' * (self.width - 2)}{char}")
        self.separator(char)

    def section_title(self, title: str) -> None:
        self.log()
        self.separator()
        self.log(f"{title:^{self.width}}")
        self.separator()
        self.log()

    def print_dict(self, dct: dict[str, Any]) -> None:
        self.log(self.format_dict(dct))

    def format_dict(
        self,
        dct: dict[str, Any],
        joiner: str = "\n",
        key_width: int = 35,
        sep: str = "",
    ) -> str:
        dct_str = []
        for key, value in dct.items():
            if isinstance(value, dict):
                value = self.format_dict(
                    value, joiner=", ", key_width=key_width // len(dct), sep=":"
                )
            dct_str.append(
                self.key_value(key + sep, value, log=False,
                               key_width=key_width)
            )
        return joiner.join(dct_str)

    def key_value(
        self, key: Any, value: Any, key_width: int = 35, log: bool = True
    ) -> str:
        line = f"{key:<{key_width}} {value}"
        if log:
            self.log(line)
        return line

    def flush(self) -> None:
        self.log.fd.flush()


def calculate_diff_overlap_matrix(setup: Setup, w: ArrayND, w_name: str, dD_p: Array2D, mix_paw_error: bool) -> Array3D | NaN:
    if "radial_" in w_name:
        assert w.ndim > 1
        if mix_paw_error:
            return calculate_diff_overlap_matrix_gaunt(setup, w[0], dD_p)
        else:
            return calculate_diff_overlap_matrix_gaunt_separate(setup, w[0], dD_p)
    else:
        assert False, "TODO"
        return calculate_3phiproj_coeffs_lebedev(setup, w, dD_p)


def calculate_3phiproj_coeffs_lebedev(setup: Setup, w_ng: Array2D, dD_p: Array2D) -> Array3D | NaN:
    r"""                                                         __
                    /        ~              ~   ~    ~           \
    Φ_{jk} = sum_i  | drdΩ   φ_k (φ_i φ_j - φ_i φ_j) p_i w(r) *  /_  Δ_ij(r).dD_ij
                    /                                            ij
    """

    if not (hasattr(setup, "data") and hasattr(setup, "xc_correction")):
        return np.nan
    xcc = setup.xc_correction
    rgd = xcc.rgd
    ni = setup.ni  # Number of partial waves
    l_j = setup.l_j  # l-index for each radial function index j
    # (Real) radial functions for the partial waves
    phi_jg = xcc.phi_jg
    phit_jg = xcc.phit_jg
    pt_jg = setup.data.pt_jg
    dn_Lg = np.dot(np.dot(xcc.B_pqL.T, dD_p), (xcc.n_qg - xcc.nt_qg))
    Lmax = len(dn_Lg)
    Phi_ii = np.zeros((ni, ni), dtype=pt_jg[0].dtype)
    for n, Y_L in enumerate(Y_nL):
        for jp, lp in enumerate(l_j):
            c = (2 * lp + 1) * weight_n[n] * w_ng[n]
            i2_counter = 0
            for j2, l2 in enumerate(l_j):
                i1_counter = 0
                dphipt_g = (
                    c
                    * (phi_jg[jp] * phi_jg[j2] - phit_jg[jp] * phit_jg[j2])
                    * pt_jg[jp][: rgd.N]
                )
                for j1, l1 in enumerate(l_j):
                    dphipt_g *= phit_jg[j1]
                    for L in range(Lmax):
                        dphi = np.dot(rgd.r_g**2 * rgd.dr_g *
                                      dphipt_g, dn_Lg[L])
                        for m1 in range(2 * l1 + 1):
                            L1 = l1**2 + m1
                            i1 = i1_counter + m1
                            for m2 in range(2 * l2 + 1):
                                L2 = l2**2 + m2
                                i2 = i2_counter + m2
                                Phi_ii[i1, i2] += dphi * \
                                    Y_L[L] * Y_L[L1] * Y_L[L2]
                    i1_counter += 2 * l1 + 1
                i2_counter += 2 * l2 + 1
    return Phi_ii


def jlist2array(rgd: RadialGridDescriptor, x_jg: list[Array1D]) -> Array2D:
    return np.array([x[: rgd.N] for x in x_jg], dtype=x_jg[0].dtype)


def calculate_3phiproj_coeffs_supergaunt(
    setup: Setup, w_g: Array1D, dD_p: Array2D
) -> Array3D | NaN:
    # deprecated
    if not (hasattr(setup, "data") and hasattr(setup, "xc_correction")):
        return np.nan
    xcc = setup.xc_correction
    rgd = xcc.rgd
    # gaunt coefficients
    Lmax = xcc.B_pqL.shape[-1]
    l_j = setup.l_j
    lmax = max(l_j)
    G_LLL = gaunt(lmax)[..., :Lmax]
    G_LLLL = super_gaunt(lmax)[..., :Lmax]
    # partial waves and projectors
    phi_jg = jlist2array(rgd, xcc.phi_jg)
    phit_jg = jlist2array(rgd, xcc.phit_jg)
    pt_jg = jlist2array(rgd, setup.data.pt_jg)
    # calculate dn^a_L
    dD_Lq = np.dot(xcc.B_pqL.T, dD_p)
    dn_Lg = np.dot(dD_Lq, xcc.n_qg - xcc.nt_qg)
    # dn_qg
    dn_jjg = (
        phi_jg[np.newaxis] * phi_jg[:, np.newaxis]
        - phit_jg[np.newaxis] * phit_jg[:, np.newaxis]
    )
    # map L -> i and j -> i
    L_i = []
    j_i = []
    for j, ll in enumerate(setup.l_j):
        for m in range(2 * ll + 1):
            L = ll**2 + m
            L_i.append(L)
            j_i.append(j)
    # radial integral
    I_jjjL = np.einsum(
        "g, ijg,kg, ig, Lg -> ijkL",
        (rgd.r_g**2) * rgd.dr_g * w_g,
        dn_jjg,
        phit_jg,
        pt_jg,
        dn_Lg,
    )
    # angular integral
    Phi_iii = np.einsum(
        "ijkL, ijM, ikLM ->ijk",
        I_jjjL[j_i][:, j_i][:, :, j_i],
        G_LLL[L_i][:, L_i],
        G_LLLL[L_i][:, L_i],
    )
    return Phi_iii.sum(axis=0).T


def calculate_diff_overlap_matrix_gaunt(
    setup: Setup, w_g: Array1D, dD_p: Array2D
) -> Array3D | NaN:
    if not hasattr(setup, "xc_correction"):
        return np.nan
    xcc = setup.xc_correction
    if xcc is None:
        return np.nan
    rgd = xcc.rgd
    # gaunt coefficients
    Lmax = xcc.B_pqL.shape[-1]
    l_j = setup.l_j
    lmax = max(l_j)
    G_LLL = gaunt(lmax)[..., :Lmax]
    # partial waves and projectors
    phi_jg = jlist2array(rgd, xcc.phi_jg)
    phit_jg = jlist2array(rgd, xcc.phit_jg)
    # calculate dn^a_L
    dD_Lq = np.dot(xcc.B_pqL.T, dD_p)
    dn_Lg = np.dot(dD_Lq, xcc.n_qg - xcc.nt_qg)
    # dn_qg
    dn_jjg = (
        phi_jg[np.newaxis] * phi_jg[:, np.newaxis]
        - phit_jg[np.newaxis] * phit_jg[:, np.newaxis]
    )
    ni = setup.ni
    Phi_ii = np.zeros((ni, ni))
    i1_counter = 0
    for j1, l1 in enumerate(l_j):
        i2_counter = 0
        for j2, l2 in enumerate(l_j):
            # no 4pi because of G_LLL
            I_L = np.dot(dn_Lg * w_g * rgd.r_g**2 * rgd.dr_g, dn_jjg[j1, j2])
            for m1 in range(2 * l1 + 1):
                L1 = l1**2 + m1
                i1 = i1_counter + m1
                for m2 in range(2 * l2 + 1):
                    L2 = l2**2 + m2
                    i2 = i2_counter + m2
                    Phi_ii[i1, i2] += np.dot(G_LLL[L1, L2], I_L)
            i2_counter += 2 * l2 + 1
        i1_counter += 2 * l1 + 1
    return Phi_ii


def calculate_diff_overlap_matrix_gaunt_separate(
    setup: Setup, w_g: Array1D, dD_p: Array2D
) -> Array3D | NaN:
    if not hasattr(setup, "xc_correction"):
        return np.nan
    xcc = setup.xc_correction
    if xcc is None:
        return np.nan
    rgd = xcc.rgd
    # gaunt coefficients
    Lmax = xcc.B_pqL.shape[-1]
    l_j = setup.l_j
    lmax = max(l_j)
    G_LLL = gaunt(lmax)[..., :Lmax]
    # partial waves and projectors
    phi_jg = jlist2array(rgd, xcc.phi_jg)
    phit_jg = jlist2array(rgd, xcc.phit_jg)
    # calculate dn^a_L
    dD_Lq = np.dot(xcc.B_pqL.T, dD_p)
    n_Lg = np.dot(dD_Lq, xcc.n_qg)
    nt_Lg = np.dot(dD_Lq, xcc.nt_qg)

    # dn_qg
    n_jjg = phi_jg[np.newaxis] * phi_jg[:, np.newaxis]
    nt_jjg = phit_jg[np.newaxis] * phit_jg[:, np.newaxis]

    ni = setup.ni
    Phi_ii = np.zeros((ni, ni))
    i1_counter = 0
    for j1, l1 in enumerate(l_j):
        i2_counter = 0
        for j2, l2 in enumerate(l_j):
            for dn_Lg, dn_jjg in zip([n_Lg, nt_Lg], [n_jjg, nt_jjg]):
                I_L = np.dot(
                    dn_Lg * rgd.dr_g * w_g * rgd.r_g**2, dn_jjg[j1, j2]
                )
                for m1 in range(2 * l1 + 1):
                    L1 = l1**2 + m1
                    i1 = i1_counter + m1
                    for m2 in range(2 * l2 + 1):
                        L2 = l2**2 + m2
                        i2 = i2_counter + m2
                        Phi_ii[i1, i2] += np.dot(G_LLL[L1, L2], I_L)
            i2_counter += 2 * l2 + 1
        i1_counter += 2 * l1 + 1
    return Phi_ii

# def calculate_3phiproj_coeffs_gaunt2(setup: Setup, w_g: Array1D, dD_p: Array2D) -> Array3D | NaN:
#     # slower than the explicit loop above
#     if not (hasattr(setup, "data") and hasattr(setup, "xc_correction")):
#         return np.nan
#     xcc = setup.xc_correction
#     rgd = xcc.rgd
#     # gaunt coefficients
#     Lmax = xcc.B_pqL.shape[-1]
#     l_j = setup.l_j
#     lmax = max(l_j)
#     G_LLL = gaunt(lmax)[..., :Lmax]
#     # partial waves and projectors
#     phi_jg = jlist2array(rgd, xcc.phi_jg)
#     phit_jg = jlist2array(rgd, xcc.phit_jg)
#     pt_jg = jlist2array(rgd, setup.data.pt_jg)
#     # calculate dn^a_L
#     dD_Lq = np.dot(xcc.B_pqL.T, dD_p)
#     dn_Lg = np.dot(dD_Lq, xcc.n_qg - xcc.nt_qg)
#     # dn_qg
#     dn_jjg = (phi_jg[np.newaxis] * phi_jg[:, np.newaxis] -
#               phit_jg[np.newaxis] * phit_jg[:, np.newaxis])
#     ni = setup.ni
#     Phi_ii = np.zeros((ni, ni))
#     # map L -> i and j -> i
#     L_i = []
#     j_i = []
#     c_j = []
#     for j, l in enumerate(l_j):
#         c_j.append((2 * l + 1) / (4 * np.pi))
#         for m in range(2 * l + 1):
#             L = l**2 + m
#             L_i.append(L)
#             j_i.append(j)
#     # sum over j'
#     I_jjL = np.einsum('g, Lg, ig, kjg, kg, k -> ijL', w_g * rgd.r_g**2 * rgd.dr_g, dn_Lg,
#                       phit_jg, dn_jjg, pt_jg, c_j)
#     # sum over L
#     Phi_ii = np.einsum('ijL, ijL->ij',
#                        G_LLL[L_i][:, L_i], I_jjL[j_i][:, j_i])
#     return Phi_ii


def get_atomic_ae_dens_radial(D_sp: Array2D, xcc: PAWXCCorrection) -> Array2D:
    n_sLg = calculate_angular_density(xcc, D_sp, xcc.n_qg, xcc.nc_g)
    return n_sLg[:, :1] / sqrt(4 * pi)


def get_atomic_ps_dens_radial(D_sp: Array2D, xcc: PAWXCCorrection) -> Array2D:
    n_sLg = calculate_angular_density(xcc, D_sp, xcc.nt_qg, xcc.nct_g)
    return n_sLg[:, :1] / sqrt(4 * pi)


def get_atomic_dens_lebedev(D_sp: Array2D, xcc: PAWXCCorrection) -> Array3D:
    n_sLg = calculate_angular_density(xcc, D_sp, xcc.n_qg, xcc.nc_g)
    Lmax = n_sLg.shape[1]
    n_sng = np.einsum("sLg,nL->sng", n_sLg, Y_nL[:, :Lmax])
    return n_sng


VALID_WEIGHTS = (
    "uniform", "zero", "density_sqrt", "density", "density_squared",
    "inverse_density", "inverse_total_density", "inverse_density_squared", "mask", "fr"
)


def name_to_weight(weights, n_sg):
    if isinstance(weights, np.ndarray):
        return weights

    if weights not in VALID_WEIGHTS:
        raise ValueError(
            f"weights must either be an array or one of the predefined weights: {', '.join(VALID_WEIGHTS)}"
        )

    weights_sg = np.zeros_like(n_sg)
    if weights == "uniform":
        weights_sg[:] = 1.0
    elif weights == "zero":
        pass  # Already initialized to zeros
    elif weights == "density_sqrt":
        weights_sg[:] = np.sqrt(n_sg)
    elif weights == "density":
        weights_sg[:] = n_sg
    elif weights == "density_squared":
        weights_sg[:] = n_sg**2
    elif weights == "inverse_density":
        weights_sg[:] = 1.0 / (n_sg + 1e-6)
    elif weights == "inverse_total_density":
        weights_sg[:] = 1.0 / (n_sg.sum(axis=0) + 1e-6)
    elif weights == "inverse_density_squared":
        weights_sg[:] = 1.0 / (n_sg**2 + 1e-12)
    elif weights == "mask":
        weights_sg[n_sg > 1e-6] = 1.0
    elif weights == "fr":
        weights_sg[:] = n_sg / (n_sg + 1e-6)

    return weights_sg


def get_atomic_weights(
    weights: str | MutableMapping, setups: Setups, D_asp: ArrayDict,
) -> tuple[MutableMapping, str]:
    nspins = D_asp.shapes_a[0][0]
    if isinstance(weights, MutableMapping):
        for a, D_sp in D_asp.items():
            w = weights[a]
            if isinstance(w, str):
                w_clean = w.removeprefix("radial_")
                n_s1g = get_atomic_ae_dens_radial(
                    D_asp[a], setups[a].xc_correction)
                if w == 'density_ratio':
                    nt_s1g = get_atomic_ps_dens_radial(
                        D_asp[a], setups[a].xc_correction)
                    weights[a] = nt_s1g / n_s1g
                else:
                    weights[a] = name_to_weight(w_clean, n_s1g)
        return weights, "radial_custom"

    if "radial_" in weights:
        get_atomic_dens = get_atomic_ae_dens_radial

        def shapes_a(a):
            if setups[a].xc_correction is None:
                return (nspins, 1, 1)
            rgd = setups[a].xc_correction.rgd
            return (nspins, 1, rgd.N)

    else:
        warnings.warn(
            "Using angular atomic weights is experimental. "
            f"Use atomic_weights='radial_{weights}' instead."
        )
        get_atomic_dens = get_atomic_dens_lebedev

        def shapes_a(a):
            if setups[a].xc_correction is None:
                return (nspins, 1, 1)
            rgd = setups[a].xc_correction.rgd
            return (nspins, len(Y_nL), rgd.N)

    weights_asng = D_asp.partition.arraydict(shapes_a, float)
    weights_clean = weights.removeprefix("radial_")
    if weights_clean == "uniform":
        for w_sng in weights_asng.values():
            w_sng.fill(1.0)
        weights_asng.check_consistency()
        return weights_asng, weights
    elif weights_clean == "zero":
        for w_sng in weights_asng.values():
            w_sng.fill(0.0)
        weights_asng.check_consistency()
        return weights_asng, weights

    for a, D_sp in D_asp.items():
        if setups[a].xc_correction is None:
            continue
        n_sng = get_atomic_dens(D_sp, setups[a].xc_correction)
        weights_asng[a][:] = name_to_weight(weights_clean, n_sng)
    weights_asng.check_consistency()
    return weights_asng, weights


def get_weights(
    weights: Array4D | str, nt_sG: Array4D, gd: GridDescriptor,
) -> tuple[Array4D, str]:
    weights_sG = gd.empty(len(nt_sG))
    nt_sG = nt_sG.clip(0.0)
    weights_name = weights
    if isinstance(weights, np.ndarray):
        weights_sG[:] = weights
        weights_name = "custom"
    else:
        weights_sG[:] = name_to_weight(weights, nt_sG)
    return weights_sG, weights_name


def get_core_regularization_weights(setups, power=1):
    """
        w = n_c / max(n_c)
        if n_c == 0 the weights are only one for regions r <= 2rcut
    """
    w_ag = {}
    for a, setup in enumerate(setups):
        xcc = setup.xc_correction
        if xcc is None:
            w_ag[a] = np.zeros((1,), dtype=float)  # AE
            continue
        nc_g = xcc.nc_g
        max_nc = nc_g.max()
        if max_nc > 1e-6:
            w_ag[a] = (nc_g / max_nc)**power
        else:
            w = np.zeros_like(nc_g)
            rcut2 = 2 * max(setup.rcut_j)
            w[xcc.rgd.r_g <= 2 * rcut2] = 1.0
            w_ag[a] = w
    return w_ag


def get_paw_boundary_weights(setups: Setups, p: float = 1.0, t: float = 2.0, c=1.5) -> dict[int, Array1D]:
    """
        E(r) = 1.0 - [0.5 * (1 + cos(pi * (r / (c * rc))**t))]**p
    """
    w_ag = {}

    for a, setup in enumerate(setups):
        xcc = setup.xc_correction

        if xcc is None:
            w_ag[a] = np.zeros(1)
            continue

        if p == 0.0:
            w_ag[a] = xcc.rgd.zeros()
            continue

        r_g = xcc.rgd.r_g
        rc = c * max(setup.rcut_j)

        x_g = np.clip(r_g / rc, 0.0, 1.0)

        E_g = 1.0 - (0.5 * (1.0 + np.cos(pi * x_g**t)))**p
        w_ag[a] = E_g

    return w_ag


def calculate_core_atomic_potentials(D_asp, setups, xc, nspins):
    v_a = {}
    vt_a = {}
    xc = XC(xc)
    # assert xc.type in {'LDA', 'GGA'}
    for a, D_sp in D_asp.items():
        setup = setups[a]
        xcc = setup.xc_correction
        if xcc is None:
            # AE setup
            v_a[a] = np.zeros(nspins, float)
            continue
        rgd = xcc.rgd
        # v_a
        n_sg = get_atomic_ae_dens_radial(D_sp, xcc).squeeze(axis=1)
        v = np.zeros_like(n_sg)
        xc.calculate_spherical(rgd, n_sg, v)
        v_a[a] = v
    for a, setup in enumerate(setups):
        # vt_a
        xcc = setup.xc_correction
        if xcc is None:
            # AE setup
            vt_a[a] = np.zeros(nspins, float)
            continue
        rgd = xcc.rgd
        nct_sg = rgd.zeros(nspins)
        nct_sg[:] = xcc.nct_g / nspins
        vt = np.zeros_like(nct_sg)
        xc.calculate_spherical(rgd, nct_sg, vt)
        vt_a[a] = vt

    return v_a, vt_a


def fakdensity2old(fake_density, old_density, old_hamiltonian):
    args = [fake_density.gd,
            fake_density.finegd,
            fake_density.nspins,
            fake_density.collinear,
            old_density.charge,
            old_density.redistributor,
            old_density.background_charge]
    from gpaw.old.pw.density import ReciprocalSpaceDensity
    if isinstance(old_density, ReciprocalSpaceDensity):
        args.insert(0, old_density.pd2.ecut)
    density = old_density.__class__(*args)
    density.initialize(old_density.setups, old_density.timer,
                       old_density.magmom_av, old_density.hund)
    density.set_positions(old_hamiltonian.spos_ac, old_density.atom_partition)
    density.initialize_directly_from_arrays(
        fake_density.nt_sG, None, fake_density.D_asp.copy())
    return density


def transform_rdm1(gd: GridDescriptor, wfs: WaveFunctions, mo_iG: Array4D, rdm1_sij: Array3D, interpolater=None) -> Array3D:
    """
    Transforms density matrices rdm1_sij obtained in the set of molecular orbitals mo_iG to the current wfs orbitals basis (only works in serial).
    """
    assert wfs.kd.gamma
    # maybe we should take the AE wfs instead?
    # if wfs.mode == 'pw':
    # mo_iG = np.asarray([wfs.pd.fft(gto_G) for gto_G in mo_iG])
    nbands = wfs.bd.nbands
    P_snm = np.empty((len(wfs.kpt_u), nbands, nbands), dtype=wfs.dtype)
    for kpt in wfs.kpt_u:
        psit_nG = kpt.psit_nG

        if wfs.mode == 'pw':
            psit_nG = np.asarray([wfs.pd.ifft(psit_G) for psit_G in psit_nG])
        if interpolater is not None:
            psit_nG = np.asarray([interpolater(psit_G) for psit_G in psit_nG])
        if not isinstance(psit_nG, np.ndarray):
            # ArrayReader
            psit_nG = np.asarray(psit_nG)
        # C_ni = wfs.integrate(psit_nG, mo_iG)
        C_ni = gd.integrate(psit_nG, mo_iG)
        P_snm[kpt.s] = np.einsum('ni, ij, mj -> nm',
                                 C_ni, rdm1_sij[kpt.s], C_ni.conj())
    return P_snm


def update_density_from_density_matrix(
        density: Density | FakeDensity, wfs: WaveFunctions, P_ksnn: Array4D | None, symmetrize: bool = True
) -> None:
    if P_ksnn is None:
        return
    if P_ksnn.ndim == 3:
        assert wfs.kd.gamma
        P_ksnn = P_ksnn[np.newaxis]
    nt_sG = density.gd.zeros(density.nspins)
    D_asp = density.setups.empty_atomic_matrix(
        density.ncomponents, density.atom_partition
    )
    for a in D_asp:
        D_asp[a][:] = 0.0
    for kpt in wfs.kpt_u:
        psit_nG = wfs.bd.collect(kpt.psit_nG, broadcast=True)
        if wfs.mode == "pw":
            psit_nG = np.asarray([wfs.pd.ifft(psit_G, q=kpt.q)
                                  for psit_G in psit_nG])
        nt_sG[kpt.s] += (
            kpt.weightk
            * np.einsum(
                "n..., nm, m...->...", psit_nG.conj(
                ), P_ksnn[kpt.k, kpt.s], psit_nG
            ).real
        )
        P_ani = kpt.P_ani
        for a in D_asp:
            P_ni = wfs.bd.collect(P_ani[a], broadcast=True)
            D_ii = np.einsum("ni, nm, mj -> ij",
                             P_ni.conj(), P_ksnn[kpt.k, kpt.s], P_ni)
            D_asp[a][kpt.s] += kpt.weightk * np.real_if_close(pack_density(D_ii))

    wfs.kd.comm.sum(nt_sG)
    for a in D_asp:
        wfs.kd.comm.sum(D_asp[a])

    if symmetrize:
        for nt_G in nt_sG:
            wfs.kd.symmetry.symmetrize(nt_G, density.gd)
        wfs.symmetrize_atomic_density_matrices(D_asp)

    if density.nct_G is not None:
        nt_sG[:] += density.nct_G
    density.initialize_directly_from_arrays(nt_sG, density.nt_vG, D_asp)


def check_bz_density_matrix_symmetry(
        Pbz_ksnn: Array4D,
        wfs: WaveFunctions,
        tol: float = 1e-4,
) -> None:
    kd = wfs.kd
    assert len(Pbz_ksnn) == len(kd.bz2ibz_k)

    hermiticity_error = np.max(np.abs(
        Pbz_ksnn - Pbz_ksnn.swapaxes(-1, -2).conj()
    ))
    assert hermiticity_error < tol, f'{hermiticity_error}'

    symmetry_error = 0.0

    for K, k in enumerate(kd.bz2ibz_k):
        K0 = kd.ibz2bz_k[k]
        P0_snn = Pbz_ksnn[K0]

        time_reversal = (
            bool(kd.time_reversal_k[K])
            != bool(kd.time_reversal_k[K0])
        )
        if time_reversal:
            P0_snn = P0_snn.conj()

        error = np.max(np.abs(Pbz_ksnn[K] - P0_snn))
        symmetry_error = max(symmetry_error, error)

    assert symmetry_error < tol, f'{symmetry_error:.3e}'


def bz_density_matrix_to_ibz(Pbz_ksnn: Array4D, wfs: WaveFunctions) -> Array4D:
    kd = wfs.kd
    assert len(Pbz_ksnn) == len(kd.bz2ibz_k)

    Pibz_ksnn = np.zeros(
        (kd.nibzkpts, *Pbz_ksnn.shape[1:]), dtype=Pbz_ksnn.dtype)

    for k in range(kd.nibzkpts):
        Pibz_ksnn[k] = Pbz_ksnn[kd.ibz2bz_k[k]]
    return Pibz_ksnn


def update_density_from_density_matrix_diff(
        density: Density, wfs: WaveFunctions, dP_ksnn: Array4D, bands: np.ndarray | None = None, check_symmetry: bool = False,
):
    assert dP_ksnn.ndim == 4
    assert wfs.bd.comm.size == 1

    if bands is None:
        bands = np.arange(0, wfs.bd.nbands)

    dnt_sG = density.gd.zeros(density.nspins)
    dD_asp = density.setups.empty_atomic_matrix(
        density.ncomponents, density.atom_partition
    )
    for a in dD_asp:
        dD_asp[a][:] = 0.0

    if len(dP_ksnn) == len(wfs.kd.bz2ibz_k):
        if check_symmetry:
            check_bz_density_matrix_symmetry(dP_ksnn, wfs)
        dP_ksnn = bz_density_matrix_to_ibz(dP_ksnn, wfs)

    for kpt in wfs.kpt_u:
        # psit_nG = wfs.bd.collect(kpt.psit_nG, broadcast=True)[bands]
        psit_nG = np.array(kpt.psit_nG)[bands]
        if wfs.mode == "pw":
            psit_nG = np.asarray([wfs.pd.ifft(psit_G, q=kpt.q)
                                  for psit_G in psit_nG])
        dnt_sG[kpt.s] += (
            kpt.weightk
            * np.einsum(
                "n..., nm, m...->...", psit_nG.conj(
                ), dP_ksnn[kpt.k, kpt.s], psit_nG
            ).real
        )
        P_ani = kpt.P_ani
        for a in dD_asp:
            P_ni = wfs.bd.collect(P_ani[a], broadcast=True)
            D_ii = np.einsum("ni, nm, mj -> ij",
                             P_ni[bands].conj(), dP_ksnn[kpt.k, kpt.s], P_ni[bands])
            dD_asp[a][kpt.s] += kpt.weightk * \
                np.real_if_close(pack_density(D_ii))
    # mpi
    wfs.kd.comm.sum(dnt_sG)
    for a in dD_asp:
        wfs.kd.comm.sum(dD_asp[a])

    # symmetrize
    for dnt_G in dnt_sG:
        wfs.kd.symmetry.symmetrize(dnt_G, density.gd)
    wfs.symmetrize_atomic_density_matrices(dD_asp)

    # update
    D_asp = density.setups.empty_atomic_matrix(
        density.ncomponents, density.atom_partition
    )
    for a in D_asp:
        D_asp[a][:] = density.D_asp[a] + dD_asp[a]

    density.initialize_directly_from_arrays(
        density.nt_sG + dnt_sG, density.nt_vG, D_asp)
