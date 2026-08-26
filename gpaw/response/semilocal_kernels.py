"""Semilocal (GGA-level) adiabatic XC kernels for the plane-wave
response code.

The semilocal fxc operator is represented exactly by a symmetric
matrix of per-point coefficient fields c_ab(r) over the derivative
slots (1, d_x, d_y, d_z) [plus tau for meta-GGAs, which additionally
requires kinetic pair-density response functions and is not wired
here], so that

    K_xc(G, G') = (1/V0) sum_ab fac_a(q+G)* FT[c_ab](G-G') fac_b(q+G')

with fac_1 = 1 and fac_i = 1j (q+G)_i.  The coefficient fields come
from the machine-generated module gpaw/xckernel_fxc.py (emitted by
the xckernel library) with Libxc second-derivative arrays obtained
through pylibxc; the (G-G') bookkeeping reuses the adiabatic-LDA
kernel machinery.

Limitations of this first version: spin-paired only; the coefficient
fields are built from the smooth pseudo density (no PAW corrections
to the kernel yet); no tau channel (LDA/GGA functionals only).
"""

from functools import partial

import numpy as np
from ase.units import Bohr

from gpaw.fd_operators import Gradient
from gpaw.response.dyson import PWKernel
from gpaw.response.fxc_kernels import AdiabaticFXCCalculator
from gpaw.response.localft import LocalPAWFTEngine, fft_from_grid


class SemilocalFXCKernel(PWKernel):
    """K_xc(G, G') assembled from coefficient-field Fourier transforms
    and reciprocal-vector factors."""

    def __init__(self, c_ab_dG, dG_K, GG_shape, Gv, volume):
        self._c_ab_dG = c_ab_dG  # {(slot, slot): c(dG) complex array}
        self._dG_K = dG_K
        self._GG_shape = GG_shape
        self._Gv = Gv  # (nG, 3) q+G Cartesian
        self._volume = volume

    def get_number_of_plane_waves(self):
        return self._GG_shape[0]

    def _factor(self, slot):
        if slot == "rho":
            return np.ones(self._GG_shape[0], dtype=complex)
        v = {"x": 0, "y": 1, "z": 2}[slot]
        return 1j * self._Gv[:, v]

    def get_Kxc_GG(self):
        K_GG = np.zeros(self._GG_shape, dtype=complex)
        for (a, b), c_dG in self._c_ab_dG.items():
            C_GG = c_dG[self._dG_K].reshape(self._GG_shape)
            fa = self._factor(a).conj()
            fb = self._factor(b)
            K_GG += fa[:, np.newaxis] * C_GG * fb[np.newaxis, :]
        return K_GG / self._volume

    def _add_to(self, x_GG):
        x_GG += self.get_Kxc_GG()


class _FieldPAWEngine(LocalPAWFTEngine):
    """LocalPAWFTEngine variant taking precomputed per-sphere field
    corrections Δc_a(n, g) (n = Lebedev direction, g = radial index)
    instead of an add_f(n(r)) callback, so that fields depending on
    density gradients can be corrected inside the spheres."""

    def calculate_field_corrections(self, qpd, R_av, micro_setups, dc_ang_a):
        nG = qpd.ngmax
        G_Gv = qpd.get_reciprocal_vectors(add_q=False)
        fPAW_G = np.zeros(nG, dtype=complex)
        G_myG = self._distribute_correction(nG)
        G_myGv = G_Gv[G_myG]
        for a, (R_v, micro_setup) in enumerate(zip(R_av, micro_setups)):
            self._add_field_correction(
                a, R_v, micro_setup, dc_ang_a[a], G_myG, G_myGv, fPAW_G
            )
        self.context.comm.sum(fPAW_G)
        return fPAW_G

    def _add_field_correction(self, a, R_v, micro_setup, dc_ng, G_myG, G_myGv, fPAW_G):
        rgd = micro_setup.rgd
        rshe, info_string = micro_setup.expand(
            dc_ng, lmax=self.rshelmax, wmin=self.rshewmin
        )
        self.print_rshe_info(a, info_string)
        (ii_MmyG, j_gMmyG, Y_MmyG) = self._expand_plane_waves(
            G_myGv, rgd.r_g, rshe.L_M, rshe.l_M
        )
        angular_coef_MmyG = ii_MmyG * Y_MmyG
        df_gM = rshe.f_gM
        radial_coef_MmyG = np.tensordot(
            j_gMmyG * df_gM[..., np.newaxis], rgd.dv_g, axes=([0, 0])
        )
        atomic_corr_myG = np.sum(angular_coef_MmyG * radial_coef_MmyG, axis=0)
        position_prefactor_myG = np.exp(-1j * np.inner(G_myGv, R_v))
        fPAW_G[G_myG] += position_prefactor_myG * atomic_corr_myG


def _sphere_coefficient_corrections(gs, family, ids, fn):
    """AE-minus-pseudo coefficient-field corrections Δc_ab(n, g) for
    every augmentation sphere, with the density gradients built like
    GPAW's radial GGA code: the radial part from d n_L/dr along the
    Lebedev direction, the angular part from r grad Y_L / r."""
    from gpaw.sphere.lebedev import R_nv
    from gpaw.xc.pawcorrection import rnablaY_nLv as rnablaY_full

    corrections = []
    for micro_setup in gs.micro_setups:
        rgd = micro_setup.rgd
        Lmax = micro_setup.n_sLg.shape[1]
        Y_nL = micro_setup.Y_nL[:, :Lmax]
        rnablaY_nLv = rnablaY_full[:, :Lmax]
        nn = Y_nL.shape[0]
        ng = rgd.N
        inv_r = np.zeros(ng)
        inv_r[1:] = 1.0 / rgd.r_g[1:]

        def side_fields(n_Lg):
            dndr_Lg = np.empty_like(n_Lg)
            for n_g, dndr_g in zip(n_Lg, dndr_Lg):
                rgd.derivative(n_g, dndr_g)
            rho_ng = Y_nL @ n_Lg
            a_ng = Y_nL @ dndr_Lg
            b_nvg = np.einsum("nLv,Lg->nvg", rnablaY_nLv, n_Lg)
            grad_nvg = a_ng[:, np.newaxis] * R_nv[:, :, np.newaxis] + b_nvg * inv_r
            grad_nvg[..., 0] = grad_nvg[..., 1]
            return rho_ng, grad_nvg

        def side_coefficients(n_Lg):
            rho_ng, grad_nvg = side_fields(n_Lg)
            ops = {"rho": np.maximum(rho_ng, 1e-10).ravel()}
            if family != "lda":
                for v, ax in enumerate("xyz"):
                    ops[f"grad_rho_{ax}"] = grad_nvg[:, v].ravel()
                ops["sigma"] = (grad_nvg**2).sum(axis=1).ravel()
            _add_libxc_arrays(ops, family, ids)
            return {
                ab: np.broadcast_to(c, (nn * ng,)).reshape(nn, ng)
                for ab, c in fn(ops).items()
            }

        c_ae = side_coefficients(micro_setup.n_sLg[0])
        c_ps = side_coefficients(micro_setup.nt_sLg[0])
        corrections.append({ab: c_ae[ab] - c_ps[ab] for ab in c_ae})
    return corrections


def _add_libxc_arrays(ops, family, ids):
    """Add Libxc first- and second-derivative arrays to an operand
    dict (in place), for 1-D flattened field arrays."""
    import pylibxc

    inp = {"rho": ops["rho"]}
    if family != "lda":
        inp["sigma"] = ops["sigma"]
    if family == "mgga_tau":
        inp["tau"] = ops["tau"]
    for fid in ids.split("+"):
        f = pylibxc.LibXCFunctional(fid, "unpolarized")
        r = f.compute(inp, do_exc=False, do_vxc=True, do_fxc=True)
        for k, v in r.items():
            v = v.reshape(v.shape[0], -1).T.squeeze()
            ops[k] = ops.get(k, 0) + v


class SemilocalFXCCalculator(AdiabaticFXCCalculator):
    """Calculates semilocal (LDA/GGA) adiabatic kernels from
    machine-generated coefficient fields."""

    #: functional name -> (family, libxc functional ids)
    LIBXC = {"ALDA": ("lda", "LDA_X+LDA_C_PW"), "APBE": ("gga", "GGA_X_PBE+GGA_C_PBE")}

    def __init__(self, localft_calc, paw=True, rshelmax=-1, rshewmin=None):
        super().__init__(localft_calc)
        self.paw = paw
        self.rshelmax = rshelmax
        self.rshewmin = rshewmin

    def __call__(self, fxc, spincomponent, qpd):
        assert spincomponent in ("00", "uu", "dd"), (
            "semilocal kernels: density-density only (for now)"
        )
        if fxc in self.LIBXC:
            family, ids = self.LIBXC[fxc]
        elif fxc.startswith("A"):
            name = fxc[1:]
            family = {"G": "gga", "L": "lda"}[name[0]]
            ids = name
        else:
            raise ValueError(f"unknown semilocal kernel {fxc!r}")

        gs = self.localft_calc.gs
        c_ab_R = _coefficient_fields(gs, family, ids)

        large_ecut = 4 * qpd.ecut
        large_qpd = qpd.copy_with(ecut=large_ecut, gammacentered=True, gd=gs.finegd)
        GG_shape, dG_K, Q_dG = self.create_unfoldable_Q_dG_mapping(qpd, large_qpd)

        dc_aab = None
        if self.paw:
            engine = _FieldPAWEngine(
                self.context, rshelmax=self.rshelmax, rshewmin=self.rshewmin
            )
            R_av = gs.atoms.positions / Bohr
            micro_setups = gs.micro_setups
            dc_aab = _sphere_coefficient_corrections(
                gs, family, ids, _generated_fn(family)
            )

        c_ab_dG = {}
        for ab, c_R in c_ab_R.items():
            c_Q = fft_from_grid(c_R, large_qpd)
            if dc_aab is not None:
                c_Q = c_Q + engine.calculate_field_corrections(
                    large_qpd, R_av, micro_setups, [dc[ab] for dc in dc_aab]
                )
            c_ab_dG[ab] = c_Q[Q_dG]

        Gv = qpd.get_reciprocal_vectors(add_q=True)
        return SemilocalFXCKernel(c_ab_dG, dG_K, GG_shape, Gv, qpd.gd.volume)


def _generated_fn(family):
    """The machine-generated coefficient-field function for a family."""
    from gpaw.xckernel_fxc import fxc_coefficient_fields_lda, fxc_coefficient_fields_gga

    return {"lda": fxc_coefficient_fields_lda, "gga": fxc_coefficient_fields_gga}[
        family
    ]


def _coefficient_fields(gs, family, ids):
    """Ground-state fields, Libxc derivative arrays, and the generated
    coefficient fields on the fine real-space grid."""
    nt_sg = gs.nt_sr
    if len(nt_sg) != 1:
        raise NotImplementedError("semilocal kernels: spin-paired only")
    finegd = gs.finegd
    nt = np.maximum(nt_sg[0], 1e-10)
    ops = {"rho": nt}
    if family != "lda":
        gradn = np.zeros((3,) + nt.shape)
        for v in range(3):
            Gradient(finegd, v, n=2).apply(nt_sg[0], gradn[v])
        for v, ax in enumerate("xyz"):
            ops[f"grad_rho_{ax}"] = gradn[v]
        ops["sigma"] = (gradn**2).sum(axis=0)

    flat = {k: np.asarray(v).ravel() for k, v in ops.items()}
    _add_libxc_arrays(flat, family, ids)
    c = _generated_fn(family)(flat)
    return {
        ab: np.ascontiguousarray(
            np.broadcast_to(v, (nt.size,)).reshape(nt.shape), dtype=float
        )
        for ab, v in c.items()
    }


def semilocal_fxc_factory(gs, context, paw=True, rshelmax=-1, rshewmin=None):
    """Convenience: a SemilocalFXCCalculator from a ground-state
    adapter and response context (mirrors
    AdiabaticFXCCalculator.from_rshe_parameters)."""
    from gpaw.response.localft import LocalGridFTCalculator

    localft_calc = LocalGridFTCalculator(gs, context)
    return SemilocalFXCCalculator(
        localft_calc, paw=paw, rshelmax=rshelmax, rshewmin=rshewmin
    )


create_semilocal_kernel = partial(SemilocalFXCCalculator)
