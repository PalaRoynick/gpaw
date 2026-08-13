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

from gpaw.fd_operators import Gradient
from gpaw.response.dyson import PWKernel
from gpaw.response.fxc_kernels import AdiabaticFXCCalculator
from gpaw.response.localft import fft_from_grid


class SemilocalFXCKernel(PWKernel):
    """K_xc(G, G') assembled from coefficient-field Fourier transforms
    and reciprocal-vector factors."""

    def __init__(self, c_ab_dG, dG_K, GG_shape, Gv, volume):
        self._c_ab_dG = c_ab_dG      # {(slot, slot): c(dG) complex array}
        self._dG_K = dG_K
        self._GG_shape = GG_shape
        self._Gv = Gv                # (nG, 3) q+G Cartesian
        self._volume = volume

    def get_number_of_plane_waves(self):
        return self._GG_shape[0]

    def _factor(self, slot):
        if slot == 'rho':
            return np.ones(self._GG_shape[0], dtype=complex)
        v = {'x': 0, 'y': 1, 'z': 2}[slot]
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


class SemilocalFXCCalculator(AdiabaticFXCCalculator):
    """Calculates semilocal (LDA/GGA) adiabatic kernels from
    machine-generated coefficient fields."""

    #: functional name -> (family, libxc functional ids)
    LIBXC = {'ALDA': ('lda', 'LDA_X+LDA_C_PW'),
             'APBE': ('gga', 'GGA_X_PBE+GGA_C_PBE')}

    def __call__(self, fxc, spincomponent, qpd):
        assert spincomponent in ('00', 'uu', 'dd'), \
            'semilocal kernels: density-density only (for now)'
        if fxc in self.LIBXC:
            family, ids = self.LIBXC[fxc]
        elif fxc.startswith('A'):
            name = fxc[1:]
            family = {'G': 'gga', 'L': 'lda'}[name[0]]
            ids = name
        else:
            raise ValueError(f'unknown semilocal kernel {fxc!r}')

        gs = self.localft_calc.gs
        c_ab_R = _coefficient_fields(gs, family, ids)

        large_ecut = 4 * qpd.ecut
        large_qpd = qpd.copy_with(ecut=large_ecut,
                                  gammacentered=True,
                                  gd=gs.finegd)
        GG_shape, dG_K, Q_dG = self.create_unfoldable_Q_dG_mapping(
            qpd, large_qpd)

        c_ab_dG = {}
        for ab, c_R in c_ab_R.items():
            c_Q = fft_from_grid(c_R, large_qpd)
            c_ab_dG[ab] = c_Q[Q_dG]

        Gv = qpd.get_reciprocal_vectors(add_q=True)
        return SemilocalFXCKernel(c_ab_dG, dG_K, GG_shape, Gv,
                                  qpd.gd.volume)


def _coefficient_fields(gs, family, ids):
    """Ground-state fields, Libxc derivative arrays, and the generated
    coefficient fields on the fine real-space grid."""
    import pylibxc

    from gpaw.xckernel_fxc import (fxc_coefficient_fields_lda,
                                   fxc_coefficient_fields_gga)

    nt_sg = gs.nt_sr
    if len(nt_sg) != 1:
        raise NotImplementedError('semilocal kernels: spin-paired only')
    finegd = gs.finegd
    nt = np.maximum(nt_sg[0], 1e-10)
    ops = {'rho': nt}
    if family != 'lda':
        gradn = np.zeros((3,) + nt.shape)
        for v in range(3):
            Gradient(finegd, v, n=2).apply(nt_sg[0], gradn[v])
        for v, ax in enumerate('xyz'):
            ops[f'grad_rho_{ax}'] = gradn[v]
        ops['sigma'] = (gradn ** 2).sum(axis=0)

    inp = {'rho': ops['rho'].ravel()}
    if family != 'lda':
        inp['sigma'] = ops['sigma'].ravel()
    for fid in ids.split('+'):
        f = pylibxc.LibXCFunctional(fid, 'unpolarized')
        r = f.compute(inp, do_exc=False, do_vxc=True, do_fxc=True)
        for k, v in r.items():
            v = v.reshape(v.shape[0], -1).T.squeeze().reshape(nt.shape)
            ops[k] = ops.get(k, 0) + v

    fn = {'lda': fxc_coefficient_fields_lda,
          'gga': fxc_coefficient_fields_gga}[family]
    c = fn(ops)
    return {ab: np.ascontiguousarray(np.broadcast_to(v, nt.shape),
                                     dtype=float)
            for ab, v in c.items()}


def semilocal_fxc_factory(gs, context):
    """Convenience: a SemilocalFXCCalculator from a ground-state
    adapter and response context (mirrors
    AdiabaticFXCCalculator.from_rshe_parameters)."""
    from gpaw.response.localft import LocalGridFTCalculator
    localft_calc = LocalGridFTCalculator(gs, context)
    return SemilocalFXCCalculator(localft_calc)


create_semilocal_kernel = partial(SemilocalFXCCalculator)
