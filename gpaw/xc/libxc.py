import warnings

import gpaw.cgpaw as cgpaw
from gpaw import debug
from gpaw.xc.kernel import XCKernel
import numpy as np
from gpaw.gpu import cupy as cp


class FunctionalNeedsLaplacianError(Exception):
    """(MGGA) Functional needs laplacian.

    """


short_names = {
    'LDA': 'LDA_X+LDA_C_PW',
    'PW91': 'GGA_X_PW91+GGA_C_PW91',
    'PBE': 'GGA_X_PBE+GGA_C_PBE',
    'PBEsol': 'GGA_X_PBE_SOL+GGA_C_PBE_SOL',
    'revPBE': 'GGA_X_PBE_R+GGA_C_PBE',
    'RPBE': 'GGA_X_RPBE+GGA_C_PBE',
    'BLYP': 'GGA_X_B88+GGA_C_LYP',
    'HCTH407': 'GGA_XC_HCTH_407',
    'WC': 'GGA_X_WC+GGA_C_PBE',
    'AM05': 'GGA_X_AM05+GGA_C_AM05',
    'M06-L': 'MGGA_X_M06_L+MGGA_C_M06_L',
    'TPSS': 'MGGA_X_TPSS+MGGA_C_TPSS',
    'revTPSS': 'MGGA_X_REVTPSS+MGGA_C_REVTPSS',
    'mBEEF': 'MGGA_X_MBEEF+GGA_C_PBE_SOL',
    'SCAN': 'MGGA_X_SCAN+MGGA_C_SCAN'}


def _to_libxc_layout(array_sg):
    """Libxc wants spin as the last index. This converts a GPAW '_sg'/'_xg'
    array to Libxc layout as a C-contiguous copy."""
    nchannels = array_sg.shape[0]
    return cp.ascontiguousarray(array_sg.reshape(nchannels, -1).T)


def _from_libxc_layout(array_gs, shape):
    """Converts 'array_gs' back to GPAW layout 'array_sg'."""
    nchannels = array_gs.shape[-1]
    return array_gs.T.reshape((nchannels,) + shape)


class LibXC(XCKernel):
    """Functionals from libxc."""
    def __init__(self, name, provides_laplacian=False, disable_fhc=True):
        if not hasattr(cgpaw, 'lxcXCFuncNum'):
            raise NameError(
                f'Unable to use {name}: GPAW not compiled with LibXC!')
        self.name = name
        self.omega = None
        self.disable_fhc = disable_fhc

        # True if the libxc installation supports GPU
        self.gpu_available = (
            hasattr(cgpaw, 'libxc_has_gpu_support') and
            cgpaw.libxc_has_gpu_support())
        self._device_xc = None
        """Holds the GPU version of libxc object. Created only on first use"""

        self.initialize(nspins=1, provides_laplacian=provides_laplacian)

    def initialize(self,
                   nspins,
                   provides_laplacian=False):
        self.nspins = nspins
        name = short_names.get(self.name, self.name)
        number = cgpaw.lxcXCFuncNum(name)
        if number is not None:
            f = number
            xc = -1
            x = -1
            c = -1
            if '_XC_' in name:
                xc = f
            elif '_C_' in name:
                c = f
            else:
                x = f
        else:
            try:
                x, c = name.split('+')
            except ValueError:
                raise NameError(f'Unknown functional: {name}.')
            xc = -1
            x = cgpaw.lxcXCFuncNum(x)
            c = cgpaw.lxcXCFuncNum(c)
            if x is None or c is None:
                raise NameError(f'Unknown functional: {name}.')

        # Always init the CPU functional, and GPU version later on first use.
        self._creation_args = xc, x, c, nspins
        """Holds creation arguments for the C++ side object"""
        self.xc = cgpaw.lxcXCFunctional(*self._creation_args, False)
        """XC functional allocated in CPU memory"""

        self.set_omega()

        if self.xc.vv10():
            raise NotImplementedError('(r)VV10 correlation is not implemented')

        if self.xc.is_mgga():
            self.type = 'MGGA'
            if self.disable_fhc:
                ok = self.xc.disable_fhc()
                if not ok:
                    warnings.warn(
                        'libxc should be compiled with --disable-fhc' +
                        ' otherwise SCF calculations might not converge.')
            if self.xc.needs_laplacian() and not provides_laplacian:
                msg = f'Functional "{name}" needs laplacian'
                msg += ' (unsupported)'
                raise FunctionalNeedsLaplacianError(msg, self.xc)
        elif self.xc.is_gga():
            self.type = 'GGA'
        else:
            self.type = 'LDA'

    def calculate(self, e_g, n_sg, dedn_sg,
                  sigma_xg=None, dedsigma_xg=None,
                  tau_sg=None, dedtau_sg=None):
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                                 tau_sg, dedtau_sg)
        nspins = len(n_sg)
        if self.nspins != nspins:
            self.initialize(nspins)
        self.xc.calculate(e_g.ravel(), n_sg, dedn_sg,
                          sigma_xg, dedsigma_xg,
                          tau_sg, dedtau_sg)

    def calculate_gpu(self, e_g, n_sg, dedn_sg,
                      sigma_xg=None, dedsigma_xg=None,
                      tau_sg=None, dedtau_sg=None):
        """Evaluate a Libxc functional on GPU arrays."""
        if debug:
            self.check_arguments(e_g, n_sg, dedn_sg, sigma_xg, dedsigma_xg,
                                 tau_sg, dedtau_sg)
        nspins = len(n_sg)
        if self.nspins != nspins:
            self.initialize(nspins)

        shape = n_sg.shape[1:]
        npoints = int(np.prod(shape))

        device_xc = self.get_device_functional()

        n_gs = _to_libxc_layout(n_sg)
        sigma_gx = None if sigma_xg is None else _to_libxc_layout(sigma_xg)
        tau_gs = None if tau_sg is None else _to_libxc_layout(tau_sg)

        # Accumulate results over each functional type. Scratch space for
        # holding individual results:
        e_g_scratch = cp.empty_like(e_g)
        dedn_gs_scratch = cp.empty_like(n_gs)
        # Derivatives are technically optional for libxc but here we always
        # pass de/dn. Hence, must always pass other first derivatives too.
        dedsigma_gx_scratch = (
            None if sigma_gx is None else cp.empty_like(sigma_gx))
        dedtau_gs_scratch = (
            None if tau_gs is None else cp.empty_like(tau_gs))

        density_g_total = n_sg.sum(axis=0)

        for index in range(device_xc.number_of_functionals()):
            cgpaw.evaluate_lxcfunctional_device(
                device_xc,
                index,
                npoints,
                n_gs,
                e_g_scratch,
                dedn_gs_scratch,
                sigma_gx,
                dedsigma_gx_scratch,
                tau_gs,
                dedtau_gs_scratch)

            # Accumulate
            if index == 0:
                # Our e_g is energy density, libxc returns energy per particle
                e_g[:] = density_g_total * e_g_scratch
                if dedsigma_xg is not None:
                    dedsigma_xg[:] = _from_libxc_layout(
                        dedsigma_gx_scratch, shape)
                if dedtau_sg is not None:
                    dedtau_sg[:] = _from_libxc_layout(dedtau_gs_scratch, shape)
            else:
                e_g += density_g_total * e_g_scratch
                if dedsigma_xg is not None:
                    dedsigma_xg += _from_libxc_layout(
                        dedsigma_gx_scratch, shape)
                if dedtau_sg is not None:
                    dedtau_sg += _from_libxc_layout(dedtau_gs_scratch, shape)

            dedn_sg += _from_libxc_layout(dedn_gs_scratch, shape)

    def set_omega(self, omega=None):
        """Set the value of gamma/omega in RSF."""
        if omega is not None:
            self.omega = omega
        if self.omega is not None:
            if not self.xc.set_omega(self.omega):
                raise ValueError('Tried setting omega on a non RSF hybrid.')

    def get_device_functional(self):
        """Returns a GPU-allocated version of the functional and initializes it
        on first use."""
        if not self.gpu_available:
            raise RuntimeError("Requires Libxc built with GPU support")

        if self._device_xc is None:
            self._device_xc = cgpaw.lxcXCFunctional(*self._creation_args, True)

            if self.omega is not None:
                self._device_xc.set_omega(self.omega)
            if self.type == 'MGGA' and self.disable_fhc:
                self._device_xc.disable_fhc()

        return self._device_xc
