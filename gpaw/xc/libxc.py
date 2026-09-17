import warnings

import numpy as np

import gpaw.cgpaw as cgpaw
from gpaw import debug
from gpaw.xc.kernel import XCKernel


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
    'R2SCAN': 'MGGA_X_R2SCAN+MGGA_C_R2SCAN',
    'TPSS': 'MGGA_X_TPSS+MGGA_C_TPSS',
    'revTPSS': 'MGGA_X_REVTPSS+MGGA_C_REVTPSS',
    'mBEEF': 'MGGA_X_MBEEF+GGA_C_PBE_SOL',
    'SCAN': 'MGGA_X_SCAN+MGGA_C_SCAN'}


class LibXC(XCKernel):
    """Functionals from libxc."""

    def __init__(self, name, provides_laplacian=False, disable_fhc=True):
        if not hasattr(cgpaw, 'lxcXCFuncNum'):
            raise NameError(
                f'Unable to use {name}: GPAW not compiled with LibXC!')
        self.name = name
        self.omega = None
        self.disable_fhc = disable_fhc
        self.device_xc = None
        self.gpu_available = (
            hasattr(cgpaw, 'lxcXCFunctionalDeviceAvailable') and
            cgpaw.lxcXCFunctionalDeviceAvailable())
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

        self.arguments = xc, x, c, nspins
        self.xc = cgpaw.lxcXCFunctional(*self.arguments, False)
        self.device_xc = None
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
        """Evaluate Libxc directly on contiguous GPU work arrays."""
        from gpaw.gpu import cupy as cp

        nspins = len(n_sg)
        if self.nspins != nspins:
            self.initialize(nspins)

        shape = n_sg.shape[1:]
        npoints = int(np.prod(shape))
        density_gs = cp.ascontiguousarray(
            cp.maximum(n_sg.reshape(nspins, npoints).T, 1.0e-10),
            dtype=float)
        sigma_gx = self._point_major(sigma_xg, npoints)
        tau_gs = self._point_major(tau_sg, npoints)
        lapl_gs = (cp.zeros_like(density_gs)
                   if self.type == 'MGGA' else None)

        energy_g = cp.zeros(npoints, dtype=float)
        density_derivative_gs = cp.zeros_like(density_gs)
        sigma_derivative_gx = (cp.zeros_like(sigma_gx)
                               if sigma_gx is not None else None)
        tau_derivative_gs = (cp.zeros_like(tau_gs)
                             if tau_gs is not None else None)
        lapl_derivative_gs = (cp.zeros_like(lapl_gs)
                              if lapl_gs is not None else None)

        device_xc = self._device_backend()
        for index in range(device_xc.number_of_functionals()):
            local_energy_g = cp.empty_like(energy_g)
            local_density_derivative_gs = cp.empty_like(
                density_derivative_gs)
            local_sigma_derivative_gx = self._zeros_like(
                sigma_derivative_gx)
            local_tau_derivative_gs = self._zeros_like(
                tau_derivative_gs)
            local_lapl_derivative_gs = self._zeros_like(
                lapl_derivative_gs)
            device_xc.calculate_device(
                index,
                npoints,
                self._address(density_gs),
                self._address(sigma_gx),
                self._address(lapl_gs),
                self._address(tau_gs),
                self._address(local_energy_g),
                self._address(local_density_derivative_gs),
                self._address(local_sigma_derivative_gx),
                self._address(local_lapl_derivative_gs),
                self._address(local_tau_derivative_gs))
            energy_g += local_energy_g
            density_derivative_gs += local_density_derivative_gs
            if sigma_derivative_gx is not None:
                sigma_derivative_gx += local_sigma_derivative_gx
            if tau_derivative_gs is not None:
                tau_derivative_gs += local_tau_derivative_gs

        e_g.reshape(-1)[:] = density_gs.sum(axis=1) * energy_g
        dedn_sg.reshape(nspins, npoints)[:] += density_derivative_gs.T
        if dedsigma_xg is not None:
            dedsigma_xg.reshape(-1, npoints)[:] = sigma_derivative_gx.T
        if dedtau_sg is not None:
            dedtau_sg.reshape(nspins, npoints)[:] = tau_derivative_gs.T

    @staticmethod
    def _point_major(array_xg, npoints):
        if array_xg is None:
            return None
        from gpaw.gpu import cupy as cp
        return cp.ascontiguousarray(
            array_xg.reshape(-1, npoints).T, dtype=float)

    @staticmethod
    def _zeros_like(array):
        if array is None:
            return None
        from gpaw.gpu import cupy as cp
        return cp.zeros_like(array)

    @staticmethod
    def _address(array) -> int:
        return 0 if array is None else array.data.ptr

    def _device_backend(self):
        if not self.gpu_available:
            raise RuntimeError('Libxc was built without CUDA or HIP support')
        if self.device_xc is None:
            self.device_xc = cgpaw.lxcXCFunctional(*self.arguments, True)
            if self.omega is not None:
                self.device_xc.set_omega(self.omega)
            if self.type == 'MGGA' and self.disable_fhc:
                self.device_xc.disable_fhc()
        return self.device_xc

    def set_omega(self, omega=None):
        """Set the value of gamma/omega in RSF."""
        if omega is not None:
            self.omega = omega
        if self.omega is not None:
            if not self.xc.set_omega(self.omega):
                raise ValueError('Tried setting omega on a non RSF hybrid.')
            if self.device_xc is not None:
                self.device_xc.set_omega(self.omega)
