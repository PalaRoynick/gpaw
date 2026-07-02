from __future__ import annotations

import warnings
from collections.abc import Callable
from functools import partial
from math import ceil, floor
from ase.units import Ha

import numpy as np
from gpaw import debug
from gpaw.core.arrays import XArray
from gpaw.core.atom_centered_functions import AtomArrays
from gpaw.core.matrix import suggest_blocking
from gpaw.gpu import as_np
from gpaw.mpi import MPIComm, serial_comm
from gpaw.new import trace, tracectx, zips
from gpaw.new.c import calculate_residuals_gpu
from gpaw.new.eigensolver import Eigensolver, calculate_weights
from gpaw.new.energies import DFTEnergies
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.utilities import as_real_dtype
from gpaw.utilities.blas import axpy
from gpaw.new.pw.hybrids import PWHybridHamiltonian
from gpaw.core import PWDesc  # , PWArray


def slparams(nbands: int,
             comm: MPIComm,
             limit: int = 1000) -> tuple[MPIComm, int, int, int]:
    """Decide on scalapack parameters."""
    if nbands < limit:
        return serial_comm, 1, 1, 0
    # How much of comm should we use?
    # At least 30,000 numbers per core, approximately:
    ncores = nbands**2 / 30_000
    # We also want ncores to factor into a product of two integers:
    ncores = int(floor(ncores**0.5) * ceil(ncores**0.5))
    if ncores < comm.size:
        pass  # comm = comm.new_communicator(range(ncores))
    else:
        ncores = comm.size
    return (comm, *suggest_blocking(nbands, ncores))


class PWFDEigensolver(Eigensolver):
    min_work_array: int = 1
    min_buffer_array: int = 1

    def __init__(self,
                 *,
                 wf_grid,
                 hamiltonian,
                 convergence: dict,
                 nbands: int,
                 domain_band_comm,
                 blocksize: int = 10,
                 max_buffer_mem: int | None = 200 * 1024 ** 2,
                 scalapack_parameters: tuple[int, int, int] | None = None):
        self.wf_grid = wf_grid
        self.converge_bands = convergence.get('bands', 'occupied')
        self.residual_target = convergence.get('eigenstates', 4e-8) / Ha**2
        self.max_blocksize = blocksize
        self.preconditioner: Callable
        self.preconditioner_factory = hamiltonian.create_preconditioner
        self.work_arrays: np.ndarray
        self.data_buffers: np.ndarray

        self.domain_band_comm = domain_band_comm

        # Maximal memory to be used for the eigensolver
        # should be infinite if hamiltonian is not band-local (hybrids)
        self.max_buffer_mem = (
            max_buffer_mem if hamiltonian.band_local else None)

        if scalapack_parameters is None:
            self.scalapack_parameters = slparams(nbands, domain_band_comm)
        else:
            r, c, b = scalapack_parameters
            if b > nbands // max(r, c):
                warnings.warn(
                    f'{nbands}x{nbands} matrix on {r}x{c} grid with '
                    f'blocksize {b} is too small for ScaLapack.  '
                    'Using Lapack instead.')
                self.scalapack_parameters = (serial_comm, 1, 1, 0)
            else:
                slcomm = domain_band_comm
                assert r * c <= slcomm.size
                self.scalapack_parameters = (slcomm, r, c, b)

    def __str__(self):
        txt = f'{self.__class__.__name__}\n'
        txt += f'Converge bands: {self.converge_bands}\n'
        _, r, c, b = self.scalapack_parameters
        if r * c > 1:
            txt += f'Scalapack: {r}x{c} grid, blocksize={b}\n'
        return txt

    def _initialize(self, ibzwfs):
        xp = ibzwfs.xp
        B = ibzwfs.nbands
        wfs = ibzwfs._wfs_u[0]
        b = wfs.psit_nX.mydims[0]

        # Decide on blocksize (used for band slicing)
        if self.max_blocksize is None:
            if xp == np:
                self.max_blocksize = 32
            else:
                # GPU's prefer larger blocksizes
                self.max_blocksize = 512

        X_max = max(1, np.prod(ibzwfs.get_max_shape()))
        self.max_blocksize = np.clip(self.max_blocksize, 1,
                                     self.max_buffer_mem // X_max)

        if isinstance(self.wf_grid, PWDesc):
            S = self.wf_grid.comm.size
            # Use a multiple of S for maximum efficiency
            self.max_blocksize = max((self.max_blocksize // S) * S, 1)

        self.blocksize = max(min(self.max_blocksize, b),
                             1)

        # Build preconditioner
        self.preconditioner = self.preconditioner_factory(self.blocksize,
                                                          xp=ibzwfs.xp)

        # Allocate work-arrays
        self._allocate_work_arrays(ibzwfs, shape=(self.min_work_array,))
        self._allocate_buffer_arrays(ibzwfs, shape=(self.min_buffer_array,))

    def _allocate_buffer_arrays(self, ibzwfs, shape):
        G_max = max(1, np.prod(ibzwfs.get_max_shape()))
        b = max(wfs.n2 - wfs.n1 for wfs in ibzwfs)
        nbands = ibzwfs.nbands
        dtype_size = ibzwfs._wfs_u[0].psit_nX.data.dtype.itemsize
        domain_size = ibzwfs.domain_comm.size

        if self.max_buffer_mem is not None:
            # Buffer size needs to ensure that the number of bands
            # of the buffer is a multiple of domain_size.
            buffer_size_per_domain = max(self.max_buffer_mem,
                                         domain_size * G_max * dtype_size,
                                         nbands * dtype_size) \
                // (domain_size * G_max * dtype_size)
            buffer_size = min(buffer_size_per_domain * domain_size
                              * G_max * dtype_size,
                              b * G_max * dtype_size)
        else:
            buffer_size = max(b * G_max * dtype_size,
                              nbands * dtype_size)

        self.data_buffers = ibzwfs.xp.empty(shape + (buffer_size,),
                                            np.byte)

    def _allocate_work_arrays(self, ibzwfs, shape):
        b = max(wfs.n2 - wfs.n1 for wfs in ibzwfs)
        shape += (b,) + ibzwfs.get_max_shape()
        dtype = ibzwfs._wfs_u[0].psit_nX.data.dtype
        self.work_arrays = ibzwfs.xp.empty(shape, dtype)

    @trace
    def iterate(self,
                ibzwfs: IBZWaveFunctions,
                density,
                potential,
                hamiltonian: Hamiltonian,
                pot_calc,
                energies: DFTEnergies) -> tuple[float, float, DFTEnergies]:
        """Iterate on state given fixed hamiltonian.

        Returns
        -------
        float:
            Weighted error of residuals:::

                           ~
              R = (H - ε S)ψ
               n        n   n
        """

        if not hasattr(self, 'preconditioner'):
            self._initialize(ibzwfs)

        wfs = ibzwfs._wfs_u[0]
        dS_aii = wfs.setups.get_overlap_corrections(wfs.P_ani.layout.atomdist,
                                                    wfs.xp)

        ibzwfs.orthonormalize()
        # ibzwfs = kpad(ibzwfs)
        hamiltonian.update_wave_functions(ibzwfs)

        apply = partial(hamiltonian.apply,
                        potential.vt_sR,
                        potential.dedtaut_sR,
                        ibzwfs, density.D_asii)  # used by hybrids

        weight_un = calculate_weights(self.converge_bands, ibzwfs)

        if isinstance(hamiltonian, PWHybridHamiltonian):
            wfs_u = ibzwfs.zero_padded_iter()
        else:
            wfs_u = ibzwfs

        wfs_error = 0.0
        eig_error = 0.0
        # Loop over k-points:
        for u, wfs in enumerate(wfs_u):
            if u < len(weight_un):
                weight_n = weight_un[u]
            elif weight_n is not None:
                weight_n *= 0.0
            Ht = partial(apply, spin=wfs.spin)
            temp_wfs_error, temp_eig_error = \
                self.iterate_kpt(wfs, weight_n, self.iterate1,
                                 Ht=Ht, potential=potential,
                                 dS_aii=dS_aii)
            wfs_error += wfs.weight * temp_wfs_error
            if eig_error < temp_eig_error:
                eig_error = temp_eig_error

        wfs_error = ibzwfs.kpt_band_comm.sum_scalar(
            float(wfs_error)) * ibzwfs.spin_degeneracy
        eig_error = ibzwfs.kpt_band_comm.max_scalar(eig_error)

        return eig_error, wfs_error, energies

    def iterate1(self, wfs, Ht, potential, dS_aii, weight_n):
        with tracectx('Initialize'):
            dH = partial(potential.deltaH, spin=wfs.spin)
            xp = wfs.psit_nX.xp
            psit_nX = wfs.psit_nX
            band_comm = psit_nX.comm
            domain_comm = psit_nX.desc.comm

            b = psit_nX.mydims[0]
            residual_nX = psit_nX.new(data=self.work_arrays[0, :b])

            wfs.subspace_diagonalize(
                Ht, dH,
                psit2_nX=residual_nX,
                data_buffer=self.data_buffers[0],
                scalapack_parameters=self.scalapack_parameters)

            if weight_n is None:
                weight_n = np.zeros(b)
                weight_n[:(len(weight_n) * 4) // 5] = 1.0

            P_ani = wfs.P_ani
            P2_ani = P_ani.new()
            P3_ani = P_ani.new()
            Ptemp_ani = P_ani.new()
            P_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
            calculate_residuals(psit_nX, residual_nX, wfs.pt_aiX, wfs.P_ani,
                                wfs.myeig_n, dH, dS_aii, P2_ani, P3_ani)
            error_n = as_np(residual_nX.norm2())
            if len(error_n.shape) > 1:
                error_n = error_n.sum(axis=1)
            error = weight_n @ error_n
            b_error = band_comm.sum_scalar(error) / \
                max(band_comm.sum_scalar(weight_n.sum()), 0.5)

        for i in range(self.niter):
            linalg_error = False
            eps = 0
            active_indicies = np.where(error_n > eps)[0]
            loop_limit = (psit_nX.dims[0] + band_comm.size - 1) // band_comm.size \
                or len(active_indicies)

            if len(active_indicies) == 0:
                break

            self.prestep(wfs, residual_nX, P2_ani, P3_ani, Ptemp_ani)

            for j in range(0, loop_limit, self.max_blocksize):
                block_slice_base = \
                    slice(min(j, len(active_indicies)),  # XXX
                        min(j + self.max_blocksize, len(active_indicies)))
                block = \
                    block_slice_base.stop - block_slice_base.start
                block_slice = active_indicies[block_slice_base]
                linalg_error = linalg_error or \
                    self.bandstep(wfs, Ht, dH, block, block_slice,
                                  i, P2_ani, P3_ani, Ptemp_ani, dS_aii)
                # TODO: Sync band eigenvalues

            if b_error < self.residual_target or linalg_error:
                if linalg_error:
                    print(f"linalg_error, {i}")
                else:
                    print(f"b_error < residual_target {i}: b_error={b_error}, residual_target={self.residual_target}")
                break

            self.poststep(wfs, residual_nX, Ht, P2_ani, P3_ani,
                          Ptemp_ani, dH, i, dS_aii)

            error_n = as_np(residual_nX.norm2())
            if len(error_n.shape) > 1:
                error_n = error_n.sum(axis=1)
            error = weight_n @ error_n
            b_error = band_comm.sum_scalar(b_error) / \
                max(band_comm.sum_scalar(weight_n.sum()), 0.5)

        if not wfs.orthonormalized:
            # wfs._P_ani = None
            wfs.orthonormalize(residual_nX)

        if debug:
            psit_nX.sanity_check()

        return error

    def prestep(self, wfs, residual_nX, P2_ani, P3_ani, Ptemp_ani) -> None:
        pass

    def bandstep(self, wfs, Ht, dH, block, block_slice,
                 i, P2_ani, P3_ani, Ptemp_ani, dS_aii) -> None:
        pass

    def poststep(self, wfs, residual_nX, Ht, P2_ani, P3_ani, Ptemp_ani, dH, i, dS_aii) -> None:
        pass


@trace
def calculate_residuals(psit_nX,
                        residual_nX: XArray,
                        pt_aiX,
                        P_ani,
                        eig_n,
                        dH: Callable[[AtomArrays, AtomArrays], AtomArrays],
                        dS_aii: AtomArrays,
                        P1_ani: AtomArrays,
                        P2_ani: AtomArrays) -> None:
    """Complete the calculation of resuduals.

    Starting from residual_nX having the values:::

       ^   ~  ~
      (T + v) ψ
               n

    add the following:::

      --   a       a   ~a ~   ~a    ~
      > (ΔH  - ε ΔS  )<p |ψ > p - ε ψ .
      --   ij   n  ij   j  n   i     n
      ij

    (P1_ani and P2_ani are work buffers).
    """
    xp = residual_nX.xp
    if xp is np:
        for r, e, p in zips(residual_nX.data, eig_n, psit_nX.data):
            axpy(-e, p, r)
    else:
        eig_n = xp.asarray(eig_n, dtype=as_real_dtype(residual_nX.data.dtype))
        calculate_residuals_gpu(residual_nX.data, eig_n, psit_nX.data)

    dH(P_ani, P1_ani)
    P_ani.block_diag_multiply(dS_aii, out_ani=P2_ani)

    P2_ani.matrix.data *= eig_n[:, np.newaxis]

    P1_ani.data -= P2_ani.data
    pt_aiX.add_to(residual_nX, P1_ani)
