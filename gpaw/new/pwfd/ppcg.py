from __future__ import annotations

from functools import partial

import numpy as np
from ase.units import Ha

from gpaw import debug

# from gpaw.typing import Array2D
from gpaw.core import PWDesc  # , PWArray
from gpaw.core.matrix import Matrix
from gpaw.gpu import as_np
from gpaw.new import trace, tracectx, prod
from gpaw.new.pwfd.davidson import sliced_preconditioner

# from gpaw.mpi import broadcast_exception
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions

# from gpaw.utilities import as_real_dtype


class PPCG(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 hamiltonian,
                 convergence,
                 domain_band_comm,
                 niter=2,
                 min_niter=1,
                 blocksize=None,
                 rr_modulo=5,
                 include_cg=True,
                 promote_inner_dtype=False,
                 tolerances: tuple[float, float, float] | None = None,
                 scalapack_parameters=None,
                 max_buffer_mem: int = 200 * 1024 ** 2):
        """
        Initialize Projected Preconditioned Conjugate Gradient eigensolver,
        a.k.a. PPCG or Not-Davidson solver.

        See https://doi.org/10.1016/j.jcp.2015.02.030 for details.

        Parameters
        ----------
        nbands : int
            Number of bands.
        wf_grid : WaveFunctionsGrid
            Grid of wave functions.
        band_comm : MPI communicator
            Communicator for band parallelization.
        hamiltonian : Hamiltonian
            Hamiltonian.
        converge_bands : str, optional
            Which bands to converge ('occupied' or 'unoccupied'). Default is
            'occupied'.
        niter : int | typle[int, int], optional
            Maximum number of iterations. Default is 2.
        min_niter : int | None, optional
            If specified as an int, the value is the minimum number of
            iterations. Where the eigensolver will stop if the residual
            is below a tolerance, and performed at least the minimum number
            of iterations, otherwise the niter mnumber of iterations is used.
            If specified as None, the minimum number of iterations is set
            to niter.
            Default is None.
        blocksize : int, optional
            Block size for the diagonal slicing. Lower values
            are more efficient on CPUs with many cores but not on GPUs. The
            value will be modified to a multiple of the number of domain
            ranks.
            Default is 128 on cpu and 1024 on gpu.
        rr_modulo : int, optional
            How often to perform subspace diagonalization. Default is 5.
        include_cg : bool, optional
            Include CG in the solver. Default is True. Can be helpful to turn
            off for single precision calculations or if memory is an issue.
        promote_inner_dtype : bool, optional
            Promote inner dtype to double precision. Default is False.
            Only relevant for single precision calculations.
        tolerances : tuple[float, float, float], optional
            Advanced setting, tolerances for the solver. Use at your own risk.
            Fist two tolerances controls freezing of converged bands, the last
            tolerance controls the early breakout criterion.
        scalapack_parameters : dict, optional
            Parameters for scalapack solver.
        max_buffer_mem : int, optional
            Maximum memory in bytes for buffer. Default is 200 * 1024 ** 2.
        """

        super().__init__(
            wf_grid=wf_grid,
            hamiltonian=hamiltonian,
            convergence=convergence,
            scalapack_parameters=scalapack_parameters,
            nbands=nbands,
            max_buffer_mem=max_buffer_mem,
            domain_band_comm=domain_band_comm)

        self.min_work_array = 2 if include_cg else 1
        self.min_buffer_array = 5
        self.band_comm = band_comm
        self.nbands = nbands
        self.niter = niter
        self.min_niter = min_niter if min_niter is not None else niter
        self.max_blocksize = blocksize
        self.rr_modulo = rr_modulo
        self.tolerances = tolerances
        self.MW_nn: Matrix
        self.MP_nn: Matrix
        self.include_cg = include_cg
        self.promote_inner_dtype = promote_inner_dtype

        # We disable dynamic breakout for hybrids, to avoid deadlocks
        self.allow_dynamic_breakout = hamiltonian.band_local

    def __str__(self):
        return super().__str__() + f'niter={self.niter}\n'

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        xp = ibzwfs.xp
        B = ibzwfs.nbands
        wfs = ibzwfs._wfs_u[0]
        b = wfs.psit_nX.mydims[0]

        band_comm = ibzwfs.band_comm
        assert isinstance(wfs, PWFDWaveFunctions)
        self.nblocksizes = 3 * self.blocksize \
            if self.include_cg else 2 * self.blocksize
        dtype = wfs.psit_nX.desc.dtype

        if self.tolerances is not None:
            # XXX: Warn
            print('oof')

        self.tol_factor = 0
        self.tolerance = 0

        self.M_nn = Matrix(B, B, dtype=dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

        if dtype != np.float32 and dtype != np.complex64:
            self.promote_inner_dtype = False

        if self.promote_inner_dtype:
            inner_dtype = np.float64 if np.issubdtype(dtype, np.floating) \
                else np.complex128
            self.buffer_bb = xp.zeros((self.nblocksizes, self.nblocksizes),
                                      dtype=dtype)
        else:
            inner_dtype = dtype

        self.H_bb = xp.zeros((self.nblocksizes, self.nblocksizes),
                             dtype=inner_dtype)
        self.S_bb = xp.zeros((self.nblocksizes, self.nblocksizes),
                             dtype=inner_dtype)

    @trace
    def prestep(self, wfs, residual_nX,
                P2_ani, P3_ani, Ptemp_ani,
                b_error, i):
        M_nn = self.M_nn
        psit_nX = wfs.psit_nX
        P_ani = wfs.P_ani
        buffer_array_nX = psit_nX.create_work_buffer(self.data_buffers[0])
        sliced_preconditioner(psit_nX, residual_nX,
                              buffer=buffer_array_nX,
                              precon=self.preconditioner)
        wfs.pt_aiX.integrate(residual_nX, out=P2_ani)
        residual_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                    domain_sum=False,
                                    symmetric=False)
        P2_ani.matrix.multiply(Ptemp_ani, opb='C', symmetric=False,
                               beta=1, out=M_nn)
        psit_nX.desc.comm.sum(M_nn.data)

        M_nn.multiply(psit_nX, out=residual_nX, beta=1.0, alpha=-1.0)
        M_nn.multiply(P_ani, out=P2_ani, beta=1.0, alpha=-1.0)

        wfs.orthonormalized = False

    @trace
    def poststep(self, wfs, residual_nX, Ht,
                 P2_ani, P3_ani, Ptemp_ani, dH, i, dS_aii):
        M_nn = self.M_nn
        psit_nX = wfs.psit_nX
        P_ani = wfs.P_ani
        b = psit_nX.mydims[0]
        domain_comm = psit_nX.desc.comm

        # Subspace diagonialization needed every once in a while
        if (i + 1) % self.rr_modulo == 0:
            wfs.subspace_diagonalize(
                Ht, dH,
                psit2_nX=residual_nX,
                data_buffer=self.data_buffers[0],
                calculate_energy=False,
                scalapack_parameters=self.scalapack_parameters)
        else:
            wfs.orthonormalize(residual_nX)
            Ht(psit_nX, out=residual_nX)

        calculate_residuals(wfs.psit_nX,
                            residual_nX,
                            wfs.pt_aiX,
                            wfs.P_ani,
                            wfs.myeig_n,
                            dH, dS_aii, P2_ani, Ptemp_ani)

        P_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
        if self.include_cg:
            P_nX = psit_nX.new(data=self.work_arrays[1, :b])
            P_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                 domain_sum=False,
                                 symmetric=False)
            P3_ani.matrix.multiply(Ptemp_ani, opb='C',
                                   symmetric=False,
                                   beta=1, out=M_nn)
            domain_comm.sum(M_nn.data)
            M_nn.multiply(psit_nX, out=P_nX, beta=1.0, alpha=-1.0)
            M_nn.multiply(P_ani, out=P3_ani, beta=1.0, alpha=-1.0)

    @trace
    def bandstep(self, wfs, Ht, dH, block, block_slice, i,
                 P2_ani, P3_ani, Ptemp_ani, dS_aii):
        psit_nX = wfs.psit_nX
        xp = psit_nX.xp
        domain_comm = psit_nX.desc.comm
        desc = psit_nX.desc
        P_ani = wfs.P_ani
        b = psit_nX.mydims[0]
        residual_nX = psit_nX.new(self.work_arrays[0, :b])

        buff1 = self.data_buffers[0:3].ravel().view(psit_nX.data.dtype)[:]
        buff2 = self.data_buffers[3:5].ravel().view(psit_nX.data.dtype)[:]
        data_size = self.blocksize * prod(desc.myshape)
        buff_bX = desc.from_data(
            data=buff1[:3 * data_size].reshape(3 * self.blocksize, -1)
        )
        Hbuff_bX = desc.from_data(
            data=buff2[:2 * data_size].reshape(2 * self.blocksize, -1)
        )

        Pbuf_abi = P_ani.layout.empty(
            (self.nblocksizes, ) + psit_nX.dims[1:])
        HPbuf_abi = P_ani.layout.empty(
            (self.nblocksizes - self.blocksize, ) + psit_nX.dims[1:])

        buff_bX.matrix.data[:block] = \
            psit_nX.matrix.data[block_slice]
        Pbuf_abi.matrix.data[:block] = \
            P_ani.matrix.data[block_slice]
        buff_bX.matrix.data[block:2 * block] = \
            residual_nX.matrix.data[block_slice]
        Pbuf_abi.matrix.data[block:2 * block] = \
            P2_ani.matrix.data[block_slice]

        if self.include_cg:
            P_nX = psit_nX.new(data=self.work_arrays[1, :b])

        if i > 0 and self.include_cg:
            nblocks = 3 * block
            buff_bX.matrix.data[2 * block:3 * block] = \
                P_nX.matrix.data[block_slice]
            Pbuf_abi.matrix.data[2 * block:3 * block] = \
                P3_ani.matrix.data[block_slice]
        else:
            nblocks = 2 * block

        H_bb = self.H_bb.ravel()[:nblocks**2].reshape(
            (nblocks, nblocks))
        S_bb = self.S_bb.ravel()[:nblocks**2].reshape(
            (nblocks, nblocks))

        MH_bb = Matrix(M=nblocks, N=nblocks,
                       data=H_bb,
                       xp=xp)
        MS_bb = Matrix(M=nblocks, N=nblocks,
                       data=S_bb,
                       xp=xp)

        if self.promote_inner_dtype:
            buffer_bb = \
                self.buffer_bb.ravel()[:nblocks**2].reshape(
                    (nblocks, nblocks))
        else:
            buffer_bb = S_bb

        MBuf_bb = Matrix(M=nblocks - block, N=nblocks,
                         data=buffer_bb[block:nblocks],
                         xp=xp)

        Pbuf_abi[:, block:nblocks].block_diag_multiply(
            dS_aii, out_ani=HPbuf_abi[:, :nblocks - block])
        buff_bX[block:nblocks].matrix_elements(
            buff_bX[:nblocks], cc=True, out=MBuf_bb,
            domain_sum=False, symmetric=False)
        if self.promote_inner_dtype:
            S_bb[:] = buffer_bb
            buffer_bb[:] = 0
        HPbuf_abi[:, :nblocks - block].matrix.multiply(
            Pbuf_abi[:, :nblocks], out=MBuf_bb,
            symmetric=False, beta=1, opb='C')
        if self.promote_inner_dtype:
            S_bb += buffer_bb
        domain_comm.sum(S_bb)

        # Scale the diagonal elements, to improve numerical
        # stability. Here, we use the expontent -0.25, which
        # makes the diagonal elements closer to 1, by the a
        # factor of sqrt(X), with X being the previous diagonal.
        # This value performed best of the ones attempted.
        diag_scale_b = xp.diag(S_bb)[block:]**(-0.25)
        S_bb[block:, :] *= diag_scale_b[:, None]
        S_bb[:, block:] *= diag_scale_b[None, :]
        buff_bX.matrix.data[block:nblocks, :] \
            *= diag_scale_b[:, None]
        Pbuf_abi.matrix.data[block:nblocks, :] \
            *= diag_scale_b[:, None]

        if not self.promote_inner_dtype:
            MBuf_bb.data = H_bb[block:nblocks]

        Ht(buff_bX[block:nblocks], out=Hbuff_bX[:nblocks - block])
        dH(Pbuf_abi[:, block:nblocks],
           out_ani=HPbuf_abi[:, :nblocks - block])
        Hbuff_bX[:nblocks - block].matrix_elements(
            buff_bX[:nblocks],
            cc=True, out=MBuf_bb,
            domain_sum=False, symmetric=False)
        if self.promote_inner_dtype:
            H_bb[:] = buffer_bb
            buffer_bb[:] = 0
        HPbuf_abi[:, :nblocks - block].matrix.multiply(
            Pbuf_abi[:, :nblocks], out=MBuf_bb,
            symmetric=False, beta=1, opb='C')
        if self.promote_inner_dtype:
            H_bb[:] += buffer_bb[:]
        domain_comm.sum(H_bb)

        H_bb[:block, :block] = xp.diag(wfs.myeig_n[block_slice])
        S_bb[:block, :block] = xp.eye(block)
        MS_bb.tril2full()

        if nblocks > 2 * block:
            # Eigvalsh approach
            with tracectx('eigvalsh', gpu=xp is not np):
                try:
                    pos_defness = xp.linalg.eigvalsh(S_bb)[0]
                    if xp is not np:
                        pos_defness = pos_defness.get()
                except np.linalg.LinAlgError:
                    pos_defness = -42
            if pos_defness < \
                    np.finfo(psit_nX.data.dtype).eps * \
                    nblocks**0.5 or np.isnan(pos_defness):
                # Insufficient numerical precision for CG,
                # thus we only do the steepest descent step
                nblocks = 2 * block
                MH_bb = Matrix(M=nblocks, N=nblocks,
                               data=H_bb[:nblocks,
                                         :nblocks],
                               xp=xp)
                MS_bb = Matrix(M=nblocks, N=nblocks,
                               data=S_bb[:nblocks,
                                         :nblocks],
                               xp=xp)
                eig_b = MH_bb.eigh(MS_bb)
            else:
                # Do the full PPCG update
                eig_b = MH_bb.eigh(MS_bb)
        else:
            eig_b = MH_bb.eigh(MS_bb)
        if self.promote_inner_dtype:
            buffer_bb[:] = H_bb.conj()
            cmin = buffer_bb[:block, :nblocks]
        else:
            cmin = H_bb[:block, :nblocks].conj()

        if not xp.isfinite(H_bb).all():
            return True  # linalg error occured

        wfs.myeig_n[block_slice] = as_np(eig_b[:block])

        with tracectx('rotations', gpu=xp is not np):
            # Ye olde updates
            buff_bX.matrix.data[:block] = \
                cmin[:, :block] @ buff_bX.matrix.data[:block]
            Pbuf_abi.matrix.data[:block] = \
                cmin[:, :block] @ Pbuf_abi.matrix.data[:block]
            buff_bX.matrix.data[block:2 * block] = \
                cmin[:, block:] @ buff_bX.matrix.data[
                    block:nblocks]
            Pbuf_abi.matrix.data[block:2 * block] = \
                cmin[:, block:] @ Pbuf_abi.matrix.data[
                    block:nblocks]

            if self.include_cg:
                P_nX.matrix.data[block_slice] = \
                    buff_bX.matrix.data[block:2 * block]
                P3_ani.matrix.data[block_slice] = \
                    Pbuf_abi.matrix.data[block:2 * block]

            psit_nX.matrix.data[block_slice] = \
                buff_bX.matrix.data[:block] \
                + buff_bX.matrix.data[block:2 * block]
            P_ani.matrix.data[block_slice] = \
                Pbuf_abi.matrix.data[:block] \
                + Pbuf_abi.matrix.data[block:2 * block]

            return False
