from __future__ import annotations

from functools import partial

import numpy as np
from ase.units import Ha

from gpaw import debug
# from gpaw.typing import Array2D
from gpaw.core import PWDesc  # , PWArray
from gpaw.core.matrix import Matrix
from gpaw.gpu import as_np
from gpaw.new import tracectx  # , trace
from gpaw.new.pwfd.davidson import sliced_preconditioner
# from gpaw.mpi import broadcast_exception
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.mpi import serial_comm

# from gpaw.utilities import as_real_dtype


class PPCG(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 hamiltonian,
                 convergence,
                 domain_band_comm,
                 max_niter=20,
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
        max_niter : int, optional
            Maximum number of iterations, to achieve minimum residual.
        niter : int, optional
            Target number of iterations. Default is 2.
        min_niter : int, optional
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
            hamiltonian=hamiltonian,
            convergence=convergence,
            scalapack_parameters=scalapack_parameters,
            nbands=nbands,
            max_buffer_mem=max_buffer_mem,
            domain_band_comm=domain_band_comm)

        self.nbands = nbands
        self.wf_grid = wf_grid
        self.band_comm = band_comm
        self.max_niter = max_niter
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
        self.hamiltonian = hamiltonian

    def __str__(self):
        return super().__str__() + f'niter={self.niter}\n'

    def _initialize(self, ibzwfs):
        xp = ibzwfs.xp

        if self.max_blocksize is None:
            if xp == np:
                self.max_blocksize = 32
            else:
                self.max_blocksize = 256

        super()._initialize(ibzwfs)
        if self.include_cg:
            self._allocate_work_arrays(ibzwfs, shape=(2,))
        else:
            self._allocate_work_arrays(ibzwfs, shape=(1,))
        self._allocate_buffer_arrays(ibzwfs, shape=(1,))

        band_comm = ibzwfs.band_comm
        self.band_comm = band_comm
        wfs = ibzwfs._wfs_u[0]
        assert isinstance(wfs, PWFDWaveFunctions)
        B = ibzwfs.nbands
        b = wfs.psit_nX.mydims[0]

        # Blocksize should be close to a divisor of b.
        bdist = (b + wfs.psit_nX.desc.comm.size - 1) \
            // wfs.psit_nX.desc.comm.size
        divisor = (bdist + self.max_blocksize - 1) \
            // self.max_blocksize
        self.blocksize = max(((bdist + divisor - 1) // divisor), 1)
        self.nblocksizes = 3 * self.blocksize \
            if self.include_cg else 2 * self.blocksize
        dtype = wfs.psit_nX.desc.dtype

        if self.tolerances is None:
            self.tolerances = (0, 4e-1, self.residual_target)

        assert len(self.tolerances) == 3
        # --------------- Convergence parameters ---------------
        # Mostly relevant for single precision, however the
        # breakout_tolerance could be used to speed up convergence
        # in double precision.
        #
        # tolerance :
        #   Freeze bands with residual < tolerance
        #   improves numerical stability at the cost of
        #   minimum achievable residual.
        self.tolerance = self.tolerances[0]
        # tolerance :
        #   Minimum residual required for breakout, unless
        #   iter > self.maximum_niter.
        self.minimum_res = self.tolerances[1] / Ha**2
        # breakout_tolerance :
        #   Stop iteration if sum(residual_ns) < breakout_tolerance
        #   breakout_tolerance saves time at the cost of minimum
        #   achievable residual. Can also be used to improve numerical
        #   stability.
        self.breakout_res = self.tolerances[2] / Ha**2

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


    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, potential,
                 dS_aii, weight_n):

        with tracectx('Initialize'):
            dH = partial(potential.deltaH, spin=wfs.spin)

            M_nn = self.M_nn

            xp = M_nn.xp

            psit_nX = wfs.psit_nX
            b = psit_nX.mydims[0]

            residual_nX = psit_nX.new(data=self.work_arrays[0, :b])
            if self.include_cg:
                P_nX = psit_nX.new(data=self.work_arrays[1, :b])

            wfs.subspace_diagonalize(
                Ht, dH,
                psit2_nX=residual_nX,
                data_buffer=self.data_buffers[0],
                scalapack_parameters=self.scalapack_parameters)

            P_ani = wfs.P_ani
            P2_ani = P_ani.new()
            P3_ani = P_ani.new()
            Ptemp_ani = P_ani.new()
            P_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)

            layout0 = P_ani.layout.new(atomdist=serial_comm)
            Pbuf_abi = layout0.empty(
                (self.nblocksizes, ) + psit_nX.dims[1:])
            HPbuf_abi = layout0.empty(
                (self.nblocksizes - self.blocksize, ) + psit_nX.dims[1:])

            domain_comm = psit_nX.desc.comm
            band_comm = psit_nX.comm

            if weight_n is None:
                weight_n = np.zeros(b)
                weight_n[:(len(weight_n) * 3) // 4] = 1.0

            buffer_array_nX = psit_nX.create_work_buffer(self.data_buffers[0])

            if domain_comm.size > 1:
                key = tuple(psit_nX.desc.kpt_c)
                desc0 = self.hamiltonian.pw_cache.get(key)
            else:
                desc0 = psit_nX.desc

            buff_bX = desc0.empty(
                (self.nblocksizes, ) +
                 psit_nX.dims[1:], xp=psit_nX.xp)
            Hbuff_bX = desc0.empty(
                (self.nblocksizes - self.blocksize, ) +
                 psit_nX.dims[1:], xp=psit_nX.xp)

            dS_Aii = dS_aii.gather(copy=True, broadcast=True)
            dH_Asii = potential.dH_asii.gather(copy=True, broadcast=True)
            dH_Aii = dH_Asii[:, wfs.spin]

            vt_sR = potential.vt_sR.gather(broadcast=True)
            Ht_args = list(Ht.args)
            Ht_args[0] = vt_sR
            Ht_local = partial(Ht.func, *Ht_args, **Ht.keywords)

        with tracectx('Residual'):
            calculate_residuals(wfs.psit_nX,
                                residual_nX,
                                wfs.pt_aiX,
                                wfs.P_ani,
                                wfs.myeig_n,
                                dH, dS_aii, P2_ani, P3_ani)

            error_n = as_np(residual_nX.norm2())
            if len(error_n.shape) > 1:
                error_n = error_n.sum(axis=1)
            if self.allow_dynamic_breakout and self.tolerance:
                active_indicies = np.greater(error_n, self.tolerance)
                active_indicies = np.where(active_indicies)[0]
            else:
                active_indicies = np.arange(b)
            error = weight_n @ error_n
            b_error = band_comm.sum_scalar(error) / \
                max(band_comm.sum_scalar(weight_n.sum()), 0.5)
            if self.check_breakout(-1, b_error, active_indicies):
                if debug:
                    psit_nX.sanity_check()
                break_after_update = True
            else:
                break_after_update = False

        for i in range(max(self.niter, self.max_niter)):
            with tracectx('Residual'):
                sliced_preconditioner(psit_nX, residual_nX,
                                      buffer=buffer_array_nX,
                                      precon=self.preconditioner)
                wfs.pt_aiX.integrate(residual_nX, out=P2_ani)
                residual_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                            domain_sum=False,
                                            symmetric=False)
                P2_ani.matrix.multiply(Ptemp_ani, opb='C', symmetric=False,
                                       beta=1, out=M_nn)
                domain_comm.sum(M_nn.data)

                M_nn.multiply(psit_nX, out=residual_nX, beta=1.0, alpha=-1.0)
                M_nn.multiply(P_ani, out=P2_ani, beta=1.0, alpha=-1.0)

            loop_limit = len(active_indicies) if self.allow_dynamic_breakout \
                else (psit_nX.dims[0] + band_comm.size - 1) // band_comm.size
            active_bands = len(active_indicies)

            with tracectx('Block-diagonal Update'):
                new_eigs_n = np.zeros_like(wfs.myeig_n)  # New eigenvalues
                mbs = self.blocksize

                iterables = (
                    psit_nX.iter_by_band(
                        buff_bX[0 * mbs:1 * mbs], out_nX=psit_nX),
                    residual_nX.iter_by_band(
                        buff_bX[1 * mbs:2 * mbs]),  # out_nX=residual_nX),
                    P_ani.iter_by_band(
                        Pbuf_abi[:, 0 * mbs:1 * mbs], out_axi=P_ani),
                    P2_ani.iter_by_band(
                        Pbuf_abi[:, 1 * mbs:2 * mbs], out_axi=None))

                if self.include_cg:
                    iterables += (
                        P_nX.iter_by_band(
                            buff_bX[2 * mbs:3 * mbs],
                            out_nX=None if break_after_update else P_nX),
                        P3_ani.iter_by_band(
                            Pbuf_abi[:, 2 * mbs:3 * mbs],
                            out_axi=None if break_after_update else P3_ani))


                for iter_info in zip(*iterables):
                    mybands = iter_info[0][0]
                    if len(mybands) == 0:
                        # Nothing to do.
                        continue

                    for info in iter_info:
                        assert all(info[0] == mybands)

                    block = len(mybands)
                    if block < mbs:
                        # Squish the buffer array together
                        buff_bX.data[block:2 * block] = \
                            buff_bX.data[mbs:mbs + block]
                        Pbuf_abi.data[block:2 * block] = \
                            Pbuf_abi.data[mbs:mbs + block]
                        if self.include_cg:
                            buff_bX.data[2 * block:3 * block] = \
                                buff_bX.data[2 * mbs:2 * mbs + block]
                            Pbuf_abi.data[2 * block:3 * block] = \
                                Pbuf_abi.data[2 * mbs:2 * mbs + block]

                    block_slice = mybands

                    if i > 0 and self.include_cg:
                        nblocks = 3 * block
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
                        buffer_bb = \
                            S_bb

                    MBuf_bb = Matrix(M=nblocks - block, N=nblocks,
                                     data=buffer_bb[block:nblocks],
                                     xp=xp)

                    Pbuf_abi[:, block:nblocks].block_diag_multiply(
                        dS_Aii, out_ani=HPbuf_abi[:, :nblocks - block])
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

                    Ht_local(buff_bX[block:nblocks],
                             out=Hbuff_bX[:nblocks - block])
                    # dH_Aii(Pbuf_abi[:, block:nblocks],
                    #        out_ani=HPbuf_abi[:, :nblocks - block])
                    Pbuf_abi[:, block:nblocks].block_diag_multiply(dH_Aii,
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

                    H_bb[:block, :block] = xp.diag(wfs.myeig_n[block_slice])

                    S_bb[:block, :block] = xp.eye(block)
                    MS_bb.tril2full()

                    if nblocks > 2 * block:
                        # Eigh approach
                        # A, temp_bb = xp.linalg.eigh(S_bb, 'L')
                        # if xp is not np:
                        #     pos_defness = A.get()[0]
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
                        break_after_update |= self.allow_dynamic_breakout
                        new_eigs_n[block_slice] = wfs.myeig_n[block_slice]
                        continue
                    new_eigs_n[block_slice] = as_np(eig_b[:block])

                    with tracectx('rotations', gpu=xp is not np):
                        # Ye olde updates
                        # First rotate the wavefunctions
                        buff_bX.matrix.data[:block] = \
                            cmin[:, :block] @ buff_bX.matrix.data[:block]
                        Pbuf_abi.matrix.data[:block] = \
                            cmin[:, :block] @ Pbuf_abi.matrix.data[:block]

                        # Now onto the residual or cg if available
                        dynslice = slice(2 * mbs, 2 * mbs + block) \
                            if self.include_cg else \
                            slice(mbs, mbs + block)

                        buff_bX.matrix.data[dynslice] = \
                            cmin[:, block:] @ buff_bX.matrix.data[
                                block:nblocks]
                        Pbuf_abi.matrix.data[dynslice] = \
                            cmin[:, block:] @ Pbuf_abi.matrix.data[
                                block:nblocks]

                        # Add the Hpsi to the wavefunctions
                        buff_bX.matrix.data[:block] += \
                            buff_bX.matrix.data[dynslice]
                        Pbuf_abi.matrix.data[:block] += \
                            Pbuf_abi.matrix.data[dynslice]

                        # Finally, calculate new Hpsi (only if we have more
                        # iterations to go). This is faster if the FFT and GEMM
                        # is faster than the all2allv involved. Currently,
                        # this rarely seems to be the case.
                        # if not break_after_update:
                        #     Ht_local(buff_bX[:block], out=buff_bX[mbs:mbs + block])

            wfs.eig_n[:] = 0
            wfs.myeig_n[:] = new_eigs_n
            band_comm.sum(wfs._eig_n)
            domain_comm.sum(wfs._eig_n)
            wfs.orthonormalized = False
            if break_after_update:
                break

            with tracectx('Residual'):
                # Subspace diagonialization needed every once in a while
                if (i + 1) % self.rr_modulo == 0:
                    wfs.subspace_diagonalize(
                        Ht, dH,  # Ht can be removed if Hpsi is calculated
                        psit2_nX=residual_nX,
                        data_buffer=self.data_buffers[0],
                        calculate_energy=False,
                        scalapack_parameters=self.scalapack_parameters)
                else:
                    wfs.orthonormalize(data_buffer=self.data_buffers[0])
                    # rotate_also=[residual_nX,]) - Use this if calculating
                    # the Hpsi in the non-communicating loop. Otherwise:
                    Ht(psit_nX, residual_nX)

                calculate_residuals(wfs.psit_nX,
                                    residual_nX,
                                    wfs.pt_aiX,
                                    wfs.P_ani,
                                    wfs.myeig_n,
                                    dH, dS_aii, P2_ani, Ptemp_ani)

                error_n = as_np(residual_nX.norm2())
                if len(error_n.shape) > 1:
                    error_n = error_n.sum(axis=1)

                if self.allow_dynamic_breakout and self.tolerance:
                    active_indicies = np.greater(error_n, self.tolerance),
                    active_indicies = np.where(active_indicies)[0]
                error = weight_n @ error_n
                b_error = band_comm.sum_scalar(error) / \
                    max(band_comm.sum_scalar(weight_n.sum()), 0.5)

                if self.check_breakout(i, b_error, active_indicies):
                    # Set 'break_after_update = True', causing the
                    # loop to break at the next iteration. This gives us
                    # one more cheap iteration (since we already
                    # calculated the residual).
                    break_after_update = True

            P_ani.block_diag_multiply(dS_aii, out_ani=Ptemp_ani)
            if self.include_cg:
                with tracectx('P-update'):
                    P_nX.matrix_elements(psit_nX, cc=True, out=M_nn,
                                         domain_sum=False,
                                         symmetric=False)
                    P3_ani.matrix.multiply(Ptemp_ani, opb='C',
                                           symmetric=False,
                                           beta=1, out=M_nn)
                    domain_comm.sum(M_nn.data)
                    M_nn.multiply(psit_nX, out=P_nX, beta=1.0, alpha=-1.0)
                    M_nn.multiply(P_ani, out=P3_ani, beta=1.0, alpha=-1.0)

        if not wfs.orthonormalized:
            wfs.orthonormalize(residual_nX)

        if debug:
            psit_nX.sanity_check()

        return error

    def check_breakout(self, iter, residual, active_indicies):
        band_comm = self.band_comm
        breakout = False
        if self.allow_dynamic_breakout:
            breakout |= band_comm.sum_scalar(len(active_indicies)) == 0
            breakout |= (residual < self.breakout_res and \
                (iter + 3) > self.min_niter)
            breakout |= (residual < self.minimum_res and (iter + 3) > self.niter)
        else:
            breakout |= (iter + 3) > self.niter
        return breakout
