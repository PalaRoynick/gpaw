from __future__ import annotations

from functools import partial
from pprint import pformat

import numpy as np
from gpaw import debug
from gpaw.core.matrix import Matrix
from gpaw.gpu import as_np
from gpaw.mpi import broadcast_exception
from gpaw.new.pwfd.eigensolver import PWFDEigensolver, calculate_residuals
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.typing import Array2D
from gpaw.new import trace, tracectx
from gpaw.utilities import as_real_dtype


class Davidson(PWFDEigensolver):
    def __init__(self,
                 nbands: int,
                 wf_grid,
                 band_comm,
                 preconditioner_factory,
                 converge_bands='occupied',
                 niter=2,
                 scalapack_parameters=None):
        super().__init__(
            preconditioner_factory,
            converge_bands)
        self.niter = niter
        self.H_NN: Matrix
        self.S_NN: Matrix
        self.M_nn: Matrix

    def __str__(self):
        return pformat(dict(name='Davidson',
                            niter=self.niter,
                            converge_bands=self.converge_bands))

    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        self._allocate_work_arrays(ibzwfs, shape=(2,))

        wfs = ibzwfs.wfs_qs[0][0]
        assert isinstance(wfs, PWFDWaveFunctions)
        domain_comm = wfs.psit_nX.desc.comm
        band_comm = wfs.band_comm

        B = ibzwfs.nbands
        xp = ibzwfs.xp
        dtype = wfs.psit_nX.desc.dtype
        if domain_comm.rank == 0 and band_comm.rank == 0:
            self.H_NN = Matrix(2 * B, 2 * B, dtype=dtype, xp=xp)
            self.S_NN = Matrix(2 * B, 2 * B, dtype=dtype, xp=xp)
        else:
            self.H_NN = self.S_NN = Matrix(0, 0)

        self.M_nn = Matrix(B, B, dtype=dtype,
                           dist=(band_comm, band_comm.size),
                           xp=xp)

    def iterate1(self,
                 wfs: PWFDWaveFunctions,
                 Ht, dH, dS_aii, weight_n):
        H_NN = self.H_NN
        S_NN = self.S_NN
        M_nn = self.M_nn

        xp = M_nn.xp

        psit_nX = wfs.psit_nX
        B = psit_nX.dims[0]  # number of bands
        eig_N = xp.empty(2 * B)
        b = psit_nX.mydims[0]

        if weight_n is not None:
            weight_n = xp.asarray(weight_n, dtype=as_real_dtype(wfs.dtype))

        psit2_nX = psit_nX.new(data=self.work_arrays[0, :b])
        psit3_nX = psit_nX.new(data=self.work_arrays[1, :b])

        wfs.subspace_diagonalize(Ht, dH,
                                 work_array=psit2_nX.data,
                                 Htpsit_nX=psit3_nX)
        residual_nX = psit3_nX  # will become (H-e*S)|psit> later

        P_ani = wfs.P_ani
        P2_ani = P_ani.new()
        P3_ani = P_ani.new()

        domain_comm = psit_nX.desc.comm
        band_comm = psit_nX.comm
        is_domain_band_master = domain_comm.rank == 0 and band_comm.rank == 0

        M0_nn = M_nn.new(dist=(band_comm, 1, 1))

        if domain_comm.rank == 0:
            eig_N[:B] = xp.asarray(wfs.eig_n)

        @trace
        def me(a, b, function=None):
            """Matrix elements"""
            return a.matrix_elements(b,
                                     domain_sum=False,
                                     out=M_nn,
                                     function=function,
                                     cc=True)

        Ht = partial(Ht, out=residual_nX)
        calculate_residuals(wfs.psit_nX,
                            residual_nX,
                            wfs.pt_aiX,
                            wfs.P_ani,
                            wfs.myeig_n,
                            dH, dS_aii, P2_ani, P3_ani)

        def copy(C_nn: Array2D) -> None:
            domain_comm.sum(M_nn.data, 0)
            if domain_comm.rank == 0:
                M_nn.redist(M0_nn)
                if band_comm.rank == 0:
                    C_nn[:] = M0_nn.data

        flag = False
        error = np.inf
        for i in range(self.niter):
            if i == self.niter - 1 or True:  # last iteration
                if flag:
                    i = self.niter - 1
                # Calculate error before we destroy residuals:
                if weight_n is None:
                    weight_n = xp.ones(b, dtype=as_real_dtype(wfs.dtype))
                n_error = as_np(weight_n @ residual_nX.norm2()).sum()
                print('n_error', np.log10(n_error))
                if error / n_error < 2 or error < 1e-4 or np.isnan(n_error):
                    print(f'Breaking at iter {i} / {self.niter}')
                    break
                error = n_error

            self.preconditioner(psit_nX, residual_nX, out=psit2_nX)

            # Calculate projections
            wfs.pt_aiX.integrate(psit2_nX, out=P2_ani)

            with tracectx('Matrix elements'):
                # <psi2 | H | psi2>
                me(psit2_nX, psit2_nX, function=Ht)
                dH(P2_ani, out_ani=P3_ani)
                P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                       out=M_nn)
                copy(H_NN.data[B:, B:])
                
                # <psi2 | H | psi>
                me(residual_nX, psit_nX)
                P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
                copy(H_NN.data[B:, :B])

                # <psi2 | S | psi2>
                me(psit2_nX, psit2_nX)
                P2_ani.block_diag_multiply(dS_aii, out_ani=P3_ani)
                P2_ani.matrix.multiply(P3_ani, opb='C', symmetric=True, beta=1,
                                       out=M_nn)
                copy(S_NN.data[B:, B:])
                M2_nn = M_nn
                
                # <psi2 | S | psi>
                me(psit2_nX, psit_nX)
                P3_ani.matrix.multiply(P_ani, opb='C', beta=1.0, out=M_nn)
                copy(S_NN.data[B:, :B])

            with tracectx('Diagonalize'):
                with broadcast_exception(domain_comm):
                    with broadcast_exception(band_comm):
                        if is_domain_band_master:
                            H_NN.tril2full()
                            S_NN.tril2full()
                            H_NN.data[:B, :B] = xp.diag(eig_N[:B])
                            S_NN.data[:B, :B] = xp.eye(B)
                            
                            '''
                            D_N = S_NN.eigh()
                            b_inds = D_N > xp.max(D_N) * 5e-6
                            b = int(np.sum(b_inds))
                            H2_NN = Matrix(b, b, dtype=H_NN.dtype, xp=H_NN.xp)
                            S2_NN = Matrix(b, b, dtype=S_NN.dtype, xp=S_NN.xp)
                            H2_NN.data[:] = S_NN.data[b_inds].conj() @ H_NN.data @ S_NN.data[b_inds].T
                            S2_NN.data[:] = xp.diag(D_N[b_inds])
                            
                            assert b >= B
                            S2_NN.data[:] = xp.diag(1/xp.sqrt(D_N[b_inds]))
                            H2_NN.data[:] = S2_NN.data.conj() @ H2_NN.data @ S2_NN.data
                            eig_N[:B] = H2_NN.eigh()[:B]
                            H_NN.data[:B, :] = H2_NN.data[:B, :] @ S2_NN.data @ S_NN.data[b_inds]
                            '''
                            
                            S_nn = S_NN.data[B:, B:]
                            D_n, U_nn = xp.linalg.eigh(S_nn)
                            b_inds = D_n > 1e-6
                            b = int(np.sum(b_inds))
                            U_nn = U_nn[:, b_inds]
                            H2_NN = Matrix(B + b, B + b, dtype=H_NN.dtype, xp=H_NN.xp)
                            S2_NN = Matrix(B + b, B + b, dtype=S_NN.dtype, xp=S_NN.xp)
                            S2_NN.data[B:, :B] = U_nn.conj().T @ S_NN.data[B:, :B]
                            H2_NN.data[B:, :B] = U_nn.conj().T @ H_NN.data[B:, :B]
                            S2_NN.data[B:, B:] = xp.diag(D_n[b_inds])
                            H2_NN.data[B:, B:] = U_nn.conj().T @ H_NN.data[B:, B:] @ U_nn
                            H2_NN.data[:B, :B] = xp.diag(eig_N[:B])
                            S2_NN.data[:B, :B] = xp.eye(B)

                            eig_N[:B] = H2_NN.eigh(S2_NN)[:B]
                            H_NN.data[:B, :B] = H2_NN.data[:B, :B]
                            H_NN.data[:B, B:] = H2_NN.data[:B, B:] @ U_nn.T
                            wfs._eig_n = as_np(eig_N[:B])
                if domain_comm.rank == 0:
                    band_comm.broadcast(wfs.eig_n, 0)
                domain_comm.broadcast(wfs.eig_n, 0)

                if domain_comm.rank == 0:
                    if band_comm.rank == 0:
                        M0_nn.data[:] = H_NN.data[:B, :B]
                        M0_nn.complex_conjugate()
                    M0_nn.redist(M_nn)
                domain_comm.broadcast(M_nn.data, 0)

            with tracectx('Rotate Psi'):
                M_nn.multiply(psit_nX, out=residual_nX)
                M_nn.multiply(P_ani, out=P3_ani)

                if domain_comm.rank == 0:
                    if band_comm.rank == 0:
                        M0_nn.data[:] = H_NN.data[:B, B:]
                        M0_nn.complex_conjugate()
                    M0_nn.redist(M_nn)
                domain_comm.broadcast(M_nn.data, 0)
                M_nn.multiply(psit2_nX, beta=1.0, out=residual_nX)
                M_nn.multiply(P2_ani, beta=1.0, out=P3_ani)
                psit_nX.data[:] = residual_nX.data
                P_ani, P3_ani = P3_ani, P_ani
                wfs._P_ani = P_ani

            if i < self.niter - 1:
                Ht(psit_nX)
                calculate_residuals(
                    wfs.psit_nX,
                    residual_nX,
                    wfs.pt_aiX, wfs.P_ani, wfs.myeig_n,
                    dH, dS_aii, P2_ani, P3_ani)
            else:
                break

        if debug:
            psit_nX.sanity_check()

        return error
