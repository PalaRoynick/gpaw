from __future__ import annotations

from abc import ABC
from functools import partial

import numpy as np
from numpy.linalg import solve

from gpaw.core.atom_arrays import AtomArrays
from gpaw.core.matrix import Matrix, suggest_blocking
from gpaw.core.uniform_grid import UGArray, UGDesc
from gpaw.new.fd.hamiltonian import FDHamiltonian, FDKickHamiltonian
from gpaw.new.hamiltonian import Hamiltonian
from gpaw.new.lcao.hamiltonian import LCAOHamiltonian
from gpaw.new.lcao.wave_functions import LCAOWaveFunctions
from gpaw.new.pw.hamiltonian import PWHamiltonian
from gpaw.new.pwfd.wave_functions import PWFDWaveFunctions
from gpaw.new.rttddft.dataclasses import RTTDDFTState
from gpaw.new.wave_functions import WaveFunctions
from gpaw.tddft.solvers.cscg import CSCG
from gpaw.utilities.timing import nulltimer


def build_wf_propagator(name: str,
                        hamiltonian: Hamiltonian,
                        state: RTTDDFTState) -> WaveFunctionPropagator:
    cls = determine_wf_propagator_class(name, hamiltonian)
    return cls(hamiltonian=hamiltonian, state=state)


def determine_wf_propagator_class(name: str,
                                  hamiltonian: Hamiltonian,
                                  ) -> type[WaveFunctionPropagator]:
    # Determine mode from the type of the hamiltonian
    if isinstance(hamiltonian, LCAOHamiltonian):
        if name == 'numpy':
            return LCAONumpyPropagator
        else:
            raise ValueError(f'Unknown propagation algorithm: {name}')
    elif isinstance(hamiltonian, FDHamiltonian):
        if name == 'numpy':
            return FDNumpyPropagator
        else:
            raise ValueError(f'Unknown propagation algorithm: {name}')
    elif isinstance(hamiltonian, PWHamiltonian):
        raise NotImplementedError('PW TDDFT is not implemented')

    raise ValueError(f'Unknown hamiltonian: {hamiltonian} '
                     f'({type(hamiltonian)})')


def propagate_wave_functions_numpy(source_C_nM: np.ndarray,
                                   target_C_nM: np.ndarray,
                                   S_MM: np.ndarray,
                                   H_MM: np.ndarray,
                                   dt: float):
    r""" Solve for the new wave functions using numpy.

    The new wave functions :math:`C(t + \Delta t)` are solved from

    .. math:

        C(t + \Delta t) (S + i \Delta t / 2)
        = C(t) (S - i \Delta t / 2)

    Parameters
    ----------
    source_C_nM
        Old wave functions :math:`C(t).
    target_C_nM
        New wave functions :math:`C(t + \Delta t).
    S_MM
        Overlap :math:`S`.
    H_MM
        Hamiltonian :math:`S`.
    dt
        Time step in atomic units.
    """
    SjH_MM = S_MM + (0.5j * dt) * H_MM
    target_C_nM[:] = source_C_nM @ SjH_MM.conj().T
    target_C_nM[:] = solve(SjH_MM.T, target_C_nM.T).T


def propagate_wave_functions_blacs(source_C_nM: Matrix,
                                   target_C_nM: Matrix,
                                   S_MM: Matrix,
                                   H_MM: Matrix,
                                   dt: float):
    # Construct (S + i dt /2 H)
    SjH_MM = H_MM.copy()
    SjH_MM.data[:] *= 0.5j * dt
    SjH_MM += S_MM

    # Construct C(t) (S - i dt /2 H)
    source_C_nM.multiply(SjH_MM, opb='C', out=target_C_nM)

    # Solve for C(t + dt), replacing the data in target_C_nM
    target_C_nM.solve(SjH_MM)


class WaveFunctionPropagator(ABC):
    """
    Propagator object.

    Implementations are specific to the type of wave functions (LCAO/FD)
    and possibly to the parallelization scheme (Numpy/BLACS).

    Objects of this class have to be rebuilt after each change of the
    state (i.e. in each time step).

    Parameters
    ----------
    hamiltonian
        The Hamiltonian object.
    state
        The current RTTDDFT state.
    """

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: RTTDDFTState):
        raise NotImplementedError

    def propagate(self,
                  source_wfs: WaveFunctions,
                  target_wfs: WaveFunctions,
                  time_step: float):
        """ Propagate wave functions

        Parameters
        ----------
        source_wfs
            Source wave functions. Unchanged by the propagation
            if they are different from target_wfs.
        target_wfs
            Target wave functions where result is written.
        time_step
            Time step.
        """

        raise NotImplementedError


class LCAONumpyPropagator(WaveFunctionPropagator):

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: RTTDDFTState):
        assert isinstance(hamiltonian, LCAOHamiltonian)
        ham_calc = hamiltonian.create_hamiltonian_matrix_calculator(
            state.potential)
        self.ham_calc = ham_calc

    def propagate(self,
                  source_wfs: WaveFunctions,
                  target_wfs: WaveFunctions,
                  time_step: float):
        assert isinstance(source_wfs, LCAOWaveFunctions)
        assert isinstance(target_wfs, LCAOWaveFunctions)
        # The root of the domain comm has the right data
        H_MM = self.ham_calc.calculate_matrix(source_wfs)
        source_wfs.domain_comm.broadcast(H_MM.data, 0)
        nM = H_MM.shape[0]

        # Phi_n <- U[H(t)] Phi_n
        if False:
            propagate_wave_functions_numpy(source_wfs.C_nM.data,
                                           target_wfs.C_nM.data,
                                           source_wfs.S_MM.data,
                                           H_MM.data,
                                           time_step)
        else:
            comm = source_wfs.C_nM.dist.comm
            r, c, b = suggest_blocking(nM, comm.size)
            C_nM = source_wfs.C_nM.new(dist=(comm, r, c, b))
            target_C_nM = source_wfs.C_nM.new(dist=(comm, r, c, b))
            source_wfs.C_nM.redist(C_nM)
            _H_MM = H_MM.new(dist=(comm, r, c, b))
            H_MM.redist(_H_MM)
            S_MM = H_MM.new(dist=(comm, r, c, b))
            source_wfs.S_MM.redist(S_MM)
            propagate_wave_functions_blacs(C_nM,
                                           target_C_nM,
                                           S_MM,
                                           _H_MM,
                                           time_step)
            target_C_nM.redist(target_wfs.C_nM)

        # Make sure wfs.C_nM and (lazy) wfs.P_ani are in sync:
        target_wfs._P_ani = None


class FDNumpyPropagator(WaveFunctionPropagator):

    def __init__(self,
                 hamiltonian: Hamiltonian,
                 state: RTTDDFTState):
        assert isinstance(hamiltonian, FDHamiltonian)

        self.timer = nulltimer
        self.preconditioner = None

        self.Ht = partial(hamiltonian.apply,
                          state.potential.vt_sR,
                          state.potential.dedtaut_sR,
                          state.ibzwfs, state.density.D_asii)
        if isinstance(hamiltonian, FDKickHamiltonian):
            self.dH = hamiltonian.dH
        else:
            self.dH = state.potential.dH

        wfs = state.ibzwfs._wfs_u[0]
        self.dS = partial(wfs.setups.get_overlap_corrections,
                          wfs.P_ani.layout.atomdist,
                          wfs.xp)

        self.dSinv = wfs.setups.inverse_overlap_correction
        self.pt_aiX = wfs.pt_aiX

    def calculate_projections(self,
                              psit_nR: UGArray,
                              P_ani: AtomArrays | None = None) -> AtomArrays:
        """ Calculate the PAW projections

        Parameters
        ----------
        psit_nR
            Wave functions
        P_ani
            Write the PAW projections here. If None, a new object is allocated

        Returns
        -------
        The projections P_ani
        """
        if P_ani is None:
            P_ani = self.pt_aiX.empty(psit_nR.dims, psit_nR.comm)
            if psit_nR.data is None:
                raise RuntimeError('There are no projections or wavefunctions')
        self.pt_aiX.integrate(psit_nR, P_ani)

        return P_ani

    def propagate(self,
                  source_wfs: WaveFunctions,
                  target_wfs: WaveFunctions,
                  time_step: float):
        assert isinstance(source_wfs, PWFDWaveFunctions)
        assert isinstance(source_wfs.psit_nX, UGArray)
        assert isinstance(target_wfs, PWFDWaveFunctions)
        assert isinstance(target_wfs.psit_nX, UGArray)
        psit_nR = target_wfs.psit_nX
        psit_nR.data[:] = source_wfs.psit_nX.data

        # Empty arrays
        rhs_nR = psit_nR.new(zeroed=True)
        init_guess_nR = psit_nR.new(zeroed=True)
        hpsit_nR = psit_nR.new(zeroed=True)
        spsit_nR = psit_nR.new(zeroed=True)
        sinvhpsit_nR = psit_nR.new(zeroed=True)

        # Calculate right-hand side of equation
        # ( S + i H dt/2 ) psit(t+dt) = ( S - i H dt/2 ) psit(t)
        self.apply_hamiltonian(psit_nR, hpsit_nR)  # hpsit_nR <- H psit(t)
        self.apply_overlap_operator(psit_nR, spsit_nR)  # spsit_nR <- S psit(t)

        rhs_nR.data[:] = spsit_nR.data - 0.5j * time_step * hpsit_nR.data

        # Calculate (1 - i S^(-1) H dt) psit(t), which is an
        # initial guess for the conjugate gradient solver
        self.apply_inverse_overlap_operator(hpsit_nR, sinvhpsit_nR)
        init_guess_nR.data[:] = (psit_nR.data -
                                 1j * time_step * sinvhpsit_nR.data)

        # Solve A x = b where A is (S + i H dt/2) and b = rhs_kpt.psit_nG
        solver = CSCGAdapter(psit_nR.desc, self)
        solver.solve(init_guess_nR, rhs_nR, psit_nR, time_step)

        self.calculate_projections(psit_nR, target_wfs.P_ani)

    def dot(self, psit_nR: UGArray, out_nR: UGArray, time_step: float):
        """Applies the propagator matrix to the given wavefunctions.

        (S + i H dt/2 ) psit_nR

        Parameters
        ----------
        psit_nR
            The known wavefunctions.
        out_nR
            The result ( S + i H dt/2 ) psit_nR.
        """
        self.timer.start('Apply time-dependent operators')

        hpsit_nR = psit_nR.new()
        spsit_nR = psit_nR.new()

        self.apply_hamiltonian(psit_nR, hpsit_nR)
        self.apply_overlap_operator(psit_nR, spsit_nR)
        self.timer.stop('Apply time-dependent operators')

        out_nR.data[:] = spsit_nR.data + 0.5j * time_step * hpsit_nR.data

    def apply_preconditioner(self, psi, psin):
        """Solves preconditioner equation.

        Parameters
        ----------
        psi
            The known wavefunctions.
        psin
            The result.

        """
        self.timer.start('Solve TDDFT preconditioner')
        if self.preconditioner is not None:
            self.preconditioner.apply(self.kpt, psi, psin)
        else:
            psin[:] = psi
        self.timer.stop('Solve TDDFT preconditioner')

    def apply_hamiltonian(self,
                          psit_nR: UGArray,
                          out_nR: UGArray,
                          spin: int = 0):
        """
        Apply the Hamiltonian on wave functions.

        Parameters
        ----------
        psit_nR
            Wave functions.
        out_nR
            Result, i.e. Hamiltonian acting on wavefunctions.
        spin
            Spin.
        """
        out_nR.data[:] = 0
        self.Ht(psit_nG=psit_nR, out=out_nR, spin=spin)

        # apply the non-local part for each nucleus
        P_ani = self.calculate_projections(psit_nR)
        P2_ani = P_ani.new()
        self.dH(P_ani, P2_ani, spin)

        # add partial wave pt_nG to psit_nG with proper coefficient
        self.pt_aiX.add_to(out_nR, P2_ani)

    def apply_overlap_operator(self,
                               psit_nR: UGArray,
                               out_nR: UGArray):
        """
        Apply the overlap operator wave functions on the wave functions:

        ^  ~         ~             ~ a       a   a
        S |ψ (t)〉= |ψ (t)〉+  Σ  |p  (t)〉ΔO   P
            n         n       aij   i        ij  nj

        Parameters
        ----------
        psit_nR
            Wave functions
        out_nR
            Result, i.e. overlap operator acting on wavefunctions
        """

        out_nR.data[:] = psit_nR.data

        P_ani = self.calculate_projections(psit_nR)
        P2_ani = P_ani.new()
        dS_aii = self.dS()
        P_ani.block_diag_multiply(dS_aii, P2_ani)
        self.pt_aiX.add_to(out_nR, P2_ani)

    def apply_inverse_overlap_operator(self,
                                       psit_nR: UGArray,
                                       out_nR: UGArray):
        """
        Apply the approximate inverse overlap operator on the wave functions:

        ^ -1  ~         ~             ~ a       a   a
        S    |ψ (t)〉= |ψ (t)〉+  Σ  |p  (t)〉ΔC   P
               n         n       aij   i        ij  nj

        Parameters
        ----------
        psit_nR
            Wave functions
        out_nR
            Result, i.e. inverse overlap operator acting on wavefunctions
        """
        out_nR.data[:] = psit_nR.data

        P_ani = self.calculate_projections(psit_nR)
        P2_ani = P_ani.new()
        self.dSinv(P_ani, out_ani=P2_ani)  # P2 is ΔC_ij @ P_nj
        self.pt_aiX.add_to(out_nR, P2_ani)


class CSCGAdapter:

    """ Adapter in order to reuse the CSCG solver from the old code

    """

    def __init__(self,
                 desc: UGDesc,
                 propagator: FDNumpyPropagator):
        self.solver = CSCG()
        self.solver.initialize(desc._gd, nulltimer)
        self.desc = desc
        self.propagator = propagator
        self._time_step: float | None = None

    @property
    def time_step(self) -> float:
        assert self._time_step is not None
        return self._time_step

    def solve(self,
              init_guess_nR: UGArray,
              rhs_nR: UGArray,
              out_nR: UGArray,
              time_step: float):
        self._time_step = time_step
        out_nR.data[:] = init_guess_nR.data
        self.solver.solve(self, out_nR.data, rhs_nR.data)
        self._time_step = None

    def dot(self, psit_nR: np.ndarray, out_nR: np.ndarray):
        """Applies the propagator matrix to the given wavefunctions.

        (S + i H dt/2 ) psi

        Parameters
        ----------
        psi: List of coarse grids
            the known wavefunctions
        psin: List of coarse grids
            the result ( S + i H dt/2 ) psi

        """
        _psit_nR = self.desc.from_data(psit_nR)
        _out_nR = self.desc.from_data(out_nR)
        self.propagator.dot(_psit_nR, _out_nR, self.time_step)

    def apply_preconditioner(self, psi, psin):
        self.propagator.apply_preconditioner(psi, psin)
