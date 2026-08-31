import numpy as np

from gpaw.new.density import Density
from gpaw.core.arrays import DomainType, XArray
from gpaw.core.atom_arrays import AtomArraysLayout, AtomArrays
from gpaw.core.atom_arrays import AtomDistribution
from gpaw.setup import Setups
from gpaw.typing import ArrayND, Self
from gpaw.mixer.metric import BaseMetric
from gpaw.mpi import MPIComm
from gpaw import GPAW_NO_C_EXTENSION


class BaseMixer:
    nmaxold = 1

    def __init__(self,
                 metric: BaseMetric,
                 desc: DomainType,
                 atomdist: AtomDistribution,
                 setups: Setups,
                 relpos_ac: ArrayND,
                 ncomponents: int,
                 world: MPIComm,
                 xp=np):
        """Base class for density mixers. Also operates as the
        not-mixing mixer.

        Parameters
        ----------
        metric:
            Metric object used to calculate inner products between densities.
        desc:
            Domain descriptor for the density (e.g., UniformGrid).
        atomdist:
            Distribution of atoms across MPI ranks.
        setups:
            Collection of PAW setups.
        relpos_ac:
            Positions of atoms relative to the unit cell.
        ncomponents:
            Number of density components (1 for non-polarized, 2 for polarized,
            4 for non-collinear).
        world:
            MPI communicator for synchronization across k-points/processes.
        xp:
            Array namespace (numpy or cupy).
        """
        self.metric = metric
        self.desc = desc
        self.ncomponents = ncomponents
        self.xp = xp
        self.setups = setups
        self.world = world  # For syncing across kpts
        self.atom_layout = AtomArraysLayout(
            [(setup.ni, setup.ni) for setup in setups],
            atomdist=atomdist, dtype=float if ncomponents < 4 else complex,
            xp=xp)
        self.histories: list[DensityHistory] = []
        self.ghat_aLr = setups.create_compensation_charges(desc,
                                                           relpos_ac,
                                                           atomdist,
                                                           xp=np)
        self.taut_hsR: DensityHistory | None = None
        # XXX: np -> xp once the GPU LFC kernels work
        self._initialize_history_()

    def _initialize_history_(self):
        self.nt_hsX = DensityHistory(
            nmaxold=1,
            ncomponents=self.ncomponents,
            desc=self.desc,
            atom_layout=self.atom_layout,
            xp=self.xp)
        self.R_hsX = DensityHistory(
            nmaxold=1,
            ncomponents=self.ncomponents,
            desc=self.desc,
            atom_layout=self.atom_layout,
            xp=self.xp)
        self.histories = [self.nt_hsX, self.R_hsX]

    def mix(self, density: Density) -> float:
        if self.taut_hsR is None and density.taut_sR is not None:
            self.taut_hsR = DensityHistory(
                nmaxold=self.nmaxold,
                ncomponents=self.ncomponents,
                desc=self.desc,
                xp=self.xp)
            self.Rtaut_hsR = DensityHistory(
                nmaxold=self.nmaxold,
                ncomponents=self.ncomponents,
                desc=self.desc,
                xp=self.xp)
            self.histories += [self.taut_hsR, self.Rtaut_hsR]

        nt_sX = density.nt_sR
        D_asii = density.D_asii
        taut_sR = density.taut_sR

        if self.ncomponents == 2:
            # Don't mix each channel,
            # mix total and magnetization instead (for metric reasons)
            nt2_sX = nt_sX.copy()
            D2_asii = D_asii.copy()
            nt2_sX.data[0] = nt_sX.data[0] + nt_sX.data[1]
            D2_asii.data[0] = D_asii.data[0] + D_asii.data[1]
            nt2_sX.data[1] = nt_sX.data[0] - nt_sX.data[1]
            D2_asii.data[1] = D_asii.data[0] - D_asii.data[1]
            if taut_sR is not None:
                taut2_sR = taut_sR.copy()
                taut2_sR.data[0] = taut_sR.data[0] + taut_sR.data[1]
                taut2_sR.data[1] = taut_sR.data[0] - taut_sR.data[1]
        else:
            nt2_sX = nt_sX
            D2_asii = D_asii
            if taut_sR is not None:
                taut2_sR = taut_sR

        alpha_h, beta_h, dnt = self._mix(nt2_sX, D2_asii, density)
        if taut_sR is not None:
            assert self.taut_hsR is not None

            if self.taut_hsR.nold == 0:
                self.taut_hsR.add(taut2_sR)
                taut2_sR.data[:] = 0
                self.taut_hsR.add_linear_combination(alpha_h, out=taut2_sR)
            else:
                Rtaut_sR = self.Rtaut_hsR.next()
                last_taut_sR = self.taut_hsR[-1]
                assert isinstance(Rtaut_sR, XArray)
                assert isinstance(last_taut_sR, XArray)
                self.calculate_residual(taut2_sR, last_taut_sR, Rtaut_sR)
                taut2_sR.data[:] = 0
                self.taut_hsR.add_linear_combination(alpha_h, out=taut2_sR)
                self.Rtaut_hsR.add_linear_combination(beta_h, out=taut2_sR)
                self.taut_hsR.maybe_delete_oldest()
                self.taut_hsR.add(taut2_sR)

        for hist in self.histories:
            hist.maybe_delete_oldest()

        for h in self.histories:
            print(h.nold, end=', ')

        if self.ncomponents == 2:
            nt_sX.data[0] = 0.5 * (nt2_sX.data[0] + nt2_sX.data[1])
            nt_sX.data[1] = 0.5 * (nt2_sX.data[0] - nt2_sX.data[1])
            D_asii.data[0] = 0.5 * (D2_asii.data[0] + D2_asii.data[1])
            D_asii.data[1] = 0.5 * (D2_asii.data[0] - D2_asii.data[1])
            if taut_sR is not None:
                taut_sR.data[0] = 0.5 * (taut2_sR.data[0] + taut2_sR.data[1])
                taut_sR.data[1] = 0.5 * (taut2_sR.data[0] - taut2_sR.data[1])

        return dnt

    def _mix(self, nt_sX: XArray,
             D_asii: AtomArrays,
             density: Density) -> tuple:
        if self.nt_hsX.nold == 0:
            self.nt_hsX.add(nt_sX, D_asii)
            return [1,], [0,], np.inf

        R_sX, R_asii = self.R_hsX.next()
        self.calculate_residual(nt_sX, self.nt_hsX[-1][0], R_sX)
        self.calculate_paw_residual(D_asii.data, self.nt_hsX[-1][1], R_asii)

        Rc_sX = self.add_compensation_charge(R_sX, R_asii, density,
                                             add_delta0=False)
        dNt = self.calculate_charge_sloshing(Rc_sX)
        self.nt_hsX.add(nt_sX, D_asii)
        return [1,], [1,], dNt

    def calculate_charge_sloshing(self, res_sX: XArray) -> float:
        slosh_sX = self.desc.empty(self.ncomponents, xp=self.xp)
        slosh_sX.data[:] = self.xp.abs(res_sX.data)
        dNt = slosh_sX.integrate().sum()
        return dNt if self.xp == np else dNt.get()

    def calculate_residual(self, ntc_sX: XArray, prev_sX: XArray,
                           res_sX: XArray) -> XArray:
        assert ntc_sX.data is not res_sX.data
        # We do density history first, in case res_sX is the same as
        # self.density_history[-1].
        res_sX.data[:] = -prev_sX.data
        res_sX.data += ntc_sX.data
        return res_sX

    def calculate_paw_residual(self,
                               D_asii: ArrayND,
                               prev_asii: ArrayND,
                               res_asii: ArrayND) -> ArrayND:
        assert D_asii is not res_asii
        res_asii[:] = -prev_asii
        res_asii += D_asii
        return res_asii

    def add_compensation_charge(self, n_sX: XArray,
                                D_asii_data: AtomArrays | ArrayND,
                                density: Density,
                                out: XArray | None = None,
                                add_delta0: bool = False) -> XArray:
        # Prepare output array
        if out is None:
            out = n_sX.copy()
        else:
            out.data[:] = n_sX.data

        if GPAW_NO_C_EXTENSION:
            # We can't compensate...
            # Hopefully one day, we will be able to handle this.
            return out

        # Get compensation charge coefficients
        ccc_asL = AtomArraysLayout(
            [delta_iiL.shape[2] for delta_iiL in density.delta_aiiL],
            atomdist=self.atom_layout.atomdist,
            xp=np).empty(self.ncomponents)
        # XXX: np -> self.xp once the GPU LFC kernels work

        if not isinstance(D_asii_data, AtomArrays):
            D_asii = AtomArrays(layout=self.atom_layout,
                                dims=self.ncomponents,
                                data=D_asii_data)
        else:
            D_asii = D_asii_data

        for a, D_sii in D_asii.items():
            Q_sL = self.xp.einsum('sij, ijL -> sL',
                                  D_sii.real,
                                  density.delta_aiiL[a])
            if add_delta0:
                # Not important for mixing, and should never be done
                # for residuals.
                Q_sL[0][0] += density.delta0_a[a]
            # XXX: Needed since we do compensation on cpu
            ccc_asL[a] = Q_sL if self.xp is np else Q_sL.get()

        # Add compensation charge to output
        if self.xp is not np:
            # XXX: Needed since we do compensation on cpu
            out_cpu = out.to_xp(np)
            self.ghat_aLr.add_to(out_cpu, ccc_asL)
            out.data[:] = self.xp.asarray(out_cpu.data)
        else:
            self.ghat_aLr.add_to(out, ccc_asL)
        return out

    def move(self, relpos_ac: ArrayND, atomdist: AtomDistribution):
        self.relpos_ac = relpos_ac
        self.atomdist = atomdist
        self.atom_layout = AtomArraysLayout(
            [(setup.ni, setup.ni) for setup in self.setups],
            atomdist=atomdist,
            dtype=float if self.ncomponents < 4 else complex,
            xp=self.xp)
        self.histories = []
        self.ghat_aLr = self.setups.create_compensation_charges(
            self.desc,
            relpos_ac,
            atomdist,
            xp=np)
        # XXX: np -> self.xp when GPU LFC works
        self._initialize_history_()

    def reset(self):
        for hist in self.histories:
            hist.reset()

    def __str__(self):
        lines = ['density mixing:',
                 'No mixer',
                 '']
        return '\n  '.join(lines)


class DensityHistory:
    marked_for_delete_oldest = False

    def __init__(self,
                 nmaxold: int,
                 ncomponents: int,
                 desc: DomainType,
                 atom_layout: AtomArraysLayout | None = None,
                 xp=np):
        self.nmaxold = nmaxold
        self.ncomponents = ncomponents
        self.desc = desc
        self.atom_layout = atom_layout
        self.xp = xp
        self._n_hsX = desc.zeros((nmaxold, ncomponents), xp=xp)
        self._D_hasii = atom_layout.zeros((nmaxold, ncomponents)) \
            if atom_layout is not None else None
        self.current_indicies: list[int] = []

    def add_density(self, density: Density) -> None:
        self.add(density.nt_sR, density.D_asii)

    def add(self, n_sX: XArray,
            D_asii: AtomArrays | None = None) -> None:
        if self.nold == self.nmaxold:
            self.delete_oldest()

        self._n_hsX.data[self.next_index] = n_sX.data
        if self._D_hasii is not None:
            assert D_asii is not None
            self._D_hasii.data[self.next_index] = D_asii.data

        self.current_indicies.append(self.next_index)

    def delete_oldest(self) -> None:
        self.current_indicies.pop(0)

    def maybe_delete_oldest(self) -> None:
        if self.marked_for_delete_oldest:
            self.delete_oldest()
            self.marked_for_delete_oldest = False

    def next(self) -> XArray | tuple[XArray, AtomArrays]:
        if self.nold == self.nmaxold:
            self.delete_oldest()

        if self._D_hasii is not None:
            ret = (self._n_hsX[self.next_index],
                   self._D_hasii.data[self.next_index])
        else:
            ret = self._n_hsX[self.next_index]

        self.current_indicies.append(self.next_index)
        return ret

    @property
    def nold(self) -> int:
        return len(self.current_indicies)

    @property
    def next_index(self) -> int:
        for i in range(self.nmaxold):
            if i not in self.current_indicies:
                return i
        raise ValueError('No available index')

    def reset(self):
        self.current_indicies = []

    def dotprod(self, other: Self | XArray) -> ArrayND:
        if isinstance(other, DensityHistory):
            H_hh = self._n_hsX.matrix_elements(other._n_hsX,
                                               symmetric=False)
            out = H_hh.data[self.current_indicies, :][
                :, other.current_indicies]
        else:
            assert len(other.dims) > 1
            out = self._n_hsX.matrix_elements(other).data[
                self.current_indicies, 0]
        return out

    def to_multisecant(self, out: XArray | None = None) -> XArray:
        if out is None:
            out = self.desc.empty((self.nold - 1, self.ncomponents),
                                  xp=self.xp)
        else:
            assert out.dims == (self.nold - 1, self.ncomponents)
        out.data[:] = self._n_hsX[self.current_indicies[:-1]].data
        out.data -= self._n_hsX[self.current_indicies[-1]].data
        return out

    def add_linear_combination(self, alpha_h: list[float],
                               out: XArray,
                               out_asii: XArray | None = None) -> None:
        for ind, alpha in enumerate(alpha_h):
            mod_ind = self.current_indicies[ind]
            out.data += alpha * self._n_hsX.data[mod_ind]
            if out_asii is not None:
                assert self._D_hasii is not None
                out_asii.data += alpha * self._D_hasii.data[mod_ind]

    def __getitem__(self, index: int) -> tuple[XArray, AtomArrays] | XArray:
        index = self.current_indicies[index]

        if self._D_hasii is not None:
            return self._n_hsX[index], self._D_hasii.data[index]
        else:
            return self._n_hsX[index]
