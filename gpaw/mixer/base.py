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
                                                           xp=xp)
        self._initialize_history_()

    def _initialize_history_(self):
        self.density_history = DensityHistory(
            nmaxold=1,
            ncomponents=self.ncomponents,
            desc=self.desc,
            xp=self.xp)
        self.histories = [self.density_history, ]

    def mix(self, density: Density) -> float:
        nt_sX = density.nt_sR
        D_asii = density.D_asii
        ntc_sX = self.add_compensation_charge(nt_sX, D_asii, density,
                                              add_delta0=True)
        if self.density_history.nold == 0:
            self.density_history.add(ntc_sX)
            return np.inf

        Rc_sX = self.calculate_residual(ntc_sX, self.density_history[-1],
                                        self.density_history[-1])
        dNt = self.calculate_charge_sloshing(Rc_sX)
        self.density_history.add(ntc_sX)
        return dNt

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

        if self.xp is not np or GPAW_NO_C_EXTENSION:
            # We can't compensate...
            # Hopefully one day, we will be able to handle this.
            return out

        # Get compensation charge coefficients
        ccc_asL = AtomArraysLayout(
            [delta_iiL.shape[2] for delta_iiL in density.delta_aiiL],
            atomdist=self.atom_layout.atomdist,
            xp=self.xp).empty(self.ncomponents)

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
                Q_sL[0] += density.delta0_a[a] / self.ncomponents
            ccc_asL[a] = Q_sL

        # Add compensation charge to output
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
            xp=self.xp)
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

    def __getitem__(self, index: int) -> tuple[XArray, AtomArrays] | XArray:
        index = self.current_indicies[index]

        if self._D_hasii is not None:
            return self._n_hsX[index], self._D_hasii.data[index]
        else:
            return self._n_hsX[index]
