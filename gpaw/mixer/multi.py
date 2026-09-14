import numpy as np

from gpaw.core.atom_arrays import AtomDistribution
from gpaw.mixer.base import BaseMixer
from gpaw.new.density import Density
from gpaw.typing import ArrayND


class MultiMixer(BaseMixer):
    def __init__(self,
                 mixer_a,  # : Mixer from gpaw.dft
                 mixer_b,  # : Mixer from gpaw.dft
                 method: str,
                 ncomponents: int,
                 noncol: bool | None = None,
                 xp=np,
                 **kwargs):
        if method not in ['dif', 'sep']:
            raise ValueError(f'Invalid method: {method}')
        if noncol is None:
            noncol = ncomponents == 4
        self.mixer_a = mixer_a.build(ncomponents=1, noncol=noncol,
                                     xp=xp, **kwargs)
        self.mixer_b = mixer_b.build(ncomponents=ncomponents - 1,
                                     noncol=noncol, xp=xp, **kwargs)
        self.method = method
        super().__init__(metric=self.mixer_a.metric, xp=xp,
                         ncomponents=ncomponents, **kwargs)

    def _initialize_history_(self):
        # Create buffers:
        self.nta_sX = self.desc.empty(1, xp=self.xp)
        self.ntb_sX = self.desc.empty(self.ncomponents - 1, xp=self.xp)
        self.Da_asii = self.atom_layout.empty(1)
        self.Db_asii = self.atom_layout.empty(self.ncomponents - 1)

    def mix(self, density: Density) -> float:
        nt_sX = density.nt_sR
        D_asii = density.D_asii

        nta_sX = self.nta_sX
        ntb_sX = self.ntb_sX
        Da_asii = self.Da_asii
        Db_asii = self.Db_asii

        method = self.method
        if self.method == 'dif' and not self.ncomponents == 2:
            # Mixer "dif" only makes sense for 2 components,
            # as non-colinear already contains the magnetizations
            # in separate components.
            method = 'sep'

        # Prepare the densities:
        match method:
            case 'dif':
                nta_sX.data[:] = nt_sX.data.sum(0)
                Da_asii.data[:] = D_asii.data.sum(0)
                ntb_sX.data[:] = nt_sX.data[0] - nt_sX.data[1]
                Db_asii.data[:] = D_asii.data[0] - D_asii.data[1]
            case 'sep':
                nta_sX.data[:] = nt_sX.data[0]
                Da_asii.data[:] = D_asii.data[0]
                ntb_sX.data[:] = nt_sX.data[1:]
                Db_asii.data[:] = D_asii.data[1:]

        # Mix the densities:
        dnt_a = self.mixer_a._mix(nta_sX, Da_asii, density)
        if self.ncomponents > 1:
            dnt_b = self.mixer_b._mix(ntb_sX, Db_asii, density)
        else:
            dnt_b = 0

        # Put the densities in their place:
        match method:
            case 'sep':
                nt_sX.data[0] = nta_sX.data
                D_asii.data[0] = Da_asii.data
                nt_sX.data[1:] = ntb_sX.data
                D_asii.data[1:] = Db_asii.data
                dnt = dnt_a + dnt_b
            case 'dif':
                nt_sX.data[0] = 0.5 * (nta_sX.data + ntb_sX.data)
                D_asii.data[0] = 0.5 * (Da_asii.data + Db_asii.data)
                nt_sX.data[1] = 0.5 * (nta_sX.data - ntb_sX.data)
                D_asii.data[1] = 0.5 * (Da_asii.data - Db_asii.data)
                dnt = dnt_a + dnt_b

        return dnt

    def _mix(self, nta_sX, Da_asii):
        raise NotImplementedError

    def move(self, relpos_ac: ArrayND, atomdist: AtomDistribution):
        self.mixer_a.move(relpos_ac, atomdist)
        self.mixer_b.move(relpos_ac, atomdist)

    def reset(self):
        self.mixer_a.reset()
        self.mixer_b.reset()

    def __str__(self):
        return f'''MultiMixer
          Method: {self.method}
          Mixer A:
          {self.mixer_a}
          Mixers B:
          {self.mixer_b if self.ncomponents > 1 else '  N/A'}
        '''
