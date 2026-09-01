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
                 xp=np,
                 **kwargs):
        if method not in ['dif', 'sep']:
            raise ValueError(f'Invalid method: {method}')
        self.mixer_a = mixer_a.build(ncomponents=1, xp=xp, **kwargs)
        self.mixers_b = [mixer_b.build(ncomponents=1, xp=xp, **kwargs)
                         for comp in range(ncomponents - 1)]
        self.method = method
        super().__init__(metric=self.mixer_a.metric, xp=xp,
                         ncomponents=ncomponents, **kwargs)

    def _initialize_history_(self):
        # Create buffers:
        self.nta_sX = self.desc.empty(1, xp=self.xp)
        self.ntb_msX = [self.desc.empty(1, xp=self.xp) for comp in
                        range(self.ncomponents - 1)]
        self.Da_asii = self.atom_layout.empty(1)
        self.Db_masii = [self.atom_layout.empty(1) for comp in
                         range(self.ncomponents - 1)]

    def mix(self, density: Density) -> float:
        nt_sX = density.nt_sR
        D_asii = density.D_asii

        nta_sX = self.nta_sX
        ntb_msX = self.ntb_msX
        Da_asii = self.Da_asii
        Db_masii = self.Db_masii

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
                ntb_msX[0].data[:] = nt_sX.data[0] - nt_sX.data[1]
                Db_masii[0].data[:] = D_asii.data[0] - D_asii.data[1]
            case 'sep':
                nta_sX.data[:] = nt_sX.data[0]
                Db_masii[0].data[:] = D_asii.data[0]
                for m, (ntb_sX, Db_asii) in enumerate(zip(ntb_msX, Db_masii)):
                    ntb_sX.data[:] = nt_sX.data[m + 1]
                    Db_asii.data[:] = D_asii.data[m + 1]

        # Mix the densities:
        dnt_a = self.mixer_a._mix(nta_sX, Da_asii, density)
        dntb_m = [mixer._mix(ntb_sX, Db_asii, density)
                  for mixer, ntb_sX, Db_asii
                  in zip(self.mixers_b, ntb_msX, Db_masii)]

        # Put the densities in their place:
        match method:
            case 'sep':
                nt_sX.data[0] = nta_sX.data[0]
                D_asii.data[0] = Da_asii.data[0]
                for m, (ntb_sX, Db_asii) in enumerate(zip(ntb_msX, Db_masii)):
                    nt_sX.data[m + 1] = ntb_sX.data[0]
                    D_asii.data[m + 1] = Db_asii.data[0]
                dnt = dnt_a + sum(dntb_m)
            case 'dif':
                ntb_sX = ntb_msX[0]
                Db_asii = Db_masii[0]
                nt_sX.data[0] = 0.5 * (nta_sX.data[0] + ntb_sX.data[0])
                D_asii.data[0] = 0.5 * (Da_asii.data[0] + Db_asii.data[0])
                nt_sX.data[1] = 0.5 * (nta_sX.data[0] - ntb_sX.data[0])
                D_asii.data[1] = 0.5 * (Da_asii.data[0] - Db_asii.data[0])
                dnt = dnt_a

        return dnt

    def _mix(self, nta_sX, Da_asii):
        raise NotImplementedError

    def move(self, relpos_ac: ArrayND, atomdist: AtomDistribution):
        self.mixer_a.move(relpos_ac, atomdist)
        for mixer_b in self.mixers_b:
            mixer_b.move(relpos_ac, atomdist)

    def reset(self):
        self.mixer_a.reset()
        for mixer_b in self.mixers_b:
            mixer_b.reset()

    def __str__(self):
        return f'''MultiMixer
          Method: {self.method}
          Mixer A:
          {self.mixer_a}
          Mixers B:
          {self.mixers_b[0] if self.ncomponents > 1 else '  N/A'}
        '''
