import numpy as np

from gpaw.new.density import Density
from gpaw.mixer.base import DensityHistory, BaseMixer


class PulayMixer(BaseMixer):
    def __init__(self,
                 nmaxold: int,
                 beta: float,
                 *args,
                 **kwargs):
        """Pulay density mixer.

        Parameters
        ----------
        nmaxold:
            Number of previous iterations to include in the Pulay optimization.
        beta:
            Linear mixing parameter (step length).
        *args, **kwargs:
            Additional arguments passed to :class:`~gpaw.mixer.base.BaseMixer`.
        """
        self.nmaxold = nmaxold
        self.beta = beta
        super().__init__(*args, **kwargs)

    def _initialize_history_(self):
        self.nt_hsX = DensityHistory(
            nmaxold=self.nmaxold,
            ncomponents=self.ncomponents,
            desc=self.desc,
            atom_layout=self.atom_layout,
            xp=self.xp)
        self.R_hsX = DensityHistory(
            nmaxold=self.nmaxold,
            ncomponents=self.ncomponents,
            desc=self.desc,
            atom_layout=self.atom_layout,
            xp=self.xp)
        self.Rc_hsX = DensityHistory(
            nmaxold=self.nmaxold,
            ncomponents=self.ncomponents,
            desc=self.desc,
            xp=self.xp)
        self.histories = [self.nt_hsX, self.R_hsX, self.Rc_hsX]

        # DIIS matrix:
        self.H_hh = self.xp.empty(
            (self.nmaxold, self.nmaxold), dtype=float)

    def mix(self, density: Density) -> float:
        # Step 1: Initialize
        nh = self.Rc_hsX.nold
        nold = self.nt_hsX.nold
        if nold == 0:
            self.nt_hsX.add_density(density)
            return np.inf

        nt_sX = density.nt_sR
        D_asii = density.D_asii

        R_sX, R_asii = self.R_hsX.next()
        self.calculate_residual(nt_sX, self.nt_hsX[-1][0], R_sX)
        self.calculate_paw_residual(D_asii.data, self.nt_hsX[-1][1], R_asii)

        Rc_sX = self.Rc_hsX.next()
        self.add_compensation_charge(R_sX, R_asii, density, out=Rc_sX)

        # Step 2: Calculate the DIIS matrix
        MRc_sX = self.metric(Rc_sX)
        MRc_1sX = MRc_sX.new(data=MRc_sX.data[None], dims=(1,) + MRc_sX.dims)
        H_h = self.Rc_hsX.dotprod(MRc_1sX)
        if nh == self.nmaxold:
            self.H_hh[:-1, :-1] = self.H_hh[1:, 1:]
        self.H_hh[:nold, nold - 1] = H_h
        self.H_hh[nold - 1, :nold] = H_h

        try:
            H_hh = self.xp.linalg.inv(self.H_hh[:nold, :nold])
        except (self.xp.linalg.LinAlgError, ZeroDivisionError):
            H_hh = self.xp.zeros((nold, nold))
            H_hh[-1, -1] = 1
        alpha_h = H_hh.sum(1)
        alpha_h /= alpha_h.sum()
        self.world.broadcast(alpha_h, 0)

        # Step 3: Mix the densities
        nt_sX.data[:] = 0
        D_asii.data[:] = 0
        for h, alpha in enumerate(alpha_h):
            nt_sX.data[:] += alpha * self.nt_hsX[h][0].data
            D_asii.data[:] += alpha * self.nt_hsX[h][1]
            nt_sX.data[:] += self.beta * alpha * self.R_hsX[h][0].data
            D_asii.data[:] += self.beta * alpha * self.R_hsX[h][1]

        # Step 4: Update the density history
        self.nt_hsX.add_density(density)

        # Step 5: Return the mixing error
        return self.calculate_charge_sloshing(Rc_sX)

    def __str__(self):
        lines = ['density mixing:',
                 'Pulay-Mixer',
                 f'Linear mixing parameters: {self.beta}',
                 f'Old densities: {self.nmaxold}',
                 self.metric.__str__(),
                 '']
        return '\n  '.join(lines)
