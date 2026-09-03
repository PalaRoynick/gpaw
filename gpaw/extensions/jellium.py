from __future__ import annotations

from gpaw.core import PWArray, PWDesc, UGArray
from gpaw.core.domain import Domain
from gpaw.dft import ExtensionInput
from gpaw.extensions import Extension
from gpaw.new.builder import DFTComponentsBuilder


class Jellium(ExtensionInput):
    name = 'jellium'

    def __init__(self,
                 charge: float):
        self.charge = charge

    def todict(self):
        return {'charge': self.charge}

    def update_mask(self, mask_r) -> None:
        mask_r.data[:] = 1.0

    def build(self, builder: DFTComponentsBuilder):
        mask_r = builder.fine_grid.zeros()
        self.update_mask(mask_r)
        # PW-mode needs this one:
        # pw = builder.electrostatic_potential_desc
        pw = builder.interpolation_desc
        return JelliumExtension(mask_r, charge=self.charge, pw=pw)


class JelliumExtension(Extension):
    def __init__(self,
                 mask_r: UGArray,
                 *,
                 charge: float,
                 pw: Domain):
        self.charge = charge
        self.pw = pw
        self.set_mask(mask_r)

    def set_mask(self, mask_r: UGArray) -> None:
        """Normalize and store the shape of the jellium region.

        Can be called again later if the region has to follow the atoms.
        """
        norm = mask_r.integrate()
        if norm == 0.0:
            raise ValueError('The jellium region contains no grid points.')
        self.mask_r = mask_r
        mask_r.data *= 1.0 / norm
        self.mask_g = None
        if isinstance(self.pw, PWDesc):
            mask0_r = mask_r.gather()
            if mask0_r is not None:
                self.mask_g = mask0_r.fft(pw=self.pw.new(comm=None))

    def update1(self, nt_r: UGArray) -> None:
        nt_r.data -= self.mask_r.data * self.charge

    def update1pw(self, nt_g: PWArray | None) -> None:
        if nt_g is None:
            return  # only rank-0 needs to do anything
        assert self.mask_g is not None
        nt_g.data -= self.mask_g.data * self.charge
