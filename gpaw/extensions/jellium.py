import numpy as np
from ase.units import Ha
from gpaw.core import PWArray, PWDesc, UGArray
from gpaw.core.domain import Domain
from gpaw.extensions import Extension
from gpaw.new.builder import DFTComponentsBuilder
from gpaw.new.ibzwfs import IBZWaveFunctions
from gpaw.dft import ExtensionInput


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
        self.mask_r = mask_r
        mask_r.data *= 1.0 / mask_r.integrate()
        self.mask_g = None
        if isinstance(pw, PWDesc):
            mask_r = mask_r.gather()
            if mask_r is not None:
                self.mask_g = mask_r.fft(pw=pw.new(comm=None))

    def update1(self, nt_r: UGArray) -> None:
        nt_r.data -= self.mask_r.data * self.charge

    def update1pw(self, nt_g: PWArray | None) -> None:
        if nt_g is None:
            return  # only rank-0 needs to do anything
        assert self.mask_g is not None
        nt_g.data -= self.mask_g.data * self.charge


class FixedPotentialJelliumExtension(JelliumExtension):
    def __init__(self,
                 mask_r: UGArray,
                 *,
                 pw: Domain,
                 workfunction_target: float,  # eV
                 excess_electrons_guess=0.0,
                 tolerance: float = 0.001):  # eV
        """Adjust jellium charge to get the desired Fermi-level."""
        super().__init__(mask_r, charge=excess_electrons_guess or 0.0, pw=pw)
        self.workfunction_target = workfunction_target / Ha
        self.tolerance = tolerance / Ha
        # (Charge, Fermi-level) history:
        self.history: list[tuple[float, float]] = []

    def post_scf_convergence(self,
                             ibzwfs: IBZWaveFunctions,
                             nelectrons: float,
                             occ_calc,
                             mixer,
                             log) -> bool:
        fl1 = ibzwfs.fermi_level
        log(f'charge: {self.charge:.6f} |e|, Fermi-level: {fl1 * Ha:.3f} eV')
        fl = -self.workfunction_target
        if abs(fl1 - fl) <= self.tolerance:
            return True
        self.history.append((self.charge, fl1))
        if len(self.history) == 1:
            area = abs(np.linalg.det(self.mask_r.desc.cell_cv[:2, :2]))
            dc = -(fl1 - fl) * area * 0.02
        else:
            (c2, fl2), (c1, fl1) = self.history[-2:]
            c = c2 + (fl - fl2) / (fl1 - fl2) * (c1 - c2)
            dc = c - c1
            if abs(dc) > abs(c2 - c1):
                dc *= abs((c2 - c1) / dc)
        self.charge += dc
        nelectrons += dc
        ibzwfs.calculate_occs(occ_calc, nelectrons)
        mixer.reset()
        return False
