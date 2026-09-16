import os
import uuid
from pathlib import Path

import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.units import Bohr, Ha

from gpaw.dft import ExtensionInput
from gpaw.extensions import Extension
from gpaw.mpi import broadcast_exception, broadcast_float, serial_comm
from gpaw.new.builder import DFTComponentsBuilder


class DNExtension(Extension):

    def __init__(self, name, atoms, xc, world, log, **kwargs):
        super().__init__()
        self.name = name
        self.atoms = atoms
        self.xc = xc
        self.world = world
        self.log = log
        self.kwargs = kwargs

        self.stress_vv = np.zeros((3, 3)) * np.nan
        self.F_av = np.zeros_like(atoms.positions) * np.nan
        self.E = np.nan
        self._calculate(self.atoms)

    def _calculate(self, atoms):
        # Circumvent a DFTD3 bug for an isolated atom ASE #1672
        if len(self.atoms) == 1 and not self.atoms.pbc.any():
            self.stress_vv = np.zeros((3, 3)) * np.nan
            self.F_av = np.zeros_like(self.atoms.positions)
            self.E = 0.0
            return

        cwd = Path.cwd()
        assert self.atoms.calc is None
        # Call DFTD3 only with single core due to #1671
        with broadcast_exception(self.world):
            if self.world.rank == 0:
                if self.name == "d3":
                    try:
                        self.calculate_single_core_d3()
                    finally:
                        os.chdir(cwd)
                elif self.name == "d4":
                    self.calculate_single_core_d4()

        self.E = broadcast_float(self.E, self.world)
        self.world.broadcast(self.F_av, 0)
        self.world.broadcast(self.stress_vv, 0)

    def calculate_single_core_d3(self):
        """Single core method to calculate D3 forces and stresses"""
        from ase.calculators.dftd3 import PureDFTD3

        label = uuid.uuid4().hex[:8]
        directory = Path("dftd3-ext-" + label).absolute()
        directory.mkdir()

        # Due to ase #1673, relative folders are not supported
        # neither are absolute folders due to 80 character limit.
        # The only way out, is to chdir to a temporary folder here.
        os.chdir(directory)
        self.log(f"Evaluating D3 corrections at"
            f"temporary folder {directory}")
        self.atoms.calc = PureDFTD3(
            xc=self.xc, directory=".", comm=serial_comm, **self.kwargs
        )

        # XXX params.xc should be taken directly from the calculator.
        # XXX What if this is changed via set?
        self.F_av = self.atoms.get_forces() / Ha * Bohr

        try:
            # Copy needed because array is not c-contigous
            self.stress_vv = (
                self.atoms.get_stress(voigt=False).copy() / Ha * Bohr**3
            )
        except PropertyNotImplementedError:
            self.stress_vv = np.zeros((3, 3)) * np.nan

        self.E = self.atoms.get_potential_energy() / Ha
        try:
            os.unlink("ase_dftd3.out")
            os.unlink("ase_dftd3.POSCAR")
            os.unlink("dftd3_cellgradient")
            os.unlink("dftd3_gradient")
            os.rmdir(directory.absolute())
        except OSError as e:
            self.log("Unable to remove files and folder", e)
        self.atoms.calc = None

    def calculate_single_core_d4(self):
        from dftd4.ase import DFTD4
        self.log(f"Evaluating D4 corrections for xc={self.xc!r}")
        self.atoms.calc = DFTD4(method=self.xc, **self.kwargs)
        self.F_av = self.atoms.get_forces() / Ha * Bohr
        try:
            # Copy needed because array is not guaranteed contiguous.
            self.stress_vv = (
                self.atoms.get_stress(voigt=False).copy() / Ha * Bohr**3
            )
        except PropertyNotImplementedError:
            self.stress_vv = np.zeros((3, 3)) * np.nan
        self.E = self.atoms.get_potential_energy() / Ha
        self.atoms.calc = None

    def get_energy_contributions(self) -> dict[str, float]:
        """Returns the energy contributions from D3/D4 in Hartree"""
        return {f"{self.name.upper()} (xc={self.xc})": self.E}

    def get_energy(self) -> float:
        """Returns the energy contribution from D3/D4 in eV"""
        return self.E * Ha

    def force_contribution(self, nt_r, vHt_r):
        return self.F_av

    def stress_contribution(self):
        if np.isnan(self.stress_vv).all():
            raise PropertyNotImplementedError
        return self.stress_vv

    def move_atoms(self, relpos_ac) -> None:
        self.atoms.set_scaled_positions(relpos_ac)
        self._calculate(self.atoms)

class DN(ExtensionInput):
    def __init__(self, *, xc, **kwargs):
        if 'name' in kwargs and self.name == "d4":
            raise TypeError('D4 does not accept reserved keyword name=...')
        self.xc = xc
        self.kwargs = kwargs

    def todict(self) -> dict:
        return {"xc": self.xc, **self.kwargs}

    def build(self, builder: DFTComponentsBuilder):
        """
        if self.name == "d3":
            from ase.calculators.dftd3 import PureDFTD3
        elif self.name == "d4":
            from dftd4.ase import DFTD4
        else:
            raise TypeError("Dispersion must be d3 or d4")
        """

        atoms = builder.atoms.copy()
        world = builder.communicators["w"]
        log = builder.log

        # Since DFTD3 is filesystem based, and GPAW has no such requirements
        # we need to be absolutely sure that there are no race-conditions
        # in files. label cannot be used, because dftd3 executable still
        # writes gradients to fixed files, thus a unique folder needs to be
        # created.

        return DNExtension(self.name, atoms, self.xc, world, log, **self.kwargs)
