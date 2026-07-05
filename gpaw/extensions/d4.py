import numpy as np
from ase.calculators.calculator import PropertyNotImplementedError
from ase.units import Bohr, Ha

from gpaw.dft import ExtensionInput
from gpaw.extensions import Extension
from gpaw.mpi import broadcast_exception, broadcast_float
from gpaw.new.builder import DFTComponentsBuilder


class D4(ExtensionInput):
    name = 'd4'

    def __init__(self, *, xc, **kwargs):
        if 'name' in kwargs:
            raise TypeError('D4 does not accept reserved keyword name=...')
        self.xc = xc
        self.kwargs = kwargs

    def todict(self) -> dict:
        return {'xc': self.xc, **self.kwargs}

    def build(self, builder: DFTComponentsBuilder):
        from dftd4.ase import DFTD4

        atoms = builder.atoms.copy()
        world = builder.communicators['w']
        log = builder.log

        class D4Extension(Extension):
            name = 'd4'

            def __init__(self):
                super().__init__()
                self.stress_vv = np.zeros((3, 3)) * np.nan
                self.F_av = np.zeros_like(atoms.positions) * np.nan
                self.E = np.nan
                self._calculate(atoms)

            def _calculate(_self, atoms):
                # Like D3, treat isolated atoms as a trivial zero-correction
                # case to avoid unnecessary API calls and numerical noise.
                if len(atoms) == 1 and not atoms.pbc.any():
                    _self.stress_vv = np.zeros((3, 3)) * np.nan
                    _self.F_av = np.zeros_like(atoms.positions)
                    _self.E = 0.0
                    return

                assert atoms.calc is None
                # DFTD4 uses an in-process Python/C API, so unlike D3 there is
                # no filesystem race to isolate. We still execute it on rank 0
                # and broadcast the result to keep extension behavior aligned.
                with broadcast_exception(world):
                    if world.rank == 0:
                        _self.calculate_single_core()
                _self.E = broadcast_float(_self.E, world)
                world.broadcast(_self.F_av, 0)
                world.broadcast(_self.stress_vv, 0)

            def calculate_single_core(_self):
                log(f'Evaluating D4 corrections for xc={self.xc!r}')
                atoms.calc = DFTD4(method=self.xc, **self.kwargs)
                _self.F_av = atoms.get_forces() / Ha * Bohr
                try:
                    # Copy needed because array is not guaranteed contiguous.
                    _self.stress_vv = (
                        atoms.get_stress(voigt=False).copy() / Ha * Bohr**3
                    )
                except PropertyNotImplementedError:
                    _self.stress_vv = np.zeros((3, 3)) * np.nan
                _self.E = atoms.get_potential_energy() / Ha
                atoms.calc = None

            def get_energy_contributions(_self) -> dict[str, float]:
                return {f'D4 (xc={self.xc})': _self.E}

            def get_energy(self) -> float:
                return self.E * Ha

            def force_contribution(self, nt_r, vHt_r):
                return self.F_av

            def stress_contribution(self):
                if np.isnan(self.stress_vv).all():
                    raise PropertyNotImplementedError
                return self.stress_vv

            def move_atoms(self, relpos_ac) -> None:
                atoms.set_scaled_positions(relpos_ac)
                self._calculate(atoms)

        return D4Extension()
