from __future__ import annotations

import numpy as np

from gpaw.core import UGArray, UGDesc
from gpaw.new.builder import create_uniform_grid
from gpaw.new.fd.hamiltonian import FDHamiltonian
from gpaw.new.fd.pot_calc import FDPotentialCalculator
from gpaw.new.gpw import as_double_precision
from gpaw.new.pot_calc import DistributedPotentialCalculator
from gpaw.new.pwfd.builder import PWFDDFTComponentsBuilder
from gpaw.new.xc import create_functional


class FDDFTComponentsBuilder(PWFDDFTComponentsBuilder):
    def __init__(self,
                 atoms,
                 params,
                 *,
                 comm,
                 log):
        self.interpolation_stencil_range = params.interpolation or 3
        super().__init__(atoms,
                         params,
                         comm=comm, log=log)
        assert not self.soc
        assert self.qspiral_v is None

        self._nct_aR = None
        self._tauct_aR = None

        self.electrostatic_potential_desc = self.fine_grid
        self.interpolation_desc = self.fine_grid

    def create_uniform_grids(self):
        grid = create_uniform_grid(
            self.mode,
            self.params.gpts,
            self.atoms.cell,
            self.atoms.pbc,
            self.ibz.symmetries,
            h=self.params.h,
            interpolation=self.interpolation_stencil_range,
            comm=self.communicators['d'])
        fine_grid = grid.new(size=grid.size_c * 2)
        # decomposition=[2 * d for d in grid.decomposition]
        return grid, fine_grid

    def create_wf_description(self) -> UGDesc:
        return self.grid.new(dtype=self.dtype)

    def get_pseudo_core_densities(self):
        if self._nct_aR is None:
            self._nct_aR = self.setups.create_pseudo_core_densities(
                self.grid, self.relpos_ac, atomdist=self.atomdist, xp=self.xp)
        return self._nct_aR

    def get_pseudo_core_ked(self):
        if self._tauct_aR is None:
            self._tauct_aR = self.setups.create_pseudo_core_ked(
                self.grid, self.relpos_ac, atomdist=self.atomdist)
        return self._tauct_aR

    def create_poisson_solver(self, extensions, fine_grid=None):
        fine_grid = fine_grid or self.fine_grid
        from gpaw.poisson import PoissonSolver as make_poisson_solver
        try:
            solver = super().create_poisson_solver(extensions)
        except NotImplementedError:
            ps = self.params.poissonsolver
            if hasattr(ps, 'params'):
                psparams = ps.params
            else:
                psparams = {'name': ps}
            solver = make_poisson_solver(
                **psparams,
                xp=self.xp).build(fine_grid, self.xp)
        return solver

    def _create_potential_calculator(self, grid, fine_grid, xc, atomdist,
                                     extensions):
        poisson_solver = self.create_poisson_solver(extensions, fine_grid)
        return FDPotentialCalculator(
            grid,
            fine_grid,
            self.setups,
            xc,
            poisson_solver,
            relpos_ac=self.relpos_ac,
            atomdist=atomdist,
            interpolation_stencil_range=self.interpolation_stencil_range,
            extensions=extensions,
            xp=self.xp)

    def create_potential_calculator(self):
        extensions = self.get_extensions()
        normal_calc = self._create_potential_calculator(
            self.grid,
            self.fine_grid,
            self.xc,
            self.atomdist,
            extensions)
        if not self.distribute_potential:
            return normal_calc

        distributed_xc = create_functional(
            self.xcfunc, self.distributed_fine_grid, self.xp)
        distributed_calc = self._create_potential_calculator(
            self.distributed_grid,
            self.distributed_fine_grid,
            distributed_xc,
            self.distributed_atomdist,
            [])
        distributed_calc.density_atomdist = self.distributed_atomdist
        return DistributedPotentialCalculator(
            normal_calc,
            distributed_calc,
            grid=self.grid,
            interpolation_desc=self.interpolation_desc,
            electrostatic_potential_desc=self.electrostatic_potential_desc,
            atomdist=self.atomdist,
            distributed_grid=self.distributed_grid,
            distributed_interpolation_desc=self.distributed_grid,
            distributed_electrostatic_potential_desc=(
                self.distributed_fine_grid),
            distributed_atomdist=self.distributed_atomdist,
            comm1=self.communicators['D'],
            comm2=self.communicators['D'])

    def create_hamiltonian_operator(self):
        return FDHamiltonian(self.wf_desc,
                             kin_stencil=self.params.mode.nn,
                             xp=self.xp)

    def read_ibz_wave_functions(self, reader):
        ibzwfs = super().read_ibz_wave_functions(reader)

        if 'coefficients' in reader.wave_functions:
            name = 'coefficients'
        elif 'values' in reader.wave_functions:
            name = 'values'  # old name
        else:
            return ibzwfs

        c = reader.bohr**1.5
        if reader.version < 0:
            c = 1  # old gpw file
        singlep = reader.get('precision', 'double') == 'single'

        for wfs in ibzwfs:
            grid = self.wf_desc.new(kpt=wfs.kpt_c, dtype=self.dtype)
            index = (wfs.spin, wfs.k)
            data = reader.wave_functions.proxy(name, *index)
            data.scale = c
            if self.communicators['w'].size == 1 and not singlep:
                # Cast to the right dtype
                wfs.psit_nX = UGArray(grid, self.nbands,
                                      data=np.array(data, dtype=grid.dtype))
            else:
                band_comm = self.communicators['b']
                wfs.psit_nX = UGArray(
                    grid, self.nbands,
                    comm=band_comm)
                if grid.comm.rank == 0:
                    mynbands = (self.nbands +
                                band_comm.size - 1) // band_comm.size
                    n1 = min(band_comm.rank * mynbands, self.nbands)
                    n2 = min((band_comm.rank + 1) * mynbands, self.nbands)
                    assert wfs.psit_nX.mydims[0] == n2 - n1
                    data = data[n1:n2]  # read from file

                if singlep:
                    wfs.psit_nX.scatter_from(as_double_precision(data))
                else:
                    wfs.psit_nX.scatter_from(data)

        return ibzwfs
