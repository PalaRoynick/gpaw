from ase.units import Ha

from gpaw.new.pwfd.eigensolver import PWFDEigensolver

class MultiEigensolver(PWFDEigensolver):
    active_solver = None

    def __init__(self,
                 *,
                 wf_grid,
                 hamiltonian,
                 convergence: dict,
                 nbands: int,
                 domain_band_comm,
                 band_comm,
                 solvers: dict,
                 setups,
                 atoms,
                 blocksize: int = 10,
                 max_buffer_mem: int | None = 200 * 1024 ** 2,
                 scalapack_parameters: tuple[int, int, int] | None = None):
        super().__init__(wf_grid=wf_grid,
                         hamiltonian=hamiltonian,
                         convergence=convergence,
                         nbands=nbands,
                         domain_band_comm=domain_band_comm,
                         blocksize=blocksize,
                         max_buffer_mem=max_buffer_mem,
                         scalapack_parameters=scalapack_parameters)
        self.solvers = {}
        for condition, solver in solvers.items():
            condition = float(condition)
            solver.max_buffer_mem = self.max_buffer_mem
            self.solvers[condition] = solver.build(
                wf_desc=wf_grid,
                hamiltonian=hamiltonian,
                convergence=convergence,
                nbands=nbands,
                band_comm=band_comm,
                domain_band_comm=domain_band_comm,
                scalapack_parameters=scalapack_parameters,
                setups=setups,
                atoms=atoms
            )
            self.solvers[condition].blocksize = blocksize
            self.min_work_array = max(self.min_work_array,
                                       self.solvers[condition].min_work_array)
            self.min_buffer_array = max(self.min_buffer_array,
                                        self.solvers[condition].min_buffer_array)


    def _initialize(self, ibzwfs):
        super()._initialize(ibzwfs)
        for solver in self.solvers.values():
            # TODO: dont actually _initialize child buffers and delete them.
            solver._initialize(ibzwfs)
            solver.work_arrays = self.work_arrays
            solver.data_buffers = self.data_buffers

    def prestep(self, *args, b_error, i, **kwargs):
        for condition in sorted(self.solvers.keys()):
            self.active_solver = self.solvers[condition]
            if b_error * Ha**2 < condition:  # XXX
                break
        self.niter = self.active_solver.niter
        self.active_solver.prestep(*args, b_error=b_error, i=i, **kwargs)

    def bandstep(self, *args, **kwargs):
        self.active_solver.bandstep(*args, **kwargs)

    def poststep(self, *args, **kwargs):
        self.active_solver.poststep(*args, **kwargs)
