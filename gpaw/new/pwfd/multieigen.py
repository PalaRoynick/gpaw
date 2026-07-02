from gpaw.new.pwfd import PWFDEigensolver

class MultiEigensolver(PWFDEigensolver):
    active_solver = None

    def __init__(self,
                 *,
                 wf_grid,
                 hamiltonian,
                 convergence: dict,
                 nbands: int,
                 domain_band_comm,
                 solvers: dict,
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
            self.solvers[condition] = solver.build(
                wf_grid=wf_grid,
                hamiltonian=hamiltonian,
                convergence=convergence,
                nbands=nbands,
                domain_band_comm=domain_band_comm,
                blocksize=blocksize,
                max_buffer_mem=max_buffer_mem,
                scalapack_parameters=scalapack_parameters
            )
            self.min_work_buffer = max(self.min_work_buffer,
                                       self.solvers[condition].min_work_buffer)
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
        for condition, solver in self.solvers.items():
            self.active_solver = solver
            if b_error > condition:  # XXX
                break
        self.active_solver.prestep(*args, **kwargs)

    def bandstep(self, *args, **kwargs):
        self.active_solver.bandstep(*args, **kwargs)

    def poststep(self, *args, **kwargs):
        self.active_solver.poststep(*args, **kwargs)
