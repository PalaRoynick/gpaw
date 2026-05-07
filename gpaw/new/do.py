from gpaw.dft import Eigensolver, LegacyGPAWError
from gpaw.new.pwfd.diropt import DirOptPWFD
from gpaw.new.lcao.dir_opt import DirOptLCAO


class DirectOptimization(Eigensolver):
    name = 'etdm-fdpw'

    def __init__(self,
                 converge_unocc: bool = False,
                 **kwargs):
        if kwargs:
            raise LegacyGPAWError
        self.converge_unocc = converge_unocc

    def todict(self):
        return {'converge_unocc': self.converge_unocc}

    def build(self,
              nbands,
              wf_desc,
              band_comm,
              domain_band_comm,
              scalapack_parameters,
              hamiltonian,
              convergence,
              setups,
              atoms):
        return DirOptPWFD(
            converge_unocc=self.converge_unocc,
            converge_bands=converge_bands,
            hamiltonian=hamiltonian)


class DirectOptimizationLcao(Eigensolver):
    name = 'etdm-lcao'

    def __init__(self,
                 representation='sparse',
                 matrix_exp='egdecomp',
                 searchdir_algo='l-bfgs',
                 linesearch_algo={'name': 'max-step',
                                  'max_step': 0.25},
                 converge_unocc: bool = False,
                 ):
        
        self.converge_unocc = converge_unocc
        self.representation = representation
        self.matrix_exp = matrix_exp
        self.searchdir_algo = searchdir_algo
        self.linesearch_algo = linesearch_algo

    def todict(self):
        return {'converge_unocc': self.converge_unocc}

    def build_lcao(self, basis, relpos_ac,
                   cell_cv, symmetries, hamiltonian,
                   create_occupation_number_calculator):
        occ_calc = create_occupation_number_calculator()

        return DirOptLCAO(
            basis,
            representation=self.representation,
            matrix_exp=self.matrix_exp,
            searchdir_algo_name=self.searchdir_algo,
            occ_calc=occ_calc,
            linesearch_algo=self.linesearch_algo)
