from gpaw.dft import Eigensolver
from gpaw.new.pwfd.dir_opt import DirOptPWFD
from gpaw.new.lcao.dir_opt import DirOptLCAO


class DirectOptimization(Eigensolver):
    name = 'etdm-fdpw'

    def __init__(self,
                 converge_unocc: bool = False,
                 **kwargs):
        if kwargs:
            raise NotImplementedError
        self.converge_unocc = converge_unocc

    def todict(self):
        return {'converge_unocc': self.converge_unocc}

    def build(self,
              nbands,
              wf_desc,
              band_comm,
              hamiltonian,
              converge_bands,
              setups,
              atoms):
        return DirOptPWFD(
            converge_unocc=self.converge_unocc,
            hamiltonian=hamiltonian)


class DirectOptimizationLcao(Eigensolver):
    name = 'etdm-lcao'

    def __init__(self,
                 representation='sparse',
                 matrix_exp='egdecomp',
                 converge_unocc: bool = False,
                 ):
        
        self.converge_unocc = converge_unocc
        self.representation = representation
        self.matrix_exp = matrix_exp

    def todict(self):
        return {'converge_unocc': self.converge_unocc}

    def build_lcao(self, basis, relpos_ac, cell_cv, symmetries, hamiltonian):
        
        return DirOptLCAO(
            basis,
            representation=self.representation,
            matrix_exp=self.matrix_exp,)
