import numpy as np
from gpaw.new.energies import DFTEnergies
from gpaw.typing import RNG
from typing import Any
from gpaw.new.etdm.tools import random_a


class DerivativesNew:

    def __init__(self,
                 egsolver,
                 shuttle,
                 update_c_ref=True,
                 random_amat=False,
                 seed= None,
                 eps=1.0e-7):

        self.eps = eps

        self.a_vec = []
        self.ind_up = []
        for i, a in enumerate(egsolver.objfunc.a_vec_u):
            self.a_vec.append(a)
            self.ind_up.append(egsolver.objfunc.a_vec_u[i].ind_up)

        if random_amat:
            for i, a in enumerate(egsolver.objfunc.a_vec_u):
                shape = a.data.shape
                self.a_vec[i].data = random_a(shape, shuttle['ibzwfs'].dtype,
                                              seed)


        # reference orbitals 
        self.c_ref = egsolver.ref_orbitals

        # FIX
        if update_c_ref:  # always true?
            egsolver.objfunc.rotate_wfs(self.a_vec,
                                        shuttle['ibzwfs'], self.c_ref)

    def get_analytical_derivatives(self,
                                   egsolver, shuttle,
                                   what2calc='gradient'):
        
        """
        Calculate analytical gradient or approximation to the Hessian
        with respect to the elements of a skew-Hermitian matrix

        ONLY LCAO CASE

        egsolver: eigensolver
        """
        # update_objfunc
        # objfunc_etdm._calc_energy_and_gradients

        assert what2calc in ['gradient', 'hessian']

        # this needs to be updated to make the following functions work
        # take chance to remove this function and pass arguments better
        shuttle['ref_orbitals'] = self.c_ref
        shuttle['a_vec_u'] = self.a_vec

        if what2calc == 'gradient':
            # calculate analytical gradient
            analytical_der = egsolver.etdm._calc_energy_and_gradient(shuttle)[1]
        else:
            # Calculate analytical approximation to hessian
            # FIX
            ibzwfs = shuttle['ibzwfs']
            analytical_der = np.hstack([egsolver.get_approx_analytical_hessian(
                wfs, wfs.dtype,
                self.ind_up) for wfs in ibzwfs])
                
            analytical_der = construct_real_hessian(analytical_der)
            analytical_der = np.diag(analytical_der)

        return analytical_der
            
    def get_numerical_derivatives(self, egsolver, shuttle,
                                  what2calc='gradient'):
        
        """
        Get numerical derivatives (ONLY LCAO CASE)
        """
        
        # total dimensionality if matrices are real
        dim = sum([len(a.data) for a in self.a_vec])
        steps = [1.0, 1.0j] if shuttle['ibzwfs'].dtype == complex else [1.0]
        use_energy_or_gradient = {'gradient': 0, 'hessian': 1}

        matrix_exp = egsolver.matrix_exp

        if what2calc == 'gradient':
            numerical_der = [np.zeros_like(a.data) for a in self.a_vec]
        else:
            numerical_der = np.zeros(shape=(len(steps) * dim,
                                            len(steps) * dim))
            # have to use exact gradient when Hessian is calculated
            egsolver.matrix_exp = 'egdecomp'

        shuttle['ref_orbitals'] = self.c_ref

        row = 0
        f = use_energy_or_gradient[what2calc]

        for step in steps:
            for u in range(len(self.a_vec)):
                for i in range(len(self.a_vec[u].data)):
                    a = self.a_vec[u].data[i]

                    a_vec_cop = self.a_vec[u].data.copy()
                    self.a_vec[u]._rotation_mat = None
                    
                    self.a_vec[u].data[i] = a + step * self.eps
                    shuttle['a_vec_u'] = self.a_vec
                    fplus = egsolver.objfunc._calc_energy_and_gradient(shuttle.copy())[f]

                    self.a_vec[u].data = a_vec_cop

                    self.a_vec[u].data[i] = a - step * self.eps
                    shuttle['a_vec_u'] = self.a_vec
                    fminus = egsolver.objfunc._calc_energy_and_gradient(shuttle.copy())[f]

                    derf = apply_central_finite_difference_approx(
                        fplus, fminus, self.eps, what2calc)

                    if what2calc == 'gradient':
                        numerical_der[u][i] += step * derf
                    else:
                        numerical_der[row] = construct_real_hessian(derf)

                    row += 1
                    self.a_vec[u].data[i] = a

        if what2calc == 'hessian':
            egsolver.matrix_exp = matrix_exp

        return numerical_der
    

def apply_central_finite_difference_approx(fplus, fminus, eps, what2calc):

    if what2calc == 'gradient':
        fplus = fplus.total_free
        fminus = fminus.total_free

    #if isinstance(fplus, DFTEnergies) and isinstance(fminus, DFTEnergies):
    #    assert (len(fplus._energies)) == (len(fminus._energies))
    #    derf = np.stack([(fplus[k] - fminus[k]) * 0.5 / eps
    #                     for k in fplus._energies.keys()])
        
    if isinstance(fplus, np.ndarray) and isinstance(fminus, np.ndarray):
        derf = (fplus - fminus) * 0.5 / eps
    elif isinstance(fplus, float) and isinstance(fminus, float):
        derf = (fplus - fminus) * 0.5 / eps
    else:
        raise ValueError()

    return derf
    

def construct_real_hessian(hess):

    if hess.dtype == complex:
        hess_real = np.hstack((np.real(hess), np.imag(hess)))
    else:
        hess_real = hess

    return hess_real
