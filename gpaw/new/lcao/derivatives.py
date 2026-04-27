import numpy as np
from gpaw.new.energies import DFTEnergies


class DerivativesNew:

    def __init__(self,
                 egsolver,
                 shuttle,
                 update_c_ref=True,
                 eps=1.0e-7):

        self.eps = eps

        self.a_vec = []
        for a in egsolver.objfunc.a_vec_u:
            self.a_vec.append(a)

        #  add random a_mat (from tools prob)

        # reference orbitals (does it work correctly?)
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
            # FIX
            analytical_der = egsolver.etdm._calc_energy_and_gradient(shuttle)[1]
        else:
            # Calculate analytical approximation to hessian
            # FIX
            ind_up = etdm.ind_up
            analytical_der = np.hstack([get_approx_analytical_hessian(
                kpt, etdm.dtype,
                ind_up[etdm.kpointval(kpt)]).copy() for kpt in wfs.kpt_u])

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
            for wfs in shuttle['ibzwfs']:
#                u = etdm.kpointval(kpt)
#                for i in range(len(self.a[u])):
                for i, a in enumerate(self.a_vec):
                    a = self.a_vec[i].data

                    self.a_vec[i].data = a + step * self.eps
                    shuttle['a_vec_u'] = self.a_vec
                    fplus = egsolver.etdm._calc_energy_and_gradient(shuttle)[f]
                    
                    self.a_vec[i].data = a - step * self.eps
                    shuttle['a_vec_u'] = self.a_vec
                    fminus = egsolver.etdm._calc_energy_and_gradient(shuttle)[f]

                    derf = apply_central_finite_difference_approx(
                        fplus, fminus, self.eps)

                    if what2calc == 'gradient':
                        numerical_der[i] += step * derf
                    else:
                        numerical_der[row] = construct_real_hessian(derf)

                    row += 1
                    self.a[i] = a

        if what2calc == 'hessian':
            egsolver.matrix_exp = matrix_exp

        return numerical_der
    

def apply_central_finite_difference_approx(fplus, fminus, eps):

    if isinstance(fplus, DFTEnergies) and isinstance(fminus, DFTEnergies):
        assert (len(fplus._energies)) == (len(fminus._energies))
        derf = np.stack([(fplus._energies[k] - fminus._energies[k]) * 0.5 / eps
                         for k in fplus._energies.keys()])
    elif isinstance(fplus, np.ndarray) and isinstance(fminus, np.ndarray):
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