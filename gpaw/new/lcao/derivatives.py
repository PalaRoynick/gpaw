import numpy as np


class DerivativesNew:

    def __init__(self,
                 egsolver,
                 ibzwfs,
                 ham,
                 dens,
                 update_c_ref=True):

        self.a_vec = []
        for a in egsolver.etdm.a_u:
            self.a_vec.append(a)

        #  add random a_mat (from tools prob)

        # initialize: the functions shousl already contain everything?

        # reference orbitals (does it work correctly?)
        self.c_ref = egsolver.ref_orbitals

        if update_c_ref:
            egsolver.objfunc.rotate_wfs()


    def get_analytical_derivatives(self,
                                   egsolver, ham, ibzwfs, dens,
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


        if what2calc == 'gradient':
            # calculate analytical gradient
            analytical_der = egsolver.etdm._calc_energy_and_gradient()[1]
        else:
            # Calculate analytical approximation to hessian
            ind_up = etdm.ind_up
            analytical_der = np.hstack([get_approx_analytical_hessian(
                kpt, etdm.dtype,
                ind_up[etdm.kpointval(kpt)]).copy() for kpt in wfs.kpt_u])

        return analytical_der
    
        
    def get_numerical_derivatives(self, 
                                       egsolver,
                                       ham,
                                       ibzwfs, dens, what2calc='gradient'):
        
        """
        Get numerical derivatives (ONLY LCAO CASE)
        """
        
        # total dimensionality if matrices are real
        dim = sum([len(a) for a in self.a_vec])
        steps = [1.0, 1.0j] if ibzwfs.dtype == complex else [1.0]
        use_energy_or_gradient = {'gradient': 0, 'hessian': 1}

        matrix_exp = egsolver.matrix_exp

        if what2calc == 'gradient':
            numerical_der = [np.zeros_like(a) for a in self.a_vec]
        else:
            numerical_der = np.zeros(shape=(len(steps) * dim,
                                            len(steps) * dim))
            # have to use exact gradient when Hessian is calculated
            egsolver.matrix_exp = 'egdecomp'

        row = 0
        f = use_energy_or_gradient[what2calc]

        for step in steps:
            for wfs in ibzwfs:
                u = etdm.kpointval(kpt)
                for i in range(len(self.a[u])):
                    a = self.a[u][i]

                    self.a[u][i] = a + step * self.eps
                    fplus = etdm.get_energy_and_gradients(
                        self.a, ham, wfs, dens, self.c_ref)[f]

                    self.a[u][i] = a - step * self.eps
                    fminus = etdm.get_energy_and_gradients(
                        self.a, ham, wfs, dens, self.c_ref)[f]

                    derf = apply_central_finite_difference_approx(
                        fplus, fminus, self.eps)

                    if what2calc == 'gradient':
                        numerical_der[u][i] += step * derf
                    else:
                        numerical_der[row] = construct_real_hessian(derf)

                    row += 1
                    self.a[u][i] = a

        if what2calc == 'hessian':
            etdm.matrix_exp = matrix_exp

        return numerical_der
    

def apply_central_finite_difference_approx()
    
def construct_real_hessian()

