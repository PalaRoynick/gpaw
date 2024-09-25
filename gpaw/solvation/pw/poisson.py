from gpaw.solvation.poisson import WeightedFDPoissonSolver
from gpaw.templogger import logg


class WrappedWeightedFDPoissonSolver(WeightedFDPoissonSolver):
    """
    XXX/cb: new class here, does realspace poisson given reciprocal density. A
    very frankenstein/simple solution...for proof of concept

    eps originally 2e-10, maxiter 1000

    Currently facing a non-convergence issue
    """
    def __init__(self, pd3, nn=3, relax='J', eps=1e-6, maxiter=10000,
                 remove_moment=None, use_charge_center=False):
        self.pd3 = pd3
        WeightedFDPoissonSolver.__init__(self, nn=nn, relax=relax, eps=eps,
                                         maxiter=maxiter,
                                         remove_moment=remove_moment,
                                         use_charge_center=use_charge_center)

    def solve(self, phi, rho, charge=None, maxcharge=1e-6,
              zero_initial_phi=False, timer=None):
        logg("start rhotq ifft")
        logg(rho.rhot_q.shape, header=False)
        rho_r = self.pd3.ifft(rho.rhot_q)
        logg("end rhotq ifft")

        logg("start phi ifft")
        phi_r = self.pd3.ifft(phi)
        logg("end phi ifft")

        ret = WeightedFDPoissonSolver.solve(self, phi_r, rho_r, charge,
                                            maxcharge, zero_initial_phi)

        logg("start phi_r fft")
        phi_p = self.pd3.fft(phi_r)
        logg("end phi_r fft")

        phi[:] = phi_p[:]

        return ret
