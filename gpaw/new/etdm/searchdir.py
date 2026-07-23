import numpy as np


class LBFGS:
    def __init__(self,
                 p_key=False,
                 *,
                 memory=3):  # set to 5 for FD
        """
        L-BFGS optimizer initialization.

        Parameters
        ----------
        p_key: bool
            If True use 'l-bfgs-p': Limited-memory BFGS with preconditioner
        memory : int
            Number of past steps to store for the L-BFGS approximation.
        """

        # Maximum number of previous steps to store for limited-memory Hessian
        self.memory = memory

        # init empty arrays
        self.reset()

        # Initialized
        self.beta0 = 1.0
        # Key to toggle l-bfgs-p
        self.p_key = p_key

    def reset(self):
        # Counter for number of local iterations performed
        self.local_iter = 0

        # Current search direction
        self.search_dir = None

        # Previous gradient and variable arrays
        self.g_old = None
        self.a_old = None

        # Arrays to store last 'memory' differences in variables and gradients
        # ds[m] ~ change in parameters (search direction)
        # dy[m] ~ change in gradients
        self.ds = ['None'] * self.memory
        self.dy = ['None'] * self.memory

        # Scaling factors for L-BFGS (1 / (y^T * s))
        self.rho = np.zeros(self.memory)

    def update(self, a_cur, g_cur, precond=None):
        """
        Compute the next search direction using L-BFGS two-loop recursion.

        Parameters
        ----------
        a_cur : ndarray
            Current variables (parameters).
        g_cur : ndarray
            Current gradient of objective function.

        Returns
        -------
        search_dir : ndarray
            Updated search direction for the next iteration.
        """

        # Step 1: First iteration initialization
        # print('self.local_iter')
        # print(self.local_iter)
        if self.local_iter == 0:
            self.ds = [0.0 * g_cur for _ in range(self.memory)]
            self.dy = [0.0 * g_cur for _ in range(self.memory)]

            # Store current gradient and variables
            self.g_old = g_cur.copy()
            self.a_old = a_cur.copy()

            # preconditioning and initial search direction
            if precond is not None:
                if self.p_key:
                    g_cur.a_ux = apply_precond(precond, g_cur.a_ux, const=-1)
                    self.search_dir = g_cur
                else:
                    g_cur.a_ux = apply_precond(precond, g_cur.a_ux, const=1)
                    self.search_dir = -g_cur
            else:
                self.search_dir = -g_cur

            # Increment iteration counter
            self.local_iter += 1

            # Return initial search direction
            return self.search_dir

        else:
            # Step 2: Determine memory index for circular storage
            m = self.local_iter % self.memory

            # Store changes in variables (ds) and gradients (dy)
            # For first few iterations, ds = previous search direction

            # FIX: these two statements should be
            # identical for both lcao and pw
            self.ds[m] = self.search_dir.copy()  # use for PW/FD
            # self.ds[m] = a_cur - self.a_old  # use for LCAO
            self.dy[m] = g_cur - self.g_old

            # Compute curvature y^T * s
            dyds = self.dy[m] @ self.ds[m]

            # Compute L-BFGS scaling factor rho = 1 / (y^T * s)
            if abs(dyds) > 1.0e-20:
                self.rho[m] = 1.0 / dyds
            else:
                # Avoid division by zero
                self.rho[m] = 1.0e20

            # Step 3: Stability check
            if self.rho[m] < 0:
                self.__init__(memory=self.memory, p_key=self.p_key) # BUG ALSO CHECK WITH NO SIC
                # self.local_iter = 0
                # self.rho[:] = 0.0
                # for x in self.ds:
                #     x[:] = 0.0
                # for x in self.dy:
                #     x[:] = 0.0
                return self.update(a_cur, g_cur, precond)

            # Step 4: Two-loop recursion to compute search direction
            q = g_cur.copy()

            # First loop: backward over stored vectors
            k = self.memory - 1
            alpha = np.zeros_like(self.rho)

            while k > -1:
                # Circular indexing over memory
                c_ind = (k + m + 1) % self.memory
                k -= 1

                # Dot product s^T * q
                sq = self.ds[c_ind] @ q

                # Scaling factor for this step
                alpha[c_ind] = self.rho[c_ind] * sq

                # Update q
                q += -alpha[c_ind] * self.dy[c_ind]

            # Step 5: Scale by initial Hessian approximation
            yy = self.dy[m] @ self.dy[m]

            # l-bfgs-p:
            if self.p_key:
                self.rho_y = self.rho[m] * yy
                if abs(self.rho_y) > 1.e-20:
                    self.beta0 = 1.0 / self.rho_y
                else:
                    self.beta0 = 1.0e20
                if precond is not None:
                    self.search_dir.a_ux = apply_precond(precond,
                                                         np.array(q.a_ux))
                else:
                    self.search_dir = q * self.beta0
            else:
                # avoid divide by zero
                devis = np.maximum(self.rho[m] * yy, 1.0e-20)
                self.search_dir = q * (1 / devis)


            # Second loop: forward over stored vectors
            for k in range(self.memory):
                if self.local_iter < self.memory:
                    c_ind = k
                else:
                    c_ind = (k + m + 1) % self.memory

                # Compute beta = rho * y^T * r
                yr = self.dy[c_ind] @ self.search_dir
                beta = self.rho[c_ind] * yr

                # Update search direction with alpha and beta corrections
                self.search_dir += self.ds[c_ind] * (alpha[c_ind] - beta)

            # Step 6: Update stored previous gradient and variable
            self.g_old = g_cur.copy()
            self.a_old = a_cur.copy()

            # Step 7: get descent direction
            self.search_dir *= -1

            self.local_iter += 1

            # Return updated search direction
            return self.search_dir


class LSR1:
    def __init__(self,
                 memory=20,
                 method='LSR1',
                 phi=None):
        """
        Initialization of LSR1 algorithm

        Parameters
        ----------

        memory : int
            Number of past steps to store for the L-SR1 approximation.
        method and phi: parameters for possible variants of LSR1,
            not implemented
        """

        # Maximum number of previous steps to store for limited-memory Hessian
        self.memory = memory

        # init empty arrays
        self.reset()

        # possible variant
        self.method = method

        # parameter for linear combination
        self.phi = phi

    def reset(self):

        # Counter for number of local iterations performed
        self.local_iter = 0

        # Current search direction
        self.search_dir = None

        # Previous gradient and variable arrays
        self.g_old = None
        self.a_old = None

        # Arrays to store last 'memory' differences in variables and gradients
        # ds[m] ~ change in parameters (search direction)
        # dy[m] ~ change in gradients
        self.du = ['None'] * self.memory
        self.dj = ['None'] * self.memory
        self.yj_k = np.zeros(self.memory)

        # here or above?
        self.phi_k = np.zeros(shape=self.memory)
        self.phi_k.fill(1.0)  # FOR NOW DO NOT USE IT

    def update(self, a_cur, g_cur, precond=None):
        """
        Compute the next search direction
        """

        # IS THIS OUTSIDE? ALWAYS DONE??
        # preconditioning and initial search direction
        self.search_dir = g_cur.copy()
        if precond is not None:
            self.search_dir.a_ux = apply_precond(precond, g_cur.a_ux)

        # Step 1: First iteration initialization
        if self.local_iter == 0:
            # Store current gradient and variables
            self.g_old = g_cur.copy()
            self.a_old = a_cur.copy()

            self.du = [0.0 * g_cur for _ in range(self.memory)]
            self.dj = [0.0 * g_cur for _ in range(self.memory)]

            # self.ds = [0.0 * g_cur for _ in range(self.memory)]
            # self.dy = [0.0 * g_cur for _ in range(self.memory)]

            # Increment iteration counter
            self.local_iter += 1

        else:
            # Step 2: Determine memory index for circular storage
            m = self.local_iter % self.memory

            # Store changes in variables (ds) and gradients (dy)
            self.ds = a_cur - self.a_old
            self.dy = g_cur - self.g_old

            # BY_K HAS TO BE DEFINED, IS IT A KARRAY? probably
            if precond is not None:
                by_k = self.g_old.copy()
                by_k.a_ux = apply_precond(precond, np.array(self.dy.a_ux))
            else:
                by_k = self.dy.copy()

            # first use of equation 19
            by_k = self.update_bv(by_k, self.dy, self.du,
                                  self.dj, self.yj_k,
                                  np.maximum(1, self.local_iter
                                             - self.memory), self.local_iter)

            self.dj[m] = self.ds - by_k
            self.yj_k[m] = self.dy @ self.dj[m]

            if self.method == 'LSR1':
                if abs(self.yj_k[m]) < 1e-12:
                    self.yj_k[m] = 1e-12
            
            self.yy = self.dy @ self.dy
            if abs(self.yy) > 1.0e-15:
                self.du[m] = self.dy * (1.0 / self.yy)
            else:
                self.du[m] = self.dy * 1.0e15

            # OTHER METHODS HERE(?)

            self.search_dir = self.update_bv(self.search_dir,
                                             g_cur, self.du, self.dj,
                                             self.yj_k, np.maximum(
                                                 1, self.local_iter -
                                                 self.memory + 1),
                                             self.local_iter + 1)
                 
            # Step 6: Update stored previous gradient and variable
            self.g_old = g_cur.copy()
            self.a_old = a_cur.copy()

            self.local_iter += 1

        # Return updated search direction
        self.search_dir.a_ux *= -1
        
        return self.search_dir
            
    def update_bv(self, bv, dy, du, dj, yj_k, i_0, i_m):

        """
        Recursive formula (Eq 19 from Journal of Chemical Theory and
        Computation, 16(11), 6968-6982.
        https://doi.org/10.1021/acs.jctc.0c00597)
        """

        # First loop: backward over stored vectors
        alpha = np.zeros(self.memory)

        for c_ind in range(i_0, i_m):
           # yr = self.dy[c_ind] @ self.search_dir
            uv = du[c_ind] @ dy
            jv = dj[c_ind] @ dy
            alpha = jv - yj_k[c_ind] * uv
            # only beta_p (no linear combination, phi=0)
            beta_p = dj[c_ind] * uv + du[c_ind] * alpha
            beta_ms = dj[c_ind] * (jv / yj_k[c_ind])
            # Update search direction with alpha and beta corrections
            # self.search_dir += self.ds[c_ind] * (alpha[c_ind] - beta)
            phi = 0.0
            beta = (1.0 - phi) * beta_ms + phi * beta_p
            bv += beta

        return bv


def apply_precond(prec, x, const=1.0):
    """
    Apply preconditioner to the gradient
    """
    y = []
    for k in range(x.shape[0]):
        if prec[k].dtype == complex:
            y.append(const * (prec[k].real * x[k].real +
                              1.0j * prec[k].imag * x[k].imag))
        else:
            y.append(const * prec[k] * x[k])
    return np.array(y)
