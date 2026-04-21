import numpy as np


class LBFGS:
    def __init__(self,
                 p_key=False,
                 *,
                 memory=5):
        """
        L-BFGS optimizer initialization.

        Parameters
        ----------
        array_shape : tuple
            Shape of the parameter array.
        dtype : type
            Data type of arrays (float or complex).
        array_unX : list[UGArray]
            Distributed parameter array.
        kpt_comm : object
            Communication object for summing quantities over k-points
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

        hess = precond

        # Step 1: First iteration initialization
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

            # Increment iteration counter
            self.local_iter += 1

            # Return initial search direction
            return self.search_dir

        else:
            # Step 2: Determine memory index for circular storage
            m = self.local_iter % self.memory

            # Store changes in variables (ds) and gradients (dy)
            # For first few iterations, ds = previous search direction
            #self.ds[m] = self.search_dir.copy()
            self.ds[m] = a_cur - self.a_old
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
                self.local_iter = 0
                self.rho[:] = 0.0
                for x in self.ds:
                    x[:] = 0.0
                for x in self.dy:
                    x[:] = 0.0
                return self.update(a_cur, g_cur, hess)

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
            # avoid divide by zero
            devis = np.maximum(self.rho[m] * yy, 1.0e-20)
            self.search_dir = q * (1 / devis)

            # l-bfgs-p:
            if self.p_key:
                self.rho_y = self.rho[m] * yy
                if abs(self.rho_y) > 1.e-20:
                    self.beta0 = 1.0 / self.rho_y
                else:
                    self.beta0 = 1.0e20
                if hess is not None:
                    self.search_dir.a_ux = apply_precond(hess,
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


def apply_precond(prec, x, const=1.0):
    """
    Apply preconditioner to the gradient
    """
    y = []
    for k in range(x.shape[0]):
        if prec[k].dtype == complex:
            y[k] = const * (prec[k].real * x[k].real
                            + 1.0j * prec[k].imag * x[k].imag)
        else:
            y.append(const * prec[k] * x[k])
    return np.array(y)
