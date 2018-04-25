import numpy as np
import scipy.sparse as sp
import cvxpy as cvx
from numba import jit
import time
from .model import Model


class Quadratic(Model):
    """Solves and stores results of cvxpy ridge regression solver, posed in its quadratic form.

        Attributes:
            name (str): "QUADRATIC" (constant)
            alpha (float): the regularization parameter for the smoothness penalty
            cache (ndarray): A cached 2D (A^T*A) array from a previous calculation (used for repeated fits).

        See Also:
            :class:`~kemitter.model.model.Model`
        """
    def __init__(self, alpha):
        super().__init__()
        self.cache = None
        self.name = "QUADRATIC"
        self.alpha = alpha

    def run(self, bases, observation, verbose=True, caching=False):
        """Runs the model calculations.

        Bases and observations are loaded into proper polarized data sets. In this step,
        arguments are checked to ensure polarization angles match in value and order. Any bases that have not been
        built already are built with their corresponding ``build()`` method.

        Bases and observations of multiple polarizations are then concatenated and given to cvxpy and MOSEK for solving.

        The problem is first factorized into its quadratic form by performing the matrix multiplication (A^T*A).
        This is an expensive operation that comes with the benefit of much faster solving times. For repeated fits
        (for example, fits of multiple frames with the same bases), this resulting matrix can be cached to avoid
        repetitive recalculations.

        Results are returned and processed in inherited ``Model`` attributes.

        Args:
            bases (list of Basis): The basis objects (built or not) of several polarizations, to be used for fitting.
            observation (list of Observation): The observation objects of several polarizations, to be used for fitting.
            verbose (bool): The console verbosity of the called solver (MOSEK) [default True].
            caching (bool): Whether or not to use or store the resulting basis-matrix multiplication [default False].
        """
        super().run(bases, observation)

        print('Bases and observations loaded in model')
        t0 = time.time()
        print('Forming QP problem:')
        print('    Setting up basis and observation matrices')
        A = sp.vstack([self.data_set(angle).basis.basis_matrix for angle in self.polarization_angles])

        basis_rows = A.shape[0]
        n = A.shape[1] + 1

        o = sp.csc_matrix((np.ones(basis_rows, ), (np.arange(basis_rows), np.zeros(basis_rows, ))))
        A = sp.hstack((A, o))

        D = np.diag(np.ones((A.shape[1] - 1,)), k=1) - np.diag(np.ones((A.shape[1],)))
        D = D[:-1, :]
        alpha = self.alpha
        A = sp.vstack((A, alpha * D))
        H = cvx.Constant(A)

        b = np.vstack([self.data_set(angle).observation.data.reshape((180 * 1024, 1), order='F') for angle in
                       self.polarization_angles])

        zz = np.zeros((A.shape[1] - 1, 1))
        b = np.vstack((b,zz))
        b = cvx.Constant(b)

        x = cvx.Variable(n)

        if self.cache is None or not caching:
            print('    Multiplying ATA')
            P = np.array(ATA(A.todense()))
            if caching:
                self.cache = P
        else:
            print('    Pulling ATA from cache')
            P = self.cache
        print('    Defining objective')
        objective = cvx.Minimize(cvx.quad_form(x, P) - 2*((H.T*b).T)*x + b.T*b)
        print('    Defining constraints')
        constraints = [x >= 0]
        print('    Defining problem')
        prob = cvx.Problem(objective, constraints)
        print('Problem Formulation DONE\n\nCalling the solver: ' + cvx.MOSEK)
        ts0 = time.time()
        prob.solve(solver=cvx.MOSEK, verbose=verbose)
        ts1 = time.time()
        print(cvx.MOSEK + ' done in {0:.2f} seconds.'.format(ts1 - ts0))
        if x.value is not None:
            print('\nProcessing solution')
            self.solver_result = np.array(x.value)
            self.background = self.solver_result[-1]
            self._process_result(self.solver_result[:-1])
            t1 = time.time()
            print('Fitting DONE:\n    Elapsed time: {0:.2f} s'.format(t1-t0))


@jit(nopython=True, parallel=True)
def ATA(A):
    return A.transpose() @ A
