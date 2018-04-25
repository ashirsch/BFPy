import numpy as np
import scipy.sparse as sp
import cvxpy as cvx
import time
from .model import Model


class Ridge(Model):
    """Solves and stores results of cvxpy ridge regression solver, implemented by the ``cvxpy.norm2()`` function.

    Attributes:
        name (str): "RIDGE" (constant)
        alpha (float): the regularization parameter for the smoothness penalty

    See Also:
        :class:`~kemitter.model.model.Model`
    """
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.name = "RIDGE"

    def run(self, bases, observation, verbose=True):
        """Runs the model calculations.

        Bases and observations are loaded into proper polarized data sets. In this step,
        arguments are checked to ensure polarization angles match in value and order. Any bases that have not been
        built already are built with their corresponding ``build()`` method.

        Bases and observations of multiple polarizations are then concatenated and given to cvxpy and MOSEK for solving.

        Results are returned and processed in inherited ``Model`` attributes.

        Args:
            bases (list of Basis): The basis objects (built or not) of several polarizations, to be used for fitting.
            observation (list of Observation): The observation objects of several polarizations, to be used for fitting.
            verbose (bool): The console verbosity of the called solver (MOSEK).
        """
        super().run(bases, observation)

        print('Bases and observations loaded in model')
        t0 = time.time()
        print('Forming Regularized problem:')
        print('    Setting up basis and observation matrices')
        A = sp.vstack([self.data_set(angle).basis.basis_matrix for angle in self.polarization_angles])

        basis_rows = A.shape[0]
        n = A.shape[1] + 1

        o = sp.csc_matrix((np.ones(basis_rows, ), (np.arange(basis_rows), np.zeros(basis_rows, ))))
        A = sp.hstack((A, o))
        H = cvx.Constant(A)

        b = np.vstack([self.data_set(angle).observation.data.reshape((180 * 1024, 1), order='F') for angle in
                       self.polarization_angles])
        b = cvx.Constant(b)

        x = cvx.Variable(n)

        print('    Defining regularization term with alpha = {0}'.format(self.alpha))
        D = np.diag(np.ones((A.shape[1] - 1,)), k=1) - np.diag(np.ones((A.shape[1],)))
        D = cvx.Constant(D[:-1, :])
        alpha = self.alpha

        print('    Defining objective')
        objective = cvx.Minimize(cvx.norm2(H * x - b) + alpha * cvx.norm2(D * x))
        print('    Defining constraints')
        constraints = [x[:-1] >= 0]
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
