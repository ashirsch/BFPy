import numpy as np
import scipy.sparse as sp
import cvxpy as cvx
import time
from .model import Model


class Ridge(Model):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
        self.name = "RIDGE"

    def run(self, bases, observation, verbose=True):
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
