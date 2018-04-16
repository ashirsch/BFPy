import numpy as np
import scipy.sparse as sp
import cvxpy as cvx
import time
from .model import Model


class Ridge(Model):

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def run(self, bases, observation, verbose=True):
        super().run(bases, observation)

        print('loaded')
        A = sp.vstack([self.data_set(angle).basis.basis_matrix for angle in self.polarization_angles])

        basis_rows = A.shape[0]
        n = A.shape[1] + 1

        o = sp.csc_matrix((np.ones(basis_rows, ), (np.arange(basis_rows), np.zeros(basis_rows, ))))
        A = sp.hstack((A, o))
        H = cvx.Constant(A)
        D = np.diag(np.ones((A.shape[1] - 1,)), k=1) - np.diag(np.ones((A.shape[1],)))
        D = cvx.Constant(D[:-1, :])

        b = np.vstack([self.data_set(angle).observation.data.reshape((180 * 1024, 1), order='F') for angle in self.polarization_angles])
        b = cvx.Constant(b)

        alpha = self.alpha
        x = cvx.Variable(n)
        print('variables and constants made')
        print('defining objective')
        objective = cvx.Minimize(cvx.norm2(H * x - b) + alpha * cvx.norm2(D * x))
        print('defining constraints')
        constraints = [x[:-1] >= 0]
        print('defining problem')
        prob = cvx.Problem(objective, constraints)
        print('\nCalling the solver...')
        t0 = time.time()
        prob.solve(solver=cvx.MOSEK, verbose=verbose)
        t1 = time.time()
        self.solver_result = np.array(x.value)
        self.background = self.solver_result[-1]
        self._process_result(self.solver_result[:-1])
        print('Done in {0:.3f} seconds.'.format(t1 - t0))
