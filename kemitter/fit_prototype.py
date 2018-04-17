import kemitter
import kemitter.vis.visualization as bfpvis
import numpy as np
import scipy.sparse as sp
import cvxpy as cvx
import time
import pickle
import matplotlib.pyplot as plt

# FIRST SUCCESSFUL FIT - Time 2 minutes - see pickled files for quick loading and result.
# Ridge Regression implemented - Time ~ 3 minutes (longer setup time)
sess = kemitter.BFPSession()

sess.load(90)
sess.load(0)

sess.data_set(90).define_basis(basis_type='EDIso',n0=1.0, n1=1.0, n2o=1.7, n2e=1.7, n3=1.5,
                       ux_range=(-1.3,1.3), uy_range=(-1.3,1.3),
                       ux_count=180, uy_count=180,
                       d=10.0, s=10.0, l=0.0, pad_w=False, trim_w=True)
sess.data_set(0).define_basis(basis_type='EDIso',n0=1.0, n1=1.0, n2o=1.7, n2e=1.7, n3=1.5,
                       ux_range=(-1.3,1.3), uy_range=(-1.3,1.3),
                       ux_count=180, uy_count=180,
                       d=10.0, s=10.0, l=0.0, pad_w=False, trim_w=True)

sess.build_bases()

# with open('test_sess.p', 'wb') as f:
#     pickle.dump(sess, f)
# with open('./test_sess.p', 'rb') as f:
#     sess = pickle.load(f)
print('loaded')
A = sp.vstack([sess.data_set(0).basis.basis_matrix, sess.data_set(90).basis.basis_matrix])

basis_rows = A.shape[0]
n = A.shape[1] + 1

o = sp.csc_matrix((np.ones(basis_rows,),(np.arange(basis_rows), np.zeros(basis_rows,))))
A = sp.hstack((A, o))
H = cvx.Constant(A)
D = np.diag(np.ones((A.shape[1] - 1,)), k=1) - np.diag(np.ones((A.shape[1],)))
D = cvx.Constant(D[:-1, :])

b = np.vstack((sess.data_set(0).observation.data.reshape((180*1024,1), order='F'),
               sess.data_set(90).observation.data.reshape((180*1024,1), order='F')))
b = cvx.Constant(b)
# plt.imshow(sess.data_set(90).observation.data)
# plt.show()
# bfpvis.basis_func_plot(sess.data_set(90).basis, 179)
### MAKE b from basis directly just to test
# x_test = np.random.rand(n, 1)
# b = A.todense() @ x_test
# plt.imshow(b.reshape((180,1024),order='F'))
# plt.show()

alpha = 0.025
x = cvx.Variable(n)
print('variables and constants made')
print('defining objective')
objective = cvx.Minimize(cvx.norm2(H*x - b) + alpha*cvx.norm2(D*x))
print('defining constraints')
constraints = [x[:-1] >= 0]
print('defining problem')
prob = cvx.Problem(objective, constraints)
print('\nSolving...')
t0 = time.time()
result = prob.solve(solver=cvx.MOSEK, verbose=True)
t1 = time.time()
with open('test_res.p', 'wb') as f:
    pickle.dump(x, f)

print('Done in {0:.3f} seconds.'.format(t1-t0))
# NORM2 worked! 2 minutes for solve!!!

# Problem data.
# m = 100
# n = 20
# np.random.seed(1)
# A = np.random.randn(m, n)
# b = np.random.randn(m, 1)
#
# # Construct the problem.
# x = Variable(n)
# objective = Minimize(sum_entries(square(A*x - b)))
# constraints = [0 <= x]
# prob = Problem(objective, constraints)
#
# print("Optimal value", prob.solve(solver=MOSEK, verbose=True))
# print("Optimal var")
# print(x.value) # A numpy matrix.