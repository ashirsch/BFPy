import pickle
import numpy as np
import tensorflow as tf
import scipy.sparse as sp
import sys
import matplotlib.pyplot as plt


print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\Alex\\Documents\\Brown\\Thesis\\untitled', 'C:/Users/Alex/Documents/Brown/Thesis/untitled'])

with open('test_sess.p', 'rb') as f:
    sess = pickle.load(f)
obs = sess.data_set(90).observation.data.reshape((180*1024,1), order='F')

# def convert_sparse_matrix_to_sparse_tensor(X):
#     coo = X.tocoo()
#     indices = np.mat([coo.row, coo.col]).transpose()
#     return tf.SparseTensor(indices, coo.data, coo.shape)

basis_rows = sess.data_set(90).basis.basis_matrix.shape[0]
n = sess.data_set(90).basis.basis_matrix.shape[1] + 1

o = sp.csc_matrix((np.ones(basis_rows,),(np.arange(basis_rows), np.zeros(basis_rows,))))
A_sp = sp.hstack((sess.data_set(90).basis.basis_matrix, o)).todense()
# A_sp = A_sp.tocoo()
# indices = np.mat([A_sp.row, A_sp.col]).transpose()

# A_tensor = convert_sparse_matrix_to_sparse_tensor(A_sp)
x = tf.Variable(np.random.rand(2049,1), dtype=tf.float64)

# sp_indices = tf.placeholder(tf.int64, [None, 2])
# sp_values = tf.placeholder(tf.float64)
# sp_shape = tf.placeholder(tf.int64, [2, ])
# A = tf.SparseTensor(sp_indices, sp_values, sp_shape)
A = tf.placeholder(tf.float64, shape=[180*1024, 2049])
b = tf.placeholder(tf.float64, shape=[180*1024, 1])
print('making recon')
reconstruction = tf.matmul(A, tf.abs(x))
print('making loss')
loss = tf.norm(reconstruction - b, 2)
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

session = tf.Session()
session.run(tf.initialize_all_variables())

print('training...')
for step in range(10):
    c, loss_val, x_val = session.run([train_step, loss, x], feed_dict={A: A_sp, b: obs})
    print(step, loss_val)
with open('tf_x_val.p', 'wb') as f:
    pickle.dump(x_val, f)
plt.plot(x_val[0:1024])

# x = tf.placeholder(tf.float32, [None, 1])
# y_ = tf.placeholder(tf.float32, [None, 1])
#
# b = tf.Variable(tf.zeros([1]))
# w = tf.Variable(tf.zeros([1, 1]))
#
# y = w * x + b
#
# loss = tf.reduce_sum((y - y_) * (y - y_))
# train_step = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
#
# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
#
# for step in range(10):
#     sess.run(train_step, feed_dict={x:[[2.3],[1.7],[-3.8],[0.5],[-4.1],[-1.5],[-2.5],[6.2]],
#                                     y_:[[-4.4],[-3.6],[7.7],[-0.9],[8.3],[2.9],[4.9],[-12.2]]})
#     print(step, sess.run(w), sess.run(b))