import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import sys


def nn_solve(A, b, x_init, tol=1e-10, m_recur_lim=100, m_iter_lim=20, k_iter_lim=10):
    # r = b - tf.matmul(A, x_init)

    loop_vars = (x_init, tf.ones_like(b), x_init, x_init, tf.zeros_like(b), tf.zeros_like(b), 1)
    def outer_cond(x, r, z, d, w, prev_r, k):
        # return tf.logical_and(tf.abs(tf.norm(prev_r) - tf.norm(r)) / tf.norm(prev_r) > tol, k < k_iter_lim)
        return k < k_iter_lim

    def outer_body(x, r, z, d, w, prev_r, k):
        L = tf.diag(tf.reshape(x, [n,]))
        prev_r = r
        r = b - tf.matmul(A, x)
        z = tf.matmul(L, tf.matmul(A, r, transpose_a=True))
        d = z
        d_mat = tf.concat([z, z], 1)
        w = tf.matmul(A, z)
        inner_loop_vars = (x, r, d, w, d_mat, tf.constant(0.0, dtype=tf.float64), tf.convert_to_tensor([[0]], dtype=tf.float64), tf.convert_to_tensor([[0]], dtype=tf.float64), prev_r, 1)

        def inner_cond(x, r, d, w, d_mat, alpha, alpha_bar, beta, prev_r, m):
            # return tf.logical_and(tf.abs(tf.norm(prev_r) - tf.norm(r)) / tf.norm(prev_r) > tol ,
            #                       tf.logical_and(alpha_bar != 0, m < m_iter_lim))
            return tf.logical_and(alpha_bar != 0, m < m_iter_lim)


        def inner_body(x, r, d, w, d_mat, alpha, alpha_bar, beta, prev_r, m):
            alpha = tf.matmul(r, w, transpose_a=True) / tf.matmul(w, w, transpose_a=True)

            zeros = tf.zeros_like(x)
            mask = tf.less(d, zeros)
            alpha_bar = tf.minimum(alpha, tf.reduce_min(tf.div(-tf.boolean_mask(x, mask), tf.boolean_mask(d, mask))))
            x = x + alpha_bar * d
            L = tf.diag(tf.reshape(x, [n,]))
            prev_r = r
            r = prev_r - alpha * w
            z = tf.matmul(A, r, transpose_a=True)
            z_bar = tf.matmul(L, z)
            Az_bar = tf.matmul(A, z_bar)

            new_beta = -tf.matmul(Az_bar, w, transpose_a=True) / tf.matmul(w, w, transpose_a=True)
            beta = tf.concat([beta, new_beta], 0)

            d = z_bar + tf.matmul(d_mat, beta)
            d_mat = tf.concat([d_mat, d], 1)

            w = tf.matmul(A, d)
            return (x, r, d, w, d_mat, alpha, alpha_bar, beta, prev_r, m+1)

        x, r, d, w = tf.while_loop(inner_cond, inner_body, inner_loop_vars,
                                   shape_invariants=(x.get_shape(), r.get_shape(), d.get_shape(), w.get_shape(),
                                                     tf.TensorShape([n, None]), tf.TensorShape(None), tf.TensorShape(None),
                                                     tf.TensorShape([None, 1]), r.get_shape(), tf.TensorShape(None)))[:4]
        return (x, r, z, d, w, prev_r, k+1)


    return tf.while_loop(outer_cond, outer_body, loop_vars)

m = 184320
n = 2048

def main():
    sys.path.insert(0, "C:\\Users\\Alex\\Documents\\Brown\\Thesis\\BFPy")
    # np.random.seed(0)
    # A_np = np.random.randn(m, n) + 10
    # x_np = np.reshape(np.sin(np.linspace(0, 2 * np.pi, n)) + 2, (n, 1))
    # b_np = A_np @ x_np
    with open('simple.p', 'rb') as f:
        obs, basis = pickle.load(f)

    # m = basis.shape[0]
    # n = basis.shape[1]

    L = .25*np.diag(np.ones((n-1,)), k=1) - np.diag(np.ones((n,)))
    A_np = np.vstack((basis.todense(), L))

    # b_np = np.random.rand(m, 1)
    # x_np = np.random.randn(n, 1) + 10
    x_init = np.random.rand(n, 1) + 1

    zz= np.zeros((L.shape[0], 1))
    b_np = np.vstack((obs.reshape((m,1), order='f'), zz))
    # expected_x = np.linalg.solve(A_np, b_np)
    print('solved np')
    A_init = tf.placeholder(tf.float64, shape=A_np.shape)
    A0 = tf.Variable(A_init)
    init = tf.global_variables_initializer()

    # A0 = tf.convert_to_tensor(A_np, dtype=tf.float64)
    b0 = tf.convert_to_tensor(b_np, dtype=tf.float64)
    x_init0 = tf.convert_to_tensor(x_init, dtype=tf.float64)
    solver = nn_solve(A0, b0, x_init0)
    with tf.Session() as sess:
        print('solving tf...')
        sess.run(init, feed_dict={A_init: A_np})
        t0 = time.time()
        solution = sess.run(solver)
        t1 = time.time()
        print(solution[-1])
        # print(x_np[0])
        # print(expected_x)
        print(solution[0][0])
        # print(np.sqrt(((x_np-solution[0]) ** 2).mean()))
        plt.plot(solution[0][:1024])
        plt.plot(solution[0][1024:])
        print(t1 - t0)
        # plt.plot(x_np)
        # plt.plot(x_init)
        plt.show()

if __name__ == "__main__":
    main()
