import tensorflow as tf
import numpy as np

def cg_solve(A, b, x_init, tol=1e-1, max_iter=1000):
    delta = tol*tf.norm(b)

    def body(x, k, r_norm_sq, r, p):
        Ap = A(p)
        alpha = r_norm_sq / tf.matmul(p, Ap, transpose_a=True)
        x = x + alpha*p
        r = r - alpha*Ap
        r_norm_sq_prev = r_norm_sq
        r_norm_sq = tf.matmul(r,r, transpose_a=True)
        beta = r_norm_sq / r_norm_sq_prev
        p = r + beta*p
        return (x, k+1, r_norm_sq, r, p)

    def cond(x, k, r_norm_sq, r, p):
        return tf.logical_and(tf.reshape(tf.sqrt(r_norm_sq) > delta, []), k < max_iter)


    r = b - A(x_init)
    loop_vars = (x_init, tf.constant(0),
                 tf.matmul(r, r, transpose_a=True), r, r)
    return tf.while_loop(cond, body, loop_vars)[:3]


def main():
    m = 5
    n = 5
    np.random.seed(0)
    A_np = np.random.randn(m, n)
    b_np = np.random.randn(m, 1)
    x_init = np.zeros((n, 1))
    shift = 1

    expected_x = np.linalg.solve(A_np, b_np)
    print('solved np')
    A0 = tf.convert_to_tensor(A_np, dtype=tf.float64)
    b0 = tf.convert_to_tensor(b_np, dtype=tf.float64)
    x_init0 = tf.convert_to_tensor(x_init, dtype=tf.float64)

    def A(x):
        return tf.matmul(A0, x)

    def AT(x):
        return tf.matmul(A0, x, transpose_a=True)

    solver = cg_solve(A, b0, x_init0)
    with tf.Session() as sess:
        print('solving tf...')
        sess_np = sess.run(solver)
        print(expected_x, sess_np[0])
        print(sess_np[2])

if __name__ == "__main__":
    main()
