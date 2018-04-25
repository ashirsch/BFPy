Quadratic
---------

The ``Quadratic`` model sets up and solves the Ridge (or Tikhonov) least squares problem:

$$\\text{minimize } || Ax+\\eta-b ||_2 + \\alpha || \\Delta x_i ||_2$$
$$\\text{subject to } x_i \\ge 0$$

where A is the a basis matrix, b is an observation, x is a vector of coefficients, and eta is a constant background
term. Alpha is a hyperparameter that specifies the weight given to the smoothness regularization term.

``Quadratic`` first factorizes the problem into it's quadratic form

$$\\text{minimize } x^T(A^TA + \\alpha^2D^TD)x - 2(A^Tb)^T + b^Tb $$

before sending the problem to cvxpy. This results in faster solving times at the cost of performing
the matrix multiplication required by the first term.

.. autoclass:: kemitter.model.Quadratic
   :members:
