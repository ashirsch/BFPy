Ridge
-----

The ``Ridge`` model sets up and solves the Ridge (or Tikhonov) least squares problem:

$$\\text{minimize } || Ax+\\eta-b ||_2 + \\alpha || \\Delta x_i ||_2$$
$$\\text{subject to } x_i \\ge 0$$

where A is the a basis matrix, b is an observation, x is a vector of coefficients, and eta is a constant background
term. Alpha is a hyperparameter that specifies the weight given to the smoothness regularization term.

``Ridge`` differs from ``Quadratic`` in that it gives defined the problem in terms of cvxpy's ``norm2()`` function,
and lets it handle the appropriate factorizations.

.. autoclass:: kemitter.model.Ridge
   :members:
