Bases
======

.. toctree::

Basis (interface)
-----------------
.. autoclass:: kemitter.basis.basis.Basis
   :members:

Basis Parameters
----------------

While basis parameters should normally be defined in the initializer of each basis class, advanced users
may want to save and set basis parameters directly in their storage format: the ``BasisParameters`` class. This can
be efficient for repeated calculations. While the parameter object has checks to verify its internal state,
basis-specific checks are only handled in the initializers of the individual basis classes. Therefore care should be
when creating a custom parameter class, as doing so may bypass these basis-specific helper functions and verification
steps.

.. autoclass:: kemitter.basis.basis.BasisParameters
   :members:
