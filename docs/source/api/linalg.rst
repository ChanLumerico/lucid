lucid.linalg
============

.. currentmodule:: lucid.linalg

Linear algebra operations backed by Apple Accelerate LAPACK/BLAS on CPU
and MLX on Metal GPU.

.. note::

   All ``lucid.linalg`` functions must be called via the ``lucid.linalg``
   namespace.  No top-level shortcuts (e.g. ``lucid.norm``, ``lucid.cross``)
   are provided — each operation has exactly one canonical path.

Decompositions
--------------

.. autofunction:: svd
.. autofunction:: eig
.. autofunction:: eigh
.. autofunction:: qr
.. autofunction:: cholesky
.. autofunction:: lu
.. autofunction:: lu_factor
.. autofunction:: lu_solve

Matrix products
---------------

.. autofunction:: matmul
.. autofunction:: multi_dot
.. autofunction:: vecdot
.. autofunction:: outer
.. autofunction:: tensordot
.. autofunction:: kron

Norms and conditions
--------------------

.. autofunction:: norm
.. autofunction:: vector_norm
.. autofunction:: matrix_norm
.. autofunction:: cond
.. autofunction:: det
.. autofunction:: slogdet

Solvers
-------

.. autofunction:: solve
.. autofunction:: lstsq
.. autofunction:: inv
.. autofunction:: pinv
.. autofunction:: matrix_power
.. autofunction:: matrix_exp

Utilities
---------

.. autofunction:: cross
.. autofunction:: diagonal
.. autofunction:: trace
.. autofunction:: vander
