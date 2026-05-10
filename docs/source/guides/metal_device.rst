Metal GPU Guide
===============

Lucid's GPU backend uses Apple's Metal Performance Shaders via the
`MLX <https://github.com/ml-explore/mlx>`_ library.  This guide covers
device selection, lazy evaluation, memory management, and profiling.

Device selection
----------------

Move tensors or entire models to Metal with ``.to("metal")``:

.. code-block:: python

   import lucid
   import lucid.nn as nn

   x     = lucid.randn(128, 128)          # cpu
   x_gpu = x.to("metal")                  # Metal GPU
   model = nn.Linear(128, 64).to("metal")

Lazy evaluation
---------------

Metal operations are **lazy** — they are recorded into a computation
graph and only executed when a *sync point* is reached:

- :func:`lucid.eval` — explicit flush
- ``.item()`` — scalar extraction forces evaluation
- ``.numpy()`` — CPU round-trip forces evaluation

.. code-block:: python

   y = x_gpu @ x_gpu.T     # no GPU work yet
   lucid.eval(y)            # graph is executed here
   print(y[0, 0].item())    # implicit sync

Memory utilities
----------------

.. code-block:: python

   print(lucid.metal_memory_allocated())         # bytes in use
   print(lucid.metal_max_memory_allocated())      # peak bytes
   lucid.reset_peak_memory_stats()

Mixed-device operations
-----------------------

Lucid does **not** silently migrate tensors between devices.
A mixed-device operation raises a ``RuntimeError``:

.. code-block:: python

   a = lucid.randn(4, 4)                # cpu
   b = lucid.randn(4, 4, device="metal")  # Metal

   a + b   # RuntimeError: device mismatch

Profiling Metal kernels
-----------------------

.. code-block:: python

   from lucid.profiler import profile, ProfilerActivity

   with profile(activities=[ProfilerActivity.Metal]) as prof:
       y = model(x_gpu)
       lucid.eval(y)

   print(prof.key_averages().table())
