lucid.profiler
==============

.. currentmodule:: lucid.profiler

Lightweight performance profiler for Lucid operations.
Wraps Metal GPU timeline events and CPU wall-clock measurements.

Context manager
---------------

.. autoclass:: profile
   :members:
   :undoc-members:
   :show-inheritance:

Events
------

.. autoclass:: ProfilerActivity
   :members:
   :undoc-members:

Key averages
------------

.. autofunction:: key_averages

Record function
---------------

.. autofunction:: record_function
