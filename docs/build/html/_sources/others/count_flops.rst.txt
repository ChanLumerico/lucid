lucid.count_flops
=================

.. autofunction:: lucid.count_flops

Overview
--------

`count_flops` temporarily **enables floating-point-operation (FLOPs) tracking** for
every tensor operation executed inside its block.  
All FLOPs are accumulated along the computational graph, so the final
tensor produced within the block exposes the total cost through its
`.flops` property.

Function Signature
------------------

.. code-block:: python

    @contextmanager
    def count_flops() -> Generator

Parameters
----------

This context manager takes **no arguments**.

Returns
-------

- **Generator** - An object that, when used in a `with` statement, enables
  FLOPs counting for the enclosed code and automatically restores the
  previous state afterwards.

Examples
--------

Context-manager usage:

.. code-block:: python

    with lucid.count_flops():
        logits = model(x)
        total_ops = logits.flops      # total FLOPs for forward pass
    print(f"Forward cost: {total_ops:,} FLOPs")

Nested usage is safe; each block restores the prior global state.

.. note::

    * The accumulated count includes **only** operations performed while the
      context manager is active.
    * Accessing `tensor.flops` outside the context returns whatever value was
      attached during creation; it does **not** keep updating after the block
      exits.

Benefits
--------

* **Quick profiling** - Estimate computational cost without external tools.
* **No code changes** - Wrap any forward pass in a single `with` block.
* **Automatic cleanup** - Original FLOPs-tracking state is restored even if
  an exception occurs.

.. caution::

    * Because it toggles a global flag, avoid mixing FLOPs tracking with
      multi-threaded or concurrently executed model runs unless you manage the
      context carefully.
    * FLOPs accounting is approximate; custom or fused kernels may not map
      one-to-one to standard floating-point operations.
