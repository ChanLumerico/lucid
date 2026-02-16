lucid.no_grad
=============

.. autoclass:: lucid.no_grad

Overview
--------

`no_grad` temporarily **disables gradient calculation**.  
It can be used either

* as a **context manager**

  .. code-block:: python

      with lucid.no_grad():
          output = model(x)

* or as a **decorator**

  .. code-block:: python

      @lucid.no_grad()
      def inference(x):
          return model(x)

Using it during inference (or any code that does **not** require gradients) saves
memory and speeds up computation.

Function Signature
------------------

.. code-block:: python

    class no_grad:
        def __enter__(self): ...
        def __exit__(self, exc_type, exc_value, traceback): ...
        def __call__(self, fn): ...

Parameters
----------

`no_grad` takes **no arguments**.  
Calling it—either in a `with` statement or as `@lucid.no_grad()`—creates a new
instance that manages gradient state for the enclosed block or function.

Returns
-------

*When used as a context manager* - returns the instance itself so that nested
usage is safe.

*When used as a decorator* - returns a **wrapped function** that will execute
with gradient tracking disabled.

Examples
--------

Context-manager usage:

.. code-block:: python

    with lucid.no_grad():
        preds = model(inputs)  # gradients are *not* stored

Decorator usage:

.. code-block:: python

    @lucid.no_grad()
    def evaluate(batch):
        return model(batch).argmax(-1)

.. note::

    * The previous global state is restored even if an exception occurs.
    * Avoid wrapping training steps with `no_grad`; doing so prevents parameter
      updates.

Benefits
--------

* **Reduced memory usage** - tensors skip gradient bookkeeping.
* **Faster execution** - no backward graph is built.
* **Clean syntax** - one line disables and then restores gradients.

.. caution::

    * Because it toggles a global flag, be mindful when mixing code that both
      requires and disables gradients.
