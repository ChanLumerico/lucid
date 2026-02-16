lucid.grad_enabled
==================

.. autofunction:: lucid.grad_enabled

The `grad_enabled` function returns the current state of gradient tracking, 
which indicates whether gradients are being computed for tensor operations.

Function Signature
------------------

.. code-block:: python

    def grad_enabled() -> bool

Returns
-------

- **bool**: 
    - `True` if gradients are being tracked.
    - `False` if gradients are disabled.

Usage
-----

The `grad_enabled` function is used to check if gradient computation is currently enabled. 

This can be useful for debugging or ensuring that gradient tracking is appropriately 
enabled or disabled in certain parts of the code.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> print(lucid.grad_enabled())  # Output: True (assuming gradients are enabled by default)
    >>> with lucid.no_grad():
    >>>     print(lucid.grad_enabled())  # Output: False (inside the no_grad context manager)

In the example above, `grad_enabled` shows whether gradient tracking is on or off at any given point in the code.

How It Works
------------

The function simply returns the value of the global `_grad_enabled` variable, which determines 
if gradient computation is enabled. When `grad_enabled()` is called, it reflects the state of the 
gradient tracking system, helping you determine if operations are being performed with or without gradient tracking.

.. note::
    The global `_grad_enabled` variable is modified by context managers like `no_grad()` 
    to disable gradients during specific operations. 

Potential Pitfalls
------------------

Using `grad_enabled` to check the gradient state is straightforward, but consider the following:

.. caution::

    - The state returned by `grad_enabled` is global, which means if other parts of your code 
      modify the gradient state, it will affect the result of `grad_enabled`.

    - Since `grad_enabled` only returns the current state, ensure that its use does not 
      interfere with the dynamic behavior of gradient tracking in your program.

Conclusion
----------

The `grad_enabled` function provides a simple and effective way to check the current 
gradient tracking state in your code. It is particularly useful for debugging and controlling 
the flow of operations that require or do not require gradients.

