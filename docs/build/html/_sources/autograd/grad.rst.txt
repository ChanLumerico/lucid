lucid.autograd.grad
===================

.. autofunction:: lucid.autograd.grad

`lucid.autograd.grad` computes gradients of one or more outputs with respect
to one or more inputs, without permanently storing gradients on the inputs.
This makes it suitable for functional-style training loops, meta-optimization,
or cases where you want gradients as values rather than side effects.

The function expects outputs that require gradients, and will seed the backward
pass with ones if `grad_outputs` is omitted. Each output can be paired with a
custom seed gradient to support vector-Jacobian products.

Function Signature
------------------

.. code-block:: python

    def grad(
        outputs,
        inputs,
        grad_outputs=None,
        retain_graph=False,
        allow_unused=False,
    )

Parameters
----------

- **outputs**: A Tensor or iterable of Tensors representing the outputs.
- **inputs**: A Tensor or iterable of Tensors to differentiate with respect to.
- **grad_outputs**: Optional seed gradients for each output. If omitted, ones
  are used.
- **retain_graph**: Whether to retain the graph after backward passes.
- **allow_unused**: If `True`, returns `None` for inputs that are not connected.

Returns
-------

- Gradients for each input. Returns a single gradient if a single input is
  provided; otherwise returns a tuple aligned with `inputs`.

Behavior Notes
--------------

- The function restores any pre-existing `.grad` values on the input tensors
  after computation.
- Input tensors do not retain gradients unless explicitly held elsewhere.
- Multiple outputs are handled by sequential backward passes while optionally
  retaining the graph.

Example
-------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.tensor([1.0, -2.0, 3.0], requires_grad=True)
    >>> y = lucid.sum(x * x)
    >>> dx = lucid.autograd.grad(y, x)
    >>> dx
    array([ 2., -4.,  6.])

Multiple Outputs
----------------

.. code-block:: python

    >>> a = lucid.tensor([2.0, -1.0, 0.5], requires_grad=True)
    >>> b = lucid.tensor([3.0, 4.0, -2.0], requires_grad=True)
    >>> out1 = lucid.sum(a * b)
    >>> out2 = lucid.sum(a * a)
    >>> da, db = lucid.autograd.grad((out1, out2), (a, b))
    >>> da
    array([ 7. ,  2. , -1. ])
    >>> db
    array([ 2. , -1. ,  0.5])
