lucid.random.bernoulli
======================

.. autofunction:: lucid.random.bernoulli

The `bernoulli` function generates a tensor by sampling from a Bernoulli distribution 
for each element in the given probability tensor. Each element in the output tensor 
is either `0` or `1` based on the corresponding probability value.

Function Signature
------------------

.. code-block:: python

    def bernoulli(
        probs: _ArrayOrScalar | Tensor,
        requires_grad: bool = False,
        keep_grad: bool = False,
    ) -> Tensor

Parameters
----------

- **probs** (*_ArrayOrScalar | Tensor*):
  A tensor or scalar representing probabilities for each element to be `1`.
  Values must lie in the range `[0, 1]`.

- **requires_grad** (*bool*, optional):
  If `True`, the resulting tensor will track gradients for automatic differentiation.
  Defaults to `False`.

- **keep_grad** (*bool*, optional):
  Determines whether gradient history should persist across multiple operations.
  Defaults to `False`.

Returns
-------

- **Tensor**:
  A tensor of the same shape as `probs`, with elements sampled from a Bernoulli distribution
  (`0` or `1`).

Example
-------

.. code-block:: python

    >>> import lucid
    >>> x = lucid.random.bernoulli([0.2, 0.8, 0.5])
    >>> print(x)
    Tensor([0, 1, 0], grad=None)

By default, the generated tensor does not track gradients.
To enable gradient tracking, set `requires_grad=True`:

.. code-block:: python

    >>> probs = lucid.Tensor([0.3, 0.7, 0.9])
    >>> y = lucid.random.bernoulli(probs, requires_grad=True)
    >>> print(y.requires_grad)
    True

.. note::

  - The `probs` values must be in the range `[0, 1]`.
  - Use `lucid.random.seed` to ensure reproducibility of random samples.
  - Since this function involves discrete random sampling, gradients cannot
    be propagated through it.
