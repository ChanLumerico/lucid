nn.init.constant_
=================

.. autofunction:: lucid.nn.init.constant_

The `constant_` function fills the input `tensor` with a constant value.

Function Signature
------------------

.. code-block:: python

    def constant_(tensor: Tensor, val: _Scalar) -> None

Parameters
----------

- **tensor** (:class:`Tensor`): 
  The tensor to be initialized.

- **val** (`_Scalar`): 
  The constant value to fill the tensor with.

Returns
-------

- **None**: 
  The function modifies the `tensor` in-place with the constant value.

Examples
--------

.. code-block:: python

    >>> import lucid
    >>> from lucid.nn.init import constant_
    >>> tensor = lucid.zeros((3, 3))
    >>> constant_(tensor, val=5)
    >>> print(tensor)
    Tensor([[ 5,  5,  5],
            [ 5,  5,  5],
            [ 5,  5,  5]], requires_grad=False)
