models.transformer_base
=======================

.. autofunction:: lucid.models.transformer_base

The `transformer_base` function creates an instance of the `Transformer` 
class with default hyperparameters, suitable for standard Transformer-based tasks.

**Total Parameters**: 62,584,544

Function Signature
------------------

.. code-block:: python

    @register_model
    def transformer_base(src_vocab_size: int = 12000, tgt_vocab_size: int = 12000) -> Transformer

Parameters
----------
- **src_vocab_size** (*int*, optional):
  Size of the source vocabulary. Default is 12,000.

- **tgt_vocab_size** (*int*, optional):
  Size of the target vocabulary. Default is 12,000.

Returns
-------
- **Transformer**:
  An instance of `Transformer` initialized with the given vocabulary sizes.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> transformer = models.transformer_base()
    >>> print(transformer)
    Transformer(src_vocab_size=12000, tgt_vocab_size=12000, d_model=512, ...)
