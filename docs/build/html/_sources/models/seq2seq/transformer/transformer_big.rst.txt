transformer_big
===============

.. autofunction:: lucid.models.transformer_big

The `transformer_big` function creates a larger variant of the `Transformer` class,
configured for more complex tasks requiring increased model capacity.

**Total Parameters**: 213,237,472

Function Signature
------------------

.. code-block:: python

    @register_model
    def transformer_big(src_vocab_size: int = 12000, tgt_vocab_size: int = 12000) -> Transformer

Parameters
----------
- **src_vocab_size** (*int*, optional):
  Size of the source vocabulary. Default is 12,000.

- **tgt_vocab_size** (*int*, optional):
  Size of the target vocabulary. Default is 12,000.

Returns
-------
- **Transformer**:
  An instance of `Transformer` with larger capacity for 
  improved performance on complex tasks.

Examples
--------

.. code-block:: python

    >>> import lucid.models as models
    >>> transformer = models.transformer_big()
    >>> print(transformer)
    Transformer(src_vocab_size=12000, tgt_vocab_size=12000, d_model=1024, ...)

