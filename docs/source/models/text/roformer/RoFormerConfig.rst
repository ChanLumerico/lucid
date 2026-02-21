RoFormerConfig
==============

.. autoclass:: lucid.models.RoFormerConfig

The `RoFormerConfig` dataclass extends `BERTConfig` with rotary embedding
controls used by RoFormer self-attention.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class RoFormerConfig(BERTConfig):
        rotary_value: bool = False
        rope_interleaved: bool = True

Additional Parameters
---------------------
- **rotary_value** (*bool*, optional):
  Whether RoPE is applied to value projections in addition to query/key.
  Default is False.
- **rope_interleaved** (*bool*, optional):
  Whether to use interleaved rotary dimensions. Default is True.

Inherited Parameters
--------------------
`RoFormerConfig` inherits all `BERTConfig` fields, including vocabulary size,
hidden dimensions, layer count, dropout settings, and decoder/cache options.

Preset Constructors
-------------------

`RoFormerConfig` also supports inherited presets:

- **RoFormerConfig.base(...)**: BERT-Base-like defaults with RoFormer fields.
- **RoFormerConfig.large(...)**: BERT-Large-like defaults with RoFormer fields.

Both methods support overrides via keyword arguments.

Basic Usage
-----------

.. code-block:: python

    from lucid.models import RoFormerConfig

    cfg = RoFormerConfig.base(
        vocab_size=50000,
        rotary_value=False,
        rope_interleaved=True,
    )
