TransformerConfig
=================

.. autoclass:: lucid.models.TransformerConfig

`TransformerConfig` stores the vocabulary sizes and structural hyperparameters
used by :class:`lucid.models.Transformer`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class TransformerConfig:
        src_vocab_size: int
        tgt_vocab_size: int
        d_model: int
        num_heads: int
        num_encoder_layers: int
        num_decoder_layers: int
        dim_feedforward: int
        dropout: float = 0.1
        max_len: int = 5000

Parameters
----------

- **src_vocab_size** (*int*):
  Source vocabulary size.
- **tgt_vocab_size** (*int*):
  Target vocabulary size.
- **d_model** (*int*):
  Hidden dimension.
- **num_heads** (*int*):
  Number of attention heads.
- **num_encoder_layers** (*int*):
  Number of encoder layers.
- **num_decoder_layers** (*int*):
  Number of decoder layers.
- **dim_feedforward** (*int*):
  Feedforward width.
- **dropout** (*float*):
  Dropout probability.
- **max_len** (*int*):
  Maximum supported positional encoding length.

Validation
----------

- All size and layer-count fields must be greater than `0`.
- `d_model` must be divisible by `num_heads`.
- `dropout` must be in `[0, 1)`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.TransformerConfig(
        src_vocab_size=12000,
        tgt_vocab_size=12000,
        d_model=512,
        num_heads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
    )
    model = models.Transformer(config)
