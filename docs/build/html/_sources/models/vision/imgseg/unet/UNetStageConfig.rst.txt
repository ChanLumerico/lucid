UNetStageConfig
===============

.. autoclass:: lucid.models.UNetStageConfig

`UNetStageConfig` describes a single encoder, decoder, or bottleneck stage
within :class:`lucid.models.UNet`. Each stage controls its output width, the
number of repeated blocks, and optional attention and regularization behavior.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class UNetStageConfig:
        channels: int
        num_blocks: int = 2
        kernel_size: int = 3
        dilation: int = 1
        use_attention: bool = False
        dropout: float = 0.0

Parameters
----------

- **channels** (*int*): Output channel width of the stage.
- **num_blocks** (*int*): Number of repeated convolutional or residual blocks
  inside the stage.
- **kernel_size** (*int*): Convolution kernel size used inside the stage.
- **dilation** (*int*): Dilation factor applied to stage convolutions.
- **use_attention** (*bool*): Whether to append a self-attention block after
  the stage blocks.
- **dropout** (*float*): Dropout probability applied inside stage blocks.

Usage
-----

.. code-block:: python

    import lucid.models as models

    stage = models.UNetStageConfig(
        channels=128,
        num_blocks=3,
        kernel_size=3,
        dilation=1,
        use_attention=True,
        dropout=0.1,
    )
