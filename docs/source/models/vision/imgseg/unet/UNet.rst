UNet
====
|convnet-badge| |segmentation-convnet-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    UNetConfig.rst
    UNetStageConfig.rst

.. autoclass:: lucid.models.UNet

`UNet` is a configurable 2D encoder-decoder segmentation model with skip
connections between encoder and decoder stages. The implementation supports
custom stage widths and depths, optional per-stage attention, different
normalization and activation choices, multiple downsampling and upsampling
strategies, and optional deep supervision heads.

Class Signature
---------------

.. code-block:: python

    class UNet(nn.Module):
        def __init__(self, config: UNetConfig) -> None

Parameters
----------

- **config** (*UNetConfig*):
  Model configuration describing the encoder/decoder stage layout, block type,
  skip merge behavior, sampling strategy, and segmentation output space.

Methods
-------

.. automethod:: lucid.models.UNet.forward

Examples
--------

**Build a Basic U-Net**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.UNetConfig.from_channels(
        in_channels=3,
        out_channels=1,
        channels=(64, 128, 256, 512),
        num_blocks=2,
        block="basic",
    )
    model = models.UNet(cfg)

    x = lucid.random.randn(1, 3, 256, 256)
    logits = model(x)
    print(logits.shape)

**Build a Residual U-Net with Deep Supervision**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.UNetConfig(
        in_channels=3,
        out_channels=4,
        encoder_stages=[
            models.UNetStageConfig(channels=32, num_blocks=2),
            models.UNetStageConfig(channels=64, num_blocks=2, use_attention=True),
            models.UNetStageConfig(channels=128, num_blocks=3),
            models.UNetStageConfig(channels=256, num_blocks=3),
        ],
        block="res",
        norm="group",
        act="silu",
        upsample_mode="bilinear",
        deep_supervision=True,
    )
    model = models.UNet(cfg)

    x = lucid.random.randn(2, 3, 256, 256)
    out = model(x)
    print(out["out"].shape)
    print(len(out["aux"]))

Notes
-----

- The current implementation is 2D-only and expects image tensors with shape
  :math:`(N, C, H, W)`.
- The current implementation supports `block="basic"` and `block="res"`.
  Although the config type reserves `convnext`, that block is not implemented yet.
- When `deep_supervision=False`, :meth:`lucid.models.UNet.forward` returns a
  single output tensor. When enabled, it returns a dictionary with `out` and
  `aux` predictions.
