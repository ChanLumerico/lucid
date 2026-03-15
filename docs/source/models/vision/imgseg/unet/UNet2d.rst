U-Net 2D
========
|convnet-badge| |segmentation-convnet-badge|

.. autoclass:: lucid.models.UNet2d

`UNet2d` is a configurable 2D encoder-decoder segmentation model with skip
connections between encoder and decoder stages. The implementation supports
custom stage widths and depths, optional per-stage attention, different
normalization and activation choices, multiple downsampling and upsampling
strategies, and optional deep supervision heads.

For volumetric inputs with shape :math:`(N, C, D, H, W)`, see
:class:`lucid.models.UNet3d`.

Class Signature
---------------

.. code-block:: python

    class UNet2d(nn.Module):
        def __init__(self, config: UNetConfig) -> None

Parameters
----------

- **config** (*UNetConfig*):
  Model configuration describing the encoder/decoder stage layout, block type,
  skip merge behavior, sampling strategy, and segmentation output space.

Methods
-------

.. automethod:: lucid.models.UNet2d.forward

Examples
--------

**Build a Basic U-Net 2D**

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
    model = models.UNet2d(cfg)

    x = lucid.random.randn(1, 3, 256, 256)
    logits = model(x)
    print(logits.shape)  # (1, 1, 256, 256)

**Build a Residual U-Net 2D with Deep Supervision**

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
    model = models.UNet2d(cfg)

    x = lucid.random.randn(2, 3, 256, 256)
    out = model(x)
    print(out["out"].shape)   # (2, 4, 256, 256)
    print(len(out["aux"]))    # 2

Notes
-----

- `UNet2d` expects image tensors with shape :math:`(N, C, H, W)`.
  For 3D volumetric inputs :math:`(N, C, D, H, W)`, use
  :class:`lucid.models.UNet3d`.
- The current implementation supports `block="basic"` and `block="res"`.
  Although the config type reserves `convnext`, that block is not implemented yet.
- When `deep_supervision=False`, :meth:`lucid.models.UNet2d.forward` returns a
  single output tensor. When enabled, it returns a dictionary with `out` and
  `aux` predictions.
