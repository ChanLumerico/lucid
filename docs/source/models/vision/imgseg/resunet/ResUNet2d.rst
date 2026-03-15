ResUNet 2D
==========
|convnet-badge| |segmentation-convnet-badge|

.. autoclass:: lucid.models.ResUNet2d

`ResUNet2d` is a residual variant of :class:`lucid.models.UNet2d` that reuses the
same configurable encoder-decoder structure but forces residual blocks
throughout the network. It is useful when deeper stage stacks or stronger
gradient flow are desirable while keeping the same segmentation-oriented U-Net
topology.

For volumetric inputs with shape :math:`(N, C, D, H, W)`, see
:class:`lucid.models.ResUNet3d`.

Class Signature
---------------

.. code-block:: python

    class ResUNet2d(UNet2d):
        def __init__(self, config: UNetConfig) -> None

Parameters
----------

- **config** (*UNetConfig*):
  U-Net configuration describing the encoder/decoder stage layout, channel
  widths, sampling strategy, and segmentation output space. `ResUNet2d`
  reuses this config and enforces `block="res"`.

Methods
-------

.. automethod:: lucid.models.ResUNet2d.forward

Examples
--------

**Build a ResUNet2d from `from_channels`**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.UNetConfig.from_channels(
        in_channels=3,
        out_channels=2,
        channels=(64, 128, 256, 512),
        num_blocks=(2, 2, 2, 3),
        norm="group",
        act="silu",
        upsample_mode="bilinear",
    )
    model = models.ResUNet2d(cfg)

    x = lucid.random.randn(1, 3, 256, 256)
    logits = model(x)
    print(logits.shape)  # (1, 2, 256, 256)

**Build a Custom ResUNet2d with Stage-Level Configuration**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.UNetConfig(
        in_channels=1,
        out_channels=3,
        encoder_stages=[
            models.UNetStageConfig(channels=32, num_blocks=2),
            models.UNetStageConfig(channels=64, num_blocks=2),
            models.UNetStageConfig(channels=128, num_blocks=3, use_attention=True),
            models.UNetStageConfig(channels=256, num_blocks=3),
        ],
        norm="group",
        act="silu",
        downsample_mode="conv",
        upsample_mode="transpose",
        deep_supervision=True,
    )
    model = models.ResUNet2d(cfg)

    x = lucid.random.randn(2, 1, 256, 256)
    out = model(x)
    print(out["out"].shape)  # (2, 3, 256, 256)
    print(len(out["aux"]))   # 2

Notes
-----

- `ResUNet2d` reuses :class:`lucid.models.UNetConfig` and
  :class:`lucid.models.UNetStageConfig`; there is no separate config class.
- If `config.block` is not already `res`, :class:`lucid.models.ResUNet2d`
  overrides it internally.
- `ResUNet2d` expects input tensors with shape :math:`(N, C, H, W)`.
  For 3D volumetric inputs, use :class:`lucid.models.ResUNet3d`.
