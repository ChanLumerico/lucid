ResUNet 3D
==========
|convnet-badge| |segmentation-convnet-badge|

.. autoclass:: lucid.models.ResUNet3d

`ResUNet3d` is a residual variant of :class:`lucid.models.UNet3d` that reuses the
same configurable 3D encoder-decoder structure but forces residual blocks
throughout the network. It is useful for volumetric segmentation tasks where
deeper stage stacks or stronger gradient flow are desirable.

Class Signature
---------------

.. code-block:: python

    class ResUNet3d(UNet3d):
        def __init__(self, config: UNetConfig) -> None

Parameters
----------

- **config** (*UNetConfig*):
  U-Net configuration describing the encoder/decoder stage layout, channel
  widths, sampling strategy, and segmentation output space. `ResUNet3d`
  reuses this config and enforces `block="res"`.

Methods
-------

.. automethod:: lucid.models.ResUNet3d.forward

Examples
--------

**Build a ResUNet3d from `from_channels`**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.UNetConfig.from_channels(
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256),
        num_blocks=(2, 2, 2, 3),
        norm="group",
        act="silu",
        upsample_mode="trilinear",
    )
    model = models.ResUNet3d(cfg)

    x = lucid.random.randn(1, 1, 64, 128, 128)
    logits = model(x)
    print(logits.shape)  # (1, 2, 64, 128, 128)

**Build a Custom ResUNet3d with Stage-Level Configuration**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.UNetConfig(
        in_channels=1,
        out_channels=4,
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
    model = models.ResUNet3d(cfg)

    x = lucid.random.randn(1, 1, 32, 64, 64)
    out = model(x)
    print(out["out"].shape)  # (1, 4, 32, 64, 64)
    print(len(out["aux"]))   # 2

Notes
-----

- `ResUNet3d` reuses :class:`lucid.models.UNetConfig` and
  :class:`lucid.models.UNetStageConfig`; there is no separate config class.
- If `config.block` is not already `res`, :class:`lucid.models.ResUNet3d`
  overrides it internally.
- `ResUNet3d` expects volumetric tensors with shape :math:`(N, C, D, H, W)`.
  For 2D inputs, use :class:`lucid.models.ResUNet2d`.
