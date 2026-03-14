ResUNet
========
|convnet-badge| |segmentation-convnet-badge|

.. autoclass:: lucid.models.ResUNet

`ResUNet` is a residual variant of :class:`lucid.models.UNet` that reuses the
same configurable encoder-decoder structure but forces residual blocks
throughout the network. It is useful when deeper stage stacks or stronger
gradient flow are desirable while keeping the same segmentation-oriented U-Net
topology.

Class Signature
---------------

.. code-block:: python

    class ResUNet(UNet):
        def __init__(self, config: UNetConfig) -> None

Parameters
----------

- **config** (*UNetConfig*):
  U-Net configuration describing the encoder/decoder stage layout, channel
  widths, sampling strategy, and segmentation output space. `ResUNet`
  reuses this config and enforces `block="res"`.

Methods
-------

.. automethod:: lucid.models.ResUNet.forward

Examples
--------

**Build a Residual U-Net from `from_channels`**

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
    model = models.ResUNet(cfg)

    x = lucid.random.randn(1, 3, 256, 256)
    logits = model(x)
    print(logits.shape)

**Build a Custom ResUNet with Stage-Level Configuration**

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
    model = models.ResUNet(cfg)

    x = lucid.random.randn(2, 1, 256, 256)
    out = model(x)
    print(out["out"].shape)
    print(len(out["aux"]))

Notes
-----

- `ResUNet` reuses :class:`lucid.models.UNetConfig` and
  :class:`lucid.models.UNetStageConfig`; there is no separate config class.
- If `config.block` is not already `res`, :class:`lucid.models.ResUNet`
  overrides it internally.
- The current implementation remains 2D-only and expects input tensors with
  shape :math:`(N, C, H, W)`.
