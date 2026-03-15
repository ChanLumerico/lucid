U-Net 3D
========
|convnet-badge| |segmentation-convnet-badge|

.. autoclass:: lucid.models.UNet3d

`UNet3d` is a configurable 3D encoder-decoder segmentation model with skip
connections between encoder and decoder stages. It shares the same
:class:`lucid.models.UNetConfig` interface as :class:`lucid.models.UNet2d` but
operates on volumetric inputs :math:`(N, C, D, H, W)`, replacing all 2D
convolutions, normalizations, pooling layers, and upsampling operations with
their 3D counterparts.

Typical use cases include medical image segmentation (CT/MRI volumes), video
feature extraction, and any task requiring dense prediction over 3D spatial data.

Class Signature
---------------

.. code-block:: python

    class UNet3d(nn.Module):
        def __init__(self, config: UNetConfig) -> None

Parameters
----------

- **config** (*UNetConfig*):
  Model configuration describing the encoder/decoder stage layout, block type,
  skip merge behavior, sampling strategy, and segmentation output space.
  When `upsample_mode="bilinear"` is passed, it is automatically remapped to
  `"trilinear"` for 3D compatibility.

Methods
-------

.. automethod:: lucid.models.UNet3d.forward

Examples
--------

**Build a Basic U-Net 3D**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.UNetConfig.from_channels(
        in_channels=1,
        out_channels=2,
        channels=(32, 64, 128, 256),
        num_blocks=2,
    )
    model = models.UNet3d(cfg)

    # (batch, channels, depth, height, width)
    x = lucid.random.randn(1, 1, 64, 128, 128)
    logits = model(x)
    print(logits.shape)  # (1, 2, 64, 128, 128)

**Build a U-Net 3D with Trilinear Upsampling and Deep Supervision**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.UNetConfig(
        in_channels=1,
        out_channels=3,
        encoder_stages=[
            models.UNetStageConfig(channels=32, num_blocks=2),
            models.UNetStageConfig(channels=64, num_blocks=2, use_attention=True),
            models.UNetStageConfig(channels=128, num_blocks=3),
            models.UNetStageConfig(channels=256, num_blocks=3),
        ],
        block="res",
        norm="group",
        act="silu",
        upsample_mode="trilinear",
        deep_supervision=True,
    )
    model = models.UNet3d(cfg)

    x = lucid.random.randn(1, 1, 32, 64, 64)
    out = model(x)
    print(out["out"].shape)  # (1, 3, 32, 64, 64)
    print(len(out["aux"]))   # 2

Notes
-----

- `UNet3d` expects volumetric tensors with shape :math:`(N, C, D, H, W)`.
  For 2D image inputs :math:`(N, C, H, W)`, use :class:`lucid.models.UNet2d`.
- Passing `upsample_mode="bilinear"` is accepted and silently remapped to
  `"trilinear"`, enabling existing 2D configs to be reused for 3D models.
- The current implementation supports `block="basic"` and `block="res"`.
- When `deep_supervision=False`, :meth:`lucid.models.UNet3d.forward` returns a
  single output tensor. When enabled, it returns a dictionary with `out` and
  `aux` predictions, identical to the 2D variant.
