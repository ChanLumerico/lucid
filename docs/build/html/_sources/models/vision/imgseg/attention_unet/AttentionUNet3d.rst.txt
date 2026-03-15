Attention U-Net 3D
==================
|convnet-badge| |segmentation-convnet-badge|

.. autoclass:: lucid.models.AttentionUNet3d

`AttentionUNet3d` is a 3D attention-gated variant of
:class:`lucid.models.UNet3d`. It extends the volumetric encoder-decoder
structure with additive attention gates on skip connections, enabling
decoder-side gating signals to suppress irrelevant encoder responses before
concatenation — the same mechanism as :class:`lucid.models.AttentionUNet2d`
but applied over :math:`(D, H, W)` spatial dimensions.

Typical use cases include volumetric medical image segmentation (CT/MRI) where
attention over 3D anatomical structures improves localization accuracy.

 Oktay, Ozan, et al. "Attention U-Net: Learning Where to Look for the
 Pancreas." *arXiv preprint arXiv:1804.03999* (2018).

Class Signature
---------------

.. code-block:: python

    class AttentionUNet3d(UNet3d):
        def __init__(self, config: AttentionUNetConfig) -> None

Parameters
----------

- **config** (*AttentionUNetConfig*):
  Attention U-Net configuration describing the encoder/decoder stage layout,
  skip-gating strategy, and segmentation output space. Set
  `upsample_mode="trilinear"` for 3D-native upsampling; passing `"bilinear"`
  is also accepted and automatically remapped.

Methods
-------

.. automethod:: lucid.models.AttentionUNet3d.forward

Examples
--------

**Build a Volumetric Attention U-Net**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.AttentionUNetConfig.from_channels(
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256),
        num_blocks=2,
        upsample_mode="trilinear",
    )
    model = models.AttentionUNet3d(cfg)

    # (batch, channels, depth, height, width)
    x = lucid.random.randn(1, 1, 32, 64, 64)
    out = model(x)
    print(out["out"].shape)  # (1, 3, 32, 64, 64)
    print(len(out["aux"]))   # 2

**Customize Gate Widths for 3D**

.. code-block:: python

    import lucid.models as models

    cfg = models.AttentionUNetConfig(
        in_channels=1,
        out_channels=2,
        encoder_stages=[
            models.UNetStageConfig(channels=32, num_blocks=2),
            models.UNetStageConfig(channels=64, num_blocks=2),
            models.UNetStageConfig(channels=128, num_blocks=2),
            models.UNetStageConfig(channels=256, num_blocks=2),
        ],
        upsample_mode="trilinear",
        attention=models.AttentionUNetGateConfig(
            inter_channels=(32, 64, 64),
        ),
    )
    model = models.AttentionUNet3d(cfg)

Notes
-----

- `AttentionUNet3d` expects volumetric tensors with shape
  :math:`(N, C, D, H, W)`. For 2D image inputs, use
  :class:`lucid.models.AttentionUNet2d`.
- The gate resampling mode follows `AttentionUNetGateConfig.resample_mode`.
  Passing `"bilinear"` is automatically remapped to `"trilinear"` inside the
  3D attention gate for compatibility with 2D configs.
- It is intentionally constrained to the paper-faithful setting:
  `block="basic"`, `skip_merge="concat"`, additive gates, sigmoid attention
  coefficients, and grid attention.
- The current default enables deep supervision, so
  :meth:`lucid.models.AttentionUNet3d.forward` returns a dictionary with `out`
  and `aux` predictions unless `deep_supervision=False`.
