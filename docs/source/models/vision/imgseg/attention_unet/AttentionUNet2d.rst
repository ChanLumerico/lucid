Attention U-Net 2D
==================
|convnet-badge| |segmentation-convnet-badge|

.. autoclass:: lucid.models.AttentionUNet2d

`AttentionUNet2d` is a paper-faithful attention-gated variant of
:class:`lucid.models.UNet2d`. It preserves the encoder-decoder segmentation
structure of U-Net but inserts additive attention gates on skip connections so
decoder-side gating features can suppress irrelevant encoder responses before
concatenation.

For volumetric inputs with shape :math:`(N, C, D, H, W)`, see
:class:`lucid.models.AttentionUNet3d`.

 Oktay, Ozan, et al. "Attention U-Net: Learning Where to Look for the
 Pancreas." *arXiv preprint arXiv:1804.03999* (2018).

Class Signature
---------------

.. code-block:: python

    class AttentionUNet2d(UNet2d):
        def __init__(self, config: AttentionUNetConfig) -> None

Parameters
----------

- **config** (*AttentionUNetConfig*):
  Attention U-Net configuration describing the encoder/decoder stage layout,
  skip-gating strategy, and segmentation output space.

Methods
-------

.. automethod:: lucid.models.AttentionUNet2d.forward

Examples
--------

**Build a Paper-Style Attention U-Net 2D**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.AttentionUNetConfig.from_channels(
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256),
        num_blocks=2,
    )
    model = models.AttentionUNet2d(cfg)

    x = lucid.random.randn(2, 1, 128, 128)
    out = model(x)
    print(out["out"].shape)  # (2, 3, 128, 128)
    print(len(out["aux"]))   # 2

**Customize Gate Widths**

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
        attention=models.AttentionUNetGateConfig(
            inter_channels=(32, 64, 64),
        ),
    )
    model = models.AttentionUNet2d(cfg)

Notes
-----

- `AttentionUNet2d` expects image tensors with shape :math:`(N, C, H, W)`.
  For 3D volumetric inputs :math:`(N, C, D, H, W)`, use
  :class:`lucid.models.AttentionUNet3d`.
- It is intentionally constrained to the paper-faithful setting:
  `block="basic"`, `skip_merge="concat"`, additive gates, sigmoid attention
  coefficients, and grid attention.
- The current default enables deep supervision, so
  :meth:`lucid.models.AttentionUNet2d.forward` returns a dictionary with `out`
  and `aux` predictions unless `deep_supervision=False`.
