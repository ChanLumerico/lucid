Attention U-Net
===============
|convnet-badge| |segmentation-convnet-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    AttentionUNetConfig.rst
    AttentionUNetGateConfig.rst

.. autoclass:: lucid.models.AttentionUNet

`AttentionUNet` is a paper-faithful attention-gated variant of
:class:`lucid.models.UNet`. It preserves the encoder-decoder segmentation
structure of U-Net but inserts additive attention gates on skip connections so
decoder-side gating features can suppress irrelevant encoder responses before
concatenation.

 Oktay, Ozan, et al. "Attention U-Net: Learning Where to Look for the
 Pancreas." *arXiv preprint arXiv:1804.03999* (2018).

Class Signature
---------------

.. code-block:: python

    class AttentionUNet(UNet):
        def __init__(self, config: AttentionUNetConfig) -> None

Parameters
----------

- **config** (*AttentionUNetConfig*):
  Attention U-Net configuration describing the encoder/decoder stage layout,
  skip-gating strategy, and segmentation output space.

Methods
-------

.. automethod:: lucid.models.AttentionUNet.forward

Examples
--------

**Build a Paper-Style Attention U-Net**

.. code-block:: python

    import lucid
    import lucid.models as models

    cfg = models.AttentionUNetConfig.from_channels(
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256),
        num_blocks=2,
    )
    model = models.AttentionUNet(cfg)

    x = lucid.random.randn(2, 1, 128, 128)
    out = model(x)
    print(out["out"].shape)
    print(len(out["aux"]))

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
    model = models.AttentionUNet(cfg)

Notes
-----

- The implementation is 2D-only and expects tensors with shape
  :math:`(N, C, H, W)`.
- It is intentionally constrained to the paper-faithful setting:
  `block="basic"`, `skip_merge="concat"`, additive gates, sigmoid attention
  coefficients, and grid attention.
- The current default enables deep supervision, so
  :meth:`lucid.models.AttentionUNet.forward` returns a dictionary with `out`
  and `aux` predictions unless `deep_supervision=False`.
