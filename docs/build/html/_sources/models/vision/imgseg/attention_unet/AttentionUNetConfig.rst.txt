AttentionUNetConfig
===================

.. autoclass:: lucid.models.AttentionUNetConfig

`AttentionUNetConfig` extends :class:`lucid.models.UNetConfig` with
paper-faithful attention gate settings for :class:`lucid.models.AttentionUNet2d`
and :class:`lucid.models.AttentionUNet3d`. It keeps the same configurable stage
layout while constraining the block and skip-merge behavior to match the
Attention U-Net formulation.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class AttentionUNetConfig(UNetConfig):
        block: Literal["basic"] = "basic"
        norm: Literal["batch", "group", "instance", "none"] = "batch"
        act: Literal["relu", "leaky_relu", "gelu", "silu"] = "relu"
        skip_merge: Literal["concat"] = "concat"
        downsample_mode: Literal["conv", "maxpool", "avgpool"] = "maxpool"
        upsample_mode: Literal["transpose", "bilinear", "trilinear", "nearest"] = "bilinear"
        deep_supervision: bool = True
        attention: AttentionUNetGateConfig = field(default_factory=AttentionUNetGateConfig)

Parameters
----------

- **in_channels** (*int*): Number of channels in the input image tensor.
- **out_channels** (*int*): Number of channels predicted by the final segmentation head.
- **encoder_stages** (*tuple[UNetStageConfig, ...] | list[UNetStageConfig]*):
  Stage specifications for the encoder path.
- **decoder_stages** (*tuple[UNetStageConfig, ...] | list[UNetStageConfig] | None*):
  Stage specifications for the decoder path. If `None`, the decoder is mirrored
  automatically from the encoder except for the deepest stage.
- **bottleneck** (*UNetStageConfig | None*): Bottleneck stage between the encoder
  and decoder. If `None`, a default bottleneck is inferred.
- **norm** (*Literal["batch", "group", "instance", "none"]*): Normalization
  layer used inside blocks.
- **act** (*Literal["relu", "leaky_relu", "gelu", "silu"]*): Activation
  function used inside blocks and the stem.
- **downsample_mode** (*Literal["conv", "maxpool", "avgpool"]*): Downsampling
  operation used between encoder stages. The paper-style default is `maxpool`.
- **upsample_mode** (*Literal["transpose", "bilinear", "trilinear", "nearest"]*):
  Upsampling operation used between decoder stages. The paper-style 2D default
  is `bilinear`; use `"trilinear"` for :class:`lucid.models.AttentionUNet3d`.
- **deep_supervision** (*bool*): Whether to attach auxiliary output heads to
  intermediate decoder stages.
- **attention** (*AttentionUNetGateConfig*): Attention gate settings applied to
  decoder-side skip connections.

Inherited Parameters
--------------------

- `stem_channels`
- `final_kernel_size`
- `align_corners`
- `bias`

Usage
-----

**2D (image segmentation)**

.. code-block:: python

    import lucid.models as models

    cfg = models.AttentionUNetConfig.from_channels(
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256),
        num_blocks=2,
        attention=models.AttentionUNetGateConfig(
            inter_channels=(32, 64, 64),
        ),
    )

    model = models.AttentionUNet2d(cfg)

**3D (volumetric segmentation)**

.. code-block:: python

    import lucid.models as models

    cfg = models.AttentionUNetConfig.from_channels(
        in_channels=1,
        out_channels=3,
        channels=(32, 64, 128, 256),
        num_blocks=2,
        upsample_mode="trilinear",
        attention=models.AttentionUNetGateConfig(
            inter_channels=(32, 64, 64),
        ),
    )

    model = models.AttentionUNet3d(cfg)
