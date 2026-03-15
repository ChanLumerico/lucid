UNetConfig
==========

.. autoclass:: lucid.models.UNetConfig

`UNetConfig` stores the architectural choices used by
:class:`lucid.models.UNet2d` and :class:`lucid.models.UNet3d`. It defines the
stage layout for the encoder and decoder along with block type, normalization,
activation, skip merge behavior, sampling strategy, and output head options.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class UNetConfig:
        in_channels: int
        out_channels: int
        encoder_stages: tuple[UNetStageConfig, ...] | list[UNetStageConfig]
        decoder_stages: tuple[UNetStageConfig, ...] | list[UNetStageConfig] | None = None
        bottleneck: UNetStageConfig | None = None
        block: Literal["basic", "res", "convnext"] = "basic"
        norm: Literal["batch", "group", "instance", "none"] = "batch"
        act: Literal["relu", "leaky_relu", "gelu", "silu"] = "relu"
        skip_merge: Literal["concat", "add"] = "concat"
        downsample_mode: Literal["conv", "maxpool", "avgpool"] = "conv"
        upsample_mode: Literal["transpose", "bilinear", "trilinear", "nearest"] = "transpose"
        stem_channels: int | None = None
        final_kernel_size: int = 1
        deep_supervision: bool = False
        align_corners: bool = False
        bias: bool | None = None

Parameters
----------

- **in_channels** (*int*): Number of channels in the input image tensor.
- **out_channels** (*int*): Number of channels predicted by the final segmentation head.
- **encoder_stages** (*tuple[UNetStageConfig, ...] | list[UNetStageConfig]*):
  Stage specifications for the encoder path.
- **decoder_stages** (*tuple[UNetStageConfig, ...] | list[UNetStageConfig] | None*):
  Stage specifications for the decoder path. If `None`, the decoder is mirrored
  automatically from the encoder except for the deepest stage.
- **bottleneck** (*UNetStageConfig | None*): Bottleneck stage inserted between
  the encoder and decoder. If `None`, a default bottleneck is inferred from the
  deepest encoder stage.
- **block** (*Literal["basic", "res", "convnext"]*): Block family identifier.
  The current implementation supports `basic` and `res`.
- **norm** (*Literal["batch", "group", "instance", "none"]*): Normalization
  layer used inside blocks and attention blocks.
- **act** (*Literal["relu", "leaky_relu", "gelu", "silu"]*): Activation
  function used inside blocks and the stem.
- **skip_merge** (*Literal["concat", "add"]*): Skip connection merge strategy.
- **downsample_mode** (*Literal["conv", "maxpool", "avgpool"]*): Downsampling
  operation used between encoder stages.
- **upsample_mode** (*Literal["transpose", "bilinear", "trilinear", "nearest"]*):
  Upsampling operation used between decoder stages. Use `"trilinear"` (or
  `"bilinear"`, which is automatically remapped) for :class:`lucid.models.UNet3d`.
- **stem_channels** (*int | None*): Output width of the input stem. If `None`,
  the first encoder stage width is used.
- **final_kernel_size** (*int*): Kernel size of the final output projection.
- **deep_supervision** (*bool*): Whether to attach auxiliary output heads to
  intermediate decoder stages.
- **align_corners** (*bool*): Value passed to interpolation operations that use
  corner alignment semantics.
- **bias** (*bool | None*): Whether convolution layers use bias terms. If `None`,
  this is inferred from the normalization choice.

Usage
-----

.. code-block:: python

    import lucid.models as models

    cfg = models.UNetConfig.from_channels(
        in_channels=3,
        out_channels=2,
        channels=(32, 64, 128, 256),
        num_blocks=(2, 2, 2, 3),
        block="res",
        norm="group",
        upsample_mode="bilinear",
    )

    model = models.UNet2d(cfg)
