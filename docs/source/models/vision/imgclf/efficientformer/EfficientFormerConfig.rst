EfficientFormerConfig
=====================

.. autoclass:: lucid.models.EfficientFormerConfig

`EfficientFormerConfig` stores the stage layout and classifier settings used by
:class:`lucid.models.EfficientFormer`. It defines the stage depths, embedding
widths, downsampling schedule, number of transformer-style blocks in the final
stage, and classifier/dropout behavior.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class EfficientFormerConfig:
        depths: tuple[int, ...] | list[int]
        embed_dims: tuple[int, ...] | list[int]
        in_channels: int = 3
        num_classes: int = 1000
        global_pool: bool = True
        downsamples: tuple[bool, ...] | list[bool] | None = None
        num_vit: int = 0
        mlp_ratios: float = 4.0
        pool_size: int = 3
        layer_scale_init_value: float = 1e-5
        act_layer: type[nn.Module] = nn.GELU
        norm_layer: type[nn.Module] = nn.BatchNorm2d
        norm_layer_cl: type[nn.Module] = nn.LayerNorm
        drop_rate: float = 0.0
        proj_drop_rate: float = 0.0
        drop_path_rate: float = 0.0

Parameters
----------

- **depths**:
  Number of blocks in each stage.
- **embed_dims**:
  Embedding width for each stage.
- **in_channels** (*int*):
  Number of input image channels.
- **num_classes** (*int*):
  Number of output classes. Set to `0` to keep an identity classifier.
- **global_pool** (*bool*):
  Whether to average the final token sequence before classification.
- **downsamples**:
  Optional explicit per-stage downsampling schedule.
- **num_vit** (*int*):
  Number of transformer-style blocks used in the final stage.
- **mlp_ratios** (*float*):
  Hidden width multiplier for MLP layers.
- **pool_size** (*int*):
  Pooling kernel size used by convolutional MetaBlocks.
- **layer_scale_init_value** (*float*):
  Initial value for layer scale parameters.
- **act_layer**, **norm_layer**, **norm_layer_cl**:
  Activation and normalization modules used by the stem, convolutional blocks,
  and final token blocks.
- **drop_rate**, **proj_drop_rate**, **drop_path_rate**:
  Head dropout, projection dropout, and stochastic depth settings.

Validation
----------

- `depths` must contain at least one positive integer.
- `embed_dims` must contain one positive width per stage.
- `in_channels` must be greater than `0`.
- `num_classes` must be greater than or equal to `0`.
- If provided, `downsamples` must match the number of stages.
- `num_vit` must be greater than or equal to `0`.
- `mlp_ratios` and `pool_size` must be greater than `0`.
- `layer_scale_init_value` must be non-negative.
- `drop_rate`, `proj_drop_rate`, and `drop_path_rate` must each be in `[0, 1)`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.EfficientFormerConfig(
        depths=(1, 1, 1, 1),
        embed_dims=(16, 32, 48, 64),
        in_channels=1,
        num_classes=10,
        num_vit=1,
        mlp_ratios=2.0,
    )
    model = models.EfficientFormer(config)
