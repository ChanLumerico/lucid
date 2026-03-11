PVTConfig
=========

.. autoclass:: lucid.models.PVTConfig

`PVTConfig` stores the stage layout and classifier settings used by
:class:`lucid.models.PVT`. It defines the image size, initial patch embedding,
hierarchical embedding widths, attention heads, depth schedule, and spatial
reduction ratios for the four-stage PVT encoder.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class PVTConfig:
        img_size: int = 224
        num_classes: int = 1000
        patch_size: int = 4
        in_channels: int = 3
        embed_dims: tuple[int, ...] | list[int] = (64, 128, 256, 512)
        num_heads: tuple[int, ...] | list[int] = (1, 2, 4, 8)
        mlp_ratios: tuple[float, ...] | list[float] = (4.0, 4.0, 4.0, 4.0)
        qkv_bias: bool = False
        qk_scale: float | None = None
        drop_rate: float = 0.0
        attn_drop_rate: float = 0.0
        drop_path_rate: float = 0.0
        norm_layer: type[nn.Module] = nn.LayerNorm
        depths: tuple[int, ...] | list[int] = (3, 4, 6, 3)
        sr_ratios: tuple[int, ...] | list[int] = (8, 4, 2, 1)

Parameters
----------

- **img_size** (*int*):
  Input image size. PVT assumes square inputs.
- **num_classes** (*int*):
  Number of output classes. Set to `0` to keep the identity head.
- **patch_size** (*int*):
  Patch size used by the first embedding stage.
- **in_channels** (*int*):
  Number of input image channels.
- **embed_dims**, **num_heads**, **mlp_ratios**, **depths**, **sr_ratios**:
  Four-stage embedding widths, head counts, feedforward ratios, block counts,
  and spatial reduction ratios.
- **qkv_bias** (*bool*):
  Whether query, key, and value projections use bias.
- **qk_scale** (*float | None*):
  Optional attention scaling override.
- **drop_rate**, **attn_drop_rate**, **drop_path_rate**:
  Dropout and stochastic depth settings.
- **norm_layer** (*type[nn.Module]*):
  Normalization layer used throughout the model.

Validation
----------

- `img_size`, `patch_size`, and `in_channels` must be greater than `0`.
- `num_classes` must be greater than or equal to `0`.
- `embed_dims`, `num_heads`, `mlp_ratios`, `depths`, and `sr_ratios` must each
  contain exactly four values.
- Embedding widths, head counts, depths, and spatial reduction ratios must be positive.
- Each embedding width must be divisible by the corresponding head count.
- Dropout rates must each be in `[0, 1)`.
- The configured image size and patch size must leave enough spatial resolution
  for all four PVT stages.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.PVTConfig(
        img_size=32,
        num_classes=10,
        patch_size=4,
        in_channels=1,
        embed_dims=(8, 16, 32, 64),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(2.0, 2.0, 2.0, 2.0),
        depths=(1, 1, 1, 1),
        sr_ratios=(8.0, 4.0, 2.0, 1.0),
        drop_path_rate=0.0,
    )
    model = models.PVT(config)
