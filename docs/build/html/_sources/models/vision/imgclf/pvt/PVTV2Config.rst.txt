PVTV2Config
===========

.. autoclass:: lucid.models.PVTV2Config

`PVTV2Config` stores the stage layout and classifier settings used by
:class:`lucid.models.PVT_V2`. It defines the overlap patch embedding, stage
depths, attention heads, spatial reduction ratios, and whether linear
attention is enabled.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class PVTV2Config:
        img_size: int = 224
        patch_size: int = 7
        in_channels: int = 3
        num_classes: int = 1000
        embed_dims: tuple[int, ...] | list[int] = (64, 128, 256, 512)
        num_heads: tuple[int, ...] | list[int] = (1, 2, 4, 8)
        mlp_ratios: tuple[int, ...] | list[int] = (4, 4, 4, 4)
        qkv_bias: bool = False
        qk_scale: float | None = None
        drop_rate: float = 0.0
        attn_drop_rate: float = 0.0
        drop_path_rate: float = 0.0
        norm_layer: type[nn.Module] = nn.LayerNorm
        depths: tuple[int, ...] | list[int] = (3, 4, 6, 3)
        sr_ratios: tuple[int, ...] | list[int] = (8, 4, 2, 1)
        num_stages: int = 4
        linear: bool = False

Parameters
----------

- **img_size** (*int*):
  Input image size. PVT-v2 assumes square inputs.
- **patch_size** (*int*):
  Patch size for the first overlap patch embedding stage.
- **in_channels** (*int*):
  Number of input image channels.
- **num_classes** (*int*):
  Number of output classes. Set to `0` to keep the identity head.
- **embed_dims**, **num_heads**, **mlp_ratios**, **depths**, **sr_ratios**:
  Per-stage embedding widths, head counts, feedforward ratios, block counts,
  and spatial reduction ratios.
- **qkv_bias** (*bool*):
  Whether query, key, and value projections use bias.
- **qk_scale** (*float | None*):
  Optional attention scaling override.
- **drop_rate**, **attn_drop_rate**, **drop_path_rate**:
  Dropout and stochastic depth settings.
- **norm_layer** (*type[nn.Module]*):
  Normalization layer used throughout the model.
- **num_stages** (*int*):
  Number of hierarchical stages.
- **linear** (*bool*):
  Whether to use the linear attention path in PVT-v2 blocks.

Validation
----------

- `img_size`, `patch_size`, `in_channels`, and `num_stages` must be greater than `0`.
- `patch_size` must be greater than `4` for the first overlap patch embedding.
- `num_classes` must be greater than or equal to `0`.
- `embed_dims`, `num_heads`, `mlp_ratios`, `depths`, and `sr_ratios` must each
  contain exactly `num_stages` values.
- Embedding widths, head counts, depths, and spatial reduction ratios must be positive.
- Each embedding width must be divisible by the corresponding head count.
- Dropout rates must each be in `[0, 1)`.
- The configured image size must leave enough spatial resolution for all stages.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.PVTV2Config(
        img_size=32,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dims=(8, 16, 32, 64),
        num_heads=(1, 2, 4, 8),
        mlp_ratios=(2, 2, 2, 2),
        depths=(1, 1, 1, 1),
        sr_ratios=(8, 4, 2, 1),
        drop_path_rate=0.0,
    )
    model = models.PVT_V2(config)
