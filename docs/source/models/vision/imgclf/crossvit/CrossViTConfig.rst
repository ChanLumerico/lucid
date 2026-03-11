CrossViTConfig
==============

.. autoclass:: lucid.models.CrossViTConfig

`CrossViTConfig` stores the dual-branch stage layout and classifier settings used
by :class:`lucid.models.CrossViT`. It defines the branch image sizes, patch
sizes, embedding widths, multi-scale block depths, attention heads, and whether
the dagger-style multi-convolution patch embedding is enabled.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class CrossViTConfig:
        img_size: tuple[int, int] | list[int] = (224, 224)
        patch_size: tuple[int, int] | list[int] = (12, 16)
        in_channels: int = 3
        num_classes: int = 1000
        embed_dim: tuple[int, int] | list[int] = (192, 384)
        depth: tuple[tuple[int, int, int], ...] | list[list[int]] = ((1, 3, 1), (1, 3, 1), (1, 3, 1))
        num_heads: tuple[int, int] | list[int] = (6, 12)
        mlp_ratio: tuple[float, float, float] | list[float] = (2.0, 2.0, 4.0)
        qkv_bias: bool = False
        qk_scale: float | None = None
        drop_rate: float = 0.0
        attn_drop_rate: float = 0.0
        drop_path_rate: float = 0.0
        norm_layer: type[nn.Module] = nn.LayerNorm
        multi_conv: bool = False

Parameters
----------

- **img_size**:
  Input resolution for the two CrossViT branches.
- **patch_size**:
  Patch size for each branch.
- **in_channels** (*int*):
  Number of input image channels.
- **num_classes** (*int*):
  Number of output classes. Set to `0` to keep identity heads.
- **embed_dim**:
  Embedding width for each branch.
- **depth**:
  Sequence of multi-scale block specs. Each entry contains two branch depths and
  one fusion depth.
- **num_heads**:
  Attention head counts for the two branches.
- **mlp_ratio**:
  MLP expansion ratios for the two branches and the cross-fusion path.
- **qkv_bias** (*bool*):
  Whether query, key, and value projections use bias.
- **qk_scale** (*float | None*):
  Optional attention scaling override.
- **drop_rate**, **attn_drop_rate**, **drop_path_rate**:
  Dropout and stochastic depth settings.
- **norm_layer** (*type[nn.Module]*):
  Normalization layer used throughout the model.
- **multi_conv** (*bool*):
  Whether to use the dagger-style multi-convolution patch embedding.

Validation
----------

- `img_size`, `patch_size`, `embed_dim`, and `num_heads` must each contain exactly two positive integers.
- `in_channels` must be greater than `0`.
- `num_classes` must be greater than or equal to `0`.
- `mlp_ratio` must contain exactly three positive values.
- `depth` must contain at least one stage spec, and each spec must contain two
  positive branch depths plus one non-negative fusion depth.
- Each embedding width must be divisible by the corresponding head count.
- Dropout rates must each be in `[0, 1)`.
- If `multi_conv=True`, every patch size must be either `12` or `16`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.CrossViTConfig(
        img_size=(32, 32),
        patch_size=(8, 16),
        in_channels=1,
        num_classes=10,
        embed_dim=(32, 64),
        depth=((1, 1, 0), (1, 1, 0)),
        num_heads=(4, 4),
        mlp_ratio=(2.0, 2.0, 1.0),
        drop_path_rate=0.0,
    )
    model = models.CrossViT(config)
