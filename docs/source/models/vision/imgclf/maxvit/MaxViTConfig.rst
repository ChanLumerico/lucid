MaxViTConfig
============

.. autoclass:: lucid.models.MaxViTConfig

`MaxViTConfig` stores the stage layout and attention settings used by
:class:`lucid.models.MaxViT`. It defines the stem width, per-stage depths and
channels, shared attention head count, window size, dropout settings, and
classifier size.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class MaxViTConfig:
        in_channels: int = 3
        depths: tuple[int, ...] | list[int] = (2, 2, 5, 2)
        channels: tuple[int, ...] | list[int] = (64, 128, 256, 512)
        num_classes: int = 1000
        embed_dim: int = 64
        num_heads: int = 32
        grid_window_size: tuple[int, int] | list[int] = (7, 7)
        attn_drop: float = 0.0
        drop: float = 0.0
        drop_path: float = 0.0
        mlp_ratio: float = 4.0
        act_layer: type[nn.Module] = nn.GELU
        norm_layer: type[nn.Module] = nn.BatchNorm2d
        norm_layer_tf: type[nn.Module] = nn.LayerNorm

Parameters
----------

- **in_channels** (*int*):
  Number of input image channels.
- **depths**:
  Number of MaxViT blocks in each stage.
- **channels**:
  Output channel width for each stage.
- **num_classes** (*int*):
  Number of output classes. Set to `0` to keep an identity classifier.
- **embed_dim** (*int*):
  Width of the convolutional stem.
- **num_heads** (*int*):
  Shared attention head count for window and grid attention.
- **grid_window_size**:
  Window size used by both attention partitioning schemes.
- **attn_drop**, **drop**, **drop_path**:
  Attention, projection, and stochastic depth dropout settings.
- **mlp_ratio** (*float*):
  Hidden width multiplier for transformer MLP layers.
- **act_layer**, **norm_layer**, **norm_layer_tf**:
  Activation and normalization modules used by the stem, MBConv path, and
  transformer blocks.

Validation
----------

- `in_channels`, `embed_dim`, and `num_heads` must be greater than `0`.
- `depths` must contain at least one positive integer.
- `channels` must contain one positive width per stage.
- Each channel width must be divisible by `num_heads`.
- `num_classes` must be greater than or equal to `0`.
- `grid_window_size` must contain exactly two positive integers.
- `attn_drop`, `drop`, and `drop_path` must each be in `[0, 1)`.
- `mlp_ratio` must be greater than `0`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.MaxViTConfig(
        in_channels=1,
        depths=(1, 1),
        channels=(16, 32),
        num_classes=10,
        embed_dim=16,
        num_heads=4,
        grid_window_size=(1, 1),
        mlp_ratio=2.0,
    )
    model = models.MaxViT(config)
