SwinTransformerConfig
=====================

.. autoclass:: lucid.models.SwinTransformerConfig

`SwinTransformerConfig` stores the hierarchical stage layout and classifier
settings used by :class:`lucid.models.SwinTransformer`. It defines the input
resolution, patch embedding setup, stage depths, attention heads, window size,
dropout rates, and positional embedding options.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class SwinTransformerConfig:
        img_size: int = 224
        patch_size: int = 4
        in_channels: int = 3
        num_classes: int = 1000
        embed_dim: int = 96
        depths: tuple[int, ...] | list[int] = (2, 2, 6, 2)
        num_heads: tuple[int, ...] | list[int] = (3, 6, 12, 24)
        window_size: int = 7
        mlp_ratio: float = 4.0
        qkv_bias: bool = True
        qk_scale: float | None = None
        drop_rate: float = 0.0
        attn_drop_rate: float = 0.0
        drop_path_rate: float = 0.1
        norm_layer: Type[nn.Module] = nn.LayerNorm
        abs_pos_emb: bool = False
        patch_norm: bool = True

Parameters
----------

- **img_size** (*int*):
  Input image size. Swin Transformer assumes square inputs.
- **patch_size** (*int*):
  Patch size used by the patch embedding convolution.
- **in_channels** (*int*):
  Number of input image channels.
- **num_classes** (*int*):
  Number of output classes. Set to `0` to keep the identity head.
- **embed_dim** (*int*):
  Base embedding width of the first stage.
- **depths**:
  Number of transformer blocks in each hierarchical stage.
- **num_heads**:
  Number of attention heads used by each stage.
- **window_size** (*int*):
  Local self-attention window size.
- **mlp_ratio** (*float*):
  Feedforward hidden width ratio inside each block.
- **qkv_bias** (*bool*):
  Whether to learn query, key, and value projection biases.
- **qk_scale** (*float | None*):
  Optional attention scaling override.
- **drop_rate** (*float*):
  Dropout applied to patch tokens and block projections.
- **attn_drop_rate** (*float*):
  Dropout applied to attention probabilities.
- **drop_path_rate** (*float*):
  Stochastic depth rate across the stage stack.
- **norm_layer** (*Type[nn.Module]*):
  Normalization layer used throughout the model.
- **abs_pos_emb** (*bool*):
  Whether to add absolute positional embeddings.
- **patch_norm** (*bool*):
  Whether to normalize tokens immediately after patch embedding.

Validation
----------

- `img_size`, `patch_size`, `in_channels`, `embed_dim`, and `window_size` must
  be greater than `0`.
- `img_size` must be greater than or equal to `patch_size`.
- `num_classes` must be greater than or equal to `0`.
- `depths` must be non-empty and contain only positive integers.
- `num_heads` must contain one positive integer per stage in `depths`.
- Each stage embedding width must be divisible by its configured head count.
- The patch resolution must be large enough to support the configured number of
  downsampling stages.
- `mlp_ratio` must be greater than `0`.
- `drop_rate`, `attn_drop_rate`, and `drop_path_rate` must each be in `[0, 1)`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.SwinTransformerConfig(
        img_size=32,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=8,
        depths=(2, 2),
        num_heads=(2, 4),
        window_size=7,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
    )
    model = models.SwinTransformer(config)
