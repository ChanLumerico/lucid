CvTConfig
=========

.. autoclass:: lucid.models.CvTConfig

`CvTConfig` stores the stage layout and classifier settings used by
:class:`lucid.models.CvT`. It defines the patch embedding parameters,
hierarchical transformer depths, attention heads, token settings, and optional
dropout schedules for each stage.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class CvTConfig:
        num_stages: int
        patch_size: tuple[int, ...] | list[int]
        patch_stride: tuple[int, ...] | list[int]
        patch_padding: tuple[int, ...] | list[int]
        dim_embed: tuple[int, ...] | list[int]
        num_heads: tuple[int, ...] | list[int]
        depth: tuple[int, ...] | list[int]
        in_channels: int = 3
        num_classes: int = 1000
        act_layer: Callable[..., nn.Module] = nn.GELU
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm
        mlp_ratio: tuple[float, ...] | list[float] = (4.0, 4.0, 4.0)
        attn_drop_rate: tuple[float, ...] | list[float] = (0.0, 0.0, 0.0)
        drop_rate: tuple[float, ...] | list[float] = (0.0, 0.0, 0.0)
        drop_path_rate: tuple[float, ...] | list[float] = (0.0, 0.0, 0.1)
        qkv_bias: tuple[bool, ...] | list[bool] = (True, True, True)
        cls_token: tuple[bool, ...] | list[bool] = (False, False, True)
        pos_embed: tuple[bool, ...] | list[bool] = (False, False, False)
        qkv_proj_method: tuple[str, ...] | list[str] = ("dw_bn", "dw_bn", "dw_bn")
        kernel_qkv: tuple[int, ...] | list[int] = (3, 3, 3)
        padding_kv: tuple[int, ...] | list[int] = (1, 1, 1)
        stride_kv: tuple[int, ...] | list[int] = (2, 2, 2)
        padding_q: tuple[int, ...] | list[int] = (1, 1, 1)
        stride_q: tuple[int, ...] | list[int] = (1, 1, 1)

Parameters
----------

- **num_stages** (*int*):
  Number of hierarchical transformer stages.
- **patch_size**, **patch_stride**, **patch_padding**:
  Per-stage convolutional embedding parameters.
- **dim_embed**:
  Per-stage embedding widths.
- **num_heads**:
  Per-stage attention head counts.
- **depth**:
  Per-stage transformer block counts.
- **in_channels** (*int*):
  Number of input image channels.
- **num_classes** (*int*):
  Number of output classes. Set to `0` to keep the identity head.
- **act_layer**, **norm_layer**:
  Activation and normalization modules used by each stage.
- **mlp_ratio**, **attn_drop_rate**, **drop_rate**, **drop_path_rate**:
  Per-stage feedforward width ratios and dropout schedules.
- **qkv_bias**, **cls_token**, **pos_embed**:
  Per-stage attention bias and token toggles.
- **qkv_proj_method**, **kernel_qkv**, **padding_kv**, **stride_kv**,
  **padding_q**, **stride_q**:
  Per-stage convolutional projection settings for attention.

Validation
----------

- `num_stages` and `in_channels` must be greater than `0`.
- `num_classes` must be greater than or equal to `0`.
- Every per-stage sequence must contain exactly `num_stages` values.
- Patch sizes, strides, embedding widths, head counts, depths, and kernel sizes
  must contain positive integers.
- Padding values must contain non-negative integers.
- `mlp_ratio` values must be greater than `0`.
- Dropout rates must each be in `[0, 1)`.
- `qkv_proj_method` values must be one of `dw_bn`, `avg`, or `lin`.
- Each `dim_embed` value must be divisible by the corresponding `num_heads`.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    config = models.CvTConfig(
        num_stages=3,
        patch_size=(3, 3, 3),
        patch_stride=(2, 2, 2),
        patch_padding=(1, 1, 1),
        dim_embed=(16, 32, 64),
        num_heads=(1, 2, 4),
        depth=(1, 1, 1),
        in_channels=1,
        num_classes=10,
        norm_layer=nn.LayerNorm,
    )
    model = models.CvT(config)
