AttentionUNetGateConfig
=======================

.. autoclass:: lucid.models.AttentionUNetGateConfig

`AttentionUNetGateConfig` describes how additive attention gates are applied to
skip features inside :class:`lucid.models.AttentionUNet`. These gates receive
encoder skip features together with decoder gating features and produce
attention-weighted skip responses before concatenation.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class AttentionUNetGateConfig:
        enabled: bool = True
        mode: Literal["additive"] = "additive"
        gate_activation: Literal["relu"] = "relu"
        attention_activation: Literal["sigmoid"] = "sigmoid"
        use_grid_attention: bool = True
        inter_channels: int | tuple[int, ...] | list[int] | None = None
        attention_channels: int = 1
        resample_mode: Literal["bilinear", "nearest"] = "bilinear"
        gate_on_skips: tuple[bool, ...] | list[bool] | None = None
        skip_low_level_gates: bool = False
        use_multi_scale_gating: bool = True
        project_skip_with_1x1: bool = True
        project_gating_with_1x1: bool = True
        init_pass_through: bool = True

Parameters
----------

- **enabled** (*bool*): Whether attention gates are active.
- **mode** (*Literal["additive"]*): Attention compatibility function. The current
  implementation supports only additive gating.
- **gate_activation** (*Literal["relu"]*): Activation used after combining
  projected skip and gating features.
- **attention_activation** (*Literal["sigmoid"]*): Activation used to convert
  gate logits into attention coefficients.
- **use_grid_attention** (*bool*): Whether to use spatially varying decoder
  gating features rather than a single pooled gating vector.
- **inter_channels** (*int | tuple[int, ...] | list[int] | None*): Internal
  channel width for the gate projection. If `None`, widths are inferred per gate.
- **attention_channels** (*int*): Number of attention coefficient channels
  predicted by each gate. The paper-style default is `1`.
- **resample_mode** (*Literal["bilinear", "nearest"]*): Interpolation mode used
  to align gating features and attention masks with skip resolutions.
- **gate_on_skips** (*tuple[bool, ...] | list[bool] | None*): Per-decoder-stage
  toggle for whether a gate is applied.
- **skip_low_level_gates** (*bool*): Whether to also gate the shallowest skip.
- **use_multi_scale_gating** (*bool*): Whether the decoder gating pathway uses
  multi-scale context. This flag documents intent for the paper-faithful setup.
- **project_skip_with_1x1** (*bool*): Whether to project skip features with a 1x1 convolution.
- **project_gating_with_1x1** (*bool*): Whether to project gating features with a 1x1 convolution.
- **init_pass_through** (*bool*): Whether to initialize gate logits close to a
  pass-through regime at the start of training.

Usage
-----

.. code-block:: python

    import lucid.models as models

    gate_cfg = models.AttentionUNetGateConfig(
        inter_channels=(32, 64, 64),
        gate_on_skips=(True, True, False),
    )
