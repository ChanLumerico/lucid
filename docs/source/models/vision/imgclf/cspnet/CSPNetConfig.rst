CSPNetConfig
============

.. autoclass:: lucid.models.CSPNetConfig

`CSPNetConfig` stores the stage layout and classifier settings used by
:class:`lucid.models.CSPNet`. It defines the CSP stage specs, backbone stack
style, normalization and activation choices, pooling mode, and head widths.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class CSPNetConfig:
        stage_specs: tuple[tuple[object, ...], ...] | list[tuple[object, ...]] | list[list[object]]
        stack_type: Literal["resnet", "resnext", "darknet"]
        in_channels: int = 3
        stem_channels: int = 64
        num_classes: int = 1000
        norm: Callable[..., nn.Module] = nn.BatchNorm2d
        act: Callable[..., nn.Module] = nn.ReLU
        split_ratio: float = 0.5
        global_pool: Literal["avg", "max"] = "avg"
        dropout: float = 0.0
        feature_channels: int | None = None
        pre_kernel_size: int = 1
        groups: int = 1
        base_width: int = 64

Parameters
----------

- **stage_specs**:
  Sequence of stage specs in the form `(stage_width, num_layers, downsample)`.
- **stack_type**:
  Backbone stack style: `"resnet"`, `"resnext"`, or `"darknet"`.
- **in_channels** (*int*):
  Number of input channels.
- **stem_channels** (*int*):
  Output width of the stem.
- **num_classes** (*int*):
  Number of output classes.
- **norm**:
  Callable normalization layer factory.
- **act**:
  Callable activation layer factory.
- **split_ratio** (*float*):
  Fraction of stage channels routed through one side of the CSP split.
- **global_pool**:
  Final pooling mode, `"avg"` or `"max"`.
- **dropout** (*float*):
  Dropout probability before the classifier.
- **feature_channels** (*int* or `None`):
  Optional pre-head projection width.
- **pre_kernel_size** (*int*):
  Kernel size of the pre-stage convolution.
- **groups** (*int*):
  Group count used by ResNeXt-style stacks.
- **base_width** (*int*):
  Base width used by ResNet/ResNeXt bottlenecks.

Validation
----------

- `stage_specs` must contain at least one three-value stage spec.
- Stage widths and layer counts must be positive integers, and `downsample` must be boolean.
- `stack_type` must be one of `"resnet"`, `"resnext"`, or `"darknet"`.
- `in_channels`, `stem_channels`, `num_classes`, `pre_kernel_size`, `groups`, and `base_width` must be greater than 0.
- `norm` and `act` must be callable.
- `split_ratio` must be in the range `(0, 1)`.
- `global_pool` must be either `"avg"` or `"max"`.
- `dropout` must be in the range `[0, 1)`.
- `feature_channels` must be greater than 0 when provided.

Usage
-----

.. code-block:: python

    from lucid.models import CSPNet, CSPNetConfig

    config = CSPNetConfig(
        stage_specs=((32, 1, False), (64, 1, True)),
        stack_type="resnet",
        in_channels=3,
        stem_channels=16,
        num_classes=10,
        feature_channels=32,
    )
    model = CSPNet(config)
