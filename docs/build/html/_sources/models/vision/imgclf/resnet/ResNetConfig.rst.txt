ResNetConfig
============

.. autoclass:: lucid.models.ResNetConfig

`ResNetConfig` stores the structural choices used by :class:`lucid.models.ResNet`.
It describes the residual block family, per-stage depths, stem settings, classifier
size, and any extra block keyword arguments required by derived ResNet variants.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class ResNetConfig:
        block: Literal["basic", "bottleneck", "preact_bottleneck"] | Type[nn.Module]
        layers: tuple[int, int, int, int] | list[int]
        num_classes: int = 1000
        in_channels: int = 3
        stem_width: int = 64
        stem_type: Literal["deep"] | None = None
        avg_down: bool = False
        channels: tuple[int, int, int, int] | list[int] = (64, 128, 256, 512)
        block_args: dict[str, Any] = field(default_factory=dict)

Parameters
----------

- **block** (*Literal["basic", "bottleneck", "preact_bottleneck"] | Type[nn.Module]*):
  Residual block selection. Built-in block names cover the standard ResNet families,
  and custom block classes are also accepted when they inherit from `nn.Module` and
  define an `expansion` attribute.
- **layers** (*tuple[int, int, int, int] | list[int]*):
  Number of residual blocks in each of the four stages.
- **num_classes** (*int*):
  Number of output classes.
- **in_channels** (*int*):
  Number of channels in the input image tensor.
- **stem_width** (*int*):
  Width parameter used by the deep stem variant.
- **stem_type** (*Literal["deep"] | None*):
  Stem style. `None` uses the classic single 7x7 convolution, while `"deep"` uses
  a three-layer stem.
- **avg_down** (*bool*):
  Whether projection shortcuts should downsample using average pooling before the 1x1 projection.
- **channels** (*tuple[int, int, int, int] | list[int]*):
  Output width of each residual stage.
- **block_args** (*dict[str, Any]*):
  Extra keyword arguments forwarded to each residual block instance.

Validation
----------

- `block` must be one of the built-in block names or a custom `nn.Module` subclass.
- Custom block classes must define an `expansion` attribute.
- `layers` and `channels` must each contain exactly four positive integers.
- `num_classes`, `in_channels`, and `stem_width` must be greater than 0.
- `stem_type` must be `None` or `"deep"`.
- `block_args` must be a dictionary.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.ResNetConfig(
        block="bottleneck",
        layers=[3, 4, 6, 3],
        num_classes=10,
        in_channels=1,
        stem_type="deep",
        stem_width=32,
        avg_down=True,
    )

    model = models.ResNet(config)
