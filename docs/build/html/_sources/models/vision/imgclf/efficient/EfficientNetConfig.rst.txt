EfficientNetConfig
==================

.. autoclass:: lucid.models.EfficientNetConfig

`EfficientNetConfig` stores the compound scaling and classifier settings used by
:class:`lucid.models.EfficientNet`. It defines the EfficientNet-B family width,
depth, and resolution scale choices together with classifier regularization.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class EfficientNetConfig:
        num_classes: int = 1000
        width_coef: float = 1.0
        depth_coef: float = 1.0
        scale: float = 1.0
        dropout: float = 0.2
        se_scale: int = 4
        stochastic_depth: bool = False
        p: float = 0.5

Parameters
----------

- **num_classes** (*int*):
  Number of output classes.
- **width_coef** (*float*):
  Compound width scaling coefficient.
- **depth_coef** (*float*):
  Compound depth scaling coefficient.
- **scale** (*float*):
  Input resolution scale factor applied by the initial upsampling layer.
- **dropout** (*float*):
  Dropout probability applied before the classifier.
- **se_scale** (*int*):
  Reduction ratio used by squeeze-and-excitation blocks.
- **stochastic_depth** (*bool*):
  Whether stochastic depth should be enabled during training.
- **p** (*float*):
  Initial survival probability used when stochastic depth is enabled.

Validation
----------

- `num_classes`, `width_coef`, `depth_coef`, and `scale` must be greater than 0.
- `dropout` must be in the range `[0, 1)`.
- `se_scale` must be a positive integer.
- `stochastic_depth` must be a boolean.
- `p` must be in the range `[0, 1]`.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.EfficientNetConfig(
        num_classes=10,
        width_coef=1.0,
        depth_coef=1.0,
        scale=1.0,
        stochastic_depth=True,
        p=0.8,
    )
    model = models.EfficientNet(config)
