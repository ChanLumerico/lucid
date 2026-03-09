FCNConfig
=========

.. autoclass:: lucid.models.FCNConfig

`FCNConfig` stores the architectural choices used by :class:`lucid.models.FCN`.
It controls the backbone preset, segmentation class count, and the widths of
the main and auxiliary classifier heads.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class FCNConfig:
        num_classes: int
        backbone: str = "resnet_50"
        in_channels: int = 3
        aux_loss: bool = True
        out_in_channels: int = 2048
        aux_in_channels: int = 1024
        classifier_hidden_channels: int = 512
        aux_hidden_channels: int = 256
        dropout: float = 0.1

Parameters
----------

- **num_classes** (*int*): Number of segmentation classes predicted per pixel.
- **backbone** (*str*): Backbone preset name. Supported values are `resnet_50`
  and `resnet_101`.
- **in_channels** (*int*): Number of input image channels.
- **aux_loss** (*bool*): Whether to include an auxiliary classifier head from
  the intermediate backbone stage.
- **out_in_channels** (*int*): Channel width of the deepest backbone feature map.
- **aux_in_channels** (*int*): Channel width of the auxiliary backbone feature map.
- **classifier_hidden_channels** (*int*): Hidden width of the main segmentation head.
- **aux_hidden_channels** (*int*): Hidden width of the auxiliary segmentation head.
- **dropout** (*float*): Dropout probability used inside both segmentation heads.

Usage
-----

.. code-block:: python

    import lucid.models as models

    cfg = models.FCNConfig(
        num_classes=21,
        backbone="resnet_50",
        in_channels=3,
        aux_loss=True,
    )

    model = models.FCN(cfg)
