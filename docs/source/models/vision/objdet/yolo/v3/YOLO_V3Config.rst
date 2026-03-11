YOLO_V3Config
=============

.. autoclass:: lucid.models.YOLO_V3Config

`YOLO_V3Config` stores the class count, 9-anchor set, input image size, and
optional custom 3-scale backbone used by :class:`lucid.models.YOLO_V3`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class YOLO_V3Config:
        num_classes: int
        anchors: list[tuple[int, int]] = field(default_factory=...)
        image_size: int = 416
        darknet: nn.Module | None = None
        darknet_out_channels_arr: list[int] | None = None

Parameters
----------

- **num_classes** (*int*):
  Number of object classes predicted at every scale.
- **anchors**:
  Flat list of 9 anchor `(width, height)` pairs used by the three detection heads.
- **image_size** (*int*):
  Input image size used for box decoding and loss scaling.
- **darknet** (*nn.Module* | *None*):
  Optional custom backbone that returns three feature maps.
- **darknet_out_channels_arr** (*list[int]* | *None*):
  Channel widths of the three backbone outputs when a custom backbone is supplied.

Validation
----------

- `num_classes` and `image_size` must be greater than `0`.
- `anchors` must contain exactly 9 positive integer `(width, height)` pairs.
- `darknet` must be an `nn.Module` or `None`.
- `darknet_out_channels_arr` must be `None` for the default backbone.
- When `darknet` is custom, `darknet_out_channels_arr` must contain exactly three positive integers.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    class ToyBackbone(nn.Module):
        def forward(self, x):
            return x, x, x

    config = models.YOLO_V3Config(
        num_classes=3,
        darknet=ToyBackbone(),
        darknet_out_channels_arr=[3, 3, 3],
    )
    model = models.YOLO_V3(config)
