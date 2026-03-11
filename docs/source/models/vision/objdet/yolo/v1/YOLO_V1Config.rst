YOLO_V1Config
=============

.. autoclass:: lucid.models.YOLO_V1Config

`YOLO_V1Config` stores the input channel count, detection grid size, number of
boxes and classes, loss weights, and convolutional backbone layout used by
:class:`lucid.models.YOLO_V1`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class YOLO_V1Config:
        in_channels: int = 3
        split_size: int = 7
        num_boxes: int = 2
        num_classes: int = 20
        lambda_coord: float = 5.0
        lambda_noobj: float = 0.5
        conv_config: list[object] = field(default_factory=...)

Parameters
----------

- **in_channels** (*int*):
  Number of input image channels consumed by the backbone.
- **split_size** (*int*):
  Detection grid size `S` used by the final prediction head.
- **num_boxes** (*int*):
  Number of bounding boxes predicted per grid cell.
- **num_classes** (*int*):
  Number of class logits predicted per grid cell.
- **lambda_coord**, **lambda_noobj** (*float*):
  Loss weights for box-coordinate and no-object confidence terms.
- **conv_config**:
  YOLO-v1 convolutional architecture definition using tuple, `"M"`, and
  repeated-block entries.

Validation
----------

- `in_channels`, `split_size`, `num_boxes`, and `num_classes` must be greater than `0`.
- `lambda_coord` and `lambda_noobj` must be non-negative.
- `conv_config` must be a non-empty list using valid YOLO-v1 architecture entries.
- `conv_config` must end with a `1024`-channel convolution because the fully connected head expects that feature width.

Usage
-----

.. code-block:: python

    import lucid.models as models

    config = models.YOLO_V1Config(
        split_size=2,
        num_boxes=2,
        num_classes=3,
        conv_config=[(1024, 1, 1, 0)],
    )
    model = models.YOLO_V1(config)
