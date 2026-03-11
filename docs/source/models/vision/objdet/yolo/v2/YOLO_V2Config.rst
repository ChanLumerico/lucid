YOLO_V2Config
=============

.. autoclass:: lucid.models.YOLO_V2Config

`YOLO_V2Config` stores the class count, anchors, loss weights, backbone
selection, route layer, image size, and passthrough option used by
:class:`lucid.models.YOLO_V2`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class YOLO_V2Config:
        num_classes: int
        num_anchors: int = 5
        anchors: list[tuple[float, float]] = field(default_factory=...)
        lambda_coord: float = 5.0
        lambda_noobj: float = 0.5
        darknet: nn.Module | None = None
        route_layer: int | None = None
        image_size: int = 416
        use_passthrough: bool = True

Parameters
----------

- **num_classes** (*int*):
  Number of object classes predicted per anchor.
- **num_anchors** (*int*):
  Number of anchors used by the detector head.
- **anchors**:
  Anchor `(width, height)` pairs used for decoding box sizes.
- **lambda_coord**, **lambda_noobj** (*float*):
  Loss weights for box-coordinate and no-object confidence terms.
- **darknet** (*nn.Module* | *None*):
  Optional custom backbone used instead of the default Darknet-19 feature extractor.
- **route_layer** (*int* | *None*):
  Feature index used by the passthrough branch when a custom backbone is supplied.
- **image_size** (*int*):
  Input image size used for anchor/grid decoding.
- **use_passthrough** (*bool*):
  Whether to use the YOLO-v2 passthrough branch.

Validation
----------

- `num_classes`, `num_anchors`, and `image_size` must be greater than `0`.
- `lambda_coord` and `lambda_noobj` must be non-negative.
- `anchors` must contain exactly `num_anchors` positive `(width, height)` pairs.
- `darknet` must be an `nn.Module` or `None`.
- `route_layer` must be non-negative or `None`.
- `use_passthrough` must be a `bool`.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    backbone = nn.Sequential(
        nn.Conv2d(3, 1024, kernel_size=1),
        nn.ReLU(),
    )
    config = models.YOLO_V2Config(
        num_classes=3,
        num_anchors=2,
        anchors=[(1.0, 1.0), (2.0, 2.0)],
        darknet=backbone,
        use_passthrough=False,
        image_size=4,
    )
    model = models.YOLO_V2(config)
