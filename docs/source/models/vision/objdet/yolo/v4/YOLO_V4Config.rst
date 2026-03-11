YOLO_V4Config
=============

.. autoclass:: lucid.models.YOLO_V4Config

`YOLO_V4Config` stores the 3-scale anchor setup, strides, backbone selection,
neck channel widths, assignment thresholds, and IoU-aware options used by
:class:`lucid.models.YOLO_V4`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class YOLO_V4Config:
        num_classes: int
        anchors: list[list[tuple[int, int]]] = field(default_factory=...)
        strides: list[int] = field(default_factory=...)
        backbone: nn.Module | None = None
        backbone_out_channels: tuple[int, int, int] | None = None
        in_channels: tuple[int, int, int] = (256, 512, 1024)
        pos_iou_thr: float = 0.25
        ignore_iou_thr: float = 0.5
        obj_balance: tuple[float, float, float] = (4.0, 1.0, 0.4)
        cls_label_smoothing: float = 0.0
        iou_aware_alpha: float = 0.5
        iou_branch_weight: float = 1.0

Validation
----------

- `num_classes` must be greater than `0`.
- `anchors` must contain three scales with three positive integer anchors each.
- `strides` must contain exactly three positive values.
- `backbone` must be an `nn.Module` or `None`.
- `backbone_out_channels` must be omitted for the default backbone and required for a custom backbone.
- `in_channels` and `backbone_out_channels` must contain exactly three positive integers.
- IoU thresholds and `iou_aware_alpha` must be in `[0, 1]`.
- `obj_balance` must contain exactly three positive values.
- `cls_label_smoothing` must be in `[0, 1)`.
- `iou_branch_weight` must be non-negative.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    class ToyBackbone(nn.Module):
        def forward(self, x):
            return x, x, x

    config = models.YOLO_V4Config(
        num_classes=3,
        backbone=ToyBackbone(),
        backbone_out_channels=(3, 3, 3),
        in_channels=(3, 3, 3),
        iou_aware_alpha=0.0,
        iou_branch_weight=0.0,
    )
    model = models.YOLO_V4(config)
