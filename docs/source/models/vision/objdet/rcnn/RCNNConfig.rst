RCNNConfig
==========

.. autoclass:: lucid.models.RCNNConfig

`RCNNConfig` stores the backbone, classifier width, input normalization, region
warping size, and post-processing thresholds used by
:class:`lucid.models.RCNN`.

Class Signature
---------------

.. code-block:: python

    @dataclass
    class RCNNConfig:
        backbone: nn.Module
        feat_dim: int
        num_classes: int
        image_means: tuple[float, float, float] | list[float] = (0.485, 0.456, 0.406)
        pixel_scale: float = 1.0
        warper_output_size: tuple[int, int] | list[int] = (224, 224)
        nms_iou_thresh: float = 0.3
        score_thresh: float = 0.0
        add_one: bool = True

Parameters
----------

- **backbone** (*nn.Module*):
  Shared region feature extractor.
- **feat_dim** (*int*):
  Flattened feature width emitted by the backbone.
- **num_classes** (*int*):
  Number of classification logits predicted per region.
- **image_means**:
  Per-channel image normalization means.
- **pixel_scale** (*float*):
  Input pixel scaling factor applied before mean subtraction.
- **warper_output_size**:
  Fixed crop resolution used by the region warper.
- **nms_iou_thresh** (*float*):
  IoU threshold used during non-maximum suppression.
- **score_thresh** (*float*):
  Minimum class probability kept during prediction.
- **add_one** (*bool*):
  Whether bounding-box delta decoding uses the `+1` width/height convention.

Validation
----------

- `backbone` must be an `nn.Module`.
- `feat_dim` and `num_classes` must be greater than `0`.
- `image_means` must contain exactly three values.
- `pixel_scale` must be greater than `0`.
- `warper_output_size` must contain exactly two positive integers.
- `nms_iou_thresh` must be in `[0, 1]`.

Usage
-----

.. code-block:: python

    import lucid.models as models
    import lucid.nn as nn

    backbone = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    config = models.RCNNConfig(backbone=backbone, feat_dim=64, num_classes=3)
    model = models.RCNN(config)
