Mask R-CNN
==========
|convnet-badge| |two-stage-det-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    mask_rcnn_resnet_50_fpn.rst
    mask_rcnn_resnet_101_fpn.rst

.. autoclass:: lucid.models.mask_rcnn.MaskRCNN

`MaskRCNN` extends Faster R-CNN by adding a parallel instance mask head on top of
RoI features. For each foreground proposal, it predicts class logits, bounding-box
deltas, and a per-instance binary segmentation mask.

Class Signature
---------------

.. code-block:: python

    class MaskRCNN(nn.Module):
        def __init__(
            self,
            backbone: nn.Module,
            feat_channels: int,
            num_classes: int,
            *,
            use_fpn: bool = False,
            anchor_sizes: tuple[int, ...] = (128, 256, 512),
            aspect_ratios: tuple[float, ...] = (0.5, 1.0, 2.0),
            anchor_stride: int = 16,
            pool_size: tuple[int, int] = (7, 7),
            hidden_dim: int = 4096,
            dropout: float = 0.5,
            mask_pool_size: tuple[int, int] = (14, 14),
            mask_hidden_channels: int = 256,
            mask_out_size: int = 28,
        )

Parameters
----------

- **backbone** (*nn.Module*):
  Backbone feature extractor used by both proposal and RoI heads.

- **feat_channels** (*int*):
  Number of channels in the backbone feature map consumed by detection/mask heads.

- **num_classes** (*int*):
  Number of target classes (background excluded).

- **use_fpn** (*bool*, optional):
  If `True`, uses `MultiScaleROIAlign` for FPN multi-level features.

- **anchor_sizes** (*tuple[int, ...]*, optional):
  Anchor scales used by the RPN.

- **aspect_ratios** (*tuple[float, ...]*, optional):
  Anchor aspect ratios used by the RPN.

- **anchor_stride** (*int*, optional):
  Anchor stride on the feature map.

- **pool_size** (*tuple[int, int]*, optional):
  RoI pooling resolution for classification/regression heads.

- **hidden_dim** (*int*, optional):
  Hidden dimension of the two-layer MLP box/class head.

- **dropout** (*float*, optional):
  Dropout probability for the MLP detection head.

- **mask_pool_size** (*tuple[int, int]*, optional):
  RoI pooling size for mask features.

- **mask_hidden_channels** (*int*, optional):
  Hidden channels in the mask head convolution stack.

- **mask_out_size** (*int*, optional):
  Final mask target/output size per instance.

Architecture
------------

1. **Backbone + RPN**:

   - Extracts shared image features.
   - Proposes candidate boxes with objectness and box regression.

2. **RoI Detection Head**:

   - Applies RoIAlign on proposals.
   - Predicts class logits and class-specific box deltas.

3. **RoI Mask Head**:

   - Applies dedicated RoIAlign for mask features.
   - Predicts class-specific mask logits per proposal.

4. **Training Losses**:

   - Combines RPN objectness/regression, RoI cls/regression, and mask BCE loss.

Loss Dictionary
---------------

.. code-block:: python

    class _MaskRCNNLoss(TypedDict):
        rpn_cls_loss: Tensor
        rpn_reg_loss: Tensor
        roi_cls_loss: Tensor
        roi_reg_loss: Tensor
        mask_loss: Tensor
        total_loss: Tensor

Methods
-------

.. automethod:: lucid.models.vision.mask_rcnn.MaskRCNN.forward
.. automethod:: lucid.models.vision.mask_rcnn.MaskRCNN.predict
.. automethod:: lucid.models.vision.mask_rcnn.MaskRCNN.get_loss

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models.vision.mask_rcnn import MaskRCNN
    import lucid
    import lucid.nn as nn

    class SimpleBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            )

        def forward(self, x):
            return self.net(x)

    model = MaskRCNN(backbone=SimpleBackbone(), feat_channels=128, num_classes=5)
    x = lucid.random.randn(1, 3, 512, 512)
    cls_logits, bbox_deltas, mask_logits = model(x)
    print(cls_logits.shape, bbox_deltas.shape, mask_logits.shape)

**Inference API**

.. code-block:: python

    detections = model.predict(x)
    print(detections[0]["boxes"].shape)
    print(detections[0]["scores"].shape)
    print(detections[0]["labels"].shape)
    print(detections[0]["masks"].shape)
