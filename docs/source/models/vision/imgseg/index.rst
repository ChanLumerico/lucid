Image Segmentation Models
=========================

.. toctree::
    :maxdepth: 1
    :hidden:

    Mask R-CNN <mask_rcnn/MaskRCNN.rst>

Mask R-CNN
----------
|convnet-badge| |two-stage-det-badge|

Mask R-CNN extends Faster R-CNN with a parallel mask prediction branch to perform
instance-level segmentation in addition to bounding box detection and classification.
The model first proposes candidate object regions, then predicts class labels, box
refinements, and a binary mask for each detected instance.

 He, Kaiming, et al. "Mask R-CNN." *Proceedings of the IEEE International
 Conference on Computer Vision* (2017): 2961-2969.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count

    * - Mask R-CNN ResNet-50 FPN
      - `mask_rcnn_resnet_50_fpn <mask_rcnn/mask_rcnn_resnet_50_fpn>`_
      - :math:`(N,3,H,W)`
      - 46,037,607

    * - Mask R-CNN ResNet-101 FPN
      - `mask_rcnn_resnet_101_fpn <mask_rcnn/mask_rcnn_resnet_101_fpn>`_
      - :math:`(N,3,H,W)`
      - 65,126,611
