Image Segmentation Models
=========================

.. toctree::
    :maxdepth: 1
    :hidden:

    Mask R-CNN <mask_rcnn/MaskRCNN.rst>
    MaskFormer <maskformer/MaskFormer.rst>
    Mask2Former <mask2former/Mask2Former.rst>

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
      - Pre-Trained

    * - Mask R-CNN ResNet-50 FPN
      - `mask_rcnn_resnet_50_fpn <mask_rcnn/mask_rcnn_resnet_50_fpn>`_
      - :math:`(N,3,H,W)`
      - 46,037,607
      - ❌

    * - Mask R-CNN ResNet-101 FPN
      - `mask_rcnn_resnet_101_fpn <mask_rcnn/mask_rcnn_resnet_101_fpn>`_
      - :math:`(N,3,H,W)`
      - 65,126,611
      - ❌

MaskFormer
----------
|transformer-badge| |segmentation-transformer-badge|

MaskFormer reformulates segmentation as mask classification with Transformer
queries. A CNN backbone and pixel decoder produce dense features, and each
query predicts a class and a mask embedding that is projected to segmentation
logits.

 Cheng, Bowen, et al. "Per-Pixel Classification is Not All You Need for
 Semantic Segmentation." *Advances in Neural Information Processing Systems*
 34 (2021): 17864-17875.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Pre-Trained

    * - MaskFormer-ResNet-18
      - `maskformer_resnet_18 <maskformer/maskformer_resnet_18>`_
      - :math:`(N,3,H,W)`
      - 24,700,119
      - ❌

    * - MaskFormer-ResNet-34
      - `maskformer_resnet_34 <maskformer/maskformer_resnet_34>`_
      - :math:`(N,3,H,W)`
      - 34,808,279
      - ❌

    * - MaskFormer-ResNet-50
      - `maskformer_resnet_50 <maskformer/maskformer_resnet_50>`_
      - :math:`(N,3,H,W)`
      - 41,307,863
      - ✅

    * - MaskFormer-ResNet-101
      - `maskformer_resnet_101 <maskformer/maskformer_resnet_101>`_
      - :math:`(N,3,H,W)`
      - 60,299,991
      - ✅

Mask2Former
-----------
|transformer-badge| |segmentation-transformer-badge|

Mask2Former introduces masked attention and multi-scale decoding for
universal segmentation. Query masks from each decoder stage are reused as
attention constraints for the next stage.

 Cheng, Bowen, et al. "Masked-attention Mask Transformer for Universal Image
 Segmentation." *Proceedings of the IEEE/CVF Conference on Computer Vision
 and Pattern Recognition* (2022): 1290-1299.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - Pre-Trained

    * - Mask2Former-ResNet-18
      - `mask2former_resnet_18 <mask2former/mask2former_resnet_18>`_
      - :math:`(N,3,H,W)`
      - 30,972,567
      - ❌

    * - Mask2Former-ResNet-34
      - `mask2former_resnet_34 <mask2former/mask2former_resnet_34>`_
      - :math:`(N,3,H,W)`
      - 41,080,727
      - ❌

    * - Mask2Former-ResNet-50
      - `mask2former_resnet_50 <mask2former/mask2former_resnet_50>`_
      - :math:`(N,3,H,W)`
      - 44,041,367
      - ❌

    * - Mask2Former-ResNet-101
      - `mask2former_resnet_101 <mask2former/mask2former_resnet_101>`_
      - :math:`(N,3,H,W)`
      - 63,033,495
      - ❌

    * - Mask2Former-Swin-Tiny
      - `mask2former_swin_tiny <mask2former/mask2former_swin_tiny>`_
      - :math:`(N,3,224,224)`
      - 47,439,633
      - ✅

    * - Mask2Former-Swin-Small
      - `mask2former_swin_small <mask2former/mask2former_swin_small>`_
      - :math:`(N,3,224,224)`
      - 68,757,537
      - ✅

    * - Mask2Former-Swin-Base
      - `mask2former_swin_base <mask2former/mask2former_swin_base>`_
      - :math:`(N,3,384,384)`
      - 106,922,191
      - ✅

    * - Mask2Former-Swin-Large
      - `mask2former_swin_large <mask2former/mask2former_swin_large>`_
      - :math:`(N,3,384,384)`
      - 215,488,779
      - ✅
