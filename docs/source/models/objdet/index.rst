Object detection
================

.. toctree::
    :maxdepth: 1
    :hidden:

    Utilities <utilities/index.rst>

    R-CNN <rcnn/RCNN.rst>
    Fast R-CNN <rcnn/FastRCNN.rst>
    Faster R-CNN <faster_rcnn/FasterRCNN.rst>
    YOLO <yolo/index.rst>

R-CNN
-----
|convnet-badge| |two-stage-det-badge| |objdet-badge|

R-CNN (Region-based CNN) detects objects by first generating region proposals using 
Selective Search, then classifies each using a shared CNN. It combines region warping, 
feature extraction, and per-region classification with Non-Maximum Suppression.

 Girshick, Ross, et al. "Rich feature hierarchies for accurate object detection and 
 semantic segmentation." *Proceedings of the IEEE conference on computer vision and 
 pattern recognition* (2014): 580-587.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape

    * - R-CNN
      - `RCNN <rcnn/RCNN>`_
      - :math:`(N,C_{in},H,W)`

Fast R-CNN
----------
|convnet-badge| |two-stage-det-badge| |objdet-badge|

Fast R-CNN improves upon R-CNN by computing the feature map once for the entire image, 
then pooling features from proposed regions using RoI Pooling. It unifies classification 
and bounding box regression into a single network with a shared backbone.

 Girshick, Ross. "Fast R-CNN." *Proceedings of the IEEE international conference on 
 computer vision* (2015): 1440-1448.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape

    * - Fast R-CNN
      - `FastRCNN <rcnn/FastRCNN>`_
      - :math:`(N,C_{in},H,W)`

Faster R-CNN
------------
|convnet-badge| |two-stage-det-badge| |objdet-badge|

Faster R-CNN builds on Fast R-CNN by introducing a Region Proposal Network (RPN) 
that shares convolutional features with the detection head, enabling end-to-end 
training and real-time inference.

 Ren, Shaoqing et al. "Faster R-CNN: Towards Real-Time Object Detection with 
 Region Proposal Networks." *IEEE Transactions on Pattern Analysis and Machine 
 Intelligence* (2017).

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count

    * - Faster R-CNN
      - `FasterRCNN <faster_rcnn/FasterRCNN>`_
      - :math:`(N,C_{in},H,W)`
      - :math:`-`
    
    * - Faster R-CNN ResNet-50 FPN
      - `faster_rcnn_resnet_50_fpn <faster_rcnn/faster_rcnn_resnet_50_fpn>`_
      - :math:`(N,3,H,W)`
      - 43,515,902
    
    * - Faster R-CNN ResNet-101 FPN
      - `faster_rcnn_resnet_101_fpn <faster_rcnn/faster_rcnn_resnet_101_fpn>`_
      - :math:`(N,3,H,W)`
      - 62,508,030


YOLO
----
|convnet-badge| |one-stage-det-badge| |objdet-badge|

YOLO is a one-stage object detector that frames detection as a single regression problem, 
directly predicting bounding boxes and class probabilities from full images in a single 
forward pass. It enables real-time detection with impressive speed and accuracy.

YOLO-v1
~~~~~~~

 Redmon, Joseph et al. "You Only Look Once: Unified, Real-Time Object Detection."
 *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (2016).

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs
    
    * - YOLO-v1
      - `yolo_v1 <yolo/v1/yolo_v1>`_
      - :math:`(N,3,448,448)`
      - 271,716,734
      - 404.84M
    
    * - YOLO-v1-Tiny
      - `yolo_v1_tiny <yolo/v1/yolo_v1_tiny>`_
      - :math:`(N,3,448,448)`
      - 236,720,462
      - 302.21M

YOLO-v2
~~~~~~~

 Redmon, Joseph, and Ali Farhadi. “YOLO9000: Better, Faster, Stronger.” 
 *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (CVPR), 
 2017, pp. 7263-7271.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs

    * - YOLO-v2
      - `yolo_v2 <yolo/v2/yolo_v2>`_
      - :math:`(N,3,416,416)`
      - 21,287,133
      - 214.26M
    
    * - YOLO-v2-Tiny
      - `yolo_v2_tiny <yolo/v2/yolo_v2_tiny>`_
      - :math:`(N,3,416,416)`
      - 15,863,821
      - 77.45M

YOLO-v3
~~~~~~~

 Redmon, Joseph, and Ali Farhadi. "YOLOv3: An Incremental Improvement." 
 *arXiv preprint* arXiv:1804.02767 (2018).

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs

    * - YOLO-v3
      - `yolo_v3 <yolo/v3/yolo_v3>`_
      - :math:`(N,3,416,416)`
      - 62,974,149
      - 558.71M
    
    * - YOLO-v3-Tiny
      - `yolo_v3_tiny <yolo/v3/yolo_v3_tiny>`_
      - :math:`(N,3,416,416)`
      - 23,106,933
      - 147.93M

YOLO-v4
~~~~~~~

 Bochkovskiy, Alexey, Chien-Yao Wang, and Hong-Yuan Mark Liao. 
 YOLOv4: Optimal Speed and Accuracy of Object Detection. 2020, arXiv:2004.10934.

.. list-table::
    :header-rows: 1
    :align: left

    * - Name
      - Model
      - Input Shape
      - Parameter Count
      - FLOPs

    * - YOLO-v4
      - `yolo_v4 <yolo/v4/yolo_v4>`_
      - :math:`(N,3,608,608)`
      - 93,488,078
      - 1.41B

*To be implemented...🔮*
