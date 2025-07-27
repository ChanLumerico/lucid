Object detection
================

.. toctree::
    :maxdepth: 1
    :hidden:

    Utilities <utilities/index.rst>

    R-CNN <rcnn/RCNN.rst>
    Fast R-CNN <rcnn/FastRCNN.rst>
    Faster R-CNN <faster_rcnn/FasterRCNN.rst>

R-CNN
-----
|convnet-badge| |region-convnet-badge| |objdet-badge|

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
|convnet-badge| |region-convnet-badge| |objdet-badge|

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
|convnet-badge| |region-convnet-badge| |objdet-badge|

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
      - -
    
    * - Faster R-CNN ResNet-50 FPN
      - `faster_rcnn_resnet_50_fpn <faster_rcnn/faster_rcnn_resnet_50_fpn>`_
      - :math:`(N,3,H,W)`
      - 43,515,902
    
    * - Faster R-CNN ResNet-101 FPN
      - `faster_rcnn_resnet_101_fpn <faster_rcnn/faster_rcnn_resnet_101_fpn>`_
      - :math:`(N,3,H,W)`
      - 62,508,030

*To be implemented...🔮*
