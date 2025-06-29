Object detection
================

.. toctree::
    :maxdepth: 1
    :hidden:

    R-CNN <rcnn/RCNN.rst>
    Fast R-CNN <rcnn/FastRCNN.rst>

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
      - Parameter Count
      - FLOPs

    * - R-CNN
      - `RCNN <rcnn/RCNN>`_
      - :math:`(N,C_{in},H,W)`
      - :math:`\mathcal{O}(P_{\text{cnn}} + F^2 + F \cdot K)`
      - :math:`\mathcal{O}\left(P_{\text{cnn}} + N \cdot (FHW + F^2 + FK)\right)`

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
      - Parameter Count
      - FLOPs

    * - Fast R-CNN
      - `FastRCNN <rcnn/FastRCNN>`_
      - :math:`(N,C_{in},H,W)`
      - :math:`\mathcal{O}(P_{\text{cnn}} + 2FH + 5CH)`
      - :math:`\mathcal{O}\left(P_{\text{cnn}} + N \cdot (2FH + 5CH)\right)`

*To be implemented...🔮*
