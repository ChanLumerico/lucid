Object detection
================

.. toctree::
    :maxdepth: 1
    :hidden:

    RCNN <rcnn/RCNN.rst>

RCNN
----
|convnet-badge| |region-convnet-badge| |objdet-badge|

RCNN (Region-based CNN) detects objects by first generating region proposals using 
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

    * - RCNN
      - `RCNN <rcnn/RCNN>`_
      - :math:`(N,C_{in},H,W)`
      - :math:`\mathcal{O}(P_{\text{cnn}} + F^2 + F \cdot K)`
      - :math:`\mathcal{O}\left(P_{\text{cnn}} + N \cdot (FHW + F^2 + FK)\right)`

*To be implemented...🔮*
