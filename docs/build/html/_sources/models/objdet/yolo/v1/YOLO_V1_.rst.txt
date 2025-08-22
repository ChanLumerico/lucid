YOLO-v1
=======
|convnet-badge| |one-stage-det-badge| |objdet-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    yolo_v1.rst
    yolo_v1_tiny.rst

.. autoclass:: lucid.models.YOLO_V1

The `YOLO_V1` class implements the original YOLO (You Only Look Once) model 
for real-time object detection, as proposed by Redmon et al. (2016).

It divides the input image into an :math:`S \times S` grid and predicts 
bounding boxes, objectness scores, and class probabilities for each cell.

.. image:: yolo_v1.png
    :width: 600
    :alt: YOLO-v1 architecture
    :align: center

Class Signature
---------------

.. code-block:: python

    class YOLO_V1(
        in_channels: int,
        split_size: int,
        num_boxes: int,
        num_classes: int,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
    )

Parameters
----------

- **in_channels** (*int*):  
  Number of input image channels, typically `3` for RGB.

- **split_size** (*int*):  
  Number of grid divisions per side (S), meaning the input is divided into :math:`S \times S` cells.

- **num_boxes** (*int*):  
  Number of bounding boxes (B) predicted per grid cell.

- **num_classes** (*int*):  
  Number of object classes to predict (C).

- **lambda_coord** (*float*):  
  Weight for the coordinate loss (default: 5.0).

- **lambda_noobj** (*float*):  
  Weight for the no-object confidence loss (default: 0.5).

Input Format
------------

The target tensor from the dataset should have shape:

.. code-block:: python

    (N, S, S,  *)

Where:

- `S` is `split_size` (grid size),
- `B` is `num_boxes` (bounding boxes per cell),
- `C` is `num_classes`.

Each vector at `(i, j)` of shape `(5 * B + C)` contains:

- For each box (B): `(x, y, w, h, conf)`
- For the cell: one-hot class vector of length `C`

Returns
-------

Use the `forward` method for predictions:

.. code-block:: python

    preds = model(x)

- **preds** (*Tensor*):  
  Tensor of shape `(N, S, S, B * 5 + C)` containing all bounding box and classification predictions.

Loss is computed using:

.. code-block:: python

    loss = model.get_loss(x, target)

- **loss** (*Tensor*):  
  Scalar total loss for object detection, including coordinate, confidence, and classification losses.

Loss Formula
------------

The total YOLO loss is defined as:

.. math::

    \begin{aligned}
    \mathcal{L} &= \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}}
    \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right] \\
    &+ \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} 
      \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right] \\
    &+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 \\
    &+ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} 
      (C_i - \hat{C}_i)^2 \\
    &+ \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c=1}^{C} (p_i(c) - \hat{p}_i(c))^2
    \end{aligned}

Where:

- :math:`\mathbb{1}_{ij}^{\text{obj}}` indicates that object exists in cell i for box j,
- :math:`\hat{C}_i` is the predicted confidence,
- :math:`(x_i, y_i, w_i, h_i)` are bounding box values,
- :math:`p_i(c)` is the class probability.

Methods
-------

.. automethod:: lucid.models.objdet.YOLO_V1.forward
.. automethod:: lucid.models.objdet.YOLO_V1.get_loss

Examples
--------

.. code-block:: python

    import lucid
    import lucid.nn as nn
    from lucid.models.objdet import YOLO_V1

    model = YOLO_V1(
        in_channels=3,
        split_size=7,
        num_boxes=2,
        num_classes=20
    )

    # Forward pass
    x = lucid.rand(16, 3, 448, 448)
    preds = model(x)

    # Compute loss
    target = lucid.rand(16, 7, 7, 5 * 2 + 20)
    loss = model.get_loss(x, target)

    loss.backward()

.. tip::

    YOLO expects bounding boxes as relative coordinates:  
    
    - `x`, `y` are center positions relative to the grid cell.  
    - `w`, `h` are normalized by image width and height.

.. warning::

    The most confident bounding box (highest IoU with ground truth) is chosen for loss computation.  
    Make sure your dataset follows the expected shape: `(N, S, S, 5 * B + C)`.
