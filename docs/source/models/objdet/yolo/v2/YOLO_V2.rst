YOLO_V2
=======
|convnet-badge| |one-stage-det-badge| |objdet-badge|

.. toctree::
    :maxdepth: 1
    :hidden:

    yolo_v2.rst
    yolo_v2_tiny.rst

.. autoclass:: lucid.models.YOLO_V2

The `YOLO_V2` class implements the YOLO-v2 object detection model. 
It is an improvement over YOLO-v1, designed to detect objects in images using 
anchor-based bounding boxes, batch normalization, and a stronger backbone (Darknet-19).

.. image:: yolo_v2.png
    :width: 600
    :alt: YOLO-v2 architecture
    :align: center

Class Signature
---------------

.. code-block:: python

    class YOLO_V2(
        num_classes: int,
        num_anchors: int = 5,
        anchors: list[tuple[float, float]] | None = None,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        darknet: nn.Module | None = None,
        route_layer: int | None = None,
        image_size: int = 416,
    )

Parameters
----------
- **darknet** (*nn.Module*, optional):
  A custom backbone model for feature extraction. If set to `None` (default), 
  the model uses the pre-configured Darknet-19 architecture.

Attributes
----------
- **darknet** (*nn.Module*):
  The feature extraction backbone model, either a custom model or Darknet-19. 
  If `None` is passed, Darknet-19 is used by default.

- **detect_head** (*nn.Module*):
  The detection head that processes the output from the backbone and generates 
  bounding boxes and class scores.

Methods
-------

.. automethod:: lucid.models.objdet.YOLO_V2.forward
.. automethod:: lucid.models.objdet.YOLO_V2.get_loss
.. automethod:: lucid.models.objdet.YOLO_V2.predict

Darknet-19 Integration
----------------------
The default backbone for the `YOLO_V2` class is **Darknet-19**, a convolutional neural network 
designed for efficient feature extraction. 

When `darknet=None` is passed during initialization, the model automatically uses the 
pre-defined Darknet-19 architecture.

To use the Darknet-19 network for object detection, the user must "pop" the network from 
the model to use it for training on the classification task. 

This can be done using `.darknet_19`:

.. code-block:: python
    
    yolo_v2_model = YOLO_V2()
    darknet_model = yolo_v2_model.darknet_19

After training on the classification task, the trained **darknet** 
can be automatically integrated back into the `YOLO_V2` model.

.. warning::

    If a custom backbone is provided (i.e., passing a custom `darknet` model during initialization), 
    the `darknet_19` attribute will raise an `AttributeError` because the custom backbone does 
    not have the full pre-built Darknet-19 structure.

Input Format
------------

The target tensor from the dataset should have shape:

.. code-block:: python

    (N, S, S, B * (5 + C))

Where:

- `S` is `split_size` (grid size),
- `B` is `num_anchors` (number of anchor boxes per grid cell),
- `C` is `num_classes`.

Each vector at `(i, j)` of shape `(B * (5 + C))` is flattened and contains:

- For each box `b = 0 .. B-1`: `(x, y, w, h, conf, cls_1, ..., cls_C)`

YOLO-v2 Loss
------------
The YOLO-v2 loss function builds upon YOLO-v1's multi-part loss and incorporates **anchor boxes**. 
For a grid of size :math:`S \times S` and :math:`B` anchors per grid cell, the predicted tensor 
shape becomes :math:`(S, S, B \times (5 + C))` where:

- 5 = [x, y, w, h, objectness]
- C = number of classes

The total loss :math:`\mathcal{L}` is composed of three parts:

.. math::

    \begin{aligned}
    \mathcal{L} &= \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} 
    \left[
        (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 + (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2
    \right] \\
    &+ \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2 \\
    &+ \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2 \\
    &+ \sum_{i=0}^{S^2} \mathbb{1}_i^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
    \end{aligned}

Where:

- :math:`\hat{x}_i, \hat{y}_i, \hat{w}_i, \hat{h}_i` are predicted box parameters
- :math:`x_i, y_i, w_i, h_i` are target box parameters
- :math:`\hat{C}_i` is predicted objectness score
- :math:`C_i` is the target confidence (IoU)
- :math:`p_i(c)` is the predicted class probability
- :math:`\hat{p}_i(c)` is the ground-truth class probability (one-hot)
- :math:`\mathbb{1}_{ij}^{\text{obj}}` indicates if anchor :math:`j` in cell 
  :math:`i` is responsible for detecting an object

- :math:`\lambda_{\text{coord}}`, :math:`\lambda_{\text{noobj}}` are weighting hyperparameters

.. note::

    Unlike YOLO-v1, YOLO-v2 uses **predefined anchors** and decouples object classification 
    and localization more clearly, improving detection stability and accuracy.

Target Encoding and Anchors
---------------------------

The YOLO_V2 model expects target bounding boxes to be encoded in **t-coordinates**,
 which are offsets relative to the anchor boxes and grid cell positions. 
 
 The target should have shape `(N, S, S, B, 5 + C)`, where each box consists of:

- `tx, ty`: offset of the center from the top-left corner of the grid cell 
  (normalized to :math:`[0,1]`)

- `tw, th`: log-scale ratio between the ground truth width/height and anchor width/height
- `objectness`: 1 if the box is responsible for the object, otherwise 0
- `C`: one-hot encoded class vector

To construct these targets properly, one must make use of the anchor box scales. 
These can be accessed from the class variable:

.. code-block:: python

   YOLO_V2.default_anchors: list[tuple[float, float]]

This list contains default anchor box widths and heights relative to the feature map scale. 

When preparing your dataset, you should assign ground truth boxes to the most appropriate 
anchor using IoU(`lucid.models.objdet.utils.iou`), and convert the box coordinates into 
t-coordinates (as above) based on the matching anchor. The same anchors must be used 
when instantiating the model via the `anchors` parameter 
(if not set, `default_anchors` is used).

Prediction Output
-----------------

Calling `model.predict(...)` returns final post-processed detections after applying 
confidence thresholding and non-maximum suppression (NMS). 

The return value is a list of length `N` (batch size), where each element is a list 
of dictionaries representing the detected objects in that image. Each dictionary 
has the following keys:

- `"box"`: Tensor of shape `(4,)` representing the absolute coordinates 
  `[x1, y1, x2, y2]` of the bounding box in pixels

- `"score"`: Confidence score after multiplying objectness with class probability
- `"class_id"`: Predicted class index

Example Usage
-------------
.. admonition:: Using YOLO-V2 with default Darknet-19

    .. code-block:: python

        >>> import lucid
        >>> import lucid.models as models
        >>> model = models.YOLO_V2()  # Uses default Darknet-19 backbone
        >>> input_tensor = lucid.Tensor(..., requires_grad=False)
        >>> output = model(input_tensor)
        >>> print(output.shape)

.. admonition:: Using YOLO-V2 with a custom backbone

    .. code-block:: python

        >>> custom_darknet = ...  # Define or load your custom backbone
        >>> model = models.YOLO_V2(darknet=custom_darknet)
        >>> input_tensor = lucid.Tensor(..., requires_grad=False)
        >>> output = model(input_tensor)
        >>> print(output.shape)

Backward Propagation
--------------------
The YOLO-V2 model supports backpropagation through its network for training purposes. 
During backpropagation, gradients are computed and propagated through the darknet layers 
as well as the detection head:

.. code-block:: python

    >>> output.backward()
    >>> print(input_tensor.grad)  # Gradients w.r.t. the input tensor
