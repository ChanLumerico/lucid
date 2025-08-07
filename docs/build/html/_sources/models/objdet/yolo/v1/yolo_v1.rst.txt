yolo_v1
=======
|convnet-badge| |one-stage-det-badge| |objdet-badge|

.. autofunction:: lucid.models.yolo_v1

The `yolo_v1` function constructs a YOLO (You Only Look Once) object detection model 
following the original YOLO-v1 architecture proposed by Redmon et al. (2016).  
This single-stage detector is optimized for speed and enables real-time detection 
of multiple objects in grid-based regions of the input image.

**Total Parameters**: 271,716,734 (ConvNet + FC)

Function Signature
------------------

.. code-block:: python

    @register_model
    def yolo_v1(num_classes: int = 20, **kwargs) -> YOLO_V1

Parameters
----------

- **num_classes** (*int*, optional):  
  Number of object classes to detect. Default is 20 (PASCAL VOC).

- **kwargs** (*dict*, optional):  
  Additional arguments to override defaults in `YOLO_V1`, such as:
  
  - `split_size` (int): Grid size for dividing the input image (default: 7).
  - `num_boxes` (int): Number of bounding boxes per grid cell (default: 2).
  - `lambda_coord` (float): Weight for coordinate loss (default: 5.0).
  - `lambda_noobj` (float): Weight for no-object confidence loss (default: 0.5).

Returns
-------

- **YOLO_V1**:  
  An instance of the YOLOv1 model ready for training or inference.

Examples
--------

**Basic Usage**

.. code-block:: python

    from lucid.models import yolo_v1

    # Create YOLOv1 model with 20 target classes
    model = yolo_v1(num_classes=20)

    # Input: batch of images with shape (N, 3, 448, 448)
    x = lucid.rand(8, 3, 448, 448)

    # Output: tensor of shape (N, 7, 7, 30) for VOC (20 classes, 2 boxes)
    preds = model(x)

    print(preds.shape)  # (8, 7, 7, 30)

Training Notes
--------------

The output shape of the model is:

.. code-block:: text

    (N, S, S, 5 * B + C)

Where:
- `S` is the grid size (`split_size`, default: 7),
- `B` is the number of boxes per cell (`num_boxes`, default: 2),
- `C` is the number of object classes (`num_classes`, e.g., 20 for VOC).

This includes:
- B bounding boxes (`x`, `y`, `w`, `h`, `conf`),
- C class probabilities.

Use the `get_loss` method of the returned model to compute the training loss 
against the corresponding ground truth targets in the same format.

.. tip::

   You can override architectural options like `split_size`, `num_boxes`, or loss coefficients 
   via `**kwargs` to create variants of the original YOLOv1 model.

.. warning::

   Make sure the ground truth targets fed into the loss function match the required shape  
   `(N, S, S, 5 * B + C)`, with coordinates normalized to the grid and confidence + 
   class vectors properly set.
